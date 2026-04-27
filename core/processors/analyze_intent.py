"""
武士団 v18 — Uchu (analyze_intent) プロセッサ
Groq JSON mode で複雑度・Intent を分類
pct239 が利用可能な場合は GBNF Grammar を使用（fallback: JSON regex）
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import time
from typing import Optional

from utils.logger import get_logger
from core.models.karasu import KarasuOutput
from core.models.uchu import ComplexityLevel, IntentType, UchuOutput

logger = get_logger(__name__)

_SYSTEM_PROMPT = """\
あなたは武士団マルチエージェントシステムの受付担当です。
ユーザーの入力を分析し、以下の JSON 形式で返答してください（他のテキストは一切不要）。

{
  "complexity": "simple|low_medium|medium|complex|strategic",
  "intent_type": "qa|code|analysis|task|rag|image",
  "has_image": false,
  "has_secret_keyword": false,
  "is_very_short": false,
  "primary_topic": "主題（最大100文字）",
  "sub_topics": ["副題1", "副題2"],
  "requires_external_knowledge": false,
  "requires_code_execution": false,
  "suggested_role": null,
  "confidence": 0.85,
  "reasoning": "分類根拠（最大200文字）"
}

complexity の基準:
- simple: 15文字以下, 単純な Q&A, 定義問い合わせ
- low_medium: 軽度コーディング, 要約, 翻訳
- medium: 汎用処理, 中程度の推論, 標準的な実装
- complex: 戦略立案, 多段階推論, 高度な設計
- strategic: 監査, システム全体の最終判断

intent_type の基準:
- qa: 質問応答, 情報検索
- code: コード生成・修正・デバッグ
- analysis: データ分析, 推論, 比較評価
- task: 実務タスク, ドキュメント作成
- rag: 外部情報が必要な検索
- image: 画像・ビジョン処理

秘密キーワード（has_secret_keyword=true）: 「極秘」「onmitsu」「忍び」
"""

# シークレットキーワード
_SECRET_KEYWORDS = ["極秘", "onmitsu", "忍び", "secret", "top secret"]


def _detect_shortcuts(user_input: str) -> dict:
    """ショートカット判定"""
    return {
        "has_image": any(t in user_input for t in ["[IMAGE]", "[画像]", "image:", "img:"]),
        "has_secret_keyword": any(kw in user_input.lower() for kw in _SECRET_KEYWORDS),
        "is_very_short": len(user_input.strip()) <= 15,
    }


def _parse_json_response(text: str) -> Optional[dict]:
    """JSON 抽出（コードブロックや前置テキスト対応）"""
    # コードブロック除去
    text = re.sub(r"```(?:json)?\s*", "", text).replace("```", "").strip()

    # 最初の { から最後の } まで抽出
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return None


def _default_output(user_input: str, shortcuts: dict) -> UchuOutput:
    """JSON パース失敗時のデフォルト出力"""
    if shortcuts["has_image"]:
        return UchuOutput(
            complexity=ComplexityLevel.MEDIUM,
            intent_type=IntentType.IMAGE,
            has_image=True,
            suggested_role="kengyo",
            confidence=0.9,
            reasoning="画像入力を検出 → kengyo にルーティング",
            **{k: v for k, v in shortcuts.items() if k != "has_image"},
        )
    if shortcuts["is_very_short"]:
        return UchuOutput(
            complexity=ComplexityLevel.SIMPLE,
            intent_type=IntentType.QA,
            confidence=0.7,
            reasoning="短入力 → simple Q&A",
            **shortcuts,
        )
    return UchuOutput(
        complexity=ComplexityLevel.MEDIUM,
        intent_type=IntentType.QA,
        confidence=0.5,
        reasoning="JSON パース失敗 → デフォルト",
        **shortcuts,
    )


def _json_to_uchu(obj: dict, shortcuts: dict) -> UchuOutput:
    """dict → UchuOutput 変換"""
    # Enum 値の正規化
    try:
        complexity = ComplexityLevel(obj.get("complexity", "medium"))
    except ValueError:
        complexity = ComplexityLevel.MEDIUM

    try:
        intent_type = IntentType(obj.get("intent_type", "qa"))
    except ValueError:
        intent_type = IntentType.QA

    # ショートカットで上書き
    has_image = bool(obj.get("has_image", shortcuts["has_image"]))
    has_secret = bool(obj.get("has_secret_keyword", shortcuts["has_secret_keyword"]))

    if has_image:
        intent_type = IntentType.IMAGE

    return UchuOutput(
        complexity=complexity,
        intent_type=intent_type,
        has_image=has_image,
        has_secret_keyword=has_secret,
        is_very_short=bool(obj.get("is_very_short", shortcuts["is_very_short"])),
        primary_topic=str(obj.get("primary_topic", ""))[:200],
        sub_topics=[str(t) for t in obj.get("sub_topics", [])][:3],
        requires_external_knowledge=bool(obj.get("requires_external_knowledge", False)),
        requires_code_execution=bool(obj.get("requires_code_execution", False)),
        suggested_role=obj.get("suggested_role") or None,
        confidence=float(obj.get("confidence", 0.7)),
        reasoning=str(obj.get("reasoning", ""))[:500],
    )


class UchuProcessor:
    """analyze_intent（受付）プロセッサ"""

    MAX_RETRIES = 3
    TIMEOUT_SEC = 8.0

    def __init__(self) -> None:
        self._groq_key = os.getenv("GROQ_API_KEY", "")
        self._groq = None

    def _get_groq(self):
        if self._groq is None and self._groq_key:
            try:
                from groq import AsyncGroq
                self._groq = AsyncGroq(api_key=self._groq_key)
            except ImportError:
                logger.warning("groq not installed")
        return self._groq

    async def process(self, user_input: str, karasu: Optional[KarasuOutput] = None) -> UchuOutput:
        """analyze_intent を実行"""
        start = time.time()
        shortcuts = _detect_shortcuts(user_input)

        # ショートカット: 画像 / 秘密キーワード
        if shortcuts["has_image"] or shortcuts["has_secret_keyword"]:
            role = "kengyo" if shortcuts["has_image"] else "onmitsu"
            return UchuOutput(
                complexity=ComplexityLevel.SIMPLE,
                intent_type=IntentType.IMAGE if shortcuts["has_image"] else IntentType.TASK,
                suggested_role=role,
                confidence=0.95,
                reasoning=f"ショートカット検出: {'画像' if shortcuts['has_image'] else '秘密キーワード'}",
                **shortcuts,
            )

        # ショートカット: 超短文
        if shortcuts["is_very_short"]:
            return UchuOutput(
                complexity=ComplexityLevel.SIMPLE,
                intent_type=IntentType.QA,
                primary_topic=user_input,
                confidence=0.8,
                reasoning="15文字以下の短入力 → simple QA",
                **shortcuts,
            )

        # Groq で JSON 分類
        system_prompt = _SYSTEM_PROMPT
        if karasu and karasu.search_results:
            from core.processors.pre_process import KarasuProcessor
            injection = KarasuProcessor().build_system_prompt_injection(karasu)
            if injection:
                system_prompt = injection + "\n\n" + system_prompt

        groq = self._get_groq()
        if groq is None:
            logger.warning("Uchu: Groq unavailable, using defaults")
            return _default_output(user_input, shortcuts)

        last_error: Optional[Exception] = None
        for attempt in range(self.MAX_RETRIES):
            try:
                response = await asyncio.wait_for(
                    groq.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_input},
                        ],
                        temperature=0.3,
                        max_tokens=400,
                        response_format={"type": "json_object"},
                    ),
                    timeout=self.TIMEOUT_SEC,
                )
                text = response.choices[0].message.content or ""
                obj = _parse_json_response(text)
                if obj:
                    output = _json_to_uchu(obj, shortcuts)
                    output.reasoning += f" (attempt={attempt + 1})"
                    elapsed = (time.time() - start) * 1000
                    logger.debug("Uchu: %.1fms complexity=%s intent=%s",
                                 elapsed, output.complexity, output.intent_type)
                    return output
            except asyncio.TimeoutError:
                last_error = asyncio.TimeoutError(f"attempt {attempt + 1}")
                logger.warning("Uchu: Groq timeout attempt %d", attempt + 1)
            except Exception as e:
                last_error = e
                logger.warning("Uchu: Groq error attempt %d: %s", attempt + 1, e)

        logger.error("Uchu: all attempts failed, using defaults. last_error=%s", last_error)
        return _default_output(user_input, shortcuts)
