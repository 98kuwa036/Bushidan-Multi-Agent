"""
武士団 v18 — Phase 2 コード品質ループ (Actor-Critic)
Groq (generate) → Haiku (review + fix) ループ

最大 2 ラウンド:
  Round 1: Groq 生成 → Haiku レビュー → PASS なら完了
  Round 2: Haiku フィックス → 最終出力
"""
from __future__ import annotations

import asyncio
import os
import re
import time
from dataclasses import dataclass, field
from typing import List, Optional

from utils.logger import get_logger

logger = get_logger(__name__)

_GROQ_MODEL = "llama-3.3-70b-versatile"
_HAIKU_MODEL = "claude-haiku-4-5-20251001"
_MAX_ROUNDS = 2
_GROQ_TIMEOUT = 15.0
_HAIKU_TIMEOUT = 20.0
_CODE_MAX_TOKENS = 2048

_REVIEW_SYSTEM = """\
あなたは武士団のコードレビュアー（目付）です。
提示されたコードを以下の観点でレビューし、JSON で返してください。

{
  "verdict": "PASS" | "FIX",
  "issues": ["問題1", "問題2"],
  "fixed_code": "修正済みコード（FIX の場合のみ）"
}

レビュー観点:
- 構文エラー・明らかなバグ
- セキュリティリスク (injection, hardcoded secrets, etc.)
- Pythonベストプラクティス違反
- 型アノテーション・docstring の整合性
- 軽微なスタイル問題は PASS でよい
"""

_GENERATE_SYSTEM_TEMPLATE = """\
あなたは武士団のコーディング担当（参謀）です。
以下の要件に従い、高品質な Python コードを生成してください。

要件:
- 動作する完全なコードを提供
- 型アノテーションを使用
- 適切なエラーハンドリング
- 日本語コメント
- セキュリティを考慮
"""


@dataclass
class CodeReviewResult:
    verdict: str  # "PASS" | "FIX"
    issues: List[str] = field(default_factory=list)
    fixed_code: Optional[str] = None
    review_ms: float = 0.0


@dataclass
class CodeQualityResult:
    final_code: str
    original_code: str
    rounds: int
    issues_found: List[str] = field(default_factory=list)
    total_ms: float = 0.0
    stage: str = ""  # e.g. "groq+haiku_fix" | "groq_pass" | "groq_only"


def _extract_code_blocks(text: str) -> str:
    """マークダウンコードブロックからコードを抽出"""
    # ```python ... ``` または ``` ... ```
    match = re.search(r"```(?:python)?\s*\n([\s\S]*?)\n```", text)
    if match:
        return match.group(1).strip()
    # コードブロックなしの場合はそのまま
    return text.strip()


def _parse_review_json(text: str) -> Optional[CodeReviewResult]:
    """レビュー JSON をパース"""
    import json
    # コードブロック除去
    text = re.sub(r"```(?:json)?\s*", "", text).replace("```", "").strip()
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return None
    try:
        obj = json.loads(match.group())
        verdict = obj.get("verdict", "PASS").upper()
        if verdict not in ("PASS", "FIX"):
            verdict = "PASS"
        return CodeReviewResult(
            verdict=verdict,
            issues=[str(i) for i in obj.get("issues", [])],
            fixed_code=obj.get("fixed_code") or None,
        )
    except Exception:
        return None


class CodeQualityLoop:
    """Actor-Critic コード品質ループ"""

    def __init__(self) -> None:
        self._groq_key = os.getenv("GROQ_API_KEY", "")
        self._anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
        self._groq = None
        self._anthropic = None

    def _get_groq(self):
        if self._groq is None and self._groq_key:
            try:
                from groq import AsyncGroq
                self._groq = AsyncGroq(api_key=self._groq_key)
            except ImportError:
                logger.warning("groq not installed")
        return self._groq

    def _get_anthropic(self):
        if self._anthropic is None and self._anthropic_key:
            try:
                import anthropic
                self._anthropic = anthropic.AsyncAnthropic(api_key=self._anthropic_key)
            except ImportError:
                logger.warning("anthropic not installed")
        return self._anthropic

    async def _generate_code(self, user_request: str, system_prompt: str) -> Optional[str]:
        """Groq でコード生成"""
        groq = self._get_groq()
        if groq is None:
            return None
        try:
            response = await asyncio.wait_for(
                groq.chat.completions.create(
                    model=_GROQ_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_request},
                    ],
                    temperature=0.4,
                    max_tokens=_CODE_MAX_TOKENS,
                ),
                timeout=_GROQ_TIMEOUT,
            )
            return response.choices[0].message.content or "" if response.choices else ""
        except asyncio.TimeoutError:
            logger.warning("CodeLoop: Groq generate timeout")
            return None
        except Exception as e:
            logger.warning("CodeLoop: Groq generate error: %s", e)
            return None

    async def _review_code(
        self, code: str, user_request: str
    ) -> CodeReviewResult:
        """Haiku でコードレビュー"""
        anthropic = self._get_anthropic()
        if anthropic is None:
            # Anthropic 未設定 → PASS として扱う
            return CodeReviewResult(verdict="PASS")

        review_prompt = f"ユーザーの要件:\n{user_request}\n\n生成コード:\n```python\n{code}\n```"

        t0 = time.time()
        try:
            response = await asyncio.wait_for(
                anthropic.messages.create(
                    model=_HAIKU_MODEL,
                    system=_REVIEW_SYSTEM,
                    messages=[{"role": "user", "content": review_prompt}],
                    max_tokens=1024,
                    temperature=0.1,
                ),
                timeout=_HAIKU_TIMEOUT,
            )
            text = response.content[0].text if response.content else ""
            result = _parse_review_json(text)
            if result is None:
                result = CodeReviewResult(verdict="PASS")
            result.review_ms = (time.time() - t0) * 1000
            return result
        except asyncio.TimeoutError:
            logger.warning("CodeLoop: Haiku review timeout")
            return CodeReviewResult(verdict="PASS", review_ms=(time.time() - t0) * 1000)
        except Exception as e:
            logger.warning("CodeLoop: Haiku review error: %s", e)
            return CodeReviewResult(verdict="PASS", review_ms=(time.time() - t0) * 1000)

    async def _fix_code(
        self, code: str, issues: List[str], user_request: str
    ) -> str:
        """Haiku でコード修正"""
        anthropic = self._get_anthropic()
        if anthropic is None:
            return code

        fix_prompt = (
            f"ユーザーの要件:\n{user_request}\n\n"
            f"問題のあるコード:\n```python\n{code}\n```\n\n"
            f"検出された問題:\n" + "\n".join(f"- {i}" for i in issues) +
            "\n\n上記の問題を修正した完全なコードを提供してください。"
        )

        try:
            response = await asyncio.wait_for(
                anthropic.messages.create(
                    model=_HAIKU_MODEL,
                    system="コードの問題を修正し、完全な修正済みコードのみを返してください。",
                    messages=[{"role": "user", "content": fix_prompt}],
                    max_tokens=_CODE_MAX_TOKENS,
                    temperature=0.2,
                ),
                timeout=_HAIKU_TIMEOUT,
            )
            text = response.content[0].text if response.content else code
            return _extract_code_blocks(text) or text
        except Exception as e:
            logger.warning("CodeLoop: Haiku fix error: %s", e)
            return code

    async def process(
        self,
        user_request: str,
        system_prompt: Optional[str] = None,
        max_rounds: int = _MAX_ROUNDS,
    ) -> CodeQualityResult:
        """Actor-Critic ループのメインエントリポイント"""
        t0 = time.time()
        sp = system_prompt or _GENERATE_SYSTEM_TEMPLATE
        all_issues: List[str] = []
        stages: List[str] = []

        # Stage 1: Groq でコード生成
        raw_code = await self._generate_code(user_request, sp)

        if raw_code is None:
            # Groq 失敗 → Haiku で直接生成
            logger.warning("CodeLoop: Groq failed, using Haiku directly")
            stages.append("haiku_direct")
            anthropic = self._get_anthropic()
            if anthropic is None:
                return CodeQualityResult(
                    final_code="# ⚠️ 生成サービスが利用不可です",
                    original_code="",
                    rounds=0,
                    total_ms=(time.time() - t0) * 1000,
                    stage="error",
                )
            try:
                response = await asyncio.wait_for(
                    anthropic.messages.create(
                        model=_HAIKU_MODEL,
                        system=sp,
                        messages=[{"role": "user", "content": user_request}],
                        max_tokens=_CODE_MAX_TOKENS,
                        temperature=0.4,
                    ),
                    timeout=_HAIKU_TIMEOUT,
                )
                code = response.content[0].text if response.content else ""
                return CodeQualityResult(
                    final_code=_extract_code_blocks(code) or code,
                    original_code=code,
                    rounds=1,
                    total_ms=(time.time() - t0) * 1000,
                    stage="haiku_direct",
                )
            except Exception as e:
                return CodeQualityResult(
                    final_code=f"# ⚠️ 生成エラー: {e}",
                    original_code="",
                    rounds=0,
                    total_ms=(time.time() - t0) * 1000,
                    stage="error",
                )

        original_code = _extract_code_blocks(raw_code) or raw_code
        current_code = original_code
        stages.append("groq_generate")

        # ラウンドループ: レビュー → 修正
        for round_num in range(1, max_rounds + 1):
            review = await self._review_code(current_code, user_request)
            all_issues.extend(review.issues)

            if review.verdict == "PASS":
                stages.append(f"haiku_review_pass_r{round_num}")
                logger.debug("CodeLoop: PASS at round %d", round_num)
                break

            # FIX: 修正済みコードがあれば使用、なければ fix メソッドを呼ぶ
            stages.append(f"haiku_review_fix_r{round_num}")
            if review.fixed_code:
                current_code = _extract_code_blocks(review.fixed_code) or review.fixed_code
            else:
                current_code = await self._fix_code(current_code, review.issues, user_request)

            logger.debug("CodeLoop: FIX round %d, issues=%d", round_num, len(review.issues))

        total_ms = (time.time() - t0) * 1000
        logger.info(
            "CodeLoop: %d rounds, %d issues fixed, %.0fms",
            len([s for s in stages if "fix" in s]),
            len(all_issues),
            total_ms,
        )

        return CodeQualityResult(
            final_code=current_code,
            original_code=original_code,
            rounds=len([s for s in stages if "review" in s]),
            issues_found=all_issues,
            total_ms=total_ms,
            stage="+".join(stages),
        )
