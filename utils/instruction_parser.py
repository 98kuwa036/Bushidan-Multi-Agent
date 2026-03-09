"""
Instruction Parser

ユーザーの自然言語指示をLLMで解釈し、構造化されたデータに変換します。
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

from utils.logger import get_logger

logger = get_logger(__name__)


class InstructionAction(Enum):
    """指示アクション"""
    APPROVE = "approve"
    REJECT = "reject"
    RETRY = "retry"
    MODIFY = "modify"
    CLARIFY = "clarify"
    UNKNOWN = "unknown"


@dataclass
class ParsedInstruction:
    """
    解析された指示

    Attributes:
        action: アクション種別
        confidence: 信頼度（0.0〜1.0）
        modifications: 修正内容（modify時）
        reason: 理由
        raw_message: 元のメッセージ
    """
    action: InstructionAction
    confidence: float
    modifications: Optional[Dict[str, Any]] = None
    reason: Optional[str] = None
    raw_message: str = ""


class InstructionParser:
    """
    自然言語指示の解析

    ユーザーのコメントを解釈し、承認/却下/修正などのアクションを特定します。
    """

    def __init__(self, llm_client=None):
        """
        初期化

        Args:
            llm_client: LLMクライアント（Gemini, Groq, Claude など）
        """
        self.llm_client = llm_client

        # キーワードベースのフォールバック
        self.keyword_patterns = {
            InstructionAction.APPROVE: [
                "承認", "ok", "OK", "approve", "yes", "いいよ",
                "よし", "go", "進めて", "続けて", "問題ない"
            ],
            InstructionAction.REJECT: [
                "却下", "no", "NO", "reject", "ダメ", "やめて",
                "中止", "stop", "待って", "ストップ"
            ],
            InstructionAction.RETRY: [
                "やり直し", "retry", "再試行", "もう一度", "再度",
                "リトライ", "再実行"
            ],
        }

    async def parse_instruction(
        self,
        user_message: str,
        context: Dict[str, Any]
    ) -> ParsedInstruction:
        """
        ユーザーの指示を解析

        Args:
            user_message: ユーザーのメッセージ
            context: コンテキスト（action_type, action_details など）

        Returns:
            ParsedInstruction: 解析結果
        """
        user_message_lower = user_message.lower().strip()

        # Step 1: キーワードベースの簡易判定
        keyword_result = self._keyword_based_parse(user_message_lower)
        if keyword_result and keyword_result.confidence >= 0.8:
            logger.info(f"📝 キーワードベース解析: {keyword_result.action.value}")
            return keyword_result

        # Step 2: LLMによる詳細解析
        if self.llm_client:
            llm_result = await self._llm_based_parse(user_message, context)
            if llm_result:
                logger.info(f"🤖 LLMベース解析: {llm_result.action.value}")
                return llm_result

        # Step 3: フォールバック
        if keyword_result:
            logger.info(f"📝 フォールバック解析: {keyword_result.action.value}")
            return keyword_result

        # Step 4: 不明な指示
        logger.warning(f"❓ 指示を解釈できませんでした: {user_message[:50]}")
        return ParsedInstruction(
            action=InstructionAction.UNKNOWN,
            confidence=0.0,
            raw_message=user_message,
            reason="指示を理解できませんでした"
        )

    def _keyword_based_parse(self, user_message: str) -> Optional[ParsedInstruction]:
        """キーワードベースの簡易解析"""
        # 承認キーワードチェック
        if any(kw in user_message for kw in self.keyword_patterns[InstructionAction.APPROVE]):
            return ParsedInstruction(
                action=InstructionAction.APPROVE,
                confidence=0.9,
                raw_message=user_message
            )

        # 却下キーワードチェック
        if any(kw in user_message for kw in self.keyword_patterns[InstructionAction.REJECT]):
            return ParsedInstruction(
                action=InstructionAction.REJECT,
                confidence=0.9,
                raw_message=user_message
            )

        # やり直しキーワードチェック
        if any(kw in user_message for kw in self.keyword_patterns[InstructionAction.RETRY]):
            return ParsedInstruction(
                action=InstructionAction.RETRY,
                confidence=0.9,
                raw_message=user_message
            )

        # 修正指示のパターン検出
        modify_patterns = [
            "に変更", "を変更", "に修正", "を修正", "代わりに",
            "change to", "modify", "instead"
        ]

        if any(pattern in user_message for pattern in modify_patterns):
            # 簡易的な修正内容抽出
            modifications = self._extract_modifications(user_message)
            return ParsedInstruction(
                action=InstructionAction.MODIFY,
                confidence=0.7,
                modifications=modifications,
                raw_message=user_message
            )

        return None

    def _extract_modifications(self, user_message: str) -> Dict[str, Any]:
        """簡易的な修正内容抽出"""
        modifications = {"raw_instruction": user_message}

        # ファイル名パターン
        import re

        # "test.py" や "example.py" などのファイル名を抽出
        filename_pattern = r'([a-zA-Z0-9_\-]+\.[a-z]{1,4})'
        filenames = re.findall(filename_pattern, user_message)
        if filenames:
            modifications["filename"] = filenames[0]

        # "test.pyに変更" のような指示
        if "に変更" in user_message or "に修正" in user_message:
            # 変更対象を抽出（簡易版）
            parts = user_message.split("に変更")[0].split("に修正")[0]
            if filenames:
                modifications["target"] = filenames[0]

        return modifications

    async def _llm_based_parse(
        self,
        user_message: str,
        context: Dict[str, Any]
    ) -> Optional[ParsedInstruction]:
        """LLMを使った詳細解析"""
        if not self.llm_client:
            return None

        try:
            # LLMプロンプトを構築
            prompt = self._build_llm_prompt(user_message, context)

            # LLMを呼び出し
            response = await self._call_llm(prompt)

            # レスポンスを解析
            parsed = self._parse_llm_response(response, user_message)

            return parsed

        except Exception as e:
            logger.error(f"❌ LLM解析エラー: {e}")
            return None

    def _build_llm_prompt(
        self,
        user_message: str,
        context: Dict[str, Any]
    ) -> str:
        """LLM用のプロンプトを構築"""
        action_type = context.get("action_type", "unknown")
        action_details = context.get("action_details", {})

        prompt = f"""あなたはユーザーの指示を解釈するアシスタントです。

コンテキスト:
- 作業種別: {action_type}
- 作業詳細: {action_details}

ユーザーのメッセージ:
"{user_message}"

このメッセージから、ユーザーの意図を以下の形式で答えてください:

アクション: [approve/reject/retry/modify/clarify/unknown]
信頼度: [0.0〜1.0]
修正内容: [modifyの場合のみ、具体的な変更内容をJSON形式で]
理由: [判断理由]

例:
ユーザー: "test2.pyに変更して"
アクション: modify
信頼度: 0.95
修正内容: {{"filename": "test2.py"}}
理由: ファイル名をtest2.pyに変更する指示

例:
ユーザー: "承認"
アクション: approve
信頼度: 1.0
修正内容: null
理由: 明確な承認の意思表示

例:
ユーザー: "やめて"
アクション: reject
信頼度: 1.0
修正内容: null
理由: 明確な却下の意思表示

では、解析してください:
"""

        return prompt

    async def _call_llm(self, prompt: str) -> str:
        """LLMを呼び出し"""
        # LLMクライアントの種類に応じて呼び出し
        if hasattr(self.llm_client, 'chat'):
            # Gemini, Claude など
            response = await self.llm_client.chat(prompt)
            return response.get("content", "")

        elif hasattr(self.llm_client, 'complete'):
            # 汎用的な complete メソッド
            response = await self.llm_client.complete(prompt)
            return response

        else:
            raise Exception("LLMクライアントが未対応です")

    def _parse_llm_response(
        self,
        response: str,
        original_message: str
    ) -> ParsedInstruction:
        """LLMレスポンスを解析"""
        import re
        import json

        # アクション抽出
        action_match = re.search(r'アクション:\s*(\w+)', response)
        action_str = action_match.group(1) if action_match else "unknown"

        try:
            action = InstructionAction(action_str.lower())
        except ValueError:
            action = InstructionAction.UNKNOWN

        # 信頼度抽出
        confidence_match = re.search(r'信頼度:\s*([\d\.]+)', response)
        confidence = float(confidence_match.group(1)) if confidence_match else 0.5

        # 修正内容抽出
        modifications = None
        modify_match = re.search(r'修正内容:\s*(\{[^}]+\}|null)', response)
        if modify_match and modify_match.group(1) != "null":
            try:
                modifications = json.loads(modify_match.group(1))
            except json.JSONDecodeError:
                logger.warning("修正内容のJSON解析に失敗しました")

        # 理由抽出
        reason_match = re.search(r'理由:\s*(.+?)(?:\n|$)', response)
        reason = reason_match.group(1).strip() if reason_match else None

        return ParsedInstruction(
            action=action,
            confidence=confidence,
            modifications=modifications,
            reason=reason,
            raw_message=original_message
        )

    def parse_filename_change(self, user_message: str) -> Optional[str]:
        """ファイル名変更指示から新しいファイル名を抽出"""
        import re

        # パターン: "XXX.pyに変更"、"XXX.pyにして"
        patterns = [
            r'([a-zA-Z0-9_\-]+\.[a-z]{1,4})に変更',
            r'([a-zA-Z0-9_\-]+\.[a-z]{1,4})にして',
            r'([a-zA-Z0-9_\-]+\.[a-z]{1,4})で',
            r'change to ([a-zA-Z0-9_\-]+\.[a-z]{1,4})',
        ]

        for pattern in patterns:
            match = re.search(pattern, user_message)
            if match:
                return match.group(1)

        return None
