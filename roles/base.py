"""
roles/base.py — ロール抽象基底クラス v14

全ロールが共通で持つ:
  - execute(state) → RoleResult
  - _get_client() — ClientRegistry 経由でクライアント取得
  - _build_system_prompt(state) — Notionコンテキスト注入
  - _format_messages(state) — 会話履歴 + 現在のメッセージを messages 形式に変換
  - _needs_followup(response, state) — 自律ループ判定
  - _call_mcp_tool(name, args) — MCPツール呼び出しヘルパー
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from utils.logger import get_logger


@dataclass
class RoleResult:
    """ロール実行結果"""
    response: str
    agent_role: str
    handled_by: str
    execution_time: float = 0.0
    error: Optional[str] = None
    mcp_tools_used: list = field(default_factory=list)
    requires_followup: bool = False
    status: str = "completed"


class BaseRole(ABC):
    """武士団ロール基底クラス v14"""

    # サブクラスで定義
    role_key: str = ""
    role_name: str = ""
    model_name: str = ""
    emoji: str = "🤖"
    default_handled_by: str = ""  # LangGraph ノード名

    def __init__(self):
        self.logger = get_logger(f"roles.{self.role_key}")

    def _get_client(self):
        """ClientRegistry からクライアントを取得"""
        from utils.client_registry import ClientRegistry
        return ClientRegistry.get().get_client(self.role_key)

    @abstractmethod
    async def execute(self, state: dict) -> RoleResult:
        """
        タスクを実行して RoleResult を返す。

        Args:
            state: BushidanState (TypedDict)

        Returns:
            RoleResult
        """

    def _build_system_prompt(self, state: dict, base_prompt: str = "") -> str:
        """Notion RAGコンテキストをシステムプロンプトに注入する共通ヘルパー"""
        prompt = base_prompt or (
            f"あなたは{self.role_name}（{self.model_name}）、武士団マルチエージェントシステムの"
            f"{self.role_key}担当です。明確・実用的な日本語で回答してください。"
        )
        notion_chunks = state.get("notion_chunks", [])
        if notion_chunks:
            kb_lines = []
            for chunk in notion_chunks[:3]:
                title = chunk.get("title", "")
                content = chunk.get("content", "")[:400]
                if content:
                    kb_lines.append(f"【{title}】\n{content}")
            if kb_lines:
                prompt += "\n\n---\n【関連ナレッジ】\n" + "\n\n".join(kb_lines)
        return prompt

    def _format_messages(self, state: dict) -> list:
        """
        会話履歴 + 現在のメッセージを messages リスト形式で返す。
        直近10往復 + 現在のメッセージ。
        """
        history = state.get("conversation_history", [])[-20:]
        message = state.get("message", "")
        messages = list(history)
        if message:
            messages.append({"role": "user", "content": message})
        return messages

    def _needs_followup(self, response: str, state: dict) -> bool:
        """
        応答がフォローアップを必要とするか判定する。

        完了シグナル (コードブロック、結論語) がなく、
        「確認」「追加情報」等のキーワードが含まれる場合に True。
        """
        if not response:
            return False
        # 完了シグナル検出
        done_signals = ["```", "以上です", "完了", "まとめ", "結論", "in conclusion"]
        for sig in done_signals:
            if sig in response.lower():
                return False
        # フォローアップ要求検出
        followup_kws = ["確認してください", "追加情報", "もう少し", "次のステップ",
                        "続きを", "詳細を教えて", "please confirm", "next step"]
        return any(kw in response.lower() for kw in followup_kws)

    async def _call_mcp_tool(self, name: str, args: dict) -> Any:
        """MCPツール呼び出しヘルパー"""
        try:
            from core.mcp_sdk import MCPToolRegistry
            registry = MCPToolRegistry.get()
            return await registry.call_tool(name, args)
        except Exception as e:
            self.logger.warning("MCP tool %s 呼び出し失敗: %s", name, e)
            return None
