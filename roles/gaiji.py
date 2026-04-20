"""roles/gaiji.py — 外事 (Command R 08-2024) ロール v18

役割: 外部情報収集・RAG・マルチステップ処理
モデル: Cohere Command R (08-2024) - RAG最適化
MCP:  tavily_search (常時 — 外部情報収集が主務)
"""

import time
from roles.base import BaseRole, RoleResult


class GaijiRole(BaseRole):
    role_key = "gaiji"
    role_name = "外事"
    model_name = "Command R (08-2024)"
    emoji = "🌐"
    default_handled_by = "gaiji_rag"

    async def execute(self, state: dict) -> RoleResult:
        start = time.time()
        client = self._get_client()
        if not client:
            return RoleResult(
                response="⚠️ 外事クライアント未設定 (COHERE_API_KEY)",
                agent_role=self.role_name, handled_by=self.default_handled_by,
                execution_time=time.time() - start, error="client not available", status="failed",
            )
        try:
            system = self._build_system_prompt(
                state,
                "あなたは外事担当 (Command R)。外部情報収集・RAG・マルチステップ処理の専門家です。"
                "最新の検索結果を活用して正確で包括的な回答を日本語で提供してください。",
            )
            mcp_used = []

            # ── Web 検索 (常時実行 — 外部情報収集が主務) ──────────────────
            query = state.get("message", "")[:300]
            web_ctx = await self._mcp_search(query, max_results=5)
            if web_ctx:
                system = self._append_mcp_context(system, "Web検索結果", web_ctx)
                mcp_used.append("tavily_search")
                self.logger.info("🌐 外事: Web検索完了")

            messages = self._format_messages(state)
            response = await client.generate(messages=messages, system=system, max_tokens=2048)
            return RoleResult(
                response=response, agent_role=self.role_name,
                handled_by=self.default_handled_by, execution_time=time.time() - start,
                requires_followup=self._needs_followup(response, state),
                mcp_tools_used=mcp_used,
            )
        except Exception as e:
            self.logger.error("外事実行失敗: %s", e)
            return RoleResult(
                response=f"❌ 外事エラー: {e}", agent_role=self.role_name,
                handled_by=self.default_handled_by, execution_time=time.time() - start,
                error=str(e), status="failed",
            )
