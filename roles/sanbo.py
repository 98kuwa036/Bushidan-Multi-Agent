"""roles/sanbo.py — 参謀 (Gemini Flash Preview) ロール v18

役割: ツール実行・コーディング・ファイル参照・Web検索
モデル: Gemini 3.1 Flash Preview
"""

import time
from roles.base import BaseRole, RoleResult


class SanboRole(BaseRole):
    role_key = "sanbo"
    role_name = "参謀"
    model_name = "Gemini Flash Preview"
    emoji = "📋"
    default_handled_by = "sanbo_mcp"

    async def execute(self, state: dict) -> RoleResult:
        start = time.time()
        client = self._get_client()
        if not client:
            return RoleResult(
                response="⚠️ 参謀クライアント未設定 (GEMINI_API_KEY を確認)",
                agent_role=self.role_name, handled_by=self.default_handled_by,
                execution_time=time.time() - start, error="client not available", status="failed",
            )
        try:
            system = self._build_system_prompt(
                state,
                "あなたは参謀担当 (Gemini Flash Preview)。ツール実行・コーディング・ファイル参照・Web検索の専門家です。"
                "必要に応じてファイルやWeb検索の結果を活用し、明確・実践的な日本語で回答してください。",
            )
            mcp_used = []
            msg = state.get("message", "")
            _GIT_KWS = ["git", "コミット", "commit", "プッシュ", "push", "プル", "pull",
                        "clone", "クローン", "ブランチ", "branch", "差分", "diff", "マージ", "merge"]

            # ── ファイル参照 ─────────────────────────────────────────────
            for ref in self._extract_file_refs(msg)[:3]:
                content = await self._mcp_read_file(ref)
                if content:
                    system = self._append_mcp_context(system, f"ファイル: {ref}", content)
                    mcp_used.append("read_file")

            # ── Git コンテキスト ─────────────────────────────────────────
            if any(kw in msg for kw in _GIT_KWS):
                git_status = await self._mcp_git_status()
                if git_status:
                    system = self._append_mcp_context(system, "git status", git_status)
                    mcp_used.append("git_status")
                git_diff = await self._mcp_git_diff()
                if git_diff:
                    system = self._append_mcp_context(system, "git diff", git_diff)
                    mcp_used.append("git_diff")

            # ── Web 検索 ─────────────────────────────────────────────────
            if self._needs_web_search(msg):
                web_ctx = await self._mcp_search(msg[:300], max_results=4)
                if web_ctx:
                    system = self._append_mcp_context(system, "Web検索結果", web_ctx)
                    mcp_used.append("tavily_search")

            if mcp_used:
                self.logger.info("🗡️ 参謀: MCP使用 %s", mcp_used)

            messages = self._format_messages(state)
            response = await client.generate(messages=messages, system=system, max_tokens=4000)
            return RoleResult(
                response=response, agent_role=self.role_name,
                handled_by=self.default_handled_by, execution_time=time.time() - start,
                requires_followup=self._needs_followup(response, state),
                mcp_tools_used=mcp_used,
            )
        except Exception as e:
            self.logger.error("参謀実行失敗: %s", e)
            return RoleResult(
                response=f"❌ 参謀エラー: {e}", agent_role=self.role_name,
                handled_by=self.default_handled_by, execution_time=time.time() - start,
                error=str(e), status="failed",
            )
