"""roles/shogun.py — 将軍 (Claude Sonnet 4.6) ロール v18

役割: メインワーカー・高難度コーディング
モデル: claude-sonnet-4-6 (Proプラン CLI 優先 → API フォールバック)
"""

import time
from roles.base import BaseRole, RoleResult

PERSONA = (
    "あなたは将軍（Claude Sonnet 4.6）、武士団マルチエージェントシステムの最高実行責任者であり、メインワーカーです。\n\n"
    "【重要任務】\n"
    "ユーザーからの実行指示（Goサイン）を受け取った直後の場合、あなたは受付との対話で詰められた「意図」を正確に把握し、"
    "目的達成のために必要なエージェント（軍師、検校、右筆など）の選定と、具体的な実行ロードマップを作成してください。\n"
    "作成したロードマップに沿って、自らも高難度コーディングや実装を担当し、チームを完遂へと導きます。\n"
    "的確かつ実践的に日本語で回答し、目的を最速で達成してください。"
)


class ShogunRole(BaseRole):
    role_key = "shogun"
    role_name = "将軍"
    model_name = "Claude Sonnet 4.6"
    emoji = "🏯"
    default_handled_by = "shogun_plan"

    async def execute(self, state: dict) -> RoleResult:
        start = time.time()
        client = self._get_client()
        if not client:
            return RoleResult(
                response="⚠️ 将軍クライアント未設定 (ANTHROPIC_API_KEY)",
                agent_role=self.role_name, handled_by=self.default_handled_by,
                execution_time=time.time() - start, error="client not available", status="failed",
            )
        try:
            system = self._build_system_prompt(state, PERSONA)
            mcp_used = []
            msg = state.get("message", "")
            _GIT_KWS = ["git", "コミット", "commit", "プッシュ", "push", "プル", "pull",
                        "clone", "クローン", "ブランチ", "branch", "差分", "diff",
                        "マージ", "merge", "PR", "プルリク", "issue", "イシュー"]

            # ── メモリ + Git (必要時) を並列取得 ─────────────────────
            needs_git = any(kw in msg for kw in _GIT_KWS)
            parallel_calls = [("read_graph", {})]
            if needs_git:
                parallel_calls += [
                    ("git_status", {"repo_path": "/mnt/Bushidan-Multi-Agent"}),
                    ("git_log",    {"repo_path": "/mnt/Bushidan-Multi-Agent", "max_count": 5}),
                ]
            results = await self._mcp_parallel(parallel_calls)

            mem_raw = results[0]
            if mem_raw:
                mem_ctx = mem_raw.content if hasattr(mem_raw, "content") else str(mem_raw)
                system = self._append_mcp_context(system, "記憶・ナレッジ", mem_ctx[:1500])
                mcp_used.append("read_graph")
            if needs_git and len(results) > 1:
                if results[1]:
                    git_s = results[1].content if hasattr(results[1], "content") else str(results[1])
                    system = self._append_mcp_context(system, "git status", git_s[:1500])
                    mcp_used.append("git_status")
                if len(results) > 2 and results[2]:
                    git_l = results[2].content if hasattr(results[2], "content") else str(results[2])
                    system = self._append_mcp_context(system, "git log", git_l[:1500])
                    mcp_used.append("git_log")

            # ── ファイル参照 (直列・最大2件) ─────────────────────────
            for ref in self._extract_file_refs(msg)[:2]:
                content = await self._mcp_read_file(ref)
                if content:
                    system = self._append_mcp_context(system, f"ファイル: {ref}", content)
                    mcp_used.append("read_file")

            if mcp_used:
                self.logger.info("🏯 将軍: MCP使用 %s", mcp_used)

            messages = self._format_messages(state)
            response = await client.generate(messages=messages, system=system, max_tokens=4000)
            notice = self._get_fallback_notice(client)
            if notice:
                response = notice + response
            return RoleResult(
                response=response, agent_role=self.role_name,
                handled_by=self.default_handled_by, execution_time=time.time() - start,
                requires_followup=self._needs_followup(response, state),
                mcp_tools_used=mcp_used,
            )
        except Exception as e:
            self.logger.error("将軍実行失敗: %s", e)
            return RoleResult(
                response=f"❌ 将軍エラー: {e}", agent_role=self.role_name,
                handled_by=self.default_handled_by, execution_time=time.time() - start,
                error=str(e), status="failed",
            )
