"""roles/daigensui.py — 大元帥 (Claude Opus 4.6) ロール v15

役割: 最終エスカレーション・最高難度判断
モデル: claude-opus-4-6 (Proプラン CLI 優先 → API フォールバック)
"""

import time
from roles.base import BaseRole, RoleResult

PERSONA = (
    "あなたは大元帥（Claude Opus 4.6）、武士団マルチエージェントシステムの総司令官です。"
    "最高難度の判断を下す最高意思決定者として、深く洞察に富んだ回答を日本語でしてください。"
)


class DaigensuiRole(BaseRole):
    role_key = "daigensui"
    role_name = "大元帥"
    model_name = "Claude Opus 4.6"
    emoji = "⚔️"
    default_handled_by = "daigensui_audit"

    async def execute(self, state: dict) -> RoleResult:
        start = time.time()
        client = self._get_client()
        if not client:
            return RoleResult(
                response="⚠️ 大元帥クライアント未設定 (ANTHROPIC_API_KEY)",
                agent_role=self.role_name, handled_by=self.default_handled_by,
                execution_time=time.time() - start, error="client not available", status="failed",
            )
        try:
            system = self._build_system_prompt(state, PERSONA)
            mcp_used = []
            msg = state.get("message", "")
            _GIT_KWS = ["git", "コミット", "commit", "プッシュ", "push", "プル", "pull",
                        "clone", "クローン", "ブランチ", "branch", "差分", "diff",
                        "マージ", "merge", "PR", "プルリク", "issue", "イシュー", "リポジトリ"]

            # ── メモリ + 戦略的思考 + Git (必要時) を並列取得 ───────────
            needs_git = any(kw in msg for kw in _GIT_KWS)
            think_args = {
                "thought": msg[:800], "thoughtNumber": 1,
                "totalThoughts": 3, "nextThoughtNeeded": False,
            }
            parallel_calls = [("read_graph", {}), ("sequentialthinking", think_args)]
            if needs_git:
                parallel_calls += [
                    ("git_status", {"repo_path": "/mnt/Bushidan-Multi-Agent"}),
                    ("git_log",    {"repo_path": "/mnt/Bushidan-Multi-Agent", "max_count": 5}),
                ]
            results = await self._mcp_parallel(parallel_calls)

            if results[0]:
                mem_ctx = results[0].content if hasattr(results[0], "content") else str(results[0])
                system = self._append_mcp_context(system, "記憶・ナレッジ", mem_ctx[:1500])
                mcp_used.append("read_graph")
            if results[1]:
                think_ctx = results[1].content if hasattr(results[1], "content") else str(results[1])
                system = self._append_mcp_context(system, "戦略的思考", think_ctx[:1500])
                mcp_used.append("sequentialthinking")
            if needs_git and len(results) > 2:
                if results[2]:
                    git_s = results[2].content if hasattr(results[2], "content") else str(results[2])
                    system = self._append_mcp_context(system, "git status", git_s[:1500])
                    mcp_used.append("git_status")
                if len(results) > 3 and results[3]:
                    git_l = results[3].content if hasattr(results[3], "content") else str(results[3])
                    system = self._append_mcp_context(system, "git log", git_l[:1500])
                    mcp_used.append("git_log")

            # ── ファイル参照 (直列・最大2件) ─────────────────────────
            for ref in self._extract_file_refs(msg)[:2]:
                content = await self._mcp_read_file(ref)
                if content:
                    system = self._append_mcp_context(system, f"ファイル: {ref}", content)
                    mcp_used.append("read_file")

            if mcp_used:
                self.logger.info("👑 大元帥: MCP使用 %s", mcp_used)

            messages = self._format_messages(state)
            response = await client.generate(messages=messages, system=system, max_tokens=4000)
            return RoleResult(
                response=response, agent_role=self.role_name,
                handled_by=self.default_handled_by, execution_time=time.time() - start,
                requires_followup=self._needs_followup(response, state),
                mcp_tools_used=mcp_used,
            )
        except Exception as e:
            self.logger.error("大元帥実行失敗: %s", e)
            return RoleResult(
                response=f"❌ 大元帥エラー: {e}", agent_role=self.role_name,
                handled_by=self.default_handled_by, execution_time=time.time() - start,
                error=str(e), status="failed",
            )
