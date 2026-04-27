"""roles/daigensui.py — 大元帥 (Claude Opus 4.6) ロール v18

役割: 最終エスカレーション・最高難度判断
モデル: claude-opus-4-6 (Proプラン CLI 優先 → API フォールバック)
"""

import time
from roles.base import BaseRole, RoleResult

PERSONA = (
    "あなたは大元帥（Claude Opus 4.6）、武士団マルチエージェントシステムの総司令官です。"
    "最高難度の判断を下す最高意思決定者として、深く洞察に富んだ回答を日本語でしてください。"
    "\n\n【クロスチェック責務】\n"
    "回答に「確信度: 低」または「要確認」を含む場合、または戦略・セキュリティ判断の場合、"
    "自身の推論に異なる視点からの反証を試みてから最終判断を下してください。"
    "異なる訓練データ由来のモデルが同じ結論に至る場合のみ高確信度とみなします。"
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

            # ── Groq クロスチェック (低確信度 or 戦略・セキュリティ判断) ──
            _LOW_CONF = ("確信度: 低", "要確認", "不確か", "推測", "わかりません")
            _HIGH_STAKES = ("セキュリティ", "アーキテクチャ", "認証", "脆弱性", "本番", "削除", "権限")
            _needs_xcheck = (
                any(s in response for s in _LOW_CONF)
                or any(s in msg for s in _HIGH_STAKES)
                or state.get("risk_level") in ("HIGH", "CRITICAL")
            )
            if _needs_xcheck:
                response = await self._groq_crosscheck(msg, response, mcp_used)

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

    async def _groq_crosscheck(self, question: str, primary_answer: str, mcp_used: list) -> str:
        """Gemini 3.1 Pro Preview (異なる訓練バイアス) で一次回答を検証し、差異を付記して返す。"""
        try:
            from utils.client_registry import ClientRegistry
            gemini_client = ClientRegistry.get().get_client("claude_fallback")  # gemini-3.1-pro-preview
            if not gemini_client:
                return primary_answer

            xcheck_prompt = (
                f"以下の質問に対する回答案を批判的に検証してください。\n"
                f"事実の誤り・論理的矛盾・重要な見落としがあれば具体的に指摘してください。\n"
                f"問題がなければ「検証OK」とだけ答えてください。\n\n"
                f"【質問】{question[:500]}\n\n【回答案】{primary_answer[:1000]}"
            )
            verdict = await gemini_client.generate(
                messages=[{"role": "user", "content": xcheck_prompt}],
                system="あなたは批判的検証AIです。回答案の誤りを見つけることに特化してください。",
                max_tokens=600,
            )
            if verdict and "検証OK" not in verdict:
                self.logger.info("👑 大元帥: Gemini 3.1 Pro クロスチェックで差異検出")
                mcp_used.append("gemini_crosscheck")
                return (
                    primary_answer
                    + f"\n\n---\n**【クロスチェック指摘 (Gemini 3.1 Pro)】**\n{verdict.strip()}"
                )
        except Exception as e:
            self.logger.debug("Geminiクロスチェックスキップ: %s", e)
        return primary_answer
