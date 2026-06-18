"""roles/sanbo.py — 参謀 (Gemini Flash Preview) ロール v18

役割: ツール実行・コーディング・ファイル参照・Web検索
モデル: Gemini 3.1 Flash Preview

HITL: 破壊的操作（削除・上書き・強制プッシュ等）を検出した場合、
      実行前に人間の承認を求める。
"""

import re
import time
from roles.base import BaseRole, RoleResult

# 破壊的操作を示すキーワード（HITL 承認が必要）
_DESTRUCTIVE_PATTERNS = [
    r"\brm\s+-rf?\b",           # rm -rf / rm -r
    r"\bgit\s+push\s+.*--force", # git push --force
    r"\bgit\s+reset\s+--hard",  # git reset --hard
    r"\bdrop\s+table\b",        # DROP TABLE
    r"\btruncate\s+table\b",    # TRUNCATE TABLE
    r"\bdelete\s+from\b",       # DELETE FROM (SQL)
    r"本番.*削除|削除.*本番",    # 本番削除
    r"\bchmod\s+777\b",         # chmod 777
    r"\bdd\s+if=",              # dd コマンド
    r"\bformat\s+[a-z]:",       # format drive
]

_DESTRUCTIVE_KWS = frozenset([
    "rm -rf", "git push --force", "git push -f", "git reset --hard",
    "DROP TABLE", "TRUNCATE", "本番環境を削除", "全データ削除",
    "強制プッシュ", "force push",
])

# HITL 承認が必要な操作（破壊的ではないが不可逆・共有影響あり）
_HITL_PATTERNS = [
    r"\bgit\s+push\b(?!\s+.*--force)",  # git push (force push は _DESTRUCTIVE で先に検出)
    r"\bgh\s+pr\s+create\b",            # gh pr create
    r"\bgh\s+issue\s+close\b",          # gh issue close
    r"\bgh\s+release\s+create\b",       # gh release create
]

_GH_KWS = [
    "github", "issue", "イシュー", "pr", "pull request", "プルリク",
    "gh ", "repo", "リポジトリ", "clone", "クローン",
]


class SanboRole(BaseRole):
    role_key = "sanbo"
    role_name = "参謀"
    model_name = "Gemini Flash Preview"
    emoji = "📋"
    default_handled_by = "sanbo_mcp"

    def _check_destructive(self, msg: str) -> str:
        """破壊的操作を検出して説明文を返す。安全なら空文字列を返す。"""
        for kw in _DESTRUCTIVE_KWS:
            if kw.lower() in msg.lower():
                return kw
        for pat in _DESTRUCTIVE_PATTERNS:
            m = re.search(pat, msg, re.IGNORECASE)
            if m:
                return m.group(0)
        return ""

    def _check_hitl_required(self, msg: str) -> str:
        """HITL 承認が必要な操作（不可逆・共有影響）を検出。安全なら空文字列を返す。"""
        for pat in _HITL_PATTERNS:
            m = re.search(pat, msg, re.IGNORECASE)
            if m:
                return m.group(0)
        return ""

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
                "必要に応じてファイルやWeb検索の結果を活用し、明確・実践的な日本語で回答してください。"
                "\n\n【コードタスク時の必須手順】\n"
                "コードに関するタスクが来た場合、必ず最初に read_file または list_directory で"
                "対象ファイル・ディレクトリの内容を確認してから作業を開始すること。"
                "ファイルを読まずにコードを生成・修正してはならない。\n"
                "リポジトリ名がメッセージに含まれる場合（例: 'Bushidan', 'Bushidan-Multi-Agent'）は"
                "`/mnt/Bushidan-Multi-Agent/{リポジトリ名}/` をワークスペースとして使用すること。"
                "変更前に必ず現状を把握し、差分を最小化する修正を心がけること。",
            )
            mcp_used = []
            msg = state.get("message", "")

            # ── HITL: 破壊的操作チェック ─────────────────────────────
            # human_response が既にある場合は承認済みとして通過
            if not state.get("human_response"):
                destructive_op = self._check_destructive(msg)
                if destructive_op:
                    self.logger.warning("🛑 参謀 HITL: 破壊的操作検出 '%s'", destructive_op)
                    return RoleResult(
                        response=f"⚠️ 承認待ち: `{destructive_op}` を実行しようとしています。",
                        agent_role=self.role_name,
                        handled_by=self.default_handled_by,
                        execution_time=time.time() - start,
                        awaiting_human_input=True,
                        human_question=(
                            f"以下の破壊的操作を実行してよいですか？\n\n"
                            f"```\n{destructive_op}\n```\n\n"
                            "「はい」または「いいえ」で答えてください。"
                        ),
                    )
                hitl_op = self._check_hitl_required(msg)
                if hitl_op:
                    self.logger.warning("🛑 参謀 HITL: 承認必要操作検出 '%s'", hitl_op)
                    return RoleResult(
                        response=f"⚠️ 承認待ち: `{hitl_op}` は外部に影響する操作です。",
                        agent_role=self.role_name,
                        handled_by=self.default_handled_by,
                        execution_time=time.time() - start,
                        awaiting_human_input=True,
                        human_question=(
                            f"以下の操作を実行してよいですか？\n\n"
                            f"```\n{hitl_op}\n```\n\n"
                            "この操作は外部（GitHub など）に影響します。「はい」または「いいえ」で答えてください。"
                        ),
                    )
            _GIT_KWS = ["git", "コミット", "commit", "プッシュ", "push", "プル", "pull",
                        "clone", "クローン", "ブランチ", "branch", "差分", "diff", "マージ", "merge"]

            # ── GitHub コンテキスト ──────────────────────────────────────
            if any(kw in msg.lower() for kw in _GH_KWS):
                owner, repo, issue_num = self._extract_github_ref(msg)
                if owner and repo and issue_num:
                    gh_ctx = await self._mcp_gh_issue(owner, repo, issue_num)
                    if gh_ctx:
                        system = self._append_mcp_context(system, f"GitHub Issue {owner}/{repo}#{issue_num}", gh_ctx)
                        mcp_used.append("gh_issue")
                elif owner and repo:
                    issues_ctx = await self._mcp_gh_list_issues(owner, repo)
                    if issues_ctx:
                        system = self._append_mcp_context(system, f"GitHub Issues {owner}/{repo}", issues_ctx)
                        mcp_used.append("gh_list_issues")

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

            # ── Python コード実行 ────────────────────────────────────────
            code_blocks = self._extract_code_blocks(msg, language="python")
            if code_blocks:
                try:
                    from utils.code_sandbox import run_code
                    exec_results = []
                    for code in code_blocks[:2]:   # 最大2ブロック
                        result = await run_code(code)
                        exec_results.append(
                            f"```\n# exit={result['exit_code']}  {result['elapsed_ms']}ms\n"
                            f"{result['stdout'] or ''}"
                            f"{('STDERR: ' + result['stderr']) if result['stderr'] else ''}\n```"
                        )
                    if exec_results:
                        system = self._append_mcp_context(
                            system, "コード実行結果", "\n".join(exec_results)
                        )
                        mcp_used.append("code_sandbox")
                except Exception as _ce:
                    self.logger.warning("code_sandbox 失敗: %s", _ce)

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
