"""roles/sanbo.py — 参謀 (Gemini 3.5 Flash) ロール v18

役割: ツール実行・コーディング・ファイル参照・Web検索
モデル: Gemini 3.5 Flash

HITL: 破壊的操作（削除・上書き・強制プッシュ等）を検出した場合、
      実行前に人間の承認を求める。
"""

import os
import re
import subprocess
import time
from roles.base import BaseRole, RoleResult

# ── ワークスペース自動セットアップ ─────────────────────────────────────────────
_WORKSPACE_BASE = "/mnt/Bushidan-Multi-Agent"
_GH_OWNER = "98kuwa036"

# 「新規プロジェクト作成」を示すキーワード
_NEW_PROJECT_KWS = [
    "新しいプロジェクト", "新規プロジェクト", "新しいアプリ", "新規アプリ",
    "repo作って", "リポジトリ作って", "新規リポジトリ", "新規repo",
    "new project", "create repo", "new repository", "start project",
]

# 「再開」を示すキーワード
_RESUME_KWS = [
    "続き", "再開", "また", "引き続き", "前回の",
    "resume", "continue", "pick up",
]

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
    model_name = "Gemini 3.5 Flash"
    emoji = "📋"
    default_handled_by = "sanbo_mcp"

    # 大規模リポジトリで優先的に読む重要ファイルの順序
    _PRIORITY_FILES = [
        "README.md", "README.rst", "README.txt",
        "pyproject.toml", "setup.py", "setup.cfg",
        "package.json", "Cargo.toml", "go.mod",
        "ARCHITECTURE.md", "CONTRIBUTING.md",
        "src/main.py", "src/index.ts", "src/index.js",
        "main.py", "app.py", "index.py", "server.py",
    ]

    async def _scan_repository_structure(self, repo_path: str) -> str:
        """
        大規模リポジトリの全体構造を段階的に把握する。
        1. ルート直下のファイル/ディレクトリ一覧を取得
        2. 重要ファイルを優先度順に最大8件読み込む
        Returns: システムプロンプトに追記するコンテキスト文字列
        """
        import os
        parts = [f"【リポジトリ構造: {repo_path}】"]

        # Step1: ルート一覧 + 主要サブディレクトリ一覧
        root_entries = await self._mcp_list_directory(repo_path)
        if not root_entries:
            return ""
        parts.append("ルート:\n" + "\n".join(f"  {e}" for e in root_entries[:40]))

        # Step2: 重要ファイルを優先読み込み
        read_count = 0
        for fname in self._PRIORITY_FILES:
            fpath = os.path.join(repo_path, fname)
            if os.path.exists(fpath):
                content = await self._mcp_read_file(fpath, max_chars=2000)
                if content:
                    parts.append(f"\n--- {fname} ---\n{content}")
                    read_count += 1
            if read_count >= 5:
                break

        # Step3: src/ または lib/ があれば一覧も取得
        for subdir in ("src", "lib", "app", "core"):
            sub_path = os.path.join(repo_path, subdir)
            if os.path.isdir(sub_path):
                sub_entries = await self._mcp_list_directory(sub_path)
                if sub_entries:
                    parts.append(f"\n{subdir}/:\n" + "\n".join(f"  {e}" for e in sub_entries[:20]))
                break

        return "\n".join(parts)

    def _extract_project_name(self, msg: str) -> str:
        """メッセージからプロジェクト名（英数字・ハイフン）を抽出。なければ空文字。"""
        # 「XXXプロジェクト」「XXXアプリ」などを抽出
        patterns = [
            r'[「『]([^」』\s]{2,40})[」』]',           # 鉤括弧内
            r'(?:project|repo|app|system)\s+(\w[\w-]{1,39})',  # 英語
            r'(\w[\w-]{2,39})(?:プロジェクト|アプリ|システム|サービス)',  # 日本語接尾
        ]
        for p in patterns:
            m = re.search(p, msg, re.IGNORECASE)
            if m:
                raw = m.group(1)
                # 英数字・ハイフンのみに正規化
                safe = re.sub(r'[^\w-]', '-', raw).strip('-').lower()
                if len(safe) >= 2:
                    return safe
        return ""

    async def _setup_coding_workspace(self, topic: str) -> tuple[str, str]:
        """
        コーディングワークスペースを準備する。
        優先順位: ローカル存在 → GitHub clone → 新規作成 (pushはHITL)
        Returns: (workspace_path, status_message)
        """
        local_path = os.path.join(_WORKSPACE_BASE, topic)

        # 1. ローカルに git リポジトリとして存在する
        if os.path.isdir(os.path.join(local_path, ".git")):
            # 最新状態を確認
            result = subprocess.run(
                ["git", "log", "--oneline", "-3"],
                cwd=local_path, capture_output=True, text=True, timeout=10,
            )
            commits = result.stdout.strip() or "(コミットなし)"
            return local_path, (
                f"✅ **ローカルで発見** → `{local_path}`\n"
                f"最近のコミット:\n```\n{commits}\n```\n作業を再開します。"
            )

        # 2. GitHub にリポジトリが存在すれば clone
        repo_info = await self._mcp_gh_get_repo(_GH_OWNER, topic)
        if repo_info:
            clone_url = repo_info.get("clone_url") or repo_info.get("html_url", "")
            if clone_url:
                result = subprocess.run(
                    ["git", "clone", clone_url, local_path],
                    capture_output=True, text=True, timeout=60,
                )
                if result.returncode == 0:
                    return local_path, (
                        f"📥 **GitHub からクローン** → `{local_path}`\n"
                        f"リポジトリ: {clone_url}\n作業を再開します。"
                    )
                self.logger.warning("clone 失敗: %s", result.stderr)

        # 3. 新規リポジトリを初期化（push は HITL で確認）
        os.makedirs(local_path, exist_ok=True)
        subprocess.run(["git", "init"], cwd=local_path, capture_output=True)
        readme = os.path.join(local_path, "README.md")
        if not os.path.exists(readme):
            with open(readme, "w") as f:
                f.write(f"# {topic}\n\n")
        subprocess.run(["git", "add", "README.md"], cwd=local_path, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "initial commit"],
            cwd=local_path, capture_output=True,
        )
        return local_path, (
            f"🆕 **新規プロジェクト初期化** → `{local_path}`\n"
            f"git init + README.md 作成済み。\n"
            f"⚠️ GitHub へのプッシュ・リポジトリ作成は承認が必要です。"
        )

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

            # ── ワークスペース自動セットアップ ───────────────────────────
            workspace_status = ""
            is_new_project = any(kw in msg for kw in _NEW_PROJECT_KWS)
            is_resume = any(kw in msg for kw in _RESUME_KWS)
            project_name = self._extract_project_name(msg)

            if project_name and (is_new_project or is_resume):
                ws_path, ws_status = await self._setup_coding_workspace(project_name)
                workspace_status = ws_status
                system = self._append_mcp_context(system, "ワークスペース", ws_status)
                mcp_used.append("workspace_setup")
                self.logger.info("🗡️ 参謀 ワークスペース: %s → %s", project_name, ws_path)

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

            # ── 大規模リポジトリ構造スキャン ────────────────────────────
            # 「全体を把握」「リファクタ」「依存関係」などのキーワードで自動トリガー
            _SCAN_KWS = [
                "全体", "構造", "把握", "概要", "リファクタ", "リファクタリング",
                "依存", "アーキテクチャ", "overview", "structure", "refactor", "codebase",
            ]
            if any(kw in msg for kw in _SCAN_KWS):
                # ワークスペース or Bushidan-Multi-Agent 自身をスキャン
                scan_target = (
                    state.get("workspace_path") or _WORKSPACE_BASE
                )
                scan_ctx = await self._scan_repository_structure(scan_target)
                if scan_ctx:
                    system = self._append_mcp_context(system, "リポジトリ構造", scan_ctx)
                    mcp_used.append("repo_scan")
                    self.logger.info("🗡️ 参謀 リポジトリスキャン完了: %s", scan_target)

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
