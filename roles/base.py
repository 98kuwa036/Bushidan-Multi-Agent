"""
roles/base.py — ロール抽象基底クラス v15

全ロールが共通で持つ:
  - execute(state) → RoleResult
  - _get_client() — ClientRegistry 経由でクライアント取得
  - _build_system_prompt(state) — Notionコンテキスト注入
  - _format_messages(state) — 会話履歴 + 現在のメッセージを messages 形式に変換
  - _needs_followup(response, state) — 自律ループ判定
  - _call_mcp_tool(name, args) — MCPツール呼び出しヘルパー
"""

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from utils.logger import get_logger

# ── 認識論的謙虚さ: 全ロール共通注入テキスト ─────────────────────────
_EPISTEMIC_HUMILITY = (
    "\n\n【認識論的謙虚さの原則】\n"
    "- 不確実な情報は「〜の可能性があります」「〜と推測されますが確認が必要です」と明示する\n"
    "- 知識カットオフ（2025年8月）以降の情報は推測であることを述べる\n"
    "- 推論と事実を明確に区別し、「〜と思われる」「〜かもしれない」を適切に使う\n"
    "- 確信が低い場合は「確信度: 低」と添え、ユーザーに確認を促す\n"
    "- 「わからない」「要確認」は正直な回答であり、不確かな推測より優れている\n"
    "- ユーザーや他エージェントから誤りを指摘された場合は素直に受け入れ修正する"
)

# ── read_graph TTLキャッシュ (30秒、プロセス全体で共有) ───────────────
_GRAPH_CACHE: Optional[str] = None
_GRAPH_CACHE_AT: float = 0.0
_GRAPH_CACHE_TTL: float = 30.0


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
    # HITL — ロールが人間の承認を要求する場合に設定
    awaiting_human_input: bool = False
    human_question: str = ""


class BaseRole(ABC):
    """武士団ロール基底クラス v15"""

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
        """Notion RAG + Bushidanファイルコンテキストをシステムプロンプトに注入する共通ヘルパー"""
        prompt = base_prompt or (
            f"あなたは{self.role_name}（{self.model_name}）、武士団マルチエージェントシステムの"
            f"{self.role_key}担当です。明確・実用的な日本語で回答してください。"
        )

        # 認識論的謙虚さを全ロールに注入
        prompt += _EPISTEMIC_HUMILITY

        # 訂正モード: ユーザーが前の回答の誤りを指摘した場合の追加指示
        if state.get("is_correction"):
            prompt += (
                "\n\n【訂正モード】\n"
                "ユーザーが直前の回答の誤りを指摘しています。\n"
                "まず「〜の点が誤りでした。訂正します」と明示的に誤りを認め、\n"
                "修正した正しい内容を提示してください。\n"
                "謝罪は1文で簡潔に、修正内容に比重を置いてください。"
            )

        # Notion RAG コンテキスト注入
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

        # Bushidan ファイルコンテキスト注入
        try:
            from utils.bushidan_files import build_file_context
            message = state.get("message", "")
            file_ctx = build_file_context(message)
            if file_ctx:
                prompt += "\n\n---\n" + file_ctx
        except Exception as e:
            self.logger.debug("Bushidanファイルコンテキスト取得スキップ: %s", e)

        return prompt

    def _format_messages(self, state: dict) -> list:
        """
        会話履歴 + 現在のメッセージを messages リスト形式で返す。
        直近10往復 + 現在のメッセージ。履歴が長い場合はサマリで先頭を圧縮。
        """
        history = state.get("conversation_history", [])
        message = state.get("message", "")

        # 既存のサマリ (check_followup で生成) を先頭に付加
        context_summary = state.get("context_summary", "")

        if len(history) > 20:
            # 20往復超: 古い部分をトークン節約のため省略し、サマリに委ねる
            history = history[-20:]

        messages: list = []
        if context_summary and len(history) <= 20:
            # サマリを system 代替として先頭 user/assistant ペアで注入
            messages.append({"role": "user", "content": f"[会話要約]\n{context_summary}"})
            messages.append({"role": "assistant", "content": "了解しました。"})

        messages.extend(history)
        if message:
            messages.append({"role": "user", "content": message})
        return messages

    def _save_response_files(self, response: str) -> list[str]:
        """
        LLMレスポンスから [FILE:xxx]...[/FILE] ブロックを検出して保存する。
        保存したファイルパスのリストを返す。
        """
        try:
            from utils.bushidan_files import extract_and_save_files
            saved = extract_and_save_files(response)
            if saved:
                self.logger.info("📁 Bushidan保存: %s", saved)
            return saved
        except Exception as e:
            self.logger.warning("Bushidanファイル保存失敗: %s", e)
            return []

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

    # ── MCP 高レベルヘルパー ────────────────────────────────────────────

    _SEARCH_TRIGGERS = frozenset([
        "最新", "調べて", "検索", "ニュース", "現在", "今日", "いつ", "だれ", "何時",
        "について教えて", "とは何", "どうなって", "いくら", "何が",
        "search", "latest", "news", "current", "find out",
    ])

    def _needs_web_search(self, message: str) -> bool:
        """メッセージが Web 検索を必要とするか簡易判定"""
        return any(kw in message for kw in self._SEARCH_TRIGGERS)

    async def _mcp_search(self, query: str, max_results: int = 3) -> str:
        """Tavily Web検索。結果をテキストで返す。失敗時は空文字。"""
        try:
            result = await self._call_mcp_tool("tavily_search", {
                "query": query, "max_results": max_results,
            })
            if not result:
                return ""
            if isinstance(result, str):
                return result[:2000]
            if isinstance(result, list):
                parts = []
                for item in result[:max_results]:
                    if isinstance(item, dict):
                        title   = item.get("title", "")
                        url     = item.get("url", "")
                        content = item.get("content", item.get("snippet", ""))[:400]
                        parts.append(f"• {title}\n  {url}\n  {content}")
                    else:
                        parts.append(str(item)[:200])
                return "\n\n".join(parts)
            # ContentBlock (LangChain ToolMessage) の場合
            if hasattr(result, "content"):
                return str(result.content)[:2000]
            return str(result)[:2000]
        except Exception as e:
            self.logger.debug("Web検索失敗 (スキップ): %s", e)
            return ""

    async def _mcp_read_memory(self) -> str:
        """ナレッジグラフ全体を読んでコンテキスト文字列を返す。30秒TTLキャッシュ付き。"""
        global _GRAPH_CACHE, _GRAPH_CACHE_AT
        now = time.monotonic()
        if _GRAPH_CACHE is not None and (now - _GRAPH_CACHE_AT) < _GRAPH_CACHE_TTL:
            return _GRAPH_CACHE
        try:
            result = await self._call_mcp_tool("read_graph", {})
            if not result:
                return ""
            text = result.content if hasattr(result, "content") else str(result)
            text = text[:1500]
            _GRAPH_CACHE = text
            _GRAPH_CACHE_AT = now
            return text
        except Exception as e:
            self.logger.debug("メモリ読み込み失敗 (スキップ): %s", e)
            return ""

    async def _mcp_think(self, thought: str) -> str:
        """Sequential thinking で問題を1ステップ分析。結果文字列を返す。"""
        try:
            result = await self._call_mcp_tool("sequentialthinking", {
                "thought": thought[:800],
                "thoughtNumber": 1,
                "totalThoughts": 3,
                "nextThoughtNeeded": False,
            })
            if not result:
                return ""
            if hasattr(result, "content"):
                return str(result.content)[:1500]
            if isinstance(result, dict):
                return result.get("thought", result.get("content", str(result)))[:1500]
            return str(result)[:1500]
        except Exception as e:
            self.logger.debug("Sequential thinking 失敗 (スキップ): %s", e)
            return ""

    async def _mcp_read_file(self, path: str) -> str:
        """ファイルを読んで内容を返す。失敗時は空文字。"""
        try:
            result = await self._call_mcp_tool("read_file", {"path": path})
            if not result:
                return ""
            text = result.content if hasattr(result, "content") else str(result)
            return text[:2000]
        except Exception as e:
            self.logger.debug("ファイル読み込み失敗 %s (スキップ): %s", path, e)
            return ""

    async def _mcp_write_file(self, path: str, content: str) -> bool:
        """ファイルを書き込む。成功なら True。"""
        try:
            await self._call_mcp_tool("write_file", {"path": path, "content": content})
            return True
        except Exception as e:
            self.logger.debug("ファイル書き込み失敗 %s: %s", path, e)
            return False

    async def _mcp_git_status(self) -> str:
        """git status を取得。失敗時は空文字。"""
        try:
            result = await self._call_mcp_tool("git_status", {"repo_path": "/mnt/Bushidan-Multi-Agent"})
            text = result.content if hasattr(result, "content") else str(result)
            return text[:1500] if result else ""
        except Exception as e:
            self.logger.debug("git_status 失敗 (スキップ): %s", e)
            return ""

    async def _mcp_git_log(self, max_count: int = 5) -> str:
        """git log を取得。失敗時は空文字。"""
        try:
            result = await self._call_mcp_tool("git_log", {
                "repo_path": "/mnt/Bushidan-Multi-Agent",
                "max_count": max_count,
            })
            text = result.content if hasattr(result, "content") else str(result)
            return text[:1500] if result else ""
        except Exception as e:
            self.logger.debug("git_log 失敗 (スキップ): %s", e)
            return ""

    async def _mcp_git_diff(self) -> str:
        """git diff (staged + unstaged) を取得。失敗時は空文字。"""
        try:
            result = await self._call_mcp_tool("git_diff_unstaged", {"repo_path": "/mnt/Bushidan-Multi-Agent"})
            text = result.content if hasattr(result, "content") else str(result)
            return text[:2000] if result else ""
        except Exception as e:
            self.logger.debug("git_diff 失敗 (スキップ): %s", e)
            return ""

    async def _mcp_screenshot(self, url: str) -> str:
        """URL のスクリーンショットを取得し base64 文字列を返す。失敗時は空文字。"""
        try:
            await self._call_mcp_tool("browser_navigate", {"url": url})
            result = await self._call_mcp_tool("browser_take_screenshot", {})
            if not result:
                return ""
            if isinstance(result, dict):
                return result.get("data", result.get("base64", ""))
            if hasattr(result, "content"):
                return str(result.content)
            return str(result)
        except Exception as e:
            self.logger.debug("スクリーンショット失敗 %s (スキップ): %s", url, e)
            return ""

    def _extract_file_refs(self, message: str) -> list[str]:
        """メッセージからファイルパス候補を抽出"""
        import re
        return re.findall(r'(?:/[\w./\-]+|[\w\-]+\.(?:py|js|ts|json|yaml|yml|md|txt|sh))', message)

    def _extract_code_blocks(self, message: str, language: str = "python") -> list[str]:
        """メッセージからコードブロック (```lang ... ```) を抽出"""
        import re
        pattern = rf'```{language}\s*([\s\S]*?)```'
        return re.findall(pattern, message, re.IGNORECASE)

    def _extract_urls(self, message: str) -> list[str]:
        """メッセージから URL を抽出"""
        import re
        return re.findall(r'https?://[^\s<>"\']+', message)

    def _append_mcp_context(self, system: str, label: str, content: str) -> str:
        """システムプロンプトに MCP コンテキストを追加"""
        if content:
            return system + f"\n\n---\n【{label}】\n{content}"
        return system

    async def _mcp_parallel(self, calls: list) -> list:
        """複数 MCP ツールを並列呼び出し。read_graph はキャッシュを利用。
        calls: [(tool_name, args_dict), ...]
        戻り値: 各結果のリスト (失敗は None)
        """
        async def _one(name: str, args: dict):
            global _GRAPH_CACHE, _GRAPH_CACHE_AT
            # read_graph はキャッシュヒット時はネットワーク不要
            if name == "read_graph":
                now = time.monotonic()
                if _GRAPH_CACHE is not None and (now - _GRAPH_CACHE_AT) < _GRAPH_CACHE_TTL:
                    return _GRAPH_CACHE
            try:
                result = await self._call_mcp_tool(name, args)
                # read_graph 結果をキャッシュ
                if name == "read_graph" and result is not None:
                    text = result.content if hasattr(result, "content") else str(result)
                    _GRAPH_CACHE = text[:1500]
                    _GRAPH_CACHE_AT = time.monotonic()
                return result
            except Exception as e:
                self.logger.debug("並列MCP %s 失敗: %s", name, e)
                return None

        results = await asyncio.gather(*[_one(n, a) for n, a in calls], return_exceptions=True)
        # 例外が混入した場合は None に変換 (個別の _one() 内で既にキャッチしているが安全弁)
        return [r if not isinstance(r, BaseException) else None for r in results]

