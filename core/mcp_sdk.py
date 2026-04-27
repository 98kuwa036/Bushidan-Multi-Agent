"""
core/mcp_sdk.py — MCP ツールレジストリ v15

MCPサーバーへの統一アクセスレイヤー。
langchain-mcp-adapters でMCPサーバーに接続し、
ロールごとのLangChainツール取得を提供する。

サーバー種別:
  stdio (npx):    filesystem / memory / sequential_thinking / playwright
  stdio (python): git (mcp-server-git via /home/claude/mcp-git-venv)
  stdio (binary): github (github-mcp-server via /home/claude/bin)
  stdio (npx):    web_search (tavily-mcp)

Usage:
    registry = MCPToolRegistry.get()
    await registry.initialize()
    tools = registry.get_tools_for_role("shogun")  # LangChain BaseTool list
    result = await registry.call_tool("search", {"query": "..."})
"""

import asyncio
import os
from typing import Any, Dict, List, Optional

from utils.logger import get_logger

logger = get_logger(__name__)


def build_mcp_server_configs() -> Dict[str, dict]:
    """
    MCP サーバー接続設定を構築する。

    stdio (npx):    memory / sequential_thinking / filesystem / web_search / playwright
    stdio (python): git  — /home/claude/mcp-git-venv/bin/python -m mcp_server_git
    stdio (binary): github — /home/claude/bin/github-mcp-server stdio
    """
    tavily_key  = os.environ.get("TAVILY_API_KEY", "")
    github_token = os.environ.get("GITHUB_TOKEN", os.environ.get("GITHUB_PERSONAL_ACCESS_TOKEN", ""))
    configs: Dict[str, dict] = {}

    # ── Memory ────────────────────────────────────────────────────────────
    configs["memory"] = {
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-memory"],
    }

    # ── Sequential Thinking ───────────────────────────────────────────────
    configs["sequential_thinking"] = {
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-sequential-thinking"],
    }

    # ── Filesystem ────────────────────────────────────────────────────────
    configs["filesystem"] = {
        "transport": "stdio",
        "command": "npx",
        "args": [
            "-y", "@modelcontextprotocol/server-filesystem",
            "/mnt/Bushidan-Multi-Agent",
            "/mnt/Bushidan",
        ],
    }

    # ── Git (ローカルリポジトリ操作) ─────────────────────────────────────
    # mcp-server-git: /home/claude/mcp-git-venv に独立インストール
    configs["git"] = {
        "transport": "stdio",
        "command": "/home/claude/mcp-git-venv/bin/python",
        "args": ["-m", "mcp_server_git", "--repository", "/mnt/Bushidan-Multi-Agent"],
    }

    # ── GitHub (リモートリポジトリ / PR / Issue / clone / push) ──────────
    # github-mcp-server: /home/claude/bin にバイナリインストール
    if github_token:
        configs["github"] = {
            "transport": "stdio",
            "command": "/home/claude/bin/github-mcp-server",
            "args": ["stdio", "--toolsets=repos,git,pull_requests,issues,users,code_security"],
            "env": {"GITHUB_PERSONAL_ACCESS_TOKEN": github_token},
        }

    # ── Tavily Web 検索 ───────────────────────────────────────────────────
    if tavily_key:
        configs["web_search"] = {
            "transport": "stdio",
            "command": "npx",
            "args": ["-y", "tavily-mcp"],
            "env": {"TAVILY_API_KEY": tavily_key},
        }

    # ── Playwright (ブラウザ自動化 / kengyo 用) ───────────────────────────
    configs["playwright"] = {
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "@playwright/mcp"],
    }

    return configs


class MCPToolRegistry:
    """
    MCP ツールレジストリ — シングルトン

    各MCPサーバーに接続してLangChainツールを取得し、
    ロールごとに許可されたツール一覧を返す。
    """

    _instance: Optional["MCPToolRegistry"] = None

    def __init__(self):
        self._client = None
        self._tools_by_server: Dict[str, list] = {}
        self._all_tools: list = []
        self._initialized = False
        self._server_configs: Dict[str, dict] = {}
        # 遅延初期化: サーバーごとの接続済みフラグ
        self._server_ready: Dict[str, bool] = {}
        self._server_init_locks: Dict[str, asyncio.Lock] = {}

        # ロール → 使用可能MCPサーバーのマッピング
        self._role_servers: Dict[str, List[str]] = {
            "daigensui": ["memory", "sequential_thinking", "filesystem", "git", "github"],
            "shogun":    ["memory", "filesystem", "git", "github"],
            "gunshi":    ["sequential_thinking", "memory"],
            "sanbo":     ["filesystem", "git", "github", "web_search"],
            "gaiji":     ["web_search"],
            "kengyo":    ["playwright", "filesystem"],
            "seppou":    ["web_search"],
            "uketuke":   [],
            "yuhitsu":   [],
            "onmitsu":   ["filesystem", "git"],
        }

    @classmethod
    def get(cls) -> "MCPToolRegistry":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def initialize(self, server_configs: Optional[Dict[str, dict]] = None) -> None:
        """
        サーバー設定をロードする (実際の接続は遅延初期化)。

        server_configs が None の場合は build_mcp_server_configs() を使用。
        """
        if self._initialized:
            return

        if server_configs is None:
            server_configs = build_mcp_server_configs()

        if not server_configs:
            logger.info("MCP サーバー設定なし — ツールレジストリは空")
            self._initialized = True
            return

        self._server_configs = server_configs
        # 各サーバーの遅延初期化ロックを準備
        for name in server_configs:
            self._server_ready[name] = False
            self._server_init_locks[name] = asyncio.Lock()

        try:
            from langchain_mcp_adapters.client import MultiServerMCPClient
            self._client = MultiServerMCPClient(server_configs)
        except ImportError as e:
            logger.warning("⚠️ langchain-mcp-adapters 未インストール: %s", e)

        self._initialized = True
        logger.info("✅ MCPToolRegistry 設定ロード完了: %d サーバー (遅延接続)", len(server_configs))

    async def _ensure_server(self, server_name: str) -> bool:
        """指定サーバーへ遅延接続し、ツールをロードする。"""
        if self._server_ready.get(server_name):
            return True
        if server_name not in self._server_init_locks:
            return False

        lock = self._server_init_locks[server_name]
        async with lock:
            # ダブルチェック
            if self._server_ready.get(server_name):
                return True
            try:
                from langchain_mcp_adapters.tools import load_mcp_tools
                async with self._client.session(server_name) as session:
                    tools = await asyncio.wait_for(
                        load_mcp_tools(session), timeout=20.0
                    )
                self._tools_by_server[server_name] = tools
                self._all_tools.extend(tools)
                self._server_ready[server_name] = True
                logger.info("✅ MCP [%s] 遅延接続: %d tools (%s)",
                            server_name, len(tools),
                            ", ".join(t.name for t in tools[:5]))
                return True
            except Exception as e:
                logger.warning("⚠️ MCP [%s] 遅延接続失敗: %s", server_name, e)
                self._server_ready[server_name] = False
                return False

    def get_tools_for_role(self, role_key: str) -> list:
        """ロールに許可されたLangChain BaseTool一覧を返す (接続済みサーバーのみ)"""
        allowed = self._role_servers.get(role_key, [])
        tools = []
        for srv in allowed:
            tools.extend(self._tools_by_server.get(srv, []))
        return tools

    async def get_tools_for_role_async(self, role_key: str) -> list:
        """ロールに許可されたサーバーへ遅延接続しツール一覧を返す"""
        allowed = self._role_servers.get(role_key, [])
        await asyncio.gather(*[self._ensure_server(s) for s in allowed], return_exceptions=True)
        return self.get_tools_for_role(role_key)

    async def call_tool(self, name: str, args: dict, role_key: Optional[str] = None) -> Any:
        """ツール名で実行。role_key を指定すると許可サーバーのみ遅延接続。"""
        # まず接続済みツールを検索
        for tool in self._all_tools:
            if tool.name == name:
                logger.info("🔧 MCP tool: %s(%s)", name, str(args)[:100])
                return await tool.ainvoke(args)

        if not self._initialized or not self._client:
            raise ValueError(f"MCPToolRegistry 未初期化 — tool: {name}")

        # role_key がある場合は許可サーバーのみ、なければ全未接続サーバーを試みる
        if role_key:
            allowed = self._role_servers.get(role_key, [])
            unready = [s for s in allowed if not self._server_ready.get(s)]
        else:
            unready = [s for s, ok in self._server_ready.items() if not ok]

        if unready:
            await asyncio.gather(*[self._ensure_server(s) for s in unready], return_exceptions=True)
            for tool in self._all_tools:
                if tool.name == name:
                    logger.info("🔧 MCP tool (遅延接続後): %s(%s)", name, str(args)[:100])
                    return await tool.ainvoke(args)

        raise ValueError(f"Unknown MCP tool: {name}")

    @property
    def available_tools(self) -> List[str]:
        return [t.name for t in self._all_tools]

    @property
    def initialized(self) -> bool:
        return self._initialized
