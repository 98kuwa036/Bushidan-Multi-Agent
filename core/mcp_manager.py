"""MCP Manager - 足軽 × 8+ (MCPサーバー群) 管理 v11.5

足軽はLLMではなくMCPサーバー (ツール実行層)。
各足軽は50-150MBの軽量プロセスで、侍大将の指示に従いツールを実行する。

v11.5 追加機能:
  - TOOL_REGISTRY: 各 MCP サーバーのツール名を静的に管理
  - list_tools(): 実行中サーバーのツール一覧を LangGraph ルーターに提供
  - get_available_tool_names(): 平坦化されたツール名リストを返す

足軽一覧:
  1. filesystem   - ファイル操作
  2. github       - Git/GitHub操作
  3. fetch        - Web情報取得
  4. memory       - 長期記憶
  5. postgres     - データベース
  6. puppeteer    - ブラウザ自動化
  7. brave-search - Web検索
  8. tavily       - Web検索 (高精度)
  9. exa          - セマンティック検索
  10. mattermost  - チーム連携 (mcp/mattermost_mcp_server.py)
  11. notion      - Notion 連携
"""

import asyncio
import json
import logging
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("shogun.mcp_manager")


@dataclass
class MCPServer:
    """Single MCP server (足軽) definition."""
    id: int
    name: str
    command: str
    args: list[str]
    env: dict[str, str] = field(default_factory=dict)
    status: str = "stopped"  # stopped | running | error
    process: Any = field(default=None, repr=False)


# Default MCP server definitions (足軽 × 8)
DEFAULT_SERVERS = [
    MCPServer(
        id=1, name="filesystem",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", "/home/claude"],
    ),
    MCPServer(
        id=2, name="github",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-github"],
        env={"GITHUB_TOKEN": "${GITHUB_TOKEN}"},
    ),
    MCPServer(
        id=3, name="fetch",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-fetch"],
    ),
    MCPServer(
        id=4, name="memory",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-memory"],
    ),
    MCPServer(
        id=5, name="postgres",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-postgres"],
        env={"DATABASE_URL": "${DATABASE_URL}"},
    ),
    MCPServer(
        id=6, name="puppeteer",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-puppeteer"],
    ),
    MCPServer(
        id=7, name="brave-search",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-brave-search"],
        env={"BRAVE_API_KEY": "${BRAVE_API_KEY}"},
    ),
    # Discord 連携は bushidan/discord_bot.py で直接処理 (MCP不要)
    # MCPServer(id=8, name="discord", ...) は不使用
]


class MCPManager:
    """MCP Server (足軽) lifecycle manager.

    In the Shogun system, MCP servers act as tool-execution agents.
    The 侍大将 (Taisho) coordinates them via the controller.
    """

    def __init__(self, config_path: str | None = None):
        self.servers: dict[str, MCPServer] = {}
        self._load_servers(config_path)

    def _load_servers(self, config_path: str | None) -> None:
        """Load MCP server definitions from config or defaults."""
        if config_path and Path(config_path).exists():
            try:
                data = json.loads(Path(config_path).read_text())
                for name, cfg in data.get("mcpServers", {}).items():
                    idx = len(self.servers) + 1
                    self.servers[name] = MCPServer(
                        id=idx,
                        name=name,
                        command=cfg["command"],
                        args=cfg.get("args", []),
                        env=cfg.get("env", {}),
                    )
                logger.info("Loaded %d MCP servers from %s", len(self.servers), config_path)
                return
            except Exception as e:
                logger.warning("Failed to load MCP config: %s", e)

        # Use defaults
        for srv in DEFAULT_SERVERS:
            self.servers[srv.name] = MCPServer(
                id=srv.id,
                name=srv.name,
                command=srv.command,
                args=list(srv.args),
                env=dict(srv.env),
            )
        logger.info("Using %d default MCP server definitions", len(self.servers))

    def _resolve_env(self, env: dict[str, str]) -> dict[str, str]:
        """Resolve ${VAR} references in env values."""
        resolved = {}
        for key, val in env.items():
            if val.startswith("${") and val.endswith("}"):
                env_var = val[2:-1]
                resolved[key] = os.environ.get(env_var, "")
            else:
                resolved[key] = val
        return resolved

    async def start_server(self, name: str) -> bool:
        """Start a single MCP server."""
        srv = self.servers.get(name)
        if not srv:
            logger.error("Unknown MCP server: %s", name)
            return False

        if srv.status == "running":
            logger.info("MCP server already running: %s", name)
            return True

        env = {**os.environ, **self._resolve_env(srv.env)}

        try:
            proc = await asyncio.create_subprocess_exec(
                srv.command, *srv.args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
            srv.process = proc
            srv.status = "running"
            logger.info("[足軽%d] %s started (PID %d)", srv.id, name, proc.pid)
            return True
        except Exception as e:
            srv.status = "error"
            logger.error("[足軽%d] %s failed to start: %s", srv.id, name, e)
            return False

    async def stop_server(self, name: str) -> None:
        """Stop a single MCP server."""
        srv = self.servers.get(name)
        if not srv or not srv.process:
            return
        try:
            srv.process.terminate()
            await asyncio.wait_for(srv.process.wait(), timeout=5)
        except asyncio.TimeoutError:
            srv.process.kill()
        srv.status = "stopped"
        srv.process = None
        logger.info("[足軽%d] %s stopped", srv.id, name)

    async def start_all(self) -> dict[str, bool]:
        """Start all MCP servers."""
        results = {}
        for name in self.servers:
            results[name] = await self.start_server(name)
        return results

    async def stop_all(self) -> None:
        """Stop all MCP servers."""
        for name in list(self.servers.keys()):
            await self.stop_server(name)

    def get_status(self) -> list[dict]:
        """Get status of all MCP servers."""
        return [
            {
                "id": srv.id,
                "name": srv.name,
                "status": srv.status,
                "pid": srv.process.pid if srv.process else None,
            }
            for srv in self.servers.values()
        ]

    def get_mcp_config_json(self) -> dict:
        """Generate MCP config JSON for claude CLI integration."""
        config = {"mcpServers": {}}
        for name, srv in self.servers.items():
            config["mcpServers"][name] = {
                "command": srv.command,
                "args": srv.args,
            }
            if srv.env:
                config["mcpServers"][name]["env"] = srv.env
        return config

    # =========================================================================
    # v11.5 新機能: ツール認識 (LangGraph Router 連携用)
    # =========================================================================

    #: 各 MCP サーバーが提供するツール名の静的レジストリ
    #: LangGraph ルーターが実行中サーバーを把握し、ルーティングに活用する
    TOOL_REGISTRY: dict[str, list[str]] = {
        "filesystem": [
            "read_file", "write_file", "edit_file", "list_directory",
            "create_directory", "move_file", "search_files", "get_file_info",
        ],
        "github": [
            "create_or_update_file", "search_repositories", "create_repository",
            "get_file_contents", "push_files", "create_issue", "create_pull_request",
            "list_commits", "list_branches",
        ],
        "fetch": [
            "fetch",
        ],
        "memory": [
            "create_entities", "create_relations", "add_observations",
            "delete_entities", "delete_observations", "delete_relations",
            "read_graph", "search_nodes", "open_nodes",
        ],
        "postgres": [
            "query", "execute",
        ],
        "puppeteer": [
            "puppeteer_navigate", "puppeteer_screenshot", "puppeteer_click",
            "puppeteer_fill", "puppeteer_evaluate",
        ],
        "brave-search": [
            "brave_web_search", "brave_local_search",
        ],
        "tavily": [
            "search", "search_context", "search_qna",
        ],
        "exa": [
            "search", "find_similar", "get_contents",
        ],
        "mattermost": [
            "post_message", "post_direct_message", "get_channel_messages",
            "search_messages", "add_reaction", "create_channel",
            "get_team_channels", "submit_task", "get_bushidan_status",
            "report_agent_progress",
        ],
        "notion": [
            "notion_get_database", "notion_query_database", "notion_get_page",
            "notion_create_page", "notion_update_page", "notion_search",
        ],
        "sequential_thinking": [
            "sequentialthinking",
        ],
        "git": [
            "git_status", "git_diff", "git_commit", "git_add", "git_reset",
            "git_log", "git_create_branch", "git_checkout",
        ],
    }

    def list_tools(self) -> dict[str, list[str]]:
        """
        実行中 MCP サーバーのツール一覧を返す。

        v11.5 LangGraph Router の fetch_context ノードから呼び出され、
        ルーティング判断に使用される。

        Returns:
            {server_name: [tool_name, ...]} 形式の辞書 (running サーバーのみ)
        """
        result: dict[str, list[str]] = {}
        for name, srv in self.servers.items():
            if srv.status == "running":
                tools = self.TOOL_REGISTRY.get(name, [])
                if tools:
                    result[name] = tools
        return result

    def get_available_tool_names(self) -> list[str]:
        """
        実行中サーバーの全ツール名を平坦化したリストで返す。

        LangGraph ルーターの route_decision で available_tools として使用。
        """
        names: list[str] = []
        for tools in self.list_tools().values():
            names.extend(tools)
        return names

    def is_tool_available(self, tool_name: str) -> bool:
        """指定ツール名が実行中のいずれかのサーバーで提供されているか確認."""
        return tool_name in self.get_available_tool_names()

    def get_server_for_tool(self, tool_name: str) -> str | None:
        """ツール名からサーバー名を逆引き (実行中サーバーのみ)."""
        for server_name, tools in self.list_tools().items():
            if tool_name in tools:
                return server_name
        return None
