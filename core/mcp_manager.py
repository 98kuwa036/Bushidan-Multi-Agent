"""MCP Manager - 足軽 × 8 (MCPサーバー群) 管理

足軽はLLMではなくMCPサーバー (ツール実行層)。
各足軽は50-150MBの軽量プロセスで、侍大将の指示に従いツールを実行する。

足軽一覧:
  1. filesystem  - ファイル操作
  2. github      - Git/GitHub操作
  3. fetch       - Web情報取得
  4. memory      - 長期記憶
  5. postgres    - データベース
  6. puppeteer   - ブラウザ自動化
  7. brave-search - Web検索
  8. slack       - チーム連携
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
    MCPServer(
        id=8, name="slack",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-slack"],
        env={
            "SLACK_BOT_TOKEN": "${SLACK_BOT_TOKEN}",
            "SLACK_TEAM_ID": "${SLACK_TEAM_ID}",
        },
    ),
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
