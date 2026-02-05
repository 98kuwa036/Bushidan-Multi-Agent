"""
Bushidan Multi-Agent System v10.1 - Smithery MCP Manager

Smithery 経由で MCP サーバーを管理する統合マネージャー。
npm 直接実行を廃止し、Smithery レジストリ経由のインストール・起動に統一。

MCP サーバー一覧:
  AI:
    - Sequential Thinking MCP   (思考チェーン)
  ブラウザ:
    - Playwright MCP             (ブラウザ操作・スクリーンショット)
  検索:
    - Tavily MCP                 (Web検索)
    - Exa MCP                    (セマンティック検索)
  データ:
    - Native Filesystem MCP      (ファイル操作)
    - Graph Memory MCP           (グラフ型記憶)
    - Prisma MCP                 (DB操作)
  連携:
    - Slack MCP                  (Slack連携)
    - Notion MCP                 (Notion連携)
    - Git MCP                    (Git操作)
"""

import asyncio
import json
import logging
import os
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("bushidan.mcp.smithery")


class MCPCategory(Enum):
    """MCP サーバーカテゴリ"""
    AI = "ai"
    BROWSER = "browser"
    SEARCH = "search"
    DATA = "data"
    INTEGRATION = "integration"


@dataclass
class MCPServerConfig:
    """MCP サーバー設定"""
    name: str
    package: str                        # Smithery パッケージ名
    category: MCPCategory
    description: str
    env_vars: Dict[str, str] = field(default_factory=dict)
    required_env: List[str] = field(default_factory=list)
    args: List[str] = field(default_factory=list)
    optional: bool = False              # True = API key なくても起動可能
    enabled: bool = True


# ==================== MCP Server Definitions ====================

MCP_SERVERS: List[MCPServerConfig] = [
    # AI
    MCPServerConfig(
        name="sequential_thinking",
        package="@modelcontextprotocol/server-sequential-thinking",
        category=MCPCategory.AI,
        description="動的思考チェーン: 分岐・修正可能な推論プロセス",
    ),

    # Browser
    MCPServerConfig(
        name="playwright",
        package="@playwright/mcp",
        category=MCPCategory.BROWSER,
        description="ブラウザ操作・スクリーンショット・UI検証",
    ),

    # Search
    MCPServerConfig(
        name="tavily",
        package="tavily-mcp",
        category=MCPCategory.SEARCH,
        description="Web検索 (1,000 searches/month free)",
        required_env=["TAVILY_API_KEY"],
        env_vars={"TAVILY_API_KEY": ""},
    ),
    MCPServerConfig(
        name="exa",
        package="exa-mcp-server",
        category=MCPCategory.SEARCH,
        description="セマンティック検索 (意味ベースの高精度検索)",
        required_env=["EXA_API_KEY"],
        env_vars={"EXA_API_KEY": ""},
        optional=True,
    ),

    # Data
    MCPServerConfig(
        name="filesystem",
        package="@modelcontextprotocol/server-filesystem",
        category=MCPCategory.DATA,
        description="ファイル読み書き・ディレクトリ操作",
        args=["./"],  # Base path
    ),
    MCPServerConfig(
        name="graph_memory",
        package="@mem0ai/mem0-mcp",
        category=MCPCategory.DATA,
        description="グラフ型記憶: エンティティ関係を保持する長期記憶",
        env_vars={"MEM0_API_KEY": ""},
        required_env=["MEM0_API_KEY"],
        optional=True,
    ),
    MCPServerConfig(
        name="prisma",
        package="prisma-mcp-server",
        category=MCPCategory.DATA,
        description="DB操作: Prisma経由のデータベースアクセス",
        optional=True,
    ),

    # Integration
    MCPServerConfig(
        name="slack",
        package="@anthropic/mcp-server-slack",
        category=MCPCategory.INTEGRATION,
        description="Slack連携: メッセージ送受信・チャンネル操作",
        required_env=["SLACK_BOT_TOKEN"],
        env_vars={"SLACK_BOT_TOKEN": ""},
        optional=True,
    ),
    MCPServerConfig(
        name="notion",
        package="@anthropic/mcp-server-notion",
        category=MCPCategory.INTEGRATION,
        description="Notion連携: ページ読み書き・データベース操作",
        required_env=["NOTION_TOKEN"],
        env_vars={"NOTION_TOKEN": ""},
        optional=True,
    ),
    MCPServerConfig(
        name="git",
        package="@modelcontextprotocol/server-git",
        category=MCPCategory.INTEGRATION,
        description="Git操作: コミット・ブランチ・差分管理",
    ),
]


class SmitheryMCPManager:
    """
    Smithery 経由の MCP サーバー管理

    機能:
    - Smithery レジストリからの MCP サーバーインストール
    - 環境変数ベースのサーバー有効/無効判定
    - サブプロセスによるサーバー起動・停止
    - JSON-RPC over stdio による通信
    """

    def __init__(self, config_dir: Optional[str] = None):
        self.config_dir = Path(config_dir or ".")
        self.servers: Dict[str, MCPServerConfig] = {
            s.name: s for s in MCP_SERVERS
        }
        self._processes: Dict[str, asyncio.subprocess.Process] = {}
        self._initialized: Dict[str, bool] = {}

    async def initialize(self) -> Dict[str, bool]:
        """
        全 MCP サーバーの初期化

        環境変数を確認し、必要な API key が設定されているサーバーのみ有効化。

        Returns:
            {server_name: is_available} の辞書
        """
        results = {}

        for name, config in self.servers.items():
            if not config.enabled:
                results[name] = False
                continue

            # 必要な環境変数チェック
            missing_env = [
                var for var in config.required_env
                if not os.getenv(var)
            ]

            if missing_env and not config.optional:
                logger.warning(
                    f"⚠️ MCP {name} requires env vars: {missing_env}"
                )
                results[name] = False
                continue

            if missing_env and config.optional:
                logger.info(
                    f"ℹ️ MCP {name} (optional) skipped: {missing_env}"
                )
                results[name] = False
                continue

            # 環境変数をサーバー設定に反映
            for var_name in config.env_vars:
                config.env_vars[var_name] = os.getenv(var_name, "")

            results[name] = True
            self._initialized[name] = True
            logger.info(f"✅ MCP {name} ({config.description}) ready")

        return results

    async def start_server(self, name: str) -> bool:
        """
        MCP サーバーをサブプロセスとして起動

        Args:
            name: サーバー名

        Returns:
            起動成功フラグ
        """
        if name not in self.servers:
            logger.error(f"❌ Unknown MCP server: {name}")
            return False

        config = self.servers[name]

        if not self._initialized.get(name):
            logger.warning(f"⚠️ MCP {name} not initialized")
            return False

        try:
            # Build command: npx @smithery/cli run <package> [args]
            cmd = ["npx", "-y", config.package]
            cmd.extend(config.args)

            # Set up environment with API keys
            env = os.environ.copy()
            for var_name, var_value in config.env_vars.items():
                if var_value:
                    env[var_name] = var_value

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )

            self._processes[name] = process
            logger.info(f"✅ MCP {name} started (PID: {process.pid})")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to start MCP {name}: {e}")
            return False

    async def stop_server(self, name: str) -> None:
        """MCP サーバー停止"""
        if name in self._processes:
            process = self._processes[name]
            try:
                process.terminate()
                await asyncio.wait_for(process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
            del self._processes[name]
            logger.info(f"✅ MCP {name} stopped")

    async def send_request(
        self,
        server_name: str,
        method: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        MCP サーバーに JSON-RPC リクエスト送信

        Args:
            server_name: サーバー名
            method: JSON-RPC メソッド名
            params: パラメータ

        Returns:
            レスポンス or None
        """
        if server_name not in self._processes:
            logger.error(f"❌ MCP {server_name} not running")
            return None

        process = self._processes[server_name]

        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": params or {},
        }

        try:
            request_bytes = (json.dumps(request) + "\n").encode("utf-8")
            process.stdin.write(request_bytes)
            await process.stdin.drain()

            response_line = await asyncio.wait_for(
                process.stdout.readline(), timeout=30.0
            )
            response = json.loads(response_line.decode("utf-8"))
            return response.get("result")

        except asyncio.TimeoutError:
            logger.error(f"❌ MCP {server_name} request timeout: {method}")
            return None
        except Exception as e:
            logger.error(f"❌ MCP {server_name} request error: {e}")
            return None

    async def shutdown(self) -> None:
        """全サーバー停止"""
        for name in list(self._processes.keys()):
            await self.stop_server(name)
        logger.info("✅ All MCP servers stopped")

    def get_available_servers(self) -> Dict[str, Dict[str, Any]]:
        """利用可能なサーバー一覧"""
        result = {}
        for name, config in self.servers.items():
            result[name] = {
                "category": config.category.value,
                "description": config.description,
                "available": self._initialized.get(name, False),
                "running": name in self._processes,
                "optional": config.optional,
            }
        return result

    def get_server_config(self, name: str) -> Optional[MCPServerConfig]:
        """サーバー設定取得"""
        return self.servers.get(name)


def generate_smithery_config() -> Dict[str, Any]:
    """
    Smithery 設定ファイル (smithery.json) を生成

    Returns:
        Smithery 設定辞書
    """
    config = {
        "mcpServers": {}
    }

    for server in MCP_SERVERS:
        server_config: Dict[str, Any] = {
            "command": "npx",
            "args": ["-y", server.package] + server.args,
        }

        if server.env_vars:
            server_config["env"] = {
                k: f"${{{k}}}" for k in server.env_vars
            }

        config["mcpServers"][server.name] = server_config

    return config
