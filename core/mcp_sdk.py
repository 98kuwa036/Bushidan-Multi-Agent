"""
core/mcp_sdk.py — MCP ツールレジストリ v14

MCPサーバーへの統一アクセスレイヤー。
langchain-mcp-adapters でMCPサーバーに接続し、
ロールごとの権限フィルタ付きツール取得を提供する。

Usage:
    registry = MCPToolRegistry.get()
    await registry.initialize(server_configs)
    tools = registry.get_tools_for_role("shogun")
    result = await registry.call_tool("search_web", {"query": "..."})
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class MCPTool:
    """MCPツール情報"""
    name: str
    description: str
    server: str
    input_schema: dict


class MCPToolRegistry:
    """
    MCP ツールレジストリ — シングルトン

    MCPサーバーに接続し、ツール一覧を管理する。
    """

    _instance: Optional["MCPToolRegistry"] = None

    def __init__(self):
        self._tools: Dict[str, MCPTool] = {}
        self._servers: Dict[str, Any] = {}
        self._initialized = False
        # ロール → 使用可能MCPサーバーのマッピング
        self._role_servers: Dict[str, List[str]] = {
            "daigensui": ["memory", "notion", "sequential_thinking", "filesystem", "git"],
            "shogun":    ["memory", "notion", "filesystem", "git", "prisma"],
            "gunshi":    ["sequential_thinking", "filesystem", "git", "memory"],
            "sanbo":     ["filesystem", "git", "prisma", "web_search"],
            "gaiji":     ["web_search", "filesystem"],
            "kengyo":    ["playwright", "filesystem"],
            "seppou":    ["web_search"],
            "uketuke":   ["web_search"],
            "yuhitsu":   [],
            "onmitsu":   ["filesystem", "git"],
        }

    @classmethod
    def get(cls) -> "MCPToolRegistry":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def initialize(self, server_configs: Optional[Dict[str, dict]] = None) -> None:
        """MCPサーバーに接続してツール一覧を取得"""
        if self._initialized:
            return

        if not server_configs:
            logger.debug("MCP サーバー設定なし — ツールレジストリは空で初期化")
            self._initialized = True
            return

        try:
            from langchain_mcp_adapters.client import MultiServerMCPClient

            client = MultiServerMCPClient(server_configs)
            lc_tools = await client.get_tools()

            for tool in lc_tools:
                mcp_tool = MCPTool(
                    name=tool.name,
                    description=getattr(tool, "description", ""),
                    server="",
                    input_schema=getattr(tool, "args_schema", {}) if hasattr(tool, "args_schema") else {},
                )
                self._tools[tool.name] = mcp_tool

            self._initialized = True
            logger.info("✅ MCP ツールレジストリ初期化完了: %d tools", len(self._tools))

        except Exception as e:
            logger.warning("⚠️ MCP ツールレジストリ初期化スキップ: %s", e)
            self._initialized = True

    def get_tools_for_role(self, role_key: str) -> List[MCPTool]:
        """ロールに許可されたツール一覧を返す"""
        allowed_servers = self._role_servers.get(role_key, [])
        if not allowed_servers:
            return []
        return [
            tool for tool in self._tools.values()
            if tool.server in allowed_servers or not tool.server
        ]

    async def call_tool(self, name: str, args: dict) -> Any:
        """ツールを実行"""
        tool = self._tools.get(name)
        if not tool:
            raise ValueError(f"Unknown MCP tool: {name}")
        logger.info("🔧 MCP tool call: %s(%s)", name, str(args)[:100])
        # langchain tool invocation
        try:
            from langchain_mcp_adapters.client import MultiServerMCPClient
            # Tool execution would go through the MCP client
            # This is a placeholder — actual execution depends on server connection
            logger.warning("MCP tool execution not yet connected: %s", name)
            return None
        except Exception as e:
            logger.error("MCP tool %s 実行失敗: %s", name, e)
            raise

    @property
    def available_tools(self) -> List[str]:
        return list(self._tools.keys())
