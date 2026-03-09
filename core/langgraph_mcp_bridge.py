"""
Bushidan Multi-Agent System v12 - LangGraph MCP Bridge
MCP と LangGraph StateGraph の統合レイヤー

langchain-mcp-adapters を使用して MCP サーバーのツールを
LangChain/LangGraph ノードから直接呼び出せるようにする。

機能:
- MultiServerMCPClient: 複数 MCP サーバーへの一括接続
- 役職別ツールフィルタリング: MCPPermissionManager と連携
- Sequential Thinking: 軍師タスク分解の強化
- Graph Memory: 委譲チェーンの永続化・検索
- Notion: リアルタイム外部ダッシュボード
- Filesystem/Git: 成果物のバージョン管理
"""

import asyncio
from typing import Any, Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass

from utils.logger import get_logger

logger = get_logger(__name__)


class MCPIntegrationError(Exception):
    """Base exception for MCP integration failures"""
    pass


class DelegationTrackingError(MCPIntegrationError):
    """Failed to track delegation chain"""
    pass


class NotionSyncError(MCPIntegrationError):
    """Failed to sync to Notion"""
    pass


class ArtifactTrackingError(MCPIntegrationError):
    """Failed to track artifacts"""
    pass


@dataclass
class DelegationEntry:
    """Delegation chain entry"""
    timestamp: str
    task_id: str
    from_agent: str
    to_agent: str
    reason: str
    task_content: str
    state_snapshot: Dict[str, Any]


class LangGraphMCPBridge:
    """
    Bridge between LangGraph StateGraph and MCP servers.

    langchain-mcp-adapters の MultiServerMCPClient を使って
    MCP サーバーのツールを LangChain ツールとして取得し、
    LangGraph ノードで直接呼び出せるようにする。

    Responsibilities:
    1. MCP サーバー接続管理 (MultiServerMCPClient)
    2. 役職別ツールフィルタリング (MCPPermissionManager)
    3. Graph Memory: Store/query delegation chains
    4. Sequential Thinking: Enhanced task analysis
    5. Notion: Real-time dashboard updates
    6. Filesystem/Git: Artifact version control
    """

    def __init__(self, orchestrator):
        """
        Initialize MCP bridge.

        Args:
            orchestrator: SystemOrchestrator instance
        """
        self.orchestrator = orchestrator

        # Legacy custom MCPs (カスタム実装)
        self.memory_mcp = orchestrator.get_mcp("memory")
        self.fs_mcp = orchestrator.get_mcp("filesystem")
        self.git_mcp = orchestrator.get_mcp("git")

        # langchain-mcp-adapters クライアント (initialize() で初期化)
        self._lc_mcp_client = None  # MultiServerMCPClient
        self._lc_tools: list = []    # LangChain BaseTool のリスト

        # Notion configuration
        self.notion_database_id: Optional[str] = None

        logger.info("🔗 LangGraph MCP Bridge 準備完了 (initialize() 待機中)")

    # =========================================================================
    # 初期化 / シャットダウン
    # =========================================================================

    async def initialize(self) -> None:
        """
        MCP サーバーに接続してツールを取得。

        SmitheryMCPManager の初期化済みサーバー設定を読み込み、
        langchain-mcp-adapters の MultiServerMCPClient で接続する。
        """
        try:
            from langchain_mcp_adapters.client import MultiServerMCPClient
        except ImportError:
            logger.warning(
                "⚠️ langchain-mcp-adapters 未インストール。"
                "'pip install langchain-mcp-adapters' を実行してください。"
                " MCP ツール統合をスキップします。"
            )
            return

        server_configs = self._build_server_configs()
        if not server_configs:
            logger.info("ℹ️ 接続可能な MCP サーバーなし (スキップ)")
            return

        try:
            self._lc_mcp_client = MultiServerMCPClient(server_configs)
            # v0.1.0+ はコンテキストマネージャー不要 — 直接 get_tools() を呼ぶ
            self._lc_tools = await self._lc_mcp_client.get_tools()

            # Notion クライアントをツールから取得
            self._setup_notion_client()

            logger.info(
                "✅ LangGraph MCP Bridge 初期化完了: %d ツール (%d サーバー)",
                len(self._lc_tools),
                len(server_configs),
            )
            for tool in self._lc_tools:
                logger.debug("  🔧 Tool: %s", tool.name)

        except Exception as e:
            logger.warning("⚠️ MCP Bridge 接続失敗: %s (MCP ツールなしで続行)", e)
            self._lc_mcp_client = None
            self._lc_tools = []

    def _build_server_configs(self) -> Dict[str, Any]:
        """
        SmitheryMCPManager の初期化済みサーバーから
        MultiServerMCPClient 用の設定辞書を構築する。
        """
        smithery = getattr(self.orchestrator, "smithery_mcp", None)
        if not smithery:
            return {}

        configs = {}
        for name, server_cfg in smithery.servers.items():
            # 初期化済みのサーバーのみ接続
            if not smithery._initialized.get(name):
                continue

            entry: Dict[str, Any] = {
                "command": "npx",
                "args": ["-y", server_cfg.package] + server_cfg.args,
                "transport": "stdio",
            }

            # 環境変数が設定済みの場合のみ追加
            env = {k: v for k, v in server_cfg.env_vars.items() if v}
            if env:
                entry["env"] = env

            configs[name] = entry
            logger.debug("📡 MCP サーバー設定: %s -> %s", name, server_cfg.package)

        return configs

    def _setup_notion_client(self) -> None:
        """ツール一覧から Notion ツールを検出して設定"""
        notion_tools = [t for t in self._lc_tools if "notion" in t.name.lower()]
        if notion_tools:
            logger.info("✅ Notion MCP ツール検出: %d 個", len(notion_tools))
        else:
            logger.debug("ℹ️ Notion MCP ツールなし")

    async def shutdown(self) -> None:
        """MCP クライアント接続を閉じる"""
        if self._lc_mcp_client:
            try:
                # aclose() があれば呼ぶ (v0.1.0+ 対応)
                if hasattr(self._lc_mcp_client, "aclose"):
                    await self._lc_mcp_client.aclose()
                logger.info("✅ MCP Bridge シャットダウン完了")
            except Exception as e:
                logger.warning("⚠️ MCP Bridge シャットダウンエラー: %s", e)

    # =========================================================================
    # ツールアクセス API (LangGraph ノード用)
    # =========================================================================

    def get_tools(self, role: Optional[str] = None) -> list:
        """
        LangGraph ノード用の MCP ツール一覧を返す。

        Args:
            role: 役職名 (指定時は権限フィルタ適用)

        Returns:
            LangChain BaseTool のリスト
        """
        if not role:
            return self._lc_tools

        smithery = getattr(self.orchestrator, "smithery_mcp", None)
        if not smithery:
            return self._lc_tools

        # 役職が使用可能なサーバー名のセットを取得
        allowed_servers = {
            name for name in smithery.servers
            if smithery.check_role_permission(role, name)
        }

        # ツール名 "server_name__tool_name" 形式でフィルタリング
        return [
            t for t in self._lc_tools
            if any(t.name.startswith(s) for s in allowed_servers)
        ]

    def get_tool_schemas(self) -> Dict[str, List[str]]:
        """
        LangGraph ルーター用のツールスキーマ辞書を返す。

        Returns:
            {server_name: [tool_name, ...]} の辞書
        """
        schemas: Dict[str, List[str]] = {}
        for tool in self._lc_tools:
            # langchain-mcp-adapters のツール名: "server_name__tool_name"
            parts = tool.name.split("__", 1)
            server = parts[0] if len(parts) > 1 else "default"
            schemas.setdefault(server, []).append(tool.name)
        return schemas

    def get_tool_by_name(self, tool_name: str):
        """ツール名でツールオブジェクトを取得"""
        for tool in self._lc_tools:
            if tool.name == tool_name or tool_name in tool.name:
                return tool
        return None

    def is_available(self) -> bool:
        """MCP ツールが利用可能か"""
        return len(self._lc_tools) > 0

    # =========================================================================
    # Sequential Thinking: Enhanced Analysis
    # =========================================================================

    async def analyze_with_sequential_thinking(
        self,
        task_content: str,
        context: dict,
    ) -> Dict[str, Any]:
        """
        Sequential Thinking MCP で深層分析を実行。

        Args:
            task_content: タスク内容
            context: 追加コンテキスト

        Returns:
            分析結果辞書、または空辞書 (失敗時)
        """
        st_tool = self.get_tool_by_name("sequential")
        if not st_tool:
            logger.debug("Sequential Thinking ツール未検出 (スキップ)")
            return {}

        try:
            prompt = (
                f"以下のタスクをステップバイステップで分解して分析してください:\n\n"
                f"{task_content}\n\n"
                f"出力形式: JSON (steps, risks, dependencies, estimated_complexity)"
            )
            result = await st_tool.ainvoke({"thought": prompt})
            logger.info("🧠 Sequential Thinking 分析完了")
            return {"analysis": result, "source": "sequential_thinking"}

        except Exception as e:
            logger.warning("⚠️ Sequential Thinking 失敗: %s", e)
            return {}

    # =========================================================================
    # Graph Memory: Delegation Chain Tracking
    # =========================================================================

    async def track_delegation(
        self,
        task_id: str,
        from_agent: str,
        to_agent: str,
        state: Dict[str, Any],
        reason: str,
    ) -> None:
        """
        委譲チェーンを Graph Memory MCP に記録。

        Args:
            task_id: タスク識別子
            from_agent: 委譲元エージェント
            to_agent: 委譲先エージェント
            state: 現在の TaskState スナップショット
            reason: 委譲理由
        """
        if not self.memory_mcp:
            logger.warning("⚠️ Memory MCP 未設定 (委譲追跡スキップ)")
            return

        try:
            entry = {
                "timestamp": datetime.now().isoformat(),
                "category": "delegation_chain",
                "task_id": task_id,
                "from_agent": from_agent,
                "to_agent": to_agent,
                "reason": reason,
                "task_content": state.get("content", "")[:500],
                "state_snapshot": {
                    "complexity":    state.get("complexity"),
                    "is_multi_step": state.get("is_multi_step"),
                    "is_action_task": state.get("is_action_task"),
                    "route":         state.get("route"),
                    "confidence":    state.get("confidence"),
                },
            }

            await self.memory_mcp.store_memory(
                content=f"Delegation: {from_agent} → {to_agent} ({reason})",
                metadata=entry,
            )
            logger.info("📝 委譲を Graph Memory に記録: %s → %s", from_agent, to_agent)

        except Exception as e:
            logger.warning("⚠️ 委譲追跡失敗: %s (タスク実行は継続)", e)

    async def query_delegation_chain(
        self,
        task_id: Optional[str] = None,
        from_agent: Optional[str] = None,
        to_agent: Optional[str] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        過去の委譲チェーンを検索。

        Returns:
            委譲エントリのリスト
        """
        if not self.memory_mcp:
            return []

        try:
            query_parts = ["delegation_chain"]
            if task_id:
                query_parts.append(task_id)
            if from_agent:
                query_parts.append(from_agent)
            if to_agent:
                query_parts.append(to_agent)

            results = await self.memory_mcp.search_memory(
                " ".join(query_parts), limit=limit
            )
            logger.info("📊 委譲エントリ検索: %d 件", len(results))
            return results

        except Exception as e:
            logger.error("❌ 委譲チェーン検索失敗: %s", e)
            return []

    # =========================================================================
    # Notion: Dashboard Sync
    # =========================================================================

    async def sync_to_notion(
        self,
        task_id: str,
        state: Dict[str, Any],
        phase: str,
    ) -> None:
        """
        Notion ダッシュボードにタスク状態を同期。

        Notion MCP ツールを使用してページを作成・更新する。
        """
        notion_tools = [t for t in self._lc_tools if "notion" in t.name.lower()]
        if not notion_tools or not self.notion_database_id:
            logger.debug("Notion MCP 未設定 (同期スキップ)")
            return

        try:
            page_id = state.get("notion_page_id")
            if not page_id:
                page_id = await self._create_notion_page_via_mcp(
                    task_id, state, notion_tools
                )
                state["notion_page_id"] = page_id

            await self._update_notion_page_via_mcp(
                page_id, state, phase, notion_tools
            )
            logger.info("📊 Notion 同期完了: %s", phase)

        except Exception as e:
            logger.warning("⚠️ Notion 同期失敗: %s (タスク実行は継続)", e)

    async def _create_notion_page_via_mcp(
        self,
        task_id: str,
        state: Dict[str, Any],
        notion_tools: list,
    ) -> str:
        """Notion MCP ツールで新規ページを作成"""
        create_tool = next(
            (t for t in notion_tools if "create" in t.name.lower()), notion_tools[0]
        )
        result = await create_tool.ainvoke({
            "database_id": self.notion_database_id,
            "title": f"Task: {task_id}",
            "content": state.get("content", "")[:2000],
            "status": "In Progress",
        })
        page_id = result.get("id", "") if isinstance(result, dict) else ""
        logger.info("📄 Notion ページ作成: %s", page_id)
        return page_id

    async def _update_notion_page_via_mcp(
        self,
        page_id: str,
        state: Dict[str, Any],
        phase: str,
        notion_tools: list,
    ) -> None:
        """Notion MCP ツールで既存ページを更新"""
        chain = state.get("delegation_chain", [])
        chain_str = (
            " → ".join(d.get("to", "") for d in chain)
            if isinstance(chain, list) and chain
            else "N/A"
        )
        update_tool = next(
            (t for t in notion_tools if "update" in t.name.lower()), notion_tools[0]
        )
        await update_tool.ainvoke({
            "page_id": page_id,
            "phase": phase.capitalize(),
            "delegation_chain": chain_str,
            "status": state.get("status", "pending").capitalize(),
            "updated_at": datetime.now().isoformat(),
        })

    # =========================================================================
    # Filesystem/Git: Artifact Tracking
    # =========================================================================

    async def track_artifact(
        self,
        task_id: str,
        file_path: str,
        operation: str,  # "create" | "modify" | "delete"
        content_preview: Optional[str] = None,
    ) -> None:
        """
        タスク成果物のファイル操作を追跡。

        1. Graph Memory MCP (永続ログ)
        2. Git MCP (バージョン管理)
        """
        timestamp = datetime.now().isoformat()

        # 1. Graph Memory
        if self.memory_mcp:
            try:
                entry = {
                    "timestamp": timestamp,
                    "category": "artifact",
                    "task_id": task_id,
                    "file_path": file_path,
                    "operation": operation,
                    "content_preview": (content_preview[:500] if content_preview else None),
                }
                await self.memory_mcp.store_memory(
                    content=f"Artifact {operation}: {file_path}",
                    metadata=entry,
                )
                logger.info("📦 成果物追跡: %s %s", operation, file_path)
            except Exception as e:
                logger.warning("⚠️ 成果物追跡失敗: %s", e)

        # 2. Git MCP
        if self.git_mcp and operation in ("create", "modify"):
            try:
                await self.git_mcp.stage_file(file_path)
                logger.info("📦 Git ステージング: %s", file_path)
            except Exception as e:
                logger.warning("⚠️ Git ステージング失敗: %s", e)

    async def commit_task_artifacts(
        self,
        task_id: str,
        message: str,
    ) -> Optional[str]:
        """
        タスク成果物をコミット。

        Returns:
            コミットハッシュ、または None (失敗時)
        """
        if not self.git_mcp:
            return None

        try:
            commit_hash = await self.git_mcp.commit(
                message=f"[Task {task_id}] {message}"
            )
            logger.info("✅ 成果物コミット完了: %s", commit_hash)
            return commit_hash
        except Exception as e:
            logger.error("❌ Git コミット失敗: %s", e)
            return None
