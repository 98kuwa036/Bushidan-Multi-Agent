"""
Bushidan Multi-Agent System v18 - System Orchestrator (簡素化)

MCP サーバー初期化 / LangGraph Router 初期化 / ヘルスチェック / process_task 委譲

10役職体制:
  受付(Gemini 3.1 Flash-Lite) / 外事(Command A) / 検校(Gemini 3.1 Flash-Image)
  将軍(Claude Sonnet 4.6) / 軍師(Mistral Large 3) / 参謀(Mistral Large 3)
  右筆(Gemini 3.1 Flash-Lite) / 斥候(Llama Groq) / 隠密(Gemma3/Nemotron)
  大元帥(Claude Opus 4.6)
"""

from typing import Any, Dict, Optional
from dataclasses import dataclass
from enum import Enum

from utils.logger import get_logger

logger = get_logger(__name__)


class SystemMode(Enum):
    """System modes"""
    BATTALION = "battalion"
    COMPANY = "company"
    PLATOON = "platoon"


@dataclass
class SystemConfig:
    """v15 システム設定"""
    mode: SystemMode
    claude_api_key: str
    gemini_api_key: str
    tavily_api_key: str

    groq_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    mistral_api_key: Optional[str] = None
    xai_api_key: Optional[str] = None

    notion_token: Optional[str] = None

    llamacpp_endpoint: str = "http://127.0.0.1:8080"
    llamacpp_model_path: str = "models/nemotron/Nemotron-3-Nano-Q4_K_M.gguf"
    llamacpp_threads: int = 6
    llamacpp_context_size: int = 4096
    llamacpp_batch_size: int = 512
    llamacpp_mlock: bool = True

    version: str = "14"
    intelligent_routing_enabled: bool = True
    prompt_caching_enabled: bool = True
    power_optimization_enabled: bool = True
    use_llamacpp: bool = True

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


class SystemOrchestrator:
    """
    v15 システムオーケストレーター

    MCP 管理 + LangGraph Router 初期化 + ヘルスチェック + process_task 委譲
    """

    VERSION = "15"

    def __init__(self, config: SystemConfig):
        self.config = config
        self.mcps: Dict[str, Any] = {}
        self.initialized = False
        self.health_status: Dict[str, bool] = {}
        self._langgraph_router = None
        self._mcp_bridge = None

    async def initialize(self) -> None:
        """初期化 — MCP + LangGraph Router"""
        logger.info("🔧 武士団 v%s 初期化開始...", self.VERSION)

        try:
            await self._initialize_mcps()
            await self._check_health()
            await self._initialize_langgraph()
            self.initialized = True
            logger.info("✅ システムオーケストレーター v%s 初期化完了", self.VERSION)
            self._log_startup_summary()
        except Exception as e:
            logger.error("❌ システム初期化失敗: %s", e)
            raise

    async def _initialize_mcps(self) -> None:
        """MCPサーバー初期化"""
        try:
            from mcp_servers.memory_mcp import MemoryMCP
            self.mcps["memory"] = MemoryMCP()
        except Exception as e:
            logger.warning("⚠️ Memory MCP: %s", e)

        try:
            from mcp_servers.filesystem_mcp import FilesystemMCP
            self.mcps["filesystem"] = FilesystemMCP("/mnt/Bushidan-Multi-Agent")
        except Exception as e:
            logger.warning("⚠️ Filesystem MCP: %s", e)

        try:
            from mcp_servers.git_mcp import GitMCP
            self.mcps["git"] = GitMCP()
        except Exception as e:
            logger.warning("⚠️ Git MCP: %s", e)

        if self.config.tavily_api_key:
            try:
                from mcp_servers.web_search_mcp import SmartWebSearchMCP as WebSearchMCP
                self.mcps["web_search"] = WebSearchMCP(self.config.tavily_api_key)
            except Exception as e:
                logger.warning("⚠️ Web Search MCP: %s", e)

        for name, mcp in self.mcps.items():
            try:
                await mcp.initialize()
                logger.info("✅ %s MCP 初期化完了", name)
            except Exception as e:
                logger.warning("⚠️ %s MCP 初期化失敗: %s", name, e)

    async def _check_health(self) -> None:
        """LLM 可用性確認"""
        try:
            from core.liveness_checker import LLMAvailabilityChecker
            checker = LLMAvailabilityChecker()
            statuses = await checker.check_all()
            logger.info(checker.print_summary())
            # LLM可用性をhealth_statusに保存
            self.health_status.update({
                name: status.available for name, status in statuses.items()
            })
        except Exception as e:
            logger.warning("⚠️ LLM 可用性確認スキップ: %s", e)

        # llama.cpp ヘルスチェック
        if self.config.use_llamacpp:
            try:
                import httpx
                async with httpx.AsyncClient(timeout=5.0) as client:
                    resp = await client.get(f"{self.config.llamacpp_endpoint}/health")
                    self.health_status["llamacpp"] = resp.status_code == 200
            except Exception:
                self.health_status["llamacpp"] = False

    async def _initialize_langgraph(self) -> None:
        """LangGraph Router 初期化"""
        try:
            from core.langgraph_router import LangGraphRouter
            self._langgraph_router = LangGraphRouter(self)
            await self._langgraph_router.initialize()
            logger.info("🔗 LangGraph Router 初期化完了")
        except Exception as e:
            import traceback
            logger.error("❌ LangGraph Router 初期化失敗: %s\n%s", e, traceback.format_exc())
            self._langgraph_router = None

        # MCP-LangGraph Bridge
        try:
            from core.langgraph_mcp_bridge import LangGraphMCPBridge
            self._mcp_bridge = LangGraphMCPBridge(self)
            await self._mcp_bridge.initialize()
            logger.info("🔗 MCP-LangGraph Bridge 初期化完了")
        except Exception as e:
            logger.warning("⚠️ MCP Bridge 初期化スキップ: %s", e)
            self._mcp_bridge = None

    def _log_startup_summary(self) -> None:
        """起動サマリー"""
        logger.info("=" * 60)
        logger.info("🏯 武士団マルチエージェントシステム v%s", self.VERSION)
        logger.info("  モード: %s", self.config.mode.value)
        logger.info("  LangGraph Router: %s", "✅" if self._langgraph_router else "❌")
        logger.info("  MCP サーバー: %s", ", ".join(self.mcps.keys()) or "なし")
        logger.info("  llama.cpp: %s", "✅" if self.health_status.get("llamacpp") else "⚠️")
        logger.info("=" * 60)

    # ── プロパティ ──────────────────────────────────────────────────

    @property
    def langgraph_router(self):
        return self._langgraph_router

    @property
    def mcp_bridge(self):
        return self._mcp_bridge

    def get_mcp(self, name: str) -> Optional[Any]:
        return self.mcps.get(name)

    def get_client(self, name: str) -> Optional[Any]:
        """後方互換: ClientRegistry に委譲"""
        from utils.client_registry import ClientRegistry
        return ClientRegistry.get().get_client(name)

    def get_router(self):
        return self._langgraph_router

    # ── タスク処理 ──────────────────────────────────────────────────

    async def process_task(
        self, task_content: str, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """LangGraph Router に委譲"""
        if not self.initialized:
            raise RuntimeError("システムが初期化されていません")

        if self._langgraph_router:
            return await self._langgraph_router.process_task(
                content=task_content,
                context=context or {},
                source="orchestrator",
            )

        raise RuntimeError("LangGraph Router が初期化されていません")

    def get_all_statistics(self) -> Dict[str, Any]:
        return {
            "version": self.VERSION,
            "mode": self.config.mode.value,
            "health_status": self.health_status,
            "mcps": list(self.mcps.keys()),
        }

    async def shutdown(self) -> None:
        """グレースフルシャットダウン"""
        logger.info("📴 システムオーケストレーター v%s シャットダウン...", self.VERSION)
        for name, mcp in self.mcps.items():
            try:
                await mcp.shutdown()
            except Exception as e:
                logger.warning("⚠️ %s MCP シャットダウンエラー: %s", name, e)
        logger.info("✅ シャットダウン完了")
