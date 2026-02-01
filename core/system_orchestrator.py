"""
Bushidan Multi-Agent System v9.3.2 - System Orchestrator

Enhanced system coordination for the 4-Tier Hybrid Architecture.
Manages: Shogun -> Karo -> Taisho -> Ashigaru with Intelligent Routing.

v9.3.2 Enhancements:
- Intelligent Router integration
- New client initialization (ClaudeClientCached, Groq, Gemini3, Qwen3, AlibabaQwen)
- 3-tier fallback chain management
- Power-saving optimization
- Enhanced health checks
"""

import asyncio
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import yaml

from utils.logger import get_logger
from mcp.memory_mcp import MemoryMCP
from mcp.filesystem_mcp import FilesystemMCP
from mcp.git_mcp import GitMCP
from mcp.web_search_mcp import SmartWebSearchMCP as WebSearchMCP


logger = get_logger(__name__)


class SystemMode(Enum):
    """v9.3.2 System modes"""
    BATTALION = "battalion"  # Full system: Shogun + Karo + Taisho + Ashigaru + All MCP
    COMPANY = "company"      # Slack mode: Karo + Taisho + Ashigaru + Memory MCP
    PLATOON = "platoon"      # HA OS mode: Taisho + Ashigaru + Dynamic MCP


@dataclass
class SystemConfig:
    """v9.3.2 Configuration structure"""
    mode: SystemMode
    claude_api_key: str
    gemini_api_key: str
    tavily_api_key: str

    # v9.3.2: Additional API keys
    groq_api_key: Optional[str] = None
    alibaba_api_key: Optional[str] = None

    # Optional tokens
    slack_token: Optional[str] = None
    notion_token: Optional[str] = None

    # Service endpoints
    ollama_endpoint: str = "http://localhost:11434"
    litellm_endpoint: str = "http://localhost:8000"

    # v9.3.2: Configuration settings
    version: str = "9.3.2"
    intelligent_routing_enabled: bool = True
    prompt_caching_enabled: bool = True
    power_optimization_enabled: bool = True

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key"""
        return getattr(self, key, default)


class SystemOrchestrator:
    """
    v9.3.2 System Orchestrator - Enhanced coordination layer

    Manages the 4-tier hybrid architecture:
    1. Shogun (Strategic Layer) - Claude Sonnet + Opus
    2. Karo (Tactical Layer) - Coordination + Groq/Gemini3
    3. Taisho (Implementation Layer) - Qwen3 + Kagemusha
    4. Ashigaru (Execution Layer) - MCP Servers

    New Features:
    - Intelligent Router for task delegation
    - 3-tier fallback chain management
    - Power-saving optimization
    - Prompt Caching for cost reduction
    """

    VERSION = "9.3.2"

    def __init__(self, config: SystemConfig):
        self.config = config
        self.mcps: Dict[str, Any] = {}
        self.clients: Dict[str, Any] = {}
        self.router = None
        self.mcp_manager = None
        self.initialized = False

        # Statistics
        self.health_status: Dict[str, bool] = {}

        # Performance targets
        self.performance_targets = {
            "simple": 2,
            "medium": 12,
            "complex": 28,
            "strategic": 45
        }

    async def initialize(self) -> None:
        """Initialize all v9.3.2 system components"""
        logger.info(f"🔧 Initializing Bushidan v{self.VERSION} components...")

        try:
            # Initialize MCP servers
            await self._initialize_mcps()

            # Initialize AI clients
            await self._initialize_clients()

            # Initialize Intelligent Router
            await self._initialize_router()

            # Verify external dependencies
            await self._verify_dependencies()

            self.initialized = True
            logger.info(f"✅ System orchestrator v{self.VERSION} initialized successfully")
            self._log_startup_summary()

        except Exception as e:
            logger.error(f"❌ System initialization failed: {e}")
            raise

    async def _initialize_mcps(self) -> None:
        """Initialize Model Context Protocol servers"""

        # Core MCPs (always required)
        self.mcps["memory"] = MemoryMCP()
        self.mcps["filesystem"] = FilesystemMCP()
        self.mcps["git"] = GitMCP()

        # Web search MCP (if Tavily API available)
        if self.config.tavily_api_key:
            self.mcps["web_search"] = WebSearchMCP(self.config.tavily_api_key)

        # Initialize all MCP servers
        for name, mcp in self.mcps.items():
            try:
                await mcp.initialize()
                logger.info(f"✅ {name.title()} MCP initialized")
            except Exception as e:
                logger.warning(f"⚠️ {name.title()} MCP initialization failed: {e}")

        # Initialize MCP Manager
        try:
            from core.mcp_manager import MCPManager
            self.mcp_manager = MCPManager()
            logger.info("✅ MCP Manager initialized")
        except Exception as e:
            logger.warning(f"⚠️ MCP Manager not available: {e}")

    async def _initialize_clients(self) -> None:
        """Initialize all v9.3.2 AI clients"""

        # Claude Client with Prompt Caching (Shogun)
        try:
            from utils.claude_client_cached import ClaudeClientCached
            self.clients["claude_cached"] = ClaudeClientCached(
                api_key=self.config.claude_api_key
            )
            logger.info("✅ Claude Client (Cached) initialized")
        except Exception as e:
            logger.warning(f"⚠️ Claude Cached client failed: {e}")
            # Fallback to standard client
            try:
                from utils.claude_client import ClaudeClient
                self.clients["claude"] = ClaudeClient(
                    api_key=self.config.claude_api_key
                )
                logger.info("✅ Claude Client (Standard) initialized as fallback")
            except Exception as e2:
                logger.warning(f"⚠️ Claude fallback also failed: {e2}")

        # Groq Client (Simple tasks)
        if self.config.groq_api_key:
            try:
                from utils.groq_client import GroqClient
                self.clients["groq"] = GroqClient(
                    api_key=self.config.groq_api_key
                )
                logger.info("✅ Groq Client initialized")
            except Exception as e:
                logger.warning(f"⚠️ Groq client failed: {e}")

        # Gemini 3.0 Flash Client (Final defense)
        try:
            from utils.gemini3_client import Gemini3Client
            self.clients["gemini3"] = Gemini3Client(
                api_key=self.config.gemini_api_key
            )
            logger.info("✅ Gemini 3.0 Flash Client initialized")
        except Exception as e:
            logger.warning(f"⚠️ Gemini 3 client failed: {e}")
            # Fallback to standard Gemini
            try:
                from utils.gemini_client import GeminiClient
                self.clients["gemini"] = GeminiClient(
                    api_key=self.config.gemini_api_key
                )
                logger.info("✅ Gemini Client (Standard) initialized as fallback")
            except Exception as e2:
                logger.warning(f"⚠️ Gemini fallback also failed: {e2}")

        # Local Qwen3 Client (Primary implementation)
        try:
            from utils.qwen3_client import Qwen3Client
            self.clients["qwen3"] = Qwen3Client(
                api_base=self.config.ollama_endpoint
            )
            logger.info("✅ Local Qwen3 Client initialized")
        except Exception as e:
            logger.warning(f"⚠️ Qwen3 client failed: {e}")

        # Alibaba Qwen Client (Kagemusha - Shadow backup)
        if self.config.alibaba_api_key:
            try:
                from utils.alibaba_qwen_client import AlibabaQwenClient
                self.clients["alibaba_qwen"] = AlibabaQwenClient(
                    api_key=self.config.alibaba_api_key
                )
                logger.info("✅ Alibaba Qwen Client (Kagemusha) initialized")
            except Exception as e:
                logger.warning(f"⚠️ Alibaba Qwen client failed: {e}")

        # Opus Client (Premium review)
        try:
            from utils.opus_client import OpusClient
            self.clients["opus"] = OpusClient(
                api_key=self.config.claude_api_key
            )
            logger.info("✅ Opus Client (Premium Review) initialized")
        except Exception as e:
            logger.warning(f"⚠️ Opus client failed: {e}")

    async def _initialize_router(self) -> None:
        """Initialize Intelligent Router for v9.3.2"""

        if not self.config.intelligent_routing_enabled:
            logger.info("ℹ️ Intelligent routing disabled in config")
            return

        try:
            from core.intelligent_router import IntelligentRouter

            router_config = {
                "performance_targets": self.performance_targets,
                "power_optimization": self.config.power_optimization_enabled
            }

            self.router = IntelligentRouter(router_config)
            logger.info("✅ Intelligent Router initialized")

        except Exception as e:
            logger.warning(f"⚠️ Intelligent Router initialization failed: {e}")
            self.router = None

    async def _verify_dependencies(self) -> None:
        """Verify external service availability with health checks"""

        import httpx

        # Check Ollama availability (for Qwen3)
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.config.ollama_endpoint}/api/tags")
                self.health_status["ollama"] = response.status_code == 200
                if self.health_status["ollama"]:
                    logger.info("✅ Ollama service available")
                else:
                    logger.warning("⚠️ Ollama service not responding")
        except Exception as e:
            self.health_status["ollama"] = False
            logger.warning(f"⚠️ Could not verify Ollama: {e}")

        # Check LiteLLM proxy
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.config.litellm_endpoint}/v1/models")
                self.health_status["litellm"] = response.status_code == 200
                if self.health_status["litellm"]:
                    logger.info("✅ LiteLLM proxy available")
                else:
                    logger.warning("⚠️ LiteLLM proxy not responding")
        except Exception as e:
            self.health_status["litellm"] = False
            logger.warning(f"⚠️ Could not verify LiteLLM: {e}")

    def _log_startup_summary(self) -> None:
        """Log startup summary with component status"""

        logger.info("=" * 60)
        logger.info(f"🏯 Bushidan Multi-Agent System v{self.VERSION}")
        logger.info("=" * 60)
        logger.info(f"Mode: {self.config.mode.value}")
        logger.info(f"Intelligent Routing: {'✅' if self.router else '❌'}")
        logger.info(f"Prompt Caching: {'✅' if self.config.prompt_caching_enabled else '❌'}")
        logger.info(f"Power Optimization: {'✅' if self.config.power_optimization_enabled else '❌'}")
        logger.info("-" * 60)
        logger.info("Components Status:")

        # Clients
        for name in ["claude_cached", "groq", "gemini3", "qwen3", "alibaba_qwen", "opus"]:
            status = "✅" if name in self.clients else "❌"
            logger.info(f"  {name}: {status}")

        # MCPs
        logger.info("MCP Servers:")
        for name in self.mcps:
            logger.info(f"  {name}: ✅")

        # Health status
        logger.info("Health Status:")
        for name, healthy in self.health_status.items():
            status = "✅" if healthy else "⚠️"
            logger.info(f"  {name}: {status}")

        logger.info("=" * 60)

    def get_mcp(self, name: str) -> Optional[Any]:
        """Get MCP server by name"""
        return self.mcps.get(name)

    def get_client(self, name: str) -> Optional[Any]:
        """Get AI client by name"""
        return self.clients.get(name)

    def get_router(self):
        """Get Intelligent Router"""
        return self.router

    def get_all_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics from all components"""

        stats = {
            "version": self.VERSION,
            "mode": self.config.mode.value,
            "health_status": self.health_status,
            "clients": {},
            "router": None
        }

        # Collect client statistics
        for name, client in self.clients.items():
            if hasattr(client, 'get_statistics'):
                try:
                    stats["clients"][name] = client.get_statistics()
                except Exception as e:
                    stats["clients"][name] = {"error": str(e)}

        # Router statistics
        if self.router:
            try:
                stats["router"] = self.router.get_statistics()
            except Exception as e:
                stats["router"] = {"error": str(e)}

        return stats

    async def shutdown(self) -> None:
        """Graceful shutdown of all components"""
        logger.info(f"📴 Shutting down system orchestrator v{self.VERSION}...")

        # Shutdown MCP servers
        for name, mcp in self.mcps.items():
            try:
                await mcp.shutdown()
                logger.info(f"✅ {name.title()} MCP shutdown complete")
            except Exception as e:
                logger.warning(f"⚠️ Error shutting down {name} MCP: {e}")

        # Log final statistics
        try:
            stats = self.get_all_statistics()
            logger.info(f"📊 Final Statistics: {stats}")
        except Exception as e:
            logger.warning(f"⚠️ Could not collect final statistics: {e}")

        logger.info(f"✅ System orchestrator v{self.VERSION} shutdown complete")
