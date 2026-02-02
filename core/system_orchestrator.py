"""
Bushidan Multi-Agent System v9.4 - System Orchestrator (システム統括)

4層ハイブリッドアーキテクチャの強化システム調整。
管理対象: 将軍 → 家老 → 大将 → 足軽（インテリジェントルーティング付き）

v9.4 機能強化:
- インテリジェントルーター統合
- 新規クライアント初期化（ClaudeClientCached, Groq, Gemini3, Qwen3 llama.cpp, AlibabaQwen）
- 3層フォールバックチェーン管理
- llama.cpp CPU最適化（HP ProDesk 600対応）
- 省電力最適化
- 強化ヘルスチェック
- BDIフレームワーク統合
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
    """v9.4 Configuration structure"""
    mode: SystemMode
    claude_api_key: str
    gemini_api_key: str
    tavily_api_key: str

    # v9.4: Additional API keys
    groq_api_key: Optional[str] = None
    alibaba_api_key: Optional[str] = None

    # Optional tokens
    slack_token: Optional[str] = None
    notion_token: Optional[str] = None

    # Service endpoints
    ollama_endpoint: str = "http://localhost:11434"  # Legacy (unused in v9.4)
    litellm_endpoint: str = "http://localhost:8000"

    # v9.4: llama.cpp configuration (HP ProDesk 600 CPU optimized)
    llamacpp_endpoint: str = "http://127.0.0.1:8080"
    llamacpp_model_path: str = "models/Qwen3-Coder-30B-A3B-instruct-q4_k_m.gguf"
    llamacpp_threads: int = 8  # HP ProDesk 600: i5/i7 with 6-8 cores
    llamacpp_context_size: int = 4096  # Optimized for CPU speed
    llamacpp_batch_size: int = 512  # CPU optimal
    llamacpp_mlock: bool = True  # Lock memory to prevent swapping

    # v9.4: Configuration settings
    version: str = "9.4"
    intelligent_routing_enabled: bool = True
    prompt_caching_enabled: bool = True
    power_optimization_enabled: bool = True
    use_llamacpp: bool = True  # Use llama.cpp instead of Ollama

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key"""
        return getattr(self, key, default)


class SystemOrchestrator:
    """
    v9.3.2 システムオーケストレーター - 強化調整層

    4層ハイブリッドアーキテクチャを管理:
    1. 将軍 (Shogun) - 戦略層: Claude Sonnet + Opus
    2. 家老 (Karo) - 戦術層: 調整 + Groq/Gemini3
    3. 大将 (Taisho) - 実装層: Qwen3 + 影武者
    4. 足軽 (Ashigaru) - 実行層: MCPサーバー

    新機能:
    - タスク委譲のインテリジェントルーター
    - 3層フォールバックチェーン管理
    - 省電力最適化
    - コスト削減のプロンプトキャッシング
    - BDIフレームワーク統合
    """

    VERSION = "9.4"

    def __init__(self, config: SystemConfig):
        self.config = config
        self.mcps: Dict[str, Any] = {}
        self.clients: Dict[str, Any] = {}
        self.router = None
        self.mcp_manager = None
        self.initialized = False

        # 4層階層コンポーネント
        self._shogun = None  # 将軍: 戦略層
        self._karo = None    # 家老: 戦術層
        self._taisho = None  # 大将: 実装層

        # 統計
        self.health_status: Dict[str, bool] = {}

        # パフォーマンス目標（秒）
        self.performance_targets = {
            "simple": 2,
            "medium": 12,
            "complex": 28,
            "strategic": 45
        }

    async def initialize(self) -> None:
        """全v9.3.2システムコンポーネントを初期化"""
        logger.info(f"🔧 武士団 v{self.VERSION} コンポーネント初期化開始...")

        try:
            # MCPサーバー初期化
            await self._initialize_mcps()

            # AIクライアント初期化
            await self._initialize_clients()

            # インテリジェントルーター初期化
            await self._initialize_router()

            # 外部依存関係の検証
            await self._verify_dependencies()

            # 4層階層コンポーネント初期化
            await self._initialize_tiers()

            self.initialized = True
            logger.info(f"✅ システムオーケストレーター v{self.VERSION} 初期化完了")
            self._log_startup_summary()

        except Exception as e:
            logger.error(f"❌ システム初期化失敗: {e}")
            raise

    async def _initialize_tiers(self) -> None:
        """4層階層コンポーネントの初期化"""
        logger.info("🏯 4層階層コンポーネント初期化...")

        # 将軍（戦略層）初期化
        try:
            from core.shogun import Shogun
            self._shogun = Shogun(self)
            await self._shogun.initialize()
            logger.info("🎌 将軍（戦略層）初期化完了")
        except Exception as e:
            logger.error(f"❌ 将軍初期化失敗: {e}")
            raise

        # 家老は将軍の初期化時に作成される
        if self._shogun and hasattr(self._shogun, 'karo'):
            self._karo = self._shogun.karo
            logger.info("👔 家老（戦術層）参照取得完了")

        # 大将は家老の初期化時に作成される
        if self._karo and hasattr(self._karo, 'taisho'):
            self._taisho = self._karo.taisho
            logger.info("⚔️ 大将（実装層）参照取得完了")

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
        # v9.4: Use llama.cpp instead of Ollama for CPU-optimized inference
        try:
            if self.config.use_llamacpp:
                from utils.qwen3_llamacpp_client import Qwen3LlamaCppClient, LlamaCppConfig
                llamacpp_config = LlamaCppConfig(
                    model_path=self.config.llamacpp_model_path,
                    host=self.config.llamacpp_endpoint.split("://")[1].split(":")[0],
                    port=int(self.config.llamacpp_endpoint.split(":")[-1]),
                    threads=self.config.llamacpp_threads,
                    context_size=self.config.llamacpp_context_size,
                    batch_size=self.config.llamacpp_batch_size,
                    mlock=self.config.llamacpp_mlock
                )
                self.clients["qwen3"] = Qwen3LlamaCppClient(config=llamacpp_config)
                logger.info("✅ Qwen3 llama.cpp Client initialized (CPU optimized)")
            else:
                # Fallback to Ollama (legacy)
                from utils.qwen3_client import Qwen3Client
                self.clients["qwen3"] = Qwen3Client(
                    api_base=self.config.ollama_endpoint
                )
                logger.info("✅ Local Qwen3 Client initialized (Ollama)")
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

        # v9.4: Check llama.cpp server availability (for Qwen3)
        if self.config.use_llamacpp:
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    response = await client.get(f"{self.config.llamacpp_endpoint}/health")
                    self.health_status["llamacpp"] = response.status_code == 200
                    if self.health_status["llamacpp"]:
                        logger.info("✅ llama.cpp server available")
                    else:
                        logger.warning("⚠️ llama.cpp server not responding")
            except Exception as e:
                self.health_status["llamacpp"] = False
                logger.warning(f"⚠️ Could not verify llama.cpp: {e}")
        else:
            # Legacy: Check Ollama availability
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
        """起動サマリーをコンポーネント状態と共にログ出力"""

        logger.info("=" * 60)
        logger.info(f"🏯 武士団マルチエージェントシステム v{self.VERSION}")
        logger.info("=" * 60)
        logger.info(f"モード: {self.config.mode.value}")
        logger.info(f"インテリジェントルーティング: {'✅' if self.router else '❌'}")
        logger.info(f"プロンプトキャッシング: {'✅' if self.config.prompt_caching_enabled else '❌'}")
        logger.info(f"省電力最適化: {'✅' if self.config.power_optimization_enabled else '❌'}")
        logger.info("-" * 60)

        # 4層階層コンポーネント
        logger.info("【4層階層コンポーネント】")
        logger.info(f"  🎌 将軍（戦略層）: {'✅' if self._shogun else '❌'}")
        logger.info(f"  👔 家老（戦術層）: {'✅' if self._karo else '❌'}")
        logger.info(f"  ⚔️ 大将（実装層）: {'✅' if self._taisho else '❌'}")

        # BDI状態
        if self._shogun and hasattr(self._shogun, 'bdi_enabled'):
            logger.info(f"  🧠 BDIフレームワーク: {'✅' if self._shogun.bdi_enabled else '❌'}")

        logger.info("-" * 60)
        logger.info("【AIクライアント】")
        client_names = {
            "claude_cached": "Claude（キャッシュ）",
            "groq": "Groq（即応）",
            "gemini3": "Gemini 3.0 Flash",
            "qwen3": f"Qwen3（{'llama.cpp CPU' if self.config.use_llamacpp else 'Ollama'}）",
            "alibaba_qwen": "Alibaba Qwen（影武者）",
            "opus": "Opus（プレミアム）"
        }
        for key, name in client_names.items():
            status = "✅" if key in self.clients else "❌"
            logger.info(f"  {name}: {status}")

        # v9.4: llama.cpp configuration info
        if self.config.use_llamacpp:
            logger.info("【llama.cpp設定】")
            logger.info(f"  エンドポイント: {self.config.llamacpp_endpoint}")
            logger.info(f"  スレッド数: {self.config.llamacpp_threads}")
            logger.info(f"  コンテキスト: {self.config.llamacpp_context_size}")
            logger.info(f"  メモリロック: {'✅' if self.config.llamacpp_mlock else '❌'}")

        # MCPサーバー
        logger.info("【MCPサーバー】")
        mcp_names = {
            "memory": "メモリMCP",
            "filesystem": "ファイルシステムMCP",
            "git": "Git MCP",
            "web_search": "Web検索MCP"
        }
        for key in self.mcps:
            name = mcp_names.get(key, key)
            logger.info(f"  {name}: ✅")

        # ヘルス状態
        logger.info("【外部サービス】")
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
        """インテリジェントルーター取得"""
        return self.router

    # ==================== 4層階層コンポーネントアクセサ ====================

    @property
    def shogun(self):
        """将軍（戦略層）取得"""
        return self._shogun

    @property
    def karo(self):
        """家老（戦術層）取得"""
        return self._karo

    @property
    def taisho(self):
        """大将（実装層）取得"""
        return self._taisho

    def get_tier_statistics(self) -> Dict[str, Any]:
        """全階層統計を取得"""
        stats = {}

        if self._shogun and hasattr(self._shogun, 'get_statistics'):
            stats["shogun"] = self._shogun.get_statistics()

        if self._karo and hasattr(self._karo, 'get_statistics'):
            stats["karo"] = self._karo.get_statistics()

        if self._taisho and hasattr(self._taisho, 'get_statistics'):
            stats["taisho"] = self._taisho.get_statistics()

        return stats

    def get_bdi_states(self) -> Dict[str, Any]:
        """全階層のBDI状態を取得"""
        bdi_states = {}

        if self._shogun and hasattr(self._shogun, 'get_bdi_state'):
            bdi_states["shogun"] = self._shogun.get_bdi_state()

        if self._karo and hasattr(self._karo, 'get_bdi_state'):
            bdi_states["karo"] = self._karo.get_bdi_state()

        if self._taisho and hasattr(self._taisho, 'get_bdi_state'):
            bdi_states["taisho"] = self._taisho.get_bdi_state()

        return bdi_states

    # ==================== タスク処理エントリポイント ====================

    async def process_task(self, task_content: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        タスク処理のメインエントリポイント

        Args:
            task_content: タスク内容
            context: オプションコンテキスト

        Returns:
            処理結果
        """
        if not self.initialized:
            raise RuntimeError("システムが初期化されていません")

        if not self._shogun:
            raise RuntimeError("将軍が初期化されていません")

        # 将軍に委譲
        from core.shogun import Task, TaskComplexity
        task = Task(
            content=task_content,
            complexity=TaskComplexity.MEDIUM,  # 将軍が再評価
            context=context
        )

        return await self._shogun.process_task(task)

    def get_all_statistics(self) -> Dict[str, Any]:
        """全コンポーネントから包括的統計を取得"""

        stats = {
            "version": self.VERSION,
            "mode": self.config.mode.value,
            "health_status": self.health_status,
            "clients": {},
            "router": None,
            "tiers": {},
            "bdi_states": {}
        }

        # クライアント統計収集
        for name, client in self.clients.items():
            if hasattr(client, 'get_statistics'):
                try:
                    stats["clients"][name] = client.get_statistics()
                except Exception as e:
                    stats["clients"][name] = {"error": str(e)}

        # ルーター統計
        if self.router:
            try:
                stats["router"] = self.router.get_statistics()
            except Exception as e:
                stats["router"] = {"error": str(e)}

        # 階層統計
        try:
            stats["tiers"] = self.get_tier_statistics()
        except Exception as e:
            stats["tiers"] = {"error": str(e)}

        # BDI状態
        try:
            stats["bdi_states"] = self.get_bdi_states()
        except Exception as e:
            stats["bdi_states"] = {"error": str(e)}

        return stats

    async def shutdown(self) -> None:
        """全コンポーネントのグレースフルシャットダウン"""
        logger.info(f"📴 システムオーケストレーター v{self.VERSION} シャットダウン開始...")

        # MCPサーバーのシャットダウン
        for name, mcp in self.mcps.items():
            try:
                await mcp.shutdown()
                logger.info(f"✅ {name} MCP シャットダウン完了")
            except Exception as e:
                logger.warning(f"⚠️ {name} MCP シャットダウンエラー: {e}")

        # 最終統計ログ
        try:
            stats = self.get_all_statistics()
            logger.info(f"📊 最終統計: {stats}")
        except Exception as e:
            logger.warning(f"⚠️ 最終統計収集失敗: {e}")

        logger.info(f"✅ システムオーケストレーター v{self.VERSION} シャットダウン完了")
