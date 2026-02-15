"""
Bushidan Multi-Agent System v10.1 - System Orchestrator (システム統括)

5層ハイブリッドアーキテクチャの強化システム調整。
管理対象: 将軍 → 軍師 → 家老 → 大将(+傭兵) → 足軽（インテリジェントルーティング付き）

v10.1 機能強化:
- 傭兵 (Kimi K2.5) 追加: 128K context, 並列サブタスク実行, マルチモーダル
- 軍師 (Gunshi) 層: Qwen3-Coder-Next 80B API (256K context, SWE-Bench 70.6%)
- 4層フォールバックチェーン: Kimi K2.5 → ローカルQwen3 → 影武者 → Gemini 3 Flash
- Smithery MCP 管理: npm → Smithery 移行
- 新MCP: Sequential Thinking, Playwright, Exa, Graph Memory, Prisma
- インテリジェントルーター統合 (GUNSHI ルート追加)
- llama.cpp CPU最適化（HP ProDesk 600対応）
- 省電力最適化
- BDIフレームワーク統合
- MCP権限マトリクス: 役職別アクセス制御
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


# =============================================================================
# MCP権限マトリクス管理
# =============================================================================

class MCPPermissionLevel(Enum):
    """MCPアクセスレベル"""
    EXCLUSIVE = "exclusive"   # 専属 - この役職のみ直接使用可能
    PRIMARY = "primary"       # 優先 - 他役職も使用可能だが優先される
    SECONDARY = "secondary"   # 補助 - 必要に応じて使用可能
    READONLY = "readonly"     # 読取専用 - 監視・分析目的
    DELEGATED = "delegated"   # 委譲 - 上位役職の指示に基づく
    FORBIDDEN = "forbidden"   # 禁止 - 使用不可


class MCPPermissionManager:
    """
    MCP権限マトリクス管理

    役職ごとのMCPアクセス権限を管理し、
    不正なアクセスを防止する。
    """

    # 役職優先度順序（上位ほど優先）
    ROLE_PRIORITY = ["shogun", "gunshi", "karo", "taisho", "kengyo", "ashigaru"]

    def __init__(self, config_path: Optional[str] = None):
        self.permissions: Dict[str, Dict[str, Dict]] = {}
        self.mcp_registry: Dict[str, Dict] = {}
        self.config_path = config_path or "config/mcp_permissions.yaml"
        self._loaded = False

    def load_permissions(self) -> bool:
        """権限設定をYAMLファイルから読み込む"""
        try:
            config_file = Path(self.config_path)
            if not config_file.exists():
                logger.warning(f"⚠️ MCP権限設定ファイルが見つかりません: {self.config_path}")
                self._load_default_permissions()
                return False

            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)

            self.mcp_registry = config.get('mcp_registry', {})
            role_permissions = config.get('role_permissions', {})

            # 権限を構造化して保存
            for role, role_config in role_permissions.items():
                self.permissions[role] = role_config.get('permissions', {})

            self._loaded = True
            logger.info(f"✅ MCP権限設定読み込み完了: {len(self.permissions)} 役職")
            return True

        except Exception as e:
            logger.error(f"❌ MCP権限設定読み込み失敗: {e}")
            self._load_default_permissions()
            return False

    def _load_default_permissions(self) -> None:
        """デフォルト権限を設定"""
        self.permissions = {
            "shogun": {
                "graph_memory": {"level": "primary"},
                "notion": {"level": "primary"},
                "slack": {"level": "primary"},
                "sequential_thinking": {"level": "secondary"},
                "filesystem": {"level": "readonly"},
                "git": {"level": "readonly"},
                "playwright": {"level": "forbidden"},
                "prisma": {"level": "forbidden"},
            },
            "gunshi": {
                "sequential_thinking": {"level": "exclusive", "priority": 1},
                "filesystem": {"level": "primary"},
                "git": {"level": "primary"},
                "graph_memory": {"level": "primary"},
                "playwright": {"level": "forbidden"},
                "slack": {"level": "forbidden"},
            },
            "karo": {
                "sequential_thinking": {"level": "primary", "priority": 2},
                "slack": {"level": "primary"},
                "filesystem": {"level": "secondary"},
                "playwright": {"level": "forbidden"},
                "prisma": {"level": "forbidden"},
            },
            "taisho": {
                "filesystem": {"level": "exclusive", "priority": 1},
                "git": {"level": "exclusive", "priority": 1},
                "prisma": {"level": "exclusive"},
                "sequential_thinking": {"level": "secondary", "priority": 3},
                "playwright": {"level": "forbidden"},
                "slack": {"level": "forbidden"},
            },
            "kengyo": {
                "playwright": {"level": "exclusive", "priority": 1},
                "filesystem": {"level": "primary"},
                "sequential_thinking": {"level": "forbidden"},
                "prisma": {"level": "forbidden"},
            },
            "ashigaru": {
                "filesystem": {"level": "delegated"},
                "git": {"level": "delegated"},
                "playwright": {"level": "delegated"},
                "prisma": {"level": "delegated"},
            },
        }
        self._loaded = True
        logger.info("ℹ️ デフォルトMCP権限設定を使用")

    def check_permission(self, role: str, mcp_name: str) -> MCPPermissionLevel:
        """
        役職がMCPにアクセス可能かチェック

        Args:
            role: 役職名 (shogun, gunshi, karo, taisho, kengyo, ashigaru)
            mcp_name: MCP名

        Returns:
            アクセスレベル
        """
        if not self._loaded:
            self.load_permissions()

        role_perms = self.permissions.get(role, {})
        mcp_perm = role_perms.get(mcp_name, {})

        level_str = mcp_perm.get('level', 'forbidden')

        try:
            return MCPPermissionLevel(level_str)
        except ValueError:
            return MCPPermissionLevel.FORBIDDEN

    def can_access(self, role: str, mcp_name: str) -> bool:
        """
        役職がMCPにアクセス可能か判定

        Returns:
            True: アクセス可能 (exclusive, primary, secondary, readonly, delegated)
            False: アクセス不可 (forbidden)
        """
        level = self.check_permission(role, mcp_name)
        return level != MCPPermissionLevel.FORBIDDEN

    def can_write(self, role: str, mcp_name: str) -> bool:
        """
        役職がMCPに書き込み可能か判定

        Returns:
            True: 書き込み可能 (exclusive, primary, secondary, delegated)
            False: 読み取りのみまたは禁止 (readonly, forbidden)
        """
        level = self.check_permission(role, mcp_name)
        return level in [
            MCPPermissionLevel.EXCLUSIVE,
            MCPPermissionLevel.PRIMARY,
            MCPPermissionLevel.SECONDARY,
            MCPPermissionLevel.DELEGATED
        ]

    def get_exclusive_owner(self, mcp_name: str) -> Optional[str]:
        """
        MCPの専属所有者を取得

        Returns:
            専属所有役職名、またはNone
        """
        for role in self.ROLE_PRIORITY:
            level = self.check_permission(role, mcp_name)
            if level == MCPPermissionLevel.EXCLUSIVE:
                return role
        return None

    def resolve_conflict(self, mcp_name: str, requesting_roles: list) -> str:
        """
        複数役職が同時にMCPを要求した場合の競合解決

        Args:
            mcp_name: MCP名
            requesting_roles: 要求している役職のリスト

        Returns:
            優先される役職名
        """
        # 1. exclusive が最優先
        for role in requesting_roles:
            if self.check_permission(role, mcp_name) == MCPPermissionLevel.EXCLUSIVE:
                return role

        # 2. priority 属性による解決
        role_priorities = []
        for role in requesting_roles:
            role_perms = self.permissions.get(role, {})
            mcp_perm = role_perms.get(mcp_name, {})
            priority = mcp_perm.get('priority', 99)
            role_priorities.append((role, priority))

        role_priorities.sort(key=lambda x: x[1])

        # 同一 priority の場合は役職階層で解決
        best_priority = role_priorities[0][1]
        candidates = [r for r, p in role_priorities if p == best_priority]

        if len(candidates) == 1:
            return candidates[0]

        # 階層順で解決
        for role in self.ROLE_PRIORITY:
            if role in candidates:
                return role

        return requesting_roles[0]

    def get_role_permissions(self, role: str) -> Dict[str, str]:
        """
        役職の全MCP権限を取得

        Returns:
            {mcp_name: level} の辞書
        """
        if not self._loaded:
            self.load_permissions()

        role_perms = self.permissions.get(role, {})
        return {mcp: perm.get('level', 'forbidden') for mcp, perm in role_perms.items()}

    def log_access_attempt(self, role: str, mcp_name: str, granted: bool) -> None:
        """アクセス試行をログ記録"""
        status = "✅ 許可" if granted else "❌ 拒否"
        level = self.check_permission(role, mcp_name)
        logger.debug(f"🔐 MCP アクセス {status}: {role} → {mcp_name} ({level.value})")


class SystemMode(Enum):
    """v10 System modes"""
    BATTALION = "battalion"  # Full system: Shogun + Gunshi + Karo + Taisho + Ashigaru + All MCP
    COMPANY = "company"      # Slack mode: Karo + Taisho + Ashigaru + Memory MCP
    PLATOON = "platoon"      # HA OS mode: Taisho + Ashigaru + Dynamic MCP


@dataclass
class SystemConfig:
    """v10 Configuration structure"""
    mode: SystemMode
    claude_api_key: str
    gemini_api_key: str
    tavily_api_key: str

    # v9.4: Additional API keys
    groq_api_key: Optional[str] = None
    alibaba_api_key: Optional[str] = None

    # v10: Qwen3-Coder-Next (軍師)
    qwen3_coder_next_api_key: Optional[str] = None
    qwen3_coder_next_provider: str = "openrouter"  # openrouter, dashscope

    # v10.1: Kimi K2.5 (傭兵)
    kimi_api_key: Optional[str] = None
    kimi_provider: str = "moonshot"  # moonshot, openrouter

    # Optional tokens
    slack_token: Optional[str] = None
    notion_token: Optional[str] = None

    # Service endpoints
    ollama_endpoint: str = "http://localhost:11434"  # Legacy (unused in v10)
    litellm_endpoint: str = "http://localhost:8000"

    # v9.4: llama.cpp configuration (HP ProDesk 600 CPU optimized)
    llamacpp_endpoint: str = "http://127.0.0.1:8080"
    llamacpp_model_path: str = "models/qwen3/Qwen3-Coder-30B-Q4_K_M.gguf"
    llamacpp_threads: int = 6  # HP ProDesk 600: i5-8500 (6C/6T)
    llamacpp_context_size: int = 4096  # Optimized for CPU speed
    llamacpp_batch_size: int = 512  # CPU optimal
    llamacpp_mlock: bool = True  # Lock memory to prevent swapping

    # v10: Configuration settings
    version: str = "10.0"
    intelligent_routing_enabled: bool = True
    prompt_caching_enabled: bool = True
    power_optimization_enabled: bool = True
    use_llamacpp: bool = True  # Use llama.cpp instead of Ollama

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key"""
        return getattr(self, key, default)


class SystemOrchestrator:
    """
    v10 システムオーケストレーター - 強化調整層

    5層ハイブリッドアーキテクチャを管理:
    1. 将軍 (Shogun) - 戦略層: Claude Sonnet + Opus
    2. 軍師 (Gunshi) - 作戦立案層: Qwen3-Coder-Next 80B API
    3. 家老 (Karo) - 戦術層: 調整 + Groq/Gemini3
    4. 大将 (Taisho) - 実装層: Qwen3 + 影武者
    5. 足軽 (Ashigaru) - 実行層: MCPサーバー

    補助役:
    - 検校 (Kengyo) - ビジュアル検証: Kimi K2.5 Vision + Playwright MCP

    新機能:
    - 軍師 (Gunshi) 層: 複雑タスクの作戦立案・コード監査
    - タスク委譲のインテリジェントルーター (GUNSHIルート)
    - 4層フォールバックチェーン管理（Kimi→Local→Kagemusha→Gemini）
    - 省電力最適化
    - コスト削減のプロンプトキャッシング
    - BDIフレームワーク統合
    """

    VERSION = "10.1"

    def __init__(self, config: SystemConfig):
        self.config = config
        self.mcps: Dict[str, Any] = {}
        self.clients: Dict[str, Any] = {}
        self.router = None
        self.mcp_manager = None
        self.initialized = False

        # MCP権限マネージャー
        self.permission_manager = MCPPermissionManager()

        # 5層階層コンポーネント
        self._shogun = None  # 将軍: 戦略層
        self._gunshi = None  # 軍師: 作戦立案層 (v10)
        self._karo = None    # 家老: 戦術層
        self._taisho = None  # 大将: 実装層
        self._kengyo = None  # 検校: ビジュアル検証 (v10.1)

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
        """全v10システムコンポーネントを初期化"""
        logger.info(f"🔧 武士団 v{self.VERSION} コンポーネント初期化開始...")

        try:
            # MCP権限マトリクス読み込み
            self.permission_manager.load_permissions()

            # MCPサーバー初期化
            await self._initialize_mcps()

            # AIクライアント初期化
            await self._initialize_clients()

            # インテリジェントルーター初期化
            await self._initialize_router()

            # 外部依存関係の検証
            await self._verify_dependencies()

            # 5層階層コンポーネント初期化
            await self._initialize_tiers()

            self.initialized = True
            logger.info(f"✅ システムオーケストレーター v{self.VERSION} 初期化完了")
            self._log_startup_summary()

        except Exception as e:
            logger.error(f"❌ システム初期化失敗: {e}")
            raise

    async def _initialize_tiers(self) -> None:
        """5層階層コンポーネントの初期化"""
        logger.info("🏯 5層階層コンポーネント初期化...")

        # 将軍（戦略層）初期化
        try:
            from core.shogun import Shogun
            self._shogun = Shogun(self)
            await self._shogun.initialize()
            logger.info("🎌 将軍（戦略層）初期化完了")
        except Exception as e:
            logger.error(f"❌ 将軍初期化失敗: {e}")
            raise

        # v10: 軍師（作戦立案層）初期化
        try:
            from core.gunshi import Gunshi
            self._gunshi = Gunshi(self)
            await self._gunshi.initialize()
            logger.info("🧠 軍師（作戦立案層）初期化完了")
        except Exception as e:
            logger.warning(f"⚠️ 軍師初期化失敗 (複雑タスクは家老に直接委譲): {e}")

        # 家老は将軍の初期化時に作成される
        if self._shogun and hasattr(self._shogun, 'karo'):
            self._karo = self._shogun.karo
            logger.info("👔 家老（戦術層）参照取得完了")

        # 大将は家老の初期化時に作成される
        if self._karo and hasattr(self._karo, 'taisho'):
            self._taisho = self._karo.taisho
            logger.info("⚔️ 大将（実装層）参照取得完了")

        # v10.1: 検校（ビジュアル・デバッガー）初期化
        try:
            from core.kengyo import Kengyo
            kimi_client = self.clients.get("kimi_k2")
            self._kengyo = Kengyo(
                kimi_client=kimi_client,
                smithery_mcp=getattr(self, 'smithery_mcp', None),
            )
            await self._kengyo.initialize()
            logger.info("👁️ 検校（ビジュアル・デバッガー）初期化完了")
        except Exception as e:
            logger.warning(f"⚠️ 検校初期化失敗 (ビジュアル検証スキップ): {e}")

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

        # Initialize MCP Manager (legacy)
        try:
            from core.mcp_manager import MCPManager
            self.mcp_manager = MCPManager()
            logger.info("✅ MCP Manager initialized")
        except Exception as e:
            logger.warning(f"⚠️ MCP Manager not available: {e}")

        # v10.1: Smithery MCP Manager
        try:
            from mcp.smithery_manager import SmitheryMCPManager
            self.smithery_mcp = SmitheryMCPManager()
            smithery_status = await self.smithery_mcp.initialize()
            available = sum(1 for v in smithery_status.values() if v)
            total = len(smithery_status)
            logger.info(f"✅ Smithery MCP Manager: {available}/{total} servers ready")
        except Exception as e:
            self.smithery_mcp = None
            logger.warning(f"⚠️ Smithery MCP Manager not available: {e}")

    async def _initialize_clients(self) -> None:
        """Initialize all v10 AI clients"""

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

        # Kagemusha Client (影武者 - qwen3-coder-plus via OpenRouter)
        if self.config.alibaba_api_key:
            try:
                from utils.alibaba_qwen_client import AlibabaQwenClient
                self.clients["alibaba_qwen"] = AlibabaQwenClient(
                    api_key=self.config.alibaba_api_key,
                    provider="openrouter",
                )
                logger.info("✅ Kagemusha Client (影武者) initialized via OpenRouter")
            except Exception as e:
                logger.warning(f"⚠️ Alibaba Qwen client failed: {e}")

        # v10.1: Kimi K2.5 Client (傭兵 - Yohei)
        if self.config.kimi_api_key:
            try:
                from utils.kimi_k2_client import KimiK2Client, KimiConfig
                kimi_config = KimiConfig(
                    api_key=self.config.kimi_api_key,
                    provider=self.config.kimi_provider
                )
                client = KimiK2Client(config=kimi_config)
                await client.initialize()
                self.clients["kimi_k2"] = client
                logger.info("✅ Kimi K2.5 Client (傭兵) initialized - 128K context")
            except Exception as e:
                logger.warning(f"⚠️ Kimi K2.5 client failed: {e}")

        # v10: Qwen3-Coder-Next Client (軍師 - Gunshi)
        if self.config.qwen3_coder_next_api_key:
            try:
                from utils.qwen3_coder_next_client import Qwen3CoderNextClient
                self.clients["qwen3_coder_next"] = Qwen3CoderNextClient(
                    api_key=self.config.qwen3_coder_next_api_key,
                    provider=self.config.qwen3_coder_next_provider
                )
                logger.info("✅ Qwen3-Coder-Next Client (軍師) initialized")
            except Exception as e:
                logger.warning(f"⚠️ Qwen3-Coder-Next client failed: {e}")

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
        """Initialize Intelligent Router for v10"""

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

        # 5層階層コンポーネント
        logger.info("【5層階層コンポーネント】")
        logger.info(f"  🎌 将軍（戦略層）: {'✅' if self._shogun else '❌'}")
        logger.info(f"  🧠 軍師（作戦立案層）: {'✅' if self._gunshi else '❌'}")
        logger.info(f"  👔 家老（戦術層）: {'✅' if self._karo else '❌'}")
        logger.info(f"  ⚔️ 大将（実装層）: {'✅' if self._taisho else '❌'}")
        logger.info(f"  👁️ 検校（ビジュアル検証）: {'✅' if self._kengyo and self._kengyo.is_available() else '❌'}")

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
            "kimi_k2": "Kimi K2.5（傭兵, 128K）",
            "qwen3_coder_next": "Qwen3-Coder-Next（軍師）",
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

    # ==================== 5層階層コンポーネントアクセサ ====================

    @property
    def shogun(self):
        """将軍（戦略層）取得"""
        return self._shogun

    @property
    def gunshi(self):
        """軍師（作戦立案層）取得"""
        return self._gunshi

    @property
    def karo(self):
        """家老（戦術層）取得"""
        return self._karo

    @property
    def taisho(self):
        """大将（実装層）取得"""
        return self._taisho

    @property
    def kengyo(self):
        """検校（ビジュアル検証）取得"""
        return self._kengyo

    # ==================== MCP権限制御 ====================

    def check_mcp_permission(self, role: str, mcp_name: str) -> MCPPermissionLevel:
        """
        役職のMCPアクセス権限をチェック

        Args:
            role: 役職名
            mcp_name: MCP名

        Returns:
            MCPPermissionLevel
        """
        return self.permission_manager.check_permission(role, mcp_name)

    def request_mcp_access(self, role: str, mcp_name: str, write: bool = False) -> bool:
        """
        MCPアクセスをリクエスト

        Args:
            role: 要求元役職
            mcp_name: MCP名
            write: 書き込み権限が必要か

        Returns:
            True: 許可, False: 拒否
        """
        if write:
            granted = self.permission_manager.can_write(role, mcp_name)
        else:
            granted = self.permission_manager.can_access(role, mcp_name)

        self.permission_manager.log_access_attempt(role, mcp_name, granted)
        return granted

    def get_mcp_for_role(self, role: str, mcp_name: str) -> Optional[Any]:
        """
        役職用にMCPを取得（権限チェック付き）

        Args:
            role: 要求元役職
            mcp_name: MCP名

        Returns:
            MCPインスタンス、または権限がない場合None
        """
        if not self.request_mcp_access(role, mcp_name):
            logger.warning(f"⚠️ {role} は {mcp_name} へのアクセス権限がありません")
            return None

        return self.mcps.get(mcp_name)

    def get_exclusive_mcp_owner(self, mcp_name: str) -> Optional[str]:
        """MCPの専属所有者を取得"""
        return self.permission_manager.get_exclusive_owner(mcp_name)

    def get_role_mcp_permissions(self, role: str) -> Dict[str, str]:
        """役職の全MCP権限を取得"""
        return self.permission_manager.get_role_permissions(role)

    def get_tier_statistics(self) -> Dict[str, Any]:
        """全階層統計を取得"""
        stats = {}

        if self._shogun and hasattr(self._shogun, 'get_statistics'):
            stats["shogun"] = self._shogun.get_statistics()

        if self._gunshi and hasattr(self._gunshi, 'get_statistics'):
            stats["gunshi"] = self._gunshi.get_statistics()

        if self._karo and hasattr(self._karo, 'get_statistics'):
            stats["karo"] = self._karo.get_statistics()

        if self._taisho and hasattr(self._taisho, 'get_statistics'):
            stats["taisho"] = self._taisho.get_statistics()

        if self._kengyo and hasattr(self._kengyo, 'get_statistics'):
            stats["kengyo"] = self._kengyo.get_statistics()

        return stats

    def get_bdi_states(self) -> Dict[str, Any]:
        """全階層のBDI状態を取得"""
        bdi_states = {}

        if self._shogun and hasattr(self._shogun, 'get_bdi_state'):
            bdi_states["shogun"] = self._shogun.get_bdi_state()

        if self._gunshi and hasattr(self._gunshi, 'get_bdi_state'):
            bdi_states["gunshi"] = self._gunshi.get_bdi_state()

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
