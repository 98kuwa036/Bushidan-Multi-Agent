"""
Bushidan Multi-Agent System v11.5 - System Orchestrator (システム統括)

9層ハイブリッドアーキテクチャの強化システム調整。
管理対象: 大元帥 → 将軍 → 軍師 → 参謀A/B → 家老A/B → 検校 → 隠密

v11.5 アーキテクチャ:
- 大元帥 (Daigensui): Claude Opus 4.6 - 最高難度・戦略設計
- 将軍 (Shogun): Claude Sonnet 4.6 - 高難度コーディング
- 軍師 (Gunshi): o3-mini (high) - 推論・設計・PDCA
- 参謀-A (Sanbo-A): Mistral Large 3 - 汎用コーディング・EU準拠
- 参謀-B (Sanbo-B): Grok 4.1 Fast - 実装・バグ修正・超高速
- 家老-A (Karo-A): Gemini 3 Flash - 軽量タスク
- 家老-B (Karo-B): Llama 3.3 70B (Groq) - アルゴリズム特化
- 検校 (Kengyo): Gemini 3 Flash Vision - マルチモーダル
- 隠密 (Onmitsu): Nemotron-3-Nano (Local) - 機密・超長文

2台構成: ローカルLLMサーバー (LLM専用, 192.168.11.239) + EliteDesk (メインオーケストレーション)

機能:
- Smithery MCP 管理
- MCP: Sequential Thinking, Playwright, Exa, Graph Memory, Prisma
- インテリジェントルーター統合
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
from mcp_servers.memory_mcp import MemoryMCP
from mcp_servers.filesystem_mcp import FilesystemMCP
from mcp_servers.git_mcp import GitMCP
from mcp_servers.web_search_mcp import SmartWebSearchMCP as WebSearchMCP


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

    # 役職優先度順序 v12 (上位ほど優先)
    # 受付→外事→検校→将軍→軍師→参謀→右筆→斥候→隠密→大元帥
    ROLE_PRIORITY = ["uketuke", "gaiji", "kengyo", "shogun", "gunshi", "sanbo", "yuhitsu", "seppou", "onmitsu", "daigensui"]

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
            "daigensui": {
                "graph_memory": {"level": "exclusive", "priority": 1},
                "notion": {"level": "exclusive", "priority": 1},
                "sequential_thinking": {"level": "primary"},
                "filesystem": {"level": "readonly"},
                "git": {"level": "readonly"},
                "playwright": {"level": "forbidden"},
                "prisma": {"level": "forbidden"},
            },
            "shogun": {
                "graph_memory": {"level": "primary"},
                "notion": {"level": "primary"},
                "sequential_thinking": {"level": "secondary"},
                "filesystem": {"level": "primary"},
                "git": {"level": "primary"},
                "playwright": {"level": "forbidden"},
                "prisma": {"level": "secondary"},
            },
            "gunshi": {
                "sequential_thinking": {"level": "exclusive", "priority": 1},
                "filesystem": {"level": "primary"},
                "git": {"level": "primary"},
                "graph_memory": {"level": "primary"},
                "playwright": {"level": "forbidden"},
            },
            "sanbo_a": {
                "filesystem": {"level": "primary"},
                "git": {"level": "primary"},
                "prisma": {"level": "primary"},
                "sequential_thinking": {"level": "secondary", "priority": 2},
                "playwright": {"level": "forbidden"},
            },
            "sanbo_b": {
                "filesystem": {"level": "primary"},
                "git": {"level": "primary"},
                "prisma": {"level": "secondary"},
                "sequential_thinking": {"level": "secondary", "priority": 3},
                "playwright": {"level": "forbidden"},
            },
            "karo_a": {
                "sequential_thinking": {"level": "secondary", "priority": 4},
                "filesystem": {"level": "secondary"},
                "playwright": {"level": "forbidden"},
                "prisma": {"level": "forbidden"},
            },
            "karo_b": {
                "sequential_thinking": {"level": "secondary", "priority": 5},
                "filesystem": {"level": "secondary"},
                "playwright": {"level": "forbidden"},
                "prisma": {"level": "forbidden"},
            },
            "kengyo": {
                "playwright": {"level": "exclusive", "priority": 1},
                "filesystem": {"level": "primary"},
                "sequential_thinking": {"level": "forbidden"},
                "prisma": {"level": "forbidden"},
            },
            "onmitsu": {
                "filesystem": {"level": "delegated"},
                "git": {"level": "delegated"},
                "playwright": {"level": "forbidden"},
                "prisma": {"level": "forbidden"},
            },
        }
        self._loaded = True
        logger.info("ℹ️ デフォルトMCP権限設定を使用")

    def check_permission(self, role: str, mcp_name: str) -> MCPPermissionLevel:
        """
        役職がMCPにアクセス可能かチェック

        Args:
            role: 役職名 (daigensui, shogun, gunshi, sanbo_a, sanbo_b, karo_a, karo_b, kengyo, onmitsu)
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
    """v11.4 System modes - 9-tier architecture"""
    BATTALION = "battalion"  # Full 9-tier: Daigensui + Shogun + Gunshi + Sanbo + Karo + Kengyo + Onmitsu + All MCP
    COMPANY = "company"      # Reduced: Karo + Onmitsu + Memory MCP
    PLATOON = "platoon"      # Minimal: Onmitsu + Dynamic MCP


@dataclass
class SystemConfig:
    """v11.5 Configuration structure"""
    mode: SystemMode
    claude_api_key: str
    gemini_api_key: str
    tavily_api_key: str

    # v11.5: API keys
    groq_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None   # o3-mini (Gunshi)
    mistral_api_key: Optional[str] = None  # Mistral Large 3 (Sanbo-A)
    xai_api_key: Optional[str] = None      # Grok 4.1 Fast (Sanbo-B)

    # Optional tokens
    discord_token: Optional[str] = None
    notion_token: Optional[str] = None

    # Service endpoints
    ollama_endpoint: str = "http://localhost:11434"  # Legacy (unused in v11.4)
    litellm_endpoint: str = "http://localhost:8000"

    # v11.4: llama.cpp configuration (HP ProDesk 600 CPU optimized) - Nemotron-3-Nano (隠密)
    llamacpp_endpoint: str = "http://127.0.0.1:8080"
    llamacpp_model_path: str = "models/nemotron/Nemotron-3-Nano-Q4_K_M.gguf"
    llamacpp_threads: int = 6  # HP ProDesk 600: i5-8500 (6C/6T)
    llamacpp_context_size: int = 4096  # Optimized for CPU speed
    llamacpp_batch_size: int = 512  # CPU optimal
    llamacpp_mlock: bool = True  # Lock memory to prevent swapping

    # v11.5: Configuration settings
    version: str = "11.5"
    intelligent_routing_enabled: bool = True
    prompt_caching_enabled: bool = True
    power_optimization_enabled: bool = True
    use_llamacpp: bool = True  # Use llama.cpp instead of Ollama

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key"""
        return getattr(self, key, default)


class SystemOrchestrator:
    """
    v11.5 システムオーケストレーター - 強化調整層

    9層ハイブリッドアーキテクチャを管理:
    1. 大元帥 (Daigensui) - 最高戦略層: Claude Opus 4.6
    2. 将軍 (Shogun) - 高難度コーディング: Claude Sonnet 4.6
    3. 軍師 (Gunshi) - 推論・設計・PDCA: o3-mini (reasoning_effort=high)
    4. 参謀-A (Sanbo-A) - 汎用コーディング: Mistral Large 3
    5. 参謀-B (Sanbo-B) - 実装・バグ修正・超高速: Grok 4.1 Fast
    6. 家老-A (Karo-A) - 軽量タスク: Gemini 3 Flash
    7. 家老-B (Karo-B) - アルゴリズム特化: Llama 3.3 70B (Groq)
    8. 検校 (Kengyo) - マルチモーダル: Gemini 3 Flash Vision
    9. 隠密 (Onmitsu) - 機密・超長文: Nemotron-3-Nano (Local llama.cpp)

    機能:
    - インテリジェントルーター統合
    - 省電力最適化
    - コスト削減のプロンプトキャッシング
    - BDIフレームワーク統合
    - MCP権限マトリクス: 役職別アクセス制御
    """

    VERSION = "11.5"

    def __init__(self, config: SystemConfig):
        self.config = config
        self.mcps: Dict[str, Any] = {}
        self.clients: Dict[str, Any] = {}
        self.router = None
        self.mcp_manager = None
        self.initialized = False

        # MCP権限マネージャー
        self.permission_manager = MCPPermissionManager()

        # 9層階層コンポーネント (v11.4)
        self._shogun = None  # 将軍: 高難度コーディング (Claude Sonnet 4.6)
        self._gunshi = None  # 軍師: 推論・設計・PDCA (o3-mini)
        self._karo = None    # 家老: 軽量タスク (Gemini Flash / Groq)
        self._taisho = None  # 大将: (deprecated, maps to sanbo layer for backward compat)
        self._kengyo = None  # 検校: マルチモーダル (Gemini Flash Vision)
        self._langgraph_router = None  # LangGraph Router

        # 統計
        self.health_status: Dict[str, bool] = {}
        self.llm_availability: Dict[str, bool] = {}  # LLM可用性確認結果

        # パフォーマンス目標（秒）
        self.performance_targets = {
            "simple": 2,
            "medium": 12,
            "complex": 28,
            "strategic": 45
        }

    async def initialize(self) -> None:
        """全v11.4システムコンポーネントを初期化"""
        logger.info(f"🔧 武士団 v{self.VERSION} コンポーネント初期化開始...")

        try:
            # MCP権限マトリクス読み込み
            self.permission_manager.load_permissions()

            # MCPサーバー初期化
            await self._initialize_mcps()

            # AIクライアント初期化
            await self._initialize_clients()

            # LLM可用性確認（新規）
            await self._check_llm_availability()

            # インテリジェントルーター初期化
            await self._initialize_router()

            # 外部依存関係の検証
            await self._verify_dependencies()

            # 9層階層コンポーネント初期化
            await self._initialize_tiers()

            self.initialized = True
            logger.info(f"✅ システムオーケストレーター v{self.VERSION} 初期化完了")
            self._log_startup_summary()

        except Exception as e:
            logger.error(f"❌ システム初期化失敗: {e}")
            raise

    async def _initialize_tiers(self) -> None:
        """9層階層コンポーネントの初期化"""
        logger.info("🏯 9層階層コンポーネント初期化...")

        # 将軍（高難度コーディング - Claude Sonnet 4.6）初期化
        try:
            from core.shogun import Shogun
            self._shogun = Shogun(self)
            await self._shogun.initialize()
            logger.info("🎌 将軍（高難度コーディング）初期化完了")
        except Exception as e:
            logger.error(f"❌ 将軍初期化失敗: {e}")
            raise

        # 軍師（推論・設計・PDCA - o3-mini）初期化
        try:
            from core.gunshi import Gunshi
            self._gunshi = Gunshi(self)
            await self._gunshi.initialize()
            logger.info("🧠 軍師（推論・設計・PDCA）初期化完了")
        except Exception as e:
            logger.warning(f"⚠️ 軍師初期化失敗 (複雑タスクは家老に直接委譲): {e}")

        # 家老は将軍の初期化時に作成される
        if self._shogun and hasattr(self._shogun, 'karo'):
            self._karo = self._shogun.karo
            logger.info("👔 家老（軽量タスク層）参照取得完了")

        # v11.4: 検校（マルチモーダル - Gemini Flash Vision）初期化
        try:
            from core.kengyo import Kengyo
            gemini_client = self.clients.get("gemini3")
            self._kengyo = Kengyo(
                kimi_client=gemini_client,
                smithery_mcp=getattr(self, 'smithery_mcp', None),
                orchestrator=self,  # For Discord reporter access
            )
            await self._kengyo.initialize()
            logger.info("👁️ 検校（マルチモーダル）初期化完了")
        except Exception as e:
            logger.warning(f"⚠️ 検校初期化失敗 (マルチモーダル検証スキップ): {e}")

        # LangGraph Router 初期化
        try:
            from core.langgraph_router import LangGraphRouter
            self._langgraph_router = LangGraphRouter(self)
            await self._langgraph_router.initialize()
            logger.info("🔗 LangGraph Router 初期化完了 ✅")
        except Exception as e:
            import traceback
            logger.error(f"❌ LangGraph Router 初期化失敗 (従来ルーティング使用)")
            logger.error(f"エラー: {e}")
            logger.error(f"トレースバック:\n{traceback.format_exc()}")
            self._langgraph_router = None

        # v12: MCP-LangGraph Bridge 初期化 (langchain-mcp-adapters)
        try:
            from core.langgraph_mcp_bridge import LangGraphMCPBridge
            self._mcp_bridge = LangGraphMCPBridge(self)
            await self._mcp_bridge.initialize()
            tool_count = len(self._mcp_bridge._lc_tools)
            logger.info(f"🔗 MCP-LangGraph Bridge 初期化完了 ✅ ({tool_count} tools)")
        except Exception as e:
            logger.warning(f"⚠️ MCP Bridge 初期化失敗 (MCP統合スキップ): {e}")
            self._mcp_bridge = None

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
            from mcp_servers.smithery_manager import SmitheryMCPManager
            self.smithery_mcp = SmitheryMCPManager()
            smithery_status = await self.smithery_mcp.initialize()
            available = sum(1 for v in smithery_status.values() if v)
            total = len(smithery_status)
            logger.info(f"✅ Smithery MCP Manager: {available}/{total} servers ready")
        except Exception as e:
            self.smithery_mcp = None
            logger.warning(f"⚠️ Smithery MCP Manager not available: {e}")

    async def _initialize_clients(self) -> None:
        """Initialize all v11.4 AI clients"""

        # Claude Client with Prompt Caching (Shogun - Claude Sonnet 4.6)
        try:
            from utils.claude_client_cached import ClaudeClientCached
            self.clients["claude_cached"] = ClaudeClientCached(
                api_key=self.config.claude_api_key
            )
            logger.info("✅ Claude Client (Cached) initialized - Shogun")
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

        # Opus Client (Daigensui - Claude Opus 4.5)
        try:
            from utils.opus_client import OpusClient
            self.clients["opus"] = OpusClient(
                api_key=self.config.claude_api_key
            )
            logger.info("✅ Opus Client (大元帥) initialized")
        except Exception as e:
            logger.warning(f"⚠️ Opus client failed: {e}")

        # v11.4: o3-mini Client (Gunshi - reasoning_effort=high)
        if self.config.openai_api_key:
            try:
                from utils.o3mini_client import O3MiniClient
                self.clients["o3_mini"] = O3MiniClient(
                    api_key=self.config.openai_api_key
                )
                logger.info("✅ o3-mini Client (軍師) initialized - reasoning_effort=high")
            except Exception as e:
                logger.warning(f"⚠️ o3-mini client failed: {e}")

        # v11.5: Mistral Large 3 Client (Sanbo-A - 汎用コーディング・EU準拠)
        if self.config.mistral_api_key:
            try:
                from utils.gpt5_client import GPT5Client
                self.clients["gpt5"] = GPT5Client(
                    api_key=self.config.mistral_api_key
                )
                logger.info("✅ Mistral Large 3 Client (参謀-A) initialized")
            except Exception as e:
                logger.warning(f"⚠️ Mistral Large 3 client failed: {e}")

        # v11.5: Grok 4.1 Fast Client (Sanbo-B - 実装・バグ修正・超高速) via xAI
        if self.config.xai_api_key:
            try:
                from utils.grok_client import GrokClient
                self.clients["grok"] = GrokClient(
                    api_key=self.config.xai_api_key
                )
                logger.info("✅ Grok 4.1 Fast Client (参謀-B) initialized")
            except Exception as e:
                logger.warning(f"⚠️ Grok 4.1 Fast client failed: {e}")

        # Gemini Flash Client (Karo-A - 軽量タスク / Kengyo - マルチモーダル)
        try:
            from utils.gemini3_client import Gemini3Client
            self.clients["gemini3"] = Gemini3Client(
                api_key=self.config.gemini_api_key
            )
            logger.info("✅ Gemini Flash Client initialized - Karo-A / Kengyo")
        except Exception as e:
            logger.warning(f"⚠️ Gemini Flash client failed: {e}")
            # Fallback to standard Gemini
            try:
                from utils.gemini_client import GeminiClient
                self.clients["gemini"] = GeminiClient(
                    api_key=self.config.gemini_api_key
                )
                logger.info("✅ Gemini Client (Standard) initialized as fallback")
            except Exception as e2:
                logger.warning(f"⚠️ Gemini fallback also failed: {e2}")

        # Groq Client (Karo-B - Llama 3.3 70B アルゴリズム特化)
        if self.config.groq_api_key:
            try:
                from utils.groq_client import GroqClient
                self.clients["groq"] = GroqClient(
                    api_key=self.config.groq_api_key
                )
                logger.info("✅ Groq Client initialized - Karo-B (Llama 3.3 70B)")
            except Exception as e:
                logger.warning(f"⚠️ Groq client failed: {e}")

        # v11.4: Nemotron-3-Nano Client (Onmitsu - 機密・超長文, local llama.cpp)
        try:
            if self.config.use_llamacpp:
                from utils.nemotron_llamacpp_client import NemotronLlamaCppClient, LlamaCppConfig
                llamacpp_config = LlamaCppConfig(
                    model_path=self.config.llamacpp_model_path,
                    host=self.config.llamacpp_endpoint.split("://")[1].split(":")[0],
                    port=int(self.config.llamacpp_endpoint.split(":")[-1]),
                    threads=self.config.llamacpp_threads,
                    context_size=self.config.llamacpp_context_size,
                    batch_size=self.config.llamacpp_batch_size,
                    mlock=self.config.llamacpp_mlock
                )
                self.clients["nemotron"] = NemotronLlamaCppClient(config=llamacpp_config)
                logger.info("✅ Nemotron-3-Nano llama.cpp Client initialized (隠密, CPU optimized)")
            else:
                # Fallback to Ollama (legacy)
                from utils.nemotron_client import NemotronClient
                self.clients["nemotron"] = NemotronClient(
                    api_base=self.config.ollama_endpoint
                )
                logger.info("✅ Nemotron Client initialized (Ollama)")
        except Exception as e:
            logger.warning(f"⚠️ Nemotron client failed: {e}")

    async def _initialize_router(self) -> None:
        """Initialize Intelligent Router for v11.4"""

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

        # v11.4: Check llama.cpp server availability (for Nemotron-3-Nano / 隠密)
        if self.config.use_llamacpp:
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    response = await client.get(f"{self.config.llamacpp_endpoint}/health")
                    self.health_status["llamacpp"] = response.status_code == 200
                    if self.health_status["llamacpp"]:
                        logger.info("✅ llama.cpp server available (Nemotron-3-Nano)")
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

        # 9層階層コンポーネント
        logger.info("【9層階層コンポーネント】")
        logger.info(f"  👑 大元帥（最高戦略）: {'✅' if 'opus' in self.clients else '❌'}")
        logger.info(f"  🎌 将軍（高難度コーディング）: {'✅' if self._shogun else '❌'}")
        logger.info(f"  🧠 軍師（推論・設計・PDCA）: {'✅' if self._gunshi else '❌'}")
        logger.info(f"  📋 参謀-A（汎用コーディング）: {'✅' if 'gpt5' in self.clients else '❌'}")
        logger.info(f"  ⚡ 参謀-B（実装・高速）: {'✅' if 'grok' in self.clients else '❌'}")
        logger.info(f"  👔 家老-A（軽量タスク）: {'✅' if self._karo else '❌'}")
        logger.info(f"  🔧 家老-B（アルゴリズム特化）: {'✅' if 'groq' in self.clients else '❌'}")
        logger.info(f"  👁️ 検校（マルチモーダル）: {'✅' if self._kengyo and self._kengyo.is_available() else '❌'}")
        logger.info(f"  🥷 隠密（機密・超長文）: {'✅' if 'nemotron' in self.clients else '❌'}")

        # BDI状態
        if self._shogun and hasattr(self._shogun, 'bdi_enabled'):
            logger.info(f"  🧠 BDIフレームワーク: {'✅' if self._shogun.bdi_enabled else '❌'}")

        logger.info("-" * 60)
        logger.info("【AIクライアント】")
        client_names = {
            "opus": "Claude Opus 4.5（大元帥）",
            "claude_cached": "Claude Sonnet 4.6（将軍, キャッシュ）",
            "o3_mini": "o3-mini（軍師, reasoning_effort=high）",
            "gpt5": "GPT-5（参謀-A）",
            "grok": "Grok-code-fast-1（参謀-B）",
            "gemini3": "Gemini Flash（家老-A / 検校）",
            "groq": "Llama 3.3 70B via Groq（家老-B）",
            "nemotron": f"Nemotron-3-Nano（隠密, {'llama.cpp CPU' if self.config.use_llamacpp else 'Ollama'}）",
        }
        for key, name in client_names.items():
            status = "✅" if key in self.clients else "❌"
            logger.info(f"  {name}: {status}")

        # llama.cpp configuration info (Nemotron)
        if self.config.use_llamacpp:
            logger.info("【llama.cpp設定 (Nemotron-3-Nano)】")
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

    # ==================== 9層階層コンポーネントアクセサ ====================

    @property
    def shogun(self):
        """将軍（高難度コーディング - Claude Sonnet 4.6）取得"""
        return self._shogun

    @property
    def gunshi(self):
        """軍師（推論・設計・PDCA - o3-mini）取得"""
        return self._gunshi

    @property
    def karo(self):
        """家老（軽量タスク - Gemini Flash / Groq）取得"""
        return self._karo

    @property
    def taisho(self):
        """大将（deprecated） - 後方互換性のため維持。v11.4では参謀(sanbo)層にマッピング。"""
        return self._taisho

    @property
    def kengyo(self):
        """検校（マルチモーダル - Gemini Flash Vision）取得"""
        return self._kengyo

    @property
    def langgraph_router(self):
        """LangGraph Router 取得"""
        return self._langgraph_router

    @property
    def mcp_bridge(self):
        """MCP-LangGraph Bridge (v10.2) 取得"""
        return self._mcp_bridge

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

        v10.2: LangGraph Router を優先使用
        フォールバック: 従来の将軍による処理

        Args:
            task_content: タスク内容
            context: オプションコンテキスト

        Returns:
            処理結果
        """
        if not self.initialized:
            raise RuntimeError("システムが初期化されていません")

        # Extract Discord reporter from context (for multi-agent presence)
        discord_reporter = context.get("discord_reporter") if context else None
        task_id = context.get("task_id") if context else None

        # Store reporter for agent access during task processing
        if discord_reporter and task_id:
            self._current_task_reporter = discord_reporter
            self._current_task_id = task_id

            # Report battle start
            await discord_reporter.report_battle_start(
                task_id,
                task_content,
                strategy="LangGraph Router による最適ルーティング、5層階層による段階的処理"
            )

        try:
            # v10.2: LangGraph Router が利用可能な場合は優先
            if self._langgraph_router:
                logger.info("🔗 LangGraph Router でタスク処理")
                return await self._langgraph_router.process_task(
                    content=task_content,
                    context=context or {},
                    priority=1,
                    source="orchestrator"
                )

            # フォールバック: 従来の将軍による処理
            if not self._shogun:
                raise RuntimeError("将軍が初期化されていません")

            logger.info("🎌 従来の将軍ルーティング（LangGraph Router 未利用）")
            from core.shogun import Task, TaskComplexity
            task = Task(
                content=task_content,
                complexity=TaskComplexity.MEDIUM,  # 将軍が再評価
                context=context
            )

            return await self._shogun.process_task(task)
        finally:
            # Clean up after task processing
            self._current_task_reporter = None
            self._current_task_id = None

    def get_reporter(self):
        """Get current Discord reporter for active task (multi-agent presence)"""
        return getattr(self, '_current_task_reporter', None)

    def get_task_id(self) -> Optional[str]:
        """Get current task ID for active task (multi-agent presence)"""
        return getattr(self, '_current_task_id', None)

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

    async def _check_llm_availability(self) -> None:
        """各LLM/APIの可用性確認（初期化後）"""
        try:
            from core.liveness_checker import LLMAvailabilityChecker

            checker = LLMAvailabilityChecker()
            statuses = await checker.check_all()

            # 可用性サマリーを保存
            self.llm_availability = checker.get_available_llms()

            # サマリー出力
            logger.info(checker.print_summary())

        except Exception as e:
            logger.warning(f"⚠️ LLM可用性確認失敗: {e}")
            self.llm_availability = {}

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
