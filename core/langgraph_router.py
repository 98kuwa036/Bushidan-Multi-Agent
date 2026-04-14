"""
武士団 Multi-Agent System v15 - LangGraph Router

10役職ルーティング + ノードタイムアウト + HITL + MCP ツール注入

StateGraph v15
  [START]
    ↓
  [analyze_intent]    タスク複雑度・種別分析
    ↓
  [notion_retrieve]   Notion RAG 検索
    ↓
  [route_decision]    ルーティング判断 (10役職 + ヘルスチェック)
    ↓
  ┌─────────────────────────────────────────────────────┐
  │ groq_qa / gunshi_haiku / gaiji_rag / sanbo_mcp        │
  │ yuhitsu_jp / uketuke_default / onmitsu_local        │
  │ metsuke_proc / shogun_plan / daigensui_audit        │
  │ kengyo_vision / parallel_groq                       │
  └─────────────────────────────────────────────────────┘
    ↓
  [check_followup]    3分岐: human / loop / done
    ├─ "human" → [human_interrupt]  (HITL 中断)
    ├─ "loop"  → [notion_retrieve]  (自律ループ)
    └─ "done"  → [notion_store]     (完了)
"""

import asyncio
import hashlib
import time
from typing import Any, Literal, Optional

import os

from langgraph.graph import StateGraph, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver

from core.state import BushidanState
from utils.logger import get_logger

# Matrix 通知チャンネルマッピング (role_key → channel)
_MATRIX_CHANNEL: dict = {
    "daigensui": "strategy",
    "shogun":    "strategy",
    "gunshi":    "strategy",
    "sanbo":     "ops",
    "gaiji":     "intel",
    "uketuke":   "general",
    "seppou":    "intel",
    "kengyo":    "intel",
    "yuhitsu":   "logs",
    "onmitsu":   "ops",
}


async def _post_to_matrix(role_key: str, message: str, response: str) -> None:
    """Matrix にルーティング結果を非同期投稿 (fire-and-forget)"""
    try:
        from utils.matrix_client import MatrixClient
        channel = _MATRIX_CHANNEL.get(role_key, "general")
        client = MatrixClient(role_key)
        if not await client.connect():
            return
        # 長すぎる場合は末尾を省略
        resp_preview = response[:300] + ("..." if len(response) > 300 else "")
        text = f"📨 [{message[:60]}]\n{resp_preview}"
        await client.send_to_channel(channel, text)
        await client.close()
    except Exception as e:
        logger.debug("Matrix 投稿スキップ (%s): %s", role_key, e)

logger = get_logger(__name__)


async def _refresh_notion_index_bg():
    """Notion インデックスをバックグラウンド更新 (起動時・エラーは無視)"""
    try:
        from integrations.notion.index import refresh_index
        n = await refresh_index()
        logger.info("📋 Notionインデックス初期構築: %d件", n)
    except Exception as e:
        logger.debug("Notionインデックス初期構築スキップ: %s", e)


async def _skill_observe(thread_id: str, message: str, handled_by: str, execution_time: float) -> None:
    """スキルトラッカーにチャット結果を非同期記録 (例外は無視)"""
    try:
        from utils.skill_tracker import observe as _st_observe
        await _st_observe(thread_id, message, handled_by, execution_time)
    except Exception as e:
        logger.debug("skill_observe スキップ: %s", e)


# ── PostgreSQL 接続設定 ────────────────────────────────────────────────────
POSTGRES_URL = os.environ.get(
    "POSTGRES_URL",
    "postgresql://postgres:kuwa1998@192.168.11.236/bushidan",
)

# ── ノード別タイムアウト (秒) ────────────────────────────────────────────────
NODE_TIMEOUTS = {
    "groq_qa":           30,
    "uketuke_default":   60,
    "gaiji_rag":         60,
    "sanbo_mcp":         60,
    "kengyo_vision":     60,
    "gunshi_haiku":      60,   # v18: 軍師(Command A)
    "metsuke_proc":      45,   # v18: 目付(Mistral Small) 低中難度
    "yuhitsu_jp":        90,
    "onmitsu_local":    120,
    "execute_step":     120,   # v16: ロードマップステップ実行
    "shogun_plan":      120,   # v16: 将軍ロードマップ作成
    "daigensui_audit":  200,   # v16: 大元帥監査 — Claude CLI + API fallback
}

# ── フォールバックマップ (障害時の代替ルート) ────────────────────────────────
_FALLBACK_MAP = {
    "groq_qa":           "uketuke_default",
    "parallel_groq":     "groq_qa",
    "gunshi_haiku":      "metsuke_proc",    # 軍師(Command A) → 目付(Mistral Small)
    "metsuke_proc":      "groq_qa",         # 目付 → 斥候(Groq)
    "gaiji_rag":         "groq_qa",         # 外事(Cohere)障害 → 斥候(Groq)で知識回答
    "sanbo_mcp":         "gunshi_haiku",    # 参謀(Gemini)障害 → 軍師(Command A)
    "yuhitsu_jp":        "metsuke_proc",    # 右筆(Gemma)障害 → 目付(Mistral Small)
    "onmitsu_local":     "uketuke_default",
    "kengyo_vision":     "gunshi_haiku",    # 検校(Gemini)障害 → 軍師(Command A)
    "shogun_plan":       "gunshi_haiku",    # 将軍プラン → 軍師(Command A)
    "execute_step":      "gunshi_haiku",    # ステップ失敗 → 軍師
    "daigensui_audit":   "shogun_plan",     # 監査障害 → 将軍プランへ
    "uketuke_default":   "groq_qa",
}


# =============================================================================
# Role Registry — 遅延インポートでロール一覧を管理
# =============================================================================

def _load_roles() -> dict:
    """roles/ パッケージから全ロールをロードする (起動時1回のみ)"""
    from roles.uketuke import UketukeRole
    from roles.gaiji import GaijiRole
    from roles.seppou import SeppouRole
    from roles.gunshi import GunshiRole
    from roles.metsuke import MetsukeRole
    from roles.sanbo import SanboRole
    from roles.shogun import ShogunRole
    from roles.daigensui import DaigensuiRole
    from roles.yuhitsu import YuhitsuRole
    from roles.onmitsu import OnmitsuRole
    from roles.kengyo import KengyoRole
    return {
        "uketuke":   UketukeRole(),
        "gaiji":     GaijiRole(),
        "seppou":    SeppouRole(),
        "gunshi":    GunshiRole(),
        "metsuke":   MetsukeRole(),
        "sanbo":     SanboRole(),
        "shogun":    ShogunRole(),
        "daigensui": DaigensuiRole(),
        "yuhitsu":   YuhitsuRole(),
        "onmitsu":   OnmitsuRole(),
        "kengyo":    KengyoRole(),
    }


# =============================================================================
# LangGraph Router v15
# =============================================================================

class LangGraphRouter:
    """LangGraph StateGraph v15 — タイムアウト + ヘルスチェック + HITL"""

    # ── 応答キャッシュ (TTL 5分、シンプルQ&A専用) ─────────────────────
    _RESP_CACHE: dict[str, tuple[dict, float]] = {}
    _RESP_CACHE_TTL: float = 300.0   # 5分

    def __init__(self, orchestrator: Any = None):
        self.orchestrator = orchestrator
        self._roles: dict = {}
        self._compiled = None
        self._compiled_fast = None  # MemorySaver版 (短文Q&A用・PGなし)
        self._checkpointer = None
        self._pool = None
        # PostgreSQL 接続状態追跡
        self._pg_status: str = "initializing"   # "connected" | "disconnected" | "reconnecting"
        self._pg_error: str = ""
        self._pg_reconnect_task: Optional[asyncio.Task] = None
        self._memory_fallback: Optional[MemorySaver] = None  # PG障害時の一時保存
        # YAML監査ログ — スレッドIDをキーにした AuditLog インスタンス保管
        self._audit_logs: dict = {}  # thread_id → AuditLog

    async def initialize(self) -> None:
        logger.info("🔗 LangGraph Router v16 初期化開始")
        try:
            # PostgresSaver を優先、失敗時 MemorySaver にフォールバック
            self._checkpointer = await self._init_checkpointer()

            # MCP ツールレジストリ初期化 (失敗しても続行)
            try:
                from core.mcp_sdk import MCPToolRegistry
                registry = MCPToolRegistry.get()
                await registry.initialize()
                logger.info("🔧 MCP tools: %s", registry.available_tools[:10] or "none")
            except Exception as e:
                logger.warning("⚠️ MCPToolRegistry 初期化スキップ: %s", e)

            # SkillTracker DB スキーマ初期化
            try:
                from utils.skill_tracker import ensure_schema as _skill_schema
                await _skill_schema()
            except Exception as _e:
                logger.warning("⚠️ SkillTracker schema skip: %s", _e)

            self._roles = _load_roles()
            self._compiled = self._build_graph().compile(
                checkpointer=self._checkpointer
            )
            # 短文Q&A用 MemorySaver グラフ (PG I/O ゼロ)
            self._compiled_fast = self._build_graph().compile(
                checkpointer=MemorySaver()
            )
            self._health_task = asyncio.create_task(
                self._background_health_check(), name="health_check_bg"
            )
            # MemorySaver 動作中の場合はバックグラウンド再接続ループ起動
            if self._pg_status == "disconnected":
                self._pg_reconnect_task = asyncio.create_task(
                    self._background_pg_reconnect(), name="pg_reconnect_bg"
                )
            checkpointer_type = type(self._checkpointer).__name__
            # Notion インデックスをバックグラウンドで初期構築
            asyncio.create_task(_refresh_notion_index_bg(), name="notion_index_init")
            logger.info("✅ LangGraph Router v16 初期化完了 (%s)", checkpointer_type)
        except Exception as e:
            import traceback
            logger.error("❌ LangGraph Router 初期化エラー: %s\n%s", e, traceback.format_exc())
            raise

    async def _init_checkpointer(self):
        """PostgresSaver を初期化。失敗時は MemorySaver にフォールバック。"""
        try:
            import psycopg
            from psycopg_pool import AsyncConnectionPool
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

            # ── テーブルセットアップ (autocommit 単独接続で実行) ────────
            async with await asyncio.wait_for(
                psycopg.AsyncConnection.connect(POSTGRES_URL, autocommit=True),
                timeout=10,
            ) as setup_conn:
                setup_saver = AsyncPostgresSaver(setup_conn)
                await asyncio.wait_for(setup_saver.setup(), timeout=15)

            # ── コネクションプール ──────────────────────────────────────
            # min_size=1: 起動直後から1接続を維持
            # max_size=5: 並列処理上限
            # open=False + await pool.open(wait=True): 非同期で確実に開く
            # reconnect_timeout=30: 切断時の再接続待ち上限
            self._pool = AsyncConnectionPool(
                conninfo=POSTGRES_URL,
                min_size=1,
                max_size=5,
                open=False,
                reconnect_timeout=30.0,
                kwargs={"autocommit": True},
            )
            await asyncio.wait_for(
                self._pool.open(wait=True), timeout=20
            )

            checkpointer = AsyncPostgresSaver(self._pool)
            self._pg_status = "connected"
            self._pg_error = ""
            logger.info("✅ PostgresSaver 初期化完了 (%s)", POSTGRES_URL.split("@")[-1])
            return checkpointer

        except Exception as e:
            self._pg_status = "disconnected"
            self._pg_error = str(e)
            logger.warning("⚠️  PostgresSaver 失敗 → MemorySaver にフォールバック: %s", e)
            if self._pool:
                try:
                    await self._pool.close()
                except Exception:
                    pass
                self._pool = None
            fallback = MemorySaver()
            self._memory_fallback = fallback
            return fallback

    async def _background_pg_reconnect(self) -> None:
        """
        PostgreSQL 再接続ループ。
        60秒ごとに接続を試み、成功したら MemorySaver のデータを移行してグラフを再コンパイルする。
        """
        logger.info("🔄 PostgreSQL 再接続ループ開始 (60秒間隔)")
        while self._pg_status != "connected":
            await asyncio.sleep(60)
            self._pg_status = "reconnecting"
            logger.info("🔄 PostgreSQL 再接続試行中...")
            try:
                import psycopg
                from psycopg_pool import AsyncConnectionPool
                from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

                # テーブルセットアップ
                async with await asyncio.wait_for(
                    psycopg.AsyncConnection.connect(POSTGRES_URL, autocommit=True),
                    timeout=10,
                ) as setup_conn:
                    setup_saver = AsyncPostgresSaver(setup_conn)
                    await asyncio.wait_for(setup_saver.setup(), timeout=15)

                # 新しいコネクションプール
                new_pool = AsyncConnectionPool(
                    conninfo=POSTGRES_URL,
                    min_size=1, max_size=5,
                    open=False,
                    reconnect_timeout=30.0,
                    kwargs={"autocommit": True},
                )
                await asyncio.wait_for(new_pool.open(wait=True), timeout=20)

                new_saver = AsyncPostgresSaver(new_pool)

                # ── MemorySaver のデータを PostgreSQL に移行 ──────────────
                migrated = 0
                if self._memory_fallback is not None:
                    storage = getattr(self._memory_fallback, "storage", {})
                    for thread_id, checkpoints in storage.items():
                        for checkpoint_id, (checkpoint, metadata) in checkpoints.items():
                            try:
                                cfg = {"configurable": {"thread_id": thread_id,
                                                         "checkpoint_id": checkpoint_id}}
                                versions: dict = {}
                                await new_saver.aput(cfg, checkpoint, metadata, versions)
                                migrated += 1
                            except Exception as me:
                                logger.warning("⚠️ チェックポイント移行失敗 %s/%s: %s",
                                               thread_id, checkpoint_id, me)
                    if migrated:
                        logger.info("✅ MemorySaver → PostgreSQL 移行: %d チェックポイント", migrated)

                # ── グラフを PostgresSaver で再コンパイル ──────────────────
                if self._pool:
                    try:
                        await self._pool.close()
                    except Exception:
                        pass
                self._pool = new_pool
                self._checkpointer = new_saver
                self._memory_fallback = None
                self._compiled = self._build_graph().compile(checkpointer=self._checkpointer)

                self._pg_status = "connected"
                self._pg_error = ""
                logger.info("✅ PostgreSQL 再接続成功。移行: %d件 グラフ再コンパイル完了", migrated)

            except Exception as e:
                self._pg_status = "disconnected"
                self._pg_error = str(e)
                logger.warning("⚠️ PostgreSQL 再接続失敗: %s", e)

    @property
    def pg_status(self) -> dict:
        """PostgreSQL 接続状態を返す"""
        return {
            "status": self._pg_status,
            "error":  self._pg_error,
            "saver":  type(self._checkpointer).__name__ if self._checkpointer else "none",
        }

    async def _background_health_check(self) -> None:
        """5分間隔で全ロールのヘルスチェックを実行し、キャッシュを更新する。"""
        await asyncio.sleep(10)  # 起動直後は待機
        while True:
            try:
                from utils.client_registry import ClientRegistry
                registry = ClientRegistry.get()
                results = await registry.health_check_all()
                unhealthy = [k for k, v in results.items() if not v]
                if unhealthy:
                    logger.warning("🏥 unhealthy ロール: %s", unhealthy)
                else:
                    logger.debug("🏥 全ロール healthy")
            except Exception as e:
                logger.debug("🏥 ヘルスチェック失敗: %s", e)
            await asyncio.sleep(300)  # 5分間隔

    def _build_graph(self) -> StateGraph:
        graph = StateGraph(BushidanState)

        # ── ノード登録 v16 ────────────────────────────────────────────
        # フロントエンド
        graph.add_node("analyze_intent",  self._analyze_intent)        # 受付 (Gemini Flash-Lite)
        graph.add_node("notion_index",    self._notion_index_node)      # v16: ローカルインデックス検索

        # 直接回答パス (軽量)
        graph.add_node("groq_qa",         self._exec_node("seppou",   "groq_qa"))
        graph.add_node("parallel_groq",   self._parallel_groq_node)
        graph.add_node("gunshi_haiku",    self._exec_node("gunshi",   "gunshi_haiku"))  # v18: 軍師統一
        graph.add_node("metsuke_proc",    self._exec_node("metsuke", "metsuke_proc"))  # v18: 目付
        graph.add_node("gaiji_rag",       self._exec_node("gaiji",    "gaiji_rag"))
        graph.add_node("sanbo_mcp",       self._exec_node("sanbo",    "sanbo_mcp"))
        graph.add_node("yuhitsu_jp",      self._exec_node("yuhitsu",  "yuhitsu_jp"))
        graph.add_node("uketuke_default", self._exec_node("uketuke",  "uketuke_default"))
        graph.add_node("onmitsu_local",   self._exec_node("onmitsu",  "onmitsu_local"))
        graph.add_node("kengyo_vision",   self._exec_node("kengyo",   "kengyo_vision"))

        # 将軍プランニングパス (v16)
        graph.add_node("shogun_plan",     self._shogun_plan_node)       # v16: ロードマップ作成
        graph.add_node("execute_step",    self._execute_step_node)      # v16: ステップ実行ループ
        graph.add_node("daigensui_audit", self._daigensui_audit_node)   # v16: Opus最終監査

        # 後処理
        graph.add_node("sandbox_verify",  self._sandbox_verify_node)
        graph.add_node("check_followup",  self._check_followup)
        graph.add_node("human_interrupt", self._human_interrupt)
        graph.add_node("notion_store",    self._notion_store)

        # ── エッジ ────────────────────────────────────────────────────
        graph.set_entry_point("analyze_intent")
        graph.add_edge("analyze_intent", "notion_index")

        # notion_index → ルーティング (v16)
        graph.add_conditional_edges(
            "notion_index",
            self._route_decision,
            {
                "groq_qa":          "groq_qa",
                "parallel_groq":    "parallel_groq",
                "gunshi_haiku":     "gunshi_haiku",     # v16
                "gunshi_pdca":      "gunshi_haiku",     # 後方互換 → Haiku
                "gaiji_rag":        "gaiji_rag",
                "sanbo_mcp":        "sanbo_mcp",
                "yuhitsu_jp":       "yuhitsu_jp",
                "uketuke_default":  "uketuke_default",
                "onmitsu_local":    "onmitsu_local",
                "kengyo_vision":    "kengyo_vision",
                "shogun_plan":      "shogun_plan",       # v16: 将軍プランへ
            },
        )

        # 直接回答パス → sandbox_verify or check_followup
        for node in ("gunshi_haiku", "metsuke_proc", "gaiji_rag", "sanbo_mcp",
                     "yuhitsu_jp", "uketuke_default", "onmitsu_local",
                     "kengyo_vision", "groq_qa", "parallel_groq"):
            graph.add_edge(node, "check_followup")

        # 将軍プランニングパス
        graph.add_edge("shogun_plan", "execute_step")
        graph.add_conditional_edges(
            "execute_step",
            self._step_decision,
            {
                "next_step": "execute_step",
                "audit":     "daigensui_audit",
                "verify":    "sandbox_verify",
                "done":      "check_followup",
            },
        )
        graph.add_edge("daigensui_audit", "sandbox_verify")
        graph.add_edge("sandbox_verify",  "check_followup")

        # check_followup → 3分岐
        graph.add_conditional_edges(
            "check_followup",
            self._followup_decision,
            {
                "human": "human_interrupt",
                "loop":  "notion_index",
                "done":  "notion_store",
            },
        )
        graph.add_edge("human_interrupt", "notion_store")
        graph.add_edge("notion_store", END)

        return graph

    # =========================================================================
    # ノード: analyze_intent — 受付 (Gemini Flash-Lite) 意図解析専念 v16
    # =========================================================================

    # ── ショートカット判定用定数 ──────────────────────────────────────
    _GREETING_SET = frozenset([
        "こんにちは", "おはよう", "こんばんは", "やあ", "ねえ", "おい",
        "hello", "hi", "hey", "yo", "morning",
    ])
    _CONF_KWS_FAST = frozenset(["機密", "秘密", "confidential", "社外秘", "オフライン", "外部送信禁止"])
    _JP_WRITING_KWS = frozenset([
        "日本語で書いて", "和訳", "添削", "翻訳して", "ビジネスメール", "敬語で",
        "メールを書", "手紙を書", "校正して",
    ])

    async def _analyze_intent(self, state: BushidanState) -> dict:
        """受付 (Gemini 3.1 Flash-Lite) でメッセージを分析し、LangGraph ルーティング情報を返す。
        ショートカット条件に合致する場合は Flash-Lite 呼び出しをスキップ。
        LLM 呼び出し失敗時はキーワードフォールバックで継続。
        """
        message = state.get("message", "")
        has_vision = bool(state.get("attachments"))
        forced = state.get("forced_role") or ("kengyo" if has_vision else None)

        # ── ① ショートカット: Flash-Lite呼び出し不要なケース ─────────────
        # 画像添付 → kengyo 確定
        if has_vision:
            logger.info("🚪 [受付] ショートカット: 画像添付 → kengyo")
            return {
                "complexity": "medium", "is_multi_step": False, "is_action_task": False,
                "is_simple_qa": False, "is_japanese_priority": False, "is_confidential": False,
                "attachments": state.get("attachments", []), "forced_role": "kengyo",
                "intent_structured": {"intent_type": "image", "domain": "general",
                                      "required_capabilities": ["image"], "user_goal": message[:80]},
            }

        # 強制ロール指定 → ルーティングのみ確定、分析スキップ
        if state.get("forced_role"):
            logger.info("🚪 [受付] ショートカット: forced_role=%s", state["forced_role"])
            return {
                "complexity": "medium", "is_multi_step": False, "is_action_task": False,
                "is_simple_qa": False, "is_japanese_priority": False, "is_confidential": False,
                "attachments": [], "forced_role": state["forced_role"],
                "intent_structured": {"intent_type": "task", "domain": "general",
                                      "required_capabilities": [], "user_goal": message[:80]},
            }

        msg_lower = message.lower().strip()
        msg_stripped = message.strip()

        # 機密キーワード → onmitsu 確定
        if any(kw in msg_stripped for kw in self._CONF_KWS_FAST):
            logger.info("🚪 [受付] ショートカット: 機密キーワード → onmitsu")
            return {
                "complexity": "medium", "is_multi_step": False, "is_action_task": False,
                "is_simple_qa": False, "is_japanese_priority": False, "is_confidential": True,
                "attachments": [], "forced_role": forced,
                "intent_structured": {"intent_type": "confidential", "domain": "security",
                                      "required_capabilities": [], "user_goal": message[:80]},
            }

        # 超短文 (15文字以下) → simple Q&A 確定 (挨拶・単語レベル)
        if len(msg_stripped) <= 15:
            is_greeting = any(g in msg_lower for g in self._GREETING_SET)
            logger.info("🚪 [受付] ショートカット: 超短文(%d文字) → simple", len(msg_stripped))
            return {
                "complexity": "simple", "is_multi_step": False, "is_action_task": False,
                "is_simple_qa": not is_greeting, "is_japanese_priority": False, "is_confidential": False,
                "attachments": [], "forced_role": forced,
                "intent_structured": {"intent_type": "qa", "domain": "general",
                                      "required_capabilities": [], "user_goal": msg_stripped},
            }

        # 日本語文章作成キーワード → japanese 確定
        if any(kw in msg_stripped for kw in self._JP_WRITING_KWS):
            logger.info("🚪 [受付] ショートカット: 日本語文章作成 → yuhitsu")
            return {
                "complexity": "medium", "is_multi_step": False, "is_action_task": False,
                "is_simple_qa": False, "is_japanese_priority": True, "is_confidential": False,
                "attachments": [], "forced_role": forced,
                "intent_structured": {"intent_type": "japanese", "domain": "general",
                                      "required_capabilities": ["japanese"], "user_goal": message[:80]},
            }

        logger.info("🚪 [受付] 分析開始: '%s'...", message[:60])

        # ── Gemini Flash-Lite による分析 ──────────────────────────────────
        try:
            from utils.client_registry import ClientRegistry
            client = ClientRegistry.get().get_client("uketuke")
            if client:
                system = (
                    "あなたは武士団マルチエージェントシステムの受付係です。"
                    "ユーザーのメッセージを分析し、以下の JSON **のみ** を返してください。余分なテキスト不要。\n\n"
                    "{\n"
                    '  "complexity": "simple"|"low_medium"|"medium"|"complex"|"strategic",\n'
                    '  "intent_type": "qa"|"task"|"analysis"|"research"|"code"|"creative"|"rag"|"image"|"japanese"|"confidential",\n'
                    '  "domain": "tech"|"business"|"creative"|"security"|"general",\n'
                    '  "required_capabilities": ["analysis","rag","web_search","code","tools","japanese","image","quick_task"],\n'
                    '  "user_goal": "ユーザーが達成したいことを1文で",\n'
                    '  "is_multi_step": true|false,\n'
                    '  "is_action_task": true|false,\n'
                    '  "is_simple_qa": true|false,\n'
                    '  "is_japanese_priority": true|false,\n'
                    '  "is_confidential": true|false\n'
                    "}\n\n"
                    "判断基準:\n"
                    "- simple: 短い質問・挨拶・事実確認\n"
                    "- low_medium: やや詳しい説明・要約・軽い比較や整理\n"
                    "- medium: 説明・分析・中程度の質問\n"
                    "- complex: コーディング・実装・複数ステップ作業\n"
                    "- strategic: アーキテクチャ設計・戦略立案\n"
                    "- required_capabilities は複数選択可 (空配列も可)\n"
                    "- rag: 既存ドキュメント検索が必要な場合\n"
                    "- web_search: 最新情報・外部情報が必要な場合\n"
                    "- tools: git/ファイル/shell操作が必要な場合\n"
                    "- is_confidential: 機密・社外秘・オフライン指定の場合 true\n"
                    "⚠️重要: is_japanese_priority は日本語で書かれているだけでは true にしない。"
                    "翻訳・添削・ビジネスメール・敬語変換など日本語文章処理が明示的に求められる場合のみ true。"
                    "挨拶・質問・コード依頼・分析依頼は false。"
                )
                raw = await client.generate(
                    messages=[{"role": "user", "content": message[:1000]}],
                    system=system,
                    max_tokens=120,
                )
                # JSON 抽出
                import re, json
                m = re.search(r'\{[^}]+\}', raw, re.DOTALL)
                if m:
                    parsed = json.loads(m.group())
                    complexity      = parsed.get("complexity", "medium")
                    is_multi        = bool(parsed.get("is_multi_step", False))
                    is_action       = bool(parsed.get("is_action_task", False))
                    is_simple_qa    = bool(parsed.get("is_simple_qa", False))
                    is_japanese     = bool(parsed.get("is_japanese_priority", False))
                    is_confidential = bool(parsed.get("is_confidential", False))
                    intent_structured = {
                        "intent_type":            parsed.get("intent_type", "general"),
                        "domain":                 parsed.get("domain", "general"),
                        "required_capabilities":  parsed.get("required_capabilities", []),
                        "user_goal":              parsed.get("user_goal", message[:120]),
                    }
                    logger.info("🚪 [受付] 分析完了: complexity=%s caps=%s goal='%s'",
                                complexity, intent_structured["required_capabilities"],
                                intent_structured["user_goal"][:40])
                    return {
                        "complexity": complexity,
                        "is_multi_step": is_multi,
                        "is_action_task": is_action,
                        "is_simple_qa": is_simple_qa,
                        "is_japanese_priority": is_japanese,
                        "is_confidential": is_confidential,
                        "attachments": state.get("attachments", []),
                        "forced_role": forced,
                        "intent_structured": intent_structured,
                    }
        except Exception as e:
            logger.warning("🚪 [受付] LLM分析失敗 → キーワードフォールバック: %s", e)

        # ── キーワードフォールバック ──────────────────────────────────────
        content_lower = message.lower()
        strategic_kws = ["設計", "アーキテクチャ", "システム全体", "戦略", "design", "architecture"]
        complex_kws   = ["実装して", "作って", "修正して", "リファクタ", "implement", "build", "refactor"]
        if any(kw in message for kw in strategic_kws) or len(message) > 800:
            complexity = "strategic"
        elif any(kw in message for kw in complex_kws) or len(message) > 300:
            complexity = "complex"
        elif len(message) < 60:
            complexity = "simple"
        else:
            complexity = "medium"
        is_multi        = any(kw in content_lower for kw in ["そして", "次に", "さらに", "まず", "step", "then"]) and len(message) > 100
        is_action       = any(kw in content_lower for kw in ["clone", "push", "pull", "install", "実行して", "削除して", "作成して"])
        is_simple_qa    = any(kw in content_lower for kw in ["とは", "って何", "ですか", "what is", "explain"]) and not is_action
        is_japanese     = any(kw in message for kw in ["日本語で書いて", "和訳", "添削", "翻訳", "ビジネスメール", "敬語で"])
        is_confidential = any(kw in content_lower for kw in ["機密", "秘密", "confidential", "オフライン", "社外秘"])
        logger.info("🚪 [受付] フォールバック分析: complexity=%s", complexity)
        return {
            "complexity": complexity,
            "is_multi_step": is_multi,
            "is_action_task": is_action,
            "is_simple_qa": is_simple_qa,
            "is_japanese_priority": is_japanese,
            "is_confidential": is_confidential,
            "attachments": state.get("attachments", []),
            "forced_role": forced,
            "intent_structured": {
                "intent_type": "general",
                "domain": "general",
                "required_capabilities": [],
                "user_goal": message[:120],
            },
        }

    # =========================================================================
    # ノード: notion_index — v16 ローカルインデックス検索
    # =========================================================================

    async def _notion_index_node(self, state: BushidanState) -> dict:
        """ローカルインデックスで高速検索。ヒットしたページのみ詳細取得。
        simple/qa では空リストを返してスキップ (Notion API 呼び出しなし)。
        """
        complexity = state.get("complexity", "medium")
        is_simple  = state.get("is_simple_qa", False)

        # simple + qa はNotion不要
        if complexity == "simple" and is_simple:
            return {"notion_chunks": []}

        # 受付が抽出した user_goal でインデックス検索
        intent = state.get("intent_structured", {})
        user_goal = intent.get("user_goal", "") or state.get("message", "")[:120]

        try:
            from integrations.notion.index import lookup
            chunks = await lookup(user_goal, top_k=3)
            if chunks:
                logger.info("📋 [notion_index] %d件ヒット (goal='%s')", len(chunks), user_goal[:40])
            return {"notion_chunks": [c.__dict__ if hasattr(c, "__dict__") else c for c in chunks]}
        except Exception as e:
            logger.debug("[notion_index] スキップ: %s", e)
            return {"notion_chunks": []}

    # =========================================================================
    # ルーティング判断 (ヘルスチェック統合)
    # =========================================================================

    def _route_decision(self, state: BushidanState) -> str:
        """v16 ルーティング — 受付の構造化インテントを活用した細粒度振り分け"""
        forced = state.get("forced_role")

        _valid = {
            "groq_qa", "parallel_groq", "gunshi_haiku", "metsuke_proc", "gaiji_rag", "sanbo_mcp",
            "yuhitsu_jp", "uketuke_default", "onmitsu_local", "kengyo_vision",
            "shogun_plan", "daigensui_audit",
        }
        # forced_role → ノード名変換
        _role_to_node = {
            "seppou":    "groq_qa",
            "gunshi":    "gunshi_haiku",
            "metsuke":   "metsuke_proc",
            "gaiji":     "gaiji_rag",
            "sanbo":     "sanbo_mcp",
            "yuhitsu":   "yuhitsu_jp",
            "uketuke":   "uketuke_default",
            "onmitsu":   "onmitsu_local",
            "shogun":    "shogun_plan",
            "daigensui": "daigensui_audit",    # 大元帥明示選択 → Opus監査ノード
            "kengyo":    "kengyo_vision",
        }
        _node_to_role = {
            "groq_qa":         "seppou",
            "gunshi_haiku":    "gunshi",
            "metsuke_proc":    "metsuke",
            "gaiji_rag":       "gaiji",
            "sanbo_mcp":       "sanbo",
            "yuhitsu_jp":      "yuhitsu",
            "uketuke_default": "uketuke",
            "onmitsu_local":   "onmitsu",
            "kengyo_vision":   "kengyo",
            "shogun_plan":     "shogun",
            "daigensui_audit": "daigensui",
        }

        def _check_health(node: str) -> str:
            role = _node_to_role.get(node)
            if not role:
                return node
            try:
                from utils.client_registry import ClientRegistry
                registry = ClientRegistry.get()
                if registry.is_healthy_cached(role):
                    return node
                fallback = _FALLBACK_MAP.get(node)
                if fallback and fallback != node:
                    logger.warning("⚠️ %s unhealthy → fallback: %s", node, fallback)
                    return fallback
            except Exception:
                pass
            return node

        # ── 強制ルーティング ──────────────────────────────────────────
        if forced:
            node = _role_to_node.get(forced, forced)
            if node in _valid:
                logger.info("🎯 Route: %s (forced_role=%s)", node, forced)
                return _check_health(node)

        # ── スキルヒント (承認済みスキルによるルーティング優先) ────────
        try:
            from utils.skill_tracker import get_route_hint as _skill_hint
            _hint_role = _skill_hint(state.get("message", ""))
            if _hint_role:
                _hint_node = _role_to_node.get(_hint_role)
                if _hint_node and _hint_node in _valid:
                    logger.info("🎯 Route: %s (skill_hint=%s)", _hint_node, _hint_role)
                    return _check_health(_hint_node)
        except Exception:
            pass

        # ── 特殊条件 (最優先: 機密・画像) ───────────────────────────
        if state.get("is_confidential"):
            logger.info("🥷 Route: onmitsu_local (機密)")
            return _check_health("onmitsu_local")

        if state.get("attachments"):
            logger.info("🖼️  Route: kengyo_vision (画像あり)")
            return _check_health("kengyo_vision")

        # ── 受付の構造化インテントを活用 ─────────────────────────────
        intent = state.get("intent_structured", {})
        intent_type = intent.get("intent_type", "")
        required_caps = intent.get("required_capabilities", [])

        # RAG専用: ドキュメント検索が主目的
        if intent_type == "rag" or (
            "rag" in required_caps and
            not any(c in required_caps for c in ["code", "tools", "web_search"])
        ):
            logger.info("📚 Route: gaiji_rag (RAG特化)")
            return _check_health("gaiji_rag")

        # Web検索: 外部最新情報が必要 → 外事(tavily_search常時実行)
        if "web_search" in required_caps or intent_type == "research":
            logger.info("🌐 Route: gaiji_rag (web_search)")
            return _check_health("gaiji_rag")

        # ── complexity ベースのルーティング (メイン判断) ─────────────
        complexity = state.get("complexity", "medium")
        is_multi   = state.get("is_multi_step", False)
        is_action  = state.get("is_action_task", False)
        is_simple  = state.get("is_simple_qa", False)

        # strategic / complex → 将軍ロードマップ
        if complexity in ("strategic", "complex") or (is_multi and complexity not in ("simple",)):
            logger.info("🗺️  Route: shogun_plan (complexity=%s multi=%s)", complexity, is_multi)
            return _check_health("shogun_plan")

        # action + tools → 参謀
        if is_action and "tools" in required_caps:
            logger.info("🔧 Route: sanbo_mcp (action+tools)")
            return _check_health("sanbo_mcp")

        # 簡易コード (simple/low_medium) → 斥候(Groq爆速) → CodeRabbitレビュー
        if "code" in required_caps and complexity in ("simple", "low_medium"):
            logger.info("⚡💻 Route: groq_qa (quick code + CodeRabbit)")
            return _check_health("groq_qa")

        # 日本語文章作成 (complex未満のみ。複雑な日本語タスクは shogun_plan へ)
        if state.get("is_japanese_priority") and complexity in ("simple", "low_medium", "medium"):
            logger.info("🖊️  Route: yuhitsu_jp (日本語文章)")
            return _check_health("yuhitsu_jp")

        # medium → 軍師(Haiku)
        if complexity == "medium" or is_action:
            message = state.get("message", "")
            if is_multi and message.count("?") + message.count("？") >= 2:
                logger.info("⚡⚡ Route: parallel_groq (multi-query)")
                return _check_health("parallel_groq")
            logger.info("🧠 Route: gunshi_haiku (medium)")
            return _check_health("gunshi_haiku")

        # low_medium → 目付(Mistral Small)
        if complexity == "low_medium":
            logger.info("🔎 Route: metsuke_proc (low_medium)")
            return _check_health("metsuke_proc")

        # simple Q&A → 斥候 (Groq 最速)
        if is_simple or complexity == "simple":
            message = state.get("message", "")
            if is_multi and message.count("?") + message.count("？") >= 2:
                logger.info("⚡⚡ Route: parallel_groq (multi-query simple)")
                return _check_health("parallel_groq")
            logger.info("⚡ Route: groq_qa (simple Q&A)")
            return _check_health("groq_qa")

        logger.info("🏯 Route: uketuke_default (フォールバック)")
        return "uketuke_default"

    # =========================================================================
    # ノード: 実行ノード生成ファクトリ (タイムアウト付き)
    # =========================================================================

    def _exec_node(self, role_key: str, node_name: str):
        """指定ロールの execute() をタイムアウト付きで呼ぶノード関数を返す。"""
        timeout = NODE_TIMEOUTS.get(node_name, 120)

        async def _node(state: BushidanState) -> dict:
            role = self._roles.get(role_key)
            if not role:
                logger.error("ロール未ロード: %s", role_key)
                return {
                    "response": f"❌ ロール {role_key} 未初期化",
                    "handled_by": node_name, "agent_role": role_key,
                    "execution_time": 0.0, "error": "role not loaded",
                    "routed_to": node_name, "mcp_tools_used": [],
                }

            # MCPツール注入 — ロール実行前にツール一覧を state に設定
            try:
                from core.mcp_sdk import MCPToolRegistry
                registry = MCPToolRegistry.get()
                lc_tools = registry.get_tools_for_role(role_key)
                if lc_tools:
                    state = dict(state)
                    state["available_tools"] = [t.name for t in lc_tools]
                    state["mcp_tools"] = lc_tools   # ロールが直接使用可能
            except Exception:
                pass

            start = time.time()
            try:
                result = await asyncio.wait_for(
                    role.execute(state), timeout=timeout
                )
            except asyncio.TimeoutError:
                elapsed = time.time() - start
                logger.warning("⏱️ %s タイムアウト (%ds)", node_name, timeout)
                return {
                    "response": f"⏱️ {role_key} タイムアウト ({timeout}秒)",
                    "handled_by": node_name, "agent_role": role_key,
                    "execution_time": elapsed, "error": f"timeout after {timeout}s",
                    "routed_to": node_name, "mcp_tools_used": [],
                    "requires_followup": False,
                    "conversation_history": [
                        {"role": "user", "content": state.get("message", "")},
                        {"role": "assistant", "content": f"⏱️ タイムアウト ({timeout}秒)"},
                    ],
                }

            # Bushidan ファイル書き込み (LLMレスポンスに [FILE:xxx]...[/FILE] があれば保存)
            saved_files: list = []
            try:
                from utils.bushidan_files import extract_and_save_files
                saved_files = extract_and_save_files(result.response)
                if saved_files:
                    logger.info("📁 Bushidan保存 [%s]: %s", node_name, saved_files)
            except Exception as _fe:
                logger.debug("Bushidanファイル保存スキップ: %s", _fe)

            new_history = [
                {"role": "user",      "content": state.get("message", "")},
                {"role": "assistant", "content": result.response},
            ]

            # Matrix に結果を非同期投稿 (エラーでも処理は止めない)
            if result.response and not result.error:
                asyncio.create_task(
                    _post_to_matrix(role_key, state.get("message", ""), result.response),
                    name=f"matrix_{node_name}",
                )

            return {
                "response":       result.response,
                "handled_by":     result.handled_by,
                "agent_role":     result.agent_role,
                "execution_time": result.execution_time,
                "error":          result.error,
                "mcp_tools_used": result.mcp_tools_used + saved_files,
                "requires_followup": result.requires_followup,
                "routed_to":      node_name,
                "available_tools": state.get("available_tools", []),
                "conversation_history": new_history,
            }

        _node.__name__ = f"exec_{node_name}"
        return _node

    # =========================================================================
    # v16 ノード: shogun_plan — 将軍がロードマップを作成
    # =========================================================================

    async def _shogun_plan_node(self, state: BushidanState) -> dict:
        """将軍 (Claude Sonnet) がユーザー要望をロードマップ JSON に分解。
        ステップごとに capability と担当ロールを割り当てる。
        """
        start = time.time()
        intent = state.get("intent_structured", {})
        complexity = state.get("complexity", "complex")
        message = state.get("message", "")

        try:
            from utils.client_registry import ClientRegistry
            client = ClientRegistry.get().get_client("shogun")
            if not client:
                raise RuntimeError("将軍クライアント未設定")

            notion_ctx = ""
            for chunk in state.get("notion_chunks", [])[:2]:
                t = chunk.get("title", "") if isinstance(chunk, dict) else getattr(chunk, "title", "")
                c = chunk.get("content", "") if isinstance(chunk, dict) else getattr(chunk, "content", "")
                if c:
                    notion_ctx += f"【{t}】{c[:300]}\n"

            system = (
                "あなたは将軍（Claude Sonnet）。ユーザーの要望を分析し、実行ロードマップを作成します。\n"
                "以下のJSON **のみ** を返してください。余分なテキスト・説明は不要です。\n\n"
                "{\n"
                '  "goal": "ユーザー目標の1文要約",\n'
                '  "steps": [\n'
                '    {\n'
                '      "id": 1,\n'
                '      "task": "具体的なタスク説明",\n'
                '      "capability": "analysis|rag|web_search|code|tools|japanese|image|quick_task",\n'
                '      "assigned_role": "gunshi|metsuke|gaiji|seppou|sanbo|yuhitsu|kengyo",\n'
                '      "can_parallel": false,\n'
                '      "status": "pending"\n'
                '    }\n'
                '  ],\n'
                '  "needs_audit": false\n'
                "}\n\n"
                "capability → assigned_role の対応:\n"
                "- analysis    → gunshi (Command A: 汎用処理・推論・分析)\n"
                "- quick_task  → metsuke (Mistral Small: 要約・整形・軽量タスク)\n"
                "- summary     → metsuke (Mistral Small: まとめ・整理)\n"
                "- rag         → gaiji (Command R: RAG・ドキュメント検索)\n"
                "- web_search  → seppou (Groq: 最新情報検索)\n"
                "- code        → sanbo (Gemini Flash: コーディング)\n"
                "- tools       → sanbo (Gemini Flash: git/ファイル操作)\n"
                "- japanese    → yuhitsu (Gemma4 Local: 日本語文章)\n"
                "- image       → kengyo (Gemini Flash Image: 画像分析)\n"
                "needs_audit: 最高難度・重大な意思決定・本番デプロイ関連の場合 true\n"
                "can_parallel: 前ステップの結果に依存しない独立したタスクの場合 true (並列実行される)\n"
            )
            if notion_ctx:
                system += f"\n【関連ナレッジ】\n{notion_ctx}"

            user_goal = intent.get("user_goal", message[:200])
            prompt = (
                f"ユーザーの要望: {message}\n\n"
                f"解釈した目標: {user_goal}\n"
                f"complexity: {complexity}\n"
                "上記に対するロードマップJSONを作成してください。"
            )

            raw = await client.generate(
                messages=[{"role": "user", "content": prompt}],
                system=system,
                max_tokens=800,
            )

            import re as _re, json as _json
            m = _re.search(r'\{.*\}', raw, _re.DOTALL)
            if m:
                roadmap = _json.loads(m.group())
                # steps の status を "pending" に初期化
                for s in roadmap.get("steps", []):
                    s["status"] = "pending"
                    s.setdefault("result", "")
                needs_audit = bool(roadmap.get("needs_audit", False)) or complexity == "strategic"
                logger.info("🗺️ [将軍] ロードマップ作成: %d steps audit=%s goal='%s'",
                            len(roadmap.get("steps", [])), needs_audit, roadmap.get("goal", "")[:40])

                # ── YAML 監査ログ開始 ─────────────────────────────────
                try:
                    from utils.audit_log import AuditLog
                    _tid = state.get("thread_id") or "default"
                    _alog = AuditLog.start(_tid, message, roadmap)
                    _alog.set_complexity(complexity)
                    self._audit_logs[_tid] = _alog
                except Exception as _e:
                    logger.debug("AuditLog.start スキップ: %s", _e)

                return {
                    "roadmap":       roadmap,
                    "roadmap_step":  0,
                    "roadmap_results": [],
                    "needs_audit":   needs_audit,
                    "routed_to":     "shogun_plan",
                    "agent_role":    "将軍",
                    "handled_by":    "shogun_plan",
                    "execution_time": time.time() - start,
                }
        except Exception as e:
            logger.warning("🗺️ [将軍] ロードマップ作成失敗 → 直接実行: %s", e)

        # フォールバック: シングルステップロードマップ
        return {
            "roadmap": {
                "goal": message[:100],
                "steps": [{"id": 1, "task": message, "capability": "analysis",
                           "assigned_role": "gunshi", "status": "pending", "result": ""}],
                "needs_audit": False,
            },
            "roadmap_step":  0,
            "roadmap_results": [],
            "needs_audit":   False,
            "routed_to":     "shogun_plan",
            "agent_role":    "将軍",
            "handled_by":    "shogun_plan",
            "execution_time": time.time() - start,
        }

    # =========================================================================
    # v16 ノード: execute_step — ロードマップステップを専門ロールで実行
    # =========================================================================

    # capability → role_key マッピング
    _CAP_TO_ROLE = {
        "analysis":   "gunshi",
        "quick_task": "metsuke",   # 軽量タスク → 目付(Mistral Small)
        "rag":        "gaiji",
        "web_search": "gaiji",     # Web検索 → 外事(tavily_search担当)
        "code":       "sanbo",
        "tools":      "sanbo",
        "japanese":   "yuhitsu",
        "image":      "kengyo",
        "summary":    "metsuke",   # 要約・整形 → 目付
    }

    async def _execute_step_node(self, state: BushidanState) -> dict:
        """現在のロードマップステップを適切なロールで実行する。
        can_parallel=True の連続ステップは asyncio.gather で並列実行。
        """
        roadmap    = state.get("roadmap", {})
        step_idx   = state.get("roadmap_step", 0)
        steps      = roadmap.get("steps", [])
        start      = time.time()

        if step_idx >= len(steps):
            return {"roadmap_step": step_idx, "execution_time": time.time() - start}

        # ── 並列バッチ検出 ────────────────────────────────────────────
        # 現在のステップが can_parallel=True なら連続する can_parallel ステップをまとめる
        batch_end = step_idx
        if steps[step_idx].get("can_parallel", False):
            for i in range(step_idx + 1, len(steps)):
                if steps[i].get("can_parallel", False):
                    batch_end = i
                else:
                    break
        batch_steps = steps[step_idx: batch_end + 1]
        is_parallel = len(batch_steps) > 1

        # ── 共通コンテキスト構築 ──────────────────────────────────────
        prev_results = state.get("roadmap_results", [])
        prev_ctx = ""
        if prev_results:
            prev_ctx = "\n".join(
                f"[Step {r.get('step_id', i+1)}] {r.get('summary', r.get('response', '')[:200])}"
                for i, r in enumerate(prev_results[-3:])
            )

        timeout = NODE_TIMEOUTS.get("execute_step", 120)

        async def _run_one_step(s: dict, idx: int):
            capability = s.get("capability", "analysis")
            assigned   = s.get("assigned_role", "")
            task_desc  = s.get("task", "")
            role_key   = assigned if assigned in self._roles else self._CAP_TO_ROLE.get(capability, "gunshi")
            role       = self._roles.get(role_key) or self._roles.get("gunshi")

            sub_state = dict(state)
            sub_state["message"] = task_desc
            sub_state["_step_task"] = task_desc
            if prev_ctx:
                sub_state["context_summary"] = (
                    f"ロードマップ目標: {roadmap.get('goal', '')}\n"
                    f"完了済みステップ:\n{prev_ctx}"
                )
            try:
                from core.mcp_sdk import MCPToolRegistry
                registry = MCPToolRegistry.get()
                lc_tools = registry.get_tools_for_role(role_key)
                if lc_tools:
                    sub_state["available_tools"] = [t.name for t in lc_tools]
                    sub_state["mcp_tools"] = lc_tools
            except Exception:
                pass

            try:
                res = await asyncio.wait_for(role.execute(sub_state), timeout=timeout)
            except asyncio.TimeoutError:
                res = type("R", (), {
                    "response": f"⏱️ ステップ{idx+1}タイムアウト",
                    "error": "timeout", "mcp_tools_used": [],
                    "agent_role": role_key, "handled_by": "execute_step",
                    "execution_time": timeout,
                })()
            return role_key, capability, task_desc, res, idx

        if is_parallel:
            logger.info("⚡ [execute_step] 並列実行 %d steps (%d〜%d)",
                        len(batch_steps), step_idx + 1, batch_end + 1)
            raw_results = await asyncio.gather(*[
                _run_one_step(s, step_idx + i)
                for i, s in enumerate(batch_steps)
            ], return_exceptions=True)
            # 例外が返ったステップはエラー結果に変換
            batch_results = []
            for ri, raw in enumerate(raw_results):
                if isinstance(raw, BaseException):
                    logger.error("⚠️ 並列ステップ %d 例外: %s", step_idx + ri + 1, raw)
                    s = batch_steps[ri]
                    from roles.base import RoleResult
                    err_result = RoleResult(
                        response=f"❌ ステップ実行エラー: {raw}",
                        agent_role=s.get("capability", "unknown"),
                        handled_by="execute_step",
                        error=str(raw), status="failed",
                    )
                    batch_results.append((s.get("capability", "unknown"), s.get("capability", ""), s.get("task", ""), err_result, step_idx + ri))
                else:
                    batch_results.append(raw)
        else:
            s = batch_steps[0]
            logger.info("▶️  [execute_step] step %d/%d: capability=%s task='%s'",
                        step_idx + 1, len(steps), s.get("capability"), s.get("task", "")[:40])
            batch_results = [await _run_one_step(s, step_idx)]

        # ── 結果集約 ────────────────────────────────────────────────────
        step_results_list = []
        updated_steps = list(steps)
        all_mcp_used = []

        _alog = self._audit_logs.get(state.get("thread_id") or "default")
        for role_key, capability, task_desc, result, abs_idx in batch_results:
            step_results_list.append({
                "step_id":    abs_idx + 1,
                "capability": capability,
                "role":       role_key,
                "task":       task_desc,
                "response":   result.response,
                "summary":    result.response[:200] if result.response else "",
                "error":      result.error,
            })
            updated_steps[abs_idx] = dict(steps[abs_idx])
            updated_steps[abs_idx]["status"] = "done" if not result.error else "error"
            updated_steps[abs_idx]["result"] = result.response[:300] if result.response else ""
            all_mcp_used.extend(getattr(result, "mcp_tools_used", []))
            # ── 監査ログにステップ結果を記録 ──────────────────────────
            if _alog:
                try:
                    _alog.add_step_result(
                        step_id=abs_idx + 1,
                        role=role_key,
                        capability=capability,
                        task=task_desc,
                        result=result.response or "",
                        execution_time=getattr(result, "execution_time", 0.0),
                        error=result.error or "",
                    )
                except Exception:
                    pass

        updated_roadmap = dict(roadmap)
        updated_roadmap["steps"] = updated_steps
        new_step_idx = batch_end + 1
        is_last = (new_step_idx >= len(steps))

        # 最終バッチは全回答を結合して response に
        if is_last:
            if is_parallel:
                new_response = "\n\n---\n\n".join(
                    r["response"] for r in step_results_list if r.get("response")
                )
            else:
                new_response = batch_results[0][3].response
        else:
            new_response = state.get("response", "")

        last_result = batch_results[-1][3]
        return {
            "roadmap":          updated_roadmap,
            "roadmap_step":     new_step_idx,
            "roadmap_results":  step_results_list,
            "response":         new_response,
            "agent_role":       last_result.agent_role,
            "handled_by":       "execute_step",
            "routed_to":        f"execute_step[{step_idx+1}~{batch_end+1}]",
            "execution_time":   time.time() - start,
            "mcp_tools_used":   all_mcp_used,
            "error":            last_result.error if is_last else None,
            "conversation_history": [
                {"role": "user",      "content": batch_steps[-1].get("task", "")},
                {"role": "assistant", "content": new_response or ""},
            ],
        }

    def _step_decision(self, state: BushidanState) -> str:
        """ロードマップステップの次遷移を決定。"""
        roadmap   = state.get("roadmap", {})
        steps     = roadmap.get("steps", [])
        step_idx  = state.get("roadmap_step", 0)

        if step_idx < len(steps):
            return "next_step"

        # 全ステップ完了
        if state.get("needs_audit", False):
            logger.info("🔍 全ステップ完了 → 大元帥監査へ")
            return "audit"

        # コードが含まれるなら sandbox_verify へ
        response = state.get("response", "") or ""
        if "```" in response:
            return "verify"

        return "done"

    # =========================================================================
    # v16 ノード: daigensui_audit — 大元帥 (Opus) による最終監査
    # =========================================================================

    async def _daigensui_audit_node(self, state: BushidanState) -> dict:
        """大元帥 (Claude Opus) がロードマップ全体を監査し、最終判断を下す。"""
        start = time.time()
        try:
            from utils.client_registry import ClientRegistry
            client = ClientRegistry.get().get_client("daigensui")
            if not client:
                raise RuntimeError("大元帥クライアント未設定")

            roadmap  = state.get("roadmap", {})
            results  = state.get("roadmap_results", [])
            message  = state.get("message", "")

            steps_summary = "\n".join(
                f"Step {r['step_id']} [{r['capability']}→{r['role']}]: {r['summary']}"
                for r in results
            )

            system = (
                "あなたは大元帥（Claude Opus）。将軍が立案・実行したロードマップを監査します。\n"
                "以下の観点でレビューし、最終的な回答・改善点・承認/否認を日本語で述べてください:\n"
                "1. 目標達成度: ユーザーの要望が満たされているか\n"
                "2. 品質評価: 各ステップの実行品質\n"
                "3. リスク確認: 見落とし・問題点\n"
                "4. 最終回答: ユーザーへの統合された回答"
            )
            prompt = (
                f"【ユーザー要望】\n{message}\n\n"
                f"【ロードマップ目標】\n{roadmap.get('goal', '')}\n\n"
                f"【実行済みステップ】\n{steps_summary}\n\n"
                f"【最終ステップの回答】\n{state.get('response', '')[:1000]}"
            )

            response = await client.generate(
                messages=[{"role": "user", "content": prompt}],
                system=system,
                max_tokens=2000,
            )
            elapsed = time.time() - start
            logger.info("⚔️ [大元帥監査] 完了 %.1fs", elapsed)

            # ── 監査ログに監査結果を記録 ──────────────────────────────
            _alog = self._audit_logs.get(state.get("thread_id") or "default")
            if _alog:
                try:
                    _alog.add_audit(
                        verdict="approved",
                        comments=response[:500],
                        execution_time=elapsed,
                    )
                except Exception:
                    pass

            return {
                "response":       response,
                "agent_role":     "大元帥",
                "handled_by":     "daigensui_audit",
                "execution_time": elapsed,
                "routed_to":      "daigensui_audit",
                "mcp_tools_used": [],
            }
        except Exception as e:
            logger.warning("大元帥監査失敗 (スキップ): %s", e)
            return {
                "agent_role":     "大元帥",
                "handled_by":     "daigensui_audit",
                "execution_time": time.time() - start,
                "error":          str(e),
            }

    # =========================================================================
    # ノード: parallel_groq — 並列斥候 (複数サブクエリを Groq で同時処理)
    # =========================================================================

    async def _parallel_groq_node(self, state: BushidanState) -> dict:
        """
        メッセージを「?」で分割し、各サブクエリを斥候(Groq)で並列実行。
        全結果をマージして返す。
        """
        message = state.get("message", "")
        start = time.time()

        # サブクエリ分割: 日本語「？」と英語「?」で分割
        import re as _re
        raw_parts = _re.split(r"[?？]\s*", message)
        sub_queries = [p.strip() for p in raw_parts if p.strip()]

        # 分割されなかった場合は通常 groq_qa にフォールバック
        if len(sub_queries) <= 1:
            exec_fn = self._exec_node("seppou", "groq_qa")
            result = await exec_fn(state)
            result["routed_to"] = "parallel_groq"
            return result

        logger.info("⚡⚡ parallel_groq: %d サブクエリを並列実行", len(sub_queries))

        role = self._roles.get("seppou")
        if not role:
            return {
                "response": "⚠️ 斥候ロール未初期化",
                "handled_by": "parallel_groq", "agent_role": "斥候",
                "execution_time": 0.0, "error": "role not loaded",
                "routed_to": "parallel_groq", "mcp_tools_used": [],
                "sub_queries": sub_queries, "sub_responses": [],
            }

        async def _run_sub(q: str) -> str:
            sub_state = dict(state)
            sub_state["message"] = q + "?"
            try:
                res = await asyncio.wait_for(role.execute(sub_state), timeout=30)
                return res.response
            except Exception as e:
                return f"❌ {q[:30]}: {e}"

        tasks = [_run_sub(q) for q in sub_queries[:4]]  # 最大4並列
        sub_responses = list(await asyncio.gather(*tasks, return_exceptions=True))
        sub_responses = [
            r if isinstance(r, str) else f"❌ エラー: {r}"
            for r in sub_responses
        ]

        # マージレスポンス
        lines = []
        for q, ans in zip(sub_queries, sub_responses):
            lines.append(f"**Q: {q}?**\n{ans}")
        merged = "\n\n".join(lines)

        elapsed = time.time() - start
        logger.info("✅ parallel_groq: 完了 %.1fs (%d サブクエリ)", elapsed, len(sub_queries))

        new_history = [
            {"role": "user",      "content": message},
            {"role": "assistant", "content": merged},
        ]

        if merged:
            asyncio.create_task(
                _post_to_matrix("seppou", message, merged),
                name="matrix_parallel_groq",
            )

        return {
            "response":       merged,
            "handled_by":     "parallel_groq",
            "agent_role":     "斥候 (並列)",
            "execution_time": elapsed,
            "error":          None,
            "mcp_tools_used": [],
            "requires_followup": False,
            "routed_to":      "parallel_groq",
            "sub_queries":    sub_queries,
            "sub_responses":  sub_responses,
            "conversation_history": new_history,
        }

    # =========================================================================
    # ノード: sandbox_verify — 生成コードを実証検証
    # =========================================================================

    async def _sandbox_verify_node(self, state: BushidanState) -> dict:
        """
        直前の実行ノードが生成したコードブロックを抽出して実行検証。
        結果をレスポンスに付記する。エラーは無視して処理を続行。
        """
        response = state.get("response", "") or ""

        # コードブロックがなければ即スキップ
        if "```" not in response:
            return {"code_verified": False, "code_verify_result": "skipped"}

        try:
            from core.code_verifier import verify_response, append_verify_note
            verify_result = await asyncio.wait_for(
                verify_response(response, max_blocks=2), timeout=25
            )
            updated_response = append_verify_note(response, verify_result)
            logger.info("[sandbox_verify] 検証結果: %s...", verify_result[:40])
            return {
                "response":          updated_response,
                "code_verified":     True,
                "code_verify_result": verify_result,
            }
        except asyncio.TimeoutError:
            logger.warning("[sandbox_verify] タイムアウト — スキップ")
            return {"response": response, "code_verified": False, "code_verify_result": "timeout"}
        except Exception as e:
            logger.debug("[sandbox_verify] スキップ: %s", e)
            return {"response": response, "code_verified": False, "code_verify_result": f"error: {e}"}

    # =========================================================================
    # ノード: check_followup — 3分岐 (human / loop / done)
    # =========================================================================

    async def _check_followup(self, state: BushidanState) -> dict:
        iteration = state.get("iteration", 0) + 1
        max_iter  = state.get("max_iterations", 3)
        followup  = state.get("requires_followup", False)
        updates: dict = {"iteration": iteration}

        # ── 会話履歴自動要約 (20往復超でGroq/Haiku で圧縮) ─────────────
        history = state.get("conversation_history", [])
        if len(history) > 20:
            try:
                from utils.client_registry import ClientRegistry
                summary_client = ClientRegistry.get().get_client("metsuke") or \
                                 ClientRegistry.get().get_client("seppou")
                if summary_client:
                    old_msgs = history[:-10]  # 古い部分を要約
                    joined = "\n".join(
                        f"{m['role']}: {str(m.get('content',''))[:300]}"
                        for m in old_msgs
                    )
                    summary_resp = await summary_client.generate(
                        messages=[{"role": "user", "content":
                            f"以下の会話を3〜5行の要点に圧縮してください:\n{joined[:3000]}"}],
                        system="会話要約アシスタント。箇条書きで簡潔に。",
                        max_tokens=300,
                    )
                    updates["context_summary"] = summary_resp
                    updates["conversation_history"] = history[-10:]  # 最新10件のみ残す
                    logger.info("📝 会話履歴要約: %d→%d turns", len(history), 10)
            except Exception as e:
                logger.debug("会話要約スキップ: %s", e)

        if followup and iteration < max_iter:
            logger.info("🔄 自律ループ継続 (%d/%d)", iteration, max_iter)
            updates["requires_followup"] = True
            return updates

        if followup and iteration >= max_iter:
            logger.info("⚠️ 最大反復回数到達 (%d) — 強制終了", max_iter)

        updates["requires_followup"] = False
        return updates

    def _followup_decision(self, state: BushidanState) -> Literal["human", "loop", "done"]:
        # HITL 判定: awaiting_human_input フラグが立っている場合
        if state.get("awaiting_human_input", False):
            return "human"
        if state.get("requires_followup", False):
            return "loop"
        return "done"

    # =========================================================================
    # ノード: human_interrupt — HITL 中断
    # =========================================================================

    async def _human_interrupt(self, state: BushidanState) -> dict:
        """
        Human-in-the-loop: 人間の応答を待つ。
        LangGraph interrupt() でグラフを一時停止し、/resume エンドポイントから
        Command(resume=...) で再開する。
        """
        question = state.get("human_question", "")
        logger.info("🙋 HITL: 人間の入力を待機中 — %s", question[:60])
        response = interrupt({"question": question})
        return {
            "human_response": response,
            "dialog_status": "active",
            "awaiting_human_input": False,
        }

    # =========================================================================
    # ノード: notion_store — Notion に結果を非同期保存
    # =========================================================================

    async def _notion_store(self, state: BushidanState) -> dict:
        if not state.get("should_save", True):
            return {"notion_page_id": None}
        try:
            from integrations.notion.storage import save_task_result_bg
            asyncio.create_task(
                save_task_result_bg(dict(state)),
                name=f"notion_store_{state.get('thread_id', '')[:8]}",
            )
            return {"notion_page_id": "pending"}
        except Exception as e:
            logger.debug("[notion_store] スキップ: %s", e)
            return {"notion_page_id": None}

    # =========================================================================
    # Public API
    # =========================================================================

    async def process_message(
        self,
        message: str,
        thread_id: str = "",
        channel_id: str = "",
        user_id: str = "",
        source: str = "api",
        forced_role: Optional[str] = None,
        attachments: Optional[list] = None,
        available_tools: Optional[list] = None,
    ) -> dict:
        """メッセージを処理して結果を返す (v16)"""
        if not self._compiled:
            await self.initialize()

        # ── 応答キャッシュチェック (Q&A 短文・強制ロール無し・添付無し) ──
        _cache_key = None
        _can_cache = (
            not forced_role
            and not attachments
            and len(message) <= 200
        )
        if _can_cache:
            _h = hashlib.md5(f"{message}|{thread_id}".encode()).hexdigest()
            _cache_key = _h
            _entry = self._RESP_CACHE.get(_h)
            if _entry and (time.time() - _entry[1]) < self._RESP_CACHE_TTL:
                logger.debug("⚡ 応答キャッシュヒット: %s", message[:40])
                return _entry[0]

        initial: BushidanState = {
            "thread_id":   thread_id or "default",
            "channel_id":  channel_id,
            "user_id":     user_id,
            "source":      source,
            "message":     message,
            "attachments": attachments or [],
            "conversation_history": [],
            "notion_chunks": [],
            "complexity":          "medium",
            "is_multi_step":       False,
            "is_action_task":      False,
            "is_simple_qa":        False,
            "is_japanese_priority": False,
            "is_confidential":     False,
            "forced_role": forced_role,
            "routed_to":   None,
            "available_tools": available_tools or [],
            "tool_schemas":    {},
            "response":       None,
            "handled_by":     None,
            "agent_role":     None,
            "execution_time": 0.0,
            "error":          None,
            "mcp_tools_used": [],
            "requires_followup": False,
            "iteration":         0,
            "max_iterations":    3,
            "should_save":    True,
            "notion_page_id": None,
            # v15
            "dialog_status":       "active",
            "awaiting_human_input": False,
            "human_question":      "",
            "human_response":      "",
            # v15
            "context_summary":     "",
            # v15: コード検証
            "code_verified":       False,
            "code_verify_result":  "",
            # v15: 並列 Groq
            "sub_queries":         [],
            "sub_responses":       [],
            # v16: 受付構造化インテント
            "intent_structured":   {},
            # v16: 将軍ロードマップ
            "roadmap":             {},
            "roadmap_step":        0,
            "roadmap_results":     [],
            "needs_audit":         False,
            # v18: ステップ実行コンテキスト
            "_step_task":          "",
        }

        # ── PG選択的チェックポイント: 短文Q&A は MemorySaver グラフを使用 ──
        _use_fast = (
            _can_cache                            # 短文・添付なし・強制ロール無し
            and self._compiled_fast is not None
        )
        _graph = self._compiled_fast if _use_fast else self._compiled

        config = {"configurable": {"thread_id": thread_id or "default"}}
        logger.info("🔗 LangGraph v16: 処理開始 thread=%s graph=%s '%s'...",
                     thread_id[:8] if thread_id else "new",
                     "fast" if _use_fast else "full",
                     message[:60])
        start = time.time()

        try:
            # ── タイムアウト: ゆったり設定（複雑なタスク対応、デフォルト 180秒）
            timeout = 180 if not _use_fast else 30
            final = await asyncio.wait_for(_graph.ainvoke(initial, config=config), timeout=timeout)
            total = time.time() - start
            logger.info(
                "✅ LangGraph v16: 完了 route=%s agent=%s time=%.2fs",
                final.get("routed_to"), final.get("agent_role"), total
            )
            result_dict = {
                "status":         "completed" if not final.get("error") else "failed",
                "response":       final.get("response", ""),
                "result":         final.get("response", ""),
                "handled_by":     final.get("handled_by", "unknown"),
                "agent_role":     final.get("agent_role", ""),
                "route":          final.get("routed_to", ""),
                "complexity":     final.get("complexity", ""),
                "mcp_tools_used": final.get("mcp_tools_used", []),
                "notion_page_id": final.get("notion_page_id"),
                "dialog_status":  final.get("dialog_status", "completed"),
                "human_question": final.get("human_question", ""),
                "execution_time": total,
                "langgraph":      True,
                "version":        "16",
            }
            # キャッシュ保存 (Q&A完了時のみ)
            if (
                _cache_key
                and result_dict["status"] == "completed"
                and not final.get("error")
                and not final.get("is_action_task")
                and not final.get("requires_followup")
            ):
                self._RESP_CACHE[_cache_key] = (result_dict, time.time())
                # キャッシュが膨らまないよう古いエントリを定期削除
                if len(self._RESP_CACHE) > 500:
                    now = time.time()
                    self._RESP_CACHE = {
                        k: v for k, v in self._RESP_CACHE.items()
                        if now - v[1] < self._RESP_CACHE_TTL
                    }

            # ── YAML 監査ログ完了 + スキルトラッカー記録 ─────────────
            _tid_key = thread_id or "default"
            _alog = self._audit_logs.pop(_tid_key, None)
            if _alog:
                try:
                    _alog.finish(final.get("response", ""))
                except Exception:
                    pass
            # スキルトラッカー (バックグラウンド — 応答には影響しない)
            asyncio.create_task(
                _skill_observe(
                    thread_id=_tid_key,
                    message=message,
                    handled_by=result_dict.get("handled_by", ""),
                    execution_time=total,
                ),
                name="skill_observe",
            )

            return result_dict
        except asyncio.TimeoutError:
            total = time.time() - start
            logger.error("⏱️ LangGraph v16 タイムアウト (%.1fs): thread=%s message='%s'",
                         total, thread_id[:8], message[:40])
            # 監査ログを異常終了で閉じる
            _alog = self._audit_logs.pop(thread_id or "default", None)
            if _alog:
                try:
                    _alog.finish(f"❌ タイムアウト ({total:.1f}秒)")
                except Exception:
                    pass
            return {
                "status":         "failed",
                "error":          f"処理タイムアウト ({total:.1f}秒)",
                "response":       f"❌ 処理がタイムアウトしました（{total:.1f}秒）。サーバー側で長時間処理が必要です。",
                "result":         f"❌ 処理タイムアウト",
                "handled_by":     "langgraph_router",
                "execution_time": total,
            }
        except Exception as e:
            logger.exception("❌ LangGraph v16 処理失敗: %s", e)
            # 監査ログを異常終了で閉じる
            _alog = self._audit_logs.pop(thread_id or "default", None)
            if _alog:
                try:
                    _alog.finish(f"❌ 処理失敗: {e}")
                except Exception:
                    pass
            return {
                "status":         "failed",
                "error":          str(e),
                "response":       f"❌ 処理失敗: {e}",
                "result":         f"❌ 処理失敗: {e}",
                "handled_by":     "langgraph_router",
                "execution_time": time.time() - start,
            }

    async def stream_message(
        self,
        message: str,
        thread_id: str = "",
        source: str = "api",
        forced_role: Optional[str] = None,
        attachments: Optional[list] = None,
    ):
        """
        メッセージをストリーミング処理。
        LangGraph astream_events で LLM トークンを yield する非同期ジェネレータ。

        Yields:
            dict: {"type": "token", "content": str}
                  {"type": "route", "route": str, "agent_role": str}
                  {"type": "done", "execution_time": float}
        """
        if not self._compiled:
            await self.initialize()

        initial: BushidanState = {
            "thread_id":   thread_id or "default",
            "channel_id":  "",
            "user_id":     "",
            "source":      source,
            "message":     message,
            "attachments": attachments or [],
            "conversation_history": [],
            "notion_chunks": [],
            "complexity":          "medium",
            "is_multi_step":       False,
            "is_action_task":      False,
            "is_simple_qa":        False,
            "is_japanese_priority": False,
            "is_confidential":     False,
            "forced_role": forced_role,
            "routed_to":   None,
            "available_tools": [],
            "tool_schemas":    {},
            "response":       None,
            "handled_by":     None,
            "agent_role":     None,
            "execution_time": 0.0,
            "error":          None,
            "mcp_tools_used": [],
            "requires_followup": False,
            "iteration":         0,
            "max_iterations":    3,
            "should_save":    True,
            "notion_page_id": None,
            "dialog_status":       "active",
            "awaiting_human_input": False,
            "human_question":      "",
            "human_response":      "",
            "context_summary":     "",
            "code_verified":       False,
            "code_verify_result":  "",
            "sub_queries":         [],
            "sub_responses":       [],
            "intent_structured":   {},
            "roadmap":             {},
            "roadmap_step":        0,
            "roadmap_results":     [],
            "needs_audit":         False,
            "_step_task":          "",
        }

        config = {"configurable": {"thread_id": thread_id or "default"}}
        start = time.time()
        route_sent = False
        roadmap_emitted = False
        last_roadmap_step = -1
        # ── ストリーミングタイムアウト (秒): ゆったり設定、デフォルト 240 秒 ──
        stream_timeout = 240

        try:
            async def _stream_with_timeout():
                try:
                    async for event in asyncio.wait_for(
                        self._compiled.astream_events(initial, config=config, version="v2"),
                        timeout=stream_timeout
                    ):
                        yield event
                except asyncio.TimeoutError:
                    logger.error("⏱️ stream_message タイムアウト (%.1fs): thread=%s",
                                 stream_timeout, thread_id[:8])
                    yield {
                        "event": "timeout",
                        "type": "error",
                        "message": f"ストリーミング処理がタイムアウトしました（{stream_timeout}秒）",
                    }

            async for event in _stream_with_timeout():
                event_type = event.get("event", "")
                name = event.get("name", "")

                # ── タイムアウトエラーを処理 ──
                if event_type == "timeout":
                    yield {"type": "error", "message": event.get("message")}
                    break

                # ロードマップ作成完了を通知 (on_chain_end for shogun_plan)
                if event_type == "on_chain_end" and name == "shogun_plan" and not roadmap_emitted:
                    output = event.get("data", {}).get("output") or {}
                    roadmap = output.get("roadmap")
                    if roadmap:
                        yield {"type": "roadmap_created", "roadmap": roadmap}
                        roadmap_emitted = True

                # ステップ実行開始を通知 (on_chain_start for execute_step)
                if event_type == "on_chain_start" and name == "execute_step":
                    inp = event.get("data", {}).get("input") or {}
                    roadmap = inp.get("roadmap") or {}
                    step_idx = inp.get("roadmap_step", 0)
                    steps = roadmap.get("steps", [])
                    if step_idx < len(steps):
                        s = steps[step_idx]
                        yield {
                            "type": "step_start",
                            "step_id": s.get("id"),
                            "task": s.get("task"),
                            "assigned_role": s.get("assigned_role"),
                            "capability": s.get("capability"),
                            "total_steps": len(steps),
                        }

                # ステップ実行完了を通知 (on_chain_end for execute_step)
                if event_type == "on_chain_end" and name == "execute_step":
                    output = event.get("data", {}).get("output") or {}
                    roadmap_results = output.get("roadmap_results", [])
                    roadmap = output.get("roadmap")
                    roadmap_step = output.get("roadmap_step", 0)
                    if roadmap_results and roadmap_step > last_roadmap_step:
                        last = roadmap_results[-1]
                        yield {
                            "type": "step_done",
                            "step_id": last.get("step_id"),
                            "role": last.get("role"),
                            "summary": last.get("summary", ""),
                            "roadmap": roadmap,
                            "roadmap_step": roadmap_step,
                        }
                        last_roadmap_step = roadmap_step

                # 監査開始を通知 (on_chain_start for daigensui_audit)
                if event_type == "on_chain_start" and name == "daigensui_audit":
                    yield {"type": "audit_start"}

                # ルーティング確定を通知
                if event_type == "on_chain_start" and not route_sent:
                    if name not in ("LangGraph", "analyze_intent", "notion_index", "execute_step", "shogun_plan"):
                        yield {"type": "route", "route": name}
                        route_sent = True

                # LLMトークンをストリーミング
                if event_type == "on_chat_model_stream":
                    chunk = event.get("data", {}).get("chunk")
                    if chunk:
                        content = ""
                        if hasattr(chunk, "content"):
                            c = chunk.content
                            if isinstance(c, str):
                                content = c
                            elif isinstance(c, list) and c:
                                content = c[0].get("text", "") if isinstance(c[0], dict) else str(c[0])
                        elif isinstance(chunk, dict):
                            content = chunk.get("content", "")
                        if content:
                            yield {"type": "token", "content": content}

        except Exception as e:
            logger.error("stream_message 失敗: %s", e)
            yield {"type": "error", "message": str(e)}
        finally:
            yield {"type": "done", "execution_time": time.time() - start}

    async def resume(self, thread_id: str, human_response: str) -> dict:
        """
        HITL Go サイン: human_interrupt で停止中のグラフを再開する。

        Args:
            thread_id:      停止中のスレッド ID
            human_response: ユーザーの応答テキスト

        Returns:
            process_message と同じ形式の dict
        """
        if not self._compiled:
            await self.initialize()

        config = {"configurable": {"thread_id": thread_id}}
        logger.info("▶️  HITL resume: thread=%s response='%s'...", thread_id[:8], human_response[:40])
        start = time.time()

        try:
            # ── HITL resume タイムアウト: 150秒 ──
            final = await asyncio.wait_for(
                self._compiled.ainvoke(
                    Command(resume=human_response),
                    config=config,
                ),
                timeout=150
            )
            total = time.time() - start
            logger.info("✅ HITL resume 完了: thread=%s time=%.2fs", thread_id[:8], total)
            return {
                "status":        "completed" if not final.get("error") else "failed",
                "response":      final.get("response", ""),
                "result":        final.get("response", ""),
                "handled_by":    final.get("handled_by", "unknown"),
                "agent_role":    final.get("agent_role", ""),
                "route":         final.get("routed_to", ""),
                "dialog_status": final.get("dialog_status", "completed"),
                "execution_time": total,
                "langgraph":     True,
                "version":       "14",
            }
        except asyncio.TimeoutError:
            total = time.time() - start
            logger.error("⏱️ HITL resume タイムアウト (%.1fs): thread=%s", total, thread_id[:8])
            return {
                "status":        "failed",
                "error":         f"resume タイムアウト ({total:.1f}秒)",
                "response":      f"❌ resume がタイムアウトしました（{total:.1f}秒）",
                "execution_time": total,
            }
        except Exception as e:
            logger.exception("❌ HITL resume 失敗: %s", e)
            return {
                "status":        "failed",
                "error":         str(e),
                "response":      f"❌ resume 失敗: {e}",
                "execution_time": time.time() - start,
            }

    # 後方互換
    async def process_task(self, content: str, context: dict = None,
                           priority: int = 1, source: str = "api") -> dict:
        ctx = context or {}
        return await self.process_message(
            message=content,
            thread_id=ctx.get("root_id", ctx.get("thread_id", "")),
            channel_id=ctx.get("channel_id", ""),
            user_id=ctx.get("user_id", ""),
            source=ctx.get("source", source),
            forced_role=ctx.get("forced_route") or ctx.get("forced_agent"),
        )
