"""
core/router/router.py — LangGraph Router v16 (Mixin 統合版)

6つの Mixin を多重継承で組み合わせた薄いオーケストレーター。
外部インターフェース (initialize / process_message / stream_message / resume / process_task)
は旧 core/langgraph_router.py と完全互換。
"""
import asyncio
import os
from typing import Any, Optional

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from core.state import BushidanState
from core.router.constants import load_roles, refresh_notion_index_bg, fire
from core.router.mixins import (
    CheckpointerMixin,
    IntentMixin,
    RoutingMixin,
    NodesMixin,
    PostprocessMixin,
    MessagingMixin,
)
from utils.logger import get_logger

logger = get_logger(__name__)


class LangGraphRouter(
    CheckpointerMixin,
    IntentMixin,
    RoutingMixin,
    NodesMixin,
    PostprocessMixin,
    MessagingMixin,
):
    """LangGraph StateGraph v16 — タイムアウト + ヘルスチェック + HITL + Batch モード"""

    def __init__(self, orchestrator: Any = None):
        self.orchestrator = orchestrator
        self._roles: dict = {}
        self._compiled = None
        self._compiled_fast = None
        self._checkpointer = None
        self._pool = None
        self._pg_status: str = "initializing"
        self._pg_error: str = ""
        self._pg_reconnect_task: Optional[asyncio.Task] = None
        self._memory_fallback: Optional[MemorySaver] = None
        self._audit_logs: dict = {}
        # 応答キャッシュ: クラス変数を共有すると複数インスタンスで競合するためインスタンスに分離
        self._RESP_CACHE: dict = {}
        self._RESP_CACHE_TTL: float = 300.0
        self._cache_lock: asyncio.Lock = asyncio.Lock()

    async def initialize(self) -> None:
        logger.info("🔗 LangGraph Router v16 初期化開始")
        try:
            self._checkpointer = await self._init_checkpointer()

            try:
                from core.mcp_sdk import MCPToolRegistry
                registry = MCPToolRegistry.get()
                await registry.initialize()
                logger.info("🔧 MCP tools: %s", registry.available_tools[:10] or "none")
            except Exception as e:
                logger.warning("⚠️ MCPToolRegistry 初期化スキップ: %s", e)

            try:
                from utils.skill_tracker import ensure_schema as _skill_schema
                await _skill_schema()
            except Exception as _e:
                logger.warning("⚠️ SkillTracker schema skip: %s", _e)

            self._roles = load_roles()
            self._compiled = self._build_graph().compile(checkpointer=self._checkpointer)
            self._compiled_fast = self._build_graph().compile(checkpointer=MemorySaver())

            self._health_task = asyncio.create_task(
                self._background_health_check(), name="health_check_bg"
            )
            if self._pg_status == "disconnected":
                self._pg_reconnect_task = asyncio.create_task(
                    self._background_pg_reconnect(), name="pg_reconnect_bg"
                )

            fire(refresh_notion_index_bg(), name="notion_index_init")
            logger.info("✅ LangGraph Router v16 初期化完了 (%s)", type(self._checkpointer).__name__)
        except Exception as e:
            import traceback
            logger.error("❌ LangGraph Router 初期化エラー: %s\n%s", e, traceback.format_exc())
            raise

    def _build_graph(self) -> StateGraph:
        graph = StateGraph(BushidanState)

        graph.add_node("analyze_intent",  self._analyze_intent)
        graph.add_node("notion_index",    self._notion_index_node)

        graph.add_node("groq_qa",         self._exec_node("seppou",    "groq_qa"))
        graph.add_node("parallel_groq",   self._parallel_groq_node)
        graph.add_node("gunshi_haiku",    self._exec_node("gunshi",    "gunshi_haiku"))
        graph.add_node("metsuke_proc",    self._exec_node("metsuke",   "metsuke_proc"))
        graph.add_node("gaiji_rag",       self._exec_node("gaiji",     "gaiji_rag"))
        graph.add_node("sanbo_mcp",       self._exec_node("sanbo",     "sanbo_mcp"))
        graph.add_node("yuhitsu_jp",      self._exec_node("yuhitsu",   "yuhitsu_jp"))
        graph.add_node("uketuke_default", self._exec_node("uketuke",   "uketuke_default"))
        graph.add_node("onmitsu_local",   self._exec_node("onmitsu",   "onmitsu_local"))
        graph.add_node("kengyo_vision",   self._exec_node("kengyo",    "kengyo_vision"))

        graph.add_node("shogun_plan",     self._shogun_plan_node)
        graph.add_node("execute_step",    self._execute_step_node)
        graph.add_node("daigensui_audit", self._daigensui_audit_node)
        graph.add_node("batch_parallel",  self._batch_parallel_node)

        graph.add_node("sandbox_verify",  self._sandbox_verify_node)
        graph.add_node("check_followup",  self._check_followup)
        graph.add_node("human_interrupt", self._human_interrupt)
        graph.add_node("notion_store",    self._notion_store)

        graph.set_entry_point("analyze_intent")
        graph.add_edge("analyze_intent", "notion_index")

        graph.add_conditional_edges(
            "notion_index",
            self._route_decision,
            {
                "groq_qa":          "groq_qa",
                "parallel_groq":    "parallel_groq",
                "gunshi_haiku":     "gunshi_haiku",
                "gunshi_pdca":      "gunshi_haiku",   # 後方互換
                "gaiji_rag":        "gaiji_rag",
                "sanbo_mcp":        "sanbo_mcp",
                "yuhitsu_jp":       "yuhitsu_jp",
                "uketuke_default":  "uketuke_default",
                "onmitsu_local":    "onmitsu_local",
                "kengyo_vision":    "kengyo_vision",
                "shogun_plan":      "shogun_plan",
                "batch_parallel":   "batch_parallel",  # BATCH: 全ステップ一括並列
            },
        )

        graph.add_edge("batch_parallel", "check_followup")

        for node in ("gunshi_haiku", "metsuke_proc", "gaiji_rag", "sanbo_mcp",
                     "yuhitsu_jp", "uketuke_default", "onmitsu_local",
                     "kengyo_vision", "groq_qa", "parallel_groq"):
            graph.add_edge(node, "check_followup")

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
