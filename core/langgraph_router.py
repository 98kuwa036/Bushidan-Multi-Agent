"""
武士団 Multi-Agent System v14 - LangGraph Router

v14 主な変更点:
  - ノードタイムアウト: _exec_node() に asyncio.wait_for ラップ
  - ヘルスチェック統合: _route_decision() でフォールバック
  - Human-in-the-loop: human_interrupt ノード + 3分岐 followup
  - context_summary: リレー間コンテキスト要約
  - MCP ツール注入: _exec_node() で state に tools 注入

StateGraph v14:
  [START]
    ↓
  [analyze_intent]    タスク複雑度・種別分析
    ↓
  [notion_retrieve]   Notion RAG 検索
    ↓
  [route_decision]    ルーティング判断 (10役職 + ヘルスチェック)
    ↓
  ┌─────────────────────────────────────────────────────┐
  │ groq_qa / gunshi_pdca / gaiji_rag / taisho_mcp      │
  │ yuhitsu_jp / karo_default / onmitsu_local           │
  │ shogun_direct / daigensui_direct / kengyo_vision    │
  └─────────────────────────────────────────────────────┘
    ↓
  [check_followup]    3分岐: human / loop / done
    ├─ "human" → [human_interrupt]  (HITL 中断)
    ├─ "loop"  → [notion_retrieve]  (自律ループ)
    └─ "done"  → [notion_store]     (完了)
"""

import asyncio
import time
from typing import Any, Literal, Optional

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from core.state import BushidanState
from utils.logger import get_logger

logger = get_logger(__name__)

# ── ノード別タイムアウト (秒) ────────────────────────────────────────────────
NODE_TIMEOUTS = {
    "groq_qa":          30,
    "karo_default":     60,
    "gaiji_rag":        60,
    "taisho_mcp":       60,
    "kengyo_vision":    60,
    "gunshi_pdca":      90,
    "yuhitsu_jp":       90,
    "onmitsu_local":   120,
    # Claude CLI (60s) + API フォールバック (140s) = 200s
    "shogun_direct":   200,
    # Claude CLI (60s) + API フォールバック (220s) = 280s
    "daigensui_direct":280,
}

# ── フォールバックマップ (障害時の代替ルート) ────────────────────────────────
_FALLBACK_MAP = {
    "groq_qa":          "karo_default",
    "gunshi_pdca":      "shogun_direct",
    "gaiji_rag":        "karo_default",
    "taisho_mcp":       "shogun_direct",
    "yuhitsu_jp":       "karo_default",
    "onmitsu_local":    "karo_default",
    "kengyo_vision":    "shogun_direct",
    "daigensui_direct": "shogun_direct",
    "shogun_direct":    "karo_default",
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
        "sanbo":     SanboRole(),
        "shogun":    ShogunRole(),
        "daigensui": DaigensuiRole(),
        "yuhitsu":   YuhitsuRole(),
        "onmitsu":   OnmitsuRole(),
        "kengyo":    KengyoRole(),
    }


# =============================================================================
# LangGraph Router v14
# =============================================================================

class LangGraphRouter:
    """LangGraph StateGraph v14 — タイムアウト + ヘルスチェック + HITL"""

    def __init__(self, orchestrator: Any = None):
        self.orchestrator = orchestrator
        self._roles: dict = {}
        self._compiled = None
        self._checkpointer = MemorySaver()

    async def initialize(self) -> None:
        logger.info("🔗 LangGraph Router v14 初期化中...")
        try:
            self._roles = _load_roles()
            self._compiled = self._build_graph().compile(
                checkpointer=self._checkpointer
            )
            # バックグラウンドヘルスチェック開始 (5分間隔でキャッシュ更新)
            self._health_task = asyncio.create_task(
                self._background_health_check(), name="health_check_bg"
            )
            logger.info("✅ LangGraph Router v14 初期化完了 (MemorySaver 有効)")
        except Exception as e:
            import traceback
            logger.error("❌ LangGraph Router 初期化エラー: %s\n%s", e, traceback.format_exc())
            raise

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

        # ── ノード登録 ────────────────────────────────────────────────
        graph.add_node("analyze_intent",   self._analyze_intent)
        graph.add_node("notion_retrieve",  self._notion_retrieve)
        graph.add_node("groq_qa",          self._exec_node("seppou",    "groq_qa"))
        graph.add_node("gunshi_pdca",      self._exec_node("gunshi",    "gunshi_pdca"))
        graph.add_node("gaiji_rag",        self._exec_node("gaiji",     "gaiji_rag"))
        graph.add_node("taisho_mcp",       self._exec_node("sanbo",     "taisho_mcp"))
        graph.add_node("yuhitsu_jp",       self._exec_node("yuhitsu",   "yuhitsu_jp"))
        graph.add_node("karo_default",     self._exec_node("uketuke",   "karo_default"))
        graph.add_node("onmitsu_local",    self._exec_node("onmitsu",   "onmitsu_local"))
        graph.add_node("shogun_direct",    self._exec_node("shogun",    "shogun_direct"))
        graph.add_node("daigensui_direct", self._exec_node("daigensui", "daigensui_direct"))
        graph.add_node("kengyo_vision",    self._exec_node("kengyo",    "kengyo_vision"))
        graph.add_node("check_followup",   self._check_followup)
        graph.add_node("human_interrupt",  self._human_interrupt)
        graph.add_node("notion_store",     self._notion_store)

        # ── エントリーポイント ──────────────────────────────────────
        graph.set_entry_point("analyze_intent")
        graph.add_edge("analyze_intent", "notion_retrieve")

        # notion_retrieve → ルーティング
        graph.add_conditional_edges(
            "notion_retrieve",
            self._route_decision,
            {
                "groq_qa":          "groq_qa",
                "gunshi_pdca":      "gunshi_pdca",
                "gaiji_rag":        "gaiji_rag",
                "taisho_mcp":       "taisho_mcp",
                "yuhitsu_jp":       "yuhitsu_jp",
                "karo_default":     "karo_default",
                "onmitsu_local":    "onmitsu_local",
                "shogun_direct":    "shogun_direct",
                "daigensui_direct": "daigensui_direct",
                "kengyo_vision":    "kengyo_vision",
            },
        )

        # 全実行ノード → check_followup
        for node in (
            "groq_qa", "gunshi_pdca", "gaiji_rag", "taisho_mcp",
            "yuhitsu_jp", "karo_default", "onmitsu_local",
            "shogun_direct", "daigensui_direct", "kengyo_vision",
        ):
            graph.add_edge(node, "check_followup")

        # check_followup → 3分岐: human / loop / done
        graph.add_conditional_edges(
            "check_followup",
            self._followup_decision,
            {
                "human": "human_interrupt",
                "loop":  "notion_retrieve",
                "done":  "notion_store",
            },
        )
        graph.add_edge("human_interrupt", "notion_store")
        graph.add_edge("notion_store", END)

        return graph

    # =========================================================================
    # ノード: analyze_intent — タスク特性分析
    # =========================================================================

    def _analyze_intent(self, state: BushidanState) -> dict:
        message = state.get("message", "")
        logger.info("📊 [analyze_intent] '%s'...", message[:60])

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

        multi_kws = ["そして", "次に", "さらに", "まず", "step", "then", "after"]
        is_multi = any(kw in content_lower for kw in multi_kws) and len(message) > 100

        action_kws = ["clone", "push", "pull", "install", "run ", "実行して", "削除して", "作成して", "検索して"]
        is_action = any(kw in content_lower for kw in action_kws)

        qa_kws = ["とは", "って何", "ですか", "what is", "what's", "how does", "explain"]
        is_simple_qa = any(kw in content_lower for kw in qa_kws) and not is_action

        jp_kws = ["日本語で書いて", "日本語で回答", "和訳", "添削", "校正", "翻訳", "ビジネスメール", "敬語で"]
        is_japanese = any(kw in message for kw in jp_kws)

        conf_kws = ["機密", "秘密", "confidential", "private", "オフライン", "社外秘", "外部送信禁止"]
        is_confidential = any(kw in content_lower or kw in message for kw in conf_kws)

        has_vision = bool(state.get("attachments"))

        return {
            "complexity": complexity,
            "is_multi_step": is_multi,
            "is_action_task": is_action,
            "is_simple_qa": is_simple_qa,
            "is_japanese_priority": is_japanese,
            "is_confidential": is_confidential,
            "attachments": state.get("attachments", []),
            "forced_role": state.get("forced_role") or ("kengyo" if has_vision else None),
        }

    # =========================================================================
    # ノード: notion_retrieve — Notion RAG 検索
    # =========================================================================

    async def _notion_retrieve(self, state: BushidanState) -> dict:
        message = state.get("message", "")
        if not message.strip():
            return {"notion_chunks": []}
        try:
            from integrations.notion.retrieval import retrieve
            chunks = await retrieve(message, top_k=4)
            logger.info("🔍 [notion_retrieve] %d件取得", len(chunks))
            return {"notion_chunks": chunks}
        except Exception as e:
            logger.debug("[notion_retrieve] スキップ: %s", e)
            return {"notion_chunks": []}

    # =========================================================================
    # ルーティング判断 (ヘルスチェック統合)
    # =========================================================================

    def _route_decision(self, state: BushidanState) -> str:
        """v14 ルーティング — ヘルスチェックフォールバック付き"""
        forced = state.get("forced_role")
        _valid = {
            "groq_qa", "gunshi_pdca", "gaiji_rag", "taisho_mcp",
            "yuhitsu_jp", "karo_default", "onmitsu_local",
            "shogun_direct", "daigensui_direct", "kengyo_vision",
        }
        _role_to_node = {
            "seppou": "groq_qa", "gunshi": "gunshi_pdca", "gaiji": "gaiji_rag",
            "sanbo": "taisho_mcp", "yuhitsu": "yuhitsu_jp", "uketuke": "karo_default",
            "onmitsu": "onmitsu_local", "shogun": "shogun_direct",
            "daigensui": "daigensui_direct", "kengyo": "kengyo_vision",
        }
        _node_to_role = {
            "groq_qa": "seppou", "gunshi_pdca": "gunshi", "gaiji_rag": "gaiji",
            "taisho_mcp": "sanbo", "yuhitsu_jp": "yuhitsu", "karo_default": "uketuke",
            "onmitsu_local": "onmitsu", "shogun_direct": "shogun",
            "daigensui_direct": "daigensui", "kengyo_vision": "kengyo",
        }

        def _check_health(node: str) -> str:
            """ヘルスチェック — unhealthy なら _FALLBACK_MAP で代替"""
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

        if forced:
            node = _role_to_node.get(forced, forced)
            if node in _valid:
                logger.info("🎯 Route: %s (forced_role=%s)", node, forced)
                return _check_health(node)

        if state.get("is_confidential"):
            logger.info("🥷 Route: onmitsu_local")
            return _check_health("onmitsu_local")

        if state.get("is_japanese_priority"):
            logger.info("🖊️ Route: yuhitsu_jp")
            return _check_health("yuhitsu_jp")

        complexity = state.get("complexity", "medium")
        is_multi   = state.get("is_multi_step", False)
        is_action  = state.get("is_action_task", False)
        is_simple  = state.get("is_simple_qa", False)

        if complexity in ("strategic", "complex") or (is_multi and complexity != "simple"):
            logger.info("🧠 Route: gunshi_pdca (complexity=%s)", complexity)
            return _check_health("gunshi_pdca")

        if is_action and state.get("available_tools"):
            logger.info("🔧 Route: taisho_mcp")
            return _check_health("taisho_mcp")

        if is_multi or is_action:
            logger.info("🌐 Route: gaiji_rag")
            return _check_health("gaiji_rag")

        if is_simple and complexity == "simple":
            logger.info("⚡ Route: groq_qa")
            return _check_health("groq_qa")

        logger.info("🏯 Route: karo_default")
        return "karo_default"

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

            # MCPツール注入
            try:
                from core.mcp_sdk import MCPToolRegistry
                registry = MCPToolRegistry.get()
                tools = registry.get_tools_for_role(role_key)
                if tools:
                    state_tools = [t.name for t in tools]
                else:
                    state_tools = state.get("available_tools", [])
            except Exception:
                state_tools = state.get("available_tools", [])

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
            return {
                "response":       result.response,
                "handled_by":     result.handled_by,
                "agent_role":     result.agent_role,
                "execution_time": result.execution_time,
                "error":          result.error,
                "mcp_tools_used": result.mcp_tools_used + saved_files,
                "requires_followup": result.requires_followup,
                "routed_to":      node_name,
                "available_tools": state_tools,
                "conversation_history": new_history,
            }

        _node.__name__ = f"exec_{node_name}"
        return _node

    # =========================================================================
    # ノード: check_followup — 3分岐 (human / loop / done)
    # =========================================================================

    def _check_followup(self, state: BushidanState) -> dict:
        iteration = state.get("iteration", 0) + 1
        max_iter  = state.get("max_iterations", 3)
        followup  = state.get("requires_followup", False)

        if followup and iteration < max_iter:
            logger.info("🔄 自律ループ継続 (%d/%d)", iteration, max_iter)
            return {"iteration": iteration, "requires_followup": True}

        if followup and iteration >= max_iter:
            logger.info("⚠️ 最大反復回数到達 (%d) — 強制終了", max_iter)

        return {"iteration": iteration, "requires_followup": False}

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
        Mattermost Bot 側が human_response を設定して再開する。
        """
        question = state.get("human_question", "")
        logger.info("🙋 HITL: 人間の入力を待機中 — %s", question[:60])
        return {
            "dialog_status": "waiting_for_human",
            "awaiting_human_input": False,  # 中断後にリセット
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
        """メッセージを v14 LangGraph ルーターで処理。"""
        if not self._compiled:
            await self.initialize()

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
            # v14: HITL
            "dialog_status":       "active",
            "awaiting_human_input": False,
            "human_question":      "",
            "human_response":      "",
            # v14: context summary
            "context_summary":     "",
        }

        config = {"configurable": {"thread_id": thread_id or "default"}}
        logger.info("🔗 LangGraph v14: 処理開始 thread=%s '%s'...",
                     thread_id[:8] if thread_id else "new", message[:60])
        start = time.time()

        try:
            final = await self._compiled.ainvoke(initial, config=config)
            total = time.time() - start
            logger.info(
                "✅ LangGraph v14: 完了 route=%s agent=%s time=%.2fs",
                final.get("routed_to"), final.get("agent_role"), total
            )
            return {
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
                "version":        "14",
            }
        except Exception as e:
            logger.exception("❌ LangGraph v14 処理失敗: %s", e)
            return {
                "status":         "failed",
                "error":          str(e),
                "response":       f"❌ 処理失敗: {e}",
                "result":         f"❌ 処理失敗: {e}",
                "handled_by":     "langgraph_router",
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
