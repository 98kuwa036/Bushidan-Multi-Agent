"""
core/router/mixins/routing.py — notion_index ノード + ルーティング判断 Mixin
"""
import asyncio
import time
from typing import TYPE_CHECKING
from utils.logger import get_logger
from core.router.constants import FALLBACK_MAP, NODE_TIMEOUTS

logger = get_logger(__name__)

if TYPE_CHECKING:
    from core.state import BushidanState


class RoutingMixin:
    """_notion_index_node と _route_decision / _exec_node を提供する"""

    if TYPE_CHECKING:
        _roles: dict

    async def _notion_index_node(self, state: "BushidanState") -> dict:
        """ローカルインデックスで高速検索。simple Q&A はスキップ。"""
        complexity = state.get("complexity", "medium")
        is_simple  = state.get("is_simple_qa", False)
        if complexity == "simple" and is_simple:
            return {"notion_chunks": []}
        intent    = state.get("intent_structured", {})
        user_goal = intent.get("user_goal", "") or state.get("message", "")[:120]
        try:
            from integrations.notion.index import lookup
            chunks = await lookup(user_goal, top_k=3)
            if chunks:
                logger.info("📋 [notion_index] %d件ヒット", len(chunks))
            return {"notion_chunks": [c.__dict__ if hasattr(c, "__dict__") else c for c in chunks]}
        except Exception as e:
            logger.debug("[notion_index] スキップ: %s", e)
            return {"notion_chunks": []}

    def _route_decision(self, state: "BushidanState") -> str:
        """v16 ルーティング — 構造化インテントを活用した細粒度振り分け"""
        forced = state.get("forced_role")

        _valid = {
            "groq_qa", "parallel_groq", "gunshi_haiku", "metsuke_proc", "gaiji_rag", "sanbo_mcp",
            "yuhitsu_jp", "uketuke_default", "onmitsu_local", "kengyo_vision",
            "shogun_plan", "daigensui_audit",
        }
        _role_to_node = {
            "seppou":    "groq_qa",      "gunshi":    "gunshi_haiku",
            "metsuke":   "metsuke_proc", "gaiji":     "gaiji_rag",
            "sanbo":     "sanbo_mcp",    "yuhitsu":   "yuhitsu_jp",
            "uketuke":   "uketuke_default", "onmitsu": "onmitsu_local",
            "shogun":    "shogun_plan",  "daigensui": "daigensui_audit",
            "kengyo":    "kengyo_vision",
        }
        _node_to_role = {v: k for k, v in _role_to_node.items()}

        def _check_health(node: str) -> str:
            role = _node_to_role.get(node)
            if not role:
                return node
            try:
                from utils.client_registry import ClientRegistry
                if ClientRegistry.get().is_healthy_cached(role):
                    return node
                fallback = FALLBACK_MAP.get(node)
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

        if state.get("is_confidential"):
            return _check_health("onmitsu_local")
        if state.get("attachments"):
            return _check_health("kengyo_vision")

        intent       = state.get("intent_structured", {})
        intent_type  = intent.get("intent_type", "")
        required_caps = intent.get("required_capabilities", [])

        if intent_type == "rag" or (
            "rag" in required_caps and
            not any(c in required_caps for c in ["code", "tools", "web_search"])
        ):
            return _check_health("gaiji_rag")

        if "web_search" in required_caps or intent_type == "research":
            return _check_health("gaiji_rag")

        complexity = state.get("complexity", "medium")
        is_multi   = state.get("is_multi_step", False)
        is_action  = state.get("is_action_task", False)
        is_simple  = state.get("is_simple_qa", False)

        # BATCH モード: 複雑・戦略タスク → batch_parallel で全ステップ一括処理
        from core.router.batch.mode import ProcessingMode, BATCH_CONFIG as _BC
        _mode = ProcessingMode(state.get("processing_mode", ProcessingMode.INTERACTIVE))
        if (
            _mode == ProcessingMode.BATCH
            and complexity in ("strategic", "complex")
            and is_multi
        ):
            logger.info("🗺️  Route: batch_parallel (BATCH mode, complexity=%s)", complexity)
            return "batch_parallel"

        if complexity in ("strategic", "complex") or (is_multi and complexity not in ("simple",)):
            logger.info("🗺️  Route: shogun_plan (complexity=%s)", complexity)
            return _check_health("shogun_plan")

        if is_action and "tools" in required_caps:
            return _check_health("sanbo_mcp")

        if "code" in required_caps and complexity in ("simple", "low_medium"):
            return _check_health("groq_qa")

        if state.get("is_japanese_priority") and complexity in ("simple", "low_medium", "medium"):
            return _check_health("yuhitsu_jp")

        if complexity == "medium" or is_action:
            msg = state.get("message", "")
            if is_multi and msg.count("?") + msg.count("？") >= 2:
                return _check_health("parallel_groq")
            return _check_health("gunshi_haiku")

        if complexity == "low_medium":
            return _check_health("metsuke_proc")

        if is_simple or complexity == "simple":
            msg = state.get("message", "")
            if is_multi and msg.count("?") + msg.count("？") >= 2:
                return _check_health("parallel_groq")
            return _check_health("groq_qa")

        logger.info("🏯 Route: uketuke_default (フォールバック)")
        return "uketuke_default"

    def _exec_node(self, role_key: str, node_name: str):
        """指定ロールの execute() をタイムアウト付きで呼ぶノード関数を生成する。"""
        from core.router.batch.mode import ProcessingMode, BATCH_CONFIG
        base_timeout = NODE_TIMEOUTS.get(node_name, 120)

        async def _node(state: "BushidanState") -> dict:
            mode = ProcessingMode(state.get("processing_mode", ProcessingMode.INTERACTIVE))
            multiplier = BATCH_CONFIG["node_timeout_multiplier"] if mode == ProcessingMode.BATCH else 1.0
            timeout = int(base_timeout * multiplier)

            role = self._roles.get(role_key)
            if not role:
                logger.error("ロール未ロード: %s", role_key)
                return {
                    "response": f"❌ ロール {role_key} 未初期化",
                    "handled_by": node_name, "agent_role": role_key,
                    "execution_time": 0.0, "error": "role not loaded",
                    "routed_to": node_name, "mcp_tools_used": [],
                }

            try:
                from core.mcp_sdk import MCPToolRegistry
                lc_tools = MCPToolRegistry.get().get_tools_for_role(role_key)
                if lc_tools:
                    state = dict(state)
                    state["available_tools"] = [t.name for t in lc_tools]
                    state["mcp_tools"] = lc_tools
            except Exception:
                pass

            start = time.time()
            try:
                result = await asyncio.wait_for(role.execute(state), timeout=timeout)
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
                        {"role": "user",      "content": state.get("message", "")},
                        {"role": "assistant", "content": f"⏱️ タイムアウト ({timeout}秒)"},
                    ],
                }

            saved_files: list = []
            try:
                from utils.bushidan_files import extract_and_save_files
                saved_files = extract_and_save_files(result.response)
                if saved_files:
                    logger.info("📁 Bushidan保存 [%s]: %s", node_name, saved_files)
            except Exception:
                pass

            state_update = {
                "response":          result.response,
                "handled_by":        result.handled_by,
                "agent_role":        result.agent_role,
                "execution_time":    result.execution_time,
                "error":             result.error,
                "mcp_tools_used":    result.mcp_tools_used + saved_files,
                "requires_followup": result.requires_followup,
                "routed_to":         node_name,
                "available_tools":   state.get("available_tools", []),
                "conversation_history": state.get("conversation_history", []) + [
                    {"role": "user",      "content": state.get("message", "")},
                    {"role": "assistant", "content": result.response},
                ],
            }
            if getattr(result, "awaiting_human_input", False):
                state_update["awaiting_human_input"] = True
                state_update["human_question"] = getattr(result, "human_question", "")
                state_update["dialog_status"] = "waiting_for_human"
            return state_update

        _node.__name__ = f"exec_{node_name}"
        return _node
