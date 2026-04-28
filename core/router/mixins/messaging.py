"""
core/router/mixins/messaging.py — 公開インターフェース Mixin

process_message / stream_message / resume / process_task
バッチモード対応: キャッシュ無効化・SemanticRouter スキップ・processing_mode 注入
"""
import asyncio
import hashlib
import time
from typing import TYPE_CHECKING, Optional
from utils.logger import get_logger
from core.router.batch.mode import ProcessingMode, BATCH_CONFIG
from core.router.constants import skill_observe, fire

logger = get_logger(__name__)

if TYPE_CHECKING:
    from core.state import BushidanState


def _make_initial_state(
    message: str,
    thread_id: str = "",
    channel_id: str = "",
    user_id: str = "",
    source: str = "api",
    forced_role: Optional[str] = None,
    attachments: Optional[list] = None,
    available_tools: Optional[list] = None,
    processing_mode: str = ProcessingMode.INTERACTIVE,
) -> "BushidanState":
    return {
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
        "processing_mode":     processing_mode,
    }


class MessagingMixin:

    def __init__(self):
        self._RESP_CACHE: dict = {}
        self._RESP_CACHE_TTL: float = 300.0
        self._cache_lock: asyncio.Lock = asyncio.Lock()

    if TYPE_CHECKING:
        _compiled: object
        _compiled_fast: object
        _audit_logs: dict

        async def initialize(self): ...

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
        mode: ProcessingMode = ProcessingMode.INTERACTIVE,
    ) -> dict:
        """メッセージを処理して結果を返す (v16)"""
        if not self._compiled:
            await self.initialize()

        is_batch = mode == ProcessingMode.BATCH

        # ── 応答キャッシュチェック (BATCH モードでは無効) ──────────────
        _cache_key = None
        _can_cache = (
            not is_batch
            and not forced_role
            and not attachments
            and len(message) <= 200
        )
        if _can_cache:
            _tid = thread_id or "default"
            _hist_len = len(self._compiled.get_state({"configurable": {"thread_id": _tid}}).values.get("conversation_history", [])) if self._compiled else 0
            _h = hashlib.md5(f"{_tid}:{_hist_len}:{message}".encode()).hexdigest()
            _cache_key = _h
            async with self._cache_lock:
                _entry = self._RESP_CACHE.get(_h)
                if _entry and (time.time() - _entry[1]) < self._RESP_CACHE_TTL:
                    logger.debug("⚡ 応答キャッシュヒット: %s", message[:40])
                    return _entry[0]

        # ── SemanticRouter 事前チェック (INTERACTIVE + 添付・強制ロールなし時のみ) ──
        if not forced_role and not attachments and not is_batch:
            try:
                from utils.semantic_router import SemanticRouter, CONFIDENT_THRESHOLD
                _sr = SemanticRouter.get()
                if not _sr.is_ready:
                    _sr.initialize()
                if _sr.is_ready:
                    _sem_route, _sem_score = _sr.route(message)
                    if _sem_route and _sem_score >= CONFIDENT_THRESHOLD:
                        _ROUTE_MAP = {
                            "groq_qa":       "uketuke",
                            "yuhitsu_jp":    "yuhitsu",
                            "metsuke_proc":  "metsuke",
                            "gunshi_haiku":  "gunshi",
                            "gaiji_rag":     "gaiji",
                            "sanbo_mcp":     "sanbo",
                            "kengyo_vision": "kengyo",
                            "onmitsu_local": "onmitsu",
                            "shogun_plan":   "shogun",
                        }
                        _sem_role = _ROUTE_MAP.get(_sem_route)
                        if _sem_role:
                            logger.info(
                                "⚡ [SR事前] %s (%.3f) → forced_role=%s (Flash-Lite スキップ)",
                                _sem_route, _sem_score, _sem_role,
                            )
                            forced_role = _sem_role
            except Exception as _sr_pre_err:
                logger.debug("SR事前チェックスキップ: %s", _sr_pre_err)

        initial = _make_initial_state(
            message=message,
            thread_id=thread_id,
            channel_id=channel_id,
            user_id=user_id,
            source=source,
            forced_role=forced_role,
            attachments=attachments,
            available_tools=available_tools,
            processing_mode=mode.value,
        )

        # BATCH: MemorySaver fast グラフも使わない（PG 永続化で追跡性確保）
        _use_fast = (
            _can_cache
            and self._compiled_fast is not None
        )
        _graph = self._compiled_fast if _use_fast else self._compiled

        config = {"configurable": {"thread_id": thread_id or "default"}}
        logger.info("🔗 LangGraph v16: 処理開始 thread=%s graph=%s mode=%s '%s'...",
                    thread_id[:8] if thread_id else "new",
                    "fast" if _use_fast else "full",
                    mode.value,
                    message[:60])
        start = time.time()

        try:
            base_timeout = 180 if not _use_fast else 30
            multiplier = BATCH_CONFIG["node_timeout_multiplier"] if is_batch else 1.0
            timeout = int(base_timeout * multiplier)
            final = await asyncio.wait_for(_graph.ainvoke(initial, config=config), timeout=timeout)
            total = time.time() - start
            logger.info(
                "✅ LangGraph v16: 完了 route=%s agent=%s time=%.2fs",
                final.get("routed_to"), final.get("agent_role"), total,
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

            if (
                _cache_key
                and result_dict["status"] == "completed"
                and not final.get("error")
                and not final.get("is_action_task")
                and not final.get("requires_followup")
            ):
                async with self._cache_lock:
                    self._RESP_CACHE[_cache_key] = (result_dict, time.time())
                    if len(self._RESP_CACHE) > 500:
                        now = time.time()
                        self._RESP_CACHE = {
                            k: v for k, v in self._RESP_CACHE.items()
                            if now - v[1] < self._RESP_CACHE_TTL
                        }

            _tid_key = thread_id or "default"
            _alog = self._audit_logs.pop(_tid_key, None)
            if _alog:
                try:
                    _alog.finish(final.get("response", ""))
                except Exception:
                    pass
            _routed = result_dict.get("route", "")
            _handled = result_dict.get("handled_by", "")
            _used_fallback = bool(_routed and _handled and _routed != _handled)
            fire(
                skill_observe(
                    thread_id=_tid_key,
                    message=message,
                    handled_by=_handled,
                    execution_time=total,
                    success=result_dict.get("status") == "completed",
                    error=str(final.get("error", "") or ""),
                    had_hitl=bool(final.get("awaiting_human_input")),
                    used_fallback=_used_fallback,
                ),
                name="skill_observe",
            )
            return result_dict

        except asyncio.TimeoutError:
            total = time.time() - start
            logger.error("⏱️ LangGraph v16 タイムアウト (%.1fs): thread=%s message='%s'",
                         total, thread_id[:8], message[:40])
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
                "result":         "❌ 処理タイムアウト",
                "handled_by":     "langgraph_router",
                "execution_time": total,
            }
        except Exception as e:
            logger.exception("❌ LangGraph v16 処理失敗: %s", e)
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
        メッセージをストリーミング処理（INTERACTIVE モード専用）。
        LangGraph astream_events で LLM トークンを yield する非同期ジェネレータ。

        Yields:
            dict: {"type": "token", "content": str}
                  {"type": "route", "route": str, "agent_role": str}
                  {"type": "done", "execution_time": float}
        """
        if not self._compiled:
            await self.initialize()

        initial = _make_initial_state(
            message=message,
            thread_id=thread_id,
            source=source,
            forced_role=forced_role,
            attachments=attachments,
            processing_mode=ProcessingMode.INTERACTIVE,
        )

        config = {"configurable": {"thread_id": thread_id or "default"}}
        start = time.time()
        route_sent = False
        roadmap_emitted = False
        last_roadmap_step = -1
        stream_timeout = 240
        _final_response: Optional[str] = None
        _final_handled_by: Optional[str] = None
        _final_agent_role: Optional[str] = None
        _final_route: Optional[str] = None
        _final_routed_by: Optional[str] = None  # "semantic_router" | "analyze_intent" | "forced"

        try:
            async def _stream_with_timeout():
                try:
                    async with asyncio.timeout(stream_timeout):
                        async for event in self._compiled.astream_events(
                            initial, config=config, version="v2"
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

                if event_type == "timeout":
                    yield {"type": "error", "message": event.get("message")}
                    break

                if event_type == "on_chain_end" and name == "shogun_plan" and not roadmap_emitted:
                    output = event.get("data", {}).get("output") or {}
                    roadmap = output.get("roadmap")
                    if roadmap:
                        yield {"type": "roadmap_created", "roadmap": roadmap}
                        roadmap_emitted = True

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

                if event_type == "on_chain_start" and name == "daigensui_audit":
                    yield {"type": "audit_start"}

                if event_type == "on_chain_start" and not route_sent:
                    if name not in ("LangGraph", "analyze_intent", "notion_index", "execute_step", "shogun_plan"):
                        yield {"type": "route", "route": name}
                        route_sent = True
                        _final_route = name

                if event_type == "on_chain_end" and name == "analyze_intent" and not _final_routed_by:
                    _final_routed_by = "analyze_intent"

                if event_type == "on_chain_end":
                    output = event.get("data", {}).get("output") or {}
                    if isinstance(output, dict):
                        resp = output.get("response")
                        if resp:
                            _final_response = resp
                            _final_handled_by = output.get("handled_by")
                            _final_agent_role = output.get("agent_role")
                            if output.get("routed_to"):
                                _final_route = output.get("routed_to")

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
            if forced_role:
                _final_routed_by = "forced"
            elif _final_routed_by is None:
                _final_routed_by = "semantic_router"
            yield {
                "type": "done",
                "execution_time": time.time() - start,
                "response":    _final_response,
                "handled_by":  _final_handled_by,
                "agent_role":  _final_agent_role,
                "route":       _final_route,
                "routed_by":   _final_routed_by,
            }

    async def resume(self, thread_id: str, human_response: str) -> dict:
        """HITL Go サイン: human_interrupt で停止中のグラフを再開する。"""
        if not self._compiled:
            await self.initialize()

        from langgraph.types import Command
        config = {"configurable": {"thread_id": thread_id}}
        logger.info("▶️  HITL resume: thread=%s response='%s'...", thread_id[:8], human_response[:40])
        start = time.time()

        try:
            final = await asyncio.wait_for(
                self._compiled.ainvoke(Command(resume=human_response), config=config),
                timeout=150,
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

    async def process_task(
        self,
        content: str,
        context: dict = None,
        priority: int = 1,
        source: str = "api",
    ) -> dict:
        """後方互換ラッパー。"""
        ctx = context or {}
        return await self.process_message(
            message=content,
            thread_id=ctx.get("root_id", ctx.get("thread_id", "")),
            channel_id=ctx.get("channel_id", ""),
            user_id=ctx.get("user_id", ""),
            source=ctx.get("source", source),
            forced_role=ctx.get("forced_route") or ctx.get("forced_agent"),
        )
