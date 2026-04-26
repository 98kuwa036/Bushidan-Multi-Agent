"""
core/router/mixins/nodes.py — 実行・特殊ノード群 Mixin

将軍プランニング (_shogun_plan_node, _execute_step_node, _step_decision)、
大元帥監査 (_daigensui_audit_node)、並列斥候 (_parallel_groq_node)、
バッチ並列 (_batch_parallel_node)、コード検証 (_sandbox_verify_node) を提供する。
"""
import asyncio
import time
from typing import TYPE_CHECKING
from utils.logger import get_logger
from core.router.constants import NODE_TIMEOUTS
from core.router.batch.mode import ProcessingMode, BATCH_CONFIG
from core.router.batch.anthropic_batch import ANTHROPIC_ROLES, AnthropicBatchProcessor

logger = get_logger(__name__)

if TYPE_CHECKING:
    from core.state import BushidanState


class NodesMixin:

    if TYPE_CHECKING:
        _roles: dict
        _audit_logs: dict

    # capability → role_key マッピング
    _CAP_TO_ROLE = {
        "analysis":   "gunshi",
        "quick_task": "metsuke",
        "rag":        "gaiji",
        "web_search": "gaiji",
        "code":       "sanbo",
        "tools":      "sanbo",
        "japanese":   "yuhitsu",
        "image":      "kengyo",
        "summary":    "metsuke",
    }

    async def _shogun_plan_node(self, state: "BushidanState") -> dict:
        """将軍 (Claude Sonnet) がユーザー要望をロードマップ JSON に分解する。"""
        start = time.time()
        intent     = state.get("intent_structured", {})
        complexity = state.get("complexity", "complex")
        message    = state.get("message", "")

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
                '{"goal":"ユーザー目標の1文要約","steps":['
                '{"id":1,"task":"具体的なタスク説明","capability":"analysis|rag|web_search|code|tools|japanese|image|quick_task",'
                '"assigned_role":"gunshi|metsuke|gaiji|seppou|sanbo|yuhitsu|kengyo","can_parallel":false,"status":"pending"}],'
                '"needs_audit":false}\n\n'
                "capability → assigned_role: analysis→gunshi, quick_task→metsuke, rag→gaiji, "
                "web_search→seppou, code→sanbo, tools→sanbo, japanese→yuhitsu, image→kengyo\n"
                "needs_audit: 最高難度・重大な意思決定・本番デプロイ関連の場合 true\n"
                "can_parallel: 前ステップの結果に依存しない独立したタスクの場合 true\n"
            )
            if notion_ctx:
                system += f"\n【関連ナレッジ】\n{notion_ctx}"

            raw = await client.generate(
                messages=[{"role": "user", "content":
                    f"ユーザーの要望: {message}\n解釈した目標: {intent.get('user_goal', message[:200])}\n"
                    f"complexity: {complexity}\nロードマップJSONを作成してください。"}],
                system=system,
                max_tokens=800,
            )

            import re as _re
            import json as _json
            m = _re.search(r'\{.*\}', raw, _re.DOTALL)
            if m:
                roadmap = _json.loads(m.group())
                for s in roadmap.get("steps", []):
                    s["status"] = "pending"
                    s.setdefault("result", "")
                needs_audit = bool(roadmap.get("needs_audit", False)) or complexity == "strategic"
                logger.info("🗺️ [将軍] ロードマップ: %d steps audit=%s",
                            len(roadmap.get("steps", [])), needs_audit)
                try:
                    from utils.audit_log import AuditLog
                    _tid  = state.get("thread_id") or "default"
                    _alog = AuditLog.start(_tid, message, roadmap)
                    _alog.set_complexity(complexity)
                    self._audit_logs[_tid] = _alog
                except Exception as e:
                    logger.debug("audit_log init skipped: %s", e)
                return {
                    "roadmap": roadmap, "roadmap_step": 0, "roadmap_results": [],
                    "needs_audit": needs_audit, "routed_to": "shogun_plan",
                    "agent_role": "将軍", "handled_by": "shogun_plan",
                    "execution_time": time.time() - start,
                }
        except Exception as e:
            logger.warning("🗺️ [将軍] ロードマップ作成失敗 → 直接実行: %s", e)

        return {
            "roadmap": {
                "goal": message[:100],
                "steps": [{"id": 1, "task": message, "capability": "analysis",
                           "assigned_role": "gunshi", "status": "pending", "result": ""}],
                "needs_audit": False,
            },
            "roadmap_step": 0, "roadmap_results": [], "needs_audit": False,
            "routed_to": "shogun_plan", "agent_role": "将軍", "handled_by": "shogun_plan",
            "execution_time": time.time() - start,
        }

    async def _execute_step_node(self, state: "BushidanState") -> dict:
        """ロードマップステップを適切なロールで実行する（並列対応）。"""
        roadmap  = state.get("roadmap", {})
        step_idx = state.get("roadmap_step", 0)
        steps    = roadmap.get("steps", [])
        start    = time.time()

        if step_idx >= len(steps):
            return {"roadmap_step": step_idx, "execution_time": time.time() - start}

        batch_end = step_idx
        if steps[step_idx].get("can_parallel", False):
            for i in range(step_idx + 1, len(steps)):
                if steps[i].get("can_parallel", False):
                    batch_end = i
                else:
                    break
        batch_steps = steps[step_idx: batch_end + 1]
        is_parallel = len(batch_steps) > 1

        prev_results = state.get("roadmap_results", [])
        prev_ctx = ""
        if prev_results:
            prev_ctx = "\n".join(
                f"[Step {r.get('step_id', i+1)}] {r.get('summary', r.get('response', '')[:200])}"
                for i, r in enumerate(prev_results[-3:])
            )

        timeout = NODE_TIMEOUTS.get("execute_step", 120)

        async def _run_one(s: dict, idx: int):
            capability = s.get("capability", "analysis")
            assigned   = s.get("assigned_role", "")
            task_desc  = s.get("task", "")
            role_key   = assigned if assigned in self._roles else self._CAP_TO_ROLE.get(capability, "gunshi")
            role       = self._roles.get(role_key) or self._roles.get("gunshi")
            sub_state  = {
                **state,
                "message":            task_desc,
                "_step_task":         task_desc,
                "conversation_history": [],
                "roadmap_results":    [],
                "sub_queries":        [],
                "sub_responses":      [],
                "notion_chunks":      list(state.get("notion_chunks", [])),
                "attachments":        list(state.get("attachments", [])),
                "mcp_tools_used":     [],
            }
            if prev_ctx:
                sub_state["context_summary"] = (
                    f"ロードマップ目標: {roadmap.get('goal', '')}\n完了済みステップ:\n{prev_ctx}"
                )
            try:
                from core.mcp_sdk import MCPToolRegistry
                lc_tools = MCPToolRegistry.get().get_tools_for_role(role_key)
                if lc_tools:
                    tool_names = [t.name for t in lc_tools]
                    sub_state["available_tools"] = tool_names
                    sub_state["mcp_tools"] = tool_names  # BaseTool オブジェクトは DB 非永続化対象のため名前のみ
            except Exception as e:
                logger.debug("mcp_tools load skipped: %s", e)
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

        # ── Anthropic Batch API (並列かつ全ステップが Anthropic ロールの場合) ──
        mode = ProcessingMode(state.get("processing_mode", ProcessingMode.INTERACTIVE))
        use_anth_batch = (
            is_parallel
            and mode == ProcessingMode.BATCH
            and BATCH_CONFIG.get("use_anthropic_batch", False)
            and all(
                (s.get("assigned_role") or self._CAP_TO_ROLE.get(s.get("capability", ""), "")) in ANTHROPIC_ROLES
                for s in batch_steps
            )
        )

        if use_anth_batch:
            batch_results = await self._run_anthropic_batch(
                batch_steps, step_idx, state, prev_ctx, roadmap
            )
        elif is_parallel:
            logger.info("⚡ [execute_step] asyncio並列 %d steps", len(batch_steps))
            raw_results = await asyncio.gather(*[
                _run_one(s, step_idx + i) for i, s in enumerate(batch_steps)
            ], return_exceptions=True)
            batch_results = []
            for ri, raw in enumerate(raw_results):
                if isinstance(raw, BaseException):
                    from roles.base import RoleResult
                    s = batch_steps[ri]
                    err_res = RoleResult(response=f"❌ エラー: {raw}", agent_role="unknown",
                                         handled_by="execute_step", error=str(raw), status="failed")
                    batch_results.append((s.get("capability", ""), s.get("capability", ""),
                                          s.get("task", ""), err_res, step_idx + ri))
                else:
                    batch_results.append(raw)
        else:
            s = batch_steps[0]
            logger.info("▶️ [execute_step] step %d/%d: %s", step_idx + 1, len(steps), s.get("task", "")[:40])
            batch_results = [await _run_one(s, step_idx)]

        step_results_list = []
        updated_steps = list(steps)
        all_mcp_used  = []
        _alog = self._audit_logs.get(state.get("thread_id") or "default")

        for role_key, capability, task_desc, result, abs_idx in batch_results:
            step_results_list.append({
                "step_id": abs_idx + 1, "capability": capability, "role": role_key,
                "task": task_desc, "response": result.response,
                "summary": result.response[:200] if result.response else "", "error": result.error,
            })
            updated_steps[abs_idx] = dict(steps[abs_idx])
            updated_steps[abs_idx]["status"] = "done" if not result.error else "error"
            updated_steps[abs_idx]["result"] = result.response[:300] if result.response else ""
            all_mcp_used.extend(getattr(result, "mcp_tools_used", []))
            if _alog:
                try:
                    _alog.add_step_result(
                        step_id=abs_idx + 1, role=role_key, capability=capability,
                        task=task_desc, result=result.response or "",
                        execution_time=getattr(result, "execution_time", 0.0),
                        error=result.error or "",
                    )
                except Exception as e:
                    logger.debug("audit_log step skipped: %s", e)

        updated_roadmap = dict(roadmap)
        updated_roadmap["steps"] = updated_steps
        new_step_idx = batch_end + 1
        is_last = (new_step_idx >= len(steps))
        new_response = (
            "\n\n---\n\n".join(r["response"] for r in step_results_list if r.get("response"))
            if is_last and is_parallel
            else (batch_results[0][3].response if is_last else state.get("response", ""))
        )

        return {
            "roadmap": updated_roadmap, "roadmap_step": new_step_idx,
            "roadmap_results": step_results_list, "response": new_response,
            "agent_role": batch_results[-1][3].agent_role, "handled_by": "execute_step",
            "routed_to": f"execute_step[{step_idx+1}~{batch_end+1}]",
            "execution_time": time.time() - start, "mcp_tools_used": all_mcp_used,
            "error": batch_results[-1][3].error if is_last else None,
            "conversation_history": [
                {"role": "user",      "content": batch_steps[-1].get("task", "")},
                {"role": "assistant", "content": new_response or ""},
            ],
        }

    def _step_decision(self, state: "BushidanState") -> str:
        """ロードマップステップの次遷移を決定する。"""
        steps    = state.get("roadmap", {}).get("steps", [])
        step_idx = state.get("roadmap_step", 0)
        if step_idx < len(steps):
            return "next_step"
        if state.get("needs_audit", False):
            return "audit"
        if "```" in (state.get("response", "") or ""):
            return "verify"
        return "done"

    def _audit_decision(self, state: "BushidanState") -> str:
        """大元帥監査後の遷移: 却下かつ反復余地あり → shogun_plan 再計画"""
        if (
            state.get("error") == "audit_rejected"
            and state.get("iteration", 0) < state.get("max_iterations", 3) - 1
        ):
            logger.info("⚔️ 大元帥が却下 — 将軍へ差し戻し (iteration=%d)", state.get("iteration", 0))
            return "replan"
        return "verify"

    async def _daigensui_audit_node(self, state: "BushidanState") -> dict:
        """大元帥 (Claude Opus) がロードマップ全体を監査し最終判断を下す。"""
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
                "1. 目標達成度 2. 品質評価 3. リスク確認 4. 最終回答 の観点でレビューし日本語で述べてください。"
            )
            response = await client.generate(
                messages=[{"role": "user", "content":
                    f"【ユーザー要望】\n{message}\n\n【ロードマップ目標】\n{roadmap.get('goal','')}\n\n"
                    f"【実行済みステップ】\n{steps_summary}\n\n"
                    f"【最終ステップの回答】\n{state.get('response','')[:1000]}"}],
                system=system,
                max_tokens=2000,
            )
            elapsed = time.time() - start
            logger.info("⚔️ [大元帥監査] 完了 %.1fs", elapsed)
            # LLM レスポンスから承認/却下を判定（キーワードベース）
            _resp_lower = response.lower()
            _rejected_kw = ["却下", "拒否", "不承認", "問題あり", "修正が必要", "reject", "denied"]
            _verdict = "rejected" if any(kw in _resp_lower for kw in _rejected_kw) else "approved"
            _alog = self._audit_logs.get(state.get("thread_id") or "default")
            if _alog:
                try:
                    _alog.add_audit(verdict=_verdict, comments=response[:500], execution_time=elapsed)
                except Exception as e:
                    logger.debug("audit_log audit skipped: %s", e)
            return {
                "response": response, "agent_role": "大元帥",
                "handled_by": "daigensui_audit", "execution_time": elapsed,
                "routed_to": "daigensui_audit", "mcp_tools_used": [],
                "iteration": state.get("iteration", 0) + 1,
                **({"error": "audit_rejected"} if _verdict == "rejected" else {}),
            }
        except Exception as e:
            logger.warning("大元帥監査失敗: %s", e)
            return {"agent_role": "大元帥", "handled_by": "daigensui_audit",
                    "execution_time": time.time() - start, "error": str(e)}

    async def _parallel_groq_node(self, state: "BushidanState") -> dict:
        """メッセージを「?」で分割し各サブクエリを斥候(Groq)で並列実行する。"""
        message = state.get("message", "")
        start   = time.time()
        import re as _re
        sub_queries = [p.strip() for p in _re.split(r"[?？]\s*", message) if p.strip()]

        if len(sub_queries) <= 1:
            exec_fn = self._exec_node("seppou", "groq_qa")
            result  = await exec_fn(state)
            result["routed_to"] = "parallel_groq"
            return result

        role = self._roles.get("seppou")
        if not role:
            return {"response": "⚠️ 斥候ロール未初期化", "handled_by": "parallel_groq",
                    "agent_role": "斥候", "execution_time": 0.0, "error": "role not loaded",
                    "routed_to": "parallel_groq", "mcp_tools_used": [],
                    "sub_queries": sub_queries, "sub_responses": []}

        async def _run_sub(q: str) -> str:
            sub_state = {
                **state,
                "message": q + "?",
                "conversation_history": [],
                "sub_queries":          [],
                "sub_responses":        [],
                "notion_chunks":        list(state.get("notion_chunks", [])),
                "attachments":          list(state.get("attachments", [])),
                "mcp_tools_used":       [],
            }
            try:
                res = await asyncio.wait_for(role.execute(sub_state), timeout=30)
                return res.response
            except Exception as e:
                return f"❌ {q[:30]}: {e}"

        sub_responses = list(await asyncio.gather(*[_run_sub(q) for q in sub_queries[:4]],
                                                   return_exceptions=True))
        sub_responses = [r if isinstance(r, str) else f"❌ エラー: {r}" for r in sub_responses]
        merged  = "\n\n".join(f"**Q: {q}?**\n{a}" for q, a in zip(sub_queries, sub_responses))
        elapsed = time.time() - start
        logger.info("✅ parallel_groq: %.1fs (%d サブクエリ)", elapsed, len(sub_queries))
        return {
            "response": merged, "handled_by": "parallel_groq", "agent_role": "斥候 (並列)",
            "execution_time": elapsed, "error": None, "mcp_tools_used": [],
            "requires_followup": False, "routed_to": "parallel_groq",
            "sub_queries": sub_queries, "sub_responses": sub_responses,
            "conversation_history": [
                {"role": "user",      "content": message},
                {"role": "assistant", "content": merged},
            ],
        }

    async def _run_anthropic_batch(
        self,
        batch_steps: list,
        step_idx: int,
        state: "BushidanState",
        prev_ctx: str,
        roadmap: dict,
    ) -> list:
        """
        Anthropic Batch API で複数ステップを一括送信し、batch_results 形式で返す。
        _execute_step_node から呼び出す内部ヘルパー。
        """
        import os
        from roles.base import RoleResult

        logger.info("📦 [execute_step] Anthropic Batch API 使用: %d steps", len(batch_steps))

        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        poll_iv = float(BATCH_CONFIG.get("anthropic_batch_poll_interval", 5.0))
        max_wt  = float(BATCH_CONFIG.get("anthropic_batch_max_wait", 3600.0))
        processor = AnthropicBatchProcessor(
            api_key=api_key, poll_interval=poll_iv, max_wait=max_wt
        )

        anth_requests = []
        for i, s in enumerate(batch_steps):
            capability = s.get("capability", "analysis")
            role_key   = s.get("assigned_role") or self._CAP_TO_ROLE.get(capability, "gunshi")
            task_desc  = s.get("task", "")
            msgs_content = task_desc
            if prev_ctx:
                msgs_content = (
                    f"ロードマップ目標: {roadmap.get('goal', '')}\n"
                    f"完了済みステップ:\n{prev_ctx}\n\n現タスク: {task_desc}"
                )
            anth_requests.append({
                "custom_id": f"step_{step_idx + i}",
                "role_key":  role_key,
                "messages":  [{"role": "user", "content": msgs_content}],
                "max_tokens": 2000,
            })

        try:
            results_map = await processor.run(anth_requests)
        except Exception as e:
            logger.warning("📦 Anthropic Batch 失敗 → asyncio.gather にフォールバック: %s", e)
            # フォールバック: 通常の並列実行
            timeout_val = NODE_TIMEOUTS.get("execute_step", 120)

            async def _fb_run(s: dict, idx: int):
                capability = s.get("capability", "analysis")
                role_key   = s.get("assigned_role") or self._CAP_TO_ROLE.get(capability, "gunshi")
                role       = self._roles.get(role_key) or self._roles.get("gunshi")
                sub_state  = {
                    **state,
                    "message":            s.get("task", ""),
                    "conversation_history": [],
                    "roadmap_results":    [],
                    "sub_queries":        [],
                    "sub_responses":      [],
                    "notion_chunks":      list(state.get("notion_chunks", [])),
                    "attachments":        list(state.get("attachments", [])),
                    "mcp_tools_used":     [],
                }
                try:
                    res = await asyncio.wait_for(role.execute(sub_state), timeout=timeout_val)
                except asyncio.TimeoutError:
                    res = type("R", (), {
                        "response": f"⏱️ ステップ{idx+1}タイムアウト",
                        "error": "timeout", "mcp_tools_used": [],
                        "agent_role": role_key, "handled_by": "execute_step",
                        "execution_time": timeout_val,
                    })()
                return role_key, capability, s.get("task", ""), res, idx

            raw = await asyncio.gather(*[_fb_run(s, step_idx + i) for i, s in enumerate(batch_steps)],
                                       return_exceptions=True)
            return [
                r if not isinstance(r, BaseException)
                else (batch_steps[ri].get("capability", ""), batch_steps[ri].get("capability", ""),
                      batch_steps[ri].get("task", ""),
                      RoleResult(response=f"❌ {r}", agent_role="unknown",
                                  handled_by="execute_step", error=str(r), status="failed"),
                      step_idx + ri)
                for ri, r in enumerate(raw)
            ]

        # results_map を batch_results 形式に変換
        batch_results = []
        for i, s in enumerate(batch_steps):
            cid        = f"step_{step_idx + i}"
            capability = s.get("capability", "analysis")
            role_key   = s.get("assigned_role") or self._CAP_TO_ROLE.get(capability, "gunshi")
            task_desc  = s.get("task", "")
            text       = results_map.get(cid, "❌ 結果なし")
            is_error   = text.startswith("❌")
            fake_result = RoleResult(
                response=text,
                agent_role=role_key,
                handled_by="anthropic_batch",
                error=text if is_error else None,
                status="failed" if is_error else "completed",
            )
            batch_results.append((role_key, capability, task_desc, fake_result, step_idx + i))

        return batch_results

    async def _batch_parallel_node(self, state: "BushidanState") -> dict:
        """
        BATCH モード専用の並列実行ノード。

        ロードマップ内の全 can_parallel ステップを上限なしで同時実行する。
        BATCH_CONFIG["max_parallel_batch_steps"] == 0 の場合は全ステップを一括処理。
        Anthropic モデルのステップは AnthropicBatchProcessor で一括送信。
        non-Anthropic ステップは asyncio.gather で並列実行。
        """
        roadmap  = state.get("roadmap", {})
        steps    = roadmap.get("steps", [])
        start    = time.time()

        if not steps:
            return {"roadmap_step": 0, "execution_time": 0.0}

        max_steps = BATCH_CONFIG.get("max_parallel_batch_steps", 0)
        target_steps = steps if max_steps == 0 else steps[:max_steps]

        # Anthropic ステップと非 Anthropic ステップに分割
        anth_steps   = [(i, s) for i, s in enumerate(target_steps)
                        if (s.get("assigned_role") or self._CAP_TO_ROLE.get(s.get("capability", ""), ""))
                        in ANTHROPIC_ROLES]
        other_steps  = [(i, s) for i, s in enumerate(target_steps)
                        if (i, s) not in anth_steps]

        timeout_val = NODE_TIMEOUTS.get("execute_step", 120)
        message     = state.get("message", "")
        all_results: dict[int, tuple] = {}

        # ── 非 Anthropic ステップ: asyncio.gather ──────────────────────────
        if other_steps:
            async def _run(idx: int, s: dict):
                capability = s.get("capability", "analysis")
                role_key   = s.get("assigned_role") or self._CAP_TO_ROLE.get(capability, "gunshi")
                role       = self._roles.get(role_key) or self._roles.get("gunshi")
                sub_state  = {
                    **state,
                    "message":            s.get("task", ""),
                    "conversation_history": [],
                    "roadmap_results":    [],
                    "sub_queries":        [],
                    "sub_responses":      [],
                    "notion_chunks":      list(state.get("notion_chunks", [])),
                    "attachments":        list(state.get("attachments", [])),
                    "mcp_tools_used":     [],
                }
                try:
                    res = await asyncio.wait_for(role.execute(sub_state), timeout=timeout_val)
                except asyncio.TimeoutError:
                    from roles.base import RoleResult
                    res = RoleResult(response=f"⏱️ Step{idx+1}タイムアウト",
                                     agent_role=role_key, handled_by="batch_parallel",
                                     error="timeout", status="failed")
                return idx, role_key, capability, s.get("task", ""), res

            logger.info("⚡ [batch_parallel] asyncio.gather: %d steps", len(other_steps))
            gathered = await asyncio.gather(*[_run(i, s) for i, s in other_steps], return_exceptions=True)
            for gi, raw in enumerate(gathered):
                if isinstance(raw, BaseException):
                    idx_o, s_o = other_steps[gi]
                    from roles.base import RoleResult
                    all_results[idx_o] = (idx_o, "unknown", s_o.get("capability", ""),
                                          s_o.get("task", ""),
                                          RoleResult(response=f"❌ {raw}", agent_role="unknown",
                                                     handled_by="batch_parallel", error=str(raw), status="failed"))
                else:
                    idx_r, role_key_r, cap_r, task_r, res_r = raw
                    all_results[idx_r] = (idx_r, role_key_r, cap_r, task_r, res_r)

        # ── Anthropic ステップ: Batch API ─────────────────────────────────
        if anth_steps and BATCH_CONFIG.get("use_anthropic_batch", False):
            try:
                import os
                processor = AnthropicBatchProcessor(
                    api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
                    poll_interval=float(BATCH_CONFIG.get("anthropic_batch_poll_interval", 5.0)),
                    max_wait=float(BATCH_CONFIG.get("anthropic_batch_max_wait", 3600.0)),
                )
                anth_requests = [
                    {
                        "custom_id": f"bp_{i}",
                        "role_key":  s.get("assigned_role") or self._CAP_TO_ROLE.get(s.get("capability", ""), "shogun"),
                        "messages":  [{"role": "user", "content": s.get("task", "")}],
                        "max_tokens": 2000,
                    }
                    for i, s in anth_steps
                ]
                logger.info("📦 [batch_parallel] Anthropic Batch API: %d steps", len(anth_steps))
                res_map = await processor.run(anth_requests)
                for req, (idx_a, s_a) in zip(anth_requests, anth_steps):
                    from roles.base import RoleResult
                    cid  = req["custom_id"]
                    text = res_map.get(cid, "❌ 結果なし")
                    rk   = req["role_key"]
                    cap  = s_a.get("capability", "analysis")
                    all_results[idx_a] = (
                        idx_a, rk, cap, s_a.get("task", ""),
                        RoleResult(response=text, agent_role=rk,
                                   handled_by="anthropic_batch",
                                   error=text if text.startswith("❌") else None,
                                   status="failed" if text.startswith("❌") else "completed"),
                    )
            except Exception as e:
                logger.warning("📦 Anthropic Batch 失敗 → asyncio.gather にフォールバック: %s", e)
                # フォールバック
                async def _run_anth(idx: int, s: dict):
                    rk   = s.get("assigned_role") or self._CAP_TO_ROLE.get(s.get("capability", ""), "shogun")
                    role = self._roles.get(rk) or self._roles.get("gunshi")
                    sub  = {
                        **state,
                        "message":            s.get("task", ""),
                        "conversation_history": [],
                        "roadmap_results":    [],
                        "sub_queries":        [],
                        "sub_responses":      [],
                        "notion_chunks":      list(state.get("notion_chunks", [])),
                        "attachments":        list(state.get("attachments", [])),
                        "mcp_tools_used":     [],
                    }
                    try:
                        res = await asyncio.wait_for(role.execute(sub), timeout=timeout_val)
                    except asyncio.TimeoutError:
                        from roles.base import RoleResult
                        res = RoleResult(response=f"⏱️ Step{idx+1}タイムアウト",
                                         agent_role=rk, handled_by="batch_parallel",
                                         error="timeout", status="failed")
                    return idx, rk, s.get("capability", ""), s.get("task", ""), res

                fb_raw = await asyncio.gather(*[_run_anth(i, s) for i, s in anth_steps], return_exceptions=True)
                for gi, raw in enumerate(fb_raw):
                    if isinstance(raw, BaseException):
                        idx_a, s_a = anth_steps[gi]
                        from roles.base import RoleResult
                        all_results[idx_a] = (idx_a, "unknown", s_a.get("capability", ""),
                                              s_a.get("task", ""),
                                              RoleResult(response=f"❌ {raw}", agent_role="unknown",
                                                         handled_by="batch_parallel", error=str(raw), status="failed"))
                    else:
                        idx_r, rk_r, cap_r, task_r, res_r = raw
                        all_results[idx_r] = (idx_r, rk_r, cap_r, task_r, res_r)
        elif anth_steps:
            # Batch API 無効: asyncio.gather
            async def _run_anth_direct(idx: int, s: dict):
                rk   = s.get("assigned_role") or self._CAP_TO_ROLE.get(s.get("capability", ""), "shogun")
                role = self._roles.get(rk) or self._roles.get("gunshi")
                sub  = {
                    **state,
                    "message":            s.get("task", ""),
                    "conversation_history": [],
                    "roadmap_results":    [],
                    "sub_queries":        [],
                    "sub_responses":      [],
                    "notion_chunks":      list(state.get("notion_chunks", [])),
                    "attachments":        list(state.get("attachments", [])),
                    "mcp_tools_used":     [],
                }
                try:
                    res = await asyncio.wait_for(role.execute(sub), timeout=timeout_val)
                except asyncio.TimeoutError:
                    from roles.base import RoleResult
                    res = RoleResult(response=f"⏱️ Step{idx+1}タイムアウト",
                                     agent_role=rk, handled_by="batch_parallel",
                                     error="timeout", status="failed")
                return idx, rk, s.get("capability", ""), s.get("task", ""), res

            ad_raw = await asyncio.gather(*[_run_anth_direct(i, s) for i, s in anth_steps], return_exceptions=True)
            for gi, raw in enumerate(ad_raw):
                if isinstance(raw, BaseException):
                    idx_a, s_a = anth_steps[gi]
                    from roles.base import RoleResult
                    all_results[idx_a] = (idx_a, "unknown", s_a.get("capability", ""),
                                          s_a.get("task", ""),
                                          RoleResult(response=f"❌ {raw}", agent_role="unknown",
                                                     handled_by="batch_parallel", error=str(raw), status="failed"))
                else:
                    idx_r, rk_r, cap_r, task_r, res_r = raw
                    all_results[idx_r] = (idx_r, rk_r, cap_r, task_r, res_r)

        # 結果を index 順に並べて state 更新を構築
        ordered = [all_results[i] for i in sorted(all_results.keys())]
        step_results_list = []
        updated_steps = list(steps)
        all_mcp_used  = []

        for idx_r, rk_r, cap_r, task_r, res_r in ordered:
            step_results_list.append({
                "step_id":   idx_r + 1,
                "capability": cap_r,
                "role":       rk_r,
                "task":       task_r,
                "response":   res_r.response,
                "summary":    res_r.response[:200] if res_r.response else "",
                "error":      res_r.error,
            })
            if idx_r < len(updated_steps):
                updated_steps[idx_r] = dict(steps[idx_r])
                updated_steps[idx_r]["status"] = "done" if not res_r.error else "error"
                updated_steps[idx_r]["result"] = (res_r.response or "")[:300]
            all_mcp_used.extend(getattr(res_r, "mcp_tools_used", []))

        updated_roadmap = dict(roadmap)
        updated_roadmap["steps"] = updated_steps
        merged_response = "\n\n---\n\n".join(
            r["response"] for r in step_results_list if r.get("response")
        )
        elapsed = time.time() - start
        logger.info("✅ [batch_parallel] 完了: %d steps %.2fs", len(ordered), elapsed)

        return {
            "roadmap":        updated_roadmap,
            "roadmap_step":   len(target_steps),
            "roadmap_results": step_results_list,
            "response":        merged_response,
            "agent_role":      ordered[-1][1] if ordered else "unknown",
            "handled_by":      "batch_parallel",
            "routed_to":       f"batch_parallel[0~{len(target_steps)}]",
            "execution_time":  elapsed,
            "mcp_tools_used":  all_mcp_used,
            "error":           None,
            "conversation_history": [
                {"role": "user",      "content": message},
                {"role": "assistant", "content": merged_response},
            ],
        }

    async def _sandbox_verify_node(self, state: "BushidanState") -> dict:
        """生成コードブロックを抽出して実行検証する。BATCH モードはスキップ。"""
        mode = ProcessingMode(state.get("processing_mode", ProcessingMode.INTERACTIVE))
        if mode == ProcessingMode.BATCH and not BATCH_CONFIG.get("sandbox_verify_enabled", True):
            return {"code_verified": False, "code_verify_result": "skipped_batch"}

        response = state.get("response", "") or ""
        if "```" not in response:
            return {"code_verified": False, "code_verify_result": "skipped"}
        try:
            from core.code_verifier import verify_response, append_verify_note
            verify_result   = await asyncio.wait_for(verify_response(response, max_blocks=2), timeout=25)
            updated_response = append_verify_note(response, verify_result)
            return {"response": updated_response, "code_verified": True,
                    "code_verify_result": verify_result}
        except asyncio.TimeoutError:
            return {"response": response, "code_verified": False, "code_verify_result": "timeout"}
        except Exception as e:
            logger.debug("[sandbox_verify] スキップ: %s", e)
            return {"response": response, "code_verified": False, "code_verify_result": f"error: {e}"}
