"""
core/router/mixins/postprocess.py — 後処理ノード Mixin

check_followup / followup_decision / human_interrupt / notion_store /
batch_completion_notify / batch_bulk_notion_store
バッチモードでの動作差分を含む。
"""
import asyncio
import os
from typing import TYPE_CHECKING, Literal
from utils.logger import get_logger
from core.router.batch.mode import ProcessingMode, BATCH_CONFIG
from core.router.constants import fire

logger = get_logger(__name__)

if TYPE_CHECKING:
    from core.state import BushidanState


class PostprocessMixin:

    async def _check_followup(self, state: "BushidanState") -> dict:
        """自律ループ判定。BATCH モードでは強制終了する。"""
        mode = ProcessingMode(state.get("processing_mode", ProcessingMode.INTERACTIVE))

        # BATCH: 自律ループ無効・HITL 無効
        if mode == ProcessingMode.BATCH:
            return {"requires_followup": False, "awaiting_human_input": False}

        updates: dict = {}
        iteration = state.get("iteration", 0) + 1
        max_iter  = state.get("max_iterations", 3)
        updates["iteration"] = iteration

        if state.get("awaiting_human_input", False):
            return updates

        if not state.get("requires_followup", False):
            return updates

        # 最大反復回数に達した場合のみ強制終了
        if iteration >= max_iter:
            logger.info("⚠️ 最大反復回数到達 (%d) — 強制終了", max_iter)
            updates["requires_followup"] = False

        return updates

    def _followup_decision(self, state: "BushidanState") -> Literal["human", "loop", "done"]:
        """BATCH モードでは "human" を "done" に変換する（ユーザー待機なし）。"""
        mode = ProcessingMode(state.get("processing_mode", ProcessingMode.INTERACTIVE))

        if state.get("awaiting_human_input", False):
            if mode == ProcessingMode.BATCH or not BATCH_CONFIG["hitl_enabled"]:
                logger.info("🤖 [BATCH] HITL スキップ → done")
                return "done"
            return "human"

        if state.get("requires_followup", False):
            return "loop"

        return "done"

    async def _human_interrupt(self, state: "BushidanState") -> dict:
        """Human-in-the-loop: LangGraph interrupt() でグラフを一時停止する。"""
        from langgraph.types import interrupt
        question = state.get("human_question", "")
        logger.info("🙋 HITL: 人間の入力を待機中 — %s", question[:60])
        response = interrupt({"question": question})
        return {
            "human_response":       response,
            "dialog_status":        "active",
            "awaiting_human_input": False,
        }

    async def _notion_store(self, state: "BushidanState") -> dict:
        """Notion に結果を保存する。BATCH モードでは同期的に await する。"""
        if not state.get("should_save", True):
            return {"notion_page_id": None}

        mode = ProcessingMode(state.get("processing_mode", ProcessingMode.INTERACTIVE))

        try:
            from integrations.notion.storage import save_task_result_bg
            if mode == ProcessingMode.BATCH and BATCH_CONFIG["notion_store_sync"]:
                # BATCH: 同期保存（完了確認）+ バッチ完了通知
                await save_task_result_bg(dict(state))
                # バッチジョブが指定されている場合は完了通知も発火
                if state.get("batch_job_id") and BATCH_CONFIG.get("notify_on_completion", True):
                    fire(
                        self._batch_completion_notify(state),
                        name=f"batch_notify_{state.get('batch_job_id', '')[:12]}",
                    )
                return {"notion_page_id": "saved"}
            else:
                # INTERACTIVE: fire-and-forget
                fire(
                    save_task_result_bg(dict(state)),
                    name=f"notion_store_{state.get('thread_id', '')[:8]}",
                )
                return {"notion_page_id": "pending"}
        except Exception as e:
            logger.warning("[notion_store] スキップ: %s", e)
            return {"notion_page_id": None}

    async def _batch_completion_notify(self, state: "BushidanState") -> None:
        """バッチジョブ完了後の通知。BATCH_NOTIFY_WEBHOOK が設定されていれば HTTP POST。"""
        batch_job_id  = state.get("batch_job_id", "")
        thread_id     = state.get("thread_id", "")
        response_text = state.get("response", "") or ""
        handled_by    = state.get("handled_by", "")
        exec_time     = state.get("execution_time", 0.0)
        error         = state.get("error")

        summary = (
            f"✅ バッチジョブ完了: {batch_job_id}\n"
            f"thread={thread_id}  handler={handled_by}  time={exec_time:.1f}s\n"
            + (f"❌ エラー: {error}" if error else f"応答先頭: {response_text[:120]}")
        )
        logger.info("📢 [batch_notify] %s", summary.replace("\n", " | "))

        webhook_url = os.environ.get("BATCH_NOTIFY_WEBHOOK", "")
        if webhook_url:
            try:
                import aiohttp
                payload = {
                    "batch_job_id": batch_job_id,
                    "thread_id":    thread_id,
                    "status":       "error" if error else "completed",
                    "handled_by":   handled_by,
                    "execution_time": exec_time,
                    "summary":      summary,
                }
                async with aiohttp.ClientSession() as session:
                    async with session.post(webhook_url, json=payload, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                        if resp.status < 300:
                            logger.info("📢 Webhook 通知成功: status=%d", resp.status)
                        else:
                            logger.warning("📢 Webhook 通知失敗: status=%d", resp.status)
            except Exception as e:
                logger.warning("📢 Webhook 通知エラー: %s", e)

    async def _crosscheck_node(self, state: "BushidanState") -> dict:
        """複雑度に応じた段階的クロスチェック。

        medium            -> crosscheck_light (Cerebras Llama3.1-8B) で検出
                             -> 問題あり -> crosscheck_heavy (Gemini 3.1 Pro) で修正
        complex/strategic -> crosscheck_heavy (Gemini 3.1 Pro) で検出・修正
        simple            -> スキップ
        """
        complexity = state.get("complexity", "simple")
        if complexity == "simple":
            return {}

        response = state.get("response", "") or ""
        if not response:
            return {}

        message = state.get("message", "") or ""
        is_medium = complexity == "medium"
        checker_key = "crosscheck_light" if is_medium else "crosscheck_heavy"

        try:
            from utils.client_registry import ClientRegistry
            registry = ClientRegistry.get()
            checker = registry.get_client(checker_key)
            if not checker:
                logger.debug("[crosscheck] %s クライアント未取得 — スキップ", checker_key)
                return {}

            check_prompt = (
                "以下の回答に事実の誤り・論理矛盾・重要な見落としがあれば指摘してください。\n"
                "問題なければ「問題なし」とだけ答えてください。\n\n"
                "【質問】" + message[:400] + "\n\n【回答】" + response[:800]
            )
            verdict = await checker.generate(
                messages=[{"role": "user", "content": check_prompt}],
                system="批判的検証AI。事実誤認・論理矛盾・重要な見落としのみ指摘。余分な説明不要。",
                max_tokens=400,
            )

            if not verdict or "問題なし" in verdict:
                return {}

            logger.info(
                "🔍 [crosscheck] 問題検出 (complexity=%s checker=%s): %s",
                complexity, checker_key, verdict[:80],
            )

            # 問題あり → crosscheck_heavy (Gemini 3.1 Pro) で修正
            verifier = registry.get_client("crosscheck_heavy")
            if not verifier:
                return {"response": response + "\n\n⚠️ **要確認**: " + verdict.strip()[:200]}

            correct_prompt = (
                "以下の回答に誤りが指摘されました。正しい回答を日本語で提示してください。\n\n"
                "【質問】" + message[:400] + "\n\n"
                "【元の回答】" + response[:800] + "\n\n"
                "【指摘内容】" + verdict[:300]
            )
            corrected = await verifier.generate(
                messages=[{"role": "user", "content": correct_prompt}],
                system="修正AI。指摘を踏まえて正確・簡潔な回答を日本語で提示してください。",
                max_tokens=2000,
            )
            if corrected:
                logger.info("🔍 [crosscheck] 修正完了 (complexity=%s)", complexity)
                return {
                    "response": corrected,
                    "mcp_tools_used": (state.get("mcp_tools_used") or []) + [checker_key],
                }

        except Exception as e:
            logger.debug("[crosscheck] スキップ: %s", e)

        return {}


    async def batch_bulk_notion_store(self, states: list) -> dict:
        """
        バッチ処理で生成された複数スレッドの結果を一括で Notion に保存する。

        MessagingMixin.process_message を BATCH モードで複数回呼び出した後、
        バッチランナーから直接呼び出す用途を想定。

        Args:
            states: BushidanState dict のリスト

        Returns:
            {"saved": int, "failed": int, "errors": list[str]}
        """
        saved  = 0
        failed = 0
        errors: list[str] = []

        try:
            from integrations.notion.storage import save_task_result_bg
        except ImportError as e:
            logger.warning("batch_bulk_notion_store: notion モジュール未ロード: %s", e)
            return {"saved": 0, "failed": len(states), "errors": [str(e)]}

        async def _save_one(state: dict) -> bool:
            try:
                if not state.get("should_save", True):
                    return False
                await save_task_result_bg(state)
                return True
            except Exception as e_inner:
                tid = state.get("thread_id", "?")
                errors.append(f"thread={tid}: {e_inner}")
                logger.warning("batch_bulk_notion_store 個別失敗: %s", e_inner)
                return False

        # 並列保存（最大 5 並列）
        semaphore = asyncio.Semaphore(5)

        async def _guarded(s: dict) -> bool:
            async with semaphore:
                return await _save_one(s)

        results = await asyncio.gather(*[_guarded(s) for s in states], return_exceptions=True)
        for r in results:
            if r is True:
                saved += 1
            else:
                failed += 1
                if isinstance(r, BaseException):
                    errors.append(str(r))

        logger.info("📋 batch_bulk_notion_store: saved=%d failed=%d", saved, failed)
        return {"saved": saved, "failed": failed, "errors": errors}
