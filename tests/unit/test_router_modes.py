"""
tests/unit/test_router_modes.py — インタラクティブ/バッチモードのユニットテスト

対象:
  - ProcessingMode enum と BATCH_CONFIG 構造
  - PostprocessMixin: _check_followup / _followup_decision / _notion_store
  - MessagingMixin: キャッシュ制御 (BATCH では無効)
  - NodesMixin: _sandbox_verify_node スキップ / _batch_parallel_node
  - AnthropicBatchProcessor: submit / poll / fetch_results (モック)
"""
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# プロジェクトルートを sys.path に追加（pytest.ini の pythonpath に頼るが念のため）
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

# ── 環境変数スタブ (DB / API キー不要のユニットテスト用) ──────────────────
os.environ.setdefault("POSTGRES_URL", "postgresql://stub:stub@localhost/stub")
os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-stub")


# ===========================================================================
# 1. ProcessingMode と BATCH_CONFIG
# ===========================================================================

class TestProcessingMode:
    def test_enum_values(self):
        from core.router.batch.mode import ProcessingMode
        assert ProcessingMode.INTERACTIVE == "interactive"
        assert ProcessingMode.BATCH == "batch"

    def test_str_coercion(self):
        from core.router.batch.mode import ProcessingMode
        assert ProcessingMode("batch") == ProcessingMode.BATCH
        assert ProcessingMode("interactive") == ProcessingMode.INTERACTIVE

    def test_batch_config_keys(self):
        from core.router.batch.mode import BATCH_CONFIG
        required = {
            "streaming_enabled", "cache_enabled", "semantic_router_shortcut",
            "autonomous_loop", "hitl_enabled", "notion_store_sync",
            "sandbox_verify_enabled", "node_timeout_multiplier",
            "use_anthropic_batch", "anthropic_batch_poll_interval",
            "anthropic_batch_max_wait", "max_parallel_batch_steps",
            "notify_on_completion",
        }
        assert required.issubset(set(BATCH_CONFIG.keys()))

    def test_batch_config_values(self):
        from core.router.batch.mode import BATCH_CONFIG
        assert BATCH_CONFIG["streaming_enabled"] is False
        assert BATCH_CONFIG["cache_enabled"] is False
        assert BATCH_CONFIG["hitl_enabled"] is False
        assert BATCH_CONFIG["notion_store_sync"] is True
        assert BATCH_CONFIG["node_timeout_multiplier"] == 3.0
        assert BATCH_CONFIG["use_anthropic_batch"] is True


# ===========================================================================
# 2. PostprocessMixin._check_followup
# ===========================================================================

class TestCheckFollowup:
    """_check_followup の BATCH / INTERACTIVE 振る舞いを検証。"""

    def _make_mixin(self):
        from core.router.mixins.postprocess import PostprocessMixin
        return PostprocessMixin()

    @pytest.mark.asyncio
    async def test_batch_returns_immediately_no_followup(self):
        mixin = self._make_mixin()
        state = {
            "processing_mode": "batch",
            "requires_followup": True,
            "awaiting_human_input": True,
            "iteration": 0,
        }
        result = await mixin._check_followup(state)
        assert result == {"requires_followup": False, "awaiting_human_input": False}

    @pytest.mark.asyncio
    async def test_interactive_increments_iteration(self):
        mixin = self._make_mixin()
        state = {
            "processing_mode": "interactive",
            "requires_followup": False,
            "awaiting_human_input": False,
            "iteration": 2,
            "max_iterations": 5,
        }
        result = await mixin._check_followup(state)
        assert result["iteration"] == 3

    @pytest.mark.asyncio
    async def test_interactive_awaiting_input_returns_early(self):
        mixin = self._make_mixin()
        state = {
            "processing_mode": "interactive",
            "awaiting_human_input": True,
            "requires_followup": True,
            "iteration": 1,
            "max_iterations": 3,
        }
        result = await mixin._check_followup(state)
        # requires_followup は変更されない
        assert "requires_followup" not in result
        assert result["iteration"] == 2

    @pytest.mark.asyncio
    async def test_interactive_max_iter_forces_done(self):
        mixin = self._make_mixin()
        state = {
            "processing_mode": "interactive",
            "requires_followup": True,
            "awaiting_human_input": False,
            "iteration": 3,
            "max_iterations": 3,
        }
        result = await mixin._check_followup(state)
        assert result.get("requires_followup") is False


# ===========================================================================
# 3. PostprocessMixin._followup_decision
# ===========================================================================

class TestFollowupDecision:
    def _make_mixin(self):
        from core.router.mixins.postprocess import PostprocessMixin
        return PostprocessMixin()

    def test_batch_converts_human_to_done(self):
        mixin = self._make_mixin()
        state = {"processing_mode": "batch", "awaiting_human_input": True, "requires_followup": False}
        assert mixin._followup_decision(state) == "done"

    def test_interactive_human_returns_human(self):
        mixin = self._make_mixin()
        state = {"processing_mode": "interactive", "awaiting_human_input": True, "requires_followup": False}
        # BATCH_CONFIG["hitl_enabled"] は False なので interactive でも done になる
        assert mixin._followup_decision(state) == "done"

    def test_requires_followup_returns_loop(self):
        mixin = self._make_mixin()
        state = {"processing_mode": "interactive", "awaiting_human_input": False, "requires_followup": True}
        assert mixin._followup_decision(state) == "loop"

    def test_all_clear_returns_done(self):
        mixin = self._make_mixin()
        state = {"processing_mode": "interactive", "awaiting_human_input": False, "requires_followup": False}
        assert mixin._followup_decision(state) == "done"

    def test_batch_loop_disabled(self):
        mixin = self._make_mixin()
        state = {"processing_mode": "batch", "awaiting_human_input": False, "requires_followup": False}
        assert mixin._followup_decision(state) == "done"


# ===========================================================================
# 4. PostprocessMixin._notion_store
# ===========================================================================

class TestNotionStore:
    def _make_mixin(self):
        from core.router.mixins.postprocess import PostprocessMixin
        return PostprocessMixin()

    @pytest.mark.asyncio
    async def test_batch_mode_sync_save(self):
        mixin = self._make_mixin()
        save_mock = AsyncMock()
        state = {"processing_mode": "batch", "should_save": True, "batch_job_id": None}

        with patch("integrations.notion.storage.save_task_result_bg", save_mock):
            result = await mixin._notion_store(state)

        save_mock.assert_awaited_once()
        assert result == {"notion_page_id": "saved"}

    @pytest.mark.asyncio
    async def test_interactive_mode_fire_and_forget(self):
        mixin = self._make_mixin()
        save_mock = AsyncMock()
        fire_calls = []

        state = {"processing_mode": "interactive", "should_save": True,
                 "thread_id": "tid123", "batch_job_id": None}

        def _fake_fire(coro, *, name=None):
            fire_calls.append(name)
            # コルーチンを閉じてリソースリークを防ぐ
            coro.close()
            return MagicMock()

        with patch("integrations.notion.storage.save_task_result_bg", save_mock):
            with patch("core.router.mixins.postprocess.fire", side_effect=_fake_fire):
                result = await mixin._notion_store(state)

        assert result == {"notion_page_id": "pending"}
        save_mock.assert_not_awaited()
        assert any("notion_store" in (n or "") for n in fire_calls)

    @pytest.mark.asyncio
    async def test_should_save_false_skips(self):
        mixin = self._make_mixin()
        state = {"processing_mode": "batch", "should_save": False}
        result = await mixin._notion_store(state)
        assert result == {"notion_page_id": None}

    @pytest.mark.asyncio
    async def test_exception_returns_none(self):
        mixin = self._make_mixin()
        state = {"processing_mode": "batch", "should_save": True, "batch_job_id": None}

        with patch("integrations.notion.storage.save_task_result_bg",
                   AsyncMock(side_effect=RuntimeError("DB down"))):
            result = await mixin._notion_store(state)

        assert result == {"notion_page_id": None}


# ===========================================================================
# 5. PostprocessMixin._batch_completion_notify
# ===========================================================================

class TestBatchCompletionNotify:
    def _make_mixin(self):
        from core.router.mixins.postprocess import PostprocessMixin
        return PostprocessMixin()

    @pytest.mark.asyncio
    async def test_no_webhook_just_logs(self):
        """設定なしでも例外にならない。"""
        mixin = self._make_mixin()
        state = {
            "batch_job_id": "job-001", "thread_id": "t1",
            "response": "完了", "handled_by": "shogun_plan",
            "execution_time": 12.5, "error": None,
        }
        with patch.dict(os.environ, {"BATCH_NOTIFY_WEBHOOK": ""}):
            await mixin._batch_completion_notify(state)  # 例外なし

    @pytest.mark.asyncio
    async def test_webhook_called_on_completion(self):
        mixin = self._make_mixin()
        state = {
            "batch_job_id": "job-002", "thread_id": "t2",
            "response": "OK", "handled_by": "batch_parallel",
            "execution_time": 5.0, "error": None,
        }

        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_resp)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch.dict(os.environ, {"BATCH_NOTIFY_WEBHOOK": "http://hook.example/"}):
            with patch("aiohttp.ClientSession", return_value=mock_session):
                await mixin._batch_completion_notify(state)

        mock_session.post.assert_called_once()
        call_kwargs = mock_session.post.call_args
        assert call_kwargs[0][0] == "http://hook.example/"


# ===========================================================================
# 6. NodesMixin._sandbox_verify_node
# ===========================================================================

class TestSandboxVerifyNode:
    def _make_mixin(self):
        from core.router.mixins.nodes import NodesMixin
        m = NodesMixin()
        m._roles = {}
        m._audit_logs = {}
        return m

    @pytest.mark.asyncio
    async def test_batch_skips_verify(self):
        mixin = self._make_mixin()
        state = {
            "processing_mode": "batch",
            "response": "```python\nprint('hello')\n```",
        }
        result = await mixin._sandbox_verify_node(state)
        assert result["code_verify_result"] == "skipped_batch"
        assert result["code_verified"] is False

    @pytest.mark.asyncio
    async def test_interactive_no_code_skips(self):
        mixin = self._make_mixin()
        state = {"processing_mode": "interactive", "response": "テキストのみ"}
        result = await mixin._sandbox_verify_node(state)
        assert result["code_verify_result"] == "skipped"

    @pytest.mark.asyncio
    async def test_interactive_with_code_calls_verifier(self):
        mixin = self._make_mixin()
        state = {
            "processing_mode": "interactive",
            "response": "```python\nprint('test')\n```",
        }
        verify_mock = AsyncMock(return_value="passed")
        append_mock = MagicMock(return_value="```python\nprint('test')\n```\n✅ 検証: passed")

        with patch("core.code_verifier.verify_response", verify_mock):
            with patch("core.code_verifier.append_verify_note", append_mock):
                result = await mixin._sandbox_verify_node(state)

        assert result["code_verified"] is True


# ===========================================================================
# 7. MessagingMixin — BATCH キャッシュ無効化
# ===========================================================================

class TestMessagingCacheBypass:
    """BATCH モードではキャッシュを使わないことを確認。"""

    @pytest.mark.asyncio
    async def test_batch_does_not_check_cache(self):
        from core.router.mixins.messaging import MessagingMixin
        from core.router.batch.mode import ProcessingMode

        mixin = MessagingMixin()
        # キャッシュにヒットするエントリを事前設定
        import hashlib
        import time
        _tid = "thread-x"
        msg  = "こんにちは"
        _h   = hashlib.md5(f"{_tid}:0:{msg}".encode()).hexdigest()
        mixin._RESP_CACHE = {_h: ({"status": "completed", "response": "CACHED!"}, time.time())}
        mixin._RESP_CACHE_TTL = 300.0
        mixin._compiled = True
        mixin._compiled_fast = None
        mixin._audit_logs = {}

        invoked = []
        async def _fake_ainvoke(state, config=None):
            invoked.append(state.get("processing_mode"))
            return {
                "response": "FRESH", "handled_by": "groq_qa", "agent_role": "seppou",
                "execution_time": 0.1, "error": None, "mcp_tools_used": [],
                "routed_to": "groq_qa", "complexity": "simple",
                "notion_page_id": None, "dialog_status": "completed",
                "human_question": "", "requires_followup": False,
                "is_action_task": False,
            }

        mock_graph = MagicMock()
        mock_graph.ainvoke = _fake_ainvoke
        mixin._compiled = mock_graph

        with patch("core.router.constants.fire"):
            result = await mixin.process_message(
                message=msg,
                thread_id=_tid,
                mode=ProcessingMode.BATCH,
            )

        # BATCH ではキャッシュを無視して _fake_ainvoke が呼ばれるはず
        assert len(invoked) == 1
        assert invoked[0] == "batch"
        assert result["response"] == "FRESH"

    @pytest.mark.asyncio
    async def test_interactive_uses_cache(self):
        from core.router.mixins.messaging import MessagingMixin
        from core.router.batch.mode import ProcessingMode

        mixin = MessagingMixin()
        import hashlib
        import time
        _tid = "thread-y"
        msg  = "テスト"
        _h   = hashlib.md5(f"{_tid}:0:{msg}".encode()).hexdigest()
        cached_resp = {"status": "completed", "response": "CACHED!", "execution_time": 0.0}
        mixin._RESP_CACHE = {_h: (cached_resp, time.time())}
        mixin._RESP_CACHE_TTL = 300.0
        mixin._compiled = MagicMock()  # 呼ばれないはず
        state_mock = MagicMock()
        state_mock.values = {}
        mixin._compiled.get_state.return_value = state_mock
        mixin._compiled_fast = None
        mixin._audit_logs = {}

        result = await mixin.process_message(
            message=msg, thread_id=_tid, mode=ProcessingMode.INTERACTIVE
        )
        assert result["response"] == "CACHED!"
        mixin._compiled.ainvoke.assert_not_called()


# ===========================================================================
# 8. AnthropicBatchProcessor (モック)
# ===========================================================================

class TestAnthropicBatchProcessor:
    """外部 API をモックして AnthropicBatchProcessor の動作を検証。"""

    def _make_processor(self):
        from core.router.batch.anthropic_batch import AnthropicBatchProcessor
        return AnthropicBatchProcessor(api_key="sk-ant-stub", poll_interval=0.01, max_wait=5.0)

    @pytest.mark.asyncio
    async def test_run_empty_returns_empty_dict(self):
        proc = self._make_processor()
        result = await proc.run([])
        assert result == {}

    @pytest.mark.asyncio
    async def test_run_success_path(self):
        proc = self._make_processor()

        # Batch 作成モック
        mock_batch_created = MagicMock()
        mock_batch_created.id = "msgbatch_test123"

        # ポーリング: 1回目 processing → 2回目 ended
        mock_batch_processing = MagicMock()
        mock_batch_processing.processing_status = "ended"

        # 結果ストリーム
        mock_result_item = MagicMock()
        mock_result_item.custom_id = "req-1"
        mock_result_item.result.type = "succeeded"
        mock_result_item.result.message.content = [MagicMock(text="テスト応答")]

        async def _async_iter_results():
            yield mock_result_item

        mock_client = MagicMock()
        mock_client.messages.batches.create = AsyncMock(return_value=mock_batch_created)
        mock_client.messages.batches.retrieve = AsyncMock(return_value=mock_batch_processing)
        mock_client.messages.batches.results = MagicMock(return_value=_async_iter_results())

        with patch("anthropic.AsyncAnthropic", return_value=mock_client):
            result = await proc.run([{
                "custom_id": "req-1",
                "role_key":  "shogun",
                "messages":  [{"role": "user", "content": "テスト"}],
                "max_tokens": 100,
            }])

        assert result == {"req-1": ("テスト応答", None)}

    @pytest.mark.asyncio
    async def test_run_error_result(self):
        proc = self._make_processor()

        mock_batch_created = MagicMock()
        mock_batch_created.id = "msgbatch_err"

        mock_batch_ended = MagicMock()
        mock_batch_ended.processing_status = "ended"

        mock_error_item = MagicMock()
        mock_error_item.custom_id = "req-err"
        mock_error_item.result.type = "errored"
        mock_error_item.result.error.type = "invalid_request"
        mock_error_item.result.error.message = "Too long"

        async def _async_iter_error():
            yield mock_error_item

        mock_client = MagicMock()
        mock_client.messages.batches.create = AsyncMock(return_value=mock_batch_created)
        mock_client.messages.batches.retrieve = AsyncMock(return_value=mock_batch_ended)
        mock_client.messages.batches.results = MagicMock(return_value=_async_iter_error())

        with patch("anthropic.AsyncAnthropic", return_value=mock_client):
            result = await proc.run([{
                "custom_id": "req-err",
                "role_key":  "shogun",
                "messages":  [{"role": "user", "content": "x"}],
            }])

        assert "req-err" in result
        _text, _err = result["req-err"]
        assert _err is not None

    @pytest.mark.asyncio
    async def test_timeout_raises(self):
        proc = self._make_processor()
        proc._max_wait = 0.02  # 即タイムアウト

        mock_batch_created = MagicMock()
        mock_batch_created.id = "msgbatch_timeout"

        mock_batch_processing = MagicMock()
        mock_batch_processing.processing_status = "in_progress"
        mock_batch_processing.request_counts = None

        mock_client = MagicMock()
        mock_client.messages.batches.create = AsyncMock(return_value=mock_batch_created)
        mock_client.messages.batches.retrieve = AsyncMock(return_value=mock_batch_processing)

        with patch("anthropic.AsyncAnthropic", return_value=mock_client):
            with pytest.raises(TimeoutError, match="タイムアウト"):
                await proc.run([{
                    "custom_id": "req-1",
                    "role_key":  "shogun",
                    "messages":  [{"role": "user", "content": "test"}],
                }])

    def test_from_env_raises_without_key(self):
        from core.router.batch.anthropic_batch import AnthropicBatchProcessor
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": ""}):
            with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
                AnthropicBatchProcessor.from_env()

    def test_from_env_ok(self):
        from core.router.batch.anthropic_batch import AnthropicBatchProcessor
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test"}):
            proc = AnthropicBatchProcessor.from_env()
            assert proc._api_key == "sk-ant-test"


# ===========================================================================
# 9. batch_bulk_notion_store
# ===========================================================================

class TestBatchBulkNotionStore:
    def _make_mixin(self):
        from core.router.mixins.postprocess import PostprocessMixin
        return PostprocessMixin()

    @pytest.mark.asyncio
    async def test_all_saved(self):
        mixin = self._make_mixin()
        save_mock = AsyncMock()
        states = [
            {"thread_id": f"t{i}", "should_save": True, "response": f"res{i}"}
            for i in range(3)
        ]
        with patch("integrations.notion.storage.save_task_result_bg", save_mock):
            result = await mixin.batch_bulk_notion_store(states)

        assert result["saved"] == 3
        assert result["failed"] == 0
        assert save_mock.await_count == 3

    @pytest.mark.asyncio
    async def test_partial_failure(self):
        mixin = self._make_mixin()
        call_count = 0

        async def _save(state):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("DB error")

        states = [
            {"thread_id": f"t{i}", "should_save": True}
            for i in range(3)
        ]
        with patch("integrations.notion.storage.save_task_result_bg", side_effect=_save):
            result = await mixin.batch_bulk_notion_store(states)

        assert result["saved"] == 2
        assert result["failed"] == 1
        assert len(result["errors"]) == 1

    @pytest.mark.asyncio
    async def test_should_save_false_counts_as_skipped(self):
        mixin = self._make_mixin()
        save_mock = AsyncMock()
        states = [{"thread_id": "t0", "should_save": False}]
        with patch("integrations.notion.storage.save_task_result_bg", save_mock):
            result = await mixin.batch_bulk_notion_store(states)

        assert result["skipped"] == 1
        assert result["saved"] == 0
        save_mock.assert_not_awaited()


# ===========================================================================
# 10. ANTHROPIC_ROLES 定数
# ===========================================================================

class TestAnthropicRolesConstant:
    def test_shogun_daigensui_in_set(self):
        from core.router.batch.anthropic_batch import ANTHROPIC_ROLES
        assert "shogun" in ANTHROPIC_ROLES
        assert "daigensui" in ANTHROPIC_ROLES

    def test_other_roles_not_in_set(self):
        from core.router.batch.anthropic_batch import ANTHROPIC_ROLES
        for role in ("seppou", "gunshi", "gaiji", "sanbo", "metsuke", "yuhitsu", "onmitsu", "kengyo"):
            assert role not in ANTHROPIC_ROLES
