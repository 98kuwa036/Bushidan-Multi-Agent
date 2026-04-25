"""
tests/integration/test_integration_smoke.py — Integration smoke tests

These tests verify module importability and basic wiring without
requiring live API keys or database connections.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

os.environ.setdefault("POSTGRES_URL", "postgresql://stub:stub@localhost/stub")
os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-stub")


class TestModuleImports:
    """Verify core modules are importable without side-effects."""

    def test_import_state(self):
        from core.state import BushidanState
        assert BushidanState is not None

    def test_import_base_client(self):
        from utils.base_client import BaseLLMClient
        assert BaseLLMClient is not None

    def test_import_processing_mode(self):
        from core.router.batch.mode import ProcessingMode, BATCH_CONFIG
        assert ProcessingMode.INTERACTIVE == "interactive"
        assert ProcessingMode.BATCH == "batch"
        assert isinstance(BATCH_CONFIG, dict)

    def test_import_auth(self):
        from console.auth import check_password, create_session, validate_session
        assert callable(check_password)
        assert callable(create_session)
        assert callable(validate_session)

    def test_import_anthropic_batch(self):
        from core.router.batch.anthropic_batch import AnthropicBatchProcessor, ANTHROPIC_ROLES
        assert "shogun" in ANTHROPIC_ROLES
        assert "daigensui" in ANTHROPIC_ROLES


class TestAuthIntegration:
    """Auth module integration: session lifecycle without a real password hash."""

    def test_session_lifecycle(self):
        from console.auth import create_session, validate_session
        with pytest.MonkeyPatch.context() as mp:
            mp.setenv("CONSOLE_PASSWORD", "testpass")
            token = create_session()
            assert isinstance(token, str) and len(token) > 20
            assert validate_session(token) is True

    def test_invalid_session_rejected(self):
        from console.auth import validate_session
        with pytest.MonkeyPatch.context() as mp:
            mp.setenv("CONSOLE_PASSWORD", "testpass")
            assert validate_session("not-a-real-token") is False

    def test_plaintext_password_check(self):
        from console.auth import check_password
        with pytest.MonkeyPatch.context() as mp:
            mp.setenv("CONSOLE_PASSWORD", "secret123")
            mp.delenv("CONSOLE_PASSWORD_HASH", raising=False)
            assert check_password("secret123") is True
            assert check_password("wrong") is False


class TestRetryLogic:
    """BaseLLMClient.generate_with_retry integration-level wiring."""

    @pytest.mark.asyncio
    async def test_retry_raises_on_non_retryable(self):
        from utils.base_client import BaseLLMClient

        class _StubClient(BaseLLMClient):
            async def generate(self, messages, system="", max_tokens=4096, **kw):
                err = Exception("bad request")
                err.status_code = 400
                raise err

        client = _StubClient()
        with pytest.raises(Exception, match="bad request"):
            await client.generate_with_retry([{"role": "user", "content": "hi"}])

    @pytest.mark.asyncio
    async def test_retry_succeeds_on_second_attempt(self):
        from utils.base_client import BaseLLMClient

        call_count = 0

        class _FlakyClient(BaseLLMClient):
            async def generate(self, messages, system="", max_tokens=4096, **kw):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    err = Exception("rate limit")
                    err.status_code = 429
                    raise err
                return "success"

        client = _FlakyClient()
        result = await client.generate_with_retry(
            [{"role": "user", "content": "hi"}],
            base_delay=0.001,
        )
        assert result == "success"
        assert call_count == 2


class TestBatchConfig:
    """BATCH_CONFIG values are production-ready."""

    def test_batch_parallel_steps_positive(self):
        from core.router.batch.mode import BATCH_CONFIG
        assert BATCH_CONFIG["max_parallel_batch_steps"] > 0

    def test_node_timeout_multiplier(self):
        from core.router.batch.mode import BATCH_CONFIG
        assert BATCH_CONFIG["node_timeout_multiplier"] >= 1.0
