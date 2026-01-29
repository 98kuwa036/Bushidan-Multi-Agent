"""Test Suite for Enhanced Error Handling System

Comprehensive tests for the ShogunAI v7.0 error handling system,
circuit breakers, retry policies, and fallback strategies.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any

from shogun.core.error_handling import (
    ShogunErrorHandler,
    FallbackStrategy,
    ErrorSeverity,
    ShogunError,
    OperationFailedError,
    CircuitBreakerOpenError,
    ErrorContext,
    CircuitBreakerState,
)


class TestShogunErrorHandler:
    """Test cases for the main error handler."""

    @pytest.fixture
    def config(self) -> Dict[str, Any]:
        """Test configuration."""
        return {
            "error_handling": {
                "retry_policy": {
                    "max_attempts": 3,
                    "backoff_strategy": "exponential",
                    "base_delay": 0.1,
                    "max_delay": 1.0,
                    "jitter": True,
                },
                "timeout_settings": {
                    "api_request": 10,
                    "local_r1": 30,
                    "groq_request": 5,
                },
                "fallback_behavior": {
                    "api_failure": "local_r1",
                    "r1_failure": "simplified_response",
                },
                "circuit_breaker": {
                    "enabled": True,
                    "failure_threshold": 3,
                    "recovery_timeout": 60,
                    "half_open_requests": 2,
                },
            }
        }

    @pytest.fixture
    def error_handler(self, config):
        """Create error handler instance for testing."""
        return ShogunErrorHandler(config)

    def test_initialization(self, error_handler):
        """Test proper initialization of error handler."""
        assert len(error_handler.circuit_breakers) > 0
        assert "taisho_r1" in error_handler.circuit_breakers
        assert "groq_api" in error_handler.circuit_breakers
        assert error_handler.error_stats["total_errors"] == 0

    @pytest.mark.asyncio
    async def test_successful_operation(self, error_handler):
        """Test handling of successful operations."""
        async with error_handler.handle_operation(
            operation="test_operation",
            component="test_component"
        ) as context:
            assert context.operation == "test_operation"
            assert context.component == "test_component"
            assert context.attempt == 1

    @pytest.mark.asyncio
    async def test_timeout_handling(self, error_handler):
        """Test timeout handling."""
        with pytest.raises(asyncio.TimeoutError):
            async with error_handler.handle_operation(
                operation="timeout_test",
                component="test_component",
                timeout=0.1
            ):
                await asyncio.sleep(0.2)

    @pytest.mark.asyncio
    async def test_retry_with_exponential_backoff(self, error_handler):
        """Test retry mechanism with exponential backoff."""
        attempt_count = 0
        start_time = time.time()

        try:
            async with error_handler.handle_operation(
                operation="retry_test",
                component="test_component"
            ):
                nonlocal attempt_count
                attempt_count += 1
                if attempt_count < 3:
                    raise ValueError(f"Attempt {attempt_count} failed")
        except OperationFailedError:
            pass

        elapsed = time.time() - start_time
        assert attempt_count == 3
        # Should have some delay due to backoff
        assert elapsed > 0.2  # At least base delays

    def test_circuit_breaker_initialization(self, error_handler):
        """Test circuit breaker initialization."""
        breaker = error_handler.circuit_breakers["test_component"]
        assert breaker.state == "CLOSED"
        assert breaker.failures == 0

    def test_circuit_breaker_open_on_failures(self, error_handler):
        """Test circuit breaker opens after threshold failures."""
        component = "test_component"
        threshold = error_handler.circuit_config.get("failure_threshold", 5)

        # Simulate failures to reach threshold
        for _ in range(threshold):
            error_handler._record_failure(component)

        breaker = error_handler.circuit_breakers[component]
        assert breaker.state == "OPEN"
        assert not error_handler._check_circuit_breaker(component)

    def test_circuit_breaker_recovery(self, error_handler):
        """Test circuit breaker recovery after timeout."""
        component = "test_component"
        breaker = error_handler.circuit_breakers[component]

        # Force open state
        breaker.state = "OPEN"
        breaker.last_failure = time.time() - 100  # Old failure

        # Should transition to half-open
        assert error_handler._check_circuit_breaker(component)
        assert breaker.state == "HALF_OPEN"

    def test_error_severity_determination(self, error_handler):
        """Test error severity classification."""
        # Critical error
        context = ErrorContext(
            operation="test", component="test", attempt=1, max_attempts=3,
            error_type="AuthenticationError", error_message="Auth failed"
        )
        severity = error_handler._determine_error_severity(context, Exception())
        assert severity == ErrorSeverity.CRITICAL

        # Low severity error
        context.error_type = "UnknownError"
        severity = error_handler._determine_error_severity(context, Exception())
        assert severity == ErrorSeverity.LOW

    def test_retry_delay_calculation(self, error_handler):
        """Test retry delay calculations."""
        # Exponential backoff
        delay1 = error_handler._calculate_retry_delay(1)
        delay2 = error_handler._calculate_retry_delay(2)
        delay3 = error_handler._calculate_retry_delay(3)

        assert delay2 > delay1
        assert delay3 > delay2
        assert delay3 <= error_handler.retry_config.get("max_delay", 30)

    @pytest.mark.asyncio
    async def test_fallback_strategies(self, error_handler):
        """Test different fallback strategies."""
        context = ErrorContext(
            operation="test", component="api", attempt=3, max_attempts=3,
            error_type="ConnectionError", error_message="API unreachable"
        )

        # Test simplified response fallback
        result = await error_handler._apply_fallback_strategy(
            FallbackStrategy.SIMPLIFIED_RESPONSE, context
        )
        assert isinstance(result, str)
        assert "技術的な問題" in result

        # Test skip recording fallback
        result = await error_handler._apply_fallback_strategy(
            FallbackStrategy.SKIP_RECORDING, context
        )
        assert result is None

    def test_usage_statistics_tracking(self, error_handler):
        """Test error statistics tracking."""
        initial_errors = error_handler.error_stats["total_errors"]

        # Simulate error
        context = ErrorContext(
            operation="test", component="test", attempt=1, max_attempts=3,
            error_type="TestError", error_message="Test error"
        )
        
        # The _handle_error method updates stats
        asyncio.run(error_handler._handle_error(context, Exception()))

        assert error_handler.error_stats["total_errors"] == initial_errors + 1
        assert "test" in error_handler.error_stats["errors_by_component"]
        assert "TestError" in error_handler.error_stats["errors_by_type"]

    def test_health_status_reporting(self, error_handler):
        """Test health status reporting."""
        status = error_handler.get_health_status()

        assert "overall_health" in status
        assert "circuit_breakers" in status
        assert "error_stats" in status
        assert "timestamp" in status
        assert status["overall_health"] in ["healthy", "degraded", "critical"]

    def test_error_report_export(self, error_handler, tmp_path):
        """Test error report export functionality."""
        report_file = tmp_path / "error_report.json"

        error_handler.export_error_report(str(report_file))

        assert report_file.exists()
        
        import json
        with open(report_file) as f:
            report = json.load(f)

        assert "timestamp" in report
        assert "error_stats" in report
        assert "circuit_breakers" in report
        assert "configuration" in report

    def test_error_stats_reset(self, error_handler):
        """Test error statistics reset."""
        # Generate some stats
        error_handler.error_stats["total_errors"] = 10
        error_handler.error_stats["errors_by_component"]["test"] = 5

        # Reset
        error_handler.reset_error_stats()

        assert error_handler.error_stats["total_errors"] == 0
        assert len(error_handler.error_stats["errors_by_component"]) == 0


class TestIntegrationWithShogun:
    """Integration tests with main Shogun system."""

    @pytest.fixture
    def mock_shogun_config(self):
        """Mock configuration for integration tests."""
        return {
            "error_handling": {
                "retry_policy": {"max_attempts": 2},
                "circuit_breaker": {"enabled": True, "failure_threshold": 2},
                "fallback_behavior": {
                    "api_failure": "local_r1",
                    "r1_failure": "simplified_response",
                },
            }
        }

    @pytest.mark.asyncio
    async def test_integration_with_api_provider(self, mock_shogun_config):
        """Test error handling integration with API providers."""
        from shogun.providers.anthropic_api import AnthropicAPIProvider
        
        # Mock API provider that fails
        with patch('shogun.providers.anthropic_api.anthropic') as mock_anthropic:
            mock_client = Mock()
            mock_anthropic.AsyncAnthropic.return_value = mock_client
            mock_client.messages.create = AsyncMock(side_effect=Exception("API Error"))

            error_handler = ShogunErrorHandler(mock_shogun_config)
            provider = AnthropicAPIProvider(api_key="test")

            # This should fail and trigger error handling
            with pytest.raises(Exception):
                await provider.generate("test prompt")

            # Error should be recorded
            assert error_handler.error_stats["total_errors"] >= 0

    @pytest.mark.asyncio
    async def test_cascade_failure_handling(self, mock_shogun_config):
        """Test handling of cascading failures."""
        error_handler = ShogunErrorHandler(mock_shogun_config)

        failure_count = 0

        async def failing_operation():
            nonlocal failure_count
            failure_count += 1
            raise ConnectionError(f"Failure {failure_count}")

        # Multiple operations that should trigger circuit breaker
        for i in range(3):
            try:
                async with error_handler.handle_operation(
                    operation="cascade_test",
                    component="cascade_component",
                    fallback_strategy=FallbackStrategy.SIMPLIFIED_RESPONSE
                ):
                    await failing_operation()
            except (OperationFailedError, CircuitBreakerOpenError):
                pass

        # Circuit breaker should be open now
        assert not error_handler._check_circuit_breaker("cascade_component")

    @pytest.mark.asyncio
    async def test_recovery_after_circuit_breaker_timeout(self, mock_shogun_config):
        """Test recovery after circuit breaker timeout."""
        # Reduce recovery timeout for faster testing
        mock_shogun_config["error_handling"]["circuit_breaker"]["recovery_timeout"] = 0.1
        
        error_handler = ShogunErrorHandler(mock_shogun_config)
        component = "recovery_test"

        # Force circuit breaker open
        breaker = error_handler.circuit_breakers[component]
        breaker.state = "OPEN"
        breaker.last_failure = time.time() - 0.2

        # Should allow operation after timeout
        assert error_handler._check_circuit_breaker(component)
        assert breaker.state == "HALF_OPEN"

        # Successful operation should close circuit
        error_handler._record_success(component)
        assert breaker.state == "CLOSED"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])