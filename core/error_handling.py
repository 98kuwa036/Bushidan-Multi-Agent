"""Enhanced Error Handling System for ShogunAI v7.0

Comprehensive error handling, retry policies, circuit breakers, and fallback mechanisms
to ensure system resilience and graceful degradation.
"""

import asyncio
import logging
import time
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Optional, Union, List
from contextlib import asynccontextmanager
import json
from pathlib import Path


logger = logging.getLogger("shogun.error_handling")


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class FallbackStrategy(Enum):
    """Fallback strategies for different failure scenarios."""
    LOCAL_R1 = "local_r1"
    SIMPLIFIED_RESPONSE = "simplified_response"
    SKIP_RECORDING = "skip_recording"
    LOCAL_CACHE = "local_cache"
    RETRY_LATER = "retry_later"
    MANUAL_INTERVENTION = "manual_intervention"


@dataclass
class ErrorContext:
    """Context information for error handling."""
    operation: str
    component: str
    attempt: int
    max_attempts: int
    error_type: str
    error_message: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CircuitBreakerState:
    """Circuit breaker state tracking."""
    name: str
    failures: int = 0
    last_failure: Optional[float] = None
    state: str = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    last_success: Optional[float] = None
    half_open_attempts: int = 0


class ShogunErrorHandler:
    """Centralized error handling system for ShogunAI."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config.get("error_handling", {})
        self.retry_config = self.config.get("retry_policy", {})
        self.timeout_config = self.config.get("timeout_settings", {})
        self.fallback_config = self.config.get("fallback_behavior", {})
        self.circuit_config = self.config.get("circuit_breaker", {})
        
        # Circuit breakers for different components
        self.circuit_breakers: Dict[str, CircuitBreakerState] = {}
        
        # Error statistics
        self.error_stats = {
            "total_errors": 0,
            "errors_by_component": {},
            "errors_by_type": {},
            "recovery_successes": 0,
            "fallback_activations": 0,
        }
        
        # Initialize circuit breakers
        self._initialize_circuit_breakers()
    
    def _initialize_circuit_breakers(self):
        """Initialize circuit breakers for key components."""
        components = [
            "taisho_r1", "groq_api", "notion_api", "anthropic_api",
            "slack_api", "github_api", "file_operations"
        ]
        
        for component in components:
            self.circuit_breakers[component] = CircuitBreakerState(name=component)
    
    @asynccontextmanager
    async def handle_operation(
        self,
        operation: str,
        component: str,
        timeout: Optional[float] = None,
        fallback_strategy: Optional[FallbackStrategy] = None,
        **metadata
    ):
        """Context manager for handling operations with comprehensive error handling.
        
        Args:
            operation: Name of the operation
            component: Component performing the operation
            timeout: Optional timeout override
            fallback_strategy: Strategy if operation fails
            **metadata: Additional context information
        """
        start_time = time.time()
        attempt = 0
        max_attempts = self.retry_config.get("max_attempts", 3)
        
        # Check circuit breaker
        if not self._check_circuit_breaker(component):
            raise CircuitBreakerOpenError(
                f"Circuit breaker OPEN for {component}. Operation blocked."
            )
        
        while attempt < max_attempts:
            attempt += 1
            error_context = ErrorContext(
                operation=operation,
                component=component,
                attempt=attempt,
                max_attempts=max_attempts,
                error_type="",
                error_message="",
                metadata=metadata
            )
            
            try:
                # Apply timeout if specified
                operation_timeout = timeout or self.timeout_config.get(
                    component, self.timeout_config.get("api_request", 60)
                )
                
                async with asyncio.timeout(operation_timeout):
                    yield error_context
                    
                # Success - record and reset circuit breaker
                self._record_success(component)
                elapsed = time.time() - start_time
                logger.info(
                    f"[ErrorHandler] ✅ {operation} successful in {elapsed:.2f}s "
                    f"(attempt {attempt}/{max_attempts})"
                )
                return
                
            except asyncio.TimeoutError as e:
                error_context.error_type = "timeout"
                error_context.error_message = f"Operation timed out after {operation_timeout}s"
                await self._handle_error(error_context, e)
                
            except CircuitBreakerOpenError:
                raise  # Don't retry if circuit breaker is open
                
            except Exception as e:
                error_context.error_type = type(e).__name__
                error_context.error_message = str(e)
                await self._handle_error(error_context, e)
            
            # Delay before retry (with exponential backoff and jitter)
            if attempt < max_attempts:
                delay = self._calculate_retry_delay(attempt)
                logger.info(f"[ErrorHandler] ⏳ Retrying in {delay:.1f}s...")
                await asyncio.sleep(delay)
        
        # All attempts failed - apply fallback strategy
        if fallback_strategy:
            logger.warning(
                f"[ErrorHandler] 🔄 All attempts failed, applying fallback: {fallback_strategy.value}"
            )
            return await self._apply_fallback_strategy(
                fallback_strategy, error_context
            )
        else:
            raise OperationFailedError(
                f"Operation {operation} failed after {max_attempts} attempts"
            )
    
    async def _handle_error(self, context: ErrorContext, exception: Exception):
        """Handle individual error occurrence."""
        self.error_stats["total_errors"] += 1
        
        # Track by component
        component_stats = self.error_stats["errors_by_component"]
        component_stats[context.component] = component_stats.get(context.component, 0) + 1
        
        # Track by error type
        type_stats = self.error_stats["errors_by_type"]
        type_stats[context.error_type] = type_stats.get(context.error_type, 0) + 1
        
        # Determine severity
        severity = self._determine_error_severity(context, exception)
        
        # Log error with appropriate level
        log_msg = (
            f"[ErrorHandler] ❌ {context.operation} failed "
            f"(attempt {context.attempt}/{context.max_attempts}): "
            f"{context.error_message}"
        )
        
        if severity == ErrorSeverity.CRITICAL:
            logger.critical(log_msg)
        elif severity == ErrorSeverity.HIGH:
            logger.error(log_msg)
        elif severity == ErrorSeverity.MEDIUM:
            logger.warning(log_msg)
        else:
            logger.info(log_msg)
        
        # Update circuit breaker
        self._record_failure(context.component)
        
        # Send alerts for high/critical errors
        if severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
            await self._send_alert(context, severity)
    
    def _determine_error_severity(
        self, context: ErrorContext, exception: Exception
    ) -> ErrorSeverity:
        """Determine error severity based on context and exception type."""
        # Critical errors that affect core functionality
        critical_errors = [
            "AuthenticationError", "PermissionError", "ConfigurationError"
        ]
        
        # High severity errors that impact user experience
        high_severity_errors = [
            "ConnectionError", "ServiceUnavailableError", "RateLimitError"
        ]
        
        # Medium severity errors that are recoverable
        medium_severity_errors = [
            "TimeoutError", "ValidationError", "ParseError"
        ]
        
        error_type = context.error_type
        
        if error_type in critical_errors:
            return ErrorSeverity.CRITICAL
        elif error_type in high_severity_errors:
            return ErrorSeverity.HIGH
        elif error_type in medium_severity_errors:
            return ErrorSeverity.MEDIUM
        else:
            return ErrorSeverity.LOW
    
    def _calculate_retry_delay(self, attempt: int) -> float:
        """Calculate retry delay with exponential backoff and jitter."""
        base_delay = self.retry_config.get("base_delay", 1.0)
        max_delay = self.retry_config.get("max_delay", 30.0)
        strategy = self.retry_config.get("backoff_strategy", "exponential")
        use_jitter = self.retry_config.get("jitter", True)
        
        if strategy == "exponential":
            delay = base_delay * (2 ** (attempt - 1))
        elif strategy == "linear":
            delay = base_delay * attempt
        else:  # constant
            delay = base_delay
        
        # Apply maximum delay limit
        delay = min(delay, max_delay)
        
        # Add jitter to prevent thundering herd
        if use_jitter:
            jitter_range = delay * 0.1
            delay += random.uniform(-jitter_range, jitter_range)
        
        return max(delay, 0.1)  # Minimum 0.1s delay
    
    def _check_circuit_breaker(self, component: str) -> bool:
        """Check if circuit breaker allows operation."""
        if not self.circuit_config.get("enabled", False):
            return True
        
        breaker = self.circuit_breakers.get(component)
        if not breaker:
            return True
        
        failure_threshold = self.circuit_config.get("failure_threshold", 5)
        recovery_timeout = self.circuit_config.get("recovery_timeout", 300)
        current_time = time.time()
        
        if breaker.state == "CLOSED":
            return True
        elif breaker.state == "OPEN":
            # Check if recovery timeout has passed
            if (breaker.last_failure and 
                current_time - breaker.last_failure > recovery_timeout):
                breaker.state = "HALF_OPEN"
                breaker.half_open_attempts = 0
                logger.info(f"[Circuit Breaker] {component} transitioning to HALF_OPEN")
                return True
            return False
        elif breaker.state == "HALF_OPEN":
            # Allow limited requests in half-open state
            max_half_open = self.circuit_config.get("half_open_requests", 3)
            if breaker.half_open_attempts < max_half_open:
                breaker.half_open_attempts += 1
                return True
            return False
        
        return False
    
    def _record_success(self, component: str):
        """Record successful operation for circuit breaker."""
        breaker = self.circuit_breakers.get(component)
        if not breaker:
            return
        
        current_time = time.time()
        breaker.last_success = current_time
        
        if breaker.state in ["HALF_OPEN", "OPEN"]:
            breaker.state = "CLOSED"
            breaker.failures = 0
            breaker.half_open_attempts = 0
            self.error_stats["recovery_successes"] += 1
            logger.info(f"[Circuit Breaker] {component} recovered, state: CLOSED")
    
    def _record_failure(self, component: str):
        """Record failed operation for circuit breaker."""
        breaker = self.circuit_breakers.get(component)
        if not breaker:
            return
        
        current_time = time.time()
        breaker.failures += 1
        breaker.last_failure = current_time
        
        failure_threshold = self.circuit_config.get("failure_threshold", 5)
        
        if breaker.failures >= failure_threshold:
            breaker.state = "OPEN"
            logger.warning(f"[Circuit Breaker] {component} opened due to failures")
        elif breaker.state == "HALF_OPEN":
            # Failed during half-open, go back to open
            breaker.state = "OPEN"
            logger.warning(f"[Circuit Breaker] {component} back to OPEN from HALF_OPEN")
    
    async def _apply_fallback_strategy(
        self, strategy: FallbackStrategy, context: ErrorContext
    ) -> Any:
        """Apply fallback strategy when operation fails."""
        self.error_stats["fallback_activations"] += 1
        
        logger.info(
            f"[ErrorHandler] 🔄 Applying fallback strategy: {strategy.value} "
            f"for {context.operation}"
        )
        
        if strategy == FallbackStrategy.LOCAL_R1:
            return await self._fallback_to_local_r1(context)
        elif strategy == FallbackStrategy.SIMPLIFIED_RESPONSE:
            return self._fallback_simplified_response(context)
        elif strategy == FallbackStrategy.SKIP_RECORDING:
            logger.info(f"[ErrorHandler] Skipping recording for {context.operation}")
            return None
        elif strategy == FallbackStrategy.LOCAL_CACHE:
            return await self._fallback_local_cache(context)
        elif strategy == FallbackStrategy.RETRY_LATER:
            await self._schedule_retry(context)
            return None
        else:
            raise NotImplementedError(f"Fallback strategy {strategy.value} not implemented")
    
    async def _fallback_to_local_r1(self, context: ErrorContext) -> str:
        """Fallback to local R1 model when cloud APIs fail."""
        logger.info("[ErrorHandler] 🏠 Falling back to local Japanese R1 model")
        
        try:
            # This would integrate with the local R1 client
            # For now, return a placeholder response
            return (
                "申し訳ございませんが、クラウドAPIが利用できないため、"
                "ローカルの日本語R1モデルで対応いたします。"
                "完全な機能ではありませんが、基本的なサポートを提供できます。"
            )
        except Exception as e:
            logger.error(f"[ErrorHandler] Local R1 fallback also failed: {e}")
            return self._fallback_simplified_response(context)
    
    def _fallback_simplified_response(self, context: ErrorContext) -> str:
        """Provide a simplified response when all else fails."""
        logger.info("[ErrorHandler] 📝 Providing simplified fallback response")
        
        return (
            f"申し訳ございません。現在、{context.component}に技術的な問題が発生しています。"
            "しばらくしてから再度お試しください。"
            f"エラー詳細: {context.error_message[:100]}..."
        )
    
    async def _fallback_local_cache(self, context: ErrorContext) -> Optional[str]:
        """Attempt to use local cache when external service fails."""
        logger.info("[ErrorHandler] 💾 Attempting local cache fallback")
        
        # Implementation would check local cache
        # For now, return None to indicate no cached data
        return None
    
    async def _schedule_retry(self, context: ErrorContext):
        """Schedule operation for retry later."""
        logger.info(f"[ErrorHandler] ⏰ Scheduling retry for {context.operation}")
        
        # Implementation would add to retry queue
        # This is a placeholder
        pass
    
    async def _send_alert(self, context: ErrorContext, severity: ErrorSeverity):
        """Send alert for high severity errors."""
        alert_msg = (
            f"🚨 ShogunAI Alert [{severity.value.upper()}]\n"
            f"Operation: {context.operation}\n"
            f"Component: {context.component}\n"
            f"Error: {context.error_message}\n"
            f"Attempt: {context.attempt}/{context.max_attempts}"
        )
        
        logger.critical(alert_msg)
        
        # In a real implementation, this would:
        # - Send Slack notification
        # - Write to alert log
        # - Potentially trigger PagerDuty/email alerts
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status of all components."""
        status = {
            "overall_health": "healthy",
            "circuit_breakers": {},
            "error_stats": dict(self.error_stats),
            "timestamp": time.time()
        }
        
        unhealthy_components = 0
        
        for name, breaker in self.circuit_breakers.items():
            breaker_status = {
                "state": breaker.state,
                "failures": breaker.failures,
                "last_failure": breaker.last_failure,
                "last_success": breaker.last_success
            }
            status["circuit_breakers"][name] = breaker_status
            
            if breaker.state == "OPEN":
                unhealthy_components += 1
        
        # Determine overall health
        if unhealthy_components > 0:
            if unhealthy_components >= len(self.circuit_breakers) / 2:
                status["overall_health"] = "critical"
            else:
                status["overall_health"] = "degraded"
        
        return status
    
    def reset_error_stats(self):
        """Reset error statistics (useful for testing or maintenance)."""
        self.error_stats = {
            "total_errors": 0,
            "errors_by_component": {},
            "errors_by_type": {},
            "recovery_successes": 0,
            "fallback_activations": 0,
        }
        logger.info("[ErrorHandler] Error statistics reset")
    
    def export_error_report(self, filepath: str):
        """Export detailed error report to file."""
        report = {
            "timestamp": time.time(),
            "error_stats": self.error_stats,
            "circuit_breakers": {
                name: {
                    "state": breaker.state,
                    "failures": breaker.failures,
                    "last_failure": breaker.last_failure,
                    "last_success": breaker.last_success
                }
                for name, breaker in self.circuit_breakers.items()
            },
            "configuration": self.config
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"[ErrorHandler] Error report exported to {filepath}")


# Custom Exception Classes
class ShogunError(Exception):
    """Base exception class for ShogunAI."""
    pass


class OperationFailedError(ShogunError):
    """Raised when an operation fails after all retries."""
    pass


class CircuitBreakerOpenError(ShogunError):
    """Raised when circuit breaker is open."""
    pass


class FallbackError(ShogunError):
    """Raised when fallback strategy fails."""
    pass


class ConfigurationError(ShogunError):
    """Raised when there's a configuration issue."""
    pass