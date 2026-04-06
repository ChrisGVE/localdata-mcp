"""
Circuit breaker pattern implementation for failure management.
"""

import logging
import threading
import time
from typing import Any, Dict

from ....logging_manager import get_logger
from ..interfaces import ConversionError
from ._types import CircuitBreakerState

logger = get_logger(__name__)


class CircuitBreaker:
    """Circuit breaker pattern implementation for failure management."""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: type = Exception,
    ):
        """
        Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before testing recovery
            expected_exception: Exception type to handle
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception

        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitBreakerState.CLOSED
        self._lock = threading.Lock()

        logger.debug(
            f"CircuitBreaker initialized",
            threshold=failure_threshold,
            timeout=recovery_timeout,
        )

    def __call__(self, func):
        """Decorator to wrap functions with circuit breaker."""

        def wrapper(*args, **kwargs):
            with self._lock:
                # Check if circuit should transition from OPEN to HALF_OPEN
                if (
                    self.state == CircuitBreakerState.OPEN
                    and self.last_failure_time
                    and time.time() - self.last_failure_time > self.recovery_timeout
                ):
                    self.state = CircuitBreakerState.HALF_OPEN
                    logger.info("Circuit breaker transitioning to HALF_OPEN")

                # Block requests if circuit is OPEN
                if self.state == CircuitBreakerState.OPEN:
                    raise ConversionError(
                        ConversionError.Type.ADAPTER_NOT_FOUND,
                        "Circuit breaker is OPEN - service unavailable",
                        {"circuit_state": self.state.value},
                    )

            try:
                # Execute function
                result = func(*args, **kwargs)

                # Success - reset failure count and close circuit
                with self._lock:
                    if self.state == CircuitBreakerState.HALF_OPEN:
                        self.state = CircuitBreakerState.CLOSED
                        logger.info(
                            "Circuit breaker CLOSED after successful recovery test"
                        )
                    self.failure_count = 0

                return result

            except self.expected_exception as e:
                with self._lock:
                    self.failure_count += 1
                    self.last_failure_time = time.time()

                    # Open circuit if threshold exceeded
                    if self.failure_count >= self.failure_threshold:
                        self.state = CircuitBreakerState.OPEN
                        logger.warning(
                            f"Circuit breaker OPENED after {self.failure_count} failures"
                        )

                raise

        return wrapper

    def get_state(self) -> Dict[str, Any]:
        """Get current circuit breaker state."""
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "failure_threshold": self.failure_threshold,
            "last_failure_time": self.last_failure_time,
            "recovery_timeout": self.recovery_timeout,
        }
