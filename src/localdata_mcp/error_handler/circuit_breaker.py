"""Circuit breaker pattern for database connections.

Provides CircuitBreakerConfig, CircuitBreakerStats, CircuitBreaker,
CircuitBreakerRegistry, and the circuit_breaker_protection context manager.
"""

import threading
import time
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Type

from ..logging_manager import get_logger, get_logging_manager
from .exceptions import (
    CircuitState,
    DatabaseConnectionError,
    ResourceExhaustionError,
)

# Get structured logger
logger = get_logger(__name__)


# ============================================================================
# Circuit Breaker Pattern
# ============================================================================


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""

    failure_threshold: int = 5  # Failures before opening circuit
    success_threshold: int = 3  # Successes before closing circuit
    timeout_duration: float = 60.0  # Seconds to wait before half-open
    monitor_window: float = 300.0  # Window to track failures (5 minutes)

    # Failure conditions
    failure_exceptions: Tuple[Type[Exception], ...] = field(
        default_factory=lambda: (DatabaseConnectionError, ResourceExhaustionError)
    )


@dataclass
class CircuitBreakerStats:
    """Statistics for circuit breaker monitoring."""

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    circuit_opened_count: int = 0
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None
    failure_window: deque = field(default_factory=lambda: deque(maxlen=100))

    @property
    def failure_rate(self) -> float:
        """Calculate current failure rate."""
        if self.total_requests == 0:
            return 0.0
        return (self.failed_requests / self.total_requests) * 100

    @property
    def recent_failure_count(self) -> int:
        """Count recent failures within monitor window."""
        current_time = time.time()
        recent_failures = [
            failure_time
            for failure_time in self.failure_window
            if current_time - failure_time <= 300.0  # 5 minute window
        ]
        return len(recent_failures)


class CircuitBreaker:
    """Circuit breaker implementation for database connections."""

    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.stats = CircuitBreakerStats()
        self.lock = threading.RLock()
        self.half_open_attempts = 0
        self.state_change_time = time.time()

    def is_request_allowed(self) -> bool:
        """Check if a request is allowed through the circuit breaker."""
        with self.lock:
            current_time = time.time()

            if self.state == CircuitState.CLOSED:
                return True
            elif self.state == CircuitState.OPEN:
                # Check if timeout has elapsed to move to half-open
                if (
                    current_time - self.state_change_time
                    >= self.config.timeout_duration
                ):
                    self._transition_to_half_open()
                    return True
                return False
            elif self.state == CircuitState.HALF_OPEN:
                # Allow limited requests in half-open state
                return self.half_open_attempts < self.config.success_threshold

        return False

    def record_success(self):
        """Record a successful operation."""
        with self.lock:
            self.stats.total_requests += 1
            self.stats.successful_requests += 1
            self.stats.last_success_time = time.time()

            if self.state == CircuitState.HALF_OPEN:
                self.half_open_attempts += 1
                if self.half_open_attempts >= self.config.success_threshold:
                    self._transition_to_closed()

    def record_failure(self, exception: Exception):
        """Record a failed operation."""
        with self.lock:
            self.stats.total_requests += 1
            self.stats.failed_requests += 1
            current_time = time.time()
            self.stats.last_failure_time = current_time
            self.stats.failure_window.append(current_time)

            # Check if this is a circuit-breaking failure
            is_circuit_failure = any(
                isinstance(exception, failure_type)
                for failure_type in self.config.failure_exceptions
            )

            if is_circuit_failure:
                if self.state == CircuitState.HALF_OPEN:
                    # Failure during half-open immediately opens circuit
                    self._transition_to_open()
                elif self.state == CircuitState.CLOSED:
                    # Check if we should open the circuit
                    if self.stats.recent_failure_count >= self.config.failure_threshold:
                        self._transition_to_open()

    def get_state_info(self) -> Dict[str, Any]:
        """Get current state information."""
        with self.lock:
            return {
                "name": self.name,
                "state": self.state.value,
                "stats": {
                    "total_requests": self.stats.total_requests,
                    "successful_requests": self.stats.successful_requests,
                    "failed_requests": self.stats.failed_requests,
                    "failure_rate": self.stats.failure_rate,
                    "recent_failure_count": self.stats.recent_failure_count,
                    "circuit_opened_count": self.stats.circuit_opened_count,
                    "last_failure_time": self.stats.last_failure_time,
                    "last_success_time": self.stats.last_success_time,
                },
                "config": {
                    "failure_threshold": self.config.failure_threshold,
                    "success_threshold": self.config.success_threshold,
                    "timeout_duration": self.config.timeout_duration,
                },
                "state_change_time": self.state_change_time,
                "half_open_attempts": self.half_open_attempts,
            }

    def reset(self):
        """Reset the circuit breaker to closed state."""
        with self.lock:
            self.state = CircuitState.CLOSED
            self.half_open_attempts = 0
            self.state_change_time = time.time()
            # Keep stats for monitoring, but could optionally reset them
            logging_manager = get_logging_manager()
            with logging_manager.context(
                operation="circuit_breaker_reset",
                component="error_handler",
                circuit_breaker_name=self.name,
            ):
                logger.info(
                    "Circuit breaker reset",
                    failure_rate=self.stats.failure_rate,
                    total_requests=self.stats.total_requests,
                    failed_requests=self.stats.failed_requests,
                )

    def _transition_to_open(self):
        """Transition circuit breaker to open state."""
        self.state = CircuitState.OPEN
        self.state_change_time = time.time()
        self.stats.circuit_opened_count += 1
        self.half_open_attempts = 0
        logging_manager = get_logging_manager()
        logging_manager.log_security_event(
            "circuit_breaker_opened",
            "medium",
            f"Circuit breaker '{self.name}' opened due to excessive failures",
            circuit_breaker_name=self.name,
            failure_rate=self.stats.failure_rate,
            failed_requests=self.stats.failed_requests,
            threshold=self.config.failure_threshold,
        )

    def _transition_to_half_open(self):
        """Transition circuit breaker to half-open state."""
        self.state = CircuitState.HALF_OPEN
        self.state_change_time = time.time()
        self.half_open_attempts = 0
        logging_manager = get_logging_manager()
        with logging_manager.context(
            operation="circuit_breaker_half_open",
            component="error_handler",
            circuit_breaker_name=self.name,
        ):
            logger.info(
                "Circuit breaker moved to half-open state",
                timeout_duration=self.config.timeout_duration,
                open_duration=time.time() - self.state_change_time,
            )

    def _transition_to_closed(self):
        """Transition circuit breaker to closed state."""
        self.state = CircuitState.CLOSED
        self.state_change_time = time.time()
        self.half_open_attempts = 0
        logging_manager = get_logging_manager()
        with logging_manager.context(
            operation="circuit_breaker_closed",
            component="error_handler",
            circuit_breaker_name=self.name,
        ):
            logger.info(
                "Circuit breaker closed after recovery",
                half_open_attempts=self.half_open_attempts,
                success_threshold=self.config.success_threshold,
                total_downtime=time.time() - self.state_change_time,
            )


class CircuitBreakerRegistry:
    """Registry for managing multiple circuit breakers."""

    def __init__(self):
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._lock = threading.RLock()

    def get_breaker(
        self, name: str, config: Optional[CircuitBreakerConfig] = None
    ) -> CircuitBreaker:
        """Get or create a circuit breaker for a given name."""
        with self._lock:
            if name not in self._breakers:
                self._breakers[name] = CircuitBreaker(name, config)
            return self._breakers[name]

    def remove_breaker(self, name: str) -> bool:
        """Remove a circuit breaker."""
        with self._lock:
            if name in self._breakers:
                del self._breakers[name]
                return True
            return False

    def get_all_breakers(self) -> Dict[str, CircuitBreaker]:
        """Get all circuit breakers."""
        with self._lock:
            return self._breakers.copy()

    def get_status_summary(self) -> Dict[str, Any]:
        """Get status summary of all circuit breakers."""
        with self._lock:
            summary = {
                "total_breakers": len(self._breakers),
                "states": defaultdict(int),
                "breakers": {},
            }

            for name, breaker in self._breakers.items():
                state_info = breaker.get_state_info()
                summary["states"][state_info["state"]] += 1
                summary["breakers"][name] = state_info

            return summary


@contextmanager
def circuit_breaker_protection(
    breaker: CircuitBreaker, operation_name: str = "unknown"
):
    """Context manager for circuit breaker protection."""
    if not breaker.is_request_allowed():
        raise DatabaseConnectionError(
            message=f"Circuit breaker '{breaker.name}' is open, rejecting request",
            metadata={
                "circuit_breaker_state": breaker.state.value,
                "operation_name": operation_name,
            },
            recovery_suggestions=[
                "Wait for circuit breaker to recover",
                "Check database connectivity",
                "Review recent error patterns",
            ],
        )

    try:
        yield
        breaker.record_success()
    except Exception as e:
        breaker.record_failure(e)
        raise
