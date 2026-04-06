"""Retry mechanism with configurable policies and backoff strategies.

Provides RetryPolicy, RetryableOperation, and the retry_on_failure decorator.
"""

import logging
import random
import time
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

from ..logging_manager import get_logger, get_logging_manager
from .exceptions import (
    LocalDataError,
    QueryExecutionError,
    RetryStrategy,
)

# Get structured logger
logger = get_logger(__name__)


# ============================================================================
# Retry Mechanism
# ============================================================================


@dataclass
class RetryPolicy:
    """Configuration for retry behavior."""

    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    backoff_multiplier: float = 2.0
    jitter: bool = True
    retry_on: Tuple[Type[Exception], ...] = field(default_factory=lambda: (Exception,))
    stop_on: Tuple[Type[Exception], ...] = field(default_factory=tuple)

    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for a specific retry attempt."""
        if self.strategy == RetryStrategy.NO_RETRY:
            return 0.0
        elif self.strategy == RetryStrategy.FIXED:
            delay = self.base_delay
        elif self.strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = self.base_delay * attempt
        elif self.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = self.base_delay * (self.backoff_multiplier ** (attempt - 1))
        elif self.strategy == RetryStrategy.FIBONACCI:
            if attempt <= 2:
                delay = self.base_delay
            else:
                # Calculate Fibonacci number for delay
                fib_delay = self.base_delay
                prev_delay = self.base_delay
                for _ in range(attempt - 2):
                    fib_delay, prev_delay = fib_delay + prev_delay, fib_delay
                delay = fib_delay
        else:
            delay = self.base_delay

        # Apply maximum delay limit
        delay = min(delay, self.max_delay)

        # Add jitter to avoid thundering herd
        if self.jitter:
            jitter_factor = random.uniform(0.1, 0.1)  # 10% jitter
            delay += delay * jitter_factor

        return delay

    def should_retry(self, exception: Exception, attempt: int) -> bool:
        """Determine if an exception should trigger a retry."""
        if attempt >= self.max_attempts:
            return False

        # Check if exception is in stop list
        if any(isinstance(exception, stop_type) for stop_type in self.stop_on):
            return False

        # Check if exception is in retry list
        return any(isinstance(exception, retry_type) for retry_type in self.retry_on)


class RetryableOperation:
    """Wrapper for operations that can be retried."""

    def __init__(
        self, operation: Callable, policy: RetryPolicy, operation_name: str = "unknown"
    ):
        self.operation = operation
        self.policy = policy
        self.operation_name = operation_name
        self.attempt_history: List[Dict[str, Any]] = []

    def execute(self, *args, **kwargs) -> Any:
        """Execute the operation with retry logic."""
        last_exception = None

        for attempt in range(1, self.policy.max_attempts + 1):
            attempt_start = time.time()

            try:
                result = self.operation(*args, **kwargs)

                # Log successful execution after retries
                if attempt > 1:
                    logging_manager = get_logging_manager()
                    with logging_manager.context(
                        operation="retry_success",
                        component="error_handler",
                        operation_name=self.operation_name,
                    ):
                        logger.info(
                            "Operation succeeded after retries",
                            attempt=attempt,
                            total_attempts=self.policy.max_attempts,
                            retry_policy=self.policy.__class__.__name__,
                        )

                return result

            except Exception as e:
                last_exception = e
                attempt_duration = time.time() - attempt_start

                # Record attempt
                self.attempt_history.append(
                    {
                        "attempt": attempt,
                        "exception": str(e),
                        "exception_type": type(e).__name__,
                        "duration": attempt_duration,
                        "timestamp": attempt_start,
                    }
                )

                # Check if we should retry
                if not self.policy.should_retry(e, attempt):
                    logging_manager = get_logging_manager()
                    logging_manager.log_error(
                        e,
                        "error_handler",
                        operation_name=self.operation_name,
                        final_attempt=attempt,
                        retry_policy=self.policy.__class__.__name__,
                    )
                    break

                # Calculate delay before next attempt
                if attempt < self.policy.max_attempts:
                    delay = self.policy.calculate_delay(attempt)
                    logging_manager = get_logging_manager()
                    with logging_manager.context(
                        operation="retry_attempt",
                        component="error_handler",
                        operation_name=self.operation_name,
                    ):
                        logger.warning(
                            "Operation failed, will retry",
                            attempt=attempt,
                            max_attempts=self.policy.max_attempts,
                            retry_delay=delay,
                            error_type=type(e).__name__,
                            error_message=str(e),
                        )
                    time.sleep(delay)

        # All retries exhausted, raise the last exception with context
        if isinstance(last_exception, LocalDataError):
            # Add retry context to existing LocalData error
            last_exception.metadata["retry_attempts"] = len(self.attempt_history)
            last_exception.metadata["retry_history"] = self.attempt_history[
                -3:
            ]  # Last 3 attempts
            raise last_exception
        else:
            # Wrap other exceptions in a LocalData error with retry context
            raise QueryExecutionError(
                message=f"Operation '{self.operation_name}' failed after {len(self.attempt_history)} attempts: {str(last_exception)}",
                cause=last_exception,
                metadata={
                    "retry_attempts": len(self.attempt_history),
                    "retry_history": self.attempt_history[-3:],
                    "operation_name": self.operation_name,
                },
            )


def retry_on_failure(policy: RetryPolicy, operation_name: str = None):
    """Decorator for adding retry logic to functions."""

    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            name = operation_name or f"{func.__module__}.{func.__name__}"
            retryable_op = RetryableOperation(func, policy, name)
            return retryable_op.execute(*args, **kwargs)

        return wrapper

    return decorator
