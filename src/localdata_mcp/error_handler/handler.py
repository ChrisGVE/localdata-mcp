"""Main error handler that coordinates all error handling components.

Provides the ErrorHandler class and global convenience functions.
"""

import threading
import time
from typing import Any, Dict, Optional, Tuple

from ..logging_manager import get_logger
from .circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerRegistry,
)
from .error_logging import ErrorLogger
from .exceptions import (
    ConfigurationError,
    DatabaseConnectionError,
    ErrorCategory,
    ErrorSeverity,
    LocalDataError,
    QueryExecutionError,
    QueryTimeoutError,
    ResourceExhaustionError,
    RetryStrategy,
    SecurityViolationError,
)
from .recovery import ErrorRecoveryManager
from .retry import RetryPolicy

# Get structured logger
logger = get_logger(__name__)


# ============================================================================
# Integration Utilities and Main Error Handler
# ============================================================================


class ErrorHandler:
    """Main error handler that coordinates all error handling components."""

    def __init__(self):
        self.logger = ErrorLogger()
        self.recovery_manager = ErrorRecoveryManager()
        self.circuit_breaker_registry = CircuitBreakerRegistry()
        self.lock = threading.RLock()

        # Default retry policies for different error types
        self.retry_policies = {
            ErrorCategory.CONNECTION: RetryPolicy(
                strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
                max_attempts=3,
                base_delay=1.0,
                max_delay=30.0,
                retry_on=(DatabaseConnectionError,),
                stop_on=(SecurityViolationError, ConfigurationError),
            ),
            ErrorCategory.TIMEOUT: RetryPolicy(
                strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
                max_attempts=2,
                base_delay=5.0,
                max_delay=60.0,
                retry_on=(QueryTimeoutError,),
            ),
            ErrorCategory.RESOURCE_EXHAUSTION: RetryPolicy(
                strategy=RetryStrategy.LINEAR_BACKOFF,
                max_attempts=2,
                base_delay=10.0,
                max_delay=120.0,
                retry_on=(ResourceExhaustionError,),
            ),
        }

    def handle_error(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
        attempt_recovery: bool = True,
    ) -> Tuple[bool, Any, Optional[LocalDataError]]:
        """Main error handling entry point.

        Args:
            error: The exception that occurred
            context: Additional context for error handling
            attempt_recovery: Whether to attempt automatic recovery

        Returns:
            Tuple[bool, Any, Optional[LocalDataError]]: (recovered, result, processed_error)
        """
        context = context or {}

        # Convert to LocalDataError if needed
        if isinstance(error, LocalDataError):
            processed_error = error
        else:
            processed_error = self._convert_to_localdata_error(error, context)

        # Log the error
        self.logger.log_error(processed_error, context)

        # Attempt recovery if requested
        recovery_result = None
        if attempt_recovery:
            recovered, recovery_result, attempted_strategies = (
                self.recovery_manager.attempt_recovery(processed_error, context)
            )

            if recovered:
                self.logger.logger.info(
                    f"Error recovery successful for {processed_error.error_code} using strategies: {attempted_strategies}"
                )
                return True, recovery_result, processed_error
            else:
                self.logger.logger.warning(
                    f"Error recovery failed for {processed_error.error_code} after attempting: {attempted_strategies}"
                )

        return False, None, processed_error

    def get_circuit_breaker(
        self, name: str, config: Optional[CircuitBreakerConfig] = None
    ) -> CircuitBreaker:
        """Get a circuit breaker for a specific resource."""
        return self.circuit_breaker_registry.get_breaker(name, config)

    def get_retry_policy(self, category: ErrorCategory) -> Optional[RetryPolicy]:
        """Get retry policy for an error category."""
        return self.retry_policies.get(category)

    def register_retry_policy(self, category: ErrorCategory, policy: RetryPolicy):
        """Register a retry policy for an error category."""
        with self.lock:
            self.retry_policies[category] = policy

    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status."""
        error_health = self.logger.get_health_status()
        circuit_breaker_status = self.circuit_breaker_registry.get_status_summary()
        recovery_stats = self.recovery_manager.get_recovery_statistics()
        error_stats = self.logger.get_error_statistics()

        return {
            "overall_health": error_health,
            "circuit_breakers": circuit_breaker_status,
            "recovery_statistics": recovery_stats,
            "error_statistics": error_stats,
            "timestamp": time.time(),
        }

    def classify_database_error(
        self,
        exception: Exception,
        db_type: str = "generic",
        database_name: Optional[str] = None,
    ) -> "StructuredErrorResponse":
        """Classify a database exception into a structured error response."""
        from ..error_classification import ErrorMapperRegistry, StructuredErrorResponse

        mapper = ErrorMapperRegistry.get_or_default(db_type)
        response = mapper.map_error(exception)
        if database_name:
            response.database = database_name

        logger.warning(
            "Database error classified",
            error_type=response.error_type.value,
            is_retryable=response.is_retryable,
            database=database_name,
            db_type=db_type,
        )
        return response

    def _convert_to_localdata_error(
        self, error: Exception, context: Dict[str, Any]
    ) -> LocalDataError:
        """Convert a generic exception to a LocalDataError."""
        error_message = str(error)
        error_type = type(error).__name__

        # Determine category based on error type and message
        if any(
            keyword in error_message.lower()
            for keyword in ["connection", "network", "host", "port"]
        ):
            category = ErrorCategory.CONNECTION
            error_class = DatabaseConnectionError
        elif any(
            keyword in error_message.lower()
            for keyword in ["timeout", "time", "expired"]
        ):
            category = ErrorCategory.TIMEOUT
            error_class = QueryTimeoutError
        elif any(
            keyword in error_message.lower()
            for keyword in ["memory", "resource", "limit"]
        ):
            category = ErrorCategory.RESOURCE_EXHAUSTION
            error_class = ResourceExhaustionError
        elif any(
            keyword in error_message.lower()
            for keyword in ["security", "permission", "access"]
        ):
            category = ErrorCategory.SECURITY_VIOLATION
            error_class = SecurityViolationError
        elif any(
            keyword in error_message.lower()
            for keyword in ["config", "setting", "parameter"]
        ):
            category = ErrorCategory.CONFIGURATION
            error_class = ConfigurationError
        else:
            category = ErrorCategory.QUERY_EXECUTION
            error_class = QueryExecutionError

        # Determine severity
        severity = ErrorSeverity.MEDIUM
        if any(
            keyword in error_message.lower()
            for keyword in ["critical", "fatal", "severe"]
        ):
            severity = ErrorSeverity.CRITICAL
        elif any(
            keyword in error_message.lower()
            for keyword in ["error", "fail", "exception"]
        ):
            severity = ErrorSeverity.HIGH
        elif any(keyword in error_message.lower() for keyword in ["warning", "warn"]):
            severity = ErrorSeverity.LOW

        return error_class(
            message=f"{error_type}: {error_message}",
            database_name=context.get("database_name"),
            query=context.get("query"),
            metadata={"original_error_type": error_type, "context": context},
            cause=error,
            severity=severity,
        )


# Global error handler instance
_error_handler: Optional[ErrorHandler] = None


def get_error_handler() -> ErrorHandler:
    """Get or create global error handler instance."""
    global _error_handler
    if _error_handler is None:
        _error_handler = ErrorHandler()
    return _error_handler


def initialize_error_handler() -> ErrorHandler:
    """Initialize a new global error handler instance."""
    global _error_handler
    _error_handler = ErrorHandler()
    return _error_handler


# Convenience functions for common operations
def handle_error(
    error: Exception, context: Optional[Dict[str, Any]] = None
) -> Tuple[bool, Any, Optional[LocalDataError]]:
    """Handle an error using the global error handler."""
    return get_error_handler().handle_error(error, context)


def get_circuit_breaker(
    name: str, config: Optional[CircuitBreakerConfig] = None
) -> CircuitBreaker:
    """Get a circuit breaker using the global error handler."""
    return get_error_handler().get_circuit_breaker(name, config)
