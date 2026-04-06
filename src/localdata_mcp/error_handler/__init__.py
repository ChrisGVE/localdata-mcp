"""Advanced Error Handling System for LocalData MCP v1.3.1.

This package provides comprehensive error handling with custom exception hierarchy,
retry mechanisms with exponential backoff, circuit breaker pattern for database
connections, and integration with existing security, timeout, and connection systems.

Key Features:
- Custom exception hierarchy for different error types
- Retry policies with exponential backoff and jitter
- Circuit breaker pattern for database connection failures
- Error recovery strategies and graceful degradation
- Comprehensive error logging and metadata preservation
- Integration with SecurityManager, ConnectionManager, and TimeoutManager
"""

from .circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerRegistry,
    CircuitBreakerStats,
    circuit_breaker_protection,
)
from .error_logging import (
    ErrorLogger,
    ErrorMetrics,
)
from .exceptions import (
    CircuitState,
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
from .handler import (
    ErrorHandler,
    get_circuit_breaker,
    get_error_handler,
    handle_error,
    initialize_error_handler,
)
from .recovery import (
    ErrorRecoveryManager,
    RecoveryAction,
    RecoveryStrategy,
)
from .retry import (
    RetryableOperation,
    RetryPolicy,
    retry_on_failure,
)

__all__ = [
    # Enums
    "ErrorCategory",
    "ErrorSeverity",
    "RetryStrategy",
    "CircuitState",
    # Exception hierarchy
    "LocalDataError",
    "DatabaseConnectionError",
    "QueryExecutionError",
    "SecurityViolationError",
    "QueryTimeoutError",
    "ResourceExhaustionError",
    "ConfigurationError",
    # Retry mechanism
    "RetryPolicy",
    "RetryableOperation",
    "retry_on_failure",
    # Circuit breaker
    "CircuitBreakerConfig",
    "CircuitBreakerStats",
    "CircuitBreaker",
    "CircuitBreakerRegistry",
    "circuit_breaker_protection",
    # Recovery
    "RecoveryStrategy",
    "RecoveryAction",
    "ErrorRecoveryManager",
    # Logging
    "ErrorMetrics",
    "ErrorLogger",
    # Main handler
    "ErrorHandler",
    "get_error_handler",
    "initialize_error_handler",
    "handle_error",
    "get_circuit_breaker",
]
