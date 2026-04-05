"""Custom exception hierarchy and error enums for LocalData MCP.

Provides ErrorCategory, ErrorSeverity, RetryStrategy, CircuitState enums
and the LocalDataError base exception with specialized subclasses.
"""

import time
from enum import Enum
from typing import Any, Dict, List, Optional


class ErrorCategory(Enum):
    """Categories of errors for classification and handling."""

    CONNECTION = "connection"  # deprecated — use CONNECTION_ERROR
    QUERY_EXECUTION = "query_execution"
    SECURITY_VIOLATION = "security_violation"
    TIMEOUT = "timeout"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    CONFIGURATION = "configuration"
    AUTHENTICATION = "authentication"
    PERMISSION = "permission"
    DATA_VALIDATION = "data_validation"
    SYSTEM = "system"
    # PRD-specified structured error classification categories
    AUTH_ERROR = "auth_error"
    SCHEMA_ERROR = "schema_error"
    SYNTAX_ERROR = "syntax_error"
    RESOURCE_ERROR = "resource_error"
    TRANSIENT_ERROR = "transient_error"
    CONSTRAINT_ERROR = "constraint_error"
    CONNECTION_ERROR = "connection_error"


class ErrorSeverity(Enum):
    """Error severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RetryStrategy(Enum):
    """Retry strategy types."""

    FIXED = "fixed"
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    FIBONACCI = "fibonacci"
    NO_RETRY = "no_retry"


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Circuit is open, rejecting requests
    HALF_OPEN = "half_open"  # Testing if service has recovered


# ============================================================================
# Custom Exception Hierarchy
# ============================================================================


class LocalDataError(Exception):
    """Base exception for all LocalData MCP errors.

    Provides structured error information with metadata, context, and recovery suggestions.
    """

    def __init__(
        self,
        message: str,
        category: ErrorCategory,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        error_code: Optional[str] = None,
        database_name: Optional[str] = None,
        query: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
        recovery_suggestions: Optional[List[str]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.category = category
        self.severity = severity
        self.error_code = error_code or f"{category.value}_{int(time.time())}"
        self.database_name = database_name
        self.query = query
        self.metadata = metadata or {}
        self.cause = cause
        self.recovery_suggestions = recovery_suggestions or []
        self.timestamp = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for logging and serialization."""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "category": self.category.value,
            "severity": self.severity.value,
            "database_name": self.database_name,
            "query": self.query[:200] if self.query else None,
            "metadata": self.metadata,
            "cause": str(self.cause) if self.cause else None,
            "recovery_suggestions": self.recovery_suggestions,
            "timestamp": self.timestamp,
        }

    def to_structured_response(self) -> "StructuredErrorResponse":
        """Convert to StructuredErrorResponse for LLM communication."""
        from ..error_classification import StructuredErrorResponse

        # Map legacy categories to new ones
        category_map = {
            ErrorCategory.CONNECTION: ErrorCategory.CONNECTION_ERROR,
            ErrorCategory.TIMEOUT: ErrorCategory.TRANSIENT_ERROR,
            ErrorCategory.RESOURCE_EXHAUSTION: ErrorCategory.RESOURCE_ERROR,
            ErrorCategory.AUTHENTICATION: ErrorCategory.AUTH_ERROR,
            ErrorCategory.SECURITY_VIOLATION: ErrorCategory.AUTH_ERROR,
        }
        error_type = category_map.get(self.category, self.category)

        is_retryable = error_type in (
            ErrorCategory.TRANSIENT_ERROR,
            ErrorCategory.CONNECTION_ERROR,
        )

        suggestion = self.recovery_suggestions[0] if self.recovery_suggestions else ""

        return StructuredErrorResponse(
            error_type=error_type,
            is_retryable=is_retryable,
            message=self.message,
            suggestion=suggestion,
            database_error_code=self.error_code,
            database=self.database_name,
        )

    def __str__(self) -> str:
        return f"[{self.category.value.upper()}] {self.message}"


class DatabaseConnectionError(LocalDataError):
    """Database connection related errors."""

    def __init__(self, message: str, database_name: Optional[str] = None, **kwargs):
        kwargs.setdefault(
            "recovery_suggestions",
            [
                "Check database connection parameters",
                "Verify network connectivity",
                "Ensure database server is running",
                "Check firewall settings",
            ],
        )
        super().__init__(
            message=message,
            category=ErrorCategory.CONNECTION,
            database_name=database_name,
            **kwargs,
        )


class QueryExecutionError(LocalDataError):
    """Query execution related errors."""

    def __init__(self, message: str, query: Optional[str] = None, **kwargs):
        kwargs.setdefault(
            "recovery_suggestions",
            [
                "Check SQL syntax",
                "Verify table and column names exist",
                "Ensure sufficient permissions",
                "Consider query optimization",
            ],
        )
        super().__init__(
            message=message,
            category=ErrorCategory.QUERY_EXECUTION,
            query=query,
            **kwargs,
        )


class SecurityViolationError(LocalDataError):
    """Security violation errors."""

    def __init__(self, message: str, query: Optional[str] = None, **kwargs):
        kwargs.setdefault(
            "recovery_suggestions",
            [
                "Review query for security issues",
                "Check SQL injection patterns",
                "Verify query permissions",
                "Contact administrator if needed",
            ],
        )
        super().__init__(
            message=message,
            category=ErrorCategory.SECURITY_VIOLATION,
            severity=ErrorSeverity.HIGH,
            query=query,
            **kwargs,
        )


class QueryTimeoutError(LocalDataError):
    """Query timeout related errors."""

    def __init__(
        self,
        message: str,
        execution_time: float = 0.0,
        timeout_limit: float = 0.0,
        **kwargs,
    ):
        metadata = kwargs.pop("metadata", {})
        metadata.update(
            {"execution_time": execution_time, "timeout_limit": timeout_limit}
        )
        kwargs.setdefault(
            "recovery_suggestions",
            [
                "Increase query timeout limit",
                "Optimize query performance",
                "Add appropriate indexes",
                "Consider data partitioning",
            ],
        )
        super().__init__(
            message=message,
            category=ErrorCategory.TIMEOUT,
            metadata=metadata,
            **kwargs,
        )


class ResourceExhaustionError(LocalDataError):
    """Resource exhaustion errors (memory, CPU, connections)."""

    def __init__(self, message: str, resource_type: str = "unknown", **kwargs):
        metadata = kwargs.pop("metadata", {})
        metadata["resource_type"] = resource_type
        kwargs.setdefault(
            "recovery_suggestions",
            [
                "Reduce query complexity",
                "Implement data pagination",
                "Close unused connections",
                "Increase system resources",
            ],
        )
        super().__init__(
            message=message,
            category=ErrorCategory.RESOURCE_EXHAUSTION,
            severity=ErrorSeverity.HIGH,
            metadata=metadata,
            **kwargs,
        )


class ConfigurationError(LocalDataError):
    """Configuration related errors."""

    def __init__(self, message: str, config_key: Optional[str] = None, **kwargs):
        metadata = kwargs.pop("metadata", {})
        if config_key:
            metadata["config_key"] = config_key
        kwargs.setdefault(
            "recovery_suggestions",
            [
                "Check configuration file syntax",
                "Verify all required parameters",
                "Review default values",
                "Consult documentation",
            ],
        )
        super().__init__(
            message=message,
            category=ErrorCategory.CONFIGURATION,
            metadata=metadata,
            **kwargs,
        )
