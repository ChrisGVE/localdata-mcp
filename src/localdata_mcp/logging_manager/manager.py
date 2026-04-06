"""Centralized structured logging manager and module-level accessors."""

import threading
import uuid
from contextlib import contextmanager
from typing import Any, Dict, Optional

import structlog

from ..config_manager import LoggingConfig
from .config import configure_stdlib_logging, configure_structlog
from .context import LogContext
from .metrics import MetricsCollector


class LoggingManager:
    """Centralized structured logging manager."""

    _instance: Optional["LoggingManager"] = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        """Singleton implementation."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
            return cls._instance

    def __init__(self, config: Optional[LoggingConfig] = None):
        """Initialize logging manager.

        Args:
            config: Logging configuration. Uses default if None.
        """
        if hasattr(self, "_initialized"):
            return

        self.config = config or LoggingConfig()
        self.metrics = MetricsCollector()
        self._context = threading.local()
        self._initialized = True

        configure_structlog(self.config, self._add_context, self._add_correlation_id)
        configure_stdlib_logging(self.config)

        self.logger = structlog.get_logger()
        self.logger.info(
            "Structured logging system initialized",
            config=self.config.__dict__,
        )

    def _add_context(self, logger, method_name, event_dict):
        """Add context information to log entries."""
        context = getattr(self._context, "context", None)
        if context:
            event_dict.update(context.to_dict())
        return event_dict

    def _add_correlation_id(self, logger, method_name, event_dict):
        """Add correlation ID to log entries."""
        if "request_id" not in event_dict:
            event_dict["request_id"] = str(uuid.uuid4())
        return event_dict

    @contextmanager
    def context(self, **kwargs):
        """Context manager for structured logging context.

        Args:
            **kwargs: Context key-value pairs.
        """
        old_context = getattr(self._context, "context", None)

        if old_context:
            new_context = LogContext(**{**old_context.__dict__, **kwargs})
        else:
            new_context = LogContext(**kwargs)

        self._context.context = new_context

        try:
            yield new_context
        finally:
            self._context.context = old_context

    def set_context(self, **kwargs):
        """Set persistent context for current thread."""
        context = getattr(self._context, "context", None)
        if context:
            for key, value in kwargs.items():
                setattr(context, key, value)
        else:
            self._context.context = LogContext(**kwargs)

    def clear_context(self):
        """Clear current thread context."""
        self._context.context = None

    def log_query_start(
        self,
        database_name: str,
        query: str,
        database_type: str = "unknown",
    ) -> str:
        """Log query execution start and return request ID.

        Args:
            database_name: Name of the database.
            query: SQL query being executed.
            database_type: Type of database.

        Returns:
            Request ID for correlation.
        """
        request_id = str(uuid.uuid4())
        query_hash = str(hash(query))

        with self.context(
            request_id=request_id,
            operation="query_start",
            database_name=database_name,
            query_hash=query_hash,
        ):
            self.logger.info(
                "Query execution started",
                database_type=database_type,
                query_length=len(query),
                query_preview=(query[:200] if len(query) > 200 else query),
            )

        return request_id

    def log_query_complete(
        self,
        request_id: str,
        database_name: str,
        database_type: str,
        duration: float,
        row_count: Optional[int] = None,
        success: bool = True,
    ):
        """Log query execution completion.

        Args:
            request_id: Request ID from query start.
            database_name: Name of the database.
            database_type: Type of database.
            duration: Query execution time in seconds.
            row_count: Number of rows returned.
            success: Whether query was successful.
        """
        status = "success" if success else "error"

        with self.context(
            request_id=request_id,
            operation="query_complete",
            database_name=database_name,
        ):
            self.logger.info(
                "Query execution completed",
                database_type=database_type,
                duration=duration,
                row_count=row_count,
                status=status,
                performance_category=(self._categorize_performance(duration)),
            )

        self.metrics.record_query(database_type, database_name, duration, status)

    def log_security_event(
        self,
        event_type: str,
        severity: str,
        description: str,
        **context,
    ):
        """Log security events.

        Args:
            event_type: Type of security event.
            severity: Severity level.
            description: Event description.
            **context: Additional context.
        """
        with self.context(operation="security_event", **context):
            self.logger.warning(
                f"Security event: {description}",
                event_type=event_type,
                severity=severity,
                **context,
            )

        self.metrics.record_security_event(event_type, severity)

    def log_performance_metrics(self, component: str, metrics: Dict[str, Any]):
        """Log performance metrics.

        Args:
            component: Component name.
            metrics: Performance metrics dictionary.
        """
        with self.context(operation="performance_metrics"):
            self.logger.info(
                f"Performance metrics for {component}",
                component=component,
                **metrics,
            )

        if "memory_usage_bytes" in metrics:
            self.metrics.update_memory_usage(component, metrics["memory_usage_bytes"])

    def log_error(
        self,
        error: Exception,
        component: str,
        database_name: Optional[str] = None,
        **context,
    ):
        """Log error with structured information.

        Args:
            error: Exception instance.
            component: Component where error occurred.
            database_name: Database name if applicable.
            **context: Additional context.
        """
        error_type = type(error).__name__

        with self.context(operation="error", **context):
            self.logger.error(
                f"Error in {component}: {str(error)}",
                error_type=error_type,
                component=component,
                database_name=database_name,
                exc_info=True,
                **context,
            )

        self.metrics.record_error(error_type, component, database_name)

    def log_timeout(
        self,
        timeout_type: str,
        database_name: str,
        timeout_value: float,
        **context,
    ):
        """Log timeout events.

        Args:
            timeout_type: Type of timeout.
            database_name: Database name.
            timeout_value: Timeout value in seconds.
            **context: Additional context.
        """
        with self.context(operation="timeout", **context):
            self.logger.warning(
                f"Timeout occurred: {timeout_type}",
                timeout_type=timeout_type,
                database_name=database_name,
                timeout_value=timeout_value,
                **context,
            )

        self.metrics.record_timeout(timeout_type, database_name)

    def _categorize_performance(self, duration: float) -> str:
        """Categorize query performance based on duration."""
        if duration < 0.1:
            return "fast"
        elif duration < 1.0:
            return "normal"
        elif duration < 5.0:
            return "slow"
        else:
            return "very_slow"

    def get_metrics(self) -> str:
        """Get Prometheus metrics."""
        return self.metrics.get_metrics()

    def get_logger(self, name: Optional[str] = None):
        """Get structured logger instance.

        Args:
            name: Logger name. Uses caller's module name if None.

        Returns:
            Structured logger instance.
        """
        return structlog.get_logger(name)


# Global logging manager instance
_logging_manager: Optional[LoggingManager] = None


def get_logging_manager(
    config: Optional[LoggingConfig] = None,
) -> LoggingManager:
    """Get global logging manager instance.

    Args:
        config: Logging configuration for initialization.

    Returns:
        Global LoggingManager instance.
    """
    global _logging_manager

    if _logging_manager is None:
        _logging_manager = LoggingManager(config)

    return _logging_manager


def get_logger(name: Optional[str] = None):
    """Get structured logger instance.

    Args:
        name: Logger name.

    Returns:
        Structured logger instance.
    """
    return get_logging_manager().get_logger(name)
