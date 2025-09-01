"""Structured logging and monitoring system for LocalData MCP.

Provides comprehensive JSON-based logging with performance metrics, security events,
and debugging capabilities using structlog and prometheus_client.
"""

import os
import sys
import time
import uuid
import logging
import logging.handlers
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from contextlib import contextmanager
from dataclasses import dataclass, field

import structlog
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, REGISTRY
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from .config_manager import LoggingConfig, LogLevel


@dataclass
class LogContext:
    """Context information for structured logging."""
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    operation: Optional[str] = None
    component: Optional[str] = None
    database_name: Optional[str] = None
    query_hash: Optional[str] = None
    start_time: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary for logging."""
        return {k: v for k, v in self.__dict__.items() if v is not None}


class MetricsCollector:
    """Prometheus metrics collector for LocalData MCP."""
    
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        """Initialize metrics collector.
        
        Args:
            registry: Prometheus registry to use. Defaults to global registry.
        """
        self.registry = registry or REGISTRY
        
        # Query metrics
        self.query_counter = Counter(
            'localdata_queries_total',
            'Total number of queries executed',
            ['database_type', 'database_name', 'status'],
            registry=self.registry
        )
        
        self.query_duration = Histogram(
            'localdata_query_duration_seconds',
            'Query execution time in seconds',
            ['database_type', 'database_name'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
            registry=self.registry
        )
        
        # Connection metrics
        self.active_connections = Gauge(
            'localdata_active_connections',
            'Number of active database connections',
            ['database_name'],
            registry=self.registry
        )
        
        self.connection_pool_size = Gauge(
            'localdata_connection_pool_size',
            'Size of connection pool',
            ['database_name'],
            registry=self.registry
        )
        
        # Memory metrics
        self.memory_usage = Gauge(
            'localdata_memory_usage_bytes',
            'Current memory usage in bytes',
            ['component'],
            registry=self.registry
        )
        
        self.buffer_utilization = Gauge(
            'localdata_buffer_utilization_ratio',
            'Query buffer utilization ratio (0-1)',
            ['buffer_type'],
            registry=self.registry
        )
        
        # Error metrics
        self.error_counter = Counter(
            'localdata_errors_total',
            'Total number of errors',
            ['error_type', 'component', 'database_name'],
            registry=self.registry
        )
        
        # Security metrics
        self.security_events = Counter(
            'localdata_security_events_total',
            'Total security events',
            ['event_type', 'severity'],
            registry=self.registry
        )
        
        # Timeout metrics
        self.timeout_counter = Counter(
            'localdata_timeouts_total',
            'Total number of timeouts',
            ['timeout_type', 'database_name'],
            registry=self.registry
        )
    
    def record_query(self, database_type: str, database_name: str, 
                    duration: float, status: str):
        """Record query execution metrics."""
        self.query_counter.labels(
            database_type=database_type,
            database_name=database_name,
            status=status
        ).inc()
        
        self.query_duration.labels(
            database_type=database_type,
            database_name=database_name
        ).observe(duration)
    
    def record_error(self, error_type: str, component: str, 
                    database_name: Optional[str] = None):
        """Record error metrics."""
        self.error_counter.labels(
            error_type=error_type,
            component=component,
            database_name=database_name or "unknown"
        ).inc()
    
    def record_security_event(self, event_type: str, severity: str):
        """Record security event metrics."""
        self.security_events.labels(
            event_type=event_type,
            severity=severity
        ).inc()
    
    def record_timeout(self, timeout_type: str, database_name: str):
        """Record timeout metrics."""
        self.timeout_counter.labels(
            timeout_type=timeout_type,
            database_name=database_name
        ).inc()
    
    def update_connection_metrics(self, database_name: str, 
                                active: int, pool_size: int):
        """Update connection pool metrics."""
        self.active_connections.labels(database_name=database_name).set(active)
        self.connection_pool_size.labels(database_name=database_name).set(pool_size)
    
    def update_memory_usage(self, component: str, bytes_used: int):
        """Update memory usage metrics."""
        self.memory_usage.labels(component=component).set(bytes_used)
    
    def update_buffer_utilization(self, buffer_type: str, ratio: float):
        """Update buffer utilization metrics."""
        self.buffer_utilization.labels(buffer_type=buffer_type).set(ratio)
    
    def get_metrics(self) -> str:
        """Get metrics in Prometheus exposition format."""
        return generate_latest(self.registry)


class LoggingManager:
    """Centralized structured logging manager."""
    
    _instance: Optional['LoggingManager'] = None
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
        # Prevent re-initialization in singleton
        if hasattr(self, '_initialized'):
            return
            
        self.config = config or LoggingConfig()
        self.metrics = MetricsCollector()
        self._context = threading.local()
        self._initialized = True
        
        # Configure structlog
        self._configure_structlog()
        
        # Set up standard library logging integration
        self._configure_stdlib_logging()
        
        # Get structured logger
        self.logger = structlog.get_logger()
        
        # Log initialization
        self.logger.info("Structured logging system initialized",
                        config=self.config.__dict__)
    
    def _configure_structlog(self):
        """Configure structlog with processors and renderers."""
        processors = [
            # Add timestamp
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            # Add context
            self._add_context,
            # Add correlation ID
            self._add_correlation_id,
            # Stack info for exceptions
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
        ]
        
        # Add JSON renderer for production
        if self.config.level != LogLevel.DEBUG:
            processors.append(structlog.processors.JSONRenderer())
        else:
            # Use colorful console output for development
            processors.append(structlog.dev.ConsoleRenderer(colors=True))
        
        structlog.configure(
            processors=processors,
            wrapper_class=structlog.stdlib.BoundLogger,
            logger_factory=structlog.stdlib.LoggerFactory(),
            context_class=dict,
            cache_logger_on_first_use=True,
        )
    
    def _configure_stdlib_logging(self):
        """Configure standard library logging integration."""
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        
        # Set log level
        level_map = {
            LogLevel.DEBUG: logging.DEBUG,
            LogLevel.INFO: logging.INFO,
            LogLevel.WARNING: logging.WARNING,
            LogLevel.ERROR: logging.ERROR,
            LogLevel.CRITICAL: logging.CRITICAL
        }
        root_logger.setLevel(level_map[self.config.level])
        
        # Console handler
        if self.config.console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level_map[self.config.level])
            if self.config.level == LogLevel.DEBUG:
                formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
            else:
                formatter = logging.Formatter('%(message)s')
            console_handler.setFormatter(formatter)
            root_logger.addHandler(console_handler)
        
        # File handler with rotation
        if self.config.file_path:
            file_path = Path(self.config.file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.handlers.RotatingFileHandler(
                file_path,
                maxBytes=self.config.max_file_size,
                backupCount=self.config.backup_count
            )
            file_handler.setLevel(level_map[self.config.level])
            file_formatter = logging.Formatter('%(message)s')
            file_handler.setFormatter(file_formatter)
            root_logger.addHandler(file_handler)
    
    def _add_context(self, logger, method_name, event_dict):
        """Add context information to log entries."""
        context = getattr(self._context, 'context', None)
        if context:
            event_dict.update(context.to_dict())
        return event_dict
    
    def _add_correlation_id(self, logger, method_name, event_dict):
        """Add correlation ID to log entries."""
        if 'request_id' not in event_dict:
            event_dict['request_id'] = str(uuid.uuid4())
        return event_dict
    
    @contextmanager
    def context(self, **kwargs):
        """Context manager for structured logging context.
        
        Args:
            **kwargs: Context key-value pairs
            
        Example:
            with logging_manager.context(operation="query", database="users"):
                logger.info("Executing query")
        """
        # Get or create context
        old_context = getattr(self._context, 'context', None)
        
        if old_context:
            # Merge with existing context
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
        context = getattr(self._context, 'context', None)
        if context:
            # Update existing context
            for key, value in kwargs.items():
                setattr(context, key, value)
        else:
            # Create new context
            self._context.context = LogContext(**kwargs)
    
    def clear_context(self):
        """Clear current thread context."""
        self._context.context = None
    
    def log_query_start(self, database_name: str, query: str, 
                       database_type: str = "unknown") -> str:
        """Log query execution start and return request ID for tracking.
        
        Args:
            database_name: Name of the database
            query: SQL query being executed
            database_type: Type of database
            
        Returns:
            Request ID for correlation
        """
        request_id = str(uuid.uuid4())
        query_hash = str(hash(query))
        
        with self.context(
            request_id=request_id,
            operation="query_start",
            database_name=database_name,
            query_hash=query_hash
        ):
            self.logger.info(
                "Query execution started",
                database_type=database_type,
                query_length=len(query),
                query_preview=query[:200] if len(query) > 200 else query
            )
        
        return request_id
    
    def log_query_complete(self, request_id: str, database_name: str,
                          database_type: str, duration: float, 
                          row_count: Optional[int] = None,
                          success: bool = True):
        """Log query execution completion.
        
        Args:
            request_id: Request ID from query start
            database_name: Name of the database
            database_type: Type of database  
            duration: Query execution time in seconds
            row_count: Number of rows returned
            success: Whether query was successful
        """
        status = "success" if success else "error"
        
        with self.context(
            request_id=request_id,
            operation="query_complete",
            database_name=database_name
        ):
            self.logger.info(
                "Query execution completed",
                database_type=database_type,
                duration=duration,
                row_count=row_count,
                status=status,
                performance_category=self._categorize_performance(duration)
            )
        
        # Record metrics
        self.metrics.record_query(database_type, database_name, duration, status)
    
    def log_security_event(self, event_type: str, severity: str, 
                          description: str, **context):
        """Log security events.
        
        Args:
            event_type: Type of security event
            severity: Severity level (low, medium, high, critical)
            description: Event description
            **context: Additional context
        """
        with self.context(operation="security_event", **context):
            self.logger.warning(
                f"Security event: {description}",
                event_type=event_type,
                severity=severity,
                **context
            )
        
        # Record metrics
        self.metrics.record_security_event(event_type, severity)
    
    def log_performance_metrics(self, component: str, metrics: Dict[str, Any]):
        """Log performance metrics.
        
        Args:
            component: Component name
            metrics: Performance metrics dictionary
        """
        with self.context(operation="performance_metrics"):
            self.logger.info(
                f"Performance metrics for {component}",
                component=component,
                **metrics
            )
        
        # Update Prometheus metrics
        if 'memory_usage_bytes' in metrics:
            self.metrics.update_memory_usage(component, metrics['memory_usage_bytes'])
    
    def log_error(self, error: Exception, component: str, 
                  database_name: Optional[str] = None, **context):
        """Log error with structured information.
        
        Args:
            error: Exception instance
            component: Component where error occurred
            database_name: Database name if applicable
            **context: Additional context
        """
        error_type = type(error).__name__
        
        with self.context(operation="error", **context):
            self.logger.error(
                f"Error in {component}: {str(error)}",
                error_type=error_type,
                component=component,
                database_name=database_name,
                exc_info=True,
                **context
            )
        
        # Record metrics
        self.metrics.record_error(error_type, component, database_name)
    
    def log_timeout(self, timeout_type: str, database_name: str, 
                   timeout_value: float, **context):
        """Log timeout events.
        
        Args:
            timeout_type: Type of timeout (query, connection, etc.)
            database_name: Database name
            timeout_value: Timeout value in seconds
            **context: Additional context
        """
        with self.context(operation="timeout", **context):
            self.logger.warning(
                f"Timeout occurred: {timeout_type}",
                timeout_type=timeout_type,
                database_name=database_name,
                timeout_value=timeout_value,
                **context
            )
        
        # Record metrics
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
            Structured logger instance
        """
        return structlog.get_logger(name)


# Global logging manager instance
_logging_manager: Optional[LoggingManager] = None


def get_logging_manager(config: Optional[LoggingConfig] = None) -> LoggingManager:
    """Get global logging manager instance.
    
    Args:
        config: Logging configuration for initialization
        
    Returns:
        Global LoggingManager instance
    """
    global _logging_manager
    
    if _logging_manager is None:
        _logging_manager = LoggingManager(config)
    
    return _logging_manager


def get_logger(name: Optional[str] = None):
    """Get structured logger instance.
    
    Args:
        name: Logger name
        
    Returns:
        Structured logger instance
    """
    return get_logging_manager().get_logger(name)