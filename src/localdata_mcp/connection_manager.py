"""Enhanced Database Connection Management for LocalData MCP.

Provides advanced connection management with per-database configuration,
connection pooling, health monitoring, and resource management. Integrates
with the configuration system and timeout management.
"""

import asyncio
import logging
import threading
import time
import psutil
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from weakref import WeakSet

from sqlalchemy import create_engine, inspect, text, event
from sqlalchemy.engine import Engine
from sqlalchemy.pool import QueuePool, StaticPool

from .config_manager import get_config_manager, DatabaseConfig, DatabaseType
from .timeout_manager import get_timeout_manager, QueryTimeoutManager, TimeoutConfig

logger = logging.getLogger(__name__)


class ConnectionState(Enum):
    """Connection states for health monitoring."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    RECONNECTING = "reconnecting"
    DISCONNECTED = "disconnected"


class ResourceType(Enum):
    """Types of resources being monitored."""
    MEMORY = "memory"
    CONNECTIONS = "connections"
    QUERY_TIME = "query_time"
    ERROR_RATE = "error_rate"


@dataclass
class ConnectionMetrics:
    """Metrics for a database connection."""
    total_queries: int = 0
    successful_queries: int = 0
    failed_queries: int = 0
    average_query_time: float = 0.0
    total_query_time: float = 0.0
    memory_usage_mb: float = 0.0
    active_connections: int = 0
    max_connections_reached: int = 0
    last_health_check: float = field(default_factory=time.time)
    connection_errors: int = 0
    reconnect_attempts: int = 0
    last_error: Optional[str] = None

    @property
    def error_rate(self) -> float:
        """Calculate error rate as a percentage."""
        if self.total_queries == 0:
            return 0.0
        return (self.failed_queries / self.total_queries) * 100

    @property
    def success_rate(self) -> float:
        """Calculate success rate as a percentage."""
        if self.total_queries == 0:
            return 100.0
        return (self.successful_queries / self.total_queries) * 100


@dataclass
class HealthCheckResult:
    """Result of a health check operation."""
    is_healthy: bool
    state: ConnectionState
    response_time_ms: float
    error_message: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    metrics_snapshot: Optional[ConnectionMetrics] = None


@dataclass
class ResourceLimit:
    """Resource limit configuration."""
    resource_type: ResourceType
    max_value: float
    warning_threshold: float
    current_value: float = 0.0
    violations: int = 0
    last_violation: Optional[float] = None

    @property
    def is_warning(self) -> bool:
        """Check if current value exceeds warning threshold."""
        return self.current_value >= self.warning_threshold

    @property
    def is_exceeded(self) -> bool:
        """Check if current value exceeds maximum limit."""
        return self.current_value >= self.max_value


class EnhancedConnectionManager:
    """Advanced connection manager with pooling, monitoring, and health checks."""

    def __init__(self):
        """Initialize the enhanced connection manager."""
        self._config_manager = get_config_manager()
        self._timeout_manager = get_timeout_manager()
        
        # Connection pools by database name
        self._engines: Dict[str, Engine] = {}
        self._db_configs: Dict[str, DatabaseConfig] = {}
        
        # Metrics and monitoring
        self._metrics: Dict[str, ConnectionMetrics] = defaultdict(ConnectionMetrics)
        self._health_status: Dict[str, HealthCheckResult] = {}
        self._resource_limits: Dict[str, Dict[ResourceType, ResourceLimit]] = {}
        
        # Thread safety
        self._lock = threading.RLock()
        self._health_check_lock = threading.Lock()
        
        # Background monitoring
        self._monitoring_active = True
        self._health_check_interval = 30.0  # 30 seconds
        self._start_monitoring_thread()
        
        # Resource tracking
        self._query_start_times: Dict[str, float] = {}
        self._active_queries: Dict[str, Set[str]] = defaultdict(set)

    def initialize_database(self, name: str, config: Optional[DatabaseConfig] = None) -> bool:
        """Initialize a database connection with advanced configuration.
        
        Args:
            name: Database name/identifier
            config: Optional database configuration. If None, loads from config manager.
            
        Returns:
            bool: Whether initialization was successful
        """
        try:
            with self._lock:
                # Get configuration
                if config is None:
                    config = self._config_manager.get_database_config(name)
                    if config is None:
                        logger.error(f"Database configuration not found for '{name}'")
                        return False
                
                # Store configuration
                self._db_configs[name] = config
                
                # Create engine with advanced configuration
                engine = self._create_enhanced_engine(name, config)
                self._engines[name] = engine
                
                # Initialize metrics
                self._metrics[name] = ConnectionMetrics()
                
                # Setup resource limits
                self._setup_resource_limits(name, config)
                
                # Perform initial health check
                health_result = self._perform_health_check(name)
                self._health_status[name] = health_result
                
                if health_result.is_healthy:
                    logger.info(f"Successfully initialized database '{name}' "
                              f"(type: {config.type.value}, state: {health_result.state.value})")
                    return True
                else:
                    logger.warning(f"Database '{name}' initialized but unhealthy: {health_result.error_message}")
                    return False
                    
        except Exception as e:
            logger.error(f"Failed to initialize database '{name}': {e}")
            return False

    def get_engine(self, name: str) -> Optional[Engine]:
        """Get database engine for a specific database.
        
        Args:
            name: Database name
            
        Returns:
            Optional[Engine]: Database engine if available
        """
        with self._lock:
            if name not in self._engines:
                # Try to initialize if configuration exists
                config = self._config_manager.get_database_config(name)
                if config and config.enabled:
                    if self.initialize_database(name, config):
                        return self._engines.get(name)
                return None
            
            return self._engines[name]

    def get_connection_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive connection information for a database.
        
        Args:
            name: Database name
            
        Returns:
            Optional[Dict[str, Any]]: Connection information
        """
        with self._lock:
            if name not in self._engines:
                return None
            
            config = self._db_configs[name]
            metrics = self._metrics[name]
            health = self._health_status.get(name)
            limits = self._resource_limits.get(name, {})
            
            return {
                "name": name,
                "type": config.type.value,
                "enabled": config.enabled,
                "tags": config.tags,
                "connection_config": {
                    "max_connections": config.max_connections,
                    "connection_timeout": config.connection_timeout,
                    "query_timeout": config.query_timeout
                },
                "metrics": {
                    "total_queries": metrics.total_queries,
                    "success_rate": metrics.success_rate,
                    "error_rate": metrics.error_rate,
                    "average_query_time": metrics.average_query_time,
                    "memory_usage_mb": metrics.memory_usage_mb,
                    "active_connections": metrics.active_connections
                },
                "health": {
                    "state": health.state.value if health else "unknown",
                    "is_healthy": health.is_healthy if health else False,
                    "last_check": health.timestamp if health else 0,
                    "response_time_ms": health.response_time_ms if health else 0
                },
                "resource_limits": {
                    rt.value: {
                        "max_value": limit.max_value,
                        "current_value": limit.current_value,
                        "warning_threshold": limit.warning_threshold,
                        "is_warning": limit.is_warning,
                        "is_exceeded": limit.is_exceeded,
                        "violations": limit.violations
                    }
                    for rt, limit in limits.items()
                }
            }

    def list_databases(self, include_tags: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """List all configured databases with optional tag filtering.
        
        Args:
            include_tags: Optional list of tags to filter by
            
        Returns:
            List[Dict[str, Any]]: List of database information
        """
        databases = []
        
        # Get all configured databases
        all_configs = self._config_manager.get_database_configs()
        
        for name, config in all_configs.items():
            # Apply tag filtering
            if include_tags:
                if not any(tag in config.tags for tag in include_tags):
                    continue
            
            db_info = self.get_connection_info(name)
            if db_info:
                databases.append(db_info)
            else:
                # Include unconfigured databases
                databases.append({
                    "name": name,
                    "type": config.type.value,
                    "enabled": config.enabled,
                    "tags": config.tags,
                    "initialized": False
                })
        
        return databases

    def get_databases_by_tag(self, tag: str) -> List[str]:
        """Get all database names that have a specific tag.
        
        Args:
            tag: Tag to search for
            
        Returns:
            List[str]: List of database names with the tag
        """
        databases = []
        all_configs = self._config_manager.get_database_configs()
        
        for name, config in all_configs.items():
            if tag in config.tags:
                databases.append(name)
        
        return databases

    @contextmanager
    def managed_query_execution(self, database_name: str, query_id: Optional[str] = None):
        """Context manager for tracked query execution with resource monitoring.
        
        Args:
            database_name: Name of the database
            query_id: Optional query identifier for tracking
            
        Yields:
            dict: Query execution context
        """
        query_id = query_id or f"query_{int(time.time() * 1000)}"
        start_time = time.time()
        
        with self._lock:
            # Track query start
            self._query_start_times[query_id] = start_time
            self._active_queries[database_name].add(query_id)
            
            # Update metrics
            metrics = self._metrics[database_name]
            metrics.total_queries += 1
            metrics.active_connections = len(self._active_queries[database_name])
        
        context = {
            'query_id': query_id,
            'database_name': database_name,
            'start_time': start_time,
            'resource_checks': []
        }
        
        try:
            # Check resource limits before execution
            self._check_resource_limits(database_name)
            
            yield context
            
            # Query succeeded
            with self._lock:
                metrics = self._metrics[database_name]
                metrics.successful_queries += 1
                
        except Exception as e:
            # Query failed
            with self._lock:
                metrics = self._metrics[database_name]
                metrics.failed_queries += 1
                metrics.last_error = str(e)
                
                # Update connection errors for connection-related failures
                if any(keyword in str(e).lower() for keyword in ['connection', 'timeout', 'network']):
                    metrics.connection_errors += 1
                    
            raise
            
        finally:
            # Clean up tracking
            end_time = time.time()
            execution_time = end_time - start_time
            
            with self._lock:
                # Update timing metrics
                metrics = self._metrics[database_name]
                metrics.total_query_time += execution_time
                metrics.average_query_time = metrics.total_query_time / metrics.total_queries
                
                # Remove from active tracking
                self._active_queries[database_name].discard(query_id)
                self._query_start_times.pop(query_id, None)
                
                # Update resource usage
                self._update_resource_usage(database_name)

    def close_database(self, name: str) -> bool:
        """Close and clean up a database connection.
        
        Args:
            name: Database name to close
            
        Returns:
            bool: Whether cleanup was successful
        """
        try:
            with self._lock:
                if name in self._engines:
                    engine = self._engines[name]
                    engine.dispose()
                    del self._engines[name]
                    
                # Clean up associated data
                self._db_configs.pop(name, None)
                self._metrics.pop(name, None)
                self._health_status.pop(name, None)
                self._resource_limits.pop(name, None)
                self._active_queries.pop(name, None)
                
                # Clean up query tracking
                query_ids_to_remove = [
                    qid for qid, start_time in self._query_start_times.items() 
                    if qid.startswith(f"{name}_")
                ]
                for qid in query_ids_to_remove:
                    self._query_start_times.pop(qid, None)
                
                logger.info(f"Successfully closed database connection '{name}'")
                return True
                
        except Exception as e:
            logger.error(f"Error closing database '{name}': {e}")
            return False

    def close_all(self):
        """Close all database connections and stop monitoring."""
        self._monitoring_active = False
        
        with self._lock:
            database_names = list(self._engines.keys())
            
        for name in database_names:
            self.close_database(name)
            
        logger.info("All database connections closed")

    def get_health_status(self, name: str) -> Optional[HealthCheckResult]:
        """Get the latest health check result for a database.
        
        Args:
            name: Database name
            
        Returns:
            Optional[HealthCheckResult]: Health check result
        """
        return self._health_status.get(name)

    def trigger_health_check(self, name: str) -> Optional[HealthCheckResult]:
        """Manually trigger a health check for a specific database.
        
        Args:
            name: Database name
            
        Returns:
            Optional[HealthCheckResult]: Health check result
        """
        if name not in self._engines:
            return None
            
        result = self._perform_health_check(name)
        with self._lock:
            self._health_status[name] = result
        
        return result

    def get_resource_status(self, name: str) -> Dict[str, Dict[str, Any]]:
        """Get resource usage and limits for a database.
        
        Args:
            name: Database name
            
        Returns:
            Dict[str, Dict[str, Any]]: Resource status information
        """
        with self._lock:
            limits = self._resource_limits.get(name, {})
            
            return {
                resource_type.value: {
                    "current_value": limit.current_value,
                    "max_value": limit.max_value,
                    "warning_threshold": limit.warning_threshold,
                    "is_warning": limit.is_warning,
                    "is_exceeded": limit.is_exceeded,
                    "violations": limit.violations,
                    "last_violation": limit.last_violation
                }
                for resource_type, limit in limits.items()
            }

    def _create_enhanced_engine(self, name: str, config: DatabaseConfig) -> Engine:
        """Create a SQLAlchemy engine with enhanced configuration.
        
        Args:
            name: Database name
            config: Database configuration
            
        Returns:
            Engine: Configured SQLAlchemy engine
        """
        # Base engine arguments
        engine_args = {
            'echo': False,  # Set to True for SQL debugging
        }
        
        # Configure connection pooling
        if config.type in {DatabaseType.POSTGRESQL, DatabaseType.MYSQL}:
            # Use connection pooling for network databases
            engine_args.update({
                'poolclass': QueuePool,
                'pool_size': min(config.max_connections, 10),
                'max_overflow': max(0, config.max_connections - 10),
                'pool_timeout': config.connection_timeout,
                'pool_recycle': 3600,  # 1 hour
                'pool_pre_ping': True,  # Verify connections before use
            })
        elif config.type in {DatabaseType.SQLITE, DatabaseType.DUCKDB}:
            # Use static pool for file-based databases
            engine_args.update({
                'poolclass': StaticPool,
                'pool_size': 1,
                'max_overflow': 0,
                'connect_args': {'check_same_thread': False}
            })
        
        # Create engine based on database type
        if config.type == DatabaseType.SQLITE:
            engine = create_engine(f"sqlite:///{config.connection_string}", **engine_args)
        elif config.type == DatabaseType.POSTGRESQL:
            engine = create_engine(config.connection_string, **engine_args)
        elif config.type == DatabaseType.MYSQL:
            engine = create_engine(config.connection_string, **engine_args)
        elif config.type == DatabaseType.DUCKDB:
            engine = create_engine(f"duckdb:///{config.connection_string}", **engine_args)
        else:
            # For other database types, use basic engine creation
            # File formats will be handled by the existing file processing logic
            if config.type.value in ["csv", "json", "yaml", "excel", "ods", "numbers"]:
                # Create in-memory SQLite for file formats
                engine = create_engine("sqlite:///:memory:", **engine_args)
            else:
                raise ValueError(f"Unsupported database type: {config.type.value}")
        
        # Set up event listeners for monitoring
        self._setup_engine_events(engine, name)
        
        return engine

    def _setup_engine_events(self, engine: Engine, database_name: str):
        """Set up SQLAlchemy event listeners for monitoring.
        
        Args:
            engine: SQLAlchemy engine
            database_name: Database name for tracking
        """
        @event.listens_for(engine, "before_cursor_execute")
        def before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
            context._query_start_time = time.time()

        @event.listens_for(engine, "after_cursor_execute")
        def after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
            if hasattr(context, '_query_start_time'):
                query_time = time.time() - context._query_start_time
                with self._lock:
                    metrics = self._metrics[database_name]
                    metrics.total_query_time += query_time
                    if metrics.total_queries > 0:
                        metrics.average_query_time = metrics.total_query_time / metrics.total_queries

    def _setup_resource_limits(self, name: str, config: DatabaseConfig):
        """Set up resource limits for a database.
        
        Args:
            name: Database name
            config: Database configuration
        """
        perf_config = self._config_manager.get_performance_config()
        
        limits = {
            ResourceType.MEMORY: ResourceLimit(
                resource_type=ResourceType.MEMORY,
                max_value=perf_config.memory_limit_mb,
                warning_threshold=perf_config.memory_limit_mb * perf_config.memory_warning_threshold
            ),
            ResourceType.CONNECTIONS: ResourceLimit(
                resource_type=ResourceType.CONNECTIONS,
                max_value=config.max_connections,
                warning_threshold=config.max_connections * 0.8
            ),
            ResourceType.QUERY_TIME: ResourceLimit(
                resource_type=ResourceType.QUERY_TIME,
                max_value=config.query_timeout,
                warning_threshold=config.query_timeout * 0.8
            ),
            ResourceType.ERROR_RATE: ResourceLimit(
                resource_type=ResourceType.ERROR_RATE,
                max_value=20.0,  # 20% error rate
                warning_threshold=10.0  # 10% warning threshold
            )
        }
        
        with self._lock:
            self._resource_limits[name] = limits

    def _check_resource_limits(self, database_name: str):
        """Check if resource limits are being exceeded.
        
        Args:
            database_name: Database name to check
            
        Raises:
            ResourceError: If critical limits are exceeded
        """
        with self._lock:
            limits = self._resource_limits.get(database_name, {})
            metrics = self._metrics[database_name]
            
            # Update current values
            limits[ResourceType.CONNECTIONS].current_value = len(self._active_queries[database_name])
            limits[ResourceType.ERROR_RATE].current_value = metrics.error_rate
            
            # Check for violations
            current_time = time.time()
            for limit in limits.values():
                if limit.is_exceeded:
                    limit.violations += 1
                    limit.last_violation = current_time
                    
                    if limit.resource_type == ResourceType.CONNECTIONS:
                        raise Exception(f"Maximum connections exceeded: {limit.current_value}/{limit.max_value}")
                    elif limit.resource_type == ResourceType.ERROR_RATE:
                        logger.warning(f"High error rate detected for '{database_name}': {limit.current_value:.1f}%")

    def _update_resource_usage(self, database_name: str):
        """Update resource usage metrics for a database.
        
        Args:
            database_name: Database name
        """
        try:
            # Get current memory usage
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            
            with self._lock:
                metrics = self._metrics[database_name]
                metrics.memory_usage_mb = memory_mb
                metrics.active_connections = len(self._active_queries[database_name])
                
                # Update resource limits
                if database_name in self._resource_limits:
                    limits = self._resource_limits[database_name]
                    limits[ResourceType.MEMORY].current_value = memory_mb
                    limits[ResourceType.CONNECTIONS].current_value = metrics.active_connections
                    limits[ResourceType.ERROR_RATE].current_value = metrics.error_rate
                    
        except Exception as e:
            logger.warning(f"Failed to update resource usage for '{database_name}': {e}")

    def _perform_health_check(self, name: str) -> HealthCheckResult:
        """Perform a health check on a database connection.
        
        Args:
            name: Database name
            
        Returns:
            HealthCheckResult: Health check result
        """
        if name not in self._engines:
            return HealthCheckResult(
                is_healthy=False,
                state=ConnectionState.DISCONNECTED,
                response_time_ms=0.0,
                error_message="Engine not found"
            )
        
        engine = self._engines[name]
        config = self._db_configs[name]
        start_time = time.time()
        
        try:
            # Perform simple health check query
            with engine.connect() as conn:
                if config.type == DatabaseType.SQLITE:
                    result = conn.execute(text("SELECT 1"))
                elif config.type in {DatabaseType.POSTGRESQL, DatabaseType.MYSQL}:
                    result = conn.execute(text("SELECT 1"))
                elif config.type == DatabaseType.DUCKDB:
                    result = conn.execute(text("SELECT 1"))
                else:
                    # For file-based databases, just check if engine is accessible
                    result = True
                
                response_time = (time.time() - start_time) * 1000  # Convert to milliseconds
                
                # Determine health state based on response time and metrics
                metrics = self._metrics[name]
                if response_time < 100 and metrics.error_rate < 5:
                    state = ConnectionState.HEALTHY
                elif response_time < 500 and metrics.error_rate < 15:
                    state = ConnectionState.DEGRADED
                else:
                    state = ConnectionState.UNHEALTHY
                
                return HealthCheckResult(
                    is_healthy=state in {ConnectionState.HEALTHY, ConnectionState.DEGRADED},
                    state=state,
                    response_time_ms=response_time,
                    metrics_snapshot=metrics
                )
                
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            logger.warning(f"Health check failed for '{name}': {e}")
            
            return HealthCheckResult(
                is_healthy=False,
                state=ConnectionState.UNHEALTHY,
                response_time_ms=response_time,
                error_message=str(e)
            )

    def _start_monitoring_thread(self):
        """Start background thread for health monitoring."""
        def monitor():
            while self._monitoring_active:
                try:
                    # Perform health checks
                    with self._lock:
                        database_names = list(self._engines.keys())
                    
                    for name in database_names:
                        if not self._monitoring_active:
                            break
                        
                        try:
                            result = self._perform_health_check(name)
                            with self._lock:
                                self._health_status[name] = result
                                
                            # Update metrics timestamp
                            with self._lock:
                                self._metrics[name].last_health_check = time.time()
                                
                        except Exception as e:
                            logger.error(f"Error during health check for '{name}': {e}")
                    
                    # Sleep until next check
                    time.sleep(self._health_check_interval)
                    
                except Exception as e:
                    logger.error(f"Error in monitoring thread: {e}")
                    time.sleep(5)  # Short sleep before retry
        
        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()
        logger.info("Started database health monitoring thread")


# Global enhanced connection manager instance
_enhanced_connection_manager: Optional[EnhancedConnectionManager] = None


def get_enhanced_connection_manager() -> EnhancedConnectionManager:
    """Get or create global enhanced connection manager instance.
    
    Returns:
        EnhancedConnectionManager: Global connection manager
    """
    global _enhanced_connection_manager
    if _enhanced_connection_manager is None:
        _enhanced_connection_manager = EnhancedConnectionManager()
    return _enhanced_connection_manager


def initialize_enhanced_connection_manager() -> EnhancedConnectionManager:
    """Initialize a new global enhanced connection manager instance.
    
    Returns:
        EnhancedConnectionManager: New connection manager
    """
    global _enhanced_connection_manager
    if _enhanced_connection_manager is not None:
        _enhanced_connection_manager.close_all()
    _enhanced_connection_manager = EnhancedConnectionManager()
    return _enhanced_connection_manager