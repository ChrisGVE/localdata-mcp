"""Enhanced Database Connection Manager implementation.

Provides advanced connection management with per-database configuration,
connection pooling, health monitoring, and resource management. Integrates
with the configuration system and timeout management.
"""

import logging
import threading
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set

from sqlalchemy.engine import Engine

from ..config_manager import get_config_manager, DatabaseConfig
from ..timeout_manager import get_timeout_manager
from .engine_factory import EngineFactoryMixin
from .health import HealthMonitorMixin
from .models import (
    ConnectionMetrics,
    HealthCheckResult,
    ResourceLimit,
    ResourceType,
)
from .query_tracking import QueryTrackingMixin
from .resources import ResourceManagerMixin

logger = logging.getLogger(__name__)


class EnhancedConnectionManager(
    EngineFactoryMixin,
    HealthMonitorMixin,
    QueryTrackingMixin,
    ResourceManagerMixin,
):
    """Advanced connection manager with pooling, monitoring, and health checks."""

    def __init__(self) -> None:
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

    def initialize_database(
        self, name: str, config: Optional[DatabaseConfig] = None
    ) -> bool:
        """Initialize a database connection with advanced configuration.

        Args:
            name: Database name/identifier.
            config: Optional database configuration. If None, loads from
                config manager.

        Returns:
            Whether initialization was successful.
        """
        try:
            with self._lock:
                if config is None:
                    config = self._config_manager.get_database_config(name)
                    if config is None:
                        logger.error(f"Database configuration not found for '{name}'")
                        return False

                self._db_configs[name] = config
                engine = self._create_enhanced_engine(name, config)
                self._engines[name] = engine
                self._metrics[name] = ConnectionMetrics()
                self._setup_resource_limits(name, config)

                health_result = self._perform_health_check(name)
                self._health_status[name] = health_result

                if health_result.is_healthy:
                    logger.info(
                        f"Successfully initialized database '{name}' "
                        f"(type: {config.type.value}, "
                        f"state: {health_result.state.value})"
                    )
                    return True

                logger.warning(
                    f"Database '{name}' initialized but unhealthy: "
                    f"{health_result.error_message}"
                )
                return False

        except Exception as e:
            logger.error(f"Failed to initialize database '{name}': {e}")
            return False

    def get_engine(self, name: str) -> Optional[Engine]:
        """Get database engine for a specific database.

        Args:
            name: Database name.

        Returns:
            Database engine if available.
        """
        with self._lock:
            if name not in self._engines:
                config = self._config_manager.get_database_config(name)
                if config and config.enabled:
                    if self.initialize_database(name, config):
                        return self._engines.get(name)
                return None
            return self._engines[name]

    def get_connection_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive connection information for a database.

        Args:
            name: Database name.

        Returns:
            Connection information dictionary.
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
                    "query_timeout": config.query_timeout,
                },
                "metrics": {
                    "total_queries": metrics.total_queries,
                    "success_rate": metrics.success_rate,
                    "error_rate": metrics.error_rate,
                    "average_query_time": metrics.average_query_time,
                    "memory_usage_mb": metrics.memory_usage_mb,
                    "active_connections": metrics.active_connections,
                },
                "health": {
                    "state": (health.state.value if health else "unknown"),
                    "is_healthy": health.is_healthy if health else False,
                    "last_check": health.timestamp if health else 0,
                    "response_time_ms": (health.response_time_ms if health else 0),
                },
                "resource_limits": {
                    rt.value: {
                        "max_value": limit.max_value,
                        "current_value": limit.current_value,
                        "warning_threshold": limit.warning_threshold,
                        "is_warning": limit.is_warning,
                        "is_exceeded": limit.is_exceeded,
                        "violations": limit.violations,
                    }
                    for rt, limit in limits.items()
                },
            }

    def list_databases(
        self, include_tags: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """List all configured databases with optional tag filtering.

        Args:
            include_tags: Optional list of tags to filter by.

        Returns:
            List of database information dictionaries.
        """
        databases = []
        all_configs = self._config_manager.get_database_configs()

        for name, config in all_configs.items():
            if include_tags:
                if not any(tag in config.tags for tag in include_tags):
                    continue

            db_info = self.get_connection_info(name)
            if db_info:
                databases.append(db_info)
            else:
                databases.append(
                    {
                        "name": name,
                        "type": config.type.value,
                        "enabled": config.enabled,
                        "tags": config.tags,
                        "initialized": False,
                    }
                )

        return databases

    def get_databases_by_tag(self, tag: str) -> List[str]:
        """Get all database names that have a specific tag.

        Args:
            tag: Tag to search for.

        Returns:
            List of database names with the tag.
        """
        all_configs = self._config_manager.get_database_configs()
        return [name for name, config in all_configs.items() if tag in config.tags]

    def close_database(self, name: str) -> bool:
        """Close and clean up a database connection.

        Args:
            name: Database name to close.

        Returns:
            Whether cleanup was successful.
        """
        try:
            with self._lock:
                if name in self._engines:
                    self._engines[name].dispose()
                    del self._engines[name]

                self._db_configs.pop(name, None)
                self._metrics.pop(name, None)
                self._health_status.pop(name, None)
                self._resource_limits.pop(name, None)
                self._active_queries.pop(name, None)

                query_ids_to_remove = [
                    qid for qid in self._query_start_times if qid.startswith(f"{name}_")
                ]
                for qid in query_ids_to_remove:
                    self._query_start_times.pop(qid, None)

                logger.info(f"Successfully closed database connection '{name}'")
                return True

        except Exception as e:
            logger.error(f"Error closing database '{name}': {e}")
            return False

    def close_all(self) -> None:
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
            name: Database name.

        Returns:
            Health check result if available.
        """
        return self._health_status.get(name)

    def trigger_health_check(self, name: str) -> Optional[HealthCheckResult]:
        """Manually trigger a health check for a specific database.

        Args:
            name: Database name.

        Returns:
            Health check result if engine exists.
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
            name: Database name.

        Returns:
            Resource status information.
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
                    "last_violation": limit.last_violation,
                }
                for resource_type, limit in limits.items()
            }
