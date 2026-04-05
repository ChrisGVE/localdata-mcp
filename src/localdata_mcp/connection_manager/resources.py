"""Resource limit management for the Enhanced Connection Manager.

Provides mixin class with methods for setting up, checking,
and updating resource limits and usage metrics.
"""

import logging
import time

import psutil

from ..config_manager import DatabaseConfig
from .models import ResourceLimit, ResourceType

logger = logging.getLogger(__name__)


class ResourceManagerMixin:
    """Mixin providing resource management for EnhancedConnectionManager."""

    def _setup_resource_limits(self, name: str, config: DatabaseConfig) -> None:
        """Set up resource limits for a database.

        Args:
            name: Database name.
            config: Database configuration.
        """
        perf_config = self._config_manager.get_performance_config()

        limits = {
            ResourceType.MEMORY: ResourceLimit(
                resource_type=ResourceType.MEMORY,
                max_value=perf_config.memory_limit_mb,
                warning_threshold=(
                    perf_config.memory_limit_mb * perf_config.memory_warning_threshold
                ),
            ),
            ResourceType.CONNECTIONS: ResourceLimit(
                resource_type=ResourceType.CONNECTIONS,
                max_value=config.max_connections,
                warning_threshold=config.max_connections * 0.8,
            ),
            ResourceType.QUERY_TIME: ResourceLimit(
                resource_type=ResourceType.QUERY_TIME,
                max_value=config.query_timeout,
                warning_threshold=config.query_timeout * 0.8,
            ),
            ResourceType.ERROR_RATE: ResourceLimit(
                resource_type=ResourceType.ERROR_RATE,
                max_value=20.0,
                warning_threshold=10.0,
            ),
        }

        with self._lock:
            self._resource_limits[name] = limits

    def _check_resource_limits(self, database_name: str) -> None:
        """Check if resource limits are being exceeded.

        Args:
            database_name: Database name to check.

        Raises:
            Exception: If critical connection limits are exceeded.
        """
        with self._lock:
            limits = self._resource_limits.get(database_name, {})
            metrics = self._metrics[database_name]

            if ResourceType.CONNECTIONS in limits:
                limits[ResourceType.CONNECTIONS].current_value = len(
                    self._active_queries[database_name]
                )
            if ResourceType.ERROR_RATE in limits:
                limits[ResourceType.ERROR_RATE].current_value = metrics.error_rate

            current_time = time.time()
            for limit in limits.values():
                if limit.is_exceeded:
                    limit.violations += 1
                    limit.last_violation = current_time

                    if limit.resource_type == ResourceType.CONNECTIONS:
                        raise Exception(
                            f"Maximum connections exceeded: "
                            f"{limit.current_value}/{limit.max_value}"
                        )
                    elif limit.resource_type == ResourceType.ERROR_RATE:
                        logger.warning(
                            f"High error rate detected for "
                            f"'{database_name}': "
                            f"{limit.current_value:.1f}%"
                        )

    def _update_resource_usage(self, database_name: str) -> None:
        """Update resource usage metrics for a database.

        Args:
            database_name: Database name.
        """
        try:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024

            with self._lock:
                metrics = self._metrics[database_name]
                metrics.memory_usage_mb = memory_mb
                metrics.active_connections = len(self._active_queries[database_name])

                if database_name in self._resource_limits:
                    limits = self._resource_limits[database_name]
                    limits[ResourceType.MEMORY].current_value = memory_mb
                    limits[
                        ResourceType.CONNECTIONS
                    ].current_value = metrics.active_connections
                    limits[ResourceType.ERROR_RATE].current_value = metrics.error_rate

        except Exception as e:
            logger.warning(
                f"Failed to update resource usage for '{database_name}': {e}"
            )
