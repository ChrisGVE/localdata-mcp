"""Health check and monitoring logic for the Enhanced Connection Manager.

Provides mixin class with methods for performing health checks
and running the background monitoring thread.
"""

import logging
import threading
import time

from sqlalchemy import text

from ..config_manager import DatabaseType
from .models import ConnectionState, HealthCheckResult

logger = logging.getLogger(__name__)


class HealthMonitorMixin:
    """Mixin providing health check methods for EnhancedConnectionManager."""

    def _perform_health_check(self, name: str) -> HealthCheckResult:
        """Perform a health check on a database connection.

        Args:
            name: Database name.

        Returns:
            Health check result.
        """
        if name not in self._engines:
            return HealthCheckResult(
                is_healthy=False,
                state=ConnectionState.DISCONNECTED,
                response_time_ms=0.0,
                error_message="Engine not found",
            )

        engine = self._engines[name]
        config = self._db_configs[name]
        start_time = time.time()

        try:
            with engine.connect() as conn:
                if config.type == DatabaseType.ORACLE:
                    conn.execute(text("SELECT 1 FROM DUAL"))
                else:
                    conn.execute(text("SELECT 1"))

                response_time = (time.time() - start_time) * 1000
                metrics = self._metrics[name]

                if response_time < 100 and metrics.error_rate < 5:
                    state = ConnectionState.HEALTHY
                elif response_time < 500 and metrics.error_rate < 15:
                    state = ConnectionState.DEGRADED
                else:
                    state = ConnectionState.UNHEALTHY

                return HealthCheckResult(
                    is_healthy=state
                    in {ConnectionState.HEALTHY, ConnectionState.DEGRADED},
                    state=state,
                    response_time_ms=response_time,
                    metrics_snapshot=metrics,
                )

        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            logger.warning(f"Health check failed for '{name}': {e}")

            return HealthCheckResult(
                is_healthy=False,
                state=ConnectionState.UNHEALTHY,
                response_time_ms=response_time,
                error_message=str(e),
            )

    def _start_monitoring_thread(self) -> None:
        """Start background thread for health monitoring."""

        def monitor():
            while self._monitoring_active:
                try:
                    with self._lock:
                        database_names = list(self._engines.keys())

                    for name in database_names:
                        if not self._monitoring_active:
                            break
                        try:
                            result = self._perform_health_check(name)
                            with self._lock:
                                self._health_status[name] = result
                                self._metrics[name].last_health_check = time.time()
                        except Exception as e:
                            logger.error(f"Error during health check for '{name}': {e}")

                    time.sleep(self._health_check_interval)

                except Exception as e:
                    logger.error(f"Error in monitoring thread: {e}")
                    time.sleep(5)

        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()
        logger.info("Started database health monitoring thread")
