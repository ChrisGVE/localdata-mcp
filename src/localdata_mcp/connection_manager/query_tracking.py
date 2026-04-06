"""Query execution tracking for the Enhanced Connection Manager.

Provides mixin class with the managed query execution context manager
and related metrics tracking.
"""

import logging
import time
from contextlib import contextmanager
from typing import Optional

logger = logging.getLogger(__name__)


class QueryTrackingMixin:
    """Mixin providing query execution tracking for EnhancedConnectionManager."""

    @contextmanager
    def managed_query_execution(
        self, database_name: str, query_id: Optional[str] = None
    ):
        """Context manager for tracked query execution.

        Args:
            database_name: Name of the database.
            query_id: Optional query identifier for tracking.

        Yields:
            Query execution context dictionary.
        """
        query_id = query_id or f"query_{int(time.time() * 1000)}"
        start_time = time.time()

        with self._lock:
            self._query_start_times[query_id] = start_time
            self._active_queries[database_name].add(query_id)
            metrics = self._metrics[database_name]
            metrics.total_queries += 1
            metrics.active_connections = len(self._active_queries[database_name])

        context = {
            "query_id": query_id,
            "database_name": database_name,
            "start_time": start_time,
            "resource_checks": [],
        }

        try:
            self._check_resource_limits(database_name)
            yield context

            with self._lock:
                self._metrics[database_name].successful_queries += 1

        except Exception as e:
            with self._lock:
                metrics = self._metrics[database_name]
                metrics.failed_queries += 1
                metrics.last_error = str(e)
                if any(
                    kw in str(e).lower() for kw in ["connection", "timeout", "network"]
                ):
                    metrics.connection_errors += 1
            raise

        finally:
            end_time = time.time()
            execution_time = end_time - start_time

            with self._lock:
                metrics = self._metrics[database_name]
                metrics.total_query_time += execution_time
                metrics.average_query_time = (
                    metrics.total_query_time / metrics.total_queries
                )
                self._active_queries[database_name].discard(query_id)
                self._query_start_times.pop(query_id, None)
                self._update_resource_usage(database_name)
