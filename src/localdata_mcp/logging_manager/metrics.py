"""Prometheus metrics collection for LocalData MCP."""

from typing import Optional

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    REGISTRY,
    generate_latest,
)


def _get_or_create_metric(metric_cls, name, description, labels, registry, **kwargs):
    """Create a Prometheus metric, reusing an existing one if already registered."""
    try:
        return metric_cls(name, description, labels, registry=registry, **kwargs)
    except ValueError:
        # Already registered -- look it up in the registry's internal mapping.
        with registry._lock:
            for collector in registry._names_to_collectors.values():
                if hasattr(collector, "_name") and collector._name == name:
                    return collector
        # Fallback: create without registry so the object is usable but not
        # double-registered.  This path should not normally be reached.
        return metric_cls(name, description, labels, registry=None, **kwargs)


class MetricsCollector:
    """Prometheus metrics collector for LocalData MCP.

    Handles duplicate registration gracefully so that multiple import paths
    (e.g. ``localdata_mcp`` vs ``src.localdata_mcp``) sharing the same
    global Prometheus REGISTRY do not crash.
    """

    def __init__(self, registry: Optional[CollectorRegistry] = None):
        """Initialize metrics collector.

        Args:
            registry: Prometheus registry to use. Defaults to global registry.
        """
        self.registry = registry or REGISTRY

        self.query_counter = _get_or_create_metric(
            Counter,
            "localdata_queries_total",
            "Total number of queries executed",
            ["database_type", "database_name", "status"],
            registry=self.registry,
        )

        self.query_duration = _get_or_create_metric(
            Histogram,
            "localdata_query_duration_seconds",
            "Query execution time in seconds",
            ["database_type", "database_name"],
            registry=self.registry,
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
        )

        self.active_connections = _get_or_create_metric(
            Gauge,
            "localdata_active_connections",
            "Number of active database connections",
            ["database_name"],
            registry=self.registry,
        )

        self.connection_pool_size = _get_or_create_metric(
            Gauge,
            "localdata_connection_pool_size",
            "Size of connection pool",
            ["database_name"],
            registry=self.registry,
        )

        self.memory_usage = _get_or_create_metric(
            Gauge,
            "localdata_memory_usage_bytes",
            "Current memory usage in bytes",
            ["component"],
            registry=self.registry,
        )

        self.buffer_utilization = _get_or_create_metric(
            Gauge,
            "localdata_buffer_utilization_ratio",
            "Query buffer utilization ratio (0-1)",
            ["buffer_type"],
            registry=self.registry,
        )

        self.error_counter = _get_or_create_metric(
            Counter,
            "localdata_errors_total",
            "Total number of errors",
            ["error_type", "component", "database_name"],
            registry=self.registry,
        )

        self.security_events = _get_or_create_metric(
            Counter,
            "localdata_security_events_total",
            "Total security events",
            ["event_type", "severity"],
            registry=self.registry,
        )

        self.timeout_counter = _get_or_create_metric(
            Counter,
            "localdata_timeouts_total",
            "Total number of timeouts",
            ["timeout_type", "database_name"],
            registry=self.registry,
        )

    def record_query(
        self,
        database_type: str,
        database_name: str,
        duration: float,
        status: str,
    ):
        """Record query execution metrics."""
        self.query_counter.labels(
            database_type=database_type,
            database_name=database_name,
            status=status,
        ).inc()

        self.query_duration.labels(
            database_type=database_type, database_name=database_name
        ).observe(duration)

    def record_error(
        self,
        error_type: str,
        component: str,
        database_name: Optional[str] = None,
    ):
        """Record error metrics."""
        self.error_counter.labels(
            error_type=error_type,
            component=component,
            database_name=database_name or "unknown",
        ).inc()

    def record_security_event(self, event_type: str, severity: str):
        """Record security event metrics."""
        self.security_events.labels(event_type=event_type, severity=severity).inc()

    def record_timeout(self, timeout_type: str, database_name: str):
        """Record timeout metrics."""
        self.timeout_counter.labels(
            timeout_type=timeout_type, database_name=database_name
        ).inc()

    def update_connection_metrics(
        self, database_name: str, active: int, pool_size: int
    ):
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
