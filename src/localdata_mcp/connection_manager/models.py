"""Data models for the Enhanced Connection Manager.

Provides enums, dataclasses, and type definitions used across
the connection management subsystem.
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


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
