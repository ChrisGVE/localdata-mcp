"""
Type definitions for the ShimRegistry framework.

Contains enums, dataclasses, and configuration types used across
the shim registry, enhanced adapters, and utility functions.
"""

import time
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum


class AdapterLifecycleState(Enum):
    """Adapter lifecycle states for managing adapter operations."""

    UNINITIALIZED = "uninitialized"
    INITIALIZED = "initialized"
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    DISPOSED = "disposed"


@dataclass
class AdapterConfig:
    """Configuration for adapter initialization and operation."""

    adapter_id: str
    config_params: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    initialization_order: int = 100  # Lower values initialize first
    enable_metrics: bool = True
    enable_health_checks: bool = True
    health_check_interval_seconds: int = 60
    max_initialization_time_seconds: int = 30
    auto_restart_on_error: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AdapterMetrics:
    """Performance and operational metrics for adapters."""

    adapter_id: str
    total_conversions: int = 0
    successful_conversions: int = 0
    failed_conversions: int = 0
    average_execution_time: float = 0.0
    last_execution_time: Optional[float] = None
    total_data_processed_mb: float = 0.0
    error_count: int = 0
    last_error: Optional[str] = None
    last_error_timestamp: Optional[float] = None
    health_check_count: int = 0
    last_health_check: Optional[float] = None
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    uptime_seconds: float = 0.0
    creation_timestamp: float = field(default_factory=time.time)


@dataclass
class HealthCheckResult:
    """Result of adapter health check."""

    adapter_id: str
    is_healthy: bool
    status: str
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metrics_snapshot: Dict[str, Any] = field(default_factory=dict)
    check_timestamp: float = field(default_factory=time.time)
