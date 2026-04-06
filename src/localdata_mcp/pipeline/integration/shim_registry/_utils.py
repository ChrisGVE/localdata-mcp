"""
Factory and utility functions for the ShimRegistry framework.

Provides convenience constructors and validation/monitoring utilities
for shim adapters and registries.
"""

import time
from typing import Any, Dict, List, Optional, Set

from ..compatibility_matrix import PipelineCompatibilityMatrix
from ..interfaces import DataFormat, ValidationResult
from ._enhanced_adapter import EnhancedShimAdapter
from ._registry import ShimRegistry
from ._types import AdapterConfig, AdapterMetrics

# Factory Functions


def create_shim_registry(
    compatibility_matrix: Optional[PipelineCompatibilityMatrix] = None, **kwargs
) -> ShimRegistry:
    """Create a ShimRegistry with standard configuration."""
    return ShimRegistry(compatibility_matrix=compatibility_matrix, **kwargs)


def create_adapter_config(adapter_id: str, **kwargs) -> AdapterConfig:
    """Create an AdapterConfig with sensible defaults."""
    return AdapterConfig(adapter_id=adapter_id, **kwargs)


# Utility Functions


def validate_adapter_dependencies(
    adapters: List[EnhancedShimAdapter],
) -> ValidationResult:
    """
    Validate adapter dependency graph for cycles and missing dependencies.

    Args:
        adapters: List of adapters to validate

    Returns:
        Validation result with dependency issues
    """
    errors = []
    warnings = []
    adapter_ids = {a.adapter_id for a in adapters}

    # Check for missing dependencies
    for adapter in adapters:
        for dep_id in adapter.config.dependencies:
            if dep_id not in adapter_ids:
                errors.append(
                    f"Adapter '{adapter.adapter_id}' has missing dependency: '{dep_id}'"
                )

    # Check for circular dependencies using DFS
    def has_cycle(adapter_id: str, visited: Set[str], rec_stack: Set[str]) -> bool:
        visited.add(adapter_id)
        rec_stack.add(adapter_id)

        adapter = next((a for a in adapters if a.adapter_id == adapter_id), None)
        if not adapter:
            return False

        for dep_id in adapter.config.dependencies:
            if dep_id not in visited:
                if has_cycle(dep_id, visited, rec_stack):
                    return True
            elif dep_id in rec_stack:
                return True

        rec_stack.remove(adapter_id)
        return False

    visited: Set[str] = set()
    for adapter in adapters:
        if adapter.adapter_id not in visited:
            if has_cycle(adapter.adapter_id, visited, set()):
                errors.append(
                    f"Circular dependency detected involving adapter '{adapter.adapter_id}'"
                )

    return ValidationResult(
        is_valid=len(errors) == 0,
        score=1.0 if len(errors) == 0 else 0.0,
        errors=errors,
        warnings=warnings,
        details={"total_adapters_checked": len(adapters)},
    )


def monitor_adapter_performance(
    adapter: EnhancedShimAdapter, duration_seconds: int = 60
) -> Dict[str, Any]:
    """
    Monitor adapter performance over a specified duration.

    Args:
        adapter: Adapter to monitor
        duration_seconds: Monitoring duration in seconds

    Returns:
        Performance monitoring report
    """
    start_time = time.time()
    start_metrics = adapter.get_metrics()

    # Wait for monitoring period
    time.sleep(duration_seconds)

    end_time = time.time()
    end_metrics = adapter.get_metrics()

    # Calculate deltas
    conversion_delta = end_metrics.total_conversions - start_metrics.total_conversions
    error_delta = end_metrics.failed_conversions - start_metrics.failed_conversions
    data_delta = (
        end_metrics.total_data_processed_mb - start_metrics.total_data_processed_mb
    )

    actual_duration = end_time - start_time

    return {
        "monitoring_duration": actual_duration,
        "conversions_per_second": conversion_delta / actual_duration,
        "errors_per_second": error_delta / actual_duration,
        "data_throughput_mb_per_second": data_delta / actual_duration,
        "current_error_rate": error_delta / max(conversion_delta, 1),
        "current_avg_execution_time": end_metrics.average_execution_time,
        "adapter_state": adapter.state.value,
        "snapshot": {"start_metrics": start_metrics, "end_metrics": end_metrics},
    }
