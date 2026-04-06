"""Memory management functions for the streaming executor.

Provides standalone functions for querying system memory status,
calculating adaptive chunk sizes, and adjusting chunk sizes based
on runtime performance metrics.
"""

from typing import List, Tuple

import psutil

from ..config_manager import PerformanceConfig
from ..logging_manager import get_logger
from .models import ChunkMetrics, MemoryStatus

logger = get_logger(__name__)


def get_memory_status(config: PerformanceConfig) -> MemoryStatus:
    """Get current memory status with recommendations.

    Args:
        config: Performance configuration for thresholds and defaults.

    Returns:
        MemoryStatus with current system memory information.
    """
    try:
        memory = psutil.virtual_memory()

        total_gb = memory.total / (1024**3)
        available_gb = memory.available / (1024**3)
        used_percent = memory.percent

        # Determine if memory is low based on threshold
        is_low_memory = used_percent > (config.memory_warning_threshold * 100)

        # Calculate recommended chunk size based on available memory
        recommended, max_safe = calculate_chunk_sizes(used_percent, config.chunk_size)

        return MemoryStatus(
            total_gb=round(total_gb, 2),
            available_gb=round(available_gb, 2),
            used_percent=used_percent,
            is_low_memory=is_low_memory,
            recommended_chunk_size=recommended,
            max_safe_chunk_size=max_safe,
        )

    except Exception as e:
        logger.warning(f"Could not get memory status: {e}")
        # Return safe defaults
        return MemoryStatus(
            total_gb=8.0,
            available_gb=4.0,
            used_percent=50.0,
            is_low_memory=False,
            recommended_chunk_size=config.chunk_size,
            max_safe_chunk_size=config.chunk_size,
        )


def calculate_chunk_sizes(used_percent: float, base_chunk_size: int) -> Tuple[int, int]:
    """Calculate recommended and max safe chunk sizes from memory usage.

    Args:
        used_percent: Current memory usage percentage.
        base_chunk_size: Base chunk size from configuration.

    Returns:
        Tuple of (recommended_chunk_size, max_safe_chunk_size).
    """
    if used_percent > 90:
        recommended = max(base_chunk_size // 8, 10)
        max_safe = max(base_chunk_size // 4, 25)
    elif used_percent > 80:
        recommended = max(base_chunk_size // 4, 25)
        max_safe = max(base_chunk_size // 2, 50)
    elif used_percent > 60:
        recommended = max(base_chunk_size // 2, 50)
        max_safe = base_chunk_size
    else:
        recommended = base_chunk_size
        max_safe = min(base_chunk_size * 2, 1000)
    return recommended, max_safe


def adapt_chunk_size(
    current_chunk_size: int,
    metrics: ChunkMetrics,
    memory_status: MemoryStatus,
    chunk_metrics_list: List[ChunkMetrics],
) -> int:
    """Adaptively adjust chunk size based on performance metrics and memory.

    Args:
        current_chunk_size: Current chunk size in rows.
        metrics: Metrics from the most recent chunk.
        memory_status: Current system memory status.
        chunk_metrics_list: Historical chunk metrics for trend analysis.

    Returns:
        Adjusted chunk size.
    """
    # Don't adjust too frequently
    if len(chunk_metrics_list) < 2:
        return current_chunk_size

    # Get recent performance trend
    recent_metrics = (
        chunk_metrics_list[-3:] if len(chunk_metrics_list) >= 3 else chunk_metrics_list
    )
    avg_processing_time = sum(m.processing_time_seconds for m in recent_metrics) / len(
        recent_metrics
    )
    avg_memory_per_row = sum(
        m.memory_used_mb / m.rows_processed for m in recent_metrics
    ) / len(recent_metrics)

    # Adjust based on performance and memory constraints
    new_chunk_size = current_chunk_size

    # If processing is too slow, reduce chunk size
    if avg_processing_time > 2.0:  # More than 2 seconds per chunk
        new_chunk_size = max(current_chunk_size // 2, 10)
        logger.debug(
            f"Reducing chunk size due to slow processing: "
            f"{current_chunk_size} -> {new_chunk_size}"
        )

    # If memory per row is high, reduce chunk size
    elif avg_memory_per_row > 0.1:  # More than 0.1MB per row
        new_chunk_size = max(current_chunk_size // 2, 25)
        logger.debug(
            f"Reducing chunk size due to high memory usage: "
            f"{current_chunk_size} -> {new_chunk_size}"
        )

    # If processing is fast and memory is available, can increase chunk size
    elif avg_processing_time < 0.5 and not memory_status.is_low_memory:
        new_chunk_size = min(current_chunk_size * 2, memory_status.max_safe_chunk_size)
        logger.debug(
            f"Increasing chunk size due to good performance: "
            f"{current_chunk_size} -> {new_chunk_size}"
        )

    # Ensure we stay within memory bounds
    new_chunk_size = min(new_chunk_size, memory_status.max_safe_chunk_size)
    new_chunk_size = max(new_chunk_size, 10)  # Minimum 10 rows

    return new_chunk_size
