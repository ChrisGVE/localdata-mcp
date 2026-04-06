"""Buffer management functions for the streaming executor.

Provides standalone functions for managing result buffers, including
memory-bounded cleanup, individual buffer clearing, buffer info
retrieval, and expiration-based cleanup.
"""

import gc
import time
from typing import Any, Callable, Dict, List, Optional

from ..logging_manager import get_logger
from .models import ChunkMetrics, MemoryStatus, ResultBuffer

logger = get_logger(__name__)


def manage_memory_bounds(
    result_buffers: Dict[str, ResultBuffer],
    chunk_metrics: List[ChunkMetrics],
    memory_status: MemoryStatus,
) -> Dict[str, Any]:
    """Monitor and manage memory usage, cleaning up as needed.

    Args:
        result_buffers: Dict of active result buffers (mutated in place).
        chunk_metrics: List of chunk metrics (mutated in place if trimmed).
        memory_status: Current system memory status.

    Returns:
        Dict with memory management actions taken.
    """
    actions_taken: List[str] = []

    if memory_status.is_low_memory:
        logger.warning(
            f"Memory usage high ({memory_status.used_percent:.1f}%), starting cleanup"
        )

        # Clear oldest result buffers first
        buffers_to_clear: List[str] = []
        buffer_ages = [
            (query_id, buffer.timestamp) for query_id, buffer in result_buffers.items()
        ]
        buffer_ages.sort(key=lambda x: x[1])  # Sort by timestamp (oldest first)

        # Clear up to half of the buffers if memory is critical
        max_to_clear = (
            len(buffer_ages) // 2
            if memory_status.used_percent > 90
            else len(buffer_ages) // 4
        )

        for query_id, _ in buffer_ages[:max_to_clear]:
            result_buffers[query_id].clear()
            buffers_to_clear.append(query_id)

        # Remove cleared buffers from tracking
        for query_id in buffers_to_clear:
            del result_buffers[query_id]

        if buffers_to_clear:
            actions_taken.append(f"Cleared {len(buffers_to_clear)} result buffers")

        # Force garbage collection
        gc.collect()
        actions_taken.append("Forced garbage collection")

        # Clear old chunk metrics
        if len(chunk_metrics) > 100:
            del chunk_metrics[:-50]  # Keep last 50
            actions_taken.append("Trimmed chunk metrics history")

    return {
        "memory_status": memory_status.__dict__,
        "actions_taken": actions_taken,
        "active_buffers": len(result_buffers),
        "chunk_metrics_count": len(chunk_metrics),
    }


def clear_buffer(result_buffers: Dict[str, ResultBuffer], query_id: str) -> bool:
    """Clear a specific result buffer.

    Args:
        result_buffers: Dict of active result buffers (mutated in place).
        query_id: Buffer to clear.

    Returns:
        True if buffer was cleared, False if not found.
    """
    if query_id in result_buffers:
        result_buffers[query_id].clear()
        del result_buffers[query_id]
        logger.info(f"Cleared result buffer: {query_id}")
        return True
    return False


def get_buffer_info(
    result_buffers: Dict[str, ResultBuffer], query_id: str
) -> Optional[Dict[str, Any]]:
    """Get information about a result buffer.

    Args:
        result_buffers: Dict of active result buffers.
        query_id: Buffer to inspect.

    Returns:
        Dict with buffer information, or None if not found.
    """
    if query_id not in result_buffers:
        return None

    buffer = result_buffers[query_id]
    return {
        "query_id": buffer.query_id,
        "db_name": buffer.db_name,
        "total_rows": buffer.total_rows,
        "chunks_count": len(buffer.chunks),
        "memory_usage_mb": buffer._current_memory_mb,
        "memory_limit_mb": buffer.max_memory_mb,
        "timestamp": buffer.timestamp,
        "is_complete": buffer.is_complete,
    }


def cleanup_expired_buffers(
    result_buffers: Dict[str, ResultBuffer],
    max_age_seconds: int,
    clear_fn: Callable[[Dict[str, ResultBuffer], str], bool],
) -> int:
    """Clean up buffers older than specified age.

    Args:
        result_buffers: Dict of active result buffers (mutated in place).
        max_age_seconds: Maximum age of buffers to keep.
        clear_fn: Function to call for clearing each buffer.

    Returns:
        Number of buffers cleaned up.
    """
    current_time = time.time()
    expired_buffers = []

    for query_id, buffer in result_buffers.items():
        if current_time - buffer.timestamp > max_age_seconds:
            expired_buffers.append(query_id)

    for query_id in expired_buffers:
        clear_fn(result_buffers, query_id)

    if expired_buffers:
        logger.info(f"Cleaned up {len(expired_buffers)} expired buffers")

    return len(expired_buffers)
