"""Core streaming execution loop and metadata builders.

Provides the unified chunk-processing loop that handles both timeout-managed
and non-timeout streaming execution, plus metadata construction helpers.
"""

import time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from ..config_manager import PerformanceConfig
from ..disk_monitor import DiskMonitor
from ..logging_manager import get_logger
from ..timeout_manager import QueryTimeoutError, TimeoutReason
from .memory import adapt_chunk_size, get_memory_status
from .models import ChunkMetrics, MemoryStatus, ResultBuffer
from .sources import StreamingDataSource

logger = get_logger(__name__)


# ------------------------------------------------------------------
# Core chunk-processing loop
# ------------------------------------------------------------------


def execute_streaming_core(
    source: StreamingDataSource,
    query_id: str,
    chunk_size: int,
    memory_status: MemoryStatus,
    result_buffer: ResultBuffer,
    config: PerformanceConfig,
    chunk_metrics_list: List[ChunkMetrics],
    timeout_manager=None,
    operation_id: Optional[str] = None,
    disk_monitor: Optional[DiskMonitor] = None,
) -> Tuple[Optional[pd.DataFrame], int, int, bool, Optional[str], MemoryStatus]:
    """Core streaming execution loop shared by timeout and non-timeout paths.

    Args:
        source: Streaming data source.
        query_id: Query identifier.
        chunk_size: Initial chunk size.
        memory_status: Initial memory status.
        result_buffer: Buffer for storing result chunks.
        config: Performance configuration.
        chunk_metrics_list: Mutable list to append chunk metrics to.
        timeout_manager: Optional timeout manager for cancellation checks.
        operation_id: Operation identifier for timeout tracking.
        disk_monitor: Optional disk space monitor for abort-on-full.

    Returns:
        Tuple of (first_chunk, total_rows_processed, chunk_number,
        truncated, abort_reason, final_memory_status).

    Raises:
        QueryTimeoutError: If query execution times out.
    """
    first_chunk = None
    total_rows_processed = 0
    chunk_number = 0
    truncated = False
    abort_reason: Optional[str] = None

    try:
        chunk_iterator = source.get_chunk_iterator(chunk_size)

        for chunk in chunk_iterator:
            if timeout_manager and timeout_manager.is_cancelled(operation_id):
                raise timeout_manager.create_timeout_error(
                    operation_id, TimeoutReason.USER_TIMEOUT
                )

            chunk_start_time = time.time()
            chunk_number += 1

            current_memory = get_memory_status(config)
            if current_memory.is_low_memory:
                logger.warning(
                    f"Low memory detected ({current_memory.used_percent:.1f}%), "
                    f"reducing chunk size from {chunk_size} "
                    f"to {current_memory.recommended_chunk_size}"
                )
                chunk_size = current_memory.recommended_chunk_size

            processed_chunk = _process_chunk(chunk, chunk_number)
            rows_in_chunk = len(processed_chunk)
            total_rows_processed += rows_in_chunk

            if first_chunk is None:
                first_chunk = processed_chunk.copy()

            buffered = result_buffer.add_chunk(processed_chunk)

            chunk_time = time.time() - chunk_start_time
            chunk_memory = processed_chunk.memory_usage(deep=True).sum() / (1024 * 1024)

            metrics = ChunkMetrics(
                chunk_number=chunk_number,
                rows_processed=rows_in_chunk,
                memory_used_mb=chunk_memory,
                processing_time_seconds=chunk_time,
            )
            chunk_metrics_list.append(metrics)

            logger.debug(
                f"Processed chunk {chunk_number}: {rows_in_chunk} rows, "
                f"{chunk_memory:.1f}MB, {chunk_time:.3f}s, buffered={buffered}"
            )

            if chunk_number > 1:
                chunk_size = adapt_chunk_size(
                    chunk_size, metrics, current_memory, chunk_metrics_list
                )

            if disk_monitor:
                can_continue, reason = disk_monitor.check_can_continue(
                    total_rows_processed
                )
                if not can_continue:
                    truncated = True
                    abort_reason = reason
                    break

            if timeout_manager and timeout_manager.is_cancelled(operation_id):
                raise timeout_manager.create_timeout_error(
                    operation_id, TimeoutReason.USER_TIMEOUT
                )

            if (
                chunk_number >= 3
                and current_memory.is_low_memory
                and total_rows_processed >= chunk_size
            ):
                logger.info(
                    f"Stopping initial streaming due to memory constraints "
                    f"after {chunk_number} chunks"
                )
                break

    except QueryTimeoutError:
        logger.warning(f"Query timed out during streaming execution: {query_id}")
        raise
    except Exception as e:
        logger.error(f"Error during streaming execution: {e}")
        raise

    result_buffer.is_complete = True
    final_memory = get_memory_status(config)
    return (
        first_chunk,
        total_rows_processed,
        chunk_number,
        truncated,
        abort_reason,
        final_memory,
    )


# ------------------------------------------------------------------
# Metadata builders
# ------------------------------------------------------------------


def build_base_metadata(
    query_id: str,
    total_rows_processed: int,
    chunk_number: int,
    execution_time: float,
    chunk_size: int,
    memory_status: MemoryStatus,
    final_memory: MemoryStatus,
    source: StreamingDataSource,
    result_buffer: ResultBuffer,
) -> Dict[str, Any]:
    """Build base metadata dict common to both execution paths."""
    return {
        "query_id": query_id,
        "total_rows_processed": total_rows_processed,
        "chunks_processed": chunk_number,
        "execution_time_seconds": execution_time,
        "final_chunk_size": chunk_size,
        "memory_status": {
            "initial_available_gb": memory_status.available_gb,
            "final_available_gb": final_memory.available_gb,
            "final_used_percent": final_memory.used_percent,
        },
        "estimated_total_rows": source.estimate_total_rows(),
        "streaming": True,
        "buffer_complete": result_buffer.is_complete,
    }


def build_timeout_metadata(
    base_metadata: Dict[str, Any],
    timeout_config,
    execution_time: float,
) -> Dict[str, Any]:
    """Augment base metadata with timeout-specific information."""
    metadata = dict(base_metadata)
    metadata["timeout_info"] = {
        "timeout_configured": True,
        "timeout_limit_seconds": timeout_config.query_timeout
        if timeout_config
        else None,
        "time_remaining_seconds": max(
            0, (timeout_config.query_timeout - execution_time)
        )
        if timeout_config
        else None,
        "database_name": timeout_config.database_name if timeout_config else None,
    }
    return metadata


def add_truncation_metadata(
    metadata: Dict[str, Any], abort_reason: Optional[str]
) -> None:
    """Add disk-monitor truncation fields to metadata in place."""
    metadata["truncated"] = True
    metadata["abort_reason"] = abort_reason
    metadata["suggestion"] = "Add LIMIT, WHERE clause, or use aggregation"
    metadata["disk_monitor_active"] = True


# ------------------------------------------------------------------
# Private helpers
# ------------------------------------------------------------------


def _process_chunk(chunk: pd.DataFrame, chunk_number: int) -> pd.DataFrame:
    """Process a single chunk (placeholder for chunk-specific processing)."""
    return chunk
