"""High-level streaming run helpers for the executor.

Contains the timeout and non-timeout execution orchestrators, result
buffer initialisation, size estimation, timeout cleanup logic, and
logging helpers. Called by StreamingQueryExecutor to keep the class thin.
"""

import gc
import time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from ..config_manager import PerformanceConfig
from ..disk_monitor import DiskMonitor
from ..error_classification import classify_error
from ..logging_manager import get_logger
from ..size_estimator import SizeEstimate, get_size_estimator
from .execution import (
    add_truncation_metadata,
    build_base_metadata,
    build_timeout_metadata,
    execute_streaming_core,
)
from .memory import get_memory_status
from .models import ChunkMetrics, MemoryStatus, ResultBuffer
from .response_metadata import generate_response_metadata
from .sources import StreamingDataSource, StreamingSQLSource

logger = get_logger(__name__)


# ------------------------------------------------------------------
# Result buffer initialisation
# ------------------------------------------------------------------


def init_result_buffer(
    query_id: str,
    config: PerformanceConfig,
    result_buffers: Dict[str, ResultBuffer],
) -> ResultBuffer:
    """Create and register a new result buffer for the given query."""
    buffer_memory_limit = min(config.memory_limit_mb // 4, 500)
    buffer = ResultBuffer(query_id, "streaming", "streaming_query", buffer_memory_limit)
    result_buffers[query_id] = buffer
    return buffer


# ------------------------------------------------------------------
# Size estimation
# ------------------------------------------------------------------


def apply_size_estimate(
    source: StreamingDataSource, chunk_size: int
) -> Optional[SizeEstimate]:
    """Attempt to get a size estimate from a SQL source."""
    if isinstance(source, StreamingSQLSource) and source.engine is not None:
        try:
            estimator = get_size_estimator(source.engine)
            estimate = estimator.estimate_result_size(source.query)
            logger.debug(
                "Size estimate available",
                estimated_rows=estimate.estimated_rows,
                estimated_bytes=estimate.estimated_total_bytes,
                confidence=estimate.confidence,
            )
            return estimate
        except Exception as exc:
            logger.debug("Size estimation failed, continuing without: %s", exc)
    return None


# ------------------------------------------------------------------
# Timeout cleanup
# ------------------------------------------------------------------


def cleanup_on_timeout(
    query_id: str,
    request_id: str,
    database_name: Optional[str],
    logging_manager,
    result_buffers: Dict[str, ResultBuffer],
) -> None:
    """Clean up resources if query times out."""
    try:
        if query_id in result_buffers:
            with logging_manager.context(
                request_id=request_id,
                operation="streaming_timeout_cleanup",
                component="streaming_executor",
            ):
                logger.info(
                    "Cleaning up result buffer for timed out query",
                    query_id=query_id,
                    buffer_size=len(result_buffers[query_id].chunks),
                )
            del result_buffers[query_id]
        gc.collect()
        logging_manager.log_timeout(
            "query_timeout",
            database_name or "file_source",
            request_id=request_id,
            query_id=query_id,
        )
    except Exception as e:
        logging_manager.log_error(
            e,
            "streaming_executor",
            database_name=database_name,
            query_id=query_id,
            request_id=request_id,
        )


# ------------------------------------------------------------------
# Logging helpers
# ------------------------------------------------------------------


def detect_db_type(source) -> str:
    """Derive database type string from a streaming data source."""
    engine = getattr(source, "engine", None)
    if engine is not None:
        dialect_name = getattr(getattr(engine, "dialect", None), "name", None)
        if dialect_name:
            return str(dialect_name)
    return "generic"


def log_streaming_success(
    start_time: float,
    request_id: str,
    database_name: Optional[str],
    logging_manager,
    memory_status: MemoryStatus,
    config: PerformanceConfig,
    result: Optional[pd.DataFrame],
    metadata: Dict[str, Any],
) -> None:
    """Log successful streaming completion and performance metrics."""
    execution_time = time.time() - start_time
    logging_manager.log_query_complete(
        request_id,
        database_name or "file_source",
        "streaming",
        execution_time,
        row_count=len(result) if result is not None else 0,
        success=True,
    )
    final_memory = get_memory_status(config)
    logging_manager.log_performance_metrics(
        "streaming_executor",
        {
            "execution_time": execution_time,
            "memory_usage_bytes": (memory_status.total_gb - final_memory.available_gb)
            * 1024**3,
            "chunk_count": metadata.get("chunk_count", 0),
            "total_rows": len(result) if result is not None else 0,
            "memory_efficiency": final_memory.available_gb / memory_status.total_gb,
        },
    )


def log_streaming_failure(
    start_time: float,
    request_id: str,
    database_name: Optional[str],
    logging_manager,
    query_id: str,
    source,
    exc: Exception,
) -> None:
    """Log failed streaming execution and attach error classification."""
    execution_time = time.time() - start_time
    logging_manager.log_query_complete(
        request_id,
        database_name or "file_source",
        "streaming",
        execution_time,
        success=False,
    )
    logging_manager.log_error(
        exc,
        "streaming_executor",
        database_name=database_name,
        query_id=query_id,
        execution_time=execution_time,
        request_id=request_id,
    )
    db_type = detect_db_type(source)
    exc.structured_error = classify_error(exc, db_type)  # type: ignore[attr-defined]


# ------------------------------------------------------------------
# High-level run helpers
# ------------------------------------------------------------------


def run_with_timeout(
    source: StreamingDataSource,
    query_id: str,
    chunk_size: int,
    memory_status: MemoryStatus,
    timeout_manager,
    operation_id: str,
    timeout_config,
    cleanup_fn,
    config: PerformanceConfig,
    result_buffers: Dict[str, ResultBuffer],
    chunk_metrics_list: List[ChunkMetrics],
    disk_monitor: Optional[DiskMonitor] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Execute streaming with timeout management."""
    buffer = init_result_buffer(query_id, config, result_buffers)

    with timeout_manager.timeout_context(
        operation_id, timeout_config, cleanup_fn
    ) as context:
        (
            first_chunk,
            total_rows,
            chunk_count,
            truncated,
            abort_reason,
            final_memory,
        ) = execute_streaming_core(
            source,
            query_id,
            chunk_size,
            memory_status,
            buffer,
            config,
            chunk_metrics_list,
            timeout_manager=timeout_manager,
            operation_id=operation_id,
            disk_monitor=disk_monitor,
        )
        ctx_start = context.get("start_time", time.time())
        execution_time = time.time() - ctx_start
        tc = context.get("timeout_config")
        base = build_base_metadata(
            query_id,
            total_rows,
            chunk_count,
            execution_time,
            chunk_size,
            memory_status,
            final_memory,
            source,
            buffer,
        )
        metadata = build_timeout_metadata(base, tc, execution_time)

    if truncated:
        add_truncation_metadata(metadata, abort_reason, total_rows)
    if first_chunk is not None and len(first_chunk) > 0:
        metadata.update(generate_response_metadata(first_chunk, total_rows))

    logger.info(
        f"Streaming execution completed: {total_rows} rows in "
        f"{execution_time:.3f}s, {chunk_count} chunks (timeout-managed)"
    )
    return (
        first_chunk if first_chunk is not None else pd.DataFrame(),
        metadata,
    )


def run_without_timeout(
    source: StreamingDataSource,
    query_id: str,
    chunk_size: int,
    memory_status: MemoryStatus,
    config: PerformanceConfig,
    result_buffers: Dict[str, ResultBuffer],
    chunk_metrics_list: List[ChunkMetrics],
    disk_monitor: Optional[DiskMonitor] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Execute streaming without timeout management."""
    loop_start = time.time()
    buffer = init_result_buffer(query_id, config, result_buffers)

    (
        first_chunk,
        total_rows,
        chunk_count,
        truncated,
        abort_reason,
        final_memory,
    ) = execute_streaming_core(
        source,
        query_id,
        chunk_size,
        memory_status,
        buffer,
        config,
        chunk_metrics_list,
        disk_monitor=disk_monitor,
    )

    execution_time = time.time() - loop_start
    metadata = build_base_metadata(
        query_id,
        total_rows,
        chunk_count,
        execution_time,
        chunk_size,
        memory_status,
        final_memory,
        source,
        buffer,
    )
    metadata["timeout_info"] = {"timeout_configured": False}

    if truncated:
        add_truncation_metadata(metadata, abort_reason, total_rows)
    if first_chunk is not None and len(first_chunk) > 0:
        metadata.update(generate_response_metadata(first_chunk, total_rows))

    logger.info(
        f"Streaming execution completed: {total_rows} rows in "
        f"{execution_time:.3f}s, {chunk_count} chunks"
    )
    return (
        first_chunk if first_chunk is not None else pd.DataFrame(),
        metadata,
    )
