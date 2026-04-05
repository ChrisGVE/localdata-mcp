"""Memory-bounded streaming query executor.

Contains the StreamingQueryExecutor class that orchestrates adaptive streaming
execution with memory management, timeout control, and response-aware metadata.
"""

import time
from typing import Any, Dict, Iterator, List, Optional, Tuple

import pandas as pd

from ..config_manager import get_config_manager, PerformanceConfig
from ..disk_monitor import DiskMonitor
from ..logging_manager import get_logger, get_logging_manager
from ..timeout_manager import get_timeout_manager, QueryTimeoutError
from .buffers import (
    clear_buffer as _clear_buffer,
    cleanup_expired_buffers as _cleanup_expired_buffers,
    get_buffer_info as _get_buffer_info,
    manage_memory_bounds as _manage_memory_bounds,
)
from .memory import get_memory_status
from .models import ResultBuffer
from .runners import (
    apply_size_estimate,
    cleanup_on_timeout,
    log_streaming_failure,
    log_streaming_success,
    run_with_timeout,
    run_without_timeout,
)

logger = get_logger(__name__)


class StreamingQueryExecutor:
    """Memory-bounded streaming query executor."""

    def __init__(self, config: Optional[PerformanceConfig] = None):
        """Initialize streaming executor with configuration.

        Args:
            config: Performance configuration. If None, loads from config manager.
        """
        self.config = config or get_config_manager().get_performance_config()
        self._result_buffers: Dict[str, ResultBuffer] = {}
        self._chunk_metrics: List = []

        logging_manager = get_logging_manager()
        with logging_manager.context(
            operation="streaming_executor_init",
            component="streaming_executor",
        ):
            logger.info(
                "StreamingQueryExecutor initialized",
                memory_limit_mb=self.config.memory_limit_mb,
                default_chunk_size=self.config.chunk_size,
                buffer_timeout=self.config.query_buffer_timeout,
                max_concurrent_connections=self.config.max_concurrent_connections,
            )

    def execute_streaming(
        self,
        source,
        query_id: str,
        initial_chunk_size: Optional[int] = None,
        database_name: Optional[str] = None,
        disk_monitor: Optional[DiskMonitor] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Execute streaming query with adaptive memory management.

        Args:
            source: Streaming data source
            query_id: Unique identifier for buffering
            initial_chunk_size: Initial chunk size (adaptive if None)
            database_name: Name of the database for timeout configuration
            disk_monitor: Optional disk space monitor for abort-on-full

        Returns:
            Tuple of (first_chunk_df, metadata_dict)

        Raises:
            QueryTimeoutError: If query execution times out
        """
        start_time = time.time()
        logging_manager = get_logging_manager()
        request_id = logging_manager.log_query_start(
            database_name or "file_source",
            f"streaming_query_{query_id}",
            "streaming",
        )

        timeout_manager = get_timeout_manager()
        timeout_config = None
        operation_id = f"streaming_{query_id}"

        if database_name:
            timeout_config = timeout_manager.get_timeout_config(database_name)
            with logging_manager.context(
                request_id=request_id,
                operation="streaming_timeout_config",
                component="streaming_executor",
                database_name=database_name,
            ):
                logger.info(
                    "Query timeout configured",
                    timeout_seconds=timeout_config.query_timeout,
                    database_name=database_name,
                )

        memory_status = get_memory_status(self.config)
        chunk_size = initial_chunk_size or memory_status.recommended_chunk_size
        chunk_size = self._adjust_for_size_estimate(source, chunk_size)

        with logging_manager.context(
            request_id=request_id,
            operation="streaming_execution_start",
            component="streaming_executor",
            query_id=query_id,
        ):
            logger.info(
                "Starting streaming execution",
                chunk_size=chunk_size,
                available_memory_gb=memory_status.available_gb,
                memory_used_percent=memory_status.used_percent,
                is_low_memory=memory_status.is_low_memory,
            )

        try:
            result, metadata = self._dispatch_execution(
                source,
                query_id,
                chunk_size,
                memory_status,
                timeout_manager,
                operation_id,
                timeout_config,
                request_id,
                database_name,
                logging_manager,
                disk_monitor,
            )
            self._attach_size_estimate(source, chunk_size, metadata)
            log_streaming_success(
                start_time,
                request_id,
                database_name,
                logging_manager,
                memory_status,
                self.config,
                result,
                metadata,
            )
            return result, metadata
        except Exception as e:
            log_streaming_failure(
                start_time,
                request_id,
                database_name,
                logging_manager,
                query_id,
                source,
                e,
            )
            raise

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _adjust_for_size_estimate(source, chunk_size: int) -> int:
        """Adjust chunk size based on optional size estimation."""
        size_estimate = apply_size_estimate(source, chunk_size)
        if size_estimate is not None:
            if size_estimate.estimated_total_bytes > 100 * 1024 * 1024:
                chunk_size = min(chunk_size, 500)
            elif size_estimate.estimated_total_bytes > 10 * 1024 * 1024:
                chunk_size = min(chunk_size, 2000)
        return chunk_size

    @staticmethod
    def _attach_size_estimate(source, chunk_size, metadata):
        """Attach size estimate to metadata if available."""
        size_estimate = apply_size_estimate(source, chunk_size)
        if size_estimate is not None:
            metadata["size_estimate"] = {
                "estimated_rows": size_estimate.estimated_rows,
                "estimated_total_bytes": size_estimate.estimated_total_bytes,
                "confidence": size_estimate.confidence,
                "source": size_estimate.source,
            }

    def _dispatch_execution(
        self,
        source,
        query_id,
        chunk_size,
        memory_status,
        timeout_manager,
        operation_id,
        timeout_config,
        request_id,
        database_name,
        logging_manager,
        disk_monitor,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Route to timeout or non-timeout execution path."""
        if timeout_config:

            def _cleanup():
                cleanup_on_timeout(
                    query_id,
                    request_id,
                    database_name,
                    logging_manager,
                    self._result_buffers,
                )

            return run_with_timeout(
                source,
                query_id,
                chunk_size,
                memory_status,
                timeout_manager,
                operation_id,
                timeout_config,
                _cleanup,
                self.config,
                self._result_buffers,
                self._chunk_metrics,
                disk_monitor,
            )
        return run_without_timeout(
            source,
            query_id,
            chunk_size,
            memory_status,
            self.config,
            self._result_buffers,
            self._chunk_metrics,
            disk_monitor,
        )

    # ------------------------------------------------------------------
    # Public buffer / memory API (thin delegations)
    # ------------------------------------------------------------------

    def get_chunk_iterator(
        self,
        query_id: str,
        start_row: int = 0,
        chunk_size: Optional[int] = None,
    ) -> Iterator[pd.DataFrame]:
        """Get iterator for buffered query results."""
        if query_id not in self._result_buffers:
            logger.error(f"Query result buffer '{query_id}' not found")
            return
        buffer = self._result_buffers[query_id]
        chunk_size = chunk_size or self.config.chunk_size
        current_row = start_row
        while True:
            chunk = buffer.get_chunk_range(current_row, chunk_size)
            if chunk is None or chunk.empty:
                break
            yield chunk
            current_row += len(chunk)

    def _get_memory_status(self):
        """Get current memory status with recommendations."""
        return get_memory_status(self.config)

    def manage_memory_bounds(self) -> Dict[str, Any]:
        """Monitor and manage memory usage, cleaning up as needed."""
        memory_status = get_memory_status(self.config)
        return _manage_memory_bounds(
            self._result_buffers,
            self._chunk_metrics,
            memory_status,
        )

    def clear_buffer(self, query_id: str) -> bool:
        """Clear a specific result buffer."""
        return _clear_buffer(self._result_buffers, query_id)

    def get_buffer_info(self, query_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a result buffer."""
        return _get_buffer_info(self._result_buffers, query_id)

    def cleanup_expired_buffers(self, max_age_seconds: int = 3600) -> int:
        """Clean up buffers older than specified age."""
        return _cleanup_expired_buffers(
            self._result_buffers,
            max_age_seconds,
            _clear_buffer,
        )
