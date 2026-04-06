"""Data models for the streaming pipeline.

Contains dataclasses and buffer classes used by the streaming executor
for tracking memory status, chunk metrics, and result buffering.
"""

import gc
import time
import weakref
from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd

from ..logging_manager import get_logger

logger = get_logger(__name__)


@dataclass
class MemoryStatus:
    """Current system memory status for adaptive decision making."""

    total_gb: float
    available_gb: float
    used_percent: float
    is_low_memory: bool
    recommended_chunk_size: int
    max_safe_chunk_size: int


@dataclass
class ChunkMetrics:
    """Metrics for a processed chunk."""

    chunk_number: int
    rows_processed: int
    memory_used_mb: float
    processing_time_seconds: float
    timestamp: float = field(default_factory=time.time)


class ResultBuffer:
    """Memory-bounded result buffer with automatic cleanup using weakref."""

    def __init__(
        self, query_id: str, db_name: str, query: str, max_memory_mb: int = 500
    ):
        """Initialize result buffer with memory bounds.

        Args:
            query_id: Unique identifier for the query result
            db_name: Database name for the query
            query: The original SQL query
            max_memory_mb: Maximum memory to use for buffering
        """
        self.query_id = query_id
        self.db_name = db_name
        self.query = query
        self.max_memory_mb = max_memory_mb
        self.chunks: List[pd.DataFrame] = []
        self.total_rows = 0
        self.timestamp = time.time()
        self.is_complete = False
        self._current_memory_mb = 0.0

        # Use weakref for automatic cleanup
        self._cleanup_ref = weakref.finalize(self, self._cleanup_buffer)

    def add_chunk(self, chunk: pd.DataFrame) -> bool:
        """Add chunk to buffer if within memory limits.

        Returns:
            bool: True if chunk was added, False if memory limit exceeded
        """
        chunk_memory_mb = self._estimate_chunk_memory(chunk)

        if self._current_memory_mb + chunk_memory_mb > self.max_memory_mb:
            # Memory limit exceeded - don't buffer this chunk
            logger.warning(
                f"Buffer {self.query_id} memory limit ({self.max_memory_mb}MB) "
                f"exceeded, not buffering chunk of {chunk_memory_mb:.1f}MB"
            )
            return False

        self.chunks.append(chunk)
        self.total_rows += len(chunk)
        self._current_memory_mb += chunk_memory_mb

        logger.debug(
            f"Added chunk to buffer {self.query_id}: {len(chunk)} rows, "
            f"{chunk_memory_mb:.1f}MB, total memory: {self._current_memory_mb:.1f}MB"
        )
        return True

    def get_chunk_range(
        self, start_row: int, chunk_size: int
    ) -> Optional[pd.DataFrame]:
        """Get specific range of rows from buffered chunks."""
        if not self.chunks:
            return None

        # Concatenate all chunks for range selection
        full_df = pd.concat(self.chunks, ignore_index=True)
        end_row = min(start_row + chunk_size, len(full_df))

        if start_row >= len(full_df):
            return None

        return full_df.iloc[start_row:end_row].copy()

    def clear(self):
        """Manually clear the buffer."""
        self.chunks.clear()
        self.total_rows = 0
        self._current_memory_mb = 0.0
        gc.collect()  # Force garbage collection

    def _estimate_chunk_memory(self, chunk: pd.DataFrame) -> float:
        """Estimate memory usage of a DataFrame chunk in MB."""
        return chunk.memory_usage(deep=True).sum() / (1024 * 1024)

    @staticmethod
    def _cleanup_buffer():
        """Cleanup method called by weakref when buffer is garbage collected."""
        gc.collect()
