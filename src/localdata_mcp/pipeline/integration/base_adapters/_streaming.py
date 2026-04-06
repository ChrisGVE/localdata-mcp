"""
Streaming-optimized adapter for memory-efficient processing of large datasets.

Extends BaseShimAdapter with chunked processing capabilities to handle
datasets that don't fit in memory.
"""

from typing import Any

import pandas as pd

from ....logging_manager import get_logger
from ..interfaces import ConversionRequest, MemoryConstraints
from ._core import BaseShimAdapter, ConversionContext

logger = get_logger(__name__)


class StreamingShimAdapter(BaseShimAdapter):
    """
    Streaming-optimized adapter for memory-efficient processing of large datasets.

    Extends BaseShimAdapter with chunked processing capabilities to handle
    datasets that don't fit in memory.
    """

    def __init__(self, adapter_id: str, chunk_size: int = 10000, **kwargs):
        """
        Initialize StreamingShimAdapter.

        Args:
            adapter_id: Unique identifier for this adapter
            chunk_size: Number of records per processing chunk
            **kwargs: Additional arguments passed to BaseShimAdapter
        """
        # Force streaming-friendly memory constraints
        if "memory_constraints" not in kwargs or kwargs["memory_constraints"] is None:
            kwargs["memory_constraints"] = MemoryConstraints(
                prefer_streaming=True, chunk_size=chunk_size, memory_efficient=True
            )

        super().__init__(adapter_id, **kwargs)
        self.chunk_size = chunk_size

        logger.info(
            f"Initialized StreamingShimAdapter",
            adapter_id=adapter_id,
            chunk_size=chunk_size,
        )

    def _perform_conversion(
        self, request: ConversionRequest, context: ConversionContext
    ) -> Any:
        """
        Perform streaming conversion for large datasets.

        Args:
            request: Conversion request
            context: Conversion context for tracking

        Returns:
            Converted data
        """
        data_size = self._estimate_data_size(request.source_data)

        # Use streaming processing for large datasets
        if data_size > 100 * 1024 * 1024:  # 100MB threshold
            return self._stream_convert(request, context)
        else:
            # Use regular processing for small data
            return super()._perform_conversion(request, context)

    def _stream_convert(
        self, request: ConversionRequest, context: ConversionContext
    ) -> Any:
        """
        Perform chunked streaming conversion.

        Args:
            request: Conversion request
            context: Conversion context

        Returns:
            Converted data assembled from chunks
        """
        # This is a base implementation - subclasses should implement
        # specific streaming logic for their conversion types

        if isinstance(request.source_data, pd.DataFrame):
            return self._stream_convert_dataframe(request, context)
        else:
            # Fallback to regular conversion
            context.warnings.append(
                "Streaming not supported for this data type, using regular conversion"
            )
            return super()._perform_conversion(request, context)

    def _stream_convert_dataframe(
        self, request: ConversionRequest, context: ConversionContext
    ) -> pd.DataFrame:
        """Stream convert a DataFrame in chunks."""
        source_df = request.source_data
        converted_chunks = []

        total_chunks = (len(source_df) + self.chunk_size - 1) // self.chunk_size

        for i in range(0, len(source_df), self.chunk_size):
            chunk = source_df.iloc[i : i + self.chunk_size]

            # Create chunk-specific request
            chunk_request = ConversionRequest(
                source_data=chunk,
                source_format=request.source_format,
                target_format=request.target_format,
                metadata=request.metadata,
                context=request.context,
                request_id=f"{request.request_id}_chunk_{i // self.chunk_size}",
            )

            # Convert chunk using base implementation
            converted_chunk = super()._perform_conversion(chunk_request, context)
            converted_chunks.append(converted_chunk)

            # Update progress
            progress = ((i // self.chunk_size) + 1) / total_chunks
            context.performance_metrics[f"chunk_{i // self.chunk_size}_progress"] = (
                progress
            )

        # Combine chunks
        result = pd.concat(converted_chunks, ignore_index=True)
        context.performance_metrics["total_chunks_processed"] = len(converted_chunks)

        return result
