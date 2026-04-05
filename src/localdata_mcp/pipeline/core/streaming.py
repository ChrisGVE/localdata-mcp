"""
Streaming pipeline components for LocalData MCP v2.0

This module provides streaming-capable pipeline classes that extend
DataSciencePipeline with memory-bounded processing capabilities:

- StreamingDataPipeline: Automatic streaming activation for large datasets
- SklearnStreamingAdapter: Bridge between sklearn pipelines and StreamingQueryExecutor
"""

import time
import uuid
import warnings
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

from ..base import (
    AnalysisPipelineBase,
    CompositionMetadata,
    PipelineError,
    PipelineResult,
    PipelineState,
    ErrorClassification,
    StreamingConfig,
)
from ...streaming import (
    StreamingQueryExecutor,
    MemoryStatus,
    StreamingDataSource,
)
from ...logging_manager import get_logging_manager, get_logger

from .pipeline_class import DataSciencePipeline, DataFrameStreamingSource

logger = get_logger(__name__)


class StreamingDataPipeline(DataSciencePipeline):
    """
    DataSciencePipeline with memory-bounded streaming capabilities.

    Extends DataSciencePipeline to handle large datasets through intelligent streaming
    while maintaining full sklearn Pipeline API compatibility. Automatically activates
    streaming for datasets above memory thresholds and provides adaptive chunk sizing
    based on real-time memory monitoring.

    Key Streaming Features:
    - Automatic streaming threshold detection (>1GB or memory pressure)
    - Memory-safe chunk processing with StreamingQueryExecutor integration
    - Intelligent result aggregation across chunks
    - Adaptive chunk sizing based on MemoryStatus feedback
    - Seamless sklearn Pipeline API compatibility

    Example Usage:
        # Automatic streaming activation
        pipeline = StreamingDataPipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier())
        ], streaming_threshold_mb=1024)

        # Works identically to sklearn Pipeline for small datasets
        pipeline.fit(small_data, y)
        result = pipeline.transform(small_data)

        # Automatically streams for large datasets
        pipeline.fit(large_data, y)  # Streaming activated automatically
        result = pipeline.transform(large_data)  # Memory-safe streaming
    """

    def __init__(
        self,
        steps: List[Tuple[str, BaseEstimator]],
        *,
        memory=None,
        verbose=False,
        analytical_intention: Optional[str] = None,
        streaming_config: Optional[StreamingConfig] = None,
        progressive_complexity: str = "auto",
        composition_aware: bool = True,
        custom_parameters: Optional[Dict[str, Any]] = None,
        streaming_threshold_mb: int = 1024,
        adaptive_chunking: bool = True,
        memory_monitoring: bool = True,
    ):
        """
        Initialize StreamingDataPipeline with enhanced streaming capabilities.

        Args:
            steps: List of (name, estimator) tuples (sklearn Pipeline compatible)
            memory: sklearn memory parameter for caching (inherited from Pipeline)
            verbose: sklearn verbose parameter (inherited from Pipeline)
            analytical_intention: Natural language description of analysis goal
            streaming_config: Configuration for streaming execution (auto-generated if None)
            progressive_complexity: Complexity level ("minimal", "auto", "comprehensive", "custom")
            composition_aware: Whether to generate metadata for tool chaining
            custom_parameters: Additional domain-specific parameters
            streaming_threshold_mb: Memory threshold for automatic streaming activation
            adaptive_chunking: Whether to enable adaptive chunk size adjustment
            memory_monitoring: Whether to enable real-time memory monitoring
        """
        # Configure streaming if not provided
        if streaming_config is None:
            streaming_config = StreamingConfig(
                enabled=False,  # Will auto-enable based on data size
                threshold_mb=streaming_threshold_mb,
                chunk_size_adaptive=adaptive_chunking,
                memory_limit_mb=streaming_threshold_mb * 2,  # 2x threshold for buffer
                memory_efficient_mode=True,
                early_termination_enabled=True,
            )

        # Initialize parent DataSciencePipeline
        super().__init__(
            steps=steps,
            memory=memory,
            verbose=verbose,
            analytical_intention=analytical_intention
            or "Large dataset processing with streaming",
            streaming_config=streaming_config,
            progressive_complexity=progressive_complexity,
            composition_aware=composition_aware,
            custom_parameters=custom_parameters,
        )

        # Streaming-specific configuration
        self.streaming_threshold_mb = streaming_threshold_mb
        self.adaptive_chunking = adaptive_chunking
        self.memory_monitoring = memory_monitoring

        # Enhanced streaming state
        self._streaming_activated = False
        self._chunk_fit_results: List[Any] = []
        self._chunk_transform_results: List[Any] = []
        self._memory_snapshots: List[MemoryStatus] = []
        self._adaptive_chunk_size = streaming_config.initial_chunk_size or 1000

        logger.info(
            "StreamingDataPipeline initialized",
            pipeline_id=self._pipeline_id,
            threshold_mb=self.streaming_threshold_mb,
            adaptive_chunking=self.adaptive_chunking,
            memory_monitoring=self.memory_monitoring,
        )

    def fit(self, X, y=None, **fit_params):
        """
        Fit the pipeline with streaming support for memory-bounded processing.

        Automatically detects if streaming is needed based on data size and memory
        pressure. For small datasets, behaves identically to sklearn Pipeline.fit().
        For large datasets, uses intelligent chunk-based fitting with result aggregation.

        Args:
            X: Training data (array-like, DataFrame, or streaming source)
            y: Target values (array-like, optional)
            **fit_params: Parameters to pass to pipeline steps

        Returns:
            self: Returns self for method chaining (sklearn compatible)
        """
        self._state = PipelineState.CONFIGURED
        start_time = time.time()

        # Start logging context
        self._request_id = self._logging_manager.log_query_start(
            database_name="streaming_pipeline",
            query=f"streaming_fit_{self._pipeline_id}",
            database_type="streaming_pipeline_fit",
        )

        with self._logging_manager.context(
            request_id=self._request_id,
            operation="streaming_pipeline_fit",
            component="streaming_data_pipeline",
        ):
            try:
                # Convert input to DataFrame for analysis
                X_df = self._ensure_dataframe(X, "streaming_fit_input")

                # Profile data and determine streaming necessity
                data_profile = self._profile_data_characteristics(X_df)
                self._execution_context["data_profile"] = data_profile

                # Check if streaming should be activated
                should_stream = self._should_use_streaming(X_df, data_profile)
                self._streaming_activated = should_stream

                if should_stream:
                    logger.info(
                        "Activating streaming mode for fit",
                        pipeline_id=self._pipeline_id,
                        data_size_mb=data_profile.get("memory_usage_mb", 0),
                        threshold_mb=self.streaming_threshold_mb,
                    )

                    # Configure streaming executor
                    if not self._streaming_executor:
                        self._streaming_executor = StreamingQueryExecutor()

                    # Execute streaming fit
                    result = self._execute_streaming_fit(X_df, y, **fit_params)
                else:
                    # Use standard DataSciencePipeline fit for small datasets
                    logger.info(
                        "Using standard fit mode",
                        pipeline_id=self._pipeline_id,
                        data_size_mb=data_profile.get("memory_usage_mb", 0),
                    )
                    result = super().fit(X, y, **fit_params)

                # Build composition metadata if enabled
                if self.composition_aware:
                    self._composition_metadata = (
                        self._build_initial_composition_metadata(X_df, y)
                    )

                # Record successful fit
                self._fit_time = time.time() - start_time
                self._state = PipelineState.FITTED

                # Log successful completion
                self._logging_manager.log_query_complete(
                    request_id=self._request_id,
                    database_name="streaming_pipeline",
                    database_type="streaming_pipeline_fit",
                    duration=self._fit_time,
                    success=True,
                )

                logger.info(
                    "StreamingDataPipeline fitted successfully",
                    pipeline_id=self._pipeline_id,
                    fit_time=self._fit_time,
                    streaming_activated=self._streaming_activated,
                    data_shape=X_df.shape,
                )

                return result

            except Exception as e:
                self._state = PipelineState.ERROR
                fit_time = time.time() - start_time

                # Log error and handle
                error_info = self._handle_pipeline_error(
                    e,
                    "streaming_fit",
                    {
                        "data_shape": X_df.shape if "X_df" in locals() else None,
                        "fit_time": fit_time,
                        "streaming_activated": self._streaming_activated,
                        "step_params": fit_params,
                    },
                )

                self._logging_manager.log_query_complete(
                    request_id=self._request_id,
                    database_name="streaming_pipeline",
                    database_type="streaming_pipeline_fit",
                    duration=fit_time,
                    success=False,
                )

                # Re-raise with enhanced error information
                raise PipelineError(
                    f"StreamingDataPipeline fit failed: {str(e)}",
                    classification=ErrorClassification.CONFIGURATION_ERROR,
                    pipeline_stage="streaming_fit",
                    context=error_info,
                    recovery_suggestions=[
                        "Check data format and pipeline step compatibility",
                        "Reduce streaming threshold or enable adaptive chunking",
                        "Verify memory limits and available system resources",
                        "Try reducing chunk size for memory-constrained environments",
                    ],
                ) from e

    def transform(self, X):
        """
        Transform data with streaming support for memory-bounded processing.

        Automatically uses streaming if it was activated during fit or if the
        transform data size exceeds thresholds. Maintains sklearn Pipeline API.

        Args:
            X: Data to transform (array-like, DataFrame, or streaming source)

        Returns:
            Transformed data with preserved format when possible
        """
        if self._state != PipelineState.FITTED:
            raise ValueError(f"Pipeline not fitted. Current state: {self._state.value}")

        self._state = PipelineState.EXECUTING
        start_time = time.time()

        # Start logging context
        request_id = self._logging_manager.log_query_start(
            database_name="streaming_pipeline",
            query=f"streaming_transform_{self._pipeline_id}",
            database_type="streaming_pipeline_transform",
        )

        with self._logging_manager.context(
            request_id=request_id,
            operation="streaming_pipeline_transform",
            component="streaming_data_pipeline",
        ):
            try:
                # Convert input to DataFrame for analysis
                X_df = self._ensure_dataframe(X, "streaming_transform_input")

                # Determine if streaming should be used for transform
                data_profile = self._profile_data_characteristics(X_df)
                should_stream = self._streaming_activated or self._should_use_streaming(
                    X_df, data_profile
                )

                if should_stream:
                    logger.info(
                        "Using streaming transform",
                        pipeline_id=self._pipeline_id,
                        data_size_mb=data_profile.get("memory_usage_mb", 0),
                    )

                    # Execute streaming transform
                    result = self._execute_streaming_transform(X_df)
                else:
                    # Use standard DataSciencePipeline transform
                    logger.info(
                        "Using standard transform",
                        pipeline_id=self._pipeline_id,
                        data_size_mb=data_profile.get("memory_usage_mb", 0),
                    )
                    result = self._transform_with_metadata_tracking(X)

                # Record successful transform
                self._transform_time = time.time() - start_time
                self._state = PipelineState.COMPLETED

                # Update composition metadata with transform results
                if self.composition_aware and self._composition_metadata:
                    self._update_composition_metadata_post_transform(X_df, result)

                # Log successful completion
                self._logging_manager.log_query_complete(
                    request_id=request_id,
                    database_name="streaming_pipeline",
                    database_type="streaming_pipeline_transform",
                    duration=self._transform_time,
                    success=True,
                )

                logger.info(
                    "StreamingDataPipeline transform completed",
                    pipeline_id=self._pipeline_id,
                    transform_time=self._transform_time,
                    streaming_used=should_stream,
                    input_shape=X_df.shape,
                    output_shape=getattr(result, "shape", "N/A"),
                )

                return result

            except Exception as e:
                self._state = PipelineState.ERROR
                transform_time = time.time() - start_time

                # Try to get partial results
                partial_results = self._get_partial_transform_results()

                # Handle error with recovery
                error_info = self._handle_pipeline_error(
                    e,
                    "streaming_transform",
                    {
                        "data_shape": X_df.shape if "X_df" in locals() else None,
                        "transform_time": transform_time,
                        "streaming_activated": self._streaming_activated,
                    },
                    partial_results,
                )

                self._logging_manager.log_query_complete(
                    request_id=request_id,
                    database_name="streaming_pipeline",
                    database_type="streaming_pipeline_transform",
                    duration=transform_time,
                    success=False,
                )

                # Re-raise with enhanced error information
                raise PipelineError(
                    f"StreamingDataPipeline transform failed: {str(e)}",
                    classification=ErrorClassification.COMPUTATION_TIMEOUT
                    if "timeout" in str(e).lower()
                    else ErrorClassification.DATA_QUALITY_FAILURE,
                    pipeline_stage="streaming_transform",
                    context=error_info,
                    partial_results=partial_results,
                    recovery_suggestions=[
                        "Check data format matches training data",
                        "Enable or increase streaming threshold for large datasets",
                        "Verify pipeline is properly fitted with streaming",
                        "Check memory availability and reduce chunk size if needed",
                    ],
                ) from e

    def _should_use_streaming(
        self, data: pd.DataFrame, profile: Dict[str, Any]
    ) -> bool:
        """
        Determine if streaming should be activated based on data characteristics and memory.

        Args:
            data: Input DataFrame to analyze
            profile: Data characteristics profile

        Returns:
            bool: True if streaming should be activated
        """
        # Check if streaming is already explicitly enabled
        if self.streaming_config.enabled:
            return True

        # Check memory-based thresholds
        data_size_mb = profile.get("memory_usage_mb", 0)
        row_count = data.shape[0]

        # Size-based activation
        if data_size_mb > self.streaming_threshold_mb:
            logger.info(
                f"Streaming activated: data size ({data_size_mb:.1f}MB) > threshold ({self.streaming_threshold_mb}MB)"
            )
            return True

        # Row count-based activation (for sparse data)
        if row_count > 100000:  # 100K rows
            logger.info(f"Streaming activated: row count ({row_count}) > 100K rows")
            return True

        # Memory pressure-based activation
        if self.memory_monitoring and self._streaming_executor:
            memory_status = self._streaming_executor._get_memory_status()
            if memory_status.is_low_memory:
                logger.info(
                    f"Streaming activated: memory pressure detected ({memory_status.used_percent:.1f}% used)"
                )
                return True

        return False

    def _execute_streaming_fit(self, X: pd.DataFrame, y=None, **fit_params):
        """
        Execute pipeline fitting with streaming for memory-bounded processing.

        Args:
            X: Training DataFrame
            y: Target values (optional)
            **fit_params: Parameters for fit method

        Returns:
            self: Returns self for method chaining
        """
        # Create streaming data source
        streaming_source = DataFrameStreamingSource(X)

        # Get initial memory status and adaptive chunk size
        memory_status = self._streaming_executor._get_memory_status()
        chunk_size = self._calculate_adaptive_chunk_size(X, memory_status)

        logger.info(
            "Starting streaming fit execution",
            pipeline_id=self._pipeline_id,
            chunk_size=chunk_size,
            total_rows=len(X),
            memory_available_gb=memory_status.available_gb,
        )

        # Execute streaming fit by processing chunks
        query_id = f"fit_{self._pipeline_id}_{time.time()}"

        try:
            # Process chunks for fitting
            chunk_number = 0
            total_samples_processed = 0

            for chunk in streaming_source.get_chunk_iterator(chunk_size):
                chunk_start_time = time.time()
                chunk_number += 1

                # Monitor memory before processing chunk
                if self.memory_monitoring:
                    current_memory = self._streaming_executor._get_memory_status()
                    self._memory_snapshots.append(current_memory)

                    if current_memory.is_low_memory:
                        # Adapt chunk size for memory pressure
                        chunk_size = min(
                            chunk_size, current_memory.recommended_chunk_size
                        )
                        logger.warning(
                            f"Memory pressure detected, reducing chunk size to {chunk_size}"
                        )

                # Get corresponding y chunk if provided
                if y is not None:
                    start_idx = (chunk_number - 1) * chunk_size
                    end_idx = start_idx + len(chunk)
                    if hasattr(y, "iloc"):
                        y_chunk = y.iloc[start_idx:end_idx]
                    else:
                        y_chunk = y[start_idx:end_idx]
                else:
                    y_chunk = None

                # Fit on chunk using sklearn's partial_fit or incremental learning
                chunk_fit_result = self._fit_chunk(
                    chunk, y_chunk, chunk_number, **fit_params
                )
                self._chunk_fit_results.append(chunk_fit_result)

                total_samples_processed += len(chunk)
                chunk_time = time.time() - chunk_start_time

                logger.debug(
                    f"Processed fit chunk {chunk_number}: {len(chunk)} samples, {chunk_time:.3f}s"
                )

                # Adaptive chunk size adjustment
                if self.adaptive_chunking and chunk_number > 1:
                    chunk_size = self._adaptive_chunk_sizing(
                        chunk_size,
                        chunk_time,
                        current_memory if self.memory_monitoring else memory_status,
                    )

            # Aggregate chunk results for final fitted pipeline
            self._aggregate_chunk_fit_results()

            logger.info(
                f"Streaming fit completed: {total_samples_processed} samples, {chunk_number} chunks"
            )
            return self

        except Exception as e:
            logger.error(f"Streaming fit failed: {e}")
            raise

    def _execute_streaming_transform(self, X: pd.DataFrame):
        """
        Execute pipeline transformation with streaming for memory-bounded processing.

        Args:
            X: Input DataFrame to transform

        Returns:
            Transformed data (aggregated from chunks)
        """
        # Create streaming data source
        streaming_source = DataFrameStreamingSource(X)

        # Get initial memory status and adaptive chunk size
        memory_status = self._streaming_executor._get_memory_status()
        chunk_size = self._calculate_adaptive_chunk_size(X, memory_status)

        logger.info(
            "Starting streaming transform execution",
            pipeline_id=self._pipeline_id,
            chunk_size=chunk_size,
            total_rows=len(X),
            memory_available_gb=memory_status.available_gb,
        )

        try:
            # Process chunks for transformation
            chunk_number = 0
            self._chunk_transform_results = []

            for chunk in streaming_source.get_chunk_iterator(chunk_size):
                chunk_start_time = time.time()
                chunk_number += 1

                # Monitor memory before processing chunk
                if self.memory_monitoring:
                    current_memory = self._streaming_executor._get_memory_status()
                    self._memory_snapshots.append(current_memory)

                    if current_memory.is_low_memory:
                        # Adapt chunk size for memory pressure
                        chunk_size = min(
                            chunk_size, current_memory.recommended_chunk_size
                        )
                        logger.warning(
                            f"Memory pressure detected, reducing chunk size to {chunk_size}"
                        )

                # Transform chunk using fitted pipeline
                chunk_result = self._transform_chunk(chunk, chunk_number)
                self._chunk_transform_results.append(chunk_result)

                chunk_time = time.time() - chunk_start_time
                logger.debug(
                    f"Processed transform chunk {chunk_number}: {len(chunk)} -> {len(chunk_result)} samples, {chunk_time:.3f}s"
                )

                # Adaptive chunk size adjustment
                if self.adaptive_chunking and chunk_number > 1:
                    chunk_size = self._adaptive_chunk_sizing(
                        chunk_size,
                        chunk_time,
                        current_memory if self.memory_monitoring else memory_status,
                    )

            # Aggregate chunk results for final output
            final_result = self._aggregate_chunk_transform_results()

            logger.info(
                f"Streaming transform completed: {len(final_result)} total samples, {chunk_number} chunks"
            )
            return final_result

        except Exception as e:
            logger.error(f"Streaming transform failed: {e}")
            raise

    def _fit_chunk(
        self, chunk: pd.DataFrame, y_chunk=None, chunk_number: int = 1, **fit_params
    ) -> Dict[str, Any]:
        """
        Fit pipeline on a single chunk with incremental/partial fitting when possible.

        Args:
            chunk: Data chunk to fit on
            y_chunk: Target values for chunk (optional)
            chunk_number: Sequential chunk number
            **fit_params: Additional parameters for fitting

        Returns:
            Dict with chunk fitting results and metadata
        """
        chunk_start_time = time.time()

        try:
            # For first chunk, use standard fit; for subsequent chunks, try partial_fit if available
            if chunk_number == 1:
                # Initial fit on first chunk
                result = super().fit(chunk, y_chunk, **fit_params)
                fit_type = "initial_fit"
            else:
                # Try to use partial_fit for incremental learning on subsequent chunks
                fit_type = "partial_fit"

                # Check if pipeline steps support partial fitting
                supports_partial = True
                for step_name, step_estimator in self.steps:
                    if not hasattr(step_estimator, "partial_fit"):
                        supports_partial = False
                        break

                if supports_partial:
                    # Use partial_fit for incremental learning
                    for step_name, step_estimator in self.steps:
                        if hasattr(step_estimator, "partial_fit"):
                            step_estimator.partial_fit(chunk, y_chunk)
                    result = self
                else:
                    # Fallback: re-fit on accumulated data (memory-intensive but necessary)
                    logger.warning(
                        f"Pipeline steps don't support partial_fit, using full refit on chunk {chunk_number}"
                    )
                    result = super().fit(chunk, y_chunk, **fit_params)
                    fit_type = "full_refit"

            chunk_time = time.time() - chunk_start_time

            return {
                "chunk_number": chunk_number,
                "fit_type": fit_type,
                "samples_processed": len(chunk),
                "processing_time": chunk_time,
                "memory_usage_mb": chunk.memory_usage(deep=True).sum() / (1024 * 1024),
                "timestamp": time.time(),
                "success": True,
            }

        except Exception as e:
            logger.error(f"Chunk fit failed on chunk {chunk_number}: {e}")
            return {
                "chunk_number": chunk_number,
                "fit_type": "failed",
                "samples_processed": len(chunk),
                "processing_time": time.time() - chunk_start_time,
                "error": str(e),
                "success": False,
            }

    def _transform_chunk(self, chunk: pd.DataFrame, chunk_number: int = 1) -> Any:
        """
        Transform a single chunk using the fitted pipeline.

        Args:
            chunk: Data chunk to transform
            chunk_number: Sequential chunk number

        Returns:
            Transformed chunk data
        """
        try:
            # Use the standard sklearn Pipeline transform method
            # This works because we've already fitted the pipeline
            chunk_result = super().transform(chunk)

            logger.debug(
                f"Successfully transformed chunk {chunk_number}: {len(chunk)} -> {len(chunk_result)}"
            )
            return chunk_result

        except Exception as e:
            logger.error(f"Chunk transform failed on chunk {chunk_number}: {e}")
            # Return empty result for failed chunks to maintain structure
            return pd.DataFrame() if hasattr(chunk, "columns") else []

    def _aggregate_chunk_fit_results(self):
        """
        Aggregate results from chunk-based fitting.

        For most sklearn estimators, the final fitted state is already correct
        after processing all chunks. This method can be extended for specific
        aggregation needs.
        """
        successful_chunks = [
            r for r in self._chunk_fit_results if r.get("success", False)
        ]
        failed_chunks = [
            r for r in self._chunk_fit_results if not r.get("success", False)
        ]

        total_samples = sum(r["samples_processed"] for r in successful_chunks)
        total_time = sum(r["processing_time"] for r in self._chunk_fit_results)

        logger.info(
            f"Fit aggregation complete: {len(successful_chunks)}/{len(self._chunk_fit_results)} chunks successful, "
            f"{total_samples} total samples, {total_time:.3f}s total time"
        )

        if failed_chunks:
            logger.warning(
                f"Some chunks failed during fit: {[r['chunk_number'] for r in failed_chunks]}"
            )

        # Store aggregation metadata
        self._execution_context["fit_aggregation"] = {
            "successful_chunks": len(successful_chunks),
            "failed_chunks": len(failed_chunks),
            "total_samples": total_samples,
            "total_processing_time": total_time,
            "average_chunk_time": total_time / len(self._chunk_fit_results)
            if self._chunk_fit_results
            else 0,
        }

    def _aggregate_chunk_transform_results(self) -> Any:
        """
        Aggregate transformation results from all chunks into final output.

        Returns:
            Aggregated transformation results
        """
        if not self._chunk_transform_results:
            return pd.DataFrame()

        try:
            # Filter out empty/failed chunk results
            valid_chunks = [
                chunk
                for chunk in self._chunk_transform_results
                if chunk is not None and (hasattr(chunk, "__len__") and len(chunk) > 0)
            ]

            if not valid_chunks:
                logger.warning("No valid chunk results to aggregate")
                return pd.DataFrame()

            # Determine aggregation strategy based on result type
            first_chunk = valid_chunks[0]

            if isinstance(first_chunk, pd.DataFrame):
                # Concatenate DataFrames
                result = pd.concat(valid_chunks, ignore_index=True)
            elif isinstance(first_chunk, np.ndarray):
                # Concatenate numpy arrays
                result = np.concatenate(valid_chunks, axis=0)
            elif isinstance(first_chunk, list):
                # Flatten lists
                result = [item for chunk in valid_chunks for item in chunk]
            else:
                # Fallback: try to convert to list and flatten
                result = [
                    item
                    for chunk in valid_chunks
                    for item in (chunk if hasattr(chunk, "__iter__") else [chunk])
                ]

            logger.info(
                f"Transform aggregation complete: {len(valid_chunks)} chunks -> {len(result)} total results"
            )
            return result

        except Exception as e:
            logger.error(f"Failed to aggregate chunk transform results: {e}")
            # Return first valid chunk as fallback
            return valid_chunks[0] if valid_chunks else pd.DataFrame()

    def _calculate_adaptive_chunk_size(
        self, data: pd.DataFrame, memory_status: MemoryStatus
    ) -> int:
        """
        Calculate optimal chunk size based on data characteristics and memory status.

        Args:
            data: Input DataFrame to analyze
            memory_status: Current memory status

        Returns:
            int: Recommended chunk size
        """
        # Base chunk size from configuration
        base_chunk_size = self.streaming_config.initial_chunk_size or 1000

        # Adjust based on memory availability
        if memory_status.is_low_memory:
            # Use memory-recommended chunk size for low memory situations
            adaptive_size = memory_status.recommended_chunk_size
        else:
            # Calculate based on data characteristics
            try:
                # Estimate memory per row
                memory_per_row_mb = (
                    data.memory_usage(deep=True).sum() / len(data) / (1024 * 1024)
                )

                # Target chunk memory usage (conservative)
                target_chunk_memory_mb = min(
                    memory_status.available_gb * 0.1 * 1024, 250
                )  # 10% of available or 250MB max

                # Calculate adaptive chunk size
                adaptive_size = (
                    int(target_chunk_memory_mb / memory_per_row_mb)
                    if memory_per_row_mb > 0
                    else base_chunk_size
                )

                # Apply bounds
                adaptive_size = max(
                    min(adaptive_size, 10000), 100
                )  # Between 100 and 10,000 rows

            except Exception as e:
                logger.warning(
                    f"Failed to calculate adaptive chunk size: {e}, using base size"
                )
                adaptive_size = base_chunk_size

        logger.info(
            f"Calculated adaptive chunk size: {adaptive_size} (base: {base_chunk_size}, memory: {memory_status.used_percent:.1f}% used)"
        )
        return adaptive_size

    def _adaptive_chunk_sizing(
        self,
        current_chunk_size: int,
        processing_time: float,
        memory_status: MemoryStatus,
    ) -> int:
        """
        Dynamically adjust chunk size based on performance feedback and memory status.

        Args:
            current_chunk_size: Current chunk size
            processing_time: Time taken to process last chunk
            memory_status: Current memory status

        Returns:
            int: Adjusted chunk size
        """
        if not self.adaptive_chunking:
            return current_chunk_size

        new_chunk_size = current_chunk_size

        # Adjust based on processing time
        if processing_time > 5.0:  # Slow processing (>5 seconds)
            new_chunk_size = max(current_chunk_size // 2, 50)
            logger.debug(
                f"Reducing chunk size due to slow processing: {current_chunk_size} -> {new_chunk_size}"
            )
        elif (
            processing_time < 1.0 and not memory_status.is_low_memory
        ):  # Fast processing and good memory
            new_chunk_size = min(
                current_chunk_size * 2, memory_status.max_safe_chunk_size
            )
            logger.debug(
                f"Increasing chunk size due to fast processing: {current_chunk_size} -> {new_chunk_size}"
            )

        # Adjust based on memory pressure
        if memory_status.is_low_memory:
            new_chunk_size = min(new_chunk_size, memory_status.recommended_chunk_size)
            logger.debug(f"Adjusting chunk size for memory pressure: {new_chunk_size}")

        # Ensure reasonable bounds
        new_chunk_size = max(
            min(new_chunk_size, 50000), 10
        )  # Between 10 and 50,000 rows

        return new_chunk_size

    def get_streaming_metadata(self) -> Dict[str, Any]:
        """
        Get comprehensive metadata about streaming execution.

        Returns:
            Dict with streaming execution details
        """
        return {
            "streaming_activated": self._streaming_activated,
            "threshold_mb": self.streaming_threshold_mb,
            "adaptive_chunking": self.adaptive_chunking,
            "memory_monitoring": self.memory_monitoring,
            "chunk_fit_results": len(self._chunk_fit_results),
            "chunk_transform_results": len(self._chunk_transform_results),
            "memory_snapshots": len(self._memory_snapshots),
            "final_chunk_size": self._adaptive_chunk_size,
            "fit_aggregation": self._execution_context.get("fit_aggregation", {}),
            "memory_efficiency": {
                "initial_memory_gb": self._memory_snapshots[0].available_gb
                if self._memory_snapshots
                else None,
                "final_memory_gb": self._memory_snapshots[-1].available_gb
                if self._memory_snapshots
                else None,
                "peak_memory_usage_percent": max(
                    (m.used_percent for m in self._memory_snapshots), default=0
                ),
            },
        }

    def clear_streaming_cache(self):
        """
        Clear streaming-related cached data and results.

        Useful for memory management between pipeline runs.
        """
        self._chunk_fit_results.clear()
        self._chunk_transform_results.clear()
        self._memory_snapshots.clear()

        if self._streaming_executor:
            # Clear any result buffers in the streaming executor
            for query_id in list(self._streaming_executor._result_buffers.keys()):
                self._streaming_executor.clear_buffer(query_id)

        logger.info("Streaming cache cleared", pipeline_id=self._pipeline_id)


class SklearnStreamingAdapter:
    """
    Adapter that bridges sklearn pipelines with StreamingQueryExecutor for seamless integration.

    This adapter enables any sklearn Pipeline (including DataSciencePipeline) to work
    seamlessly with the existing StreamingQueryExecutor infrastructure, providing
    memory-bounded processing capabilities while maintaining full sklearn API compatibility.

    Key Features:
    - Sklearn pipeline execution within streaming chunks
    - Partial fitting support for incremental learning algorithms
    - Result aggregation across streaming chunks with memory management
    - Memory monitoring integration with adaptive chunk sizing
    - Error handling and recovery in streaming context
    - Performance monitoring and optimization

    Usage:
        # Create adapter for any sklearn pipeline
        adapter = SklearnStreamingAdapter(pipeline, streaming_executor)

        # Execute with memory-bounded streaming
        results = adapter.execute_streaming(data, operation='transform')

        # Or use for fitting with streaming
        adapter.execute_streaming(X_train, y_train, operation='fit')
    """

    def __init__(
        self,
        sklearn_pipeline: Pipeline,
        streaming_executor: Optional[StreamingQueryExecutor] = None,
        memory_monitoring: bool = True,
        adaptive_chunking: bool = True,
        performance_tracking: bool = True,
    ):
        """
        Initialize SklearnStreamingAdapter.

        Args:
            sklearn_pipeline: Any sklearn Pipeline (or DataSciencePipeline)
            streaming_executor: StreamingQueryExecutor instance (auto-created if None)
            memory_monitoring: Enable real-time memory monitoring
            adaptive_chunking: Enable adaptive chunk size optimization
            performance_tracking: Enable performance metrics tracking
        """
        self.sklearn_pipeline = sklearn_pipeline
        self.streaming_executor = streaming_executor or StreamingQueryExecutor()
        self.memory_monitoring = memory_monitoring
        self.adaptive_chunking = adaptive_chunking
        self.performance_tracking = performance_tracking

        # Adapter state
        self.adapter_id = str(uuid.uuid4())[:8]
        self.execution_history: List[Dict[str, Any]] = []
        self.performance_metrics: Dict[str, Any] = {}
        self._chunk_results_cache: Dict[str, List[Any]] = {}

        logger.info(
            "SklearnStreamingAdapter initialized",
            adapter_id=self.adapter_id,
            pipeline_type=type(sklearn_pipeline).__name__,
            memory_monitoring=memory_monitoring,
            adaptive_chunking=adaptive_chunking,
        )

    def execute_streaming(
        self,
        X,
        y=None,
        operation: str = "transform",
        initial_chunk_size: Optional[int] = None,
        **operation_params,
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Execute sklearn pipeline operation with streaming support.

        Args:
            X: Input data (DataFrame, array, or streaming source)
            y: Target data for fitting (optional)
            operation: Operation to perform ('fit', 'transform', 'fit_transform', 'predict')
            initial_chunk_size: Initial chunk size (adaptive if None)
            **operation_params: Additional parameters for the operation

        Returns:
            Tuple of (operation_results, execution_metadata)
        """
        execution_id = f"{operation}_{self.adapter_id}_{int(time.time())}"
        start_time = time.time()

        logger.info(
            f"Starting streaming {operation} execution",
            adapter_id=self.adapter_id,
            execution_id=execution_id,
            operation=operation,
        )

        try:
            # Convert input to streaming source
            if isinstance(X, pd.DataFrame):
                data_source = DataFrameStreamingSource(X)
            elif hasattr(X, "get_chunk_iterator"):
                data_source = X  # Already a streaming source
            else:
                # Convert array-like to DataFrame
                X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
                data_source = DataFrameStreamingSource(X_df)

            # Get initial memory status
            memory_status = self.streaming_executor._get_memory_status()
            chunk_size = self._calculate_initial_chunk_size(
                data_source, memory_status, initial_chunk_size
            )

            # Execute streaming operation
            operation_results, chunk_metadata = self._execute_operation_streaming(
                data_source, y, operation, chunk_size, **operation_params
            )

            # Calculate execution metadata
            execution_time = time.time() - start_time
            execution_metadata = self._build_execution_metadata(
                execution_id, operation, execution_time, chunk_metadata
            )

            # Store execution history
            if self.performance_tracking:
                self.execution_history.append(execution_metadata)
                self._update_performance_metrics(execution_metadata)

            logger.info(
                f"Streaming {operation} execution completed",
                adapter_id=self.adapter_id,
                execution_id=execution_id,
                execution_time=execution_time,
                chunks_processed=chunk_metadata.get("total_chunks", 0),
            )

            return operation_results, execution_metadata

        except Exception as e:
            execution_time = time.time() - start_time
            error_metadata = {
                "execution_id": execution_id,
                "operation": operation,
                "execution_time": execution_time,
                "error_type": type(e).__name__,
                "error_message": str(e),
                "adapter_id": self.adapter_id,
                "success": False,
            }

            logger.error(
                f"Streaming {operation} execution failed",
                adapter_id=self.adapter_id,
                execution_id=execution_id,
                error=str(e),
            )

            raise PipelineError(
                f"SklearnStreamingAdapter {operation} failed: {str(e)}",
                classification=ErrorClassification.STREAMING_ERROR,
                context=error_metadata,
            ) from e

    def _execute_operation_streaming(
        self,
        data_source: StreamingDataSource,
        y,
        operation: str,
        chunk_size: int,
        **operation_params,
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Execute the specific sklearn operation in streaming mode.

        Args:
            data_source: Streaming data source
            y: Target data (for fitting operations)
            operation: Operation to perform
            chunk_size: Size of chunks to process
            **operation_params: Additional parameters

        Returns:
            Tuple of (operation_results, chunk_processing_metadata)
        """
        chunk_results = []
        chunk_metadata = {
            "total_chunks": 0,
            "successful_chunks": 0,
            "failed_chunks": 0,
            "processing_times": [],
            "memory_snapshots": [],
            "chunk_sizes": [],
        }

        try:
            # Process data in chunks
            chunk_number = 0
            current_chunk_size = chunk_size

            for chunk in data_source.get_chunk_iterator(current_chunk_size):
                chunk_start_time = time.time()
                chunk_number += 1
                chunk_metadata["total_chunks"] = chunk_number

                # Monitor memory if enabled
                if self.memory_monitoring:
                    memory_status = self.streaming_executor._get_memory_status()
                    chunk_metadata["memory_snapshots"].append(
                        {
                            "chunk_number": chunk_number,
                            "memory_used_percent": memory_status.used_percent,
                            "available_gb": memory_status.available_gb,
                            "is_low_memory": memory_status.is_low_memory,
                        }
                    )

                    # Adapt chunk size if needed
                    if self.adaptive_chunking and memory_status.is_low_memory:
                        current_chunk_size = min(
                            current_chunk_size, memory_status.recommended_chunk_size
                        )
                        logger.warning(
                            f"Reducing chunk size due to memory pressure: {current_chunk_size}"
                        )

                try:
                    # Get corresponding y chunk if provided
                    if y is not None:
                        start_idx = (chunk_number - 1) * current_chunk_size
                        end_idx = start_idx + len(chunk)
                        if hasattr(y, "iloc"):
                            y_chunk = y.iloc[start_idx:end_idx]
                        else:
                            y_chunk = (
                                y[start_idx:end_idx]
                                if hasattr(y, "__getitem__")
                                else None
                            )
                    else:
                        y_chunk = None

                    # Execute operation on chunk
                    chunk_result = self._execute_chunk_operation(
                        chunk, y_chunk, operation, chunk_number, **operation_params
                    )

                    chunk_results.append(chunk_result)
                    chunk_metadata["successful_chunks"] += 1

                    # Track chunk processing time
                    chunk_time = time.time() - chunk_start_time
                    chunk_metadata["processing_times"].append(chunk_time)
                    chunk_metadata["chunk_sizes"].append(len(chunk))

                    logger.debug(
                        f"Processed chunk {chunk_number}: {len(chunk)} samples, {chunk_time:.3f}s"
                    )

                    # Adaptive chunk sizing based on performance
                    if self.adaptive_chunking and chunk_number > 1:
                        current_chunk_size = self._adapt_chunk_size(
                            current_chunk_size,
                            chunk_time,
                            memory_status if self.memory_monitoring else None,
                        )

                except Exception as chunk_error:
                    chunk_metadata["failed_chunks"] += 1
                    logger.error(
                        f"Chunk {chunk_number} processing failed: {chunk_error}"
                    )

                    # For transform operations, continue with next chunk
                    if operation in ["transform", "predict"]:
                        chunk_results.append(None)  # Placeholder for failed chunk
                        continue
                    else:
                        # For fit operations, chunk failures are more serious
                        raise chunk_error

            # Aggregate results based on operation type
            final_result = self._aggregate_chunk_results(chunk_results, operation)

            return final_result, chunk_metadata

        except Exception as e:
            logger.error(f"Streaming operation {operation} failed: {e}")
            raise

    def _execute_chunk_operation(
        self, chunk, y_chunk, operation: str, chunk_number: int, **operation_params
    ) -> Any:
        """
        Execute sklearn operation on a single chunk.

        Args:
            chunk: Data chunk
            y_chunk: Target chunk (for fitting)
            operation: Operation to perform
            chunk_number: Sequential chunk number
            **operation_params: Additional operation parameters

        Returns:
            Operation result for this chunk
        """
        try:
            if operation == "fit":
                if chunk_number == 1:
                    # Initial fit on first chunk
                    result = self.sklearn_pipeline.fit(
                        chunk, y_chunk, **operation_params
                    )
                else:
                    # Try partial_fit for subsequent chunks if supported
                    if hasattr(self.sklearn_pipeline, "partial_fit"):
                        result = self.sklearn_pipeline.partial_fit(chunk, y_chunk)
                    else:
                        # Check if individual steps support partial fitting
                        self._partial_fit_steps(chunk, y_chunk)
                        result = self.sklearn_pipeline

            elif operation == "transform":
                result = self.sklearn_pipeline.transform(chunk)

            elif operation == "fit_transform":
                if chunk_number == 1:
                    result = self.sklearn_pipeline.fit_transform(
                        chunk, y_chunk, **operation_params
                    )
                else:
                    # For subsequent chunks: partial fit + transform
                    if hasattr(self.sklearn_pipeline, "partial_fit"):
                        self.sklearn_pipeline.partial_fit(chunk, y_chunk)
                    else:
                        self._partial_fit_steps(chunk, y_chunk)
                    result = self.sklearn_pipeline.transform(chunk)

            elif operation == "predict":
                result = self.sklearn_pipeline.predict(chunk)

            else:
                # Generic operation - try to call the method
                if hasattr(self.sklearn_pipeline, operation):
                    method = getattr(self.sklearn_pipeline, operation)
                    result = (
                        method(chunk, y_chunk, **operation_params)
                        if y_chunk is not None
                        else method(chunk, **operation_params)
                    )
                else:
                    raise ValueError(f"Unknown operation: {operation}")

            return result

        except Exception as e:
            logger.error(
                f"Chunk operation {operation} failed on chunk {chunk_number}: {e}"
            )
            raise

    def _partial_fit_steps(self, chunk, y_chunk):
        """
        Apply partial_fit to individual pipeline steps that support it.

        Args:
            chunk: Data chunk
            y_chunk: Target chunk
        """
        # Transform data through the pipeline up to each step
        current_data = chunk

        for step_name, step_estimator in self.sklearn_pipeline.steps:
            if hasattr(step_estimator, "partial_fit"):
                # Apply partial fit on this step
                step_estimator.partial_fit(current_data, y_chunk)

            # Transform data for next step
            if hasattr(step_estimator, "transform"):
                current_data = step_estimator.transform(current_data)

    def _aggregate_chunk_results(self, chunk_results: List[Any], operation: str) -> Any:
        """
        Aggregate results from all chunks based on operation type.

        Args:
            chunk_results: List of results from each chunk
            operation: Operation that was performed

        Returns:
            Aggregated result
        """
        # Filter out None results from failed chunks
        valid_results = [r for r in chunk_results if r is not None]

        if not valid_results:
            logger.warning("No valid chunk results to aggregate")
            return None

        try:
            if operation in ["fit", "fit_transform"] and len(valid_results) > 0:
                # For fit operations, return the fitted pipeline (last valid result)
                if operation == "fit":
                    return self.sklearn_pipeline  # Return the fitted pipeline
                else:
                    # For fit_transform, aggregate the transformed results
                    return self._concatenate_results(valid_results)

            elif operation in ["transform", "predict"]:
                # Concatenate transformation/prediction results
                return self._concatenate_results(valid_results)

            else:
                # For other operations, try to concatenate or return list
                try:
                    return self._concatenate_results(valid_results)
                except:
                    return valid_results  # Return as list if concatenation fails

        except Exception as e:
            logger.error(f"Failed to aggregate {operation} results: {e}")
            return valid_results[0] if valid_results else None

    def _concatenate_results(self, results: List[Any]) -> Any:
        """
        Concatenate results intelligently based on their type.

        Args:
            results: List of results to concatenate

        Returns:
            Concatenated result
        """
        if not results:
            return None

        first_result = results[0]

        if isinstance(first_result, pd.DataFrame):
            return pd.concat(results, ignore_index=True)
        elif isinstance(first_result, np.ndarray):
            return np.concatenate(results, axis=0)
        elif isinstance(first_result, list):
            return [item for result in results for item in result]
        else:
            # Fallback: return as list
            return results

    def _calculate_initial_chunk_size(
        self,
        data_source: StreamingDataSource,
        memory_status: MemoryStatus,
        provided_chunk_size: Optional[int],
    ) -> int:
        """
        Calculate initial chunk size based on data characteristics and memory.

        Args:
            data_source: Streaming data source
            memory_status: Current memory status
            provided_chunk_size: User-provided chunk size (takes precedence)

        Returns:
            Calculated chunk size
        """
        if provided_chunk_size is not None:
            return provided_chunk_size

        # Use memory-based calculation
        if memory_status.is_low_memory:
            chunk_size = memory_status.recommended_chunk_size
        else:
            # Conservative default based on available memory
            available_gb = memory_status.available_gb
            if available_gb > 8:
                chunk_size = 5000  # Large chunks for high-memory systems
            elif available_gb > 4:
                chunk_size = 2000  # Medium chunks
            else:
                chunk_size = 1000  # Small chunks for constrained systems

        # Estimate memory per row if possible
        try:
            memory_per_row = data_source.estimate_memory_per_row()
            if memory_per_row > 0:
                # Target 100MB per chunk max
                target_memory_mb = 100
                calculated_chunk_size = int(
                    (target_memory_mb * 1024 * 1024) / memory_per_row
                )
                chunk_size = min(chunk_size, max(calculated_chunk_size, 100))
        except:
            pass  # Use default chunk size if estimation fails

        logger.info(
            f"Calculated initial chunk size: {chunk_size} (memory: {memory_status.used_percent:.1f}% used)"
        )
        return chunk_size

    def _adapt_chunk_size(
        self,
        current_chunk_size: int,
        processing_time: float,
        memory_status: Optional[MemoryStatus],
    ) -> int:
        """
        Adapt chunk size based on performance feedback.

        Args:
            current_chunk_size: Current chunk size
            processing_time: Time taken for last chunk
            memory_status: Current memory status (optional)

        Returns:
            Adapted chunk size
        """
        new_chunk_size = current_chunk_size

        # Adapt based on processing time
        if processing_time > 10.0:  # Very slow (>10 seconds)
            new_chunk_size = max(current_chunk_size // 4, 50)
        elif processing_time > 5.0:  # Slow (>5 seconds)
            new_chunk_size = max(current_chunk_size // 2, 100)
        elif processing_time < 0.5 and (
            not memory_status or not memory_status.is_low_memory
        ):
            # Fast processing and good memory - increase chunk size
            new_chunk_size = min(current_chunk_size * 2, 10000)

        # Apply memory constraints
        if memory_status and memory_status.is_low_memory:
            new_chunk_size = min(new_chunk_size, memory_status.recommended_chunk_size)

        # Ensure reasonable bounds
        new_chunk_size = max(min(new_chunk_size, 50000), 10)

        if new_chunk_size != current_chunk_size:
            logger.debug(
                f"Adapted chunk size: {current_chunk_size} -> {new_chunk_size} (time: {processing_time:.3f}s)"
            )

        return new_chunk_size

    def _build_execution_metadata(
        self,
        execution_id: str,
        operation: str,
        execution_time: float,
        chunk_metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Build comprehensive execution metadata.

        Args:
            execution_id: Unique execution identifier
            operation: Operation that was performed
            execution_time: Total execution time
            chunk_metadata: Metadata from chunk processing

        Returns:
            Comprehensive execution metadata
        """
        total_chunks = chunk_metadata.get("total_chunks", 0)
        successful_chunks = chunk_metadata.get("successful_chunks", 0)
        processing_times = chunk_metadata.get("processing_times", [])
        chunk_sizes = chunk_metadata.get("chunk_sizes", [])

        metadata = {
            "execution_id": execution_id,
            "adapter_id": self.adapter_id,
            "operation": operation,
            "execution_time_seconds": execution_time,
            "success": True,
            "timestamp": time.time(),
            # Chunk processing statistics
            "chunk_statistics": {
                "total_chunks": total_chunks,
                "successful_chunks": successful_chunks,
                "failed_chunks": chunk_metadata.get("failed_chunks", 0),
                "success_rate": successful_chunks / total_chunks
                if total_chunks > 0
                else 0,
                "average_chunk_processing_time": sum(processing_times)
                / len(processing_times)
                if processing_times
                else 0,
                "total_samples_processed": sum(chunk_sizes),
                "average_chunk_size": sum(chunk_sizes) / len(chunk_sizes)
                if chunk_sizes
                else 0,
            },
            # Performance metrics
            "performance_metrics": {
                "samples_per_second": sum(chunk_sizes) / execution_time
                if execution_time > 0
                else 0,
                "chunks_per_second": total_chunks / execution_time
                if execution_time > 0
                else 0,
                "memory_efficiency": self._calculate_memory_efficiency(chunk_metadata),
            },
            # Pipeline information
            "pipeline_info": {
                "pipeline_type": type(self.sklearn_pipeline).__name__,
                "pipeline_steps": [
                    step_name for step_name, _ in self.sklearn_pipeline.steps
                ]
                if hasattr(self.sklearn_pipeline, "steps")
                else [],
                "supports_partial_fit": self._check_partial_fit_support(),
            },
        }

        return metadata

    def _calculate_memory_efficiency(
        self, chunk_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate memory efficiency metrics from chunk processing.

        Args:
            chunk_metadata: Metadata from chunk processing

        Returns:
            Memory efficiency metrics
        """
        memory_snapshots = chunk_metadata.get("memory_snapshots", [])

        if not memory_snapshots:
            return {"memory_monitoring_enabled": False}

        memory_usage_percentages = [s["memory_used_percent"] for s in memory_snapshots]
        available_gb_values = [s["available_gb"] for s in memory_snapshots]
        low_memory_count = sum(1 for s in memory_snapshots if s["is_low_memory"])

        return {
            "memory_monitoring_enabled": True,
            "peak_memory_usage_percent": max(memory_usage_percentages)
            if memory_usage_percentages
            else 0,
            "average_memory_usage_percent": sum(memory_usage_percentages)
            / len(memory_usage_percentages)
            if memory_usage_percentages
            else 0,
            "minimum_available_gb": min(available_gb_values)
            if available_gb_values
            else 0,
            "memory_pressure_chunks": low_memory_count,
            "memory_stability": (len(memory_snapshots) - low_memory_count)
            / len(memory_snapshots)
            if memory_snapshots
            else 1.0,
        }

    def _check_partial_fit_support(self) -> bool:
        """
        Check if the pipeline supports partial fitting.

        Returns:
            True if partial fitting is supported
        """
        if hasattr(self.sklearn_pipeline, "partial_fit"):
            return True

        if hasattr(self.sklearn_pipeline, "steps"):
            # Check if any steps support partial_fit
            for step_name, step_estimator in self.sklearn_pipeline.steps:
                if hasattr(step_estimator, "partial_fit"):
                    return True

        return False

    def _update_performance_metrics(self, execution_metadata: Dict[str, Any]):
        """
        Update internal performance metrics with execution data.

        Args:
            execution_metadata: Metadata from execution
        """
        operation = execution_metadata["operation"]

        if operation not in self.performance_metrics:
            self.performance_metrics[operation] = {
                "execution_count": 0,
                "total_execution_time": 0,
                "total_samples_processed": 0,
                "average_samples_per_second": 0,
                "success_rate": 0,
                "memory_efficiency_scores": [],
            }

        metrics = self.performance_metrics[operation]
        metrics["execution_count"] += 1
        metrics["total_execution_time"] += execution_metadata["execution_time_seconds"]

        chunk_stats = execution_metadata.get("chunk_statistics", {})
        metrics["total_samples_processed"] += chunk_stats.get(
            "total_samples_processed", 0
        )
        metrics["success_rate"] = (
            metrics["success_rate"] * (metrics["execution_count"] - 1)
            + (1.0 if execution_metadata["success"] else 0.0)
        ) / metrics["execution_count"]

        # Update samples per second average
        if metrics["total_execution_time"] > 0:
            metrics["average_samples_per_second"] = (
                metrics["total_samples_processed"] / metrics["total_execution_time"]
            )

        # Track memory efficiency
        perf_metrics = execution_metadata.get("performance_metrics", {})
        memory_efficiency = perf_metrics.get("memory_efficiency", {})
        if memory_efficiency.get("memory_monitoring_enabled", False):
            stability_score = memory_efficiency.get("memory_stability", 0)
            metrics["memory_efficiency_scores"].append(stability_score)

    # Public utility methods
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive performance summary across all operations.

        Returns:
            Performance summary dictionary
        """
        return {
            "adapter_id": self.adapter_id,
            "pipeline_type": type(self.sklearn_pipeline).__name__,
            "total_executions": len(self.execution_history),
            "operations_performed": list(self.performance_metrics.keys()),
            "performance_by_operation": self.performance_metrics,
            "configuration": {
                "memory_monitoring": self.memory_monitoring,
                "adaptive_chunking": self.adaptive_chunking,
                "performance_tracking": self.performance_tracking,
            },
        }

    def clear_performance_history(self):
        """
        Clear performance history and metrics.

        Useful for starting fresh performance tracking.
        """
        self.execution_history.clear()
        self.performance_metrics.clear()
        self._chunk_results_cache.clear()

        logger.info("Performance history cleared", adapter_id=self.adapter_id)
