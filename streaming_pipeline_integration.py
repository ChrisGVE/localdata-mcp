"""
Streaming Architecture Integration for LocalData MCP v2.0

This module provides the streaming integration layer that preserves LocalData MCP's
proven memory-safe processing while enabling complex data science pipelines.

Design Principles:
1. Memory-Safe Processing Preservation - Build on proven StreamingQueryExecutor
2. Pipeline Streaming Integration - Multi-stage streaming with intermediate results
3. Performance Optimization - Maintain sub-100ms tool discovery
4. Large Dataset Handling - Strategies for complex analytics on massive datasets
5. Resource Management - Adaptive resource allocation and monitoring

Architectural Integration:
- Bridges StreamingQueryExecutor with AnalysisPipelineBase
- Provides streaming-aware pipeline composition
- Enables chunk-based pipeline execution
- Maintains compatibility with all 15 LocalData domains
"""

from typing import Any, Dict, List, Optional, Tuple, Union, Iterator, Protocol
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import time
import logging
import weakref
from contextlib import contextmanager

import pandas as pd

# Import existing streaming architecture
from src.localdata_mcp.streaming_executor import (
    StreamingQueryExecutor,
    StreamingDataSource,
    ResultBuffer,
    MemoryStatus,
    ChunkMetrics,
    create_streaming_source
)

# Import existing pipeline architecture
from src.localdata_mcp.pipeline.base import (
    AnalysisPipelineBase,
    PipelineResult,
    PipelineState,
    StreamingConfig,
    CompositionMetadata,
    PipelineError,
    ErrorClassification
)

# Import configuration and logging
from src.localdata_mcp.config_manager import PerformanceConfig
from src.localdata_mcp.logging_manager import get_logger

logger = get_logger(__name__)


# ============================================================================
# Core Streaming Pipeline Architecture
# ============================================================================

class StreamingStage(Enum):
    """Pipeline execution stages for streaming."""
    DATA_INGESTION = "data_ingestion"
    PREPROCESSING = "preprocessing" 
    ANALYSIS = "analysis"
    POSTPROCESSING = "postprocessing"
    RESULTS_AGGREGATION = "results_aggregation"


class ChunkProcessingMode(Enum):
    """Modes for processing chunks in streaming pipelines."""
    SEQUENTIAL = "sequential"  # Process chunks one by one
    BATCH_ACCUMULATE = "batch_accumulate"  # Accumulate chunks for batch processing
    STATELESS = "stateless"  # Each chunk independent
    STATEFUL = "stateful"  # Maintain state across chunks


@dataclass
class StreamingPipelineConfig:
    """Configuration for streaming-aware pipeline execution."""
    
    # Memory management
    memory_limit_mb: int = 1000
    chunk_size: int = 1000
    buffer_limit_mb: int = 500
    
    # Pipeline streaming behavior
    processing_mode: ChunkProcessingMode = ChunkProcessingMode.SEQUENTIAL
    intermediate_results_caching: bool = True
    early_results_enabled: bool = True
    
    # Performance optimization
    parallel_stages: bool = False
    memory_monitoring_interval_seconds: float = 1.0
    adaptive_chunk_sizing: bool = True
    
    # Error handling and recovery
    partial_results_on_error: bool = True
    max_retry_attempts: int = 3
    fallback_to_memory_mode: bool = True
    
    # Cross-domain compatibility
    maintain_tool_discovery_performance: bool = True
    preserve_metadata_flow: bool = True


@dataclass 
class ChunkProcessingContext:
    """Context information for processing individual chunks."""
    
    chunk_number: int
    total_chunks_estimate: Optional[int]
    chunk_size: int
    
    # Memory status
    memory_status: MemoryStatus
    buffer_usage_mb: float
    
    # Pipeline context
    current_stage: StreamingStage
    pipeline_state: Dict[str, Any]
    intermediate_results: Dict[str, Any]
    
    # Performance tracking
    chunk_start_time: float
    cumulative_processing_time: float
    
    # Error handling
    errors_encountered: List[Dict[str, Any]] = field(default_factory=list)
    retry_count: int = 0


@dataclass
class StreamingPipelineResult:
    """Extended pipeline result with streaming context."""
    
    # Core result (extends PipelineResult)
    success: bool
    data: Optional[pd.DataFrame]
    metadata: Dict[str, Any]
    
    # Streaming-specific information
    chunks_processed: int
    total_rows_processed: int
    streaming_enabled: bool
    
    # Performance metrics
    execution_time_seconds: float
    memory_peak_mb: float
    throughput_rows_per_second: float
    
    # Pipeline composition context
    composition_metadata: Optional[CompositionMetadata]
    intermediate_results: Dict[str, Any] = field(default_factory=dict)
    
    # Error and recovery information
    partial_results: bool = False
    error_info: Optional[Dict[str, Any]] = None
    recovery_options: List[Dict[str, Any]] = field(default_factory=list)


# ============================================================================
# Streaming-Aware Pipeline Interface
# ============================================================================

class StreamingPipelineProtocol(Protocol):
    """Protocol for pipelines that support streaming execution."""
    
    def supports_streaming(self) -> bool:
        """Check if this pipeline supports streaming execution."""
        ...
    
    def get_streaming_requirements(self) -> Dict[str, Any]:
        """Get streaming requirements and constraints."""
        ...
    
    def process_chunk(
        self, 
        chunk: pd.DataFrame, 
        context: ChunkProcessingContext
    ) -> Tuple[Any, Dict[str, Any]]:
        """Process a single chunk of data."""
        ...
    
    def aggregate_chunk_results(
        self, 
        chunk_results: List[Tuple[Any, Dict[str, Any]]], 
        context: Dict[str, Any]
    ) -> Tuple[Any, Dict[str, Any]]:
        """Aggregate results from multiple chunks."""
        ...
    
    def handle_streaming_error(
        self, 
        error: Exception, 
        context: ChunkProcessingContext
    ) -> Dict[str, Any]:
        """Handle errors during streaming execution."""
        ...


# ============================================================================
# Core Streaming Pipeline Executor
# ============================================================================

class StreamingPipelineExecutor:
    """
    Integration layer between StreamingQueryExecutor and AnalysisPipelineBase.
    
    Provides streaming execution for complex data science pipelines while
    preserving LocalData MCP's proven memory-safe processing architecture.
    """
    
    def __init__(
        self, 
        streaming_config: Optional[StreamingPipelineConfig] = None,
        performance_config: Optional[PerformanceConfig] = None
    ):
        """Initialize streaming pipeline executor."""
        self.streaming_config = streaming_config or StreamingPipelineConfig()
        self.performance_config = performance_config
        
        # Initialize core streaming executor
        self.streaming_executor = StreamingQueryExecutor(performance_config)
        
        # Pipeline state management
        self._active_pipelines: Dict[str, AnalysisPipelineBase] = {}
        self._chunk_state: Dict[str, Dict[str, Any]] = {}
        self._intermediate_buffers: Dict[str, ResultBuffer] = {}
        
        # Performance monitoring
        self._execution_metrics: List[Dict[str, Any]] = []
        self._memory_snapshots: List[MemoryStatus] = []
        
        logger.info(
            "StreamingPipelineExecutor initialized",
            memory_limit_mb=self.streaming_config.memory_limit_mb,
            chunk_size=self.streaming_config.chunk_size,
            processing_mode=self.streaming_config.processing_mode.value
        )
    
    def execute_pipeline(
        self,
        pipeline: AnalysisPipelineBase,
        data_source: Union[StreamingDataSource, pd.DataFrame],
        pipeline_id: Optional[str] = None
    ) -> StreamingPipelineResult:
        """
        Execute analysis pipeline with streaming support.
        
        Args:
            pipeline: Analysis pipeline to execute
            data_source: Streaming data source or DataFrame
            pipeline_id: Unique identifier for pipeline execution
            
        Returns:
            StreamingPipelineResult with analysis results and streaming metadata
        """
        pipeline_id = pipeline_id or self._generate_pipeline_id()
        start_time = time.time()
        
        try:
            # Register pipeline for state management
            self._active_pipelines[pipeline_id] = pipeline
            self._chunk_state[pipeline_id] = {
                "processed_chunks": 0,
                "total_rows": 0,
                "intermediate_results": {},
                "errors": [],
                "stage": StreamingStage.DATA_INGESTION
            }
            
            # Determine execution strategy
            if self._should_use_streaming(data_source, pipeline):
                result = self._execute_streaming_pipeline(
                    pipeline, data_source, pipeline_id
                )
            else:
                result = self._execute_standard_pipeline(
                    pipeline, data_source, pipeline_id
                )
            
            # Calculate performance metrics
            execution_time = time.time() - start_time
            result.execution_time_seconds = execution_time
            result.throughput_rows_per_second = (
                result.total_rows_processed / execution_time if execution_time > 0 else 0
            )
            
            # Log performance
            logger.info(
                "Pipeline execution completed",
                pipeline_id=pipeline_id,
                execution_time=execution_time,
                chunks_processed=result.chunks_processed,
                total_rows=result.total_rows_processed,
                throughput=result.throughput_rows_per_second
            )
            
            return result
            
        except Exception as e:
            # Handle pipeline errors with recovery
            execution_time = time.time() - start_time
            return self._handle_pipeline_execution_error(
                e, pipeline_id, execution_time
            )
        
        finally:
            # Cleanup pipeline state
            self._cleanup_pipeline_state(pipeline_id)
    
    def _should_use_streaming(
        self, 
        data_source: Union[StreamingDataSource, pd.DataFrame], 
        pipeline: AnalysisPipelineBase
    ) -> bool:
        """Determine if streaming execution should be used."""
        
        # Always use streaming for StreamingDataSource
        if isinstance(data_source, StreamingDataSource):
            return True
            
        # Check if pipeline explicitly requires streaming
        if hasattr(pipeline, 'streaming_config') and pipeline.streaming_config.enabled:
            return True
            
        # Check data size threshold
        if isinstance(data_source, pd.DataFrame):
            data_size_mb = data_source.memory_usage(deep=True).sum() / (1024 * 1024)
            if data_size_mb > self.streaming_config.memory_limit_mb / 2:
                return True
                
        return False
    
    def _execute_streaming_pipeline(
        self,
        pipeline: AnalysisPipelineBase,
        data_source: Union[StreamingDataSource, pd.DataFrame],
        pipeline_id: str
    ) -> StreamingPipelineResult:
        """
        Execute pipeline with streaming processing.
        
        This is the core integration that bridges StreamingQueryExecutor
        with complex multi-stage analysis pipelines.
        """
        logger.info(f"Starting streaming pipeline execution: {pipeline_id}")
        
        # Convert DataFrame to streaming source if needed
        if isinstance(data_source, pd.DataFrame):
            data_source = self._dataframe_to_streaming_source(data_source)
        
        # Initialize streaming context
        streaming_context = self._initialize_streaming_context(
            pipeline, data_source, pipeline_id
        )
        
        # Execute streaming stages
        chunk_results = []
        total_rows_processed = 0
        memory_peak = 0.0
        
        try:
            # Stage 1: Data Ingestion with Streaming
            self._update_pipeline_stage(pipeline_id, StreamingStage.DATA_INGESTION)
            
            for chunk_num, chunk in enumerate(
                self._get_chunk_iterator(data_source, streaming_context)
            ):
                # Update chunk processing context
                chunk_context = self._create_chunk_context(
                    chunk_num, chunk, streaming_context, pipeline_id
                )
                
                # Memory monitoring
                current_memory = self.streaming_executor._get_memory_status()
                memory_peak = max(memory_peak, current_memory.used_percent)
                
                # Check memory bounds
                if current_memory.is_low_memory:
                    logger.warning(
                        f"Low memory detected during chunk {chunk_num}",
                        memory_percent=current_memory.used_percent
                    )
                    
                    # Apply memory pressure handling
                    chunk = self._handle_memory_pressure(
                        chunk, chunk_context, pipeline_id
                    )
                
                # Stage 2: Preprocessing (chunk-aware)
                self._update_pipeline_stage(pipeline_id, StreamingStage.PREPROCESSING)
                preprocessed_chunk = self._preprocess_chunk(
                    chunk, chunk_context, pipeline
                )
                
                # Stage 3: Analysis (pipeline-specific)
                self._update_pipeline_stage(pipeline_id, StreamingStage.ANALYSIS)
                analysis_result = self._analyze_chunk(
                    preprocessed_chunk, chunk_context, pipeline
                )
                
                # Stage 4: Postprocessing
                self._update_pipeline_stage(pipeline_id, StreamingStage.POSTPROCESSING)
                processed_result = self._postprocess_chunk(
                    analysis_result, chunk_context, pipeline
                )
                
                # Collect chunk result
                chunk_results.append({
                    'chunk_number': chunk_num,
                    'rows_processed': len(chunk),
                    'result': processed_result,
                    'metadata': chunk_context.__dict__
                })
                
                total_rows_processed += len(chunk)
                
                # Update pipeline state
                self._update_chunk_state(pipeline_id, chunk_num, processed_result)
                
                # Early results if enabled
                if (self.streaming_config.early_results_enabled and 
                    chunk_num == 0 and 
                    processed_result is not None):
                    logger.info(f"Early results available after chunk {chunk_num}")
            
            # Stage 5: Results Aggregation
            self._update_pipeline_stage(pipeline_id, StreamingStage.RESULTS_AGGREGATION)
            final_result, final_metadata = self._aggregate_streaming_results(
                chunk_results, pipeline, pipeline_id
            )
            
            # Build streaming pipeline result
            return StreamingPipelineResult(
                success=True,
                data=final_result,
                metadata=final_metadata,
                chunks_processed=len(chunk_results),
                total_rows_processed=total_rows_processed,
                streaming_enabled=True,
                execution_time_seconds=0.0,  # Set by caller
                memory_peak_mb=memory_peak,
                throughput_rows_per_second=0.0,  # Set by caller
                composition_metadata=pipeline._composition_metadata,
                intermediate_results=self._chunk_state[pipeline_id]["intermediate_results"]
            )
            
        except Exception as e:
            # Handle streaming pipeline errors
            return self._handle_streaming_error(
                e, chunk_results, total_rows_processed, pipeline_id
            )
    
    def _execute_standard_pipeline(
        self,
        pipeline: AnalysisPipelineBase,
        data_source: Union[StreamingDataSource, pd.DataFrame],
        pipeline_id: str
    ) -> StreamingPipelineResult:
        """
        Execute pipeline with standard (non-streaming) processing.
        
        Used for smaller datasets or pipelines that don't support streaming.
        """
        logger.info(f"Starting standard pipeline execution: {pipeline_id}")
        
        # Convert streaming source to DataFrame if needed
        if isinstance(data_source, StreamingDataSource):
            data_df = self._streaming_source_to_dataframe(data_source)
        else:
            data_df = data_source
        
        # Execute standard pipeline
        pipeline.fit(data_df)
        pipeline_result = pipeline.transform(data_df)
        
        # Convert to streaming result format
        return StreamingPipelineResult(
            success=pipeline_result.success,
            data=pipeline_result.data,
            metadata=pipeline_result.metadata,
            chunks_processed=1,  # Single "chunk"
            total_rows_processed=len(data_df),
            streaming_enabled=False,
            execution_time_seconds=pipeline_result.execution_time_seconds,
            memory_peak_mb=pipeline_result.memory_used_mb,
            throughput_rows_per_second=0.0,  # Set by caller
            composition_metadata=pipeline_result.composition_metadata,
            partial_results=False,
            error_info=pipeline_result.error
        )
    
    # ========================================================================
    # Chunk Processing Methods
    # ========================================================================
    
    def _get_chunk_iterator(
        self, 
        data_source: StreamingDataSource, 
        context: Dict[str, Any]
    ) -> Iterator[pd.DataFrame]:
        """Get chunk iterator with adaptive sizing."""
        chunk_size = self.streaming_config.chunk_size
        
        if self.streaming_config.adaptive_chunk_sizing:
            # Adapt chunk size based on memory status
            memory_status = self.streaming_executor._get_memory_status()
            chunk_size = memory_status.recommended_chunk_size
        
        return data_source.get_chunk_iterator(chunk_size)
    
    def _preprocess_chunk(
        self,
        chunk: pd.DataFrame,
        context: ChunkProcessingContext,
        pipeline: AnalysisPipelineBase
    ) -> pd.DataFrame:
        """Apply preprocessing to individual chunk."""
        
        # Use pipeline's preprocessing if it supports chunk processing
        if hasattr(pipeline, 'process_chunk'):
            result, _ = pipeline.process_chunk(chunk, context)
            return result
        
        # Default preprocessing (basic cleaning)
        return self._apply_default_preprocessing(chunk, context)
    
    def _analyze_chunk(
        self,
        chunk: pd.DataFrame,
        context: ChunkProcessingContext,
        pipeline: AnalysisPipelineBase
    ) -> Any:
        """Apply analysis to individual chunk."""
        
        # For stateless operations, analyze chunk independently
        if self.streaming_config.processing_mode == ChunkProcessingMode.STATELESS:
            return self._analyze_chunk_stateless(chunk, context, pipeline)
        
        # For stateful operations, maintain state across chunks
        elif self.streaming_config.processing_mode == ChunkProcessingMode.STATEFUL:
            return self._analyze_chunk_stateful(chunk, context, pipeline)
        
        # Default: sequential processing
        return self._analyze_chunk_sequential(chunk, context, pipeline)
    
    def _postprocess_chunk(
        self,
        result: Any,
        context: ChunkProcessingContext,
        pipeline: AnalysisPipelineBase
    ) -> Any:
        """Apply postprocessing to chunk result."""
        
        # Apply any necessary postprocessing transformations
        if hasattr(pipeline, 'postprocess_chunk_result'):
            return pipeline.postprocess_chunk_result(result, context)
        
        return result
    
    def _aggregate_streaming_results(
        self,
        chunk_results: List[Dict[str, Any]],
        pipeline: AnalysisPipelineBase,
        pipeline_id: str
    ) -> Tuple[Any, Dict[str, Any]]:
        """Aggregate results from all processed chunks."""
        
        # Use pipeline's aggregation method if available
        if hasattr(pipeline, 'aggregate_chunk_results'):
            results_only = [cr['result'] for cr in chunk_results]
            context = self._chunk_state[pipeline_id]
            return pipeline.aggregate_chunk_results(results_only, context)
        
        # Default aggregation strategy
        return self._default_result_aggregation(chunk_results, pipeline_id)
    
    # ========================================================================
    # Memory and Performance Management
    # ========================================================================
    
    def _handle_memory_pressure(
        self,
        chunk: pd.DataFrame,
        context: ChunkProcessingContext,
        pipeline_id: str
    ) -> pd.DataFrame:
        """Handle memory pressure by optimizing chunk processing."""
        
        # Reduce chunk size for next iteration
        if self.streaming_config.adaptive_chunk_sizing:
            new_chunk_size = max(len(chunk) // 2, 100)
            logger.info(
                f"Reducing chunk size due to memory pressure",
                old_size=len(chunk),
                new_size=new_chunk_size
            )
        
        # Clean up intermediate buffers if needed
        if self.streaming_config.intermediate_results_caching:
            self._cleanup_intermediate_buffers(pipeline_id)
        
        # Optimize chunk data types
        return self._optimize_chunk_datatypes(chunk)
    
    def _optimize_chunk_datatypes(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """Optimize chunk data types to reduce memory usage."""
        optimized = chunk.copy()
        
        # Downcast numeric types
        for col in optimized.select_dtypes(include=['int64']).columns:
            optimized[col] = pd.to_numeric(optimized[col], downcast='integer')
        
        for col in optimized.select_dtypes(include=['float64']).columns:
            optimized[col] = pd.to_numeric(optimized[col], downcast='float')
        
        # Convert object columns to category if beneficial
        for col in optimized.select_dtypes(include=['object']).columns:
            if optimized[col].nunique() / len(optimized) < 0.5:  # Low cardinality
                optimized[col] = optimized[col].astype('category')
        
        return optimized
    
    def monitor_performance(self) -> Dict[str, Any]:
        """Monitor streaming pipeline performance."""
        memory_status = self.streaming_executor._get_memory_status()
        
        performance_metrics = {
            "memory_status": memory_status.__dict__,
            "active_pipelines": len(self._active_pipelines),
            "intermediate_buffers": len(self._intermediate_buffers),
            "recent_execution_metrics": self._execution_metrics[-10:],  # Last 10
            "memory_snapshots": self._memory_snapshots[-10:]  # Last 10
        }
        
        # Store snapshot
        self._memory_snapshots.append(memory_status)
        
        # Trim history
        if len(self._memory_snapshots) > 100:
            self._memory_snapshots = self._memory_snapshots[-50:]
        
        return performance_metrics
    
    # ========================================================================
    # Utility and Helper Methods
    # ========================================================================
    
    def _generate_pipeline_id(self) -> str:
        """Generate unique pipeline execution ID."""
        import uuid
        return f"pipeline_{uuid.uuid4().hex[:8]}"
    
    def _initialize_streaming_context(
        self,
        pipeline: AnalysisPipelineBase,
        data_source: StreamingDataSource,
        pipeline_id: str
    ) -> Dict[str, Any]:
        """Initialize streaming execution context."""
        return {
            "pipeline_id": pipeline_id,
            "pipeline_class": pipeline.__class__.__name__,
            "data_source_type": type(data_source).__name__,
            "estimated_total_rows": data_source.estimate_total_rows(),
            "start_time": time.time(),
            "streaming_config": self.streaming_config
        }
    
    def _create_chunk_context(
        self,
        chunk_num: int,
        chunk: pd.DataFrame,
        streaming_context: Dict[str, Any],
        pipeline_id: str
    ) -> ChunkProcessingContext:
        """Create context for processing individual chunk."""
        memory_status = self.streaming_executor._get_memory_status()
        
        return ChunkProcessingContext(
            chunk_number=chunk_num,
            total_chunks_estimate=None,  # Unknown for streaming
            chunk_size=len(chunk),
            memory_status=memory_status,
            buffer_usage_mb=sum(
                buffer._current_memory_mb 
                for buffer in self._intermediate_buffers.values()
            ),
            current_stage=self._chunk_state[pipeline_id].get(
                "current_stage", StreamingStage.DATA_INGESTION
            ),
            pipeline_state=self._chunk_state[pipeline_id].copy(),
            intermediate_results=self._chunk_state[pipeline_id]["intermediate_results"],
            chunk_start_time=time.time(),
            cumulative_processing_time=streaming_context.get(
                "cumulative_time", 0.0
            )
        )
    
    def _update_pipeline_stage(self, pipeline_id: str, stage: StreamingStage):
        """Update current pipeline execution stage."""
        if pipeline_id in self._chunk_state:
            self._chunk_state[pipeline_id]["current_stage"] = stage
    
    def _update_chunk_state(
        self, 
        pipeline_id: str, 
        chunk_num: int, 
        result: Any
    ):
        """Update pipeline state after processing chunk."""
        if pipeline_id in self._chunk_state:
            state = self._chunk_state[pipeline_id]
            state["processed_chunks"] += 1
            
            # Store intermediate result if caching enabled
            if self.streaming_config.intermediate_results_caching:
                state["intermediate_results"][f"chunk_{chunk_num}"] = {
                    "result": result,
                    "timestamp": time.time()
                }
    
    def _cleanup_pipeline_state(self, pipeline_id: str):
        """Clean up pipeline state after execution."""
        if pipeline_id in self._active_pipelines:
            del self._active_pipelines[pipeline_id]
        
        if pipeline_id in self._chunk_state:
            del self._chunk_state[pipeline_id]
        
        # Clean up intermediate buffers
        self._cleanup_intermediate_buffers(pipeline_id)
    
    def _cleanup_intermediate_buffers(self, pipeline_id: str):
        """Clean up intermediate result buffers for pipeline."""
        buffers_to_remove = [
            buffer_id for buffer_id in self._intermediate_buffers.keys()
            if buffer_id.startswith(pipeline_id)
        ]
        
        for buffer_id in buffers_to_remove:
            self._intermediate_buffers[buffer_id].clear()
            del self._intermediate_buffers[buffer_id]
    
    def _handle_pipeline_execution_error(
        self,
        error: Exception,
        pipeline_id: str,
        execution_time: float
    ) -> StreamingPipelineResult:
        """Handle errors during pipeline execution."""
        
        # Get partial results if available
        partial_results = None
        chunks_processed = 0
        total_rows = 0
        
        if pipeline_id in self._chunk_state:
            state = self._chunk_state[pipeline_id]
            chunks_processed = state.get("processed_chunks", 0)
            total_rows = state.get("total_rows", 0)
            partial_results = state.get("intermediate_results")
        
        # Log error
        logger.error(
            f"Pipeline execution failed: {pipeline_id}",
            error=str(error),
            chunks_processed=chunks_processed,
            execution_time=execution_time
        )
        
        # Return error result with partial data
        return StreamingPipelineResult(
            success=False,
            data=None,
            metadata={"error": str(error), "pipeline_id": pipeline_id},
            chunks_processed=chunks_processed,
            total_rows_processed=total_rows,
            streaming_enabled=True,
            execution_time_seconds=execution_time,
            memory_peak_mb=0.0,
            throughput_rows_per_second=0.0,
            composition_metadata=None,
            intermediate_results=partial_results or {},
            partial_results=partial_results is not None,
            error_info={
                "exception_type": type(error).__name__,
                "message": str(error),
                "pipeline_id": pipeline_id,
                "execution_time": execution_time
            }
        )
    
    # ========================================================================
    # Default Processing Implementations
    # ========================================================================
    
    def _apply_default_preprocessing(self, chunk: pd.DataFrame, context: ChunkProcessingContext) -> pd.DataFrame:
        """Apply default preprocessing to chunk."""
        # Basic data cleaning
        processed = chunk.copy()
        
        # Handle missing values (basic strategy)
        for col in processed.select_dtypes(include=['number']).columns:
            processed[col] = processed[col].fillna(processed[col].median())
        
        for col in processed.select_dtypes(include=['object']).columns:
            processed[col] = processed[col].fillna('unknown')
        
        return processed
    
    def _analyze_chunk_stateless(self, chunk: pd.DataFrame, context: ChunkProcessingContext, pipeline: AnalysisPipelineBase) -> Any:
        """Analyze chunk independently (stateless processing)."""
        # Each chunk processed independently
        if hasattr(pipeline, '_execute_analysis_step'):
            # Use pipeline's analysis method if available
            steps = pipeline._configure_analysis_pipeline()
            results = []
            
            for step in steps:
                result, metadata = pipeline._execute_analysis_step(step, chunk, context.pipeline_state)
                results.append(result)
            
            return results
        
        # Fallback: return chunk as-is
        return chunk
    
    def _analyze_chunk_stateful(self, chunk: pd.DataFrame, context: ChunkProcessingContext, pipeline: AnalysisPipelineBase) -> Any:
        """Analyze chunk with state maintained across chunks."""
        # Maintain state across chunks for operations like running statistics
        pipeline_state = context.pipeline_state
        
        # Update running statistics or other stateful computations
        if "running_stats" not in pipeline_state:
            pipeline_state["running_stats"] = {
                "count": 0,
                "sum": 0,
                "mean": 0,
                "variance": 0
            }
        
        # Update statistics with new chunk
        chunk_stats = self._calculate_chunk_statistics(chunk)
        pipeline_state["running_stats"] = self._update_running_statistics(
            pipeline_state["running_stats"], chunk_stats
        )
        
        return chunk_stats
    
    def _analyze_chunk_sequential(self, chunk: pd.DataFrame, context: ChunkProcessingContext, pipeline: AnalysisPipelineBase) -> Any:
        """Analyze chunk sequentially (default processing)."""
        # Process chunk using standard pipeline methods
        return self._analyze_chunk_stateless(chunk, context, pipeline)
    
    def _calculate_chunk_statistics(self, chunk: pd.DataFrame) -> Dict[str, Any]:
        """Calculate basic statistics for a chunk."""
        numeric_cols = chunk.select_dtypes(include=['number']).columns
        
        if len(numeric_cols) == 0:
            return {"row_count": len(chunk)}
        
        stats = {
            "row_count": len(chunk),
            "numeric_summary": chunk[numeric_cols].describe().to_dict()
        }
        
        return stats
    
    def _update_running_statistics(self, running_stats: Dict[str, Any], chunk_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Update running statistics with new chunk statistics."""
        # Simple running count update
        running_stats["count"] += chunk_stats["row_count"]
        
        # For more complex running statistics, implement incremental algorithms
        # This is a placeholder for demonstration
        
        return running_stats
    
    def _default_result_aggregation(self, chunk_results: List[Dict[str, Any]], pipeline_id: str) -> Tuple[Any, Dict[str, Any]]:
        """Default strategy for aggregating chunk results."""
        
        if not chunk_results:
            return None, {"error": "No chunk results to aggregate"}
        
        # Simple concatenation for DataFrame results
        try:
            results = [cr['result'] for cr in chunk_results if cr['result'] is not None]
            
            if not results:
                return None, {"error": "No valid results to aggregate"}
            
            # If all results are DataFrames, concatenate them
            if all(isinstance(result, pd.DataFrame) for result in results):
                aggregated_df = pd.concat(results, ignore_index=True)
                metadata = {
                    "aggregation_method": "concatenation",
                    "chunks_aggregated": len(results),
                    "total_rows": len(aggregated_df)
                }
                return aggregated_df, metadata
            
            # For other result types, return as list
            metadata = {
                "aggregation_method": "list",
                "chunks_aggregated": len(results),
                "result_types": [type(r).__name__ for r in results]
            }
            return results, metadata
            
        except Exception as e:
            logger.error(f"Error aggregating results: {e}")
            return None, {"error": f"Aggregation failed: {str(e)}"}
    
    def _dataframe_to_streaming_source(self, df: pd.DataFrame) -> StreamingDataSource:
        """Convert DataFrame to streaming source for consistent processing."""
        # This would typically create a temporary file or in-memory source
        # For now, return a mock streaming source
        # In practice, this might create a CSV file temporarily or use a memory-based source
        pass  # Placeholder - implementation depends on specific requirements
    
    def _streaming_source_to_dataframe(self, source: StreamingDataSource) -> pd.DataFrame:
        """Convert streaming source to DataFrame for standard processing."""
        chunks = list(source.get_chunk_iterator(10000))  # Large chunks
        if chunks:
            return pd.concat(chunks, ignore_index=True)
        return pd.DataFrame()
    
    def _handle_streaming_error(
        self,
        error: Exception,
        chunk_results: List[Dict[str, Any]],
        total_rows_processed: int,
        pipeline_id: str
    ) -> StreamingPipelineResult:
        """Handle errors during streaming execution."""
        
        # Get partial results
        partial_data = None
        if chunk_results:
            try:
                partial_data = self._default_result_aggregation(
                    chunk_results, pipeline_id
                )[0]
            except Exception as agg_error:
                logger.error(f"Failed to aggregate partial results: {agg_error}")
        
        return StreamingPipelineResult(
            success=False,
            data=partial_data,
            metadata={"error": str(error)},
            chunks_processed=len(chunk_results),
            total_rows_processed=total_rows_processed,
            streaming_enabled=True,
            execution_time_seconds=0.0,  # Set by caller
            memory_peak_mb=0.0,
            throughput_rows_per_second=0.0,
            composition_metadata=None,
            intermediate_results={},
            partial_results=partial_data is not None,
            error_info={
                "exception_type": type(error).__name__,
                "message": str(error),
                "chunks_before_error": len(chunk_results)
            }
        )


# ============================================================================
# Integration Bridge Classes
# ============================================================================

class StreamingAwarePipeline(AnalysisPipelineBase, ABC):
    """
    Extended AnalysisPipelineBase with streaming support.
    
    Provides the bridge between existing pipeline architecture and
    new streaming integration capabilities.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Streaming-specific configuration
        self.streaming_pipeline_config = StreamingPipelineConfig()
        self._streaming_executor: Optional[StreamingPipelineExecutor] = None
    
    def supports_streaming(self) -> bool:
        """Check if this pipeline supports streaming execution."""
        return True  # All streaming-aware pipelines support streaming
    
    def get_streaming_requirements(self) -> Dict[str, Any]:
        """Get streaming requirements and constraints."""
        return {
            "memory_limit_mb": self.streaming_pipeline_config.memory_limit_mb,
            "processing_mode": self.streaming_pipeline_config.processing_mode.value,
            "supports_parallel": self.streaming_pipeline_config.parallel_stages,
            "requires_stateful_processing": self._requires_stateful_processing(),
            "intermediate_results_needed": self.streaming_pipeline_config.intermediate_results_caching
        }
    
    @abstractmethod
    def _requires_stateful_processing(self) -> bool:
        """Check if pipeline requires stateful processing across chunks."""
        pass
    
    def process_chunk(
        self, 
        chunk: pd.DataFrame, 
        context: ChunkProcessingContext
    ) -> Tuple[Any, Dict[str, Any]]:
        """Process a single chunk of data."""
        # Default implementation - subclasses should override for optimized chunk processing
        return self._execute_standard_analysis(chunk)
    
    def aggregate_chunk_results(
        self, 
        chunk_results: List[Tuple[Any, Dict[str, Any]]], 
        context: Dict[str, Any]
    ) -> Tuple[Any, Dict[str, Any]]:
        """Aggregate results from multiple chunks."""
        # Default implementation - concatenate DataFrames
        results = [result for result, _ in chunk_results]
        
        if all(isinstance(result, pd.DataFrame) for result in results):
            aggregated = pd.concat(results, ignore_index=True)
            metadata = {
                "aggregation_method": "concatenation",
                "chunks_aggregated": len(results)
            }
            return aggregated, metadata
        
        return results, {"aggregation_method": "list"}
    
    def handle_streaming_error(
        self, 
        error: Exception, 
        context: ChunkProcessingContext
    ) -> Dict[str, Any]:
        """Handle errors during streaming execution."""
        return {
            "error_type": type(error).__name__,
            "message": str(error),
            "chunk_number": context.chunk_number,
            "recovery_possible": self._can_recover_from_error(error),
            "suggested_actions": self._get_recovery_suggestions(error, context)
        }
    
    def _can_recover_from_error(self, error: Exception) -> bool:
        """Check if pipeline can recover from the given error."""
        # Memory errors might be recoverable by reducing chunk size
        if isinstance(error, MemoryError):
            return True
        
        # Data quality errors might be recoverable by skipping chunk
        if "data" in str(error).lower():
            return True
        
        return False
    
    def _get_recovery_suggestions(self, error: Exception, context: ChunkProcessingContext) -> List[str]:
        """Get suggestions for recovering from error."""
        suggestions = []
        
        if isinstance(error, MemoryError):
            suggestions.append("Reduce chunk size")
            suggestions.append("Enable memory-efficient mode")
        
        if context.memory_status.is_low_memory:
            suggestions.append("Clean up intermediate buffers")
            suggestions.append("Force garbage collection")
        
        return suggestions


# ============================================================================
# Usage Example and Integration Points
# ============================================================================

def create_streaming_pipeline_example() -> StreamingPipelineExecutor:
    """
    Example: Create streaming-enabled pipeline executor for large dataset analysis.
    
    Demonstrates the integration between proven LocalData MCP streaming
    architecture and new pipeline composition capabilities.
    """
    
    # Configure streaming for large dataset processing
    streaming_config = StreamingPipelineConfig(
        memory_limit_mb=2048,  # 2GB limit
        chunk_size=5000,       # 5k rows per chunk
        processing_mode=ChunkProcessingMode.STATEFUL,  # Maintain state
        intermediate_results_caching=True,
        early_results_enabled=True,
        adaptive_chunk_sizing=True
    )
    
    # Create executor with streaming support
    executor = StreamingPipelineExecutor(streaming_config=streaming_config)
    
    logger.info(
        "Created streaming pipeline executor",
        memory_limit_mb=streaming_config.memory_limit_mb,
        chunk_size=streaming_config.chunk_size,
        processing_mode=streaming_config.processing_mode.value
    )
    
    return executor


if __name__ == "__main__":
    # Example usage demonstration
    executor = create_streaming_pipeline_example()
    
    print("Streaming Pipeline Integration Architecture Created")
    print(f"Memory Limit: {executor.streaming_config.memory_limit_mb}MB")
    print(f"Processing Mode: {executor.streaming_config.processing_mode.value}")
    print(f"Adaptive Chunking: {executor.streaming_config.adaptive_chunk_sizing}")
