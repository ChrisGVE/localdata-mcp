"""
Base classes and interfaces for the Core Pipeline Framework.

This module defines the foundational abstractions that all analysis pipelines
inherit from, implementing the Five First Principles Architecture:

1. Intention-Driven Interface - Natural language analytical goals
2. Context-Aware Composition - Enriched metadata for tool chaining
3. Progressive Disclosure - Simple defaults, powerful customization
4. Streaming-First - Memory-bounded processing for large datasets
5. Modular Domain Integration - Extensible cross-domain architecture
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from ..streaming_executor import MemoryStatus, StreamingDataSource
from ..logging_manager import get_logger

logger = get_logger(__name__)


class PipelineState(Enum):
    """Pipeline execution states."""
    INITIALIZED = "initialized"
    CONFIGURED = "configured" 
    FITTED = "fitted"
    EXECUTING = "executing"
    COMPLETED = "completed"
    ERROR = "error"


class DataSourceType(Enum):
    """Supported data source types."""
    SQL_QUERY = "sql_query"
    CSV_FILE = "csv_file"
    EXCEL_FILE = "excel_file"
    JSON_FILE = "json_file"  
    PARQUET_FILE = "parquet_file"
    DATABASE_CONNECTION = "database_connection"
    DATAFRAME = "dataframe"


class PreprocessingIntent(Enum):
    """Preprocessing complexity levels."""
    MINIMAL = "minimal"  # Basic cleaning and type inference
    AUTO = "auto"  # Automatic comprehensive preprocessing
    COMPREHENSIVE = "comprehensive"  # Full preprocessing with advanced features
    CUSTOM = "custom"  # User-defined preprocessing pipeline


class ErrorClassification(Enum):
    """Pipeline error classifications for recovery strategies."""
    MEMORY_OVERFLOW = "memory_overflow"
    COMPUTATION_TIMEOUT = "computation_timeout" 
    DATA_QUALITY_FAILURE = "data_quality_failure"
    STREAMING_INTERRUPTION = "streaming_interruption"
    CONFIGURATION_ERROR = "configuration_error"
    EXTERNAL_DEPENDENCY_ERROR = "external_dependency_error"


@dataclass
class CompositionMetadata:
    """Metadata for downstream tool composition and workflow chaining."""
    
    # Core composition information
    domain: str  # "time_series", "ml", "statistical", etc.
    analysis_type: str  # "forecast", "classification", "correlation", etc.
    result_type: str  # "predictions", "model", "statistics", etc.
    
    # Downstream compatibility
    compatible_tools: List[str] = field(default_factory=list)
    suggested_compositions: List[Dict[str, Any]] = field(default_factory=list)
    data_artifacts: Dict[str, Any] = field(default_factory=dict)
    
    # Pipeline chaining context
    input_schema: Dict[str, Any] = field(default_factory=dict)
    output_schema: Dict[str, Any] = field(default_factory=dict)
    transformation_summary: Dict[str, Any] = field(default_factory=dict)
    
    # Quality and confidence
    confidence_level: float = 0.0
    quality_score: float = 0.0
    limitations: List[str] = field(default_factory=list)
    
    # Next step recommendations
    recommended_next_steps: List[Dict[str, Any]] = field(default_factory=list)
    alternative_approaches: List[Dict[str, Any]] = field(default_factory=list)


@dataclass  
class StreamingConfig:
    """Configuration for streaming-compatible pipeline execution."""
    
    enabled: bool = False
    threshold_mb: int = 100  # Enable streaming above this data size
    chunk_size_adaptive: bool = True
    initial_chunk_size: Optional[int] = None
    memory_limit_mb: int = 1000
    buffer_timeout_seconds: int = 300
    
    # Performance tuning
    parallel_processing: bool = False
    memory_efficient_mode: bool = True
    early_termination_enabled: bool = True


@dataclass
class PipelineResult:
    """Standardized pipeline execution result."""
    
    # Core result data
    success: bool
    data: Optional[Union[pd.DataFrame, Any]]
    metadata: Dict[str, Any]
    
    # Execution information
    execution_time_seconds: float
    memory_used_mb: float
    pipeline_stage: str
    
    # Composition context
    composition_metadata: Optional[CompositionMetadata] = None
    
    # Error information (if success=False)
    error: Optional[Dict[str, Any]] = None
    partial_results: Optional[Any] = None
    recovery_options: List[Dict[str, Any]] = field(default_factory=list)


class PipelineError(Exception):
    """Base exception for pipeline errors with recovery context."""
    
    def __init__(self, 
                 message: str,
                 classification: ErrorClassification,
                 pipeline_stage: str,
                 context: Optional[Dict[str, Any]] = None,
                 partial_results: Optional[Any] = None,
                 recovery_suggestions: Optional[List[str]] = None):
        super().__init__(message)
        self.classification = classification
        self.pipeline_stage = pipeline_stage
        self.context = context or {}
        self.partial_results = partial_results
        self.recovery_suggestions = recovery_suggestions or []
        self.timestamp = time.time()


class AnalysisPipelineBase(BaseEstimator, TransformerMixin, ABC):
    """
    Abstract base class for all analysis pipelines.
    
    Implements the Five First Principles:
    1. Intention-Driven Interface - Natural language configuration
    2. Context-Aware Composition - Enriched metadata for tool chaining  
    3. Progressive Disclosure - Simple defaults, powerful options
    4. Streaming-First - Memory-bounded processing
    5. Modular Domain Integration - Extensible architecture
    """
    
    def __init__(self, 
                 analytical_intention: str,
                 streaming_config: Optional[StreamingConfig] = None,
                 progressive_complexity: str = "auto",
                 composition_aware: bool = True,
                 custom_parameters: Optional[Dict[str, Any]] = None):
        """
        Initialize analysis pipeline with intention-driven configuration.
        
        Args:
            analytical_intention: Natural language description of analysis goal
            streaming_config: Configuration for streaming execution
            progressive_complexity: "minimal", "auto", "comprehensive", "custom"
            composition_aware: Include metadata for downstream tool composition
            custom_parameters: Domain-specific custom configuration
        """
        self.analytical_intention = analytical_intention
        self.streaming_config = streaming_config or StreamingConfig()
        self.progressive_complexity = progressive_complexity
        self.composition_aware = composition_aware
        self.custom_parameters = custom_parameters or {}
        
        # Pipeline state management
        self._state = PipelineState.INITIALIZED
        self._execution_context: Dict[str, Any] = {}
        self._composition_metadata: Optional[CompositionMetadata] = None
        self._error_handler: Optional[Any] = None  # Imported when needed
        
        # Performance tracking
        self._start_time: Optional[float] = None
        self._memory_profile: List[MemoryStatus] = []
        
        logger.info(f"Initialized {self.__class__.__name__}",
                   intention=analytical_intention,
                   complexity=progressive_complexity,
                   streaming_enabled=streaming_config.enabled if streaming_config else False)
    
    @property
    def state(self) -> PipelineState:
        """Get current pipeline state."""
        return self._state
    
    @property
    def domain(self) -> str:
        """Get the domain this pipeline belongs to."""
        return self.__class__.__module__.split('.')[-1]  # Extract from module path
    
    @abstractmethod
    def get_analysis_type(self) -> str:
        """Get the specific analysis type this pipeline performs."""
        pass
    
    @abstractmethod
    def _configure_analysis_pipeline(self) -> List[Callable]:
        """
        Configure analysis steps based on intention and complexity level.
        
        Returns:
            List of callable analysis functions to execute in sequence
        """
        pass
    
    @abstractmethod  
    def _execute_analysis_step(self, 
                              step: Callable, 
                              data: pd.DataFrame, 
                              context: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        """
        Execute individual analysis step with error handling and metadata.
        
        Args:
            step: Analysis function to execute
            data: Input data for the step
            context: Execution context with metadata
            
        Returns:
            Tuple of (step_result, step_metadata)
        """
        pass
    
    def fit(self, X: pd.DataFrame, y=None) -> 'AnalysisPipelineBase':
        """
        Fit the analysis pipeline to the data.
        
        Analyzes data characteristics and configures streaming/processing options.
        
        Args:
            X: Input data to fit the pipeline to
            y: Optional target variable (for supervised learning)
            
        Returns:
            Self for method chaining
        """
        self._state = PipelineState.CONFIGURED
        self._start_time = time.time()
        
        try:
            # Profile data characteristics
            data_profile = self._profile_data_characteristics(X)
            
            # Configure streaming if needed
            if self._should_enable_streaming(X, data_profile):
                self._configure_streaming(X, data_profile)
            
            # Configure analysis pipeline based on intention and data
            self._analysis_pipeline = self._configure_analysis_pipeline()
            
            # Build composition metadata if enabled
            if self.composition_aware:
                self._composition_metadata = self._build_composition_context(X, data_profile)
            
            # Prepare execution context
            self._execution_context = {
                "data_profile": data_profile,
                "fit_time": time.time(),
                "pipeline_steps": [step.__name__ for step in self._analysis_pipeline],
                "streaming_enabled": self.streaming_config.enabled
            }
            
            self._state = PipelineState.FITTED
            logger.info(f"Pipeline fitted successfully",
                       steps=len(self._analysis_pipeline),
                       streaming=self.streaming_config.enabled)
            
            return self
            
        except Exception as e:
            self._state = PipelineState.ERROR
            self._handle_pipeline_error(e, "fit", context={"data_shape": X.shape})
            raise
    
    def transform(self, X: pd.DataFrame) -> PipelineResult:
        """
        Execute analysis pipeline with streaming support and enriched metadata.
        
        Args:
            X: Input data to transform/analyze
            
        Returns:
            PipelineResult with analysis results and composition metadata
        """
        if self._state != PipelineState.FITTED:
            raise ValueError(f"Pipeline not fitted. Current state: {self._state.value}")
        
        self._state = PipelineState.EXECUTING
        start_time = time.time()
        
        try:
            # Execute with streaming if configured
            if self.streaming_config.enabled:
                result_data, metadata = self._execute_streaming_analysis(X)
            else:
                result_data, metadata = self._execute_standard_analysis(X)
            
            # Calculate execution metrics
            execution_time = time.time() - start_time
            memory_used = self._calculate_memory_usage()
            
            # Build pipeline result
            pipeline_result = PipelineResult(
                success=True,
                data=result_data,
                metadata=metadata,
                execution_time_seconds=execution_time,
                memory_used_mb=memory_used,
                pipeline_stage="transform",
                composition_metadata=self._composition_metadata
            )
            
            self._state = PipelineState.COMPLETED
            logger.info(f"Pipeline executed successfully",
                       execution_time=execution_time,
                       memory_used_mb=memory_used)
            
            return pipeline_result
            
        except Exception as e:
            self._state = PipelineState.ERROR
            execution_time = time.time() - start_time
            
            # Try to get partial results for graceful degradation
            partial_results = self._get_partial_results()
            
            error_result = self._handle_pipeline_error(
                e, "transform", 
                context={
                    "data_shape": X.shape,
                    "execution_time": execution_time
                },
                partial_results=partial_results
            )
            
            return PipelineResult(
                success=False,
                data=partial_results,
                metadata={"error": True, "execution_time_seconds": execution_time},
                execution_time_seconds=execution_time,
                memory_used_mb=self._calculate_memory_usage(),
                pipeline_stage="transform",
                error=error_result,
                partial_results=partial_results,
                recovery_options=error_result.get("recovery_options", [])
            )
    
    def _profile_data_characteristics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Profile data characteristics for pipeline configuration."""
        profile = {
            "shape": data.shape,
            "memory_usage_mb": data.memory_usage(deep=True).sum() / (1024 * 1024),
            "column_types": dict(data.dtypes),
            "null_percentages": (data.isnull().sum() / len(data) * 100).to_dict(),
            "numeric_columns": data.select_dtypes(include=['number']).columns.tolist(),
            "categorical_columns": data.select_dtypes(include=['object', 'category']).columns.tolist(),
            "datetime_columns": data.select_dtypes(include=['datetime64']).columns.tolist()
        }
        
        # Statistical summary for numeric columns
        if profile["numeric_columns"]:
            profile["numeric_summary"] = data[profile["numeric_columns"]].describe().to_dict()
        
        return profile
    
    def _should_enable_streaming(self, data: pd.DataFrame, profile: Dict[str, Any]) -> bool:
        """Determine if streaming should be enabled based on data characteristics."""
        data_size_mb = profile["memory_usage_mb"]
        return (data_size_mb > self.streaming_config.threshold_mb or
                len(data) > 100000 or  # Large row count
                self.streaming_config.enabled)  # Explicitly enabled
    
    def _configure_streaming(self, data: pd.DataFrame, profile: Dict[str, Any]):
        """Configure streaming parameters based on data characteristics."""
        if not self.streaming_config.enabled:
            self.streaming_config.enabled = True
        
        # Adaptive chunk sizing based on data characteristics
        if self.streaming_config.chunk_size_adaptive:
            estimated_row_size = profile["memory_usage_mb"] / len(data) * 1024 * 1024  # bytes per row
            target_chunk_memory_mb = min(self.streaming_config.memory_limit_mb // 4, 250)
            adaptive_chunk_size = int(target_chunk_memory_mb * 1024 * 1024 / estimated_row_size)
            self.streaming_config.initial_chunk_size = max(min(adaptive_chunk_size, 10000), 100)
        
        logger.info("Streaming configured",
                   chunk_size=self.streaming_config.initial_chunk_size,
                   memory_limit_mb=self.streaming_config.memory_limit_mb)
    
    def _build_composition_context(self, data: pd.DataFrame, profile: Dict[str, Any]) -> CompositionMetadata:
        """Build metadata for downstream tool composition."""
        return CompositionMetadata(
            domain=self.domain,
            analysis_type=self.get_analysis_type(),
            result_type="analysis_results",  # Default, subclasses should override
            input_schema={
                "columns": list(data.columns),
                "dtypes": dict(data.dtypes),
                "shape": data.shape
            },
            transformation_summary={
                "intention": self.analytical_intention,
                "complexity": self.progressive_complexity,
                "streaming_enabled": self.streaming_config.enabled
            }
        )
    
    @abstractmethod
    def _execute_streaming_analysis(self, data: pd.DataFrame) -> Tuple[Any, Dict[str, Any]]:
        """Execute analysis with streaming support for large datasets."""
        pass
    
    @abstractmethod
    def _execute_standard_analysis(self, data: pd.DataFrame) -> Tuple[Any, Dict[str, Any]]:
        """Execute analysis on full dataset in memory."""
        pass
    
    def _get_partial_results(self) -> Optional[Any]:
        """Get partial results if available during error recovery."""
        return self._execution_context.get("partial_results")
    
    def _calculate_memory_usage(self) -> float:
        """Calculate current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except ImportError:
            return 0.0
    
    def _handle_pipeline_error(self, 
                              error: Exception, 
                              stage: str,
                              context: Optional[Dict[str, Any]] = None,
                              partial_results: Optional[Any] = None) -> Dict[str, Any]:
        """Handle pipeline errors with recovery strategies."""
        # Import error handler when needed to avoid circular imports
        if self._error_handler is None:
            from .error_handling import PipelineErrorHandler
            self._error_handler = PipelineErrorHandler()
        
        pipeline_context = {
            "current_stage": stage,
            "analytical_intention": self.analytical_intention,
            "progressive_complexity": self.progressive_complexity,
            "streaming_enabled": self.streaming_config.enabled,
            **(context or {})
        }
        
        return self._error_handler.handle_pipeline_error(
            error, pipeline_context, partial_results
        )
    
    # Utility methods for subclasses
    def get_execution_context(self) -> Dict[str, Any]:
        """Get current execution context."""
        return self._execution_context.copy()
    
    def update_composition_metadata(self, updates: Dict[str, Any]):
        """Update composition metadata with additional information."""
        if self._composition_metadata:
            for key, value in updates.items():
                if hasattr(self._composition_metadata, key):
                    setattr(self._composition_metadata, key, value)
    
    def log_performance_metric(self, metric_name: str, value: Any):
        """Log a performance metric for monitoring."""
        logger.info(f"Performance metric: {metric_name}", 
                   value=value, 
                   pipeline=self.__class__.__name__)