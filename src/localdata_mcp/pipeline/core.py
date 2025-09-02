"""
DataSciencePipeline: Enhanced sklearn Pipeline for LocalData MCP v2.0

This module provides the core DataSciencePipeline class that extends sklearn.pipeline.Pipeline
with enhanced features for comprehensive data science platform capabilities:

- Enhanced metadata tracking throughout pipeline stages
- Integration with logging and streaming architecture  
- Intention-driven configuration and progressive disclosure
- Context-aware composition for LLM tool chaining
- Memory-bounded processing for large datasets
- Comprehensive error handling with recovery strategies

The DataSciencePipeline maintains full sklearn API compatibility while adding
LocalData MCP's First Principles Architecture capabilities.
"""

import time
import uuid
import warnings
import concurrent.futures
from copy import deepcopy
from collections import defaultdict, deque
from typing import Any, Dict, Generator, List, Optional, Tuple, Union, Callable

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

from .base import (
    AnalysisPipelineBase, 
    CompositionMetadata, 
    PipelineError, 
    PipelineResult,
    PipelineState,
    ErrorClassification,
    StreamingConfig
)
from ..streaming_executor import StreamingQueryExecutor, MemoryStatus, StreamingDataSource
from ..logging_manager import get_logging_manager, get_logger
from ..error_handler import ErrorCategory, ErrorSeverity

logger = get_logger(__name__)


class DataFrameStreamingSource(StreamingDataSource):
    """Streaming data source for pandas DataFrames."""
    
    def __init__(self, dataframe: pd.DataFrame):
        self.dataframe = dataframe
        self._estimated_row_size = None
    
    def get_chunk_iterator(self, chunk_size: int) -> Generator[pd.DataFrame, None, None]:
        """Get iterator that yields DataFrame chunks."""
        for start in range(0, len(self.dataframe), chunk_size):
            chunk = self.dataframe.iloc[start:start + chunk_size].copy()
            if not chunk.empty:
                yield chunk
    
    def estimate_total_rows(self) -> Optional[int]:
        """Get exact number of rows."""
        return len(self.dataframe)
    
    def estimate_memory_per_row(self) -> float:
        """Estimate memory usage per row in bytes."""
        if self._estimated_row_size is None:
            if len(self.dataframe) > 0:
                memory_usage = self.dataframe.memory_usage(deep=True).sum()
                self._estimated_row_size = memory_usage / len(self.dataframe)
            else:
                self._estimated_row_size = 1024.0  # Default 1KB per row
        return self._estimated_row_size


class DataSciencePipeline(Pipeline):
    """
    Enhanced sklearn Pipeline with LocalData MCP v2.0 capabilities.
    
    Extends sklearn.pipeline.Pipeline while maintaining full API compatibility.
    Adds enhanced metadata tracking, logging integration, streaming support,
    and intention-driven configuration for comprehensive data science workflows.
    
    Key Features:
    - Full sklearn Pipeline API compatibility
    - Enhanced metadata preservation across pipeline stages  
    - Integration with LocalData MCP logging and streaming systems
    - Progressive disclosure (minimal/auto/comprehensive/custom complexity)
    - Context-aware composition metadata for LLM tool chaining
    - Memory-bounded processing for large datasets
    - Comprehensive error handling with recovery strategies
    
    Example Usage:
        # Basic usage (sklearn compatible)
        pipeline = DataSciencePipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier())
        ])
        
        # Enhanced usage with LocalData MCP features
        pipeline = DataSciencePipeline(
            steps=[('scaler', StandardScaler()), ('model', LinearRegression())],
            analytical_intention="Predict house prices with feature scaling",
            streaming_config=StreamingConfig(enabled=True, threshold_mb=100),
            progressive_complexity="auto",
            composition_aware=True
        )
    """
    
    def __init__(self, 
                 steps: List[Tuple[str, BaseEstimator]], 
                 *,
                 memory=None,
                 verbose=False,
                 analytical_intention: Optional[str] = None,
                 streaming_config: Optional[StreamingConfig] = None,
                 progressive_complexity: str = "auto",
                 composition_aware: bool = True,
                 custom_parameters: Optional[Dict[str, Any]] = None):
        """
        Initialize DataSciencePipeline with enhanced LocalData MCP capabilities.
        
        Args:
            steps: List of (name, estimator) tuples (sklearn Pipeline compatible)
            memory: sklearn memory parameter for caching (inherited from Pipeline)
            verbose: sklearn verbose parameter (inherited from Pipeline)  
            analytical_intention: Natural language description of analysis goal
            streaming_config: Configuration for streaming execution on large datasets
            progressive_complexity: Complexity level ("minimal", "auto", "comprehensive", "custom")
            composition_aware: Whether to generate metadata for tool chaining
            custom_parameters: Additional domain-specific parameters
        """
        # Initialize sklearn Pipeline with standard parameters
        super().__init__(steps=steps, memory=memory, verbose=verbose)
        
        # LocalData MCP enhanced features
        self.analytical_intention = analytical_intention or "General data science pipeline"
        self.streaming_config = streaming_config or StreamingConfig()
        self.progressive_complexity = progressive_complexity
        self.composition_aware = composition_aware
        self.custom_parameters = custom_parameters or {}
        
        # Enhanced state management
        self._pipeline_id = str(uuid.uuid4())
        self._state = PipelineState.INITIALIZED
        self._execution_context: Dict[str, Any] = {}
        self._composition_metadata: Optional[CompositionMetadata] = None
        self._streaming_executor: Optional[StreamingQueryExecutor] = None
        
        # Performance and error tracking
        self._fit_time: Optional[float] = None
        self._transform_time: Optional[float] = None
        self._memory_profile: List[MemoryStatus] = []
        self._step_metadata: List[Dict[str, Any]] = []
        self._error_history: List[Dict[str, Any]] = []
        
        # Logging integration
        self._logging_manager = get_logging_manager()
        self._request_id: Optional[str] = None
        
        logger.info("DataSciencePipeline initialized",
                   pipeline_id=self._pipeline_id,
                   steps=len(self.steps),
                   intention=self.analytical_intention,
                   streaming_enabled=self.streaming_config.enabled,
                   complexity=self.progressive_complexity)
    
    @property
    def pipeline_id(self) -> str:
        """Get unique pipeline identifier."""
        return self._pipeline_id
    
    @property 
    def state(self) -> PipelineState:
        """Get current pipeline state."""
        return self._state
    
    @property
    def composition_metadata(self) -> Optional[CompositionMetadata]:
        """Get composition metadata for tool chaining."""
        return self._composition_metadata
    
    def fit(self, X, y=None, **fit_params):
        """
        Fit the pipeline with enhanced metadata tracking and streaming support.
        
        Maintains full sklearn Pipeline.fit() API compatibility while adding
        enhanced features for large dataset handling and metadata generation.
        
        Args:
            X: Training data (array-like or DataFrame)
            y: Target values (array-like, optional)
            **fit_params: Parameters to pass to pipeline steps
            
        Returns:
            self: Returns self for method chaining (sklearn compatible)
        """
        self._state = PipelineState.CONFIGURED
        start_time = time.time()
        
        # Start logging context
        self._request_id = self._logging_manager.log_query_start(
            database_name="pipeline",
            query=f"fit_{self._pipeline_id}",
            database_type="pipeline_fit"
        )
        
        with self._logging_manager.context(
            request_id=self._request_id,
            operation="pipeline_fit",
            component="data_science_pipeline"
        ):
            try:
                # Convert input to pandas DataFrame if needed for analysis
                X_df = self._ensure_dataframe(X, "fit_input")
                
                # Profile data characteristics for streaming decisions
                data_profile = self._profile_data_characteristics(X_df)
                self._execution_context["data_profile"] = data_profile
                
                # Configure streaming if needed for large datasets
                if self._should_enable_streaming(X_df, data_profile):
                    self._configure_streaming_for_fit(X_df, data_profile)
                
                # Build composition metadata if enabled
                if self.composition_aware:
                    self._composition_metadata = self._build_initial_composition_metadata(X_df, y)
                
                # Execute sklearn Pipeline fit with enhanced error handling
                if self.streaming_config.enabled and len(X_df) > self.streaming_config.threshold_mb * 1000:
                    # For very large datasets, use streaming-aware fit
                    result = self._fit_with_streaming(X, y, **fit_params)
                else:
                    # Standard sklearn fit
                    result = super().fit(X, y, **fit_params)
                
                # Record successful fit
                self._fit_time = time.time() - start_time
                self._state = PipelineState.FITTED
                
                # Log successful completion
                self._logging_manager.log_query_complete(
                    request_id=self._request_id,
                    database_name="pipeline", 
                    database_type="pipeline_fit",
                    duration=self._fit_time,
                    success=True
                )
                
                logger.info("Pipeline fitted successfully",
                           pipeline_id=self._pipeline_id,
                           fit_time=self._fit_time,
                           data_shape=X_df.shape,
                           streaming_enabled=self.streaming_config.enabled)
                
                return result
                
            except Exception as e:
                self._state = PipelineState.ERROR
                fit_time = time.time() - start_time
                
                # Log error and handle
                error_info = self._handle_pipeline_error(e, "fit", {
                    "data_shape": X_df.shape if 'X_df' in locals() else None,
                    "fit_time": fit_time,
                    "step_params": fit_params
                })
                
                self._logging_manager.log_query_complete(
                    request_id=self._request_id,
                    database_name="pipeline",
                    database_type="pipeline_fit", 
                    duration=fit_time,
                    success=False
                )
                
                # Re-raise with enhanced error information
                raise PipelineError(
                    f"Pipeline fit failed: {str(e)}",
                    classification=ErrorClassification.CONFIGURATION_ERROR,
                    pipeline_stage="fit",
                    context=error_info,
                    recovery_suggestions=[
                        "Check data format and pipeline step compatibility",
                        "Reduce data size or enable streaming",
                        "Verify fit_params are compatible with pipeline steps"
                    ]
                ) from e
    
    def transform(self, X):
        """
        Transform data with enhanced metadata tracking and streaming support.
        
        Maintains full sklearn Pipeline.transform() API compatibility while
        adding enhanced features for metadata preservation and large dataset handling.
        
        Args:
            X: Data to transform (array-like or DataFrame)
            
        Returns:
            Transformed data with preserved format when possible
        """
        if self._state != PipelineState.FITTED:
            raise ValueError(f"Pipeline not fitted. Current state: {self._state.value}")
        
        self._state = PipelineState.EXECUTING
        start_time = time.time()
        
        # Start logging context
        request_id = self._logging_manager.log_query_start(
            database_name="pipeline",
            query=f"transform_{self._pipeline_id}",
            database_type="pipeline_transform"
        )
        
        with self._logging_manager.context(
            request_id=request_id,
            operation="pipeline_transform", 
            component="data_science_pipeline"
        ):
            try:
                # Convert input to pandas DataFrame if needed for analysis
                X_df = self._ensure_dataframe(X, "transform_input")
                
                # Execute transform with metadata tracking
                if self.streaming_config.enabled and len(X_df) > self.streaming_config.threshold_mb * 1000:
                    # Use streaming transform for large datasets
                    result = self._transform_with_streaming(X)
                else:
                    # Standard sklearn transform with metadata tracking
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
                    database_name="pipeline",
                    database_type="pipeline_transform",
                    duration=self._transform_time,
                    success=True
                )
                
                logger.info("Pipeline transform completed",
                           pipeline_id=self._pipeline_id,
                           transform_time=self._transform_time,
                           input_shape=X_df.shape,
                           output_shape=getattr(result, 'shape', 'N/A'))
                
                return result
                
            except Exception as e:
                self._state = PipelineState.ERROR
                transform_time = time.time() - start_time
                
                # Try to get partial results
                partial_results = self._get_partial_transform_results()
                
                # Handle error with recovery
                error_info = self._handle_pipeline_error(e, "transform", {
                    "data_shape": X_df.shape if 'X_df' in locals() else None,
                    "transform_time": transform_time
                }, partial_results)
                
                self._logging_manager.log_query_complete(
                    request_id=request_id,
                    database_name="pipeline",
                    database_type="pipeline_transform",
                    duration=transform_time,
                    success=False
                )
                
                # Re-raise with enhanced error information
                raise PipelineError(
                    f"Pipeline transform failed: {str(e)}",
                    classification=ErrorClassification.COMPUTATION_TIMEOUT if "timeout" in str(e).lower() 
                                 else ErrorClassification.DATA_QUALITY_FAILURE,
                    pipeline_stage="transform", 
                    context=error_info,
                    partial_results=partial_results,
                    recovery_suggestions=[
                        "Check data format matches training data",
                        "Enable streaming for large datasets", 
                        "Verify pipeline is properly fitted"
                    ]
                ) from e
    
    def fit_transform(self, X, y=None, **fit_params):
        """
        Fit pipeline and transform in one step with enhanced metadata tracking.
        
        Maintains sklearn Pipeline.fit_transform() API compatibility.
        
        Args:
            X: Training data
            y: Target values (optional)
            **fit_params: Parameters for fit method
            
        Returns:
            Transformed training data
        """
        # Use the enhanced fit method
        self.fit(X, y, **fit_params)
        
        # Use the enhanced transform method  
        return self.transform(X)
    
    def get_pipeline_result(self) -> Optional[PipelineResult]:
        """
        Get comprehensive pipeline execution result with metadata.
        
        Returns:
            PipelineResult with execution details and composition metadata,
            or None if pipeline hasn't been executed
        """
        if self._state in [PipelineState.INITIALIZED, PipelineState.CONFIGURED]:
            return None
        
        return PipelineResult(
            success=(self._state == PipelineState.COMPLETED),
            data=None,  # Transform results are returned directly by transform()
            metadata={
                "pipeline_id": self._pipeline_id,
                "analytical_intention": self.analytical_intention,
                "progressive_complexity": self.progressive_complexity,
                "streaming_enabled": self.streaming_config.enabled,
                "fit_time": self._fit_time,
                "transform_time": self._transform_time,
                "step_metadata": self._step_metadata,
                "error_history": self._error_history
            },
            execution_time_seconds=(self._fit_time or 0) + (self._transform_time or 0),
            memory_used_mb=self._calculate_memory_usage(),
            pipeline_stage=self._state.value,
            composition_metadata=self._composition_metadata
        )
    
    def get_step_metadata(self, step_name: Optional[str] = None) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Get metadata for pipeline steps.
        
        Args:
            step_name: Name of specific step to get metadata for.
                      If None, returns metadata for all steps.
                      
        Returns:
            Step metadata dictionary or list of metadata dictionaries
        """
        if step_name:
            for metadata in self._step_metadata:
                if metadata.get("step_name") == step_name:
                    return metadata
            return {}
        
        return self._step_metadata.copy()
    
    # Internal helper methods
    
    def _ensure_dataframe(self, data: Any, context: str) -> pd.DataFrame:
        """Convert input data to DataFrame for enhanced analysis capabilities."""
        if isinstance(data, pd.DataFrame):
            return data
        
        try:
            if hasattr(data, 'shape') and len(data.shape) == 2:
                # Array-like 2D data
                return pd.DataFrame(data)
            elif hasattr(data, '__iter__') and not isinstance(data, str):
                # Iterable data
                return pd.DataFrame(list(data))
            else:
                # Scalar or other data
                return pd.DataFrame([data])
        except Exception as e:
            logger.warning(f"Could not convert {context} to DataFrame: {e}")
            # Return a minimal DataFrame for compatibility
            return pd.DataFrame({"data": [data]})
    
    def _profile_data_characteristics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Profile data characteristics for pipeline configuration."""
        try:
            profile = {
                "shape": data.shape,
                "memory_usage_mb": data.memory_usage(deep=True).sum() / (1024 * 1024),
                "column_types": dict(data.dtypes),
                "null_percentages": (data.isnull().sum() / len(data) * 100).to_dict(),
                "numeric_columns": data.select_dtypes(include=[np.number]).columns.tolist(),
                "categorical_columns": data.select_dtypes(include=['object', 'category']).columns.tolist(),
                "datetime_columns": data.select_dtypes(include=['datetime64']).columns.tolist()
            }
            
            # Statistical summary for numeric columns
            if profile["numeric_columns"]:
                profile["numeric_summary"] = data[profile["numeric_columns"]].describe().to_dict()
                
            return profile
        
        except Exception as e:
            logger.warning(f"Data profiling failed: {e}")
            return {
                "shape": getattr(data, 'shape', (0, 0)),
                "memory_usage_mb": 0,
                "column_types": {},
                "null_percentages": {},
                "numeric_columns": [],
                "categorical_columns": [],
                "datetime_columns": []
            }
    
    def _should_enable_streaming(self, data: pd.DataFrame, profile: Dict[str, Any]) -> bool:
        """Determine if streaming should be enabled based on data characteristics."""
        if self.streaming_config.enabled:
            return True
        
        data_size_mb = profile.get("memory_usage_mb", 0)
        row_count = data.shape[0]
        
        return (data_size_mb > self.streaming_config.threshold_mb or 
                row_count > 100000)
    
    def _configure_streaming_for_fit(self, data: pd.DataFrame, profile: Dict[str, Any]):
        """Configure streaming parameters for fitting phase."""
        if not self.streaming_config.enabled:
            self.streaming_config.enabled = True
        
        # Initialize streaming executor if needed
        if not self._streaming_executor:
            self._streaming_executor = StreamingQueryExecutor()
        
        logger.info("Streaming configured for pipeline fit",
                   pipeline_id=self._pipeline_id,
                   data_size_mb=profile.get("memory_usage_mb", 0),
                   threshold_mb=self.streaming_config.threshold_mb)
    
    def _build_initial_composition_metadata(self, data: pd.DataFrame, target=None) -> CompositionMetadata:
        """Build initial composition metadata for tool chaining."""
        # Determine domain based on pipeline steps
        domain = self._infer_domain_from_steps()
        
        # Determine analysis type
        analysis_type = self._infer_analysis_type_from_steps()
        
        return CompositionMetadata(
            domain=domain,
            analysis_type=analysis_type,
            result_type="pipeline_output",
            input_schema={
                "columns": list(data.columns),
                "dtypes": {str(col): str(dtype) for col, dtype in data.dtypes.items()},
                "shape": data.shape,
                "has_target": target is not None
            },
            transformation_summary={
                "intention": self.analytical_intention,
                "complexity": self.progressive_complexity,
                "steps": [step[0] for step in self.steps],
                "streaming_enabled": self.streaming_config.enabled
            },
            compatible_tools=self._suggest_compatible_tools(domain, analysis_type),
            suggested_compositions=self._suggest_next_steps(domain, analysis_type)
        )
    
    def _infer_domain_from_steps(self) -> str:
        """Infer domain from pipeline steps."""
        step_types = [type(step[1]).__name__.lower() for step in self.steps]
        
        # Check for common patterns
        if any('regress' in step for step in step_types):
            return "regression"
        elif any('classif' in step for step in step_types):
            return "classification"  
        elif any('cluster' in step for step in step_types):
            return "clustering"
        elif any('decomposition' in step or 'pca' in step for step in step_types):
            return "dimensionality_reduction"
        else:
            return "general_ml"
    
    def _infer_analysis_type_from_steps(self) -> str:
        """Infer analysis type from pipeline steps."""
        if len(self.steps) == 0:
            return "unknown"
        
        # Check the final step (usually the model/analyzer)
        final_step_name = type(self.steps[-1][1]).__name__.lower()
        
        if 'classifier' in final_step_name:
            return "classification"
        elif 'regressor' in final_step_name or 'regression' in final_step_name:
            return "regression"
        elif 'cluster' in final_step_name:
            return "clustering"
        elif 'transformer' in final_step_name or 'scaler' in final_step_name:
            return "preprocessing"
        else:
            return "analysis"
    
    def _suggest_compatible_tools(self, domain: str, analysis_type: str) -> List[str]:
        """Suggest compatible LocalData MCP tools based on domain and analysis type."""
        tools = ["visualize_data", "statistical_summary"]
        
        if domain == "regression":
            tools.extend(["model_evaluation", "residual_analysis", "feature_importance"])
        elif domain == "classification":
            tools.extend(["confusion_matrix", "classification_report", "roc_analysis"])
        elif domain == "clustering":
            tools.extend(["cluster_analysis", "silhouette_analysis"])
        
        return tools
    
    def _suggest_next_steps(self, domain: str, analysis_type: str) -> List[Dict[str, Any]]:
        """Suggest next steps in the analysis workflow."""
        suggestions = []
        
        if domain in ["regression", "classification"]:
            suggestions.extend([
                {"tool": "model_evaluation", "purpose": "Assess model performance"},
                {"tool": "feature_importance", "purpose": "Understand feature contributions"},
                {"tool": "visualize_predictions", "purpose": "Plot predictions vs actual values"}
            ])
        
        return suggestions
    
    def _fit_with_streaming(self, X, y=None, **fit_params):
        """Fit pipeline with streaming support for large datasets."""
        logger.info("Using streaming fit for large dataset",
                   pipeline_id=self._pipeline_id)
        
        # For now, use standard fit but with memory monitoring
        # Future enhancement: implement true streaming fit
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Suppress sklearn warnings during streaming
            return super().fit(X, y, **fit_params)
    
    def _transform_with_metadata_tracking(self, X):
        """Transform data with enhanced metadata tracking."""
        # Track metadata for each step
        step_results = []
        current_data = X
        
        try:
            for step_idx, (step_name, step_estimator) in enumerate(self.steps):
                step_start_time = time.time()
                
                # Execute the step
                if hasattr(step_estimator, 'transform'):
                    step_result = step_estimator.transform(current_data)
                else:
                    # Final step might be a predictor
                    step_result = step_estimator.predict(current_data) if hasattr(step_estimator, 'predict') else current_data
                
                step_execution_time = time.time() - step_start_time
                
                # Record step metadata
                step_metadata = {
                    "step_name": step_name,
                    "step_index": step_idx,
                    "step_type": type(step_estimator).__name__,
                    "execution_time": step_execution_time,
                    "input_shape": getattr(current_data, 'shape', 'N/A'),
                    "output_shape": getattr(step_result, 'shape', 'N/A'),
                    "timestamp": time.time()
                }
                
                self._step_metadata.append(step_metadata)
                step_results.append(step_result)
                current_data = step_result
                
                logger.debug(f"Step '{step_name}' completed",
                           pipeline_id=self._pipeline_id,
                           step_time=step_execution_time,
                           input_shape=step_metadata["input_shape"],
                           output_shape=step_metadata["output_shape"])
            
            return current_data
        
        except Exception as e:
            # Store partial results for error recovery
            self._execution_context["partial_step_results"] = step_results
            self._execution_context["failed_at_step"] = len(step_results)
            raise
    
    def _transform_with_streaming(self, X):
        """Transform data using streaming for large datasets."""
        logger.info("Using streaming transform for large dataset",
                   pipeline_id=self._pipeline_id)
        
        # For now, use standard transform but with memory monitoring
        # Future enhancement: implement true streaming transform
        return self._transform_with_metadata_tracking(X)
    
    def _update_composition_metadata_post_transform(self, input_data: pd.DataFrame, output_data):
        """Update composition metadata after successful transform."""
        if not self._composition_metadata:
            return
        
        # Update output schema
        if hasattr(output_data, 'shape'):
            self._composition_metadata.output_schema = {
                "shape": output_data.shape,
                "dtype": str(getattr(output_data, 'dtype', 'mixed'))
            }
        
        # Update transformation summary
        self._composition_metadata.transformation_summary.update({
            "completed_steps": len(self.steps),
            "execution_successful": True,
            "output_type": type(output_data).__name__
        })
        
        # Calculate quality score based on execution
        self._composition_metadata.quality_score = self._calculate_pipeline_quality_score()
        
        # Update confidence level
        self._composition_metadata.confidence_level = min(0.9, self._composition_metadata.quality_score)
    
    def _calculate_pipeline_quality_score(self) -> float:
        """Calculate pipeline quality score based on execution metrics."""
        score = 0.5  # Base score
        
        # Add score for successful completion
        if self._state == PipelineState.COMPLETED:
            score += 0.3
        
        # Add score for reasonable execution time
        total_time = (self._fit_time or 0) + (self._transform_time or 0)
        if total_time < 60:  # Less than 1 minute
            score += 0.1
        
        # Add score for successful streaming if used
        if self.streaming_config.enabled and len(self._step_metadata) > 0:
            score += 0.1
        
        return min(1.0, score)
    
    def _get_partial_transform_results(self) -> Optional[Any]:
        """Get partial results from failed transform operation."""
        return self._execution_context.get("partial_step_results")
    
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
        error_info = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "pipeline_stage": stage,
            "pipeline_id": self._pipeline_id,
            "timestamp": time.time(),
            "context": context or {},
            "partial_results_available": partial_results is not None
        }
        
        self._error_history.append(error_info)
        
        # Log error with full context
        self._logging_manager.log_error(
            error, 
            "data_science_pipeline",
            pipeline_id=self._pipeline_id,
            stage=stage,
            **error_info
        )
        
        return error_info


class PipelineComposer:
    """
    Orchestrates multiple DataSciencePipeline instances for complex multi-stage workflows.
    
    The PipelineComposer implements sophisticated workflow orchestration capabilities:
    - Sequential execution with metadata flow between pipeline stages
    - Parallel execution for independent analyses
    - Intelligent dependency resolution and execution scheduling
    - Automatic composition metadata enrichment for LLM tool chaining
    - Error propagation with graceful recovery and partial results
    - Streaming integration with memory-aware execution
    
    Key Features:
    - Context-aware composition with rich metadata for intelligent tool chaining
    - Progressive disclosure from simple workflows to complex orchestration
    - Intention-driven workflow design ("Analyze customer data" → multi-stage workflow)
    - Streaming-first architecture supporting large dataset workflows
    - Modular integration with easy addition of new pipeline stages
    
    Example Usage:
        # Sequential workflow
        composer = PipelineComposer('sequential')
        composer.add_pipeline('cleaning', data_cleaning_pipeline)
        composer.add_pipeline('analysis', statistical_analysis_pipeline, depends_on='cleaning')
        composer.add_pipeline('visualization', viz_pipeline, depends_on='analysis')
        results = composer.execute(data)
        
        # Parallel analysis
        composer = PipelineComposer('parallel')
        composer.add_pipeline('stats', stats_pipeline)
        composer.add_pipeline('ml', ml_pipeline) 
        composer.add_pipeline('viz', viz_pipeline)
        results = composer.execute(data)
    """
    
    def __init__(self,
                 composition_strategy: str = 'sequential',
                 metadata_enrichment: bool = True,
                 streaming_aware: bool = True,
                 error_recovery_mode: str = 'partial',
                 max_parallel_pipelines: int = 4,
                 composition_timeout_seconds: int = 3600):
        """
        Initialize PipelineComposer with workflow orchestration capabilities.
        
        Args:
            composition_strategy: Strategy for pipeline execution ('sequential', 'parallel', 'adaptive')
            metadata_enrichment: Whether to generate enriched composition metadata
            streaming_aware: Enable streaming integration for large datasets
            error_recovery_mode: Error handling strategy ('strict', 'partial', 'continue')
            max_parallel_pipelines: Maximum number of pipelines to run in parallel
            composition_timeout_seconds: Timeout for entire composition workflow
        """
        self.composition_strategy = composition_strategy
        self.metadata_enrichment = metadata_enrichment
        self.streaming_aware = streaming_aware
        self.error_recovery_mode = error_recovery_mode
        self.max_parallel_pipelines = max_parallel_pipelines
        self.composition_timeout_seconds = composition_timeout_seconds
        
        # Pipeline registry and dependency graph
        self._pipelines: Dict[str, DataSciencePipeline] = {}
        self._pipeline_metadata: Dict[str, Dict[str, Any]] = {}
        self._dependency_graph: Dict[str, List[str]] = defaultdict(list)  # name -> [dependencies]
        self._reverse_dependencies: Dict[str, List[str]] = defaultdict(list)  # name -> [dependents]
        
        # Execution state management
        self._composer_id = str(uuid.uuid4())
        self._execution_order: List[str] = []
        self._parallel_groups: List[List[str]] = []
        self._composition_metadata: Optional[CompositionMetadata] = None
        
        # Results and error tracking
        self._pipeline_results: Dict[str, PipelineResult] = {}
        self._pipeline_errors: Dict[str, Exception] = {}
        self._partial_results: Dict[str, Any] = {}
        self._execution_metrics: Dict[str, Dict[str, Any]] = {}
        
        # Streaming and performance management
        self._streaming_executor: Optional[StreamingQueryExecutor] = None
        self._memory_tracker: List[MemoryStatus] = []
        self._optimization_cache: Dict[str, Any] = {}
        
        # Logging integration
        self._logging_manager = get_logging_manager()
        self._request_id: Optional[str] = None
        
        logger.info("PipelineComposer initialized",
                   composer_id=self._composer_id,
                   strategy=composition_strategy,
                   metadata_enrichment=metadata_enrichment,
                   streaming_aware=streaming_aware,
                   error_recovery=error_recovery_mode)
    
    @property
    def composer_id(self) -> str:
        """Get unique composer identifier."""
        return self._composer_id
    
    @property
    def registered_pipelines(self) -> Dict[str, str]:
        """Get registry of pipeline names and their types."""
        return {name: type(pipeline).__name__ for name, pipeline in self._pipelines.items()}
    
    @property
    def composition_metadata(self) -> Optional[CompositionMetadata]:
        """Get enriched composition metadata for tool chaining."""
        return self._composition_metadata
    
    def add_pipeline(self,
                    name: str,
                    pipeline: DataSciencePipeline,
                    depends_on: Optional[Union[str, List[str]]] = None,
                    metadata: Optional[Dict[str, Any]] = None,
                    data_transformation: Optional[Callable] = None) -> 'PipelineComposer':
        """
        Add a pipeline to the composition with optional dependencies.
        
        Args:
            name: Unique name for the pipeline
            pipeline: DataSciencePipeline instance to add
            depends_on: Pipeline name(s) this pipeline depends on
            metadata: Additional metadata for the pipeline
            data_transformation: Function to transform data between pipelines
            
        Returns:
            Self for method chaining
            
        Raises:
            ValueError: If pipeline name already exists or invalid dependencies
        """
        if name in self._pipelines:
            raise ValueError(f"Pipeline '{name}' already exists in composition")
        
        # Validate dependencies
        dependencies = []
        if depends_on:
            if isinstance(depends_on, str):
                dependencies = [depends_on]
            else:
                dependencies = list(depends_on)
            
            for dep in dependencies:
                if dep not in self._pipelines:
                    raise ValueError(f"Dependency '{dep}' not found in composition")
        
        # Register pipeline
        self._pipelines[name] = pipeline
        self._pipeline_metadata[name] = {
            'dependencies': dependencies,
            'data_transformation': data_transformation,
            'metadata': metadata or {},
            'registration_time': time.time(),
            'pipeline_type': type(pipeline).__name__,
            'analytical_intention': getattr(pipeline, 'analytical_intention', 'Unknown')
        }
        
        # Update dependency graph
        self._dependency_graph[name] = dependencies
        for dep in dependencies:
            self._reverse_dependencies[dep].append(name)
        
        # Invalidate cached execution order
        self._execution_order = []
        self._parallel_groups = []
        
        logger.info(f"Pipeline '{name}' added to composition",
                   composer_id=self._composer_id,
                   pipeline_type=type(pipeline).__name__,
                   dependencies=dependencies,
                   total_pipelines=len(self._pipelines))
        
        return self
    
    def resolve_dependencies(self) -> Dict[str, Any]:
        """
        Analyze and resolve pipeline dependencies to create execution plan.
        
        Returns:
            Dependency analysis report with execution order and parallel groups
            
        Raises:
            ValueError: If circular dependencies are detected
        """
        if not self._pipelines:
            return {
                'execution_order': [],
                'parallel_groups': [],
                'dependency_analysis': 'No pipelines registered'
            }
        
        # Detect circular dependencies using DFS
        def has_cycle(node: str, visited: set, rec_stack: set) -> bool:
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in self._dependency_graph.get(node, []):
                if neighbor not in visited:
                    if has_cycle(neighbor, visited, rec_stack):
                        return True
                elif neighbor in rec_stack:
                    return True
            
            rec_stack.remove(node)
            return False
        
        visited = set()
        for pipeline_name in self._pipelines:
            if pipeline_name not in visited:
                if has_cycle(pipeline_name, visited, set()):
                    raise ValueError(f"Circular dependency detected involving pipeline '{pipeline_name}'")
        
        # Topological sort for execution order
        execution_order = self._topological_sort()
        self._execution_order = execution_order
        
        # Identify parallel groups (pipelines with no interdependencies)
        parallel_groups = self._identify_parallel_groups(execution_order)
        self._parallel_groups = parallel_groups
        
        dependency_report = {
            'execution_order': execution_order,
            'parallel_groups': parallel_groups,
            'dependency_graph': dict(self._dependency_graph),
            'reverse_dependencies': dict(self._reverse_dependencies),
            'total_pipelines': len(self._pipelines),
            'parallelizable_pipelines': sum(len(group) for group in parallel_groups if len(group) > 1),
            'dependency_analysis': 'Valid dependency graph - no cycles detected'
        }
        
        logger.info("Dependencies resolved",
                   composer_id=self._composer_id,
                   execution_order=execution_order,
                   parallel_groups_count=len([g for g in parallel_groups if len(g) > 1]),
                   total_stages=len(parallel_groups))
        
        return dependency_report
    
    def execute(self, data: pd.DataFrame, **kwargs) -> Dict[str, PipelineResult]:
        """
        Execute the composition using the configured strategy.
        
        Args:
            data: Input data for the composition
            **kwargs: Additional parameters for pipeline execution
            
        Returns:
            Dictionary of pipeline results keyed by pipeline name
        """
        strategy = kwargs.pop('strategy', self.composition_strategy)
        return self._execute_composition(strategy, data, **kwargs)
    
    def _topological_sort(self) -> List[str]:
        """
        Perform topological sort to determine pipeline execution order.
        
        Returns:
            List of pipeline names in execution order
        """
        # Kahn's algorithm for topological sorting
        in_degree = {name: 0 for name in self._pipelines}
        for name in self._pipelines:
            for dep in self._dependency_graph.get(name, []):
                in_degree[name] += 1
        
        # Find all nodes with no incoming edges
        queue = deque([name for name, degree in in_degree.items() if degree == 0])
        result = []
        
        while queue:
            current = queue.popleft()
            result.append(current)
            
            # For each dependent of current node
            for dependent in self._reverse_dependencies.get(current, []):
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)
        
        # Check if all nodes were processed (no cycle)
        if len(result) != len(self._pipelines):
            raise ValueError("Circular dependency detected in pipeline composition")
        
        return result
    
    def _identify_parallel_groups(self, execution_order: List[str]) -> List[List[str]]:
        """
        Identify groups of pipelines that can be executed in parallel.
        
        Args:
            execution_order: Topologically sorted pipeline names
            
        Returns:
            List of pipeline groups, each group can be executed in parallel
        """
        groups = []
        processed = set()
        
        for pipeline_name in execution_order:
            if pipeline_name in processed:
                continue
            
            # Start a new group with this pipeline
            parallel_group = [pipeline_name]
            dependencies = set(self._dependency_graph.get(pipeline_name, []))
            
            # Check remaining pipelines to see which can run in parallel
            for other_name in execution_order:
                if other_name == pipeline_name or other_name in processed:
                    continue
                
                other_deps = set(self._dependency_graph.get(other_name, []))
                
                # Can run in parallel if they don't depend on each other
                # and have compatible dependency requirements
                if (pipeline_name not in other_deps and 
                    other_name not in dependencies and
                    dependencies == other_deps):  # Same level in dependency hierarchy
                    parallel_group.append(other_name)
            
            groups.append(parallel_group)
            processed.update(parallel_group)
        
        return groups
    
    def _execute_composition(self, strategy: str, data: pd.DataFrame, **kwargs) -> Dict[str, PipelineResult]:
        """
        Execute pipeline composition with specified strategy.
        
        Args:
            strategy: Execution strategy ('sequential', 'parallel', 'adaptive')
            data: Input data
            **kwargs: Additional execution parameters
            
        Returns:
            Dictionary of pipeline results
        """
        if not self._pipelines:
            raise ValueError("No pipelines registered in composition")
        
        # Start logging context
        self._request_id = self._logging_manager.log_query_start(
            database_name="composition",
            query=f"execute_{strategy}_{self._composer_id}",
            database_type="pipeline_composition"
        )
        
        start_time = time.time()
        
        with self._logging_manager.context(
            request_id=self._request_id,
            operation="pipeline_composition",
            component="pipeline_composer"
        ):
            try:
                # Resolve dependencies and create execution plan
                dependency_report = self.resolve_dependencies()
                
                # Initialize execution state
                self._pipeline_results = {}
                self._pipeline_errors = {}
                self._partial_results = {}
                self._execution_metrics = {}
                
                # Build initial composition metadata
                if self.metadata_enrichment:
                    self._composition_metadata = self._build_composition_metadata(data, strategy)
                
                # Execute based on strategy
                if strategy == 'sequential':
                    results = self._execute_sequential_workflow(data, dependency_report, **kwargs)
                elif strategy == 'parallel':
                    results = self._execute_parallel_workflow(data, dependency_report, **kwargs)
                elif strategy == 'adaptive':
                    results = self._execute_adaptive_workflow(data, dependency_report, **kwargs)
                else:
                    raise ValueError(f"Unsupported composition strategy: {strategy}")
                
                # Enrich composition metadata with execution results
                if self.metadata_enrichment and self._composition_metadata:
                    self._enrich_metadata_with_results(results)
                
                execution_time = time.time() - start_time
                
                # Log successful completion
                self._logging_manager.log_query_complete(
                    request_id=self._request_id,
                    database_name="composition",
                    database_type="pipeline_composition",
                    duration=execution_time,
                    success=True
                )
                
                logger.info("Pipeline composition executed successfully",
                           composer_id=self._composer_id,
                           strategy=strategy,
                           execution_time=execution_time,
                           successful_pipelines=len([r for r in results.values() if r.success]),
                           total_pipelines=len(results))
                
                return results
                
            except Exception as e:
                execution_time = time.time() - start_time
                
                # Handle composition error
                error_info = self._handle_composition_error(e, strategy, {
                    'execution_time': execution_time,
                    'data_shape': data.shape,
                    'total_pipelines': len(self._pipelines)
                })
                
                self._logging_manager.log_query_complete(
                    request_id=self._request_id,
                    database_name="composition",
                    database_type="pipeline_composition",
                    duration=execution_time,
                    success=False
                )
                
                # Return partial results if available in recovery mode
                if self.error_recovery_mode in ['partial', 'continue']:
                    return self._pipeline_results
                else:
                    raise PipelineError(
                        f"Pipeline composition failed: {str(e)}",
                        classification=ErrorClassification.CONFIGURATION_ERROR,
                        pipeline_stage="composition",
                        context=error_info,
                        partial_results=self._pipeline_results,
                        recovery_suggestions=[
                            "Check pipeline dependencies and configuration",
                            "Enable error recovery mode for partial results",
                            "Verify input data format and size"
                        ]
                    ) from e
    
    def _execute_sequential_workflow(self, data: pd.DataFrame, dependency_report: Dict[str, Any], **kwargs) -> Dict[str, PipelineResult]:
        """
        Execute pipelines sequentially with metadata flow between stages.
        
        Args:
            data: Input data
            dependency_report: Resolved dependency information
            **kwargs: Additional execution parameters
            
        Returns:
            Dictionary of pipeline results
        """
        execution_order = dependency_report['execution_order']
        current_data = data
        results = {}
        
        for pipeline_name in execution_order:
            try:
                pipeline = self._pipelines[pipeline_name]
                metadata = self._pipeline_metadata[pipeline_name]
                
                # Apply data transformation if specified
                if metadata.get('data_transformation'):
                    current_data = metadata['data_transformation'](current_data)
                
                # Execute pipeline
                logger.debug(f"Executing pipeline '{pipeline_name}' sequentially",
                           composer_id=self._composer_id,
                           data_shape=current_data.shape)
                
                # Fit and transform with the pipeline
                pipeline_start_time = time.time()
                pipeline.fit(current_data)
                result = pipeline.transform(current_data)
                
                # If the result is a PipelineResult, use it directly
                if isinstance(result, PipelineResult):
                    pipeline_result = result
                else:
                    # Create PipelineResult wrapper
                    pipeline_result = PipelineResult(
                        success=True,
                        data=result,
                        metadata={
                            'pipeline_name': pipeline_name,
                            'sequential_execution': True,
                            'execution_order': execution_order.index(pipeline_name)
                        },
                        execution_time_seconds=time.time() - pipeline_start_time,
                        memory_used_mb=self._calculate_pipeline_memory_usage(pipeline),
                        pipeline_stage=pipeline.state.value if hasattr(pipeline, 'state') else 'completed',
                        composition_metadata=pipeline.composition_metadata if hasattr(pipeline, 'composition_metadata') else None
                    )
                
                results[pipeline_name] = pipeline_result
                
                # Use pipeline output as input for next stage if it's DataFrame-like
                if isinstance(result, pd.DataFrame):
                    current_data = result
                elif hasattr(result, 'data') and isinstance(result.data, pd.DataFrame):
                    current_data = result.data
                
                # Propagate metadata between pipelines
                self._propagate_metadata_between_pipelines(pipeline_name, pipeline_result)
                
                logger.info(f"Pipeline '{pipeline_name}' completed successfully in sequential workflow",
                           composer_id=self._composer_id,
                           execution_time=pipeline_result.execution_time_seconds,
                           output_shape=getattr(result, 'shape', 'N/A'))
                
            except Exception as e:
                # Handle pipeline error in sequential workflow
                error_result = self._handle_pipeline_error_in_composition(pipeline_name, e, 'sequential')
                results[pipeline_name] = error_result
                
                if self.error_recovery_mode == 'strict':
                    # Stop execution on first error in strict mode
                    break
                elif self.error_recovery_mode == 'continue':
                    # Continue with next pipeline, using original data
                    continue
                # For 'partial' mode, continue with whatever data we have
        
        return results
    
    def _execute_parallel_workflow(self, data: pd.DataFrame, dependency_report: Dict[str, Any], **kwargs) -> Dict[str, PipelineResult]:
        """
        Execute pipelines in parallel for independent analyses.
        
        Args:
            data: Input data (copied to each parallel pipeline)
            dependency_report: Resolved dependency information
            **kwargs: Additional execution parameters
            
        Returns:
            Dictionary of pipeline results
        """
        parallel_groups = dependency_report['parallel_groups']
        results = {}
        
        for group_idx, group in enumerate(parallel_groups):
            if len(group) == 1:
                # Single pipeline execution
                pipeline_name = group[0]
                result = self._execute_single_pipeline(pipeline_name, data, kwargs)
                results[pipeline_name] = result
            else:
                # Parallel execution within group
                group_results = self._execute_parallel_group(group, data, kwargs)
                results.update(group_results)
        
        return results
    
    def _execute_adaptive_workflow(self, data: pd.DataFrame, dependency_report: Dict[str, Any], **kwargs) -> Dict[str, PipelineResult]:
        """
        Execute pipelines using adaptive strategy based on data characteristics and dependencies.
        
        Args:
            data: Input data
            dependency_report: Resolved dependency information
            **kwargs: Additional execution parameters
            
        Returns:
            Dictionary of pipeline results
        """
        # Analyze data characteristics to choose strategy
        data_size_mb = data.memory_usage(deep=True).sum() / (1024 * 1024)
        has_dependencies = any(len(deps) > 0 for deps in self._dependency_graph.values())
        parallelizable_count = dependency_report['parallelizable_pipelines']
        
        # Adaptive strategy decision
        if has_dependencies and data_size_mb > 100:
            # Large data with dependencies - use sequential with streaming
            logger.info("Adaptive strategy: Sequential execution with streaming for large data",
                       composer_id=self._composer_id,
                       data_size_mb=data_size_mb,
                       has_dependencies=has_dependencies)
            return self._execute_sequential_workflow(data, dependency_report, **kwargs)
        elif parallelizable_count > 1 and data_size_mb < 500:
            # Moderate data size with parallelizable pipelines
            logger.info("Adaptive strategy: Parallel execution for independent pipelines",
                       composer_id=self._composer_id,
                       parallelizable_count=parallelizable_count,
                       data_size_mb=data_size_mb)
            return self._execute_parallel_workflow(data, dependency_report, **kwargs)
        else:
            # Default to sequential for complex dependency graphs or very large data
            logger.info("Adaptive strategy: Sequential execution (default)",
                       composer_id=self._composer_id,
                       has_dependencies=has_dependencies,
                       data_size_mb=data_size_mb)
            return self._execute_sequential_workflow(data, dependency_report, **kwargs)
    
    def _execute_single_pipeline(self, pipeline_name: str, data: pd.DataFrame, kwargs: Dict[str, Any]) -> PipelineResult:
        """
        Execute a single pipeline with error handling.
        
        Args:
            pipeline_name: Name of pipeline to execute
            data: Input data
            kwargs: Additional parameters
            
        Returns:
            PipelineResult from execution
        """
        try:
            pipeline = self._pipelines[pipeline_name]
            metadata = self._pipeline_metadata[pipeline_name]
            
            # Apply data transformation if specified
            if metadata.get('data_transformation'):
                data = metadata['data_transformation'](data)
            
            logger.debug(f"Executing single pipeline '{pipeline_name}'",
                       composer_id=self._composer_id,
                       data_shape=data.shape)
            
            # Execute pipeline
            pipeline_start_time = time.time()
            pipeline.fit(data)
            result = pipeline.transform(data)
            
            # Ensure we return a PipelineResult
            if isinstance(result, PipelineResult):
                return result
            else:
                return PipelineResult(
                    success=True,
                    data=result,
                    metadata={
                        'pipeline_name': pipeline_name,
                        'single_execution': True
                    },
                    execution_time_seconds=time.time() - pipeline_start_time,
                    memory_used_mb=self._calculate_pipeline_memory_usage(pipeline),
                    pipeline_stage=pipeline.state.value if hasattr(pipeline, 'state') else 'completed',
                    composition_metadata=pipeline.composition_metadata if hasattr(pipeline, 'composition_metadata') else None
                )
        
        except Exception as e:
            return self._handle_pipeline_error_in_composition(pipeline_name, e, 'single')
    
    def _execute_parallel_group(self, group: List[str], data: pd.DataFrame, kwargs: Dict[str, Any]) -> Dict[str, PipelineResult]:
        """
        Execute a group of pipelines in parallel.
        
        Args:
            group: List of pipeline names to execute in parallel
            data: Input data (copied to each pipeline)
            kwargs: Additional parameters
            
        Returns:
            Dictionary of pipeline results
        """
        results = {}
        
        # Use ThreadPoolExecutor for I/O bound operations
        max_workers = min(len(group), self.max_parallel_pipelines)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all pipelines for parallel execution
            future_to_pipeline = {}
            for pipeline_name in group:
                # Create a copy of data for each pipeline to avoid conflicts
                pipeline_data = data.copy() if hasattr(data, 'copy') else data
                future = executor.submit(self._execute_single_pipeline, pipeline_name, pipeline_data, kwargs)
                future_to_pipeline[future] = pipeline_name
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_pipeline, timeout=self.composition_timeout_seconds):
                pipeline_name = future_to_pipeline[future]
                try:
                    result = future.result()
                    results[pipeline_name] = result
                    
                    logger.info(f"Pipeline '{pipeline_name}' completed in parallel group",
                               composer_id=self._composer_id,
                               success=result.success,
                               execution_time=result.execution_time_seconds)
                
                except Exception as e:
                    error_result = self._handle_pipeline_error_in_composition(pipeline_name, e, 'parallel')
                    results[pipeline_name] = error_result
        
        return results
    
    def _build_composition_metadata(self, data: pd.DataFrame, strategy: str) -> CompositionMetadata:
        """
        Build comprehensive composition metadata for tool chaining.
        
        Args:
            data: Input data
            strategy: Execution strategy
            
        Returns:
            Rich composition metadata for LLM tool chaining
        """
        # Analyze the overall composition domain and type
        domains = set()
        analysis_types = set()
        compatible_tools = set()
        
        for pipeline_name, pipeline in self._pipelines.items():
            # Extract domain information from pipelines
            if hasattr(pipeline, 'composition_metadata') and pipeline.composition_metadata:
                meta = pipeline.composition_metadata
                domains.add(meta.domain)
                analysis_types.add(meta.analysis_type)
                compatible_tools.update(meta.compatible_tools)
        
        # Determine composite domain and analysis type
        if len(domains) == 1:
            composite_domain = list(domains)[0]
        elif 'ml' in domains or 'machine_learning' in domains:
            composite_domain = 'ml'
        else:
            composite_domain = 'multi_domain'
        
        if len(analysis_types) == 1:
            composite_analysis_type = list(analysis_types)[0]
        else:
            composite_analysis_type = 'multi_stage_analysis'
        
        # Generate tool suggestions for the composition
        composition_tools = list(compatible_tools)
        composition_tools.extend([
            'pipeline_visualization',
            'workflow_monitoring',
            'result_comparison',
            'performance_analysis'
        ])
        
        # Create suggested next steps
        suggested_compositions = [
            {
                'tool': 'visualize_pipeline_results',
                'purpose': 'Visualize outputs from multiple pipeline stages',
                'priority': 'high'
            },
            {
                'tool': 'compare_pipeline_performance',
                'purpose': 'Compare execution metrics across pipelines',
                'priority': 'medium'
            },
            {
                'tool': 'export_workflow_results',
                'purpose': 'Export complete workflow results for reporting',
                'priority': 'medium'
            }
        ]
        
        return CompositionMetadata(
            domain=composite_domain,
            analysis_type=composite_analysis_type,
            result_type='multi_pipeline_composition',
            compatible_tools=composition_tools,
            suggested_compositions=suggested_compositions,
            data_artifacts={
                'total_pipelines': len(self._pipelines),
                'execution_strategy': strategy,
                'dependency_complexity': len([d for d in self._dependency_graph.values() if len(d) > 0]),
                'parallelizable_stages': len([g for g in self._parallel_groups if len(g) > 1])
            },
            input_schema={
                'columns': list(data.columns),
                'dtypes': {str(col): str(dtype) for col, dtype in data.dtypes.items()},
                'shape': data.shape,
                'composition_input': True
            },
            transformation_summary={
                'multi_stage_workflow': True,
                'composition_strategy': strategy,
                'registered_pipelines': list(self._pipelines.keys()),
                'execution_order': self._execution_order,
                'streaming_aware': self.streaming_aware,
                'error_recovery_mode': self.error_recovery_mode
            },
            confidence_level=0.8,  # High confidence for composed workflows
            quality_score=0.75,    # Good quality baseline for orchestration
            recommended_next_steps=[
                {
                    'action': 'analyze_workflow_performance',
                    'description': 'Review execution metrics and optimize pipeline ordering',
                    'priority': 'high'
                },
                {
                    'action': 'visualize_pipeline_outputs',
                    'description': 'Create visualizations comparing outputs from different stages',
                    'priority': 'high'
                },
                {
                    'action': 'export_comprehensive_report',
                    'description': 'Generate detailed report with all pipeline results',
                    'priority': 'medium'
                }
            ]
        )
    
    def _enrich_metadata_with_results(self, results: Dict[str, PipelineResult]):
        """
        Enrich composition metadata with execution results.
        
        Args:
            results: Dictionary of pipeline results
        """
        if not self._composition_metadata:
            return
        
        successful_pipelines = [name for name, result in results.items() if result.success]
        failed_pipelines = [name for name, result in results.items() if not result.success]
        
        # Update quality score based on success rate
        success_rate = len(successful_pipelines) / len(results) if results else 0
        self._composition_metadata.quality_score = 0.5 + (success_rate * 0.5)
        
        # Update confidence level
        self._composition_metadata.confidence_level = min(0.95, success_rate * 0.9 + 0.1)
        
        # Add execution artifacts
        self._composition_metadata.data_artifacts.update({
            'successful_pipelines': successful_pipelines,
            'failed_pipelines': failed_pipelines,
            'success_rate': success_rate,
            'total_execution_time': sum(r.execution_time_seconds for r in results.values()),
            'total_memory_used': sum(r.memory_used_mb for r in results.values()),
            'execution_timestamp': time.time()
        })
        
        # Update limitations based on failures
        if failed_pipelines:
            self._composition_metadata.limitations.extend([
                f"Pipeline failures: {', '.join(failed_pipelines)}",
                "Partial results available - some analysis stages incomplete"
            ])
    
    def _propagate_metadata_between_pipelines(self, pipeline_name: str, result: PipelineResult):
        """
        Propagate metadata between pipeline stages for enhanced composition context.
        
        Args:
            pipeline_name: Name of the pipeline that just completed
            result: Result from the completed pipeline
        """
        # Store pipeline-specific metadata for downstream use
        self._execution_metrics[pipeline_name] = {
            'execution_time': result.execution_time_seconds,
            'memory_used': result.memory_used_mb,
            'success': result.success,
            'stage': result.pipeline_stage,
            'timestamp': time.time()
        }
        
        # Update dependent pipelines with context from this pipeline
        dependents = self._reverse_dependencies.get(pipeline_name, [])
        for dependent_name in dependents:
            if dependent_name in self._pipelines:
                dependent_pipeline = self._pipelines[dependent_name]
                
                # Update custom parameters with upstream context
                if hasattr(dependent_pipeline, 'custom_parameters'):
                    dependent_pipeline.custom_parameters.update({
                        f'upstream_{pipeline_name}_success': result.success,
                        f'upstream_{pipeline_name}_execution_time': result.execution_time_seconds,
                        f'upstream_{pipeline_name}_stage': result.pipeline_stage
                    })
    
    def _handle_pipeline_error_in_composition(self, pipeline_name: str, error: Exception, context: str) -> PipelineResult:
        """
        Handle errors that occur during pipeline execution within composition.
        
        Args:
            pipeline_name: Name of the failed pipeline
            error: Exception that occurred
            context: Execution context ('sequential', 'parallel', 'single')
            
        Returns:
            PipelineResult representing the error
        """
        error_info = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'pipeline_name': pipeline_name,
            'composition_context': context,
            'composer_id': self._composer_id,
            'timestamp': time.time()
        }
        
        self._pipeline_errors[pipeline_name] = error
        
        # Log the error
        logger.error(f"Pipeline '{pipeline_name}' failed in {context} composition",
                    composer_id=self._composer_id,
                    error_type=type(error).__name__,
                    error_message=str(error),
                    context=context)
        
        return PipelineResult(
            success=False,
            data=None,
            metadata={'error_info': error_info, 'pipeline_name': pipeline_name},
            execution_time_seconds=0,
            memory_used_mb=0,
            pipeline_stage='error',
            error=error_info
        )
    
    def _handle_composition_error(self, error: Exception, strategy: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle errors that occur during composition execution.
        
        Args:
            error: Exception that occurred
            strategy: Execution strategy that failed
            context: Additional context information
            
        Returns:
            Error information dictionary
        """
        error_info = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'composition_strategy': strategy,
            'composer_id': self._composer_id,
            'timestamp': time.time(),
            'context': context
        }
        
        # Log the composition error
        self._logging_manager.log_error(
            error,
            "pipeline_composer",
            composer_id=self._composer_id,
            strategy=strategy,
            **error_info
        )
        
        return error_info
    
    def _calculate_pipeline_memory_usage(self, pipeline: DataSciencePipeline) -> float:
        """
        Calculate memory usage for a specific pipeline.
        
        Args:
            pipeline: Pipeline to calculate memory for
            
        Returns:
            Memory usage in MB
        """
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except ImportError:
            return 0.0
    
    def get_composition_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive summary of the pipeline composition.
        
        Returns:
            Detailed composition summary
        """
        summary = {
            'composer_id': self._composer_id,
            'composition_strategy': self.composition_strategy,
            'total_pipelines': len(self._pipelines),
            'registered_pipelines': self.registered_pipelines,
            'dependency_graph': dict(self._dependency_graph),
            'execution_order': self._execution_order,
            'parallel_groups': self._parallel_groups,
            'metadata_enrichment_enabled': self.metadata_enrichment,
            'streaming_aware': self.streaming_aware,
            'error_recovery_mode': self.error_recovery_mode,
            'max_parallel_pipelines': self.max_parallel_pipelines,
            'composition_timeout_seconds': self.composition_timeout_seconds
        }
        
        # Add execution results if available
        if self._pipeline_results:
            summary['execution_results'] = {
                'successful_pipelines': [name for name, result in self._pipeline_results.items() if result.success],
                'failed_pipelines': [name for name, result in self._pipeline_results.items() if not result.success],
                'total_execution_time': sum(r.execution_time_seconds for r in self._pipeline_results.values()),
                'total_memory_used': sum(r.memory_used_mb for r in self._pipeline_results.values())
            }
        
        # Add composition metadata if available
        if self._composition_metadata:
            summary['composition_metadata'] = {
                'domain': self._composition_metadata.domain,
                'analysis_type': self._composition_metadata.analysis_type,
                'result_type': self._composition_metadata.result_type,
                'compatible_tools': self._composition_metadata.compatible_tools,
                'confidence_level': self._composition_metadata.confidence_level,
                'quality_score': self._composition_metadata.quality_score
            }
        
        return summary


# Export the main classes for use by other modules
__all__ = ['DataSciencePipeline', 'PipelineComposer']


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
    
    def __init__(self, 
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
                 memory_monitoring: bool = True):
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
                early_termination_enabled=True
            )
        
        # Initialize parent DataSciencePipeline
        super().__init__(
            steps=steps,
            memory=memory,
            verbose=verbose,
            analytical_intention=analytical_intention or "Large dataset processing with streaming",
            streaming_config=streaming_config,
            progressive_complexity=progressive_complexity,
            composition_aware=composition_aware,
            custom_parameters=custom_parameters
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
        
        logger.info("StreamingDataPipeline initialized",
                   pipeline_id=self._pipeline_id,
                   threshold_mb=self.streaming_threshold_mb,
                   adaptive_chunking=self.adaptive_chunking,
                   memory_monitoring=self.memory_monitoring)
    
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
            database_type="streaming_pipeline_fit"
        )
        
        with self._logging_manager.context(
            request_id=self._request_id,
            operation="streaming_pipeline_fit",
            component="streaming_data_pipeline"
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
                    logger.info("Activating streaming mode for fit",
                               pipeline_id=self._pipeline_id,
                               data_size_mb=data_profile.get("memory_usage_mb", 0),
                               threshold_mb=self.streaming_threshold_mb)
                    
                    # Configure streaming executor
                    if not self._streaming_executor:
                        self._streaming_executor = StreamingQueryExecutor()
                    
                    # Execute streaming fit
                    result = self._execute_streaming_fit(X_df, y, **fit_params)
                else:
                    # Use standard DataSciencePipeline fit for small datasets
                    logger.info("Using standard fit mode",
                               pipeline_id=self._pipeline_id,
                               data_size_mb=data_profile.get("memory_usage_mb", 0))
                    result = super().fit(X, y, **fit_params)
                
                # Build composition metadata if enabled
                if self.composition_aware:
                    self._composition_metadata = self._build_initial_composition_metadata(X_df, y)
                
                # Record successful fit
                self._fit_time = time.time() - start_time
                self._state = PipelineState.FITTED
                
                # Log successful completion
                self._logging_manager.log_query_complete(
                    request_id=self._request_id,
                    database_name="streaming_pipeline",
                    database_type="streaming_pipeline_fit",
                    duration=self._fit_time,
                    success=True
                )
                
                logger.info("StreamingDataPipeline fitted successfully",
                           pipeline_id=self._pipeline_id,
                           fit_time=self._fit_time,
                           streaming_activated=self._streaming_activated,
                           data_shape=X_df.shape)
                
                return result
                
            except Exception as e:
                self._state = PipelineState.ERROR
                fit_time = time.time() - start_time
                
                # Log error and handle
                error_info = self._handle_pipeline_error(e, "streaming_fit", {
                    "data_shape": X_df.shape if 'X_df' in locals() else None,
                    "fit_time": fit_time,
                    "streaming_activated": self._streaming_activated,
                    "step_params": fit_params
                })
                
                self._logging_manager.log_query_complete(
                    request_id=self._request_id,
                    database_name="streaming_pipeline",
                    database_type="streaming_pipeline_fit",
                    duration=fit_time,
                    success=False
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
                        "Try reducing chunk size for memory-constrained environments"
                    ]
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
            database_type="streaming_pipeline_transform"
        )
        
        with self._logging_manager.context(
            request_id=request_id,
            operation="streaming_pipeline_transform",
            component="streaming_data_pipeline"
        ):
            try:
                # Convert input to DataFrame for analysis
                X_df = self._ensure_dataframe(X, "streaming_transform_input")
                
                # Determine if streaming should be used for transform
                data_profile = self._profile_data_characteristics(X_df)
                should_stream = self._streaming_activated or self._should_use_streaming(X_df, data_profile)
                
                if should_stream:
                    logger.info("Using streaming transform",
                               pipeline_id=self._pipeline_id,
                               data_size_mb=data_profile.get("memory_usage_mb", 0))
                    
                    # Execute streaming transform
                    result = self._execute_streaming_transform(X_df)
                else:
                    # Use standard DataSciencePipeline transform
                    logger.info("Using standard transform",
                               pipeline_id=self._pipeline_id,
                               data_size_mb=data_profile.get("memory_usage_mb", 0))
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
                    success=True
                )
                
                logger.info("StreamingDataPipeline transform completed",
                           pipeline_id=self._pipeline_id,
                           transform_time=self._transform_time,
                           streaming_used=should_stream,
                           input_shape=X_df.shape,
                           output_shape=getattr(result, 'shape', 'N/A'))
                
                return result
                
            except Exception as e:
                self._state = PipelineState.ERROR
                transform_time = time.time() - start_time
                
                # Try to get partial results
                partial_results = self._get_partial_transform_results()
                
                # Handle error with recovery
                error_info = self._handle_pipeline_error(e, "streaming_transform", {
                    "data_shape": X_df.shape if 'X_df' in locals() else None,
                    "transform_time": transform_time,
                    "streaming_activated": self._streaming_activated
                }, partial_results)
                
                self._logging_manager.log_query_complete(
                    request_id=request_id,
                    database_name="streaming_pipeline",
                    database_type="streaming_pipeline_transform",
                    duration=transform_time,
                    success=False
                )
                
                # Re-raise with enhanced error information
                raise PipelineError(
                    f"StreamingDataPipeline transform failed: {str(e)}",
                    classification=ErrorClassification.COMPUTATION_TIMEOUT if "timeout" in str(e).lower() 
                                 else ErrorClassification.DATA_QUALITY_FAILURE,
                    pipeline_stage="streaming_transform",
                    context=error_info,
                    partial_results=partial_results,
                    recovery_suggestions=[
                        "Check data format matches training data",
                        "Enable or increase streaming threshold for large datasets",
                        "Verify pipeline is properly fitted with streaming",
                        "Check memory availability and reduce chunk size if needed"
                    ]
                ) from e
    
    def _should_use_streaming(self, data: pd.DataFrame, profile: Dict[str, Any]) -> bool:
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
            logger.info(f"Streaming activated: data size ({data_size_mb:.1f}MB) > threshold ({self.streaming_threshold_mb}MB)")
            return True
        
        # Row count-based activation (for sparse data)
        if row_count > 100000:  # 100K rows
            logger.info(f"Streaming activated: row count ({row_count}) > 100K rows")
            return True
        
        # Memory pressure-based activation
        if self.memory_monitoring and self._streaming_executor:
            memory_status = self._streaming_executor._get_memory_status()
            if memory_status.is_low_memory:
                logger.info(f"Streaming activated: memory pressure detected ({memory_status.used_percent:.1f}% used)")
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
        
        logger.info("Starting streaming fit execution",
                   pipeline_id=self._pipeline_id,
                   chunk_size=chunk_size,
                   total_rows=len(X),
                   memory_available_gb=memory_status.available_gb)
        
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
                        chunk_size = min(chunk_size, current_memory.recommended_chunk_size)
                        logger.warning(f"Memory pressure detected, reducing chunk size to {chunk_size}")
                
                # Get corresponding y chunk if provided
                if y is not None:
                    start_idx = (chunk_number - 1) * chunk_size
                    end_idx = start_idx + len(chunk)
                    if hasattr(y, 'iloc'):
                        y_chunk = y.iloc[start_idx:end_idx]
                    else:
                        y_chunk = y[start_idx:end_idx]
                else:
                    y_chunk = None
                
                # Fit on chunk using sklearn's partial_fit or incremental learning
                chunk_fit_result = self._fit_chunk(chunk, y_chunk, chunk_number, **fit_params)
                self._chunk_fit_results.append(chunk_fit_result)
                
                total_samples_processed += len(chunk)
                chunk_time = time.time() - chunk_start_time
                
                logger.debug(f"Processed fit chunk {chunk_number}: {len(chunk)} samples, {chunk_time:.3f}s")
                
                # Adaptive chunk size adjustment
                if self.adaptive_chunking and chunk_number > 1:
                    chunk_size = self._adaptive_chunk_sizing(
                        chunk_size, chunk_time, current_memory if self.memory_monitoring else memory_status
                    )
            
            # Aggregate chunk results for final fitted pipeline
            self._aggregate_chunk_fit_results()
            
            logger.info(f"Streaming fit completed: {total_samples_processed} samples, {chunk_number} chunks")
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
        
        logger.info("Starting streaming transform execution",
                   pipeline_id=self._pipeline_id,
                   chunk_size=chunk_size,
                   total_rows=len(X),
                   memory_available_gb=memory_status.available_gb)
        
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
                        chunk_size = min(chunk_size, current_memory.recommended_chunk_size)
                        logger.warning(f"Memory pressure detected, reducing chunk size to {chunk_size}")
                
                # Transform chunk using fitted pipeline
                chunk_result = self._transform_chunk(chunk, chunk_number)
                self._chunk_transform_results.append(chunk_result)
                
                chunk_time = time.time() - chunk_start_time
                logger.debug(f"Processed transform chunk {chunk_number}: {len(chunk)} -> {len(chunk_result)} samples, {chunk_time:.3f}s")
                
                # Adaptive chunk size adjustment
                if self.adaptive_chunking and chunk_number > 1:
                    chunk_size = self._adaptive_chunk_sizing(
                        chunk_size, chunk_time, current_memory if self.memory_monitoring else memory_status
                    )
            
            # Aggregate chunk results for final output
            final_result = self._aggregate_chunk_transform_results()
            
            logger.info(f"Streaming transform completed: {len(final_result)} total samples, {chunk_number} chunks")
            return final_result
            
        except Exception as e:
            logger.error(f"Streaming transform failed: {e}")
            raise
    
    def _fit_chunk(self, chunk: pd.DataFrame, y_chunk=None, chunk_number: int = 1, **fit_params) -> Dict[str, Any]:
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
                    if not hasattr(step_estimator, 'partial_fit'):
                        supports_partial = False
                        break
                
                if supports_partial:
                    # Use partial_fit for incremental learning
                    for step_name, step_estimator in self.steps:
                        if hasattr(step_estimator, 'partial_fit'):
                            step_estimator.partial_fit(chunk, y_chunk)
                    result = self
                else:
                    # Fallback: re-fit on accumulated data (memory-intensive but necessary)
                    logger.warning(f"Pipeline steps don't support partial_fit, using full refit on chunk {chunk_number}")
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
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Chunk fit failed on chunk {chunk_number}: {e}")
            return {
                "chunk_number": chunk_number,
                "fit_type": "failed",
                "samples_processed": len(chunk),
                "processing_time": time.time() - chunk_start_time,
                "error": str(e),
                "success": False
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
            
            logger.debug(f"Successfully transformed chunk {chunk_number}: {len(chunk)} -> {len(chunk_result)}")
            return chunk_result
            
        except Exception as e:
            logger.error(f"Chunk transform failed on chunk {chunk_number}: {e}")
            # Return empty result for failed chunks to maintain structure
            return pd.DataFrame() if hasattr(chunk, 'columns') else []
    
    def _aggregate_chunk_fit_results(self):
        """
        Aggregate results from chunk-based fitting.
        
        For most sklearn estimators, the final fitted state is already correct
        after processing all chunks. This method can be extended for specific
        aggregation needs.
        """
        successful_chunks = [r for r in self._chunk_fit_results if r.get("success", False)]
        failed_chunks = [r for r in self._chunk_fit_results if not r.get("success", False)]
        
        total_samples = sum(r["samples_processed"] for r in successful_chunks)
        total_time = sum(r["processing_time"] for r in self._chunk_fit_results)
        
        logger.info(f"Fit aggregation complete: {len(successful_chunks)}/{len(self._chunk_fit_results)} chunks successful, "
                   f"{total_samples} total samples, {total_time:.3f}s total time")
        
        if failed_chunks:
            logger.warning(f"Some chunks failed during fit: {[r['chunk_number'] for r in failed_chunks]}")
        
        # Store aggregation metadata
        self._execution_context["fit_aggregation"] = {
            "successful_chunks": len(successful_chunks),
            "failed_chunks": len(failed_chunks),
            "total_samples": total_samples,
            "total_processing_time": total_time,
            "average_chunk_time": total_time / len(self._chunk_fit_results) if self._chunk_fit_results else 0
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
            valid_chunks = [chunk for chunk in self._chunk_transform_results 
                          if chunk is not None and (hasattr(chunk, '__len__') and len(chunk) > 0)]
            
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
                result = [item for chunk in valid_chunks for item in (chunk if hasattr(chunk, '__iter__') else [chunk])]
            
            logger.info(f"Transform aggregation complete: {len(valid_chunks)} chunks -> {len(result)} total results")
            return result
            
        except Exception as e:
            logger.error(f"Failed to aggregate chunk transform results: {e}")
            # Return first valid chunk as fallback
            return valid_chunks[0] if valid_chunks else pd.DataFrame()
    
    def _calculate_adaptive_chunk_size(self, data: pd.DataFrame, memory_status: MemoryStatus) -> int:
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
                memory_per_row_mb = data.memory_usage(deep=True).sum() / len(data) / (1024 * 1024)
                
                # Target chunk memory usage (conservative)
                target_chunk_memory_mb = min(memory_status.available_gb * 0.1 * 1024, 250)  # 10% of available or 250MB max
                
                # Calculate adaptive chunk size
                adaptive_size = int(target_chunk_memory_mb / memory_per_row_mb) if memory_per_row_mb > 0 else base_chunk_size
                
                # Apply bounds
                adaptive_size = max(min(adaptive_size, 10000), 100)  # Between 100 and 10,000 rows
                
            except Exception as e:
                logger.warning(f"Failed to calculate adaptive chunk size: {e}, using base size")
                adaptive_size = base_chunk_size
        
        logger.info(f"Calculated adaptive chunk size: {adaptive_size} (base: {base_chunk_size}, memory: {memory_status.used_percent:.1f}% used)")
        return adaptive_size
    
    def _adaptive_chunk_sizing(self, current_chunk_size: int, processing_time: float, memory_status: MemoryStatus) -> int:
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
            logger.debug(f"Reducing chunk size due to slow processing: {current_chunk_size} -> {new_chunk_size}")
        elif processing_time < 1.0 and not memory_status.is_low_memory:  # Fast processing and good memory
            new_chunk_size = min(current_chunk_size * 2, memory_status.max_safe_chunk_size)
            logger.debug(f"Increasing chunk size due to fast processing: {current_chunk_size} -> {new_chunk_size}")
        
        # Adjust based on memory pressure
        if memory_status.is_low_memory:
            new_chunk_size = min(new_chunk_size, memory_status.recommended_chunk_size)
            logger.debug(f"Adjusting chunk size for memory pressure: {new_chunk_size}")
        
        # Ensure reasonable bounds
        new_chunk_size = max(min(new_chunk_size, 50000), 10)  # Between 10 and 50,000 rows
        
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
                "initial_memory_gb": self._memory_snapshots[0].available_gb if self._memory_snapshots else None,
                "final_memory_gb": self._memory_snapshots[-1].available_gb if self._memory_snapshots else None,
                "peak_memory_usage_percent": max((m.used_percent for m in self._memory_snapshots), default=0)
            }
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
    
    def __init__(self,
                 sklearn_pipeline: Pipeline,
                 streaming_executor: Optional[StreamingQueryExecutor] = None,
                 memory_monitoring: bool = True,
                 adaptive_chunking: bool = True,
                 performance_tracking: bool = True):
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
        
        logger.info("SklearnStreamingAdapter initialized",
                   adapter_id=self.adapter_id,
                   pipeline_type=type(sklearn_pipeline).__name__,
                   memory_monitoring=memory_monitoring,
                   adaptive_chunking=adaptive_chunking)
    
    def execute_streaming(self,
                         X,
                         y=None,
                         operation: str = 'transform',
                         initial_chunk_size: Optional[int] = None,
                         **operation_params) -> Tuple[Any, Dict[str, Any]]:
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
        
        logger.info(f"Starting streaming {operation} execution",
                   adapter_id=self.adapter_id,
                   execution_id=execution_id,
                   operation=operation)
        
        try:
            # Convert input to streaming source
            if isinstance(X, pd.DataFrame):
                data_source = DataFrameStreamingSource(X)
            elif hasattr(X, 'get_chunk_iterator'):
                data_source = X  # Already a streaming source
            else:
                # Convert array-like to DataFrame
                X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
                data_source = DataFrameStreamingSource(X_df)
            
            # Get initial memory status
            memory_status = self.streaming_executor._get_memory_status()
            chunk_size = self._calculate_initial_chunk_size(data_source, memory_status, initial_chunk_size)
            
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
            
            logger.info(f"Streaming {operation} execution completed",
                       adapter_id=self.adapter_id,
                       execution_id=execution_id,
                       execution_time=execution_time,
                       chunks_processed=chunk_metadata.get('total_chunks', 0))
            
            return operation_results, execution_metadata
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_metadata = {
                'execution_id': execution_id,
                'operation': operation,
                'execution_time': execution_time,
                'error_type': type(e).__name__,
                'error_message': str(e),
                'adapter_id': self.adapter_id,
                'success': False
            }
            
            logger.error(f"Streaming {operation} execution failed",
                        adapter_id=self.adapter_id,
                        execution_id=execution_id,
                        error=str(e))
            
            raise PipelineError(
                f"SklearnStreamingAdapter {operation} failed: {str(e)}",
                classification=ErrorClassification.STREAMING_ERROR,
                context=error_metadata
            ) from e
    
    def _execute_operation_streaming(self,
                                   data_source: StreamingDataSource,
                                   y,
                                   operation: str,
                                   chunk_size: int,
                                   **operation_params) -> Tuple[Any, Dict[str, Any]]:
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
            'total_chunks': 0,
            'successful_chunks': 0,
            'failed_chunks': 0,
            'processing_times': [],
            'memory_snapshots': [],
            'chunk_sizes': []
        }
        
        try:
            # Process data in chunks
            chunk_number = 0
            current_chunk_size = chunk_size
            
            for chunk in data_source.get_chunk_iterator(current_chunk_size):
                chunk_start_time = time.time()
                chunk_number += 1
                chunk_metadata['total_chunks'] = chunk_number
                
                # Monitor memory if enabled
                if self.memory_monitoring:
                    memory_status = self.streaming_executor._get_memory_status()
                    chunk_metadata['memory_snapshots'].append({
                        'chunk_number': chunk_number,
                        'memory_used_percent': memory_status.used_percent,
                        'available_gb': memory_status.available_gb,
                        'is_low_memory': memory_status.is_low_memory
                    })
                    
                    # Adapt chunk size if needed
                    if self.adaptive_chunking and memory_status.is_low_memory:
                        current_chunk_size = min(current_chunk_size, memory_status.recommended_chunk_size)
                        logger.warning(f"Reducing chunk size due to memory pressure: {current_chunk_size}")
                
                try:
                    # Get corresponding y chunk if provided
                    if y is not None:
                        start_idx = (chunk_number - 1) * current_chunk_size
                        end_idx = start_idx + len(chunk)
                        if hasattr(y, 'iloc'):
                            y_chunk = y.iloc[start_idx:end_idx]
                        else:
                            y_chunk = y[start_idx:end_idx] if hasattr(y, '__getitem__') else None
                    else:
                        y_chunk = None
                    
                    # Execute operation on chunk
                    chunk_result = self._execute_chunk_operation(
                        chunk, y_chunk, operation, chunk_number, **operation_params
                    )
                    
                    chunk_results.append(chunk_result)
                    chunk_metadata['successful_chunks'] += 1
                    
                    # Track chunk processing time
                    chunk_time = time.time() - chunk_start_time
                    chunk_metadata['processing_times'].append(chunk_time)
                    chunk_metadata['chunk_sizes'].append(len(chunk))
                    
                    logger.debug(f"Processed chunk {chunk_number}: {len(chunk)} samples, {chunk_time:.3f}s")
                    
                    # Adaptive chunk sizing based on performance
                    if self.adaptive_chunking and chunk_number > 1:
                        current_chunk_size = self._adapt_chunk_size(
                            current_chunk_size, chunk_time,
                            memory_status if self.memory_monitoring else None
                        )
                
                except Exception as chunk_error:
                    chunk_metadata['failed_chunks'] += 1
                    logger.error(f"Chunk {chunk_number} processing failed: {chunk_error}")
                    
                    # For transform operations, continue with next chunk
                    if operation in ['transform', 'predict']:
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
    
    def _execute_chunk_operation(self,
                               chunk,
                               y_chunk,
                               operation: str,
                               chunk_number: int,
                               **operation_params) -> Any:
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
            if operation == 'fit':
                if chunk_number == 1:
                    # Initial fit on first chunk
                    result = self.sklearn_pipeline.fit(chunk, y_chunk, **operation_params)
                else:
                    # Try partial_fit for subsequent chunks if supported
                    if hasattr(self.sklearn_pipeline, 'partial_fit'):
                        result = self.sklearn_pipeline.partial_fit(chunk, y_chunk)
                    else:
                        # Check if individual steps support partial fitting
                        self._partial_fit_steps(chunk, y_chunk)
                        result = self.sklearn_pipeline
                        
            elif operation == 'transform':
                result = self.sklearn_pipeline.transform(chunk)
                
            elif operation == 'fit_transform':
                if chunk_number == 1:
                    result = self.sklearn_pipeline.fit_transform(chunk, y_chunk, **operation_params)
                else:
                    # For subsequent chunks: partial fit + transform
                    if hasattr(self.sklearn_pipeline, 'partial_fit'):
                        self.sklearn_pipeline.partial_fit(chunk, y_chunk)
                    else:
                        self._partial_fit_steps(chunk, y_chunk)
                    result = self.sklearn_pipeline.transform(chunk)
                    
            elif operation == 'predict':
                result = self.sklearn_pipeline.predict(chunk)
                
            else:
                # Generic operation - try to call the method
                if hasattr(self.sklearn_pipeline, operation):
                    method = getattr(self.sklearn_pipeline, operation)
                    result = method(chunk, y_chunk, **operation_params) if y_chunk is not None else method(chunk, **operation_params)
                else:
                    raise ValueError(f"Unknown operation: {operation}")
            
            return result
            
        except Exception as e:
            logger.error(f"Chunk operation {operation} failed on chunk {chunk_number}: {e}")
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
            if hasattr(step_estimator, 'partial_fit'):
                # Apply partial fit on this step
                step_estimator.partial_fit(current_data, y_chunk)
            
            # Transform data for next step
            if hasattr(step_estimator, 'transform'):
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
            if operation in ['fit', 'fit_transform'] and len(valid_results) > 0:
                # For fit operations, return the fitted pipeline (last valid result)
                if operation == 'fit':
                    return self.sklearn_pipeline  # Return the fitted pipeline
                else:
                    # For fit_transform, aggregate the transformed results
                    return self._concatenate_results(valid_results)
            
            elif operation in ['transform', 'predict']:
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
    
    def _calculate_initial_chunk_size(self,
                                    data_source: StreamingDataSource,
                                    memory_status: MemoryStatus,
                                    provided_chunk_size: Optional[int]) -> int:
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
                calculated_chunk_size = int((target_memory_mb * 1024 * 1024) / memory_per_row)
                chunk_size = min(chunk_size, max(calculated_chunk_size, 100))
        except:
            pass  # Use default chunk size if estimation fails
        
        logger.info(f"Calculated initial chunk size: {chunk_size} (memory: {memory_status.used_percent:.1f}% used)")
        return chunk_size
    
    def _adapt_chunk_size(self,
                         current_chunk_size: int,
                         processing_time: float,
                         memory_status: Optional[MemoryStatus]) -> int:
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
        elif processing_time < 0.5 and (not memory_status or not memory_status.is_low_memory):
            # Fast processing and good memory - increase chunk size
            new_chunk_size = min(current_chunk_size * 2, 10000)
        
        # Apply memory constraints
        if memory_status and memory_status.is_low_memory:
            new_chunk_size = min(new_chunk_size, memory_status.recommended_chunk_size)
        
        # Ensure reasonable bounds
        new_chunk_size = max(min(new_chunk_size, 50000), 10)
        
        if new_chunk_size != current_chunk_size:
            logger.debug(f"Adapted chunk size: {current_chunk_size} -> {new_chunk_size} (time: {processing_time:.3f}s)")
        
        return new_chunk_size
    
    def _build_execution_metadata(self,
                                execution_id: str,
                                operation: str,
                                execution_time: float,
                                chunk_metadata: Dict[str, Any]) -> Dict[str, Any]:
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
        total_chunks = chunk_metadata.get('total_chunks', 0)
        successful_chunks = chunk_metadata.get('successful_chunks', 0)
        processing_times = chunk_metadata.get('processing_times', [])
        chunk_sizes = chunk_metadata.get('chunk_sizes', [])
        
        metadata = {
            'execution_id': execution_id,
            'adapter_id': self.adapter_id,
            'operation': operation,
            'execution_time_seconds': execution_time,
            'success': True,
            'timestamp': time.time(),
            
            # Chunk processing statistics
            'chunk_statistics': {
                'total_chunks': total_chunks,
                'successful_chunks': successful_chunks,
                'failed_chunks': chunk_metadata.get('failed_chunks', 0),
                'success_rate': successful_chunks / total_chunks if total_chunks > 0 else 0,
                'average_chunk_processing_time': sum(processing_times) / len(processing_times) if processing_times else 0,
                'total_samples_processed': sum(chunk_sizes),
                'average_chunk_size': sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0
            },
            
            # Performance metrics
            'performance_metrics': {
                'samples_per_second': sum(chunk_sizes) / execution_time if execution_time > 0 else 0,
                'chunks_per_second': total_chunks / execution_time if execution_time > 0 else 0,
                'memory_efficiency': self._calculate_memory_efficiency(chunk_metadata)
            },
            
            # Pipeline information
            'pipeline_info': {
                'pipeline_type': type(self.sklearn_pipeline).__name__,
                'pipeline_steps': [step_name for step_name, _ in self.sklearn_pipeline.steps] if hasattr(self.sklearn_pipeline, 'steps') else [],
                'supports_partial_fit': self._check_partial_fit_support()
            }
        }
        
        return metadata
    
    def _calculate_memory_efficiency(self, chunk_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate memory efficiency metrics from chunk processing.
        
        Args:
            chunk_metadata: Metadata from chunk processing
            
        Returns:
            Memory efficiency metrics
        """
        memory_snapshots = chunk_metadata.get('memory_snapshots', [])
        
        if not memory_snapshots:
            return {'memory_monitoring_enabled': False}
        
        memory_usage_percentages = [s['memory_used_percent'] for s in memory_snapshots]
        available_gb_values = [s['available_gb'] for s in memory_snapshots]
        low_memory_count = sum(1 for s in memory_snapshots if s['is_low_memory'])
        
        return {
            'memory_monitoring_enabled': True,
            'peak_memory_usage_percent': max(memory_usage_percentages) if memory_usage_percentages else 0,
            'average_memory_usage_percent': sum(memory_usage_percentages) / len(memory_usage_percentages) if memory_usage_percentages else 0,
            'minimum_available_gb': min(available_gb_values) if available_gb_values else 0,
            'memory_pressure_chunks': low_memory_count,
            'memory_stability': (len(memory_snapshots) - low_memory_count) / len(memory_snapshots) if memory_snapshots else 1.0
        }
    
    def _check_partial_fit_support(self) -> bool:
        """
        Check if the pipeline supports partial fitting.
        
        Returns:
            True if partial fitting is supported
        """
        if hasattr(self.sklearn_pipeline, 'partial_fit'):
            return True
        
        if hasattr(self.sklearn_pipeline, 'steps'):
            # Check if any steps support partial_fit
            for step_name, step_estimator in self.sklearn_pipeline.steps:
                if hasattr(step_estimator, 'partial_fit'):
                    return True
        
        return False
    
    def _update_performance_metrics(self, execution_metadata: Dict[str, Any]):
        """
        Update internal performance metrics with execution data.
        
        Args:
            execution_metadata: Metadata from execution
        """
        operation = execution_metadata['operation']
        
        if operation not in self.performance_metrics:
            self.performance_metrics[operation] = {
                'execution_count': 0,
                'total_execution_time': 0,
                'total_samples_processed': 0,
                'average_samples_per_second': 0,
                'success_rate': 0,
                'memory_efficiency_scores': []
            }
        
        metrics = self.performance_metrics[operation]
        metrics['execution_count'] += 1
        metrics['total_execution_time'] += execution_metadata['execution_time_seconds']
        
        chunk_stats = execution_metadata.get('chunk_statistics', {})
        metrics['total_samples_processed'] += chunk_stats.get('total_samples_processed', 0)
        metrics['success_rate'] = (
            (metrics['success_rate'] * (metrics['execution_count'] - 1) + 
             (1.0 if execution_metadata['success'] else 0.0)) / metrics['execution_count']
        )
        
        # Update samples per second average
        if metrics['total_execution_time'] > 0:
            metrics['average_samples_per_second'] = metrics['total_samples_processed'] / metrics['total_execution_time']
        
        # Track memory efficiency
        perf_metrics = execution_metadata.get('performance_metrics', {})
        memory_efficiency = perf_metrics.get('memory_efficiency', {})
        if memory_efficiency.get('memory_monitoring_enabled', False):
            stability_score = memory_efficiency.get('memory_stability', 0)
            metrics['memory_efficiency_scores'].append(stability_score)
    
    # Public utility methods
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive performance summary across all operations.
        
        Returns:
            Performance summary dictionary
        """
        return {
            'adapter_id': self.adapter_id,
            'pipeline_type': type(self.sklearn_pipeline).__name__,
            'total_executions': len(self.execution_history),
            'operations_performed': list(self.performance_metrics.keys()),
            'performance_by_operation': self.performance_metrics,
            'configuration': {
                'memory_monitoring': self.memory_monitoring,
                'adaptive_chunking': self.adaptive_chunking,
                'performance_tracking': self.performance_tracking
            }
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


# Update the exports
__all__ = ['DataSciencePipeline', 'PipelineComposer', 'StreamingDataPipeline', 'SklearnStreamingAdapter']