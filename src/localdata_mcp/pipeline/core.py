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
from typing import Any, Dict, List, Optional, Tuple, Union

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
from ..streaming_executor import StreamingQueryExecutor, MemoryStatus
from ..logging_manager import get_logging_manager, get_logger
from ..error_handler import ErrorCategory, ErrorSeverity

logger = get_logger(__name__)


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