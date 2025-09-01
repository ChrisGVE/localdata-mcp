"""
Pipeline Data Flow Engine - Concrete Implementation

This module implements the data flow architecture for pipeline compositions,
including automatic type conversion, streaming support, and metadata preservation.
"""

from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import json
import logging
from datetime import datetime
import hashlib
import sys

from pipeline_composition_framework import (
    CompositionStage,
    AnalysisComposition, 
    ConversionResult,
    CompositionError
)

logger = logging.getLogger(__name__)


# ============================================================================
# Core Data Flow Components
# ============================================================================

@dataclass
class CompositionContext:
    """
    Execution context for pipeline compositions.
    
    Manages data flow, intermediate results, and metadata across stages.
    """
    composition_id: str
    stage_outputs: Dict[int, Any] = field(default_factory=dict)
    stage_metadata: Dict[int, Dict] = field(default_factory=dict)
    performance_stats: Dict[str, Any] = field(default_factory=dict)
    execution_start: datetime = field(default_factory=datetime.now)
    
    def set_stage_output(self, stage_index: int, output: Any, metadata: Optional[Dict] = None):
        """Store output from a stage."""
        self.stage_outputs[stage_index] = output
        if metadata:
            self.stage_metadata[stage_index] = metadata
    
    def get_stage_output(self, stage_index: int) -> Any:
        """Get output from a specific stage."""
        return self.stage_outputs.get(stage_index)
    
    def get_final_output(self) -> Any:
        """Get output from the last stage."""
        if not self.stage_outputs:
            return None
        last_stage = max(self.stage_outputs.keys())
        return self.stage_outputs[last_stage]
    
    def get_composition_metadata(self) -> Dict[str, Any]:
        """Get comprehensive metadata about the composition execution."""
        execution_duration = (datetime.now() - self.execution_start).total_seconds()
        
        return {
            'composition_id': self.composition_id,
            'execution_duration_seconds': execution_duration,
            'stages_completed': len(self.stage_outputs),
            'stage_metadata': self.stage_metadata,
            'performance_stats': self.performance_stats
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return self.performance_stats
    
    def cleanup_intermediate_results(self):
        """Clean up intermediate results to free memory."""
        # Keep only the last few stages for debugging
        if len(self.stage_outputs) > 3:
            stages_to_keep = sorted(self.stage_outputs.keys())[-2:]  # Keep last 2 stages
            
            for stage_index in list(self.stage_outputs.keys()):
                if stage_index not in stages_to_keep:
                    del self.stage_outputs[stage_index]
                    if stage_index in self.stage_metadata:
                        del self.stage_metadata[stage_index]


@dataclass  
class CompositionResult:
    """Result of pipeline composition execution."""
    final_result: Any
    metadata: Dict[str, Any]
    performance_stats: Dict[str, Any]
    success: bool = True
    error_message: Optional[str] = None
    partial_results: Dict[int, Any] = field(default_factory=dict)


# ============================================================================
# Type Conversion Engine
# ============================================================================

class TypeConversionEngine:
    """
    Handles automatic conversion between incompatible data types in pipeline compositions.
    
    Supports conversions between pandas, numpy, sklearn, scipy, and other data science libraries.
    """
    
    def __init__(self):
        self.conversion_registry = self._initialize_conversions()
        self.metadata_extractors = self._initialize_metadata_extractors()
    
    def convert(self, data: Any, from_type: str, to_type: str, context: Dict = None) -> ConversionResult:
        """
        Convert data from one type to another with metadata preservation.
        
        Args:
            data: Source data to convert
            from_type: Source data type identifier  
            to_type: Target data type identifier
            context: Additional context for conversion
            
        Returns:
            ConversionResult with converted data or error information
        """
        conversion_key = (from_type, to_type)
        context = context or {}
        
        if conversion_key not in self.conversion_registry:
            return ConversionResult(
                success=False,
                error=f"No conversion path from {from_type} to {to_type}"
            )
        
        converter_func = self.conversion_registry[conversion_key]
        
        try:
            converted_data = converter_func(data, context)
            
            # Extract metadata from both original and converted data
            metadata = self._extract_conversion_metadata(data, converted_data, from_type, to_type)
            
            return ConversionResult(
                success=True,
                data=converted_data,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Conversion failed from {from_type} to {to_type}: {str(e)}")
            return ConversionResult(
                success=False,
                error=f"Conversion failed: {str(e)}"
            )
    
    def _initialize_conversions(self) -> Dict:
        """Initialize conversion function registry."""
        return {
            ('pandas.DataFrame', 'numpy.ndarray'): self._dataframe_to_array,
            ('numpy.ndarray', 'pandas.DataFrame'): self._array_to_dataframe,
            ('sklearn.base.BaseEstimator', 'dict'): self._model_to_dict,
            ('scipy.optimize.OptimizeResult', 'pandas.DataFrame'): self._optimize_result_to_dataframe,
            ('dict', 'pandas.DataFrame'): self._dict_to_dataframe,
            ('pandas.DataFrame', 'dict'): self._dataframe_to_dict,
            ('list', 'numpy.ndarray'): self._list_to_array,
            ('numpy.ndarray', 'list'): self._array_to_list,
            ('pandas.Series', 'numpy.ndarray'): self._series_to_array,
            ('numpy.ndarray', 'pandas.Series'): self._array_to_series
        }
    
    def _initialize_metadata_extractors(self) -> Dict:
        """Initialize metadata extraction functions."""
        return {
            'pandas.DataFrame': self._extract_dataframe_metadata,
            'numpy.ndarray': self._extract_array_metadata,
            'sklearn.base.BaseEstimator': self._extract_model_metadata,
            'dict': self._extract_dict_metadata
        }
    
    # ========================================================================
    # Conversion Functions
    # ========================================================================
    
    def _dataframe_to_array(self, df: pd.DataFrame, context: Dict) -> np.ndarray:
        """Convert DataFrame to numpy array for sklearn/scipy tools."""
        # Handle mixed types by separating numeric and categorical
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == len(df.columns):
            # All numeric - direct conversion
            return df.values
        
        # Mixed types - need encoding strategy
        encoded_df = df.copy()
        
        # Encode categorical columns
        for col in df.select_dtypes(include=['object', 'category']).columns:
            if context.get('preserve_categories', False):
                # Use label encoding to preserve category information
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                encoded_df[col] = le.fit_transform(df[col].astype(str))
            else:
                # Simple numeric encoding
                unique_values = df[col].unique()
                value_map = {val: idx for idx, val in enumerate(unique_values)}
                encoded_df[col] = df[col].map(value_map)
        
        return encoded_df.values
    
    def _array_to_dataframe(self, arr: np.ndarray, context: Dict) -> pd.DataFrame:
        """Convert numpy array to DataFrame with appropriate column names."""
        # Generate column names
        if 'column_names' in context:
            columns = context['column_names']
        elif arr.ndim == 2:
            columns = [f'feature_{i}' for i in range(arr.shape[1])]
        else:
            columns = ['value']
        
        if arr.ndim == 1:
            return pd.DataFrame(arr, columns=['value'])
        else:
            return pd.DataFrame(arr, columns=columns[:arr.shape[1]])
    
    def _model_to_dict(self, model, context: Dict) -> Dict:
        """Convert sklearn model to dictionary representation."""
        model_dict = {
            'model_type': type(model).__name__,
            'model_module': type(model).__module__
        }
        
        # Extract model parameters
        if hasattr(model, 'get_params'):
            model_dict['parameters'] = model.get_params()
        
        # Extract fitted attributes if model is trained
        if hasattr(model, 'coef_'):
            model_dict['coefficients'] = model.coef_.tolist() if hasattr(model.coef_, 'tolist') else str(model.coef_)
        
        if hasattr(model, 'feature_importances_'):
            model_dict['feature_importances'] = model.feature_importances_.tolist()
        
        if hasattr(model, 'intercept_'):
            model_dict['intercept'] = float(model.intercept_) if np.isscalar(model.intercept_) else model.intercept_.tolist()
        
        # Training metrics if available
        if hasattr(model, 'score_'):
            model_dict['training_score'] = float(model.score_)
        
        return model_dict
    
    def _optimize_result_to_dataframe(self, result, context: Dict) -> pd.DataFrame:
        """Convert scipy optimization result to DataFrame."""
        result_dict = {
            'optimization_success': [result.success],
            'optimization_message': [result.message if hasattr(result, 'message') else 'No message'],
            'function_value': [float(result.fun) if hasattr(result, 'fun') else None],
            'iterations': [int(result.nit) if hasattr(result, 'nit') else None],
            'function_evaluations': [int(result.nfev) if hasattr(result, 'nfev') else None]
        }
        
        # Add parameter values if available
        if hasattr(result, 'x') and result.x is not None:
            if np.isscalar(result.x):
                result_dict['optimal_parameter'] = [float(result.x)]
            else:
                for i, param_value in enumerate(result.x):
                    result_dict[f'parameter_{i}'] = [float(param_value)]
        
        return pd.DataFrame(result_dict)
    
    def _dict_to_dataframe(self, data_dict: Dict, context: Dict) -> pd.DataFrame:
        """Convert dictionary to DataFrame."""
        if not data_dict:
            return pd.DataFrame()
        
        # Handle different dictionary structures
        if all(isinstance(v, (list, tuple, np.ndarray)) for v in data_dict.values()):
            # Dictionary of arrays - each key becomes a column
            return pd.DataFrame(data_dict)
        
        elif all(isinstance(v, (int, float, str, bool)) for v in data_dict.values()):
            # Dictionary of scalars - create single row DataFrame  
            return pd.DataFrame([data_dict])
        
        else:
            # Mixed types - convert to string representation
            converted_dict = {}
            for key, value in data_dict.items():
                if isinstance(value, (list, tuple, np.ndarray)):
                    converted_dict[key] = value
                else:
                    converted_dict[key] = [str(value)]
            
            return pd.DataFrame(converted_dict)
    
    def _dataframe_to_dict(self, df: pd.DataFrame, context: Dict) -> Dict:
        """Convert DataFrame to dictionary."""
        if context.get('orient', 'records') == 'records':
            return df.to_dict('records')
        elif context.get('orient') == 'dict':
            return df.to_dict()
        else:
            return df.to_dict('records')  # Default
    
    def _list_to_array(self, data_list: List, context: Dict) -> np.ndarray:
        """Convert list to numpy array."""
        return np.array(data_list)
    
    def _array_to_list(self, arr: np.ndarray, context: Dict) -> List:
        """Convert numpy array to list."""
        return arr.tolist()
    
    def _series_to_array(self, series: pd.Series, context: Dict) -> np.ndarray:
        """Convert pandas Series to numpy array."""
        return series.values
    
    def _array_to_series(self, arr: np.ndarray, context: Dict) -> pd.Series:
        """Convert numpy array to pandas Series."""
        name = context.get('series_name', 'values')
        return pd.Series(arr, name=name)
    
    # ========================================================================
    # Metadata Extraction Functions
    # ========================================================================
    
    def _extract_conversion_metadata(self, original_data: Any, converted_data: Any, 
                                    from_type: str, to_type: str) -> Dict[str, Any]:
        """Extract metadata from conversion process."""
        metadata = {
            'conversion': {
                'from_type': from_type,
                'to_type': to_type,
                'conversion_timestamp': datetime.now().isoformat()
            }
        }
        
        # Extract metadata from original data
        if from_type in self.metadata_extractors:
            original_metadata = self.metadata_extractors[from_type](original_data)
            metadata['original_data'] = original_metadata
        
        # Extract metadata from converted data
        if to_type in self.metadata_extractors:
            converted_metadata = self.metadata_extractors[to_type](converted_data)
            metadata['converted_data'] = converted_metadata
        
        return metadata
    
    def _extract_dataframe_metadata(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Extract metadata from pandas DataFrame."""
        return {
            'type': 'pandas.DataFrame',
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
            'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 * 1024),
            'null_counts': df.isnull().sum().to_dict(),
            'numeric_columns': list(df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(df.select_dtypes(include=['object', 'category']).columns)
        }
    
    def _extract_array_metadata(self, arr: np.ndarray) -> Dict[str, Any]:
        """Extract metadata from numpy array."""
        return {
            'type': 'numpy.ndarray',
            'shape': arr.shape,
            'dtype': str(arr.dtype),
            'memory_usage_mb': arr.nbytes / (1024 * 1024),
            'min_value': float(np.min(arr)) if arr.size > 0 else None,
            'max_value': float(np.max(arr)) if arr.size > 0 else None,
            'mean_value': float(np.mean(arr)) if arr.size > 0 else None
        }
    
    def _extract_model_metadata(self, model) -> Dict[str, Any]:
        """Extract metadata from sklearn model."""
        metadata = {
            'type': 'sklearn.base.BaseEstimator',
            'model_class': type(model).__name__,
            'model_module': type(model).__module__
        }
        
        if hasattr(model, 'n_features_in_'):
            metadata['n_features'] = int(model.n_features_in_)
        
        if hasattr(model, 'feature_names_in_'):
            metadata['feature_names'] = list(model.feature_names_in_)
        
        return metadata
    
    def _extract_dict_metadata(self, data_dict: Dict) -> Dict[str, Any]:
        """Extract metadata from dictionary."""
        return {
            'type': 'dict',
            'keys': list(data_dict.keys()),
            'key_count': len(data_dict),
            'value_types': {key: type(value).__name__ for key, value in data_dict.items()},
            'memory_usage_mb': sys.getsizeof(data_dict) / (1024 * 1024)
        }


# ============================================================================
# Pipeline Data Flow Manager
# ============================================================================

class PipelineDataFlow:
    """
    Manages data flow between pipeline stages with automatic conversion and streaming support.
    
    Key Features:
    - Automatic type conversion between incompatible stages
    - Memory-efficient streaming for large datasets  
    - Rich metadata preservation through compositions
    - Performance optimization with intelligent caching
    """
    
    def __init__(self, memory_manager, conversion_engine: Optional[TypeConversionEngine] = None):
        self.memory_manager = memory_manager
        self.conversion_engine = conversion_engine or TypeConversionEngine()
        self.execution_stats = {}
    
    def execute_composition(self, composition: AnalysisComposition, 
                          tool_executor: 'ToolExecutor') -> CompositionResult:
        """
        Execute a validated composition with streaming data flow.
        
        Args:
            composition: Validated pipeline composition
            tool_executor: Executor for individual tools
            
        Returns:
            CompositionResult with final results and metadata
        """
        context = CompositionContext(composition_id=composition.composition_id)
        
        try:
            for stage_index, stage in enumerate(composition.stages):
                logger.info(f"Executing stage {stage_index}: {stage.tool_name}")
                
                # Get input data from previous stage or initial data
                if stage_index == 0:
                    input_data = composition.initial_data
                else:
                    input_data = context.get_stage_output(stage_index - 1)
                
                # Apply automatic type conversion if needed
                if stage.requires_conversion and stage.conversion_path:
                    conversion_result = self._apply_conversion(input_data, stage, context)
                    
                    if not conversion_result.success:
                        raise CompositionError(
                            f"Type conversion failed at stage {stage_index}: {conversion_result.error}",
                            error_type='type_conversion_error',
                            stage_index=stage_index
                        )
                    
                    input_data = conversion_result.data
                    
                    # Store conversion metadata
                    stage_metadata = context.stage_metadata.get(stage_index, {})
                    stage_metadata['conversion'] = conversion_result.metadata
                    context.stage_metadata[stage_index] = stage_metadata
                
                # Execute stage with streaming support
                stage_start = datetime.now()
                stage_result = self._execute_stage_streaming(stage, input_data, tool_executor, context)
                stage_duration = (datetime.now() - stage_start).total_seconds()
                
                # Store result and performance stats
                context.set_stage_output(stage_index, stage_result.data)
                context.performance_stats[f'stage_{stage_index}_duration'] = stage_duration
                context.performance_stats[f'stage_{stage_index}_tool'] = stage.tool_name
                
                # Memory management
                if self.memory_manager.should_cleanup(context):
                    context.cleanup_intermediate_results()
            
            return CompositionResult(
                final_result=context.get_final_output(),
                metadata=context.get_composition_metadata(), 
                performance_stats=context.get_performance_stats(),
                success=True
            )
            
        except CompositionError as e:
            logger.error(f"Composition execution failed: {str(e)}")
            return CompositionResult(
                final_result=context.get_final_output(),
                metadata=context.get_composition_metadata(),
                performance_stats=context.get_performance_stats(),
                success=False,
                error_message=str(e),
                partial_results=context.stage_outputs
            )
        
        except Exception as e:
            logger.error(f"Unexpected error in composition execution: {str(e)}")
            return CompositionResult(
                final_result=None,
                metadata=context.get_composition_metadata(),
                performance_stats=context.get_performance_stats(), 
                success=False,
                error_message=f"Unexpected error: {str(e)}",
                partial_results=context.stage_outputs
            )
    
    def _apply_conversion(self, data: Any, stage: CompositionStage, 
                         context: CompositionContext) -> ConversionResult:
        """Apply automatic type conversion for a stage."""
        # Determine source and target types
        source_type = self._infer_data_type(data)
        target_type = stage.expected_input_types[0] if stage.expected_input_types else 'unknown'
        
        # Prepare conversion context
        conversion_context = {
            'stage_index': len(context.stage_outputs),
            'tool_name': stage.tool_name,
            'preserve_categories': stage.parameters.get('preserve_categories', False),
            'column_names': getattr(data, 'columns', None) if hasattr(data, 'columns') else None
        }
        
        return self.conversion_engine.convert(data, source_type, target_type, conversion_context)
    
    def _execute_stage_streaming(self, stage: CompositionStage, input_data: Any,
                                tool_executor: 'ToolExecutor', 
                                context: CompositionContext) -> 'StageResult':
        """Execute a single stage with streaming support."""
        
        # For now, execute directly - streaming implementation would be added here
        # This is where we'd integrate with the streaming architecture from ARCHITECTURE.md
        
        result = tool_executor.execute_tool(
            tool_name=stage.tool_name,
            function=stage.function, 
            input_data=input_data,
            parameters=stage.parameters
        )
        
        return result
    
    def _infer_data_type(self, data: Any) -> str:
        """Infer data type from data object."""
        if isinstance(data, pd.DataFrame):
            return 'pandas.DataFrame'
        elif isinstance(data, np.ndarray):
            return 'numpy.ndarray'
        elif isinstance(data, pd.Series):
            return 'pandas.Series'
        elif isinstance(data, dict):
            return 'dict'
        elif isinstance(data, list):
            return 'list'
        else:
            return type(data).__name__


# ============================================================================
# Protocol Definitions
# ============================================================================

@dataclass
class StageResult:
    """Result of executing a single stage."""
    data: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error_message: Optional[str] = None


class ToolExecutor(Protocol):
    """Protocol for tool executors."""
    
    def execute_tool(self, tool_name: str, function: str, input_data: Any, 
                    parameters: Dict[str, Any]) -> StageResult:
        """Execute a tool with given parameters."""
        ...


# ============================================================================
# Example Usage
# ============================================================================

def create_example_data_flow():
    """Example of using the data flow engine."""
    from pipeline_composition_framework import create_example_composition
    
    # Create example composition
    composition = create_example_composition()
    
    # Initialize data flow components
    memory_manager = None  # Would be actual MemoryManager instance
    data_flow = PipelineDataFlow(memory_manager)
    
    # Execute composition (would need actual tool executor)
    # result = data_flow.execute_composition(composition, tool_executor)
    
    print(f"Data flow engine ready for composition: {composition.composition_id}")


if __name__ == "__main__":
    create_example_data_flow()