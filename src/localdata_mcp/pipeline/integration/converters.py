"""
Core Data Format Converters for LocalData MCP v2.0 Integration Shims Framework.

This module provides bidirectional data format conversion capabilities between
common data science formats with comprehensive metadata preservation and
memory-efficient processing.

Key Features:
- PandasConverter: DataFrame ↔ other formats with index/column preservation
- NumpyConverter: Array ↔ other formats with shape/dtype preservation  
- SparseMatrixConverter: Sparse matrices ↔ other formats with density management
- Streaming-first architecture for memory efficiency
- Sklearn-compatible fit/transform patterns
- Comprehensive error handling and quality scoring
"""

import logging
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union, Set
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .base_adapters import BaseShimAdapter, StreamingShimAdapter, ConversionContext
from .interfaces import (
    DataFormat, ConversionRequest, ConversionResult, ConversionError,
    MemoryConstraints, PerformanceRequirements
)
from .type_detection import TypeDetectionEngine, FormatDetectionResult
from .metadata_manager import MetadataManager, PreservationStrategy
from ...logging_manager import get_logger

logger = get_logger(__name__)


class ConversionQuality(Enum):
    """Quality levels for data conversion."""
    LOSSLESS = "lossless"           # No data loss
    HIGH_FIDELITY = "high_fidelity" # Minimal data loss
    MODERATE = "moderate"           # Some data loss acceptable
    LOW = "low"                     # Significant data loss


@dataclass
class ConversionOptions:
    """Options for controlling conversion behavior."""
    preserve_index: bool = True
    preserve_columns: bool = True
    handle_mixed_types: bool = True
    categorical_threshold: float = 0.1  # Unique ratio below which to treat as categorical
    sparse_density_threshold: float = 0.1  # Below which to use sparse representation
    chunk_size_rows: int = 10000
    memory_efficient: bool = True
    quality_target: ConversionQuality = ConversionQuality.HIGH_FIDELITY


class PandasConverter(BaseShimAdapter):
    """
    Bidirectional converter for pandas DataFrame format.
    
    Handles conversions between DataFrame and other formats while preserving
    index information, column names, data types, and categorical data.
    """
    
    def __init__(self,
                 adapter_id: str = "pandas_converter",
                 conversion_options: Optional[ConversionOptions] = None,
                 **kwargs):
        """
        Initialize PandasConverter.
        
        Args:
            adapter_id: Unique identifier for this converter
            conversion_options: Options controlling conversion behavior
            **kwargs: Additional arguments passed to BaseShimAdapter
        """
        # Define supported conversions
        supported_conversions = [
            # From DataFrame
            (DataFormat.PANDAS_DATAFRAME, DataFormat.NUMPY_ARRAY),
            (DataFormat.PANDAS_DATAFRAME, DataFormat.SCIPY_SPARSE),
            (DataFormat.PANDAS_DATAFRAME, DataFormat.PYTHON_DICT),
            (DataFormat.PANDAS_DATAFRAME, DataFormat.PYTHON_LIST),
            (DataFormat.PANDAS_DATAFRAME, DataFormat.TIME_SERIES),
            (DataFormat.PANDAS_DATAFRAME, DataFormat.CATEGORICAL),
            
            # To DataFrame
            (DataFormat.NUMPY_ARRAY, DataFormat.PANDAS_DATAFRAME),
            (DataFormat.SCIPY_SPARSE, DataFormat.PANDAS_DATAFRAME),
            (DataFormat.PYTHON_DICT, DataFormat.PANDAS_DATAFRAME),
            (DataFormat.PYTHON_LIST, DataFormat.PANDAS_DATAFRAME),
            (DataFormat.TIME_SERIES, DataFormat.PANDAS_DATAFRAME),
            (DataFormat.CATEGORICAL, DataFormat.PANDAS_DATAFRAME),
        ]
        
        super().__init__(
            adapter_id=adapter_id,
            supported_conversions=supported_conversions,
            **kwargs
        )
        
        self.conversion_options = conversion_options or ConversionOptions()
        
        # Initialize components
        self._type_detector = TypeDetectionEngine()
        self._metadata_manager = MetadataManager()
        
        # Conversion statistics
        self._conversion_stats = {
            'successful_conversions': 0,
            'failed_conversions': 0,
            'total_rows_processed': 0,
            'average_conversion_time': 0.0
        }
        
        logger.info(f"PandasConverter initialized",
                   adapter_id=adapter_id,
                   supported_conversions_count=len(supported_conversions))
    
    def _perform_conversion(self, request: ConversionRequest, 
                          context: ConversionContext) -> Any:
        """
        Perform the actual data conversion.
        
        Args:
            request: Conversion request with source data and target format
            context: Conversion context for tracking
            
        Returns:
            Converted data
        """
        source_format = request.source_format
        target_format = request.target_format
        source_data = request.source_data
        
        logger.debug(f"Converting from {source_format.value} to {target_format.value}")
        
        try:
            # Route to appropriate conversion method
            if source_format == DataFormat.PANDAS_DATAFRAME:
                return self._convert_from_dataframe(source_data, target_format, context)
            elif target_format == DataFormat.PANDAS_DATAFRAME:
                return self._convert_to_dataframe(source_data, source_format, context)
            else:
                raise ConversionError(
                    ConversionError.Type.CONVERSION_FAILED,
                    f"Unsupported conversion path: {source_format.value} -> {target_format.value}"
                )
        
        except Exception as e:
            self._conversion_stats['failed_conversions'] += 1
            logger.error(f"Conversion failed: {e}")
            raise ConversionError(
                ConversionError.Type.CONVERSION_FAILED,
                f"Pandas conversion failed: {str(e)}"
            )
    
    def _convert_from_dataframe(self, df: pd.DataFrame, 
                              target_format: DataFormat, 
                              context: ConversionContext) -> Any:
        """Convert DataFrame to other formats."""
        if not isinstance(df, pd.DataFrame):
            raise ConversionError(
                ConversionError.Type.TYPE_MISMATCH,
                f"Expected pandas DataFrame, got {type(df)}"
            )
        
        # Store metadata for preservation
        original_metadata = self._metadata_manager.extract_metadata(df, DataFormat.PANDAS_DATAFRAME)
        context.intermediate_results['original_metadata'] = original_metadata
        
        if target_format == DataFormat.NUMPY_ARRAY:
            return self._dataframe_to_numpy(df, context)
        elif target_format == DataFormat.SCIPY_SPARSE:
            return self._dataframe_to_sparse(df, context)
        elif target_format == DataFormat.PYTHON_DICT:
            return self._dataframe_to_dict(df, context)
        elif target_format == DataFormat.PYTHON_LIST:
            return self._dataframe_to_list(df, context)
        elif target_format == DataFormat.TIME_SERIES:
            return self._dataframe_to_timeseries(df, context)
        elif target_format == DataFormat.CATEGORICAL:
            return self._dataframe_to_categorical(df, context)
        else:
            raise ConversionError(
                ConversionError.Type.CONVERSION_FAILED,
                f"Unsupported target format: {target_format.value}"
            )
    
    def _convert_to_dataframe(self, data: Any, 
                            source_format: DataFormat, 
                            context: ConversionContext) -> pd.DataFrame:
        """Convert other formats to DataFrame."""
        if source_format == DataFormat.NUMPY_ARRAY:
            return self._numpy_to_dataframe(data, context)
        elif source_format == DataFormat.SCIPY_SPARSE:
            return self._sparse_to_dataframe(data, context)
        elif source_format == DataFormat.PYTHON_DICT:
            return self._dict_to_dataframe(data, context)
        elif source_format == DataFormat.PYTHON_LIST:
            return self._list_to_dataframe(data, context)
        elif source_format == DataFormat.TIME_SERIES:
            return self._timeseries_to_dataframe(data, context)
        elif source_format == DataFormat.CATEGORICAL:
            return self._categorical_to_dataframe(data, context)
        else:
            raise ConversionError(
                ConversionError.Type.CONVERSION_FAILED,
                f"Unsupported source format: {source_format.value}"
            )
    
    def _dataframe_to_numpy(self, df: pd.DataFrame, context: ConversionContext) -> np.ndarray:
        """Convert DataFrame to NumPy array."""
        try:
            # Handle mixed types by attempting numeric conversion
            if self.conversion_options.handle_mixed_types:
                numeric_df = df.select_dtypes(include=[np.number])
                
                if len(numeric_df.columns) == 0:
                    # No numeric columns - try to convert everything
                    context.warnings.append("No numeric columns found, attempting conversion of all data")
                    converted_df = pd.get_dummies(df, drop_first=True)
                elif len(numeric_df.columns) < len(df.columns):
                    # Mixed types - warn and use only numeric
                    non_numeric_cols = set(df.columns) - set(numeric_df.columns)
                    context.warnings.append(f"Dropping non-numeric columns: {list(non_numeric_cols)}")
                    converted_df = numeric_df
                else:
                    # All numeric
                    converted_df = df
            else:
                converted_df = df
            
            # Convert to numpy array
            array_data = converted_df.values
            
            # Store conversion metadata
            context.intermediate_results['columns_preserved'] = list(converted_df.columns)
            context.intermediate_results['shape_change'] = (df.shape, array_data.shape)
            context.performance_metrics['rows_converted'] = len(df)
            
            self._conversion_stats['total_rows_processed'] += len(df)
            
            return array_data
            
        except Exception as e:
            raise ConversionError(
                ConversionError.Type.CONVERSION_FAILED,
                f"DataFrame to NumPy conversion failed: {str(e)}"
            )
    
    def _dataframe_to_sparse(self, df: pd.DataFrame, context: ConversionContext) -> sparse.spmatrix:
        """Convert DataFrame to scipy sparse matrix."""
        try:
            # First convert to numpy
            array_data = self._dataframe_to_numpy(df, context)
            
            # Calculate density
            density = np.count_nonzero(array_data) / array_data.size
            context.performance_metrics['density'] = density
            
            if density > self.conversion_options.sparse_density_threshold:
                context.warnings.append(
                    f"Data density {density:.3f} above threshold "
                    f"{self.conversion_options.sparse_density_threshold}, but creating sparse matrix anyway"
                )
            
            # Create sparse matrix (CSR format for efficiency)
            sparse_matrix = sparse.csr_matrix(array_data)
            
            context.intermediate_results['sparsity_ratio'] = 1 - density
            context.intermediate_results['sparse_format'] = 'csr'
            
            return sparse_matrix
            
        except Exception as e:
            raise ConversionError(
                ConversionError.Type.CONVERSION_FAILED,
                f"DataFrame to sparse matrix conversion failed: {str(e)}"
            )
    
    def _dataframe_to_dict(self, df: pd.DataFrame, context: ConversionContext) -> Dict[str, Any]:
        """Convert DataFrame to dictionary."""
        try:
            # Use pandas to_dict with records orientation for row-based structure
            dict_data = df.to_dict('records')
            
            # Store additional metadata
            metadata_dict = {
                'data': dict_data,
                'columns': list(df.columns),
                'index': list(df.index) if self.conversion_options.preserve_index else None,
                'dtypes': df.dtypes.to_dict(),
                'shape': df.shape
            }
            
            context.intermediate_results['preservation_mode'] = 'full_metadata'
            context.performance_metrics['dict_entries'] = len(dict_data)
            
            return metadata_dict
            
        except Exception as e:
            raise ConversionError(
                ConversionError.Type.CONVERSION_FAILED,
                f"DataFrame to dict conversion failed: {str(e)}"
            )
    
    def _dataframe_to_list(self, df: pd.DataFrame, context: ConversionContext) -> List[Any]:
        """Convert DataFrame to list."""
        try:
            # Convert to list of lists (rows as lists)
            list_data = df.values.tolist()
            
            # Optionally include column names as first row
            if self.conversion_options.preserve_columns:
                list_data.insert(0, list(df.columns))
                context.intermediate_results['header_included'] = True
            
            context.performance_metrics['list_length'] = len(list_data)
            
            return list_data
            
        except Exception as e:
            raise ConversionError(
                ConversionError.Type.CONVERSION_FAILED,
                f"DataFrame to list conversion failed: {str(e)}"
            )
    
    def _dataframe_to_timeseries(self, df: pd.DataFrame, context: ConversionContext) -> pd.DataFrame:
        """Convert DataFrame to time series format."""
        try:
            # Check if already has datetime index
            if isinstance(df.index, pd.DatetimeIndex):
                ts_df = df.copy()
                context.intermediate_results['already_timeseries'] = True
            else:
                # Try to find datetime column to use as index
                datetime_cols = df.select_dtypes(include=['datetime64']).columns
                
                if len(datetime_cols) > 0:
                    # Use first datetime column as index
                    ts_df = df.set_index(datetime_cols[0])
                    context.intermediate_results['datetime_column_used'] = datetime_cols[0]
                else:
                    # No datetime columns - create a simple integer index and warn
                    ts_df = df.copy()
                    ts_df.index = pd.date_range('2024-01-01', periods=len(df), freq='D')
                    context.warnings.append("No datetime columns found, created synthetic date range")
            
            # Sort by index for time series convention
            ts_df = ts_df.sort_index()
            
            context.intermediate_results['time_range'] = (ts_df.index.min(), ts_df.index.max())
            
            return ts_df
            
        except Exception as e:
            raise ConversionError(
                ConversionError.Type.CONVERSION_FAILED,
                f"DataFrame to time series conversion failed: {str(e)}"
            )
    
    def _dataframe_to_categorical(self, df: pd.DataFrame, context: ConversionContext) -> pd.DataFrame:
        """Convert DataFrame to categorical format."""
        try:
            categorical_df = df.copy()
            converted_columns = []
            
            for col in df.columns:
                # Check if column should be categorical
                unique_ratio = df[col].nunique() / len(df) if len(df) > 0 else 0
                
                if (unique_ratio < self.conversion_options.categorical_threshold and 
                    df[col].nunique() < 100):  # Additional constraint for large cardinality
                    
                    categorical_df[col] = df[col].astype('category')
                    converted_columns.append(col)
            
            if not converted_columns:
                context.warnings.append("No columns met criteria for categorical conversion")
            
            context.intermediate_results['categorical_columns'] = converted_columns
            context.performance_metrics['categories_created'] = len(converted_columns)
            
            return categorical_df
            
        except Exception as e:
            raise ConversionError(
                ConversionError.Type.CONVERSION_FAILED,
                f"DataFrame to categorical conversion failed: {str(e)}"
            )
    
    def _numpy_to_dataframe(self, array: np.ndarray, context: ConversionContext) -> pd.DataFrame:
        """Convert NumPy array to DataFrame."""
        try:
            # Generate column names if not preserved
            if 'original_metadata' in context.intermediate_results:
                metadata = context.intermediate_results['original_metadata']
                columns = metadata.get('columns', None)
            else:
                columns = None
            
            if columns is None or len(columns) != array.shape[1]:
                columns = [f'col_{i}' for i in range(array.shape[1])]
                context.warnings.append("Generated column names as original names not available")
            
            # Create DataFrame
            df = pd.DataFrame(array, columns=columns)
            
            context.intermediate_results['columns_generated'] = columns
            context.performance_metrics['dataframe_shape'] = df.shape
            
            return df
            
        except Exception as e:
            raise ConversionError(
                ConversionError.Type.CONVERSION_FAILED,
                f"NumPy to DataFrame conversion failed: {str(e)}"
            )
    
    def _sparse_to_dataframe(self, sparse_matrix: sparse.spmatrix, context: ConversionContext) -> pd.DataFrame:
        """Convert sparse matrix to DataFrame."""
        try:
            # Convert to dense first
            dense_array = sparse_matrix.toarray()
            
            # Use numpy conversion path
            df = self._numpy_to_dataframe(dense_array, context)
            
            context.intermediate_results['original_sparse_format'] = type(sparse_matrix).__name__
            context.performance_metrics['sparsity_lost'] = True
            
            return df
            
        except Exception as e:
            raise ConversionError(
                ConversionError.Type.CONVERSION_FAILED,
                f"Sparse to DataFrame conversion failed: {str(e)}"
            )
    
    def _dict_to_dataframe(self, data: Dict[str, Any], context: ConversionContext) -> pd.DataFrame:
        """Convert dictionary to DataFrame."""
        try:
            if 'data' in data and 'columns' in data:
                # Structured metadata format
                df = pd.DataFrame(data['data'])
                
                # Restore index if preserved
                if 'index' in data and data['index'] is not None:
                    df.index = data['index']
                
                context.intermediate_results['metadata_restored'] = True
            else:
                # Simple dictionary conversion
                df = pd.DataFrame.from_dict(data, orient='index').T
                context.warnings.append("No metadata structure found, used simple conversion")
            
            context.performance_metrics['dict_keys'] = len(data)
            
            return df
            
        except Exception as e:
            raise ConversionError(
                ConversionError.Type.CONVERSION_FAILED,
                f"Dict to DataFrame conversion failed: {str(e)}"
            )
    
    def _list_to_dataframe(self, data: List[Any], context: ConversionContext) -> pd.DataFrame:
        """Convert list to DataFrame."""
        try:
            if not data:
                # Empty list
                df = pd.DataFrame()
                context.warnings.append("Empty list provided, created empty DataFrame")
                return df
            
            # Check if first row might be headers
            if (len(data) > 1 and 
                isinstance(data[0], list) and 
                all(isinstance(x, str) for x in data[0])):
                # Assume first row is headers
                columns = data[0]
                data_rows = data[1:]
                df = pd.DataFrame(data_rows, columns=columns)
                context.intermediate_results['header_detected'] = True
            else:
                # No headers detected
                df = pd.DataFrame(data)
                context.warnings.append("No headers detected, used default column names")
            
            context.performance_metrics['rows_processed'] = len(data)
            
            return df
            
        except Exception as e:
            raise ConversionError(
                ConversionError.Type.CONVERSION_FAILED,
                f"List to DataFrame conversion failed: {str(e)}"
            )
    
    def _timeseries_to_dataframe(self, ts_data: pd.DataFrame, context: ConversionContext) -> pd.DataFrame:
        """Convert time series to regular DataFrame."""
        try:
            # Time series is already a DataFrame, just ensure proper formatting
            df = ts_data.copy()
            
            # Reset index if it's a DatetimeIndex and we want to preserve it as a column
            if isinstance(df.index, pd.DatetimeIndex) and self.conversion_options.preserve_index:
                df.reset_index(inplace=True)
                context.intermediate_results['datetime_index_preserved'] = True
            
            return df
            
        except Exception as e:
            raise ConversionError(
                ConversionError.Type.CONVERSION_FAILED,
                f"Time series to DataFrame conversion failed: {str(e)}"
            )
    
    def _categorical_to_dataframe(self, cat_data: pd.DataFrame, context: ConversionContext) -> pd.DataFrame:
        """Convert categorical DataFrame to regular DataFrame."""
        try:
            df = cat_data.copy()
            
            # Convert categorical columns back to original types where sensible
            for col in df.columns:
                if pd.api.types.is_categorical_dtype(df[col]):
                    # Try to convert back to numeric if possible
                    try:
                        df[col] = pd.to_numeric(df[col].astype(str))
                        context.intermediate_results.setdefault('reconverted_numeric', []).append(col)
                    except (ValueError, TypeError):
                        # Keep as object type
                        df[col] = df[col].astype(str)
                        context.intermediate_results.setdefault('converted_to_string', []).append(col)
            
            return df
            
        except Exception as e:
            raise ConversionError(
                ConversionError.Type.CONVERSION_FAILED,
                f"Categorical to DataFrame conversion failed: {str(e)}"
            )
    
    def _calculate_quality_score(self, request: ConversionRequest, 
                                converted_data: Any, context: ConversionContext) -> float:
        """Calculate conversion quality score."""
        base_score = 1.0
        
        # Reduce score for warnings
        if context.warnings:
            base_score -= len(context.warnings) * 0.05
        
        # Reduce score for data loss indicators
        if 'shape_change' in context.intermediate_results:
            original_shape, new_shape = context.intermediate_results['shape_change']
            if original_shape != new_shape:
                base_score -= 0.1
        
        # Reduce score for type conversions that might lose information
        if 'sparsity_lost' in context.performance_metrics:
            base_score -= 0.1
        
        # Bonus for metadata preservation
        if context.intermediate_results.get('metadata_restored', False):
            base_score += 0.05
        
        return max(base_score, 0.0)


class NumpyConverter(BaseShimAdapter):
    """
    Bidirectional converter for NumPy array format.
    
    Handles conversions between NumPy arrays and other formats while preserving
    shape, data types, and array properties.
    """
    
    def __init__(self,
                 adapter_id: str = "numpy_converter",
                 conversion_options: Optional[ConversionOptions] = None,
                 **kwargs):
        """Initialize NumpyConverter."""
        
        supported_conversions = [
            # From NumPy array
            (DataFormat.NUMPY_ARRAY, DataFormat.PANDAS_DATAFRAME),
            (DataFormat.NUMPY_ARRAY, DataFormat.SCIPY_SPARSE),
            (DataFormat.NUMPY_ARRAY, DataFormat.PYTHON_LIST),
            (DataFormat.NUMPY_ARRAY, DataFormat.PYTHON_DICT),
            
            # To NumPy array
            (DataFormat.PANDAS_DATAFRAME, DataFormat.NUMPY_ARRAY),
            (DataFormat.SCIPY_SPARSE, DataFormat.NUMPY_ARRAY),
            (DataFormat.PYTHON_LIST, DataFormat.NUMPY_ARRAY),
            (DataFormat.PYTHON_DICT, DataFormat.NUMPY_ARRAY),
        ]
        
        super().__init__(
            adapter_id=adapter_id,
            supported_conversions=supported_conversions,
            **kwargs
        )
        
        self.conversion_options = conversion_options or ConversionOptions()
        self._type_detector = TypeDetectionEngine()
        self._metadata_manager = MetadataManager()
        
        logger.info(f"NumpyConverter initialized", adapter_id=adapter_id)
    
    def _perform_conversion(self, request: ConversionRequest, 
                          context: ConversionContext) -> Any:
        """Perform NumPy array conversion."""
        source_format = request.source_format
        target_format = request.target_format
        source_data = request.source_data
        
        try:
            if source_format == DataFormat.NUMPY_ARRAY:
                return self._convert_from_numpy(source_data, target_format, context)
            elif target_format == DataFormat.NUMPY_ARRAY:
                return self._convert_to_numpy(source_data, source_format, context)
            else:
                raise ConversionError(
                    ConversionError.Type.CONVERSION_FAILED,
                    f"Unsupported conversion path: {source_format.value} -> {target_format.value}"
                )
        
        except Exception as e:
            logger.error(f"NumPy conversion failed: {e}")
            raise ConversionError(
                ConversionError.Type.CONVERSION_FAILED,
                f"NumPy conversion failed: {str(e)}"
            )
    
    def _convert_from_numpy(self, array: np.ndarray, 
                          target_format: DataFormat, 
                          context: ConversionContext) -> Any:
        """Convert NumPy array to other formats."""
        if not isinstance(array, np.ndarray):
            raise ConversionError(
                ConversionError.Type.TYPE_MISMATCH,
                f"Expected NumPy array, got {type(array)}"
            )
        
        # Store metadata
        original_metadata = self._metadata_manager.extract_metadata(array, DataFormat.NUMPY_ARRAY)
        context.intermediate_results['original_metadata'] = original_metadata
        
        if target_format == DataFormat.PANDAS_DATAFRAME:
            return self._numpy_to_dataframe(array, context)
        elif target_format == DataFormat.SCIPY_SPARSE:
            return self._numpy_to_sparse(array, context)
        elif target_format == DataFormat.PYTHON_LIST:
            return self._numpy_to_list(array, context)
        elif target_format == DataFormat.PYTHON_DICT:
            return self._numpy_to_dict(array, context)
        else:
            raise ConversionError(
                ConversionError.Type.CONVERSION_FAILED,
                f"Unsupported target format: {target_format.value}"
            )
    
    def _convert_to_numpy(self, data: Any, 
                        source_format: DataFormat, 
                        context: ConversionContext) -> np.ndarray:
        """Convert other formats to NumPy array."""
        if source_format == DataFormat.PANDAS_DATAFRAME:
            return self._dataframe_to_numpy(data, context)
        elif source_format == DataFormat.SCIPY_SPARSE:
            return self._sparse_to_numpy(data, context)
        elif source_format == DataFormat.PYTHON_LIST:
            return self._list_to_numpy(data, context)
        elif source_format == DataFormat.PYTHON_DICT:
            return self._dict_to_numpy(data, context)
        else:
            raise ConversionError(
                ConversionError.Type.CONVERSION_FAILED,
                f"Unsupported source format: {source_format.value}"
            )
    
    def _numpy_to_dataframe(self, array: np.ndarray, context: ConversionContext) -> pd.DataFrame:
        """Convert NumPy array to DataFrame."""
        try:
            if array.ndim == 1:
                # 1D array becomes single column
                df = pd.DataFrame(array, columns=['values'])
                context.intermediate_results['dimension_handling'] = '1d_to_column'
            elif array.ndim == 2:
                # 2D array becomes standard DataFrame
                columns = [f'col_{i}' for i in range(array.shape[1])]
                df = pd.DataFrame(array, columns=columns)
                context.intermediate_results['dimension_handling'] = '2d_standard'
            else:
                # Higher dimensions - flatten to 2D with warning
                reshaped = array.reshape(array.shape[0], -1)
                columns = [f'col_{i}' for i in range(reshaped.shape[1])]
                df = pd.DataFrame(reshaped, columns=columns)
                context.warnings.append(f"Flattened {array.ndim}D array to 2D for DataFrame conversion")
                context.intermediate_results['dimension_handling'] = f'{array.ndim}d_flattened'
            
            context.performance_metrics['shape_preserved'] = array.shape
            return df
            
        except Exception as e:
            raise ConversionError(
                ConversionError.Type.CONVERSION_FAILED,
                f"NumPy to DataFrame conversion failed: {str(e)}"
            )
    
    def _numpy_to_sparse(self, array: np.ndarray, context: ConversionContext) -> sparse.spmatrix:
        """Convert NumPy array to sparse matrix."""
        try:
            # Ensure 2D
            if array.ndim == 1:
                array = array.reshape(-1, 1)
                context.intermediate_results['reshaped_1d'] = True
            elif array.ndim > 2:
                original_shape = array.shape
                array = array.reshape(array.shape[0], -1)
                context.warnings.append(f"Reshaped {original_shape} array to {array.shape} for sparse conversion")
            
            # Calculate density
            density = np.count_nonzero(array) / array.size
            context.performance_metrics['density'] = density
            
            # Choose sparse format based on density and shape
            if density < 0.05:  # Very sparse
                sparse_matrix = sparse.coo_matrix(array)
                context.intermediate_results['sparse_format'] = 'coo'
            elif array.shape[0] > array.shape[1]:  # Tall matrix
                sparse_matrix = sparse.csc_matrix(array)
                context.intermediate_results['sparse_format'] = 'csc'
            else:  # Wide matrix or square
                sparse_matrix = sparse.csr_matrix(array)
                context.intermediate_results['sparse_format'] = 'csr'
            
            return sparse_matrix
            
        except Exception as e:
            raise ConversionError(
                ConversionError.Type.CONVERSION_FAILED,
                f"NumPy to sparse conversion failed: {str(e)}"
            )
    
    def _numpy_to_list(self, array: np.ndarray, context: ConversionContext) -> List[Any]:
        """Convert NumPy array to Python list."""
        try:
            # Convert to list preserving structure
            list_data = array.tolist()
            
            context.performance_metrics['original_shape'] = array.shape
            context.performance_metrics['list_nesting'] = array.ndim
            
            return list_data
            
        except Exception as e:
            raise ConversionError(
                ConversionError.Type.CONVERSION_FAILED,
                f"NumPy to list conversion failed: {str(e)}"
            )
    
    def _numpy_to_dict(self, array: np.ndarray, context: ConversionContext) -> Dict[str, Any]:
        """Convert NumPy array to dictionary with metadata."""
        try:
            dict_data = {
                'data': array.tolist(),
                'shape': array.shape,
                'dtype': str(array.dtype),
                'ndim': array.ndim,
                'size': array.size
            }
            
            # Add memory layout information
            dict_data['flags'] = {
                'c_contiguous': array.flags.c_contiguous,
                'f_contiguous': array.flags.f_contiguous,
                'writeable': array.flags.writeable
            }
            
            context.performance_metrics['metadata_included'] = True
            
            return dict_data
            
        except Exception as e:
            raise ConversionError(
                ConversionError.Type.CONVERSION_FAILED,
                f"NumPy to dict conversion failed: {str(e)}"
            )
    
    def _dataframe_to_numpy(self, df: pd.DataFrame, context: ConversionContext) -> np.ndarray:
        """Convert DataFrame to NumPy array."""
        try:
            # Select numeric columns only if mixed types
            numeric_df = df.select_dtypes(include=[np.number])
            
            if len(numeric_df.columns) == 0:
                # No numeric columns - try to convert
                try:
                    array = df.values.astype(float)
                    context.warnings.append("Forced conversion of non-numeric data to float")
                except (ValueError, TypeError):
                    # Use object array
                    array = df.values
                    context.warnings.append("Created object array due to mixed/non-numeric types")
            elif len(numeric_df.columns) < len(df.columns):
                # Mixed types - warn and use only numeric
                array = numeric_df.values
                dropped_cols = set(df.columns) - set(numeric_df.columns)
                context.warnings.append(f"Dropped non-numeric columns: {list(dropped_cols)}")
            else:
                # All numeric
                array = df.values
            
            context.performance_metrics['original_df_shape'] = df.shape
            context.performance_metrics['final_array_shape'] = array.shape
            
            return array
            
        except Exception as e:
            raise ConversionError(
                ConversionError.Type.CONVERSION_FAILED,
                f"DataFrame to NumPy conversion failed: {str(e)}"
            )
    
    def _sparse_to_numpy(self, sparse_matrix: sparse.spmatrix, context: ConversionContext) -> np.ndarray:
        """Convert sparse matrix to NumPy array."""
        try:
            array = sparse_matrix.toarray()
            
            context.performance_metrics['original_sparse_format'] = type(sparse_matrix).__name__
            context.performance_metrics['density_restored'] = np.count_nonzero(array) / array.size
            
            return array
            
        except Exception as e:
            raise ConversionError(
                ConversionError.Type.CONVERSION_FAILED,
                f"Sparse to NumPy conversion failed: {str(e)}"
            )
    
    def _list_to_numpy(self, data: List[Any], context: ConversionContext) -> np.ndarray:
        """Convert Python list to NumPy array."""
        try:
            array = np.array(data)
            
            context.performance_metrics['inferred_shape'] = array.shape
            context.performance_metrics['inferred_dtype'] = str(array.dtype)
            
            return array
            
        except Exception as e:
            raise ConversionError(
                ConversionError.Type.CONVERSION_FAILED,
                f"List to NumPy conversion failed: {str(e)}"
            )
    
    def _dict_to_numpy(self, data: Dict[str, Any], context: ConversionContext) -> np.ndarray:
        """Convert dictionary to NumPy array."""
        try:
            if 'data' in data and 'shape' in data:
                # Structured format with metadata
                array = np.array(data['data'])
                
                # Try to restore original shape
                if 'shape' in data:
                    try:
                        array = array.reshape(data['shape'])
                        context.intermediate_results['shape_restored'] = True
                    except ValueError:
                        context.warnings.append("Could not restore original shape")
                
                # Try to restore dtype
                if 'dtype' in data:
                    try:
                        array = array.astype(data['dtype'])
                        context.intermediate_results['dtype_restored'] = True
                    except (ValueError, TypeError):
                        context.warnings.append(f"Could not restore dtype {data['dtype']}")
                
            else:
                # Simple dictionary - try to convert values
                if all(isinstance(v, (list, tuple)) for v in data.values()):
                    # Dictionary of arrays
                    array = np.array(list(data.values()))
                else:
                    # Dictionary of scalars
                    array = np.array(list(data.values()))
                
                context.warnings.append("Converted simple dictionary, some structure may be lost")
            
            return array
            
        except Exception as e:
            raise ConversionError(
                ConversionError.Type.CONVERSION_FAILED,
                f"Dict to NumPy conversion failed: {str(e)}"
            )


class SparseMatrixConverter(BaseShimAdapter):
    """
    Bidirectional converter for scipy sparse matrix formats.
    
    Handles conversions between sparse matrices and other formats with
    intelligent density management and format optimization.
    """
    
    def __init__(self,
                 adapter_id: str = "sparse_converter",
                 conversion_options: Optional[ConversionOptions] = None,
                 **kwargs):
        """Initialize SparseMatrixConverter."""
        
        supported_conversions = [
            # From sparse matrix
            (DataFormat.SCIPY_SPARSE, DataFormat.PANDAS_DATAFRAME),
            (DataFormat.SCIPY_SPARSE, DataFormat.NUMPY_ARRAY),
            (DataFormat.SCIPY_SPARSE, DataFormat.PYTHON_DICT),
            (DataFormat.SCIPY_SPARSE, DataFormat.PYTHON_LIST),
            
            # To sparse matrix
            (DataFormat.PANDAS_DATAFRAME, DataFormat.SCIPY_SPARSE),
            (DataFormat.NUMPY_ARRAY, DataFormat.SCIPY_SPARSE),
            (DataFormat.PYTHON_DICT, DataFormat.SCIPY_SPARSE),
            (DataFormat.PYTHON_LIST, DataFormat.SCIPY_SPARSE),
        ]
        
        super().__init__(
            adapter_id=adapter_id,
            supported_conversions=supported_conversions,
            **kwargs
        )
        
        self.conversion_options = conversion_options or ConversionOptions()
        self._metadata_manager = MetadataManager()
        
        # Sparse format preferences based on operations
        self._format_preferences = {
            'row_operations': sparse.csr_matrix,
            'column_operations': sparse.csc_matrix,
            'construction': sparse.coo_matrix,
            'general': sparse.csr_matrix
        }
        
        logger.info(f"SparseMatrixConverter initialized", adapter_id=adapter_id)
    
    def _perform_conversion(self, request: ConversionRequest, 
                          context: ConversionContext) -> Any:
        """Perform sparse matrix conversion."""
        source_format = request.source_format
        target_format = request.target_format
        source_data = request.source_data
        
        try:
            if source_format == DataFormat.SCIPY_SPARSE:
                return self._convert_from_sparse(source_data, target_format, context)
            elif target_format == DataFormat.SCIPY_SPARSE:
                return self._convert_to_sparse(source_data, source_format, context)
            else:
                raise ConversionError(
                    ConversionError.Type.CONVERSION_FAILED,
                    f"Unsupported conversion path: {source_format.value} -> {target_format.value}"
                )
        
        except Exception as e:
            logger.error(f"Sparse matrix conversion failed: {e}")
            raise ConversionError(
                ConversionError.Type.CONVERSION_FAILED,
                f"Sparse conversion failed: {str(e)}"
            )
    
    def _convert_from_sparse(self, sparse_matrix: sparse.spmatrix, 
                           target_format: DataFormat, 
                           context: ConversionContext) -> Any:
        """Convert sparse matrix to other formats."""
        if not sparse.issparse(sparse_matrix):
            raise ConversionError(
                ConversionError.Type.TYPE_MISMATCH,
                f"Expected sparse matrix, got {type(sparse_matrix)}"
            )
        
        # Store sparse matrix metadata
        context.intermediate_results['original_sparse_info'] = {
            'format': type(sparse_matrix).__name__,
            'shape': sparse_matrix.shape,
            'nnz': sparse_matrix.nnz,
            'density': sparse_matrix.nnz / (sparse_matrix.shape[0] * sparse_matrix.shape[1])
        }
        
        if target_format == DataFormat.PANDAS_DATAFRAME:
            return self._sparse_to_dataframe(sparse_matrix, context)
        elif target_format == DataFormat.NUMPY_ARRAY:
            return self._sparse_to_numpy(sparse_matrix, context)
        elif target_format == DataFormat.PYTHON_DICT:
            return self._sparse_to_dict(sparse_matrix, context)
        elif target_format == DataFormat.PYTHON_LIST:
            return self._sparse_to_list(sparse_matrix, context)
        else:
            raise ConversionError(
                ConversionError.Type.CONVERSION_FAILED,
                f"Unsupported target format: {target_format.value}"
            )
    
    def _convert_to_sparse(self, data: Any, 
                         source_format: DataFormat, 
                         context: ConversionContext) -> sparse.spmatrix:
        """Convert other formats to sparse matrix."""
        if source_format == DataFormat.PANDAS_DATAFRAME:
            return self._dataframe_to_sparse(data, context)
        elif source_format == DataFormat.NUMPY_ARRAY:
            return self._numpy_to_sparse(data, context)
        elif source_format == DataFormat.PYTHON_DICT:
            return self._dict_to_sparse(data, context)
        elif source_format == DataFormat.PYTHON_LIST:
            return self._list_to_sparse(data, context)
        else:
            raise ConversionError(
                ConversionError.Type.CONVERSION_FAILED,
                f"Unsupported source format: {source_format.value}"
            )
    
    def _sparse_to_dataframe(self, sparse_matrix: sparse.spmatrix, context: ConversionContext) -> pd.DataFrame:
        """Convert sparse matrix to DataFrame."""
        try:
            # Convert to dense array first
            dense_array = sparse_matrix.toarray()
            
            # Create DataFrame with generated column names
            columns = [f'col_{i}' for i in range(dense_array.shape[1])]
            df = pd.DataFrame(dense_array, columns=columns)
            
            context.performance_metrics['sparsity_lost'] = True
            context.performance_metrics['memory_increase'] = 'significant'
            context.warnings.append("Converted sparse to dense - significant memory increase possible")
            
            return df
            
        except Exception as e:
            raise ConversionError(
                ConversionError.Type.CONVERSION_FAILED,
                f"Sparse to DataFrame conversion failed: {str(e)}"
            )
    
    def _sparse_to_numpy(self, sparse_matrix: sparse.spmatrix, context: ConversionContext) -> np.ndarray:
        """Convert sparse matrix to NumPy array."""
        try:
            dense_array = sparse_matrix.toarray()
            
            context.performance_metrics['densification'] = True
            context.performance_metrics['original_nnz'] = sparse_matrix.nnz
            context.performance_metrics['final_size'] = dense_array.size
            
            return dense_array
            
        except Exception as e:
            raise ConversionError(
                ConversionError.Type.CONVERSION_FAILED,
                f"Sparse to NumPy conversion failed: {str(e)}"
            )
    
    def _sparse_to_dict(self, sparse_matrix: sparse.spmatrix, context: ConversionContext) -> Dict[str, Any]:
        """Convert sparse matrix to dictionary."""
        try:
            # Store in COO format for easier serialization
            coo_matrix = sparse_matrix.tocoo()
            
            dict_data = {
                'format': 'scipy_sparse_coo',
                'shape': sparse_matrix.shape,
                'data': coo_matrix.data.tolist(),
                'row': coo_matrix.row.tolist(),
                'col': coo_matrix.col.tolist(),
                'dtype': str(sparse_matrix.dtype),
                'nnz': sparse_matrix.nnz
            }
            
            context.intermediate_results['preservation_format'] = 'coo_coordinates'
            context.performance_metrics['compression_ratio'] = sparse_matrix.nnz / sparse_matrix.size
            
            return dict_data
            
        except Exception as e:
            raise ConversionError(
                ConversionError.Type.CONVERSION_FAILED,
                f"Sparse to dict conversion failed: {str(e)}"
            )
    
    def _sparse_to_list(self, sparse_matrix: sparse.spmatrix, context: ConversionContext) -> List[Any]:
        """Convert sparse matrix to list."""
        try:
            # Convert to dense first, then to list
            dense_array = sparse_matrix.toarray()
            list_data = dense_array.tolist()
            
            context.warnings.append("Sparse structure lost in list conversion")
            
            return list_data
            
        except Exception as e:
            raise ConversionError(
                ConversionError.Type.CONVERSION_FAILED,
                f"Sparse to list conversion failed: {str(e)}"
            )
    
    def _dataframe_to_sparse(self, df: pd.DataFrame, context: ConversionContext) -> sparse.spmatrix:
        """Convert DataFrame to sparse matrix."""
        try:
            # Convert to numpy first, handling mixed types
            numeric_df = df.select_dtypes(include=[np.number])
            
            if len(numeric_df.columns) == 0:
                # No numeric columns - try dummy encoding
                encoded_df = pd.get_dummies(df, drop_first=True)
                array_data = encoded_df.values
                context.warnings.append("Applied dummy encoding for non-numeric data")
            elif len(numeric_df.columns) < len(df.columns):
                # Mixed types
                array_data = numeric_df.values
                dropped_cols = set(df.columns) - set(numeric_df.columns)
                context.warnings.append(f"Dropped non-numeric columns: {list(dropped_cols)}")
            else:
                array_data = df.values
            
            # Calculate density to choose format
            density = np.count_nonzero(array_data) / array_data.size
            context.performance_metrics['density'] = density
            
            if density > self.conversion_options.sparse_density_threshold:
                context.warnings.append(
                    f"Data density {density:.3f} above threshold "
                    f"{self.conversion_options.sparse_density_threshold}"
                )
            
            # Choose sparse format based on shape and density
            if density < 0.01:  # Very sparse
                sparse_matrix = sparse.coo_matrix(array_data)
            elif array_data.shape[0] > array_data.shape[1]:  # Tall matrix
                sparse_matrix = sparse.csc_matrix(array_data)
            else:  # Wide matrix
                sparse_matrix = sparse.csr_matrix(array_data)
            
            context.intermediate_results['sparse_format_chosen'] = type(sparse_matrix).__name__
            
            return sparse_matrix
            
        except Exception as e:
            raise ConversionError(
                ConversionError.Type.CONVERSION_FAILED,
                f"DataFrame to sparse conversion failed: {str(e)}"
            )
    
    def _numpy_to_sparse(self, array: np.ndarray, context: ConversionContext) -> sparse.spmatrix:
        """Convert NumPy array to sparse matrix."""
        try:
            # Ensure 2D
            if array.ndim == 1:
                array = array.reshape(-1, 1)
                context.intermediate_results['reshaped_1d'] = True
            elif array.ndim > 2:
                original_shape = array.shape
                array = array.reshape(array.shape[0], -1)
                context.warnings.append(f"Flattened {original_shape} to {array.shape} for sparse conversion")
            
            # Calculate density and choose format
            density = np.count_nonzero(array) / array.size
            context.performance_metrics['density'] = density
            
            if density < 0.01:
                sparse_matrix = sparse.coo_matrix(array)
                context.intermediate_results['format_reason'] = 'very_sparse'
            elif density < 0.1:
                sparse_matrix = sparse.csr_matrix(array)
                context.intermediate_results['format_reason'] = 'moderately_sparse'
            else:
                sparse_matrix = sparse.csr_matrix(array)
                context.warnings.append(f"Dense data (density={density:.3f}) converted to sparse")
                context.intermediate_results['format_reason'] = 'dense_but_requested'
            
            return sparse_matrix
            
        except Exception as e:
            raise ConversionError(
                ConversionError.Type.CONVERSION_FAILED,
                f"NumPy to sparse conversion failed: {str(e)}"
            )
    
    def _dict_to_sparse(self, data: Dict[str, Any], context: ConversionContext) -> sparse.spmatrix:
        """Convert dictionary to sparse matrix."""
        try:
            if 'format' in data and data['format'] == 'scipy_sparse_coo':
                # Restore from COO format
                sparse_matrix = sparse.coo_matrix(
                    (data['data'], (data['row'], data['col'])),
                    shape=data['shape'],
                    dtype=data.get('dtype', 'float64')
                )
                
                # Convert to CSR for efficiency
                sparse_matrix = sparse_matrix.tocsr()
                
                context.intermediate_results['restored_from_coo'] = True
                context.performance_metrics['nnz_restored'] = data['nnz']
                
            else:
                # Convert dictionary values to array first, then to sparse
                if isinstance(data, dict) and all(isinstance(v, (list, np.ndarray)) for v in data.values()):
                    # Dictionary of arrays
                    array = np.array(list(data.values()))
                else:
                    # Try to convert to array
                    array = np.array(list(data.values())).reshape(-1, 1)
                    context.warnings.append("Dictionary structure not preserved in sparse conversion")
                
                sparse_matrix = sparse.csr_matrix(array)
            
            return sparse_matrix
            
        except Exception as e:
            raise ConversionError(
                ConversionError.Type.CONVERSION_FAILED,
                f"Dict to sparse conversion failed: {str(e)}"
            )
    
    def _list_to_sparse(self, data: List[Any], context: ConversionContext) -> sparse.spmatrix:
        """Convert list to sparse matrix."""
        try:
            # Convert list to numpy array first
            array = np.array(data)
            
            # Ensure 2D
            if array.ndim == 1:
                array = array.reshape(-1, 1)
                context.intermediate_results['reshaped_from_1d'] = True
            elif array.ndim > 2:
                original_shape = array.shape
                array = array.reshape(array.shape[0], -1)
                context.warnings.append(f"Flattened shape from {original_shape} to {array.shape}")
            
            # Create sparse matrix
            sparse_matrix = sparse.csr_matrix(array)
            
            density = sparse_matrix.nnz / sparse_matrix.size
            context.performance_metrics['density'] = density
            
            if density > 0.5:
                context.warnings.append(f"High density ({density:.3f}) list converted to sparse - may be inefficient")
            
            return sparse_matrix
            
        except Exception as e:
            raise ConversionError(
                ConversionError.Type.CONVERSION_FAILED,
                f"List to sparse conversion failed: {str(e)}"
            )
    
    def _calculate_quality_score(self, request: ConversionRequest, 
                                converted_data: Any, context: ConversionContext) -> float:
        """Calculate conversion quality score for sparse operations."""
        base_score = 1.0
        
        # Reduce score for warnings
        if context.warnings:
            base_score -= len(context.warnings) * 0.05
        
        # Reduce score for sparsity loss
        if context.performance_metrics.get('sparsity_lost', False):
            base_score -= 0.2
        
        # Reduce score for poor density match
        if 'density' in context.performance_metrics:
            density = context.performance_metrics['density']
            if density > 0.5 and request.target_format == DataFormat.SCIPY_SPARSE:
                base_score -= 0.1  # Dense data converted to sparse
            elif density < 0.1 and request.source_format == DataFormat.SCIPY_SPARSE:
                base_score -= 0.05  # Very sparse data densified
        
        return max(base_score, 0.0)


# Factory functions for easy converter creation

def create_pandas_converter(conversion_options: Optional[ConversionOptions] = None,
                           **kwargs) -> PandasConverter:
    """Create a PandasConverter with optional configuration."""
    return PandasConverter(conversion_options=conversion_options, **kwargs)


def create_numpy_converter(conversion_options: Optional[ConversionOptions] = None,
                          **kwargs) -> NumpyConverter:
    """Create a NumpyConverter with optional configuration."""
    return NumpyConverter(conversion_options=conversion_options, **kwargs)


def create_sparse_converter(conversion_options: Optional[ConversionOptions] = None,
                           **kwargs) -> SparseMatrixConverter:
    """Create a SparseMatrixConverter with optional configuration."""
    return SparseMatrixConverter(conversion_options=conversion_options, **kwargs)


# Utility functions for conversion options

def create_memory_efficient_options() -> ConversionOptions:
    """Create conversion options optimized for memory efficiency."""
    return ConversionOptions(
        chunk_size_rows=5000,
        memory_efficient=True,
        sparse_density_threshold=0.05,
        quality_target=ConversionQuality.MODERATE
    )


def create_high_fidelity_options() -> ConversionOptions:
    """Create conversion options optimized for data preservation."""
    return ConversionOptions(
        preserve_index=True,
        preserve_columns=True,
        handle_mixed_types=True,
        quality_target=ConversionQuality.HIGH_FIDELITY
    )


def create_streaming_options(chunk_size: int = 10000) -> ConversionOptions:
    """Create conversion options optimized for streaming processing."""
    return ConversionOptions(
        chunk_size_rows=chunk_size,
        memory_efficient=True,
        quality_target=ConversionQuality.HIGH_FIDELITY
    )