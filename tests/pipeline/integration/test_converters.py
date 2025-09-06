"""
Unit tests for core data format converters in the Integration Shims Framework.

Tests cover:
- PandasConverter: DataFrame ↔ other formats with metadata preservation
- NumpyConverter: Array ↔ other formats with shape/dtype preservation  
- SparseMatrixConverter: Sparse matrices ↔ other formats with density management
- Bidirectional conversion accuracy and quality
- Memory efficiency and streaming capabilities
- Edge cases and error handling
"""

import pytest
import pandas as pd
import numpy as np
from scipy import sparse
import warnings
from unittest.mock import Mock, patch

from src.localdata_mcp.pipeline.integration.converters import (
    PandasConverter,
    NumpyConverter,
    SparseMatrixConverter,
    ConversionOptions,
    ConversionQuality,
    create_pandas_converter,
    create_numpy_converter,
    create_sparse_converter,
    create_memory_efficient_options,
    create_high_fidelity_options,
    create_streaming_options
)
from src.localdata_mcp.pipeline.integration.interfaces import (
    DataFormat,
    ConversionRequest,
    ConversionError,
    create_conversion_request
)


class TestPandasConverter:
    """Test cases for PandasConverter."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.converter = PandasConverter()
        
        # Sample datasets
        self.sample_df = pd.DataFrame({
            'numeric_int': [1, 2, 3, 4, 5],
            'numeric_float': [1.1, 2.2, 3.3, 4.4, 5.5],
            'string_col': ['a', 'b', 'c', 'd', 'e'],
            'bool_col': [True, False, True, False, True]
        })
        
        # Time series data
        self.ts_df = pd.DataFrame({
            'value': [10, 20, 30, 40, 50],
            'date': pd.date_range('2024-01-01', periods=5, freq='D')
        }).set_index('date')
        
        # Categorical data
        self.cat_df = pd.DataFrame({
            'category': pd.Categorical(['A', 'B', 'A', 'C', 'B']),
            'values': [1, 2, 3, 4, 5]
        })
        
        # Mixed types DataFrame
        self.mixed_df = pd.DataFrame({
            'numbers': [1, 2, 3],
            'strings': ['x', 'y', 'z'],
            'dates': pd.date_range('2024-01-01', periods=3)
        })
    
    def test_initialization(self):
        """Test PandasConverter initialization."""
        assert self.converter.adapter_id == "pandas_converter"
        assert len(self.converter.supported_conversions) > 0
        assert isinstance(self.converter.conversion_options, ConversionOptions)
        
        # Check specific conversions are supported
        conversions = self.converter.get_supported_conversions()
        assert (DataFormat.PANDAS_DATAFRAME, DataFormat.NUMPY_ARRAY) in conversions
        assert (DataFormat.NUMPY_ARRAY, DataFormat.PANDAS_DATAFRAME) in conversions
    
    def test_dataframe_to_numpy_conversion(self):
        """Test DataFrame to NumPy array conversion."""
        request = create_conversion_request(
            self.sample_df,
            DataFormat.PANDAS_DATAFRAME,
            DataFormat.NUMPY_ARRAY
        )
        
        result = self.converter.convert(request)
        
        assert result.success is True
        assert isinstance(result.converted_data, np.ndarray)
        assert result.converted_data.shape[0] == len(self.sample_df)
        assert result.quality_score > 0.8  # Good quality conversion
        
        # Should handle mixed types by selecting numeric columns
        assert len(result.warnings) > 0  # Should warn about dropping string columns
    
    def test_numpy_to_dataframe_conversion(self):
        """Test NumPy array to DataFrame conversion."""
        test_array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        
        request = create_conversion_request(
            test_array,
            DataFormat.NUMPY_ARRAY,
            DataFormat.PANDAS_DATAFRAME
        )
        
        result = self.converter.convert(request)
        
        assert result.success is True
        assert isinstance(result.converted_data, pd.DataFrame)
        assert result.converted_data.shape == test_array.shape
        assert list(result.converted_data.columns) == ['col_0', 'col_1', 'col_2']
    
    def test_dataframe_to_sparse_conversion(self):
        """Test DataFrame to sparse matrix conversion."""
        # Create sparse-like DataFrame
        sparse_data = pd.DataFrame({
            'col1': [1, 0, 0, 2, 0],
            'col2': [0, 0, 3, 0, 0],
            'col3': [0, 4, 0, 0, 5]
        })
        
        request = create_conversion_request(
            sparse_data,
            DataFormat.PANDAS_DATAFRAME,
            DataFormat.SCIPY_SPARSE
        )
        
        result = self.converter.convert(request)
        
        assert result.success is True
        assert sparse.issparse(result.converted_data)
        assert result.converted_data.shape == sparse_data.shape
    
    def test_dataframe_to_dict_conversion(self):
        """Test DataFrame to dictionary conversion."""
        request = create_conversion_request(
            self.sample_df,
            DataFormat.PANDAS_DATAFRAME,
            DataFormat.PYTHON_DICT
        )
        
        result = self.converter.convert(request)
        
        assert result.success is True
        assert isinstance(result.converted_data, dict)
        assert 'data' in result.converted_data
        assert 'columns' in result.converted_data
        assert len(result.converted_data['data']) == len(self.sample_df)
    
    def test_dataframe_to_list_conversion(self):
        """Test DataFrame to list conversion."""
        request = create_conversion_request(
            self.sample_df,
            DataFormat.PANDAS_DATAFRAME,
            DataFormat.PYTHON_LIST
        )
        
        result = self.converter.convert(request)
        
        assert result.success is True
        assert isinstance(result.converted_data, list)
        # Should include header row plus data rows
        assert len(result.converted_data) == len(self.sample_df) + 1
        assert result.converted_data[0] == list(self.sample_df.columns)
    
    def test_timeseries_conversion(self):
        """Test time series specific conversion."""
        request = create_conversion_request(
            self.sample_df,  # Regular DataFrame
            DataFormat.PANDAS_DATAFRAME,
            DataFormat.TIME_SERIES
        )
        
        result = self.converter.convert(request)
        
        assert result.success is True
        assert isinstance(result.converted_data, pd.DataFrame)
        # Should warn about creating synthetic date range
        assert len(result.warnings) > 0
    
    def test_categorical_conversion(self):
        """Test categorical conversion."""
        # Create DataFrame with low cardinality columns suitable for categorical
        low_cardinality_df = pd.DataFrame({
            'category1': ['A', 'B', 'A', 'A', 'B'] * 4,  # Low cardinality
            'category2': ['X', 'Y', 'X', 'Y', 'X'] * 4,  # Low cardinality
            'high_card': list(range(20))  # High cardinality - shouldn't convert
        })
        
        request = create_conversion_request(
            low_cardinality_df,
            DataFormat.PANDAS_DATAFRAME,
            DataFormat.CATEGORICAL
        )
        
        result = self.converter.convert(request)
        
        assert result.success is True
        assert isinstance(result.converted_data, pd.DataFrame)
        
        # Check that low cardinality columns were converted to categorical
        converted_df = result.converted_data
        assert pd.api.types.is_categorical_dtype(converted_df['category1'])
        assert pd.api.types.is_categorical_dtype(converted_df['category2'])
        # High cardinality column should remain unchanged
        assert not pd.api.types.is_categorical_dtype(converted_df['high_card'])
    
    def test_bidirectional_conversion_consistency(self):
        """Test bidirectional conversion consistency."""
        # DataFrame -> NumPy -> DataFrame
        df_to_numpy_request = create_conversion_request(
            self.sample_df,
            DataFormat.PANDAS_DATAFRAME,
            DataFormat.NUMPY_ARRAY
        )
        
        numpy_result = self.converter.convert(df_to_numpy_request)
        assert numpy_result.success is True
        
        numpy_to_df_request = create_conversion_request(
            numpy_result.converted_data,
            DataFormat.NUMPY_ARRAY,
            DataFormat.PANDAS_DATAFRAME
        )
        
        df_result = self.converter.convert(numpy_to_df_request)
        assert df_result.success is True
        
        # Check shape consistency
        original_numeric = self.sample_df.select_dtypes(include=[np.number])
        assert df_result.converted_data.shape == original_numeric.shape
    
    def test_conversion_options_handling(self):
        """Test different conversion options."""
        options = ConversionOptions(
            preserve_index=False,
            preserve_columns=False,
            handle_mixed_types=False
        )
        
        converter_with_options = PandasConverter(conversion_options=options)
        
        request = create_conversion_request(
            self.sample_df,
            DataFormat.PANDAS_DATAFRAME,
            DataFormat.PYTHON_LIST
        )
        
        result = converter_with_options.convert(request)
        
        assert result.success is True
        # With preserve_columns=False, shouldn't include header
        assert len(result.converted_data) == len(self.sample_df)  # No extra header row
    
    def test_error_handling(self):
        """Test error handling for invalid conversions."""
        # Test with invalid source data type
        request = create_conversion_request(
            "not_a_dataframe",
            DataFormat.PANDAS_DATAFRAME,
            DataFormat.NUMPY_ARRAY
        )
        
        result = self.converter.convert(request)
        
        assert result.success is False
        assert len(result.errors) > 0
        assert "pandas DataFrame" in result.errors[0]
    
    def test_large_dataframe_handling(self):
        """Test handling of large DataFrames (memory efficiency)."""
        # Create a larger DataFrame for testing
        large_df = pd.DataFrame(np.random.randn(1000, 10))
        
        request = create_conversion_request(
            large_df,
            DataFormat.PANDAS_DATAFRAME,
            DataFormat.NUMPY_ARRAY
        )
        
        result = self.converter.convert(request)
        
        assert result.success is True
        assert result.converted_data.shape == large_df.shape
        # Check performance metrics are recorded
        assert 'rows_converted' in result.performance_metrics


class TestNumpyConverter:
    """Test cases for NumpyConverter."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.converter = NumpyConverter()
        
        # Sample arrays
        self.array_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        self.array_1d = np.array([1, 2, 3, 4, 5])
        self.array_3d = np.random.randn(2, 3, 4)
        self.float_array = np.array([[1.1, 2.2], [3.3, 4.4]])
        
        # Sample DataFrame for conversion to numpy
        self.sample_df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [4, 5, 6],
            'col3': [7, 8, 9]
        })
    
    def test_initialization(self):
        """Test NumpyConverter initialization."""
        assert self.converter.adapter_id == "numpy_converter"
        assert len(self.converter.supported_conversions) > 0
        
        conversions = self.converter.get_supported_conversions()
        assert (DataFormat.NUMPY_ARRAY, DataFormat.PANDAS_DATAFRAME) in conversions
        assert (DataFormat.PANDAS_DATAFRAME, DataFormat.NUMPY_ARRAY) in conversions
    
    def test_numpy_to_dataframe_2d(self):
        """Test 2D NumPy array to DataFrame conversion."""
        request = create_conversion_request(
            self.array_2d,
            DataFormat.NUMPY_ARRAY,
            DataFormat.PANDAS_DATAFRAME
        )
        
        result = self.converter.convert(request)
        
        assert result.success is True
        assert isinstance(result.converted_data, pd.DataFrame)
        assert result.converted_data.shape == self.array_2d.shape
        assert list(result.converted_data.columns) == ['col_0', 'col_1', 'col_2']
        np.testing.assert_array_equal(result.converted_data.values, self.array_2d)
    
    def test_numpy_to_dataframe_1d(self):
        """Test 1D NumPy array to DataFrame conversion."""
        request = create_conversion_request(
            self.array_1d,
            DataFormat.NUMPY_ARRAY,
            DataFormat.PANDAS_DATAFRAME
        )
        
        result = self.converter.convert(request)
        
        assert result.success is True
        assert isinstance(result.converted_data, pd.DataFrame)
        assert result.converted_data.shape == (len(self.array_1d), 1)
        assert list(result.converted_data.columns) == ['values']
    
    def test_numpy_to_dataframe_3d(self):
        """Test 3D NumPy array to DataFrame conversion."""
        request = create_conversion_request(
            self.array_3d,
            DataFormat.NUMPY_ARRAY,
            DataFormat.PANDAS_DATAFRAME
        )
        
        result = self.converter.convert(request)
        
        assert result.success is True
        assert isinstance(result.converted_data, pd.DataFrame)
        assert len(result.warnings) > 0  # Should warn about flattening
        assert result.converted_data.shape[0] == self.array_3d.shape[0]
    
    def test_numpy_to_sparse_conversion(self):
        """Test NumPy array to sparse matrix conversion."""
        # Create sparse-like array
        sparse_array = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
        
        request = create_conversion_request(
            sparse_array,
            DataFormat.NUMPY_ARRAY,
            DataFormat.SCIPY_SPARSE
        )
        
        result = self.converter.convert(request)
        
        assert result.success is True
        assert sparse.issparse(result.converted_data)
        assert result.converted_data.shape == sparse_array.shape
        
        # Verify sparse structure is preserved
        dense_back = result.converted_data.toarray()
        np.testing.assert_array_equal(dense_back, sparse_array)
    
    def test_numpy_to_list_conversion(self):
        """Test NumPy array to list conversion."""
        request = create_conversion_request(
            self.array_2d,
            DataFormat.NUMPY_ARRAY,
            DataFormat.PYTHON_LIST
        )
        
        result = self.converter.convert(request)
        
        assert result.success is True
        assert isinstance(result.converted_data, list)
        assert result.converted_data == self.array_2d.tolist()
    
    def test_numpy_to_dict_conversion(self):
        """Test NumPy array to dictionary conversion."""
        request = create_conversion_request(
            self.array_2d,
            DataFormat.NUMPY_ARRAY,
            DataFormat.PYTHON_DICT
        )
        
        result = self.converter.convert(request)
        
        assert result.success is True
        assert isinstance(result.converted_data, dict)
        assert 'data' in result.converted_data
        assert 'shape' in result.converted_data
        assert 'dtype' in result.converted_data
        assert result.converted_data['shape'] == self.array_2d.shape
    
    def test_dataframe_to_numpy_conversion(self):
        """Test DataFrame to NumPy array conversion."""
        request = create_conversion_request(
            self.sample_df,
            DataFormat.PANDAS_DATAFRAME,
            DataFormat.NUMPY_ARRAY
        )
        
        result = self.converter.convert(request)
        
        assert result.success is True
        assert isinstance(result.converted_data, np.ndarray)
        assert result.converted_data.shape == self.sample_df.shape
        np.testing.assert_array_equal(result.converted_data, self.sample_df.values)
    
    def test_mixed_type_dataframe_to_numpy(self):
        """Test conversion of DataFrame with mixed types."""
        mixed_df = pd.DataFrame({
            'numbers': [1, 2, 3],
            'strings': ['a', 'b', 'c'],
            'bools': [True, False, True]
        })
        
        request = create_conversion_request(
            mixed_df,
            DataFormat.PANDAS_DATAFRAME,
            DataFormat.NUMPY_ARRAY
        )
        
        result = self.converter.convert(request)
        
        assert result.success is True
        assert len(result.warnings) > 0  # Should warn about non-numeric columns
        # Should only convert numeric columns
        assert result.converted_data.shape[1] == 1  # Only 'numbers' column
    
    def test_bidirectional_conversion(self):
        """Test bidirectional conversion consistency."""
        # Array -> DataFrame -> Array
        array_to_df_request = create_conversion_request(
            self.array_2d,
            DataFormat.NUMPY_ARRAY,
            DataFormat.PANDAS_DATAFRAME
        )
        
        df_result = self.converter.convert(array_to_df_request)
        assert df_result.success is True
        
        df_to_array_request = create_conversion_request(
            df_result.converted_data,
            DataFormat.PANDAS_DATAFRAME,
            DataFormat.NUMPY_ARRAY
        )
        
        array_result = self.converter.convert(df_to_array_request)
        assert array_result.success is True
        
        # Should be identical
        np.testing.assert_array_equal(array_result.converted_data, self.array_2d)
    
    def test_dtype_preservation(self):
        """Test data type preservation in conversions."""
        int_array = np.array([[1, 2], [3, 4]], dtype=np.int32)
        
        request = create_conversion_request(
            int_array,
            DataFormat.NUMPY_ARRAY,
            DataFormat.PYTHON_DICT
        )
        
        result = self.converter.convert(request)
        
        assert result.success is True
        assert result.converted_data['dtype'] == str(int_array.dtype)


class TestSparseMatrixConverter:
    """Test cases for SparseMatrixConverter."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.converter = SparseMatrixConverter()
        
        # Sample sparse matrices
        self.csr_matrix = sparse.csr_matrix([[1, 0, 2], [0, 3, 0], [4, 0, 5]])
        self.csc_matrix = sparse.csc_matrix([[1, 0, 0], [2, 3, 0], [0, 0, 4]])
        self.coo_matrix = sparse.coo_matrix([[1, 2, 0], [0, 0, 3], [0, 4, 5]])
        
        # Dense array for sparse conversion
        self.sparse_array = np.array([[1, 0, 0, 2], [0, 0, 3, 0], [4, 0, 0, 0], [0, 5, 0, 0]])
        
        # DataFrame for sparse conversion
        self.sparse_df = pd.DataFrame(self.sparse_array, columns=['a', 'b', 'c', 'd'])
    
    def test_initialization(self):
        """Test SparseMatrixConverter initialization."""
        assert self.converter.adapter_id == "sparse_converter"
        assert len(self.converter.supported_conversions) > 0
        
        conversions = self.converter.get_supported_conversions()
        assert (DataFormat.SCIPY_SPARSE, DataFormat.PANDAS_DATAFRAME) in conversions
        assert (DataFormat.PANDAS_DATAFRAME, DataFormat.SCIPY_SPARSE) in conversions
    
    def test_sparse_to_dataframe_conversion(self):
        """Test sparse matrix to DataFrame conversion."""
        request = create_conversion_request(
            self.csr_matrix,
            DataFormat.SCIPY_SPARSE,
            DataFormat.PANDAS_DATAFRAME
        )
        
        result = self.converter.convert(request)
        
        assert result.success is True
        assert isinstance(result.converted_data, pd.DataFrame)
        assert result.converted_data.shape == self.csr_matrix.shape
        assert len(result.warnings) > 0  # Should warn about sparsity loss
        
        # Check data integrity
        np.testing.assert_array_equal(result.converted_data.values, self.csr_matrix.toarray())
    
    def test_sparse_to_numpy_conversion(self):
        """Test sparse matrix to NumPy array conversion."""
        request = create_conversion_request(
            self.csc_matrix,
            DataFormat.SCIPY_SPARSE,
            DataFormat.NUMPY_ARRAY
        )
        
        result = self.converter.convert(request)
        
        assert result.success is True
        assert isinstance(result.converted_data, np.ndarray)
        assert result.converted_data.shape == self.csc_matrix.shape
        
        # Check data integrity
        np.testing.assert_array_equal(result.converted_data, self.csc_matrix.toarray())
    
    def test_sparse_to_dict_conversion(self):
        """Test sparse matrix to dictionary conversion."""
        request = create_conversion_request(
            self.coo_matrix,
            DataFormat.SCIPY_SPARSE,
            DataFormat.PYTHON_DICT
        )
        
        result = self.converter.convert(request)
        
        assert result.success is True
        assert isinstance(result.converted_data, dict)
        assert result.converted_data['format'] == 'scipy_sparse_coo'
        assert 'data' in result.converted_data
        assert 'row' in result.converted_data
        assert 'col' in result.converted_data
        assert result.converted_data['shape'] == self.coo_matrix.shape
    
    def test_dataframe_to_sparse_conversion(self):
        """Test DataFrame to sparse matrix conversion."""
        request = create_conversion_request(
            self.sparse_df,
            DataFormat.PANDAS_DATAFRAME,
            DataFormat.SCIPY_SPARSE
        )
        
        result = self.converter.convert(request)
        
        assert result.success is True
        assert sparse.issparse(result.converted_data)
        assert result.converted_data.shape == self.sparse_df.shape
        
        # Check data integrity
        np.testing.assert_array_equal(result.converted_data.toarray(), self.sparse_df.values)
    
    def test_numpy_to_sparse_conversion(self):
        """Test NumPy array to sparse matrix conversion."""
        request = create_conversion_request(
            self.sparse_array,
            DataFormat.NUMPY_ARRAY,
            DataFormat.SCIPY_SPARSE
        )
        
        result = self.converter.convert(request)
        
        assert result.success is True
        assert sparse.issparse(result.converted_data)
        assert result.converted_data.shape == self.sparse_array.shape
        
        # Check data integrity
        np.testing.assert_array_equal(result.converted_data.toarray(), self.sparse_array)
    
    def test_dict_to_sparse_restoration(self):
        """Test restoration of sparse matrix from dictionary."""
        # First convert to dict
        sparse_to_dict_request = create_conversion_request(
            self.csr_matrix,
            DataFormat.SCIPY_SPARSE,
            DataFormat.PYTHON_DICT
        )
        
        dict_result = self.converter.convert(sparse_to_dict_request)
        assert dict_result.success is True
        
        # Then restore from dict
        dict_to_sparse_request = create_conversion_request(
            dict_result.converted_data,
            DataFormat.PYTHON_DICT,
            DataFormat.SCIPY_SPARSE
        )
        
        sparse_result = self.converter.convert(dict_to_sparse_request)
        assert sparse_result.success is True
        assert sparse.issparse(sparse_result.converted_data)
        
        # Check data integrity
        np.testing.assert_array_equal(
            sparse_result.converted_data.toarray(),
            self.csr_matrix.toarray()
        )
    
    def test_density_threshold_warnings(self):
        """Test warnings for high-density data converted to sparse."""
        # Create dense-like array
        dense_array = np.random.randn(10, 10)  # High density
        
        request = create_conversion_request(
            dense_array,
            DataFormat.NUMPY_ARRAY,
            DataFormat.SCIPY_SPARSE
        )
        
        result = self.converter.convert(request)
        
        assert result.success is True
        # Should warn about high density
        assert any("dense data" in warning.lower() for warning in result.warnings)
    
    def test_different_sparse_formats(self):
        """Test handling of different sparse matrix formats."""
        formats = [self.csr_matrix, self.csc_matrix, self.coo_matrix]
        
        for sparse_mat in formats:
            request = create_conversion_request(
                sparse_mat,
                DataFormat.SCIPY_SPARSE,
                DataFormat.NUMPY_ARRAY
            )
            
            result = self.converter.convert(request)
            
            assert result.success is True
            assert isinstance(result.converted_data, np.ndarray)
            # All should produce the same dense result for equivalent matrices
    
    def test_1d_array_to_sparse_handling(self):
        """Test 1D array to sparse matrix conversion."""
        array_1d = np.array([1, 0, 2, 0, 3])
        
        request = create_conversion_request(
            array_1d,
            DataFormat.NUMPY_ARRAY,
            DataFormat.SCIPY_SPARSE
        )
        
        result = self.converter.convert(request)
        
        assert result.success is True
        assert sparse.issparse(result.converted_data)
        assert result.converted_data.shape == (len(array_1d), 1)  # Reshaped to 2D


class TestConversionOptions:
    """Test cases for ConversionOptions and factory functions."""
    
    def test_default_options(self):
        """Test default conversion options."""
        options = ConversionOptions()
        
        assert options.preserve_index is True
        assert options.preserve_columns is True
        assert options.handle_mixed_types is True
        assert options.quality_target == ConversionQuality.HIGH_FIDELITY
    
    def test_memory_efficient_options(self):
        """Test memory efficient options."""
        options = create_memory_efficient_options()
        
        assert options.memory_efficient is True
        assert options.chunk_size_rows == 5000
        assert options.quality_target == ConversionQuality.MODERATE
    
    def test_high_fidelity_options(self):
        """Test high fidelity options."""
        options = create_high_fidelity_options()
        
        assert options.preserve_index is True
        assert options.preserve_columns is True
        assert options.quality_target == ConversionQuality.HIGH_FIDELITY
    
    def test_streaming_options(self):
        """Test streaming options."""
        options = create_streaming_options(chunk_size=20000)
        
        assert options.chunk_size_rows == 20000
        assert options.memory_efficient is True


class TestFactoryFunctions:
    """Test cases for converter factory functions."""
    
    def test_create_pandas_converter(self):
        """Test pandas converter factory."""
        converter = create_pandas_converter()
        
        assert isinstance(converter, PandasConverter)
        assert converter.adapter_id == "pandas_converter"
    
    def test_create_numpy_converter(self):
        """Test numpy converter factory."""
        converter = create_numpy_converter()
        
        assert isinstance(converter, NumpyConverter)
        assert converter.adapter_id == "numpy_converter"
    
    def test_create_sparse_converter(self):
        """Test sparse converter factory."""
        converter = create_sparse_converter()
        
        assert isinstance(converter, SparseMatrixConverter)
        assert converter.adapter_id == "sparse_converter"
    
    def test_factory_with_custom_options(self):
        """Test factory functions with custom options."""
        custom_options = ConversionOptions(
            preserve_index=False,
            sparse_density_threshold=0.2
        )
        
        converter = create_pandas_converter(conversion_options=custom_options)
        
        assert converter.conversion_options.preserve_index is False
        assert converter.conversion_options.sparse_density_threshold == 0.2


class TestEdgeCases:
    """Test cases for edge cases and error conditions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.pandas_converter = PandasConverter()
        self.numpy_converter = NumpyConverter()
        self.sparse_converter = SparseMatrixConverter()
    
    def test_empty_dataframe_conversion(self):
        """Test conversion of empty DataFrame."""
        empty_df = pd.DataFrame()
        
        request = create_conversion_request(
            empty_df,
            DataFormat.PANDAS_DATAFRAME,
            DataFormat.NUMPY_ARRAY
        )
        
        result = self.pandas_converter.convert(request)
        
        # Should handle gracefully
        assert result.success is True or len(result.errors) > 0
    
    def test_single_value_conversions(self):
        """Test conversions with single values."""
        single_value_df = pd.DataFrame({'value': [42]})
        
        request = create_conversion_request(
            single_value_df,
            DataFormat.PANDAS_DATAFRAME,
            DataFormat.NUMPY_ARRAY
        )
        
        result = self.pandas_converter.convert(request)
        
        assert result.success is True
        assert result.converted_data.shape == (1, 1)
    
    def test_very_sparse_matrix_conversion(self):
        """Test conversion of very sparse matrices."""
        # Create very sparse matrix (99% zeros)
        very_sparse = sparse.random(1000, 1000, density=0.01, format='csr')
        
        request = create_conversion_request(
            very_sparse,
            DataFormat.SCIPY_SPARSE,
            DataFormat.PYTHON_DICT
        )
        
        result = self.sparse_converter.convert(request)
        
        assert result.success is True
        assert result.converted_data['format'] == 'scipy_sparse_coo'
    
    def test_unsupported_conversion_paths(self):
        """Test handling of unsupported conversion paths."""
        # Try to convert directly between incompatible formats
        request = create_conversion_request(
            pd.DataFrame({'a': [1, 2, 3]}),
            DataFormat.PANDAS_DATAFRAME,
            DataFormat.JSON  # Not supported by pandas converter
        )
        
        result = self.pandas_converter.convert(request)
        
        assert result.success is False
        assert len(result.errors) > 0
    
    def test_corrupted_data_handling(self):
        """Test handling of corrupted or invalid data."""
        # Test with None as source data
        request = create_conversion_request(
            None,
            DataFormat.PANDAS_DATAFRAME,
            DataFormat.NUMPY_ARRAY
        )
        
        result = self.pandas_converter.convert(request)
        
        assert result.success is False
        assert len(result.errors) > 0
    
    def test_memory_limit_simulation(self):
        """Test behavior under simulated memory constraints."""
        # This test would typically use memory profiling tools
        # For now, we test with large datasets
        large_df = pd.DataFrame(np.random.randn(10000, 100))
        
        request = create_conversion_request(
            large_df,
            DataFormat.PANDAS_DATAFRAME,
            DataFormat.NUMPY_ARRAY
        )
        
        result = self.pandas_converter.convert(request)
        
        # Should complete successfully for reasonably sized data
        assert result.success is True
        assert 'rows_converted' in result.performance_metrics


if __name__ == '__main__':
    pytest.main([__file__])