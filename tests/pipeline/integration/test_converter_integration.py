"""
Integration tests for core data format converters.

Tests comprehensive bidirectional conversion workflows and integration
with metadata management and type detection systems.
"""

import pytest
import pandas as pd
import numpy as np
from scipy import sparse
import time
import tempfile
import os

from src.localdata_mcp.pipeline.integration.converters import (
    PandasConverter, NumpyConverter, SparseMatrixConverter,
    ConversionOptions, ConversionQuality
)
from src.localdata_mcp.pipeline.integration.interfaces import (
    DataFormat, create_conversion_request
)
from src.localdata_mcp.pipeline.integration.type_detection import TypeDetectionEngine
from src.localdata_mcp.pipeline.integration.metadata_manager import MetadataManager


class TestBidirectionalConversions:
    """Test comprehensive bidirectional conversion workflows."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.pandas_converter = PandasConverter()
        self.numpy_converter = NumpyConverter()
        self.sparse_converter = SparseMatrixConverter()
        
        # Create comprehensive test datasets
        self.test_data = self._create_test_datasets()
    
    def _create_test_datasets(self):
        """Create comprehensive test datasets for integration testing."""
        datasets = {}
        
        # Standard tabular data
        datasets['tabular'] = pd.DataFrame({
            'int_col': [1, 2, 3, 4, 5],
            'float_col': [1.1, 2.2, 3.3, 4.4, 5.5],
            'bool_col': [True, False, True, False, True],
            'str_col': ['a', 'b', 'c', 'd', 'e']
        })
        
        # Time series data
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        datasets['timeseries'] = pd.DataFrame({
            'temperature': np.random.normal(20, 5, 100),
            'humidity': np.random.normal(60, 10, 100),
            'pressure': np.random.normal(1013, 20, 100)
        }, index=dates)
        
        # Sparse-like data
        sparse_data = np.zeros((20, 15))
        sparse_indices = [(i, j) for i in range(0, 20, 3) for j in range(0, 15, 4)]
        for i, j in sparse_indices:
            sparse_data[i, j] = np.random.randn()
        datasets['sparse'] = pd.DataFrame(sparse_data)
        
        # Categorical data
        categories = ['A', 'B', 'C'] * 20
        np.random.shuffle(categories)
        datasets['categorical'] = pd.DataFrame({
            'category': categories,
            'values': np.random.randn(60),
            'group': np.random.choice(['X', 'Y'], 60)
        })
        
        # High-dimensional data
        datasets['high_dim'] = pd.DataFrame(np.random.randn(50, 100))
        
        # Arrays
        datasets['array_2d'] = np.random.randn(10, 8)
        datasets['array_1d'] = np.random.randn(20)
        datasets['array_3d'] = np.random.randn(5, 4, 3)
        
        # Sparse matrices
        datasets['sparse_csr'] = sparse.random(15, 20, density=0.1, format='csr')
        datasets['sparse_csc'] = sparse.random(15, 20, density=0.1, format='csc')
        datasets['sparse_coo'] = sparse.random(15, 20, density=0.1, format='coo')
        
        return datasets
    
    def test_dataframe_numpy_roundtrip(self):
        """Test DataFrame -> NumPy -> DataFrame roundtrip."""
        original_df = self.test_data['tabular']
        
        # DataFrame to NumPy
        df_to_array_request = create_conversion_request(
            original_df,
            DataFormat.PANDAS_DATAFRAME,
            DataFormat.NUMPY_ARRAY
        )
        
        array_result = self.pandas_converter.convert(df_to_array_request)
        assert array_result.success is True
        
        # NumPy back to DataFrame
        array_to_df_request = create_conversion_request(
            array_result.converted_data,
            DataFormat.NUMPY_ARRAY,
            DataFormat.PANDAS_DATAFRAME
        )
        
        df_result = self.numpy_converter.convert(array_to_df_request)
        assert df_result.success is True
        
        # Verify data integrity for numeric columns
        original_numeric = original_df.select_dtypes(include=[np.number])
        result_df = df_result.converted_data
        
        assert result_df.shape == original_numeric.shape
        np.testing.assert_array_almost_equal(
            result_df.values, 
            original_numeric.values
        )
    
    def test_dataframe_sparse_roundtrip(self):
        """Test DataFrame -> Sparse -> DataFrame roundtrip."""
        original_df = self.test_data['sparse']
        
        # DataFrame to Sparse
        df_to_sparse_request = create_conversion_request(
            original_df,
            DataFormat.PANDAS_DATAFRAME,
            DataFormat.SCIPY_SPARSE
        )
        
        sparse_result = self.pandas_converter.convert(df_to_sparse_request)
        assert sparse_result.success is True
        assert sparse.issparse(sparse_result.converted_data)
        
        # Sparse back to DataFrame
        sparse_to_df_request = create_conversion_request(
            sparse_result.converted_data,
            DataFormat.SCIPY_SPARSE,
            DataFormat.PANDAS_DATAFRAME
        )
        
        df_result = self.sparse_converter.convert(sparse_to_df_request)
        assert df_result.success is True
        
        # Verify data integrity
        np.testing.assert_array_almost_equal(
            df_result.converted_data.values,
            original_df.values
        )
    
    def test_numpy_sparse_roundtrip(self):
        """Test NumPy -> Sparse -> NumPy roundtrip."""
        original_array = self.test_data['array_2d']
        # Make it sparse-like by zeroing many elements
        sparse_array = original_array.copy()
        mask = np.random.rand(*sparse_array.shape) > 0.3
        sparse_array[mask] = 0
        
        # NumPy to Sparse
        array_to_sparse_request = create_conversion_request(
            sparse_array,
            DataFormat.NUMPY_ARRAY,
            DataFormat.SCIPY_SPARSE
        )
        
        sparse_result = self.numpy_converter.convert(array_to_sparse_request)
        assert sparse_result.success is True
        
        # Sparse back to NumPy
        sparse_to_array_request = create_conversion_request(
            sparse_result.converted_data,
            DataFormat.SCIPY_SPARSE,
            DataFormat.NUMPY_ARRAY
        )
        
        array_result = self.sparse_converter.convert(sparse_to_array_request)
        assert array_result.success is True
        
        # Verify data integrity
        np.testing.assert_array_almost_equal(
            array_result.converted_data,
            sparse_array
        )
    
    def test_multi_hop_conversion_chain(self):
        """Test multi-hop conversion chains."""
        original_df = self.test_data['categorical']
        
        # Complex conversion chain: DataFrame -> Dict -> List -> NumPy -> DataFrame
        conversions = [
            (DataFormat.PANDAS_DATAFRAME, DataFormat.PYTHON_DICT, self.pandas_converter),
            (DataFormat.PYTHON_DICT, DataFormat.PYTHON_LIST, None),  # Custom handling needed
            (DataFormat.PYTHON_LIST, DataFormat.NUMPY_ARRAY, self.numpy_converter),
            (DataFormat.NUMPY_ARRAY, DataFormat.PANDAS_DATAFRAME, self.numpy_converter)
        ]
        
        current_data = original_df
        current_format = DataFormat.PANDAS_DATAFRAME
        
        # Execute conversion chain
        for target_format, converter in [(fmt, conv) for _, fmt, conv in conversions]:
            if converter is not None:
                request = create_conversion_request(
                    current_data,
                    current_format,
                    target_format
                )
                result = converter.convert(request)
                
                if result.success:
                    current_data = result.converted_data
                    current_format = target_format
                else:
                    # Some conversions may not be directly supported
                    break
        
        # At minimum, first conversion should work
        assert current_format != DataFormat.PANDAS_DATAFRAME  # Should have made some progress
    
    def test_timeseries_specialized_conversion(self):
        """Test time series specific conversion workflows."""
        ts_data = self.test_data['timeseries']
        
        # Time series to standard DataFrame
        ts_to_df_request = create_conversion_request(
            ts_data,
            DataFormat.TIME_SERIES,
            DataFormat.PANDAS_DATAFRAME
        )
        
        df_result = self.pandas_converter.convert(ts_to_df_request)
        assert df_result.success is True
        
        # Standard DataFrame back to time series
        df_to_ts_request = create_conversion_request(
            df_result.converted_data,
            DataFormat.PANDAS_DATAFRAME,
            DataFormat.TIME_SERIES
        )
        
        ts_result = self.pandas_converter.convert(df_to_ts_request)
        assert ts_result.success is True
        
        # Should preserve temporal structure
        assert isinstance(ts_result.converted_data.index, pd.DatetimeIndex)
    
    def test_high_dimensional_handling(self):
        """Test handling of high-dimensional data."""
        high_dim_df = self.test_data['high_dim']
        
        # High-dim DataFrame to sparse (should be efficient for sparse data)
        request = create_conversion_request(
            high_dim_df,
            DataFormat.PANDAS_DATAFRAME,
            DataFormat.SCIPY_SPARSE
        )
        
        result = self.pandas_converter.convert(request)
        assert result.success is True
        
        # Check that sparsity is detected and handled appropriately
        if 'density' in result.performance_metrics:
            density = result.performance_metrics['density']
            # For random data, density should be high
            assert density > 0.5


class TestMetadataIntegration:
    """Test integration with metadata management system."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.pandas_converter = PandasConverter()
        self.metadata_manager = MetadataManager()
        
        self.test_df = pd.DataFrame({
            'col_A': [1, 2, 3, 4],
            'col_B': ['x', 'y', 'z', 'w'],
            'col_C': [1.1, 2.2, 3.3, 4.4]
        })
    
    def test_metadata_preservation_in_conversion(self):
        """Test metadata preservation during conversion."""
        # Extract original metadata
        original_metadata = self.metadata_manager.extract_metadata(
            self.test_df, 
            DataFormat.PANDAS_DATAFRAME
        )
        
        # Perform conversion with metadata
        request = create_conversion_request(
            self.test_df,
            DataFormat.PANDAS_DATAFRAME,
            DataFormat.NUMPY_ARRAY,
            metadata=original_metadata
        )
        
        result = self.pandas_converter.convert(request)
        assert result.success is True
        
        # Check metadata preservation
        assert 'original_metadata' in result.metadata
        assert result.metadata['conversion_adapter'] == 'pandas_converter'
    
    def test_metadata_transformation(self):
        """Test metadata transformation during format conversion."""
        original_metadata = self.metadata_manager.extract_metadata(
            self.test_df,
            DataFormat.PANDAS_DATAFRAME
        )
        
        # Transform metadata for different target format
        transformed_metadata = self.metadata_manager.transform_metadata(
            original_metadata,
            DataFormat.PANDAS_DATAFRAME,
            DataFormat.NUMPY_ARRAY,
            'pandas_converter'
        )
        
        # Check transformation
        assert 'transformation_timestamp' in transformed_metadata
        assert transformed_metadata['target_format'] == DataFormat.NUMPY_ARRAY.value
        assert 'transformation_history' in transformed_metadata


class TestTypeDetectionIntegration:
    """Test integration with type detection system."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.type_detector = TypeDetectionEngine()
        self.pandas_converter = PandasConverter()
        
        self.mixed_data = pd.DataFrame({
            'numbers': [1, 2, 3],
            'strings': ['a', 'b', 'c'],
            'dates': pd.date_range('2024-01-01', periods=3)
        })
    
    def test_automatic_format_detection(self):
        """Test automatic format detection before conversion."""
        # Detect format of mixed data
        detection_result = self.type_detector.detect_format(self.mixed_data)
        
        assert detection_result.detected_format == DataFormat.PANDAS_DATAFRAME
        assert detection_result.confidence_score > 0.8
        
        # Use detection result to inform conversion
        if detection_result.schema_info:
            schema = detection_result.schema_info
            assert 'columns' in schema.additional_properties or schema.columns is not None
    
    def test_format_compatibility_validation(self):
        """Test format compatibility validation."""
        # Create numpy array
        test_array = np.array([[1, 2, 3], [4, 5, 6]])
        
        # Validate compatibility with expected format
        validation_result = self.type_detector.validate_format_compatibility(
            test_array,
            DataFormat.NUMPY_ARRAY
        )
        
        assert validation_result.is_valid is True
        assert validation_result.score > 0.9
        
        # Test incompatible format
        incompatible_validation = self.type_detector.validate_format_compatibility(
            test_array,
            DataFormat.PANDAS_DATAFRAME
        )
        
        assert incompatible_validation.is_valid is False


class TestPerformanceBenchmarks:
    """Performance benchmarks for converters."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.pandas_converter = PandasConverter()
        self.numpy_converter = NumpyConverter()
        self.sparse_converter = SparseMatrixConverter()
    
    def test_large_dataframe_conversion_performance(self):
        """Benchmark large DataFrame conversion performance."""
        # Create large DataFrame
        large_df = pd.DataFrame(np.random.randn(10000, 50))
        
        start_time = time.time()
        
        request = create_conversion_request(
            large_df,
            DataFormat.PANDAS_DATAFRAME,
            DataFormat.NUMPY_ARRAY
        )
        
        result = self.pandas_converter.convert(request)
        
        end_time = time.time()
        conversion_time = end_time - start_time
        
        assert result.success is True
        assert conversion_time < 5.0  # Should complete in reasonable time
        assert result.execution_time > 0
        
        # Check performance metrics
        assert 'rows_converted' in result.performance_metrics
        assert result.performance_metrics['rows_converted'] == len(large_df)
    
    def test_sparse_matrix_memory_efficiency(self):
        """Test memory efficiency of sparse matrix operations."""
        # Create very sparse matrix
        sparse_matrix = sparse.random(5000, 5000, density=0.01, format='csr')
        
        # Test conversion to dense formats
        request = create_conversion_request(
            sparse_matrix,
            DataFormat.SCIPY_SPARSE,
            DataFormat.PYTHON_DICT
        )
        
        result = self.sparse_converter.convert(request)
        
        assert result.success is True
        
        # Dictionary format should preserve sparsity information
        dict_data = result.converted_data
        assert dict_data['format'] == 'scipy_sparse_coo'
        assert len(dict_data['data']) < sparse_matrix.size  # Much smaller than full matrix
    
    def test_conversion_quality_metrics(self):
        """Test conversion quality metrics."""
        test_data = pd.DataFrame({
            'clean_data': [1.0, 2.0, 3.0, 4.0],
            'with_nulls': [1.0, np.nan, 3.0, 4.0],
            'mixed_types': [1, 'two', 3.0, '4']
        })
        
        request = create_conversion_request(
            test_data,
            DataFormat.PANDAS_DATAFRAME,
            DataFormat.NUMPY_ARRAY
        )
        
        result = self.pandas_converter.convert(request)
        
        # Quality score should reflect data issues
        assert 0.0 <= result.quality_score <= 1.0
        
        # Should have warnings about mixed types
        assert len(result.warnings) > 0
    
    def test_memory_usage_tracking(self):
        """Test memory usage tracking during conversions."""
        # Create moderately large dataset
        test_df = pd.DataFrame(np.random.randn(1000, 100))
        
        request = create_conversion_request(
            test_df,
            DataFormat.PANDAS_DATAFRAME,
            DataFormat.NUMPY_ARRAY
        )
        
        result = self.pandas_converter.convert(request)
        
        assert result.success is True
        
        # Should track memory-related metrics
        assert 'rows_converted' in result.performance_metrics
        assert result.execution_time > 0


class TestErrorHandlingIntegration:
    """Test comprehensive error handling across converters."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.converters = [
            PandasConverter(),
            NumpyConverter(),
            SparseMatrixConverter()
        ]
    
    def test_invalid_data_graceful_failure(self):
        """Test graceful failure with invalid data."""
        invalid_data_cases = [
            (None, DataFormat.PANDAS_DATAFRAME),
            ("string", DataFormat.NUMPY_ARRAY),
            ([], DataFormat.SCIPY_SPARSE),
            ({}, DataFormat.PANDAS_DATAFRAME)
        ]
        
        for invalid_data, source_format in invalid_data_cases:
            for converter in self.converters:
                if any(source_format == src for src, _ in converter.get_supported_conversions()):
                    request = create_conversion_request(
                        invalid_data,
                        source_format,
                        DataFormat.NUMPY_ARRAY  # Arbitrary target
                    )
                    
                    result = converter.convert(request)
                    
                    # Should fail gracefully
                    assert result.success is False
                    assert len(result.errors) > 0
                    assert result.converted_data is not None  # Should return original or safe default
    
    def test_unsupported_conversion_handling(self):
        """Test handling of unsupported conversion requests."""
        test_df = pd.DataFrame({'a': [1, 2, 3]})
        
        # Try unsupported conversion path
        request = create_conversion_request(
            test_df,
            DataFormat.PANDAS_DATAFRAME,
            DataFormat.UNKNOWN  # Unsupported target
        )
        
        result = self.converters[0].convert(request)  # PandasConverter
        
        assert result.success is False
        assert len(result.errors) > 0
        assert "unsupported" in result.errors[0].lower()
    
    def test_conversion_chain_error_propagation(self):
        """Test error propagation in conversion chains."""
        # Start with valid data
        valid_df = pd.DataFrame({'values': [1, 2, 3]})
        
        # First conversion should work
        request1 = create_conversion_request(
            valid_df,
            DataFormat.PANDAS_DATAFRAME,
            DataFormat.NUMPY_ARRAY
        )
        
        result1 = self.converters[0].convert(request1)
        assert result1.success is True
        
        # Second conversion with incompatible target should fail gracefully
        request2 = create_conversion_request(
            result1.converted_data,
            DataFormat.NUMPY_ARRAY,
            DataFormat.UNKNOWN  # Invalid target
        )
        
        result2 = self.converters[1].convert(request2)
        assert result2.success is False


if __name__ == '__main__':
    pytest.main([__file__])