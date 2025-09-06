"""
Unit tests for enhanced type detection system in the Integration Shims Framework.

Tests cover:
- TypeDetectionEngine with format-specific detectors
- FormatDetectionResult and SchemaInfo functionality
- Format-specific detectors (pandas, numpy, time series, categorical)
- Conversion complexity assessment and recommendations
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, date
import tempfile
import json

from src.localdata_mcp.pipeline.integration.type_detection import (
    TypeDetectionEngine,
    FormatDetectionResult,
    SchemaInfo,
    PandasDataFrameDetector,
    NumpyArrayDetector,
    TimeSeriesDetector,
    CategoricalDetector,
    detect_data_format
)
from src.localdata_mcp.pipeline.integration.interfaces import (
    DataFormat,
    ValidationResult
)


class TestFormatDetectionResult:
    """Test cases for FormatDetectionResult dataclass."""
    
    def test_detection_result_creation(self):
        """Test FormatDetectionResult creation."""
        result = FormatDetectionResult(
            detected_format=DataFormat.PANDAS_DATAFRAME,
            confidence_score=0.95,
            alternative_formats=[(DataFormat.NUMPY_ARRAY, 0.3)],
            detection_time=0.05,
            sample_size=100
        )
        
        assert result.detected_format == DataFormat.PANDAS_DATAFRAME
        assert result.confidence_score == 0.95
        assert len(result.alternative_formats) == 1
        assert result.detection_time == 0.05
        assert result.sample_size == 100
        assert isinstance(result.warnings, list)


class TestSchemaInfo:
    """Test cases for SchemaInfo dataclass."""
    
    def test_schema_info_creation(self):
        """Test SchemaInfo creation."""
        schema = SchemaInfo(
            data_format=DataFormat.PANDAS_DATAFRAME,
            structure_type='tabular',
            columns={'A': 'int64', 'B': 'float64'},
            shape=(100, 2)
        )
        
        assert schema.data_format == DataFormat.PANDAS_DATAFRAME
        assert schema.structure_type == 'tabular'
        assert schema.columns['A'] == 'int64'
        assert schema.shape == (100, 2)
        assert isinstance(schema.creation_time, datetime)


class TestPandasDataFrameDetector:
    """Test cases for PandasDataFrameDetector."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.detector = PandasDataFrameDetector()
        self.sample_df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [1.1, 2.2, 3.3, 4.4, 5.5],
            'C': ['a', 'b', 'c', 'd', 'e']
        })
    
    def test_detect_pandas_dataframe(self):
        """Test detecting pandas DataFrame."""
        confidence, details = self.detector.detect_format(self.sample_df)
        
        assert confidence == 1.0
        assert 'shape' in details
        assert 'dtypes' in details
        assert 'memory_usage' in details
        assert details['shape'] == (5, 3)
    
    def test_detect_non_dataframe(self):
        """Test detecting non-DataFrame data."""
        confidence, details = self.detector.detect_format([1, 2, 3])
        
        assert confidence == 0.0
        assert details == {}
    
    def test_extract_schema(self):
        """Test schema extraction from DataFrame."""
        schema = self.detector.extract_schema(self.sample_df)
        
        assert schema.data_format == DataFormat.PANDAS_DATAFRAME
        assert schema.structure_type == 'tabular'
        assert schema.shape == (5, 3)
        assert len(schema.columns) == 3
        assert 'completeness' in schema.quality_metrics
        assert 'consistency' in schema.quality_metrics
        assert 'uniqueness' in schema.quality_metrics
    
    def test_quality_metrics_calculation(self):
        """Test quality metrics calculation."""
        # Create DataFrame with missing values
        df_with_nulls = pd.DataFrame({
            'A': [1, 2, None, 4, 5],
            'B': [1.1, None, 3.3, None, 5.5],
            'C': ['a', 'b', 'c', 'd', 'e']
        })
        
        schema = self.detector.extract_schema(df_with_nulls)
        
        # Completeness should be less than 1.0 due to missing values
        assert schema.quality_metrics['completeness'] < 1.0
        assert schema.null_info['total_nulls'] > 0
        assert schema.null_info['null_percentage'] > 0


class TestNumpyArrayDetector:
    """Test cases for NumpyArrayDetector."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.detector = NumpyArrayDetector()
        self.sample_array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    
    def test_detect_numpy_array(self):
        """Test detecting numpy array."""
        confidence, details = self.detector.detect_format(self.sample_array)
        
        assert confidence == 1.0
        assert 'shape' in details
        assert 'dtype' in details
        assert 'ndim' in details
        assert details['shape'] == (3, 3)
        assert details['ndim'] == 2
    
    def test_detect_non_array(self):
        """Test detecting non-array data."""
        confidence, details = self.detector.detect_format([1, 2, 3])
        
        assert confidence == 0.0
        assert details == {}
    
    def test_extract_schema(self):
        """Test schema extraction from numpy array."""
        schema = self.detector.extract_schema(self.sample_array)
        
        assert schema.data_format == DataFormat.NUMPY_ARRAY
        assert schema.structure_type == 'array'
        assert schema.shape == (3, 3)
        assert 'int' in schema.element_type  # Should contain 'int'
        assert 'completeness' in schema.quality_metrics
        assert 'density' in schema.quality_metrics
    
    def test_floating_point_array_schema(self):
        """Test schema extraction from floating point array with NaN."""
        float_array = np.array([[1.0, 2.0, np.nan], [4.0, np.inf, 6.0]])
        schema = self.detector.extract_schema(float_array)
        
        assert schema.null_info['nan_count'] > 0
        assert schema.null_info['inf_count'] > 0
        assert schema.quality_metrics['completeness'] < 1.0


class TestTimeSeriesDetector:
    """Test cases for TimeSeriesDetector."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.detector = TimeSeriesDetector()
        
        # Time series DataFrame with datetime index
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        self.timeseries_df = pd.DataFrame({
            'value': range(10),
            'category': ['A', 'B'] * 5
        }, index=dates)
        
        # Regular DataFrame without time index
        self.regular_df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6]
        })
    
    def test_detect_time_series_with_datetime_index(self):
        """Test detecting time series with datetime index."""
        confidence, details = self.detector.detect_format(self.timeseries_df)
        
        assert confidence > 0.7  # High confidence for datetime index
        assert details['has_datetime_index'] is True
        assert details['potential_frequency'] is not None
    
    def test_detect_regular_dataframe(self):
        """Test detecting regular DataFrame."""
        confidence, details = self.detector.detect_format(self.regular_df)
        
        assert confidence < 0.5  # Low confidence for non-time-series
        assert details['has_datetime_index'] is False
    
    def test_detect_dataframe_with_temporal_columns(self):
        """Test detecting DataFrame with temporal column names."""
        temporal_df = pd.DataFrame({
            'timestamp': ['2023-01-01', '2023-01-02', '2023-01-03'],
            'date_column': ['2023-01-01', '2023-01-02', '2023-01-03'],
            'value': [1, 2, 3]
        })
        
        confidence, details = self.detector.detect_format(temporal_df)
        
        assert confidence > 0.0  # Should detect temporal patterns
        assert len(details['temporal_columns']) > 0
    
    def test_extract_time_series_schema(self):
        """Test schema extraction from time series."""
        schema = self.detector.extract_schema(self.timeseries_df)
        
        assert schema.data_format == DataFormat.TIME_SERIES
        assert schema.additional_properties['temporal_index'] is True
        assert 'time_range' in schema.additional_properties
        assert 'frequency' in schema.additional_properties
    
    def test_frequency_inference(self):
        """Test frequency inference from time series."""
        frequency = self.detector._infer_frequency(self.timeseries_df)
        assert frequency == 'D'  # Daily frequency
        
        # Test with irregular index
        irregular_dates = pd.to_datetime(['2023-01-01', '2023-01-03', '2023-01-07'])
        irregular_df = pd.DataFrame({'value': [1, 2, 3]}, index=irregular_dates)
        
        frequency = self.detector._infer_frequency(irregular_df)
        assert frequency is None  # No regular frequency


class TestCategoricalDetector:
    """Test cases for CategoricalDetector."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.detector = CategoricalDetector()
        
        # DataFrame with categorical columns
        self.categorical_df = pd.DataFrame({
            'category': ['A', 'B', 'A', 'C', 'B', 'A'] * 10,  # Low cardinality
            'high_card': range(60),  # High cardinality
            'explicit_cat': pd.Categorical(['X', 'Y', 'Z'] * 20)  # Explicit categorical
        })
        
        # Categorical Series
        self.categorical_series = pd.Series(['red', 'blue', 'red', 'green', 'blue'] * 5)
    
    def test_detect_categorical_dataframe(self):
        """Test detecting categorical DataFrame."""
        confidence, details = self.detector.detect_format(self.categorical_df)
        
        assert confidence > 0.0
        assert len(details['categorical_columns']) >= 1  # At least 'category' and 'explicit_cat'
        assert 'category' in details['categorical_columns']
        assert 'explicit_cat' in details['categorical_columns']
        assert 'high_card' not in details['categorical_columns']  # High cardinality excluded
    
    def test_detect_categorical_series(self):
        """Test detecting categorical Series."""
        confidence, details = self.detector.detect_format(self.categorical_series)
        
        assert confidence > 0.5  # Should detect low cardinality
        assert details['unique_values'] == 3  # red, blue, green
        assert len(details['sample_values']) == 3
    
    def test_detect_explicit_categorical(self):
        """Test detecting explicit categorical type."""
        explicit_cat = pd.Categorical(['A', 'B', 'C'] * 10)
        confidence, details = self.detector.detect_format(explicit_cat)
        
        assert confidence == 1.0  # Perfect confidence for explicit categorical
    
    def test_extract_categorical_schema(self):
        """Test schema extraction from categorical data."""
        schema = self.detector.extract_schema(self.categorical_df)
        
        assert schema.data_format == DataFormat.CATEGORICAL
        assert schema.structure_type == 'array'  # For DataFrame it's treated as array structure


class TestTypeDetectionEngine:
    """Test cases for TypeDetectionEngine."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.engine = TypeDetectionEngine(
            confidence_threshold=0.7,
            enable_schema_inference=True,
            max_sample_size=1000
        )
        
        # Sample data for testing
        self.df_data = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [1.1, 2.2, 3.3, 4.4, 5.5],
            'C': ['a', 'b', 'c', 'd', 'e']
        })
        
        self.array_data = np.array([[1, 2], [3, 4], [5, 6]])
        
        dates = pd.date_range('2023-01-01', periods=5, freq='D')
        self.timeseries_data = pd.DataFrame({'value': range(5)}, index=dates)
    
    def test_engine_initialization(self):
        """Test TypeDetectionEngine initialization."""
        assert self.engine.confidence_threshold == 0.7
        assert self.engine.enable_schema_inference is True
        assert self.engine.max_sample_size == 1000
        assert len(self.engine._detectors) >= 4  # At least 4 built-in detectors
    
    def test_detect_pandas_dataframe(self):
        """Test detecting pandas DataFrame."""
        result = self.engine.detect_format(self.df_data)
        
        assert isinstance(result, FormatDetectionResult)
        assert result.detected_format == DataFormat.PANDAS_DATAFRAME
        assert result.confidence_score >= 0.7
        assert result.schema_info is not None
        assert result.sample_size == 5
        assert result.detection_time > 0
    
    def test_detect_numpy_array(self):
        """Test detecting numpy array."""
        result = self.engine.detect_format(self.array_data)
        
        assert result.detected_format == DataFormat.NUMPY_ARRAY
        assert result.confidence_score >= 0.7
        assert result.schema_info is not None
        assert result.schema_info.structure_type == 'array'
    
    def test_detect_time_series(self):
        """Test detecting time series."""
        result = self.engine.detect_format(self.timeseries_data)
        
        # Time series detector should have higher confidence than regular DataFrame detector
        assert result.detected_format in [DataFormat.TIME_SERIES, DataFormat.PANDAS_DATAFRAME]
        assert result.confidence_score > 0.0
    
    def test_format_compatibility_validation(self):
        """Test format compatibility validation."""
        # Test compatible format
        compatible_result = self.engine.validate_format_compatibility(
            self.df_data, 
            DataFormat.PANDAS_DATAFRAME
        )
        
        assert isinstance(compatible_result, ValidationResult)
        assert compatible_result.is_valid is True
        assert compatible_result.score > 0.7
        
        # Test incompatible format
        incompatible_result = self.engine.validate_format_compatibility(
            self.df_data,
            DataFormat.JSON
        )
        
        assert incompatible_result.is_valid is False
        assert len(incompatible_result.errors) > 0
    
    def test_conversion_requirements_inference(self):
        """Test conversion requirements inference."""
        requirements = self.engine.infer_conversion_requirements(
            self.df_data,
            DataFormat.NUMPY_ARRAY
        )
        
        assert 'source_format' in requirements
        assert 'target_format' in requirements
        assert 'conversion_needed' in requirements
        assert 'estimated_complexity' in requirements
        assert requirements['source_format'] == DataFormat.PANDAS_DATAFRAME
        assert requirements['target_format'] == DataFormat.NUMPY_ARRAY
        assert requirements['conversion_needed'] is True
    
    def test_caching_functionality(self):
        """Test detection result caching."""
        # First detection
        result1 = self.engine.detect_format(self.df_data)
        cache_size_after_first = len(self.engine._detection_cache)
        
        # Second detection of same data
        result2 = self.engine.detect_format(self.df_data)
        
        assert cache_size_after_first == 1
        assert len(self.engine._detection_cache) == 1  # No new cache entry
        assert result1.detected_format == result2.detected_format
    
    def test_data_sampling(self):
        """Test data sampling for large datasets."""
        # Create large DataFrame
        large_df = pd.DataFrame({
            'A': range(10000),
            'B': [f'item_{i}' for i in range(10000)]
        })
        
        sampled_data = self.engine._sample_data(large_df)
        
        assert len(sampled_data) <= self.engine.max_sample_size
        assert len(sampled_data) == self.engine.max_sample_size  # Should be exactly max_sample_size
    
    def test_fallback_detection(self):
        """Test fallback detection for unrecognized formats."""
        # Test with basic Python types
        dict_data = {'a': 1, 'b': 2}
        list_data = [1, 2, 3, 4, 5]
        string_data = "test string"
        
        dict_format, dict_confidence = self.engine._fallback_detection(dict_data)
        list_format, list_confidence = self.engine._fallback_detection(list_data)
        string_format, string_confidence = self.engine._fallback_detection(string_data)
        
        assert dict_format == DataFormat.PYTHON_DICT
        assert list_format == DataFormat.PYTHON_LIST
        assert dict_confidence > 0
        assert list_confidence > 0
        assert string_confidence > 0
    
    def test_schema_quality_validation(self):
        """Test schema quality validation warnings."""
        # Create DataFrame with quality issues
        poor_quality_df = pd.DataFrame({
            'A': [1, None, None, None, 5],  # High null percentage
            'B': [1.1, 2.2, 3.3, 4.4, 5.5]
        })
        
        result = self.engine.detect_format(poor_quality_df)
        
        if result.schema_info:
            warnings = self.engine._validate_schema_quality(result.schema_info)
            assert len(warnings) > 0  # Should have warnings about data quality


class TestConversionRecommendations:
    """Test cases for conversion recommendations."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.engine = TypeDetectionEngine()
    
    def test_pandas_to_numpy_recommendations(self):
        """Test recommendations for pandas to numpy conversion."""
        df_data = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        detection_result = self.engine.detect_format(df_data)
        
        recommendations = self.engine._generate_conversion_recommendations(
            detection_result,
            DataFormat.NUMPY_ARRAY
        )
        
        assert len(recommendations) > 0
        assert any('values attribute' in rec for rec in recommendations)
    
    def test_time_series_recommendations(self):
        """Test recommendations for time series conversion."""
        df_data = pd.DataFrame({'A': [1, 2, 3]})
        detection_result = self.engine.detect_format(df_data)
        
        recommendations = self.engine._generate_conversion_recommendations(
            detection_result,
            DataFormat.TIME_SERIES
        )
        
        assert len(recommendations) > 0
        assert any('datetime index' in rec for rec in recommendations)
    
    def test_conversion_complexity_assessment(self):
        """Test conversion complexity assessment."""
        # Simple conversion
        simple_complexity = self.engine._assess_conversion_complexity(
            DataFormat.PANDAS_DATAFRAME,
            DataFormat.NUMPY_ARRAY,
            None
        )
        assert simple_complexity == 'low'
        
        # Complex conversion
        complex_complexity = self.engine._assess_conversion_complexity(
            DataFormat.JSON,
            DataFormat.PANDAS_DATAFRAME,
            None
        )
        assert complex_complexity == 'high'
        
        # Same format
        same_complexity = self.engine._assess_conversion_complexity(
            DataFormat.PANDAS_DATAFRAME,
            DataFormat.PANDAS_DATAFRAME,
            None
        )
        assert same_complexity == 'none'


class TestUtilityFunction:
    """Test cases for utility functions."""
    
    def test_detect_data_format_function(self):
        """Test the utility detect_data_format function."""
        df_data = pd.DataFrame({'A': [1, 2, 3]})
        
        result = detect_data_format(
            df_data,
            confidence_threshold=0.8,
            include_schema=True
        )
        
        assert isinstance(result, FormatDetectionResult)
        assert result.detected_format == DataFormat.PANDAS_DATAFRAME
        assert result.schema_info is not None
        assert result.confidence_score >= 0.8


if __name__ == '__main__':
    pytest.main([__file__])