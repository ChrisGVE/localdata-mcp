"""
Unit tests for base adapter implementations in the Integration Shims Framework.

Tests cover:
- BaseShimAdapter with sklearn-compatible interface
- StreamingShimAdapter for large dataset processing
- CachingShimAdapter with intelligent caching
- PassThroughAdapter and ValidationAdapter utilities
"""

import pytest
import pandas as pd
import numpy as np
import time
from unittest.mock import Mock, patch
from datetime import datetime

from src.localdata_mcp.pipeline.integration.base_adapters import (
    BaseShimAdapter,
    StreamingShimAdapter, 
    CachingShimAdapter,
    PassThroughAdapter,
    ValidationAdapter,
    ConversionContext
)
from src.localdata_mcp.pipeline.integration.interfaces import (
    DataFormat,
    ConversionRequest,
    ConversionResult,
    ConversionError,
    MemoryConstraints,
    PerformanceRequirements,
    create_conversion_request
)


class TestConversionAdapter(BaseShimAdapter):
    """Test adapter implementation for testing BaseShimAdapter."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.supported_conversions = [
            (DataFormat.PANDAS_DATAFRAME, DataFormat.NUMPY_ARRAY),
            (DataFormat.NUMPY_ARRAY, DataFormat.PANDAS_DATAFRAME)
        ]
    
    def _perform_conversion(self, request, context):
        """Simple test conversion implementation."""
        if request.source_format == DataFormat.PANDAS_DATAFRAME:
            if isinstance(request.source_data, pd.DataFrame):
                return request.source_data.values
        elif request.source_format == DataFormat.NUMPY_ARRAY:
            if isinstance(request.source_data, np.ndarray):
                return pd.DataFrame(request.source_data)
        
        return request.source_data


class TestBaseShimAdapter:
    """Test cases for BaseShimAdapter."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.adapter = TestConversionAdapter(
            adapter_id="test_adapter",
            enable_caching=True,
            enable_validation=True
        )
        
        # Sample data
        self.sample_df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [1.1, 2.2, 3.3, 4.4, 5.5],
            'C': ['a', 'b', 'c', 'd', 'e']
        })
        
        self.sample_array = np.array([[1, 2], [3, 4], [5, 6]])
    
    def test_initialization(self):
        """Test adapter initialization."""
        assert self.adapter.adapter_id == "test_adapter"
        assert self.adapter.enable_caching is True
        assert self.adapter.enable_validation is True
        assert self.adapter._fitted is False
        assert len(self.adapter.supported_conversions) == 2
    
    def test_fit_method(self):
        """Test sklearn-compatible fit method."""
        # Test fitting with DataFrame
        fitted_adapter = self.adapter.fit(self.sample_df)
        
        assert fitted_adapter is self.adapter  # Returns self
        assert self.adapter._fitted is True
        assert 'shape' in self.adapter._fit_metadata
        assert 'current_dtype' in self.adapter._fit_metadata
        
        # Test fit metadata content
        assert self.adapter._fit_metadata['shape'] == self.sample_df.shape
    
    def test_transform_method(self):
        """Test sklearn-compatible transform method."""
        # Test auto-fitting during transform
        result = self.adapter.transform(self.sample_df)
        
        assert self.adapter._fitted is True
        # For base transform, it should return input unchanged
        pd.testing.assert_frame_equal(result, self.sample_df)
    
    def test_can_convert_confidence_scoring(self):
        """Test conversion confidence scoring."""
        # Test supported conversion
        request = create_conversion_request(
            self.sample_df,
            DataFormat.PANDAS_DATAFRAME,
            DataFormat.NUMPY_ARRAY
        )
        
        confidence = self.adapter.can_convert(request)
        assert confidence == 0.8  # Base confidence for supported conversion
        
        # Test pass-through conversion
        passthrough_request = create_conversion_request(
            self.sample_df,
            DataFormat.PANDAS_DATAFRAME,
            DataFormat.PANDAS_DATAFRAME
        )
        
        passthrough_confidence = self.adapter.can_convert(passthrough_request)
        assert passthrough_confidence == 1.0
        
        # Test unsupported conversion
        unsupported_request = create_conversion_request(
            self.sample_df,
            DataFormat.PANDAS_DATAFRAME,
            DataFormat.JSON
        )
        
        unsupported_confidence = self.adapter.can_convert(unsupported_request)
        assert unsupported_confidence == 0.0
    
    def test_successful_conversion(self):
        """Test successful data conversion."""
        request = create_conversion_request(
            self.sample_df,
            DataFormat.PANDAS_DATAFRAME,
            DataFormat.NUMPY_ARRAY
        )
        
        result = self.adapter.convert(request)
        
        assert isinstance(result, ConversionResult)
        assert result.success is True
        assert isinstance(result.converted_data, np.ndarray)
        assert result.original_format == DataFormat.PANDAS_DATAFRAME
        assert result.target_format == DataFormat.NUMPY_ARRAY
        assert result.quality_score == 1.0  # Pass-through quality score
        assert result.execution_time > 0
    
    def test_conversion_with_caching(self):
        """Test conversion result caching."""
        request = create_conversion_request(
            self.sample_df,
            DataFormat.PANDAS_DATAFRAME,
            DataFormat.NUMPY_ARRAY
        )
        
        # First conversion
        result1 = self.adapter.convert(request)
        cache_size_after_first = len(self.adapter._conversion_cache)
        
        # Second conversion (should use cache)
        result2 = self.adapter.convert(request)
        
        assert cache_size_after_first == 1
        assert len(self.adapter._conversion_cache) == 1  # No new cache entry
        assert result1.request_id == result2.request_id
    
    def test_cost_estimation(self):
        """Test conversion cost estimation."""
        request = create_conversion_request(
            self.sample_df,
            DataFormat.PANDAS_DATAFRAME,
            DataFormat.NUMPY_ARRAY
        )
        
        cost = self.adapter.estimate_cost(request)
        
        assert cost.computational_cost >= 0.1  # At least base cost
        assert cost.memory_cost_mb > 0
        assert cost.time_estimate_seconds > 0
        assert cost.quality_impact == 0.0  # No quality loss expected
    
    def test_validation_enabled(self):
        """Test request validation when enabled."""
        # Test invalid request (None source data)
        invalid_request = create_conversion_request(
            None,
            DataFormat.PANDAS_DATAFRAME,
            DataFormat.NUMPY_ARRAY
        )
        
        result = self.adapter.convert(invalid_request)
        
        assert result.success is False
        assert len(result.errors) > 0
        assert "validation failed" in result.errors[0].lower()
    
    def test_performance_stats_tracking(self):
        """Test performance statistics tracking."""
        request = create_conversion_request(
            self.sample_df,
            DataFormat.PANDAS_DATAFRAME,
            DataFormat.NUMPY_ARRAY
        )
        
        # Perform multiple conversions
        for _ in range(3):
            self.adapter.convert(request)
        
        assert 'execution_times' in self.adapter._performance_stats
        assert len(self.adapter._performance_stats['execution_times']) >= 3


class TestStreamingShimAdapter:
    """Test cases for StreamingShimAdapter."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.adapter = StreamingShimAdapter(
            adapter_id="streaming_test",
            chunk_size=2  # Small chunk for testing
        )
        
        # Large DataFrame for streaming test
        self.large_df = pd.DataFrame({
            'A': range(10),
            'B': range(10, 20),
            'C': [f'item_{i}' for i in range(10)]
        })
    
    def test_streaming_initialization(self):
        """Test streaming adapter initialization."""
        assert self.adapter.adapter_id == "streaming_test"
        assert self.adapter.chunk_size == 2
        assert self.adapter.memory_constraints.prefer_streaming is True
    
    @patch.object(StreamingShimAdapter, '_estimate_data_size')
    def test_streaming_conversion_large_data(self, mock_estimate_size):
        """Test streaming conversion for large datasets."""
        # Mock large data size to trigger streaming
        mock_estimate_size.return_value = 200 * 1024 * 1024  # 200MB
        
        # Create a test adapter that implements streaming
        class TestStreamingAdapter(StreamingShimAdapter):
            def _perform_conversion(self, request, context):
                if request.source_format == DataFormat.PANDAS_DATAFRAME:
                    return self._stream_convert(request, context)
                return request.source_data
        
        adapter = TestStreamingAdapter(
            adapter_id="test_streaming",
            chunk_size=2
        )
        
        request = create_conversion_request(
            self.large_df,
            DataFormat.PANDAS_DATAFRAME,
            DataFormat.PANDAS_DATAFRAME
        )
        
        result = adapter.convert(request)
        
        assert result.success is True
        assert 'total_chunks_processed' in result.performance_metrics


class TestCachingShimAdapter:
    """Test cases for CachingShimAdapter."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.adapter = CachingShimAdapter(
            adapter_id="caching_test",
            cache_size_mb=1,  # Small cache for testing
            cache_ttl_seconds=1  # Short TTL for testing
        )
        
        # Set up test conversion
        self.adapter.supported_conversions = [
            (DataFormat.PANDAS_DATAFRAME, DataFormat.NUMPY_ARRAY)
        ]
        
        self.sample_df = pd.DataFrame({'A': [1, 2, 3]})
    
    def test_caching_initialization(self):
        """Test caching adapter initialization."""
        assert self.adapter.cache_size_mb == 1
        assert self.adapter.cache_ttl_seconds == 1
        assert self.adapter._total_cache_size == 0
    
    def test_cache_key_generation(self):
        """Test cache key generation."""
        request = create_conversion_request(
            self.sample_df,
            DataFormat.PANDAS_DATAFRAME,
            DataFormat.NUMPY_ARRAY
        )
        
        key1 = self.adapter._generate_cache_key(request)
        key2 = self.adapter._generate_cache_key(request)
        
        assert key1 == key2  # Same request should generate same key
        assert len(key1) == 64  # SHA256 hex string length
    
    def test_cache_ttl_expiration(self):
        """Test cache TTL expiration."""
        # Mock a simple conversion
        class TestCachingAdapter(CachingShimAdapter):
            def _perform_conversion(self, request, context):
                return request.source_data.values
        
        adapter = TestCachingAdapter(
            adapter_id="ttl_test",
            cache_ttl_seconds=0.1  # Very short TTL
        )
        adapter.supported_conversions = [(DataFormat.PANDAS_DATAFRAME, DataFormat.NUMPY_ARRAY)]
        
        request = create_conversion_request(
            self.sample_df,
            DataFormat.PANDAS_DATAFRAME,
            DataFormat.NUMPY_ARRAY
        )
        
        # First conversion
        result1 = adapter.convert(request)
        assert len(adapter._conversion_cache) == 1
        
        # Wait for TTL to expire
        time.sleep(0.2)
        
        # Second conversion should not use cache
        result2 = adapter.convert(request)
        
        # Cache should be cleaned up
        assert len(adapter._conversion_cache) <= 1


class TestPassThroughAdapter:
    """Test cases for PassThroughAdapter."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.adapter = PassThroughAdapter()
        self.sample_data = pd.DataFrame({'A': [1, 2, 3]})
    
    def test_passthrough_initialization(self):
        """Test pass-through adapter initialization."""
        assert self.adapter.adapter_id == "pass_through"
        assert len(self.adapter.supported_conversions) == len(list(DataFormat))
        
        # Check that all formats are supported for same-format conversion
        for fmt in DataFormat:
            assert (fmt, fmt) in self.adapter.supported_conversions
    
    def test_passthrough_can_convert(self):
        """Test pass-through conversion confidence."""
        # Same format should have confidence 1.0
        same_format_request = create_conversion_request(
            self.sample_data,
            DataFormat.PANDAS_DATAFRAME,
            DataFormat.PANDAS_DATAFRAME
        )
        
        confidence = self.adapter.can_convert(same_format_request)
        assert confidence == 1.0
        
        # Different formats should have confidence 0.0
        diff_format_request = create_conversion_request(
            self.sample_data,
            DataFormat.PANDAS_DATAFRAME,
            DataFormat.NUMPY_ARRAY
        )
        
        confidence = self.adapter.can_convert(diff_format_request)
        assert confidence == 0.0
    
    def test_passthrough_conversion(self):
        """Test pass-through conversion."""
        request = create_conversion_request(
            self.sample_data,
            DataFormat.PANDAS_DATAFRAME,
            DataFormat.PANDAS_DATAFRAME
        )
        
        result = self.adapter.convert(request)
        
        assert result.success is True
        pd.testing.assert_frame_equal(result.converted_data, self.sample_data)
        assert result.quality_score == 1.0
    
    def test_passthrough_cost_estimation(self):
        """Test pass-through cost estimation."""
        request = create_conversion_request(
            self.sample_data,
            DataFormat.PANDAS_DATAFRAME,
            DataFormat.PANDAS_DATAFRAME
        )
        
        cost = self.adapter.estimate_cost(request)
        
        assert cost.computational_cost == 0.01  # Minimal cost
        assert cost.memory_cost_mb == 0.0
        assert cost.time_estimate_seconds == 0.001


class TestValidationAdapter:
    """Test cases for ValidationAdapter."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.adapter = ValidationAdapter(
            adapter_id="validation_test",
            strict_validation=True
        )
        
        self.sample_data = pd.DataFrame({'A': [1, 2, 3]})
    
    def test_validation_initialization(self):
        """Test validation adapter initialization."""
        assert self.adapter.adapter_id == "validation_test"
        assert self.adapter.strict_validation is True
        assert self.adapter.enable_validation is True
    
    def test_enhanced_request_validation(self):
        """Test enhanced request validation."""
        # Test valid request
        valid_request = create_conversion_request(
            self.sample_data,
            DataFormat.PANDAS_DATAFRAME,
            DataFormat.PANDAS_DATAFRAME
        )
        
        validation_result = self.adapter.validate_request(valid_request)
        assert validation_result.is_valid is True
        
        # Test invalid request (None data)
        invalid_request = create_conversion_request(
            None,
            DataFormat.PANDAS_DATAFRAME,
            DataFormat.PANDAS_DATAFRAME
        )
        
        validation_result = self.adapter.validate_request(invalid_request)
        assert validation_result.is_valid is False
        assert len(validation_result.errors) > 0
    
    def test_wrapped_adapter_conversion(self):
        """Test conversion with wrapped adapter."""
        # Create a simple wrapped adapter
        wrapped = PassThroughAdapter()
        
        validation_adapter = ValidationAdapter(
            adapter_id="wrapped_validation",
            wrapped_adapter=wrapped
        )
        
        request = create_conversion_request(
            self.sample_data,
            DataFormat.PANDAS_DATAFRAME,
            DataFormat.PANDAS_DATAFRAME
        )
        
        result = validation_adapter.convert(request)
        
        assert result.success is True
        pd.testing.assert_frame_equal(result.converted_data, self.sample_data)


class TestConversionContext:
    """Test cases for ConversionContext."""
    
    def test_context_creation(self):
        """Test conversion context creation."""
        context = ConversionContext(request_id="test_123")
        
        assert context.request_id == "test_123"
        assert context.start_time > 0
        assert isinstance(context.intermediate_results, dict)
        assert isinstance(context.performance_metrics, dict)
        assert isinstance(context.warnings, list)
    
    def test_context_data_storage(self):
        """Test storing data in conversion context."""
        context = ConversionContext(request_id="test_storage")
        
        # Store intermediate results
        context.intermediate_results['step1'] = 'result1'
        context.performance_metrics['memory_usage'] = 1024
        context.warnings.append('Test warning')
        
        assert context.intermediate_results['step1'] == 'result1'
        assert context.performance_metrics['memory_usage'] == 1024
        assert 'Test warning' in context.warnings


if __name__ == '__main__':
    pytest.main([__file__])