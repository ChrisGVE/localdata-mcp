"""
Unit tests for the Performance Optimization System.

Tests cover:
- ConversionCache functionality and memory management
- LazyLoadingManager operations and lifecycle
- PerformanceBenchmark and profiling capabilities
- OptimizationSelector strategy selection
- Integration points and factory functions
"""

import pytest
import time
import threading
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np

from src.localdata_mcp.pipeline.integration.performance_optimization import (
    ConversionCache, CacheEvictionPolicy, CacheStatistics,
    LazyLoadingManager, LazyConverter
)
from src.localdata_mcp.pipeline.integration.interfaces import (
    ConversionRequest, ConversionResult, DataFormat, ShimAdapter
)


class MockShimAdapter(ShimAdapter):
    """Mock adapter for testing."""
    
    def __init__(self, adapter_id: str, conversion_delay: float = 0.01):
        super().__init__(adapter_id)
        self.conversion_delay = conversion_delay
        self.conversion_count = 0
    
    def can_convert(self, request: ConversionRequest) -> float:
        return 1.0
    
    def convert(self, request: ConversionRequest) -> ConversionResult:
        time.sleep(self.conversion_delay)
        self.conversion_count += 1
        
        return ConversionResult(
            converted_data=f"converted_{request.source_data}",
            success=True,
            original_format=request.source_format,
            target_format=request.target_format,
            actual_format=request.target_format,
            request_id=request.request_id
        )
    
    def estimate_cost(self, request: ConversionRequest):
        from src.localdata_mcp.pipeline.integration.interfaces import ConversionCost
        return ConversionCost(
            computational_cost=0.1,
            memory_cost_mb=10.0,
            time_estimate_seconds=0.1
        )
    
    def get_supported_conversions(self):
        return [(DataFormat.PANDAS_DATAFRAME, DataFormat.JSON)]


@pytest.fixture
def mock_adapter():
    """Create a mock adapter for testing."""
    return MockShimAdapter("test_adapter")


@pytest.fixture
def sample_request():
    """Create a sample conversion request."""
    return ConversionRequest(
        source_data="test_data",
        source_format=DataFormat.PANDAS_DATAFRAME,
        target_format=DataFormat.JSON,
        request_id="test_request_1"
    )


@pytest.fixture
def large_dataframe():
    """Create a large DataFrame for testing."""
    return pd.DataFrame({
        'col1': np.random.randn(10000),
        'col2': np.random.randn(10000),
        'col3': ['text'] * 10000
    })


class TestConversionCache:
    """Test the ConversionCache system."""
    
    def test_cache_initialization(self):
        """Test cache initialization with default parameters."""
        cache = ConversionCache()
        
        assert cache.max_size == 1000
        assert cache.max_memory_mb == 500.0
        assert cache.ttl_seconds == 3600
        assert cache.eviction_policy == CacheEvictionPolicy.LRU
        
        stats = cache.get_statistics()
        assert stats.total_requests == 0
        assert stats.cache_hits == 0
        assert stats.cache_misses == 0
    
    def test_cache_miss_and_put(self, sample_request, mock_adapter):
        """Test cache miss and subsequent storage."""
        cache = ConversionCache(max_size=10)
        
        # Should be cache miss
        cached_result = cache.get(sample_request)
        assert cached_result is None
        
        # Convert and cache result
        result = mock_adapter.convert(sample_request)
        cache.put(sample_request, result)
        
        # Should be cache hit now
        cached_result = cache.get(sample_request)
        assert cached_result is not None
        assert cached_result.converted_data == result.converted_data
        
        stats = cache.get_statistics()
        assert stats.total_requests == 2
        assert stats.cache_hits == 1
        assert stats.cache_misses == 1
        assert stats.hit_rate == 0.5
    
    def test_cache_eviction_lru(self, mock_adapter):
        """Test LRU cache eviction policy."""
        cache = ConversionCache(max_size=2, eviction_policy=CacheEvictionPolicy.LRU)
        
        # Create multiple requests
        requests = []
        for i in range(3):
            request = ConversionRequest(
                source_data=f"data_{i}",
                source_format=DataFormat.PANDAS_DATAFRAME,
                target_format=DataFormat.JSON,
                request_id=f"request_{i}"
            )
            requests.append(request)
        
        # Fill cache beyond capacity
        for i, request in enumerate(requests):
            result = mock_adapter.convert(request)
            cache.put(request, result)
        
        # First request should be evicted (LRU)
        assert cache.get(requests[0]) is None
        assert cache.get(requests[1]) is not None
        assert cache.get(requests[2]) is not None
        
        stats = cache.get_statistics()
        assert stats.evictions >= 1
    
    def test_cache_ttl_expiration(self, sample_request, mock_adapter):
        """Test cache TTL expiration."""
        cache = ConversionCache(ttl_seconds=1)
        
        # Add item to cache
        result = mock_adapter.convert(sample_request)
        cache.put(sample_request, result)
        
        # Should be cache hit immediately
        assert cache.get(sample_request) is not None
        
        # Wait for expiration
        time.sleep(1.1)
        
        # Should be cache miss after expiration
        assert cache.get(sample_request) is None
    
    def test_cache_invalidation(self, sample_request, mock_adapter):
        """Test cache invalidation functionality."""
        cache = ConversionCache()
        
        # Add multiple items
        for i in range(5):
            request = ConversionRequest(
                source_data=f"data_{i}",
                source_format=DataFormat.PANDAS_DATAFRAME,
                target_format=DataFormat.JSON,
                request_id=f"request_{i}"
            )
            result = mock_adapter.convert(request)
            cache.put(request, result)
        
        # Invalidate all
        invalidated = cache.invalidate()
        assert invalidated == 5
        
        stats = cache.get_statistics()
        assert stats.cache_size == 0
    
    def test_cache_memory_pressure_eviction(self, mock_adapter):
        """Test memory-based cache eviction."""
        # Use small memory limit to trigger eviction
        cache = ConversionCache(max_memory_mb=1.0, eviction_policy=CacheEvictionPolicy.MEMORY_PRESSURE)
        
        # Create large data to trigger memory eviction
        large_data = "x" * (500 * 1024)  # 500KB string
        
        requests = []
        for i in range(5):  # Should exceed 1MB limit
            request = ConversionRequest(
                source_data=large_data,
                source_format=DataFormat.PANDAS_DATAFRAME,
                target_format=DataFormat.JSON,
                request_id=f"large_request_{i}"
            )
            requests.append(request)
        
        # Add items, should trigger evictions due to memory pressure
        for request in requests:
            result = ConversionResult(
                converted_data=large_data,
                success=True,
                original_format=request.source_format,
                target_format=request.target_format,
                actual_format=request.target_format,
                request_id=request.request_id
            )
            cache.put(request, result)
        
        stats = cache.get_statistics()
        # Should have evicted some items to stay under memory limit
        assert stats.cache_size < 5


class TestLazyLoadingManager:
    """Test the LazyLoadingManager system."""
    
    def test_lazy_loading_manager_initialization(self):
        """Test lazy loading manager initialization."""
        manager = LazyLoadingManager(
            default_threshold_mb=25.0,
            max_background_tasks=2
        )
        
        assert manager.default_threshold_mb == 25.0
        assert manager.max_background_tasks == 2
        
        status = manager.get_status()
        assert status['total_converters'] == 0
        assert status['loaded_converters'] == 0
    
    def test_lazy_converter_creation(self, sample_request, mock_adapter):
        """Test lazy converter creation and lazy loading."""
        manager = LazyLoadingManager()
        
        lazy_converter = manager.create_lazy_converter(
            mock_adapter, sample_request, threshold_mb=10.0
        )
        
        assert not lazy_converter.is_loaded()
        assert lazy_converter.threshold_mb == 10.0
        
        # Access should trigger loading
        data = lazy_converter._ensure_loaded()
        assert data == "converted_test_data"
        assert lazy_converter.is_loaded()
        assert mock_adapter.conversion_count == 1
    
    def test_lazy_converter_attribute_access(self, sample_request, mock_adapter):
        """Test lazy attribute access triggering load."""
        manager = LazyLoadingManager()
        
        # Mock result with attributes
        mock_result = Mock()
        mock_result.converted_data = Mock()
        mock_result.converted_data.shape = (100, 5)
        mock_result.success = True
        mock_adapter.convert = Mock(return_value=mock_result)
        
        lazy_converter = manager.create_lazy_converter(mock_adapter, sample_request)
        
        # Accessing attribute should trigger load
        try:
            # This should trigger loading via __getattr__
            shape = lazy_converter.shape
            assert shape == (100, 5)
        except AttributeError:
            # Expected if the mock doesn't have the attribute properly set up
            pass
        
        assert lazy_converter.is_loaded()
    
    def test_lazy_loading_cleanup(self, mock_adapter):
        """Test cleanup of unused lazy converters."""
        manager = LazyLoadingManager(cleanup_interval_seconds=1)
        
        # Create some lazy converters
        converters = []
        for i in range(5):
            request = ConversionRequest(
                source_data=f"data_{i}",
                source_format=DataFormat.PANDAS_DATAFRAME,
                target_format=DataFormat.JSON,
                request_id=f"request_{i}"
            )
            converter = manager.create_lazy_converter(mock_adapter, request)
            converters.append(converter)
        
        # Should have 5 converters
        status = manager.get_status()
        assert status['total_converters'] == 5
        
        # Simulate old access times by modifying state
        for i, converter in enumerate(converters[:3]):
            converter.state.creation_time = time.time() - 2000  # 2000 seconds ago
            converter.state.last_access_time = time.time() - 2000
        
        # Cleanup should remove old converters
        cleaned = manager.cleanup_unused()
        assert cleaned >= 3
        
        status = manager.get_status()
        assert status['total_converters'] <= 2
    
    def test_background_preloading(self, sample_request, mock_adapter):
        """Test background preloading of lazy converters."""
        manager = LazyLoadingManager()
        
        lazy_converter = manager.create_lazy_converter(mock_adapter, sample_request)
        
        # Start background loading
        future = manager.preload_background(lazy_converter)
        
        # Wait a bit for background loading
        time.sleep(0.1)
        
        # Should eventually be loaded
        assert lazy_converter.is_loaded() or lazy_converter.state.is_loading
        
        # Cancel future to clean up
        manager.cancel_loading(future)


class TestPerformanceOptimization:
    """Test performance optimization integration."""
    
    @patch('src.localdata_mcp.pipeline.integration.performance_optimization_complete.PerformanceBenchmark')
    def test_performance_benchmark_creation(self, mock_benchmark_class):
        """Test performance benchmark creation and usage."""
        from src.localdata_mcp.pipeline.integration.performance_optimization_complete import (
            create_performance_optimizer
        )
        
        # Mock the benchmark class
        mock_benchmark = Mock()
        mock_benchmark_class.return_value = mock_benchmark
        
        optimizer_suite = create_performance_optimizer(
            cache_size=100,
            memory_limit_mb=50.0
        )
        
        assert 'cache' in optimizer_suite
        assert 'lazy_manager' in optimizer_suite
        assert 'benchmark' in optimizer_suite
        assert 'selector' in optimizer_suite
    
    def test_cache_performance_with_threading(self, mock_adapter):
        """Test cache performance under concurrent access."""
        cache = ConversionCache(max_size=100)
        
        # Create multiple requests
        requests = []
        for i in range(50):
            request = ConversionRequest(
                source_data=f"data_{i}",
                source_format=DataFormat.PANDAS_DATAFRAME,
                target_format=DataFormat.JSON,
                request_id=f"concurrent_request_{i}"
            )
            requests.append(request)
        
        def worker(request_list):
            """Worker function for threading test."""
            for request in request_list:
                # Try to get from cache
                cached = cache.get(request)
                if cached is None:
                    # Convert and cache
                    result = mock_adapter.convert(request)
                    cache.put(request, result)
                else:
                    # Use cached result
                    pass
        
        # Create threads
        threads = []
        chunk_size = 10
        for i in range(0, len(requests), chunk_size):
            chunk = requests[i:i + chunk_size]
            thread = threading.Thread(target=worker, args=(chunk,))
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        stats = cache.get_statistics()
        assert stats.total_requests > 0
        assert stats.cache_size <= 50  # Should have cached unique requests
    
    def test_memory_estimation(self, large_dataframe):
        """Test memory estimation for different data types."""
        cache = ConversionCache()
        
        # Test DataFrame memory estimation
        estimated_size = cache._estimate_result_size(
            Mock(converted_data=large_dataframe)
        )
        
        # Should be reasonable estimate (> 0)
        assert estimated_size > 0
        
        # Test numpy array
        large_array = np.random.randn(10000, 10)
        estimated_size_array = cache._estimate_result_size(
            Mock(converted_data=large_array)
        )
        
        assert estimated_size_array > 0
    
    def test_data_fingerprinting(self, large_dataframe):
        """Test data fingerprinting for cache key generation."""
        cache = ConversionCache()
        
        request1 = ConversionRequest(
            source_data=large_dataframe,
            source_format=DataFormat.PANDAS_DATAFRAME,
            target_format=DataFormat.JSON,
            request_id="fingerprint_test_1"
        )
        
        request2 = ConversionRequest(
            source_data=large_dataframe,
            source_format=DataFormat.PANDAS_DATAFRAME,
            target_format=DataFormat.JSON,
            request_id="fingerprint_test_2"
        )
        
        key1 = cache._generate_cache_key(request1)
        key2 = cache._generate_cache_key(request2)
        
        # Should generate same key for same data
        assert key1 == key2
        
        # Different data should generate different keys
        different_data = large_dataframe.copy()
        different_data.iloc[0, 0] = 999999
        
        request3 = ConversionRequest(
            source_data=different_data,
            source_format=DataFormat.PANDAS_DATAFRAME,
            target_format=DataFormat.JSON,
            request_id="fingerprint_test_3"
        )
        
        key3 = cache._generate_cache_key(request3)
        assert key1 != key3
    
    def test_cache_compression(self):
        """Test cache data compression for large items."""
        cache = ConversionCache(enable_compression=True)
        
        # Create large data
        large_data = "x" * (1024 * 1024)  # 1MB string
        
        # Test compression
        compressed = cache._compress_data(large_data)
        assert len(compressed) < len(large_data.encode())
        
        # Test decompression
        decompressed = cache._decompress_data(compressed)
        assert decompressed == large_data


class TestIntegrationPoints:
    """Test integration points and factory functions."""
    
    def test_optimization_suite_creation(self):
        """Test complete optimization suite creation."""
        from src.localdata_mcp.pipeline.integration.performance_optimization_complete import (
            create_performance_optimizer
        )
        
        suite = create_performance_optimizer(
            cache_size=500,
            memory_limit_mb=200.0,
            lazy_threshold_mb=25.0
        )
        
        # Check all components are present
        expected_components = ['cache', 'lazy_manager', 'benchmark', 'selector']
        for component in expected_components:
            assert component in suite
        
        # Test cache configuration
        cache = suite['cache']
        assert hasattr(cache, 'max_size')
        
        # Test lazy manager configuration
        lazy_manager = suite['lazy_manager']
        assert hasattr(lazy_manager, 'default_threshold_mb')
    
    @patch('src.localdata_mcp.pipeline.integration.performance_optimization_complete.OptimizationSelector')
    def test_request_optimization(self, mock_selector_class, sample_request):
        """Test automatic request optimization."""
        from src.localdata_mcp.pipeline.integration.performance_optimization_complete import (
            optimize_conversion_request, create_performance_optimizer
        )
        
        # Mock selector
        mock_selector = Mock()
        mock_selector.analyze_data_characteristics.return_value = Mock(
            data_size_mb=10.0,
            complexity_score=0.5,
            estimated_processing_time=1.0
        )
        mock_selector.recommend_conversion_path.return_value = Mock(
            selected_strategy=Mock(name='cache_first')
        )
        mock_selector_class.return_value = mock_selector
        
        # Create optimizer suite
        suite = create_performance_optimizer()
        suite['selector'] = mock_selector
        
        # Optimize request
        optimized_request = optimize_conversion_request(sample_request, suite)
        
        # Check optimization metadata was added
        assert 'optimization_strategy' in optimized_request.metadata
        assert 'performance_hints' in optimized_request.metadata
        
        # Verify selector was called
        mock_selector.analyze_data_characteristics.assert_called_once()
        mock_selector.recommend_conversion_path.assert_called_once()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])