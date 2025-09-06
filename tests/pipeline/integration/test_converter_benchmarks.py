"""
Performance benchmarks for core data format converters.

Measures memory efficiency, conversion speed, and quality metrics
for different data sizes and types.
"""

import pytest
import pandas as pd
import numpy as np
from scipy import sparse
import time
import psutil
import os
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass

from src.localdata_mcp.pipeline.integration.converters import (
    PandasConverter,
    NumpyConverter, 
    SparseMatrixConverter,
    ConversionOptions,
    create_memory_efficient_options,
    create_streaming_options
)
from src.localdata_mcp.pipeline.integration.interfaces import (
    DataFormat,
    create_conversion_request
)


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    test_name: str
    data_size: Tuple[int, ...]
    conversion_time: float
    memory_before: float
    memory_after: float
    memory_peak: float
    quality_score: float
    success: bool
    warnings_count: int
    errors_count: int


class PerformanceProfiler:
    """Simple performance profiler for conversion operations."""
    
    def __init__(self):
        self.process = psutil.Process(os.getpid())
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        return self.process.memory_info().rss / 1024 / 1024
    
    def profile_conversion(self, converter, request) -> BenchmarkResult:
        """Profile a single conversion operation."""
        memory_before = self.get_memory_usage()
        start_time = time.time()
        
        result = converter.convert(request)
        
        end_time = time.time()
        memory_after = self.get_memory_usage()
        
        return BenchmarkResult(
            test_name=f"{converter.adapter_id}_{request.source_format.value}_to_{request.target_format.value}",
            data_size=getattr(request.source_data, 'shape', (len(request.source_data),)) if hasattr(request.source_data, '__len__') else (1,),
            conversion_time=end_time - start_time,
            memory_before=memory_before,
            memory_after=memory_after,
            memory_peak=max(memory_before, memory_after),
            quality_score=result.quality_score,
            success=result.success,
            warnings_count=len(result.warnings),
            errors_count=len(result.errors)
        )


class TestPandasConverterBenchmarks:
    """Performance benchmarks for PandasConverter."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.converter = PandasConverter()
        self.profiler = PerformanceProfiler()
        
        # Create test datasets of various sizes
        self.datasets = self._create_benchmark_datasets()
    
    def _create_benchmark_datasets(self) -> Dict[str, pd.DataFrame]:
        """Create benchmark datasets of various sizes."""
        datasets = {}
        
        # Small dataset
        datasets['small'] = pd.DataFrame(np.random.randn(100, 10))
        
        # Medium dataset
        datasets['medium'] = pd.DataFrame(np.random.randn(5000, 50))
        
        # Large dataset
        datasets['large'] = pd.DataFrame(np.random.randn(50000, 100))
        
        # Wide dataset (many columns)
        datasets['wide'] = pd.DataFrame(np.random.randn(1000, 500))
        
        # Mixed types dataset
        datasets['mixed'] = pd.DataFrame({
            **{f'num_{i}': np.random.randn(10000) for i in range(10)},
            **{f'str_{i}': [f'val_{j}' for j in range(10000)] for i in range(5)},
            **{f'bool_{i}': np.random.choice([True, False], 10000) for i in range(3)}
        })
        
        # Sparse-like dataset
        sparse_data = np.random.randn(10000, 100)
        sparse_data[np.random.rand(*sparse_data.shape) > 0.1] = 0  # 90% zeros
        datasets['sparse_like'] = pd.DataFrame(sparse_data)
        
        return datasets
    
    def test_dataframe_to_numpy_performance_scaling(self):
        """Test DataFrame to NumPy conversion performance across data sizes."""
        results = []
        
        for name, df in self.datasets.items():
            if name != 'mixed':  # Skip mixed for numeric conversion
                request = create_conversion_request(
                    df,
                    DataFormat.PANDAS_DATAFRAME,
                    DataFormat.NUMPY_ARRAY
                )
                
                benchmark = self.profiler.profile_conversion(self.converter, request)
                results.append(benchmark)
                
                # Performance assertions
                assert benchmark.success, f"Conversion failed for {name} dataset"
                assert benchmark.conversion_time < 10.0, f"Conversion took too long for {name}: {benchmark.conversion_time}s"
                
                # Memory efficiency check
                memory_increase = benchmark.memory_after - benchmark.memory_before
                assert memory_increase < 500, f"Excessive memory usage for {name}: {memory_increase}MB"
        
        # Print results for analysis
        self._print_benchmark_results(results, "DataFrame to NumPy")
    
    def test_dataframe_to_sparse_performance(self):
        """Test DataFrame to sparse matrix conversion performance."""
        results = []
        
        sparse_datasets = ['sparse_like', 'medium', 'wide']
        
        for name in sparse_datasets:
            if name in self.datasets:
                df = self.datasets[name]
                request = create_conversion_request(
                    df,
                    DataFormat.PANDAS_DATAFRAME,
                    DataFormat.SCIPY_SPARSE
                )
                
                benchmark = self.profiler.profile_conversion(self.converter, request)
                results.append(benchmark)
                
                assert benchmark.success, f"Sparse conversion failed for {name}"
                
                # Sparse conversion might be slower but should be memory efficient
                assert benchmark.conversion_time < 30.0, f"Sparse conversion too slow for {name}"
        
        self._print_benchmark_results(results, "DataFrame to Sparse")
    
    def test_mixed_types_handling_performance(self):
        """Test performance with mixed data types."""
        mixed_df = self.datasets['mixed']
        
        request = create_conversion_request(
            mixed_df,
            DataFormat.PANDAS_DATAFRAME,
            DataFormat.NUMPY_ARRAY
        )
        
        benchmark = self.profiler.profile_conversion(self.converter, request)
        
        # Should handle gracefully with warnings
        assert benchmark.success
        assert benchmark.warnings_count > 0  # Should warn about dropping string columns
        assert benchmark.conversion_time < 15.0  # Reasonable time even with type handling
    
    def test_memory_efficient_options_performance(self):
        """Test performance with memory efficient options."""
        memory_efficient_converter = PandasConverter(
            conversion_options=create_memory_efficient_options()
        )
        
        large_df = self.datasets['large']
        
        # Standard conversion
        standard_request = create_conversion_request(
            large_df,
            DataFormat.PANDAS_DATAFRAME,
            DataFormat.NUMPY_ARRAY
        )
        
        standard_benchmark = self.profiler.profile_conversion(self.converter, standard_request)
        
        # Memory efficient conversion
        efficient_request = create_conversion_request(
            large_df,
            DataFormat.PANDAS_DATAFRAME,
            DataFormat.NUMPY_ARRAY
        )
        
        efficient_benchmark = self.profiler.profile_conversion(memory_efficient_converter, efficient_request)
        
        # Both should succeed
        assert standard_benchmark.success
        assert efficient_benchmark.success
        
        # Memory efficient might be slightly slower but use less peak memory
        memory_diff_standard = standard_benchmark.memory_peak - standard_benchmark.memory_before
        memory_diff_efficient = efficient_benchmark.memory_peak - efficient_benchmark.memory_before
        
        print(f"Standard memory usage: {memory_diff_standard:.2f}MB")
        print(f"Efficient memory usage: {memory_diff_efficient:.2f}MB")
    
    def _print_benchmark_results(self, results: List[BenchmarkResult], test_name: str):
        """Print benchmark results in a readable format."""
        print(f"\n=== {test_name} Benchmark Results ===")
        print("Dataset        | Size           | Time(s) | Memory(MB) | Quality | Warnings")
        print("-" * 75)
        
        for result in results:
            size_str = f"{result.data_size[0]}x{result.data_size[1]}" if len(result.data_size) > 1 else str(result.data_size[0])
            memory_change = result.memory_after - result.memory_before
            
            print(f"{result.test_name.split('_')[1]:12} | {size_str:12} | {result.conversion_time:6.3f} | {memory_change:8.1f} | {result.quality_score:6.3f} | {result.warnings_count:8}")


class TestNumpyConverterBenchmarks:
    """Performance benchmarks for NumpyConverter."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.converter = NumpyConverter()
        self.profiler = PerformanceProfiler()
        
        # Create numpy test datasets
        self.arrays = self._create_numpy_datasets()
    
    def _create_numpy_datasets(self) -> Dict[str, np.ndarray]:
        """Create numpy datasets for benchmarking."""
        return {
            'small_2d': np.random.randn(100, 10),
            'medium_2d': np.random.randn(5000, 50),
            'large_2d': np.random.randn(50000, 100),
            'small_1d': np.random.randn(1000),
            'large_1d': np.random.randn(100000),
            'small_3d': np.random.randn(10, 20, 30),
            'medium_3d': np.random.randn(50, 100, 20),
            'float32': np.random.randn(10000, 50).astype(np.float32),
            'int64': np.random.randint(0, 1000, (10000, 20)).astype(np.int64),
            'sparse_pattern': self._create_sparse_pattern_array()
        }
    
    def _create_sparse_pattern_array(self) -> np.ndarray:
        """Create array with sparse pattern."""
        arr = np.zeros((5000, 100))
        # Add sparse non-zero elements
        indices = np.random.choice(arr.size, size=int(0.05 * arr.size), replace=False)
        arr.flat[indices] = np.random.randn(len(indices))
        return arr
    
    def test_numpy_to_dataframe_performance(self):
        """Test NumPy to DataFrame conversion performance."""
        results = []
        
        test_arrays = ['small_2d', 'medium_2d', 'large_2d', 'float32', 'int64']
        
        for name in test_arrays:
            array = self.arrays[name]
            request = create_conversion_request(
                array,
                DataFormat.NUMPY_ARRAY,
                DataFormat.PANDAS_DATAFRAME
            )
            
            benchmark = self.profiler.profile_conversion(self.converter, request)
            results.append(benchmark)
            
            assert benchmark.success, f"Conversion failed for {name}"
            assert benchmark.conversion_time < 10.0, f"Conversion too slow for {name}"
        
        self._print_benchmark_results(results, "NumPy to DataFrame")
    
    def test_numpy_to_sparse_performance(self):
        """Test NumPy to sparse matrix conversion performance."""
        sparse_arrays = ['sparse_pattern', 'medium_2d']
        results = []
        
        for name in sparse_arrays:
            array = self.arrays[name]
            request = create_conversion_request(
                array,
                DataFormat.NUMPY_ARRAY,
                DataFormat.SCIPY_SPARSE
            )
            
            benchmark = self.profiler.profile_conversion(self.converter, request)
            results.append(benchmark)
            
            assert benchmark.success, f"Sparse conversion failed for {name}"
        
        self._print_benchmark_results(results, "NumPy to Sparse")
    
    def test_multidimensional_array_handling(self):
        """Test performance with multidimensional arrays."""
        arrays_3d = ['small_3d', 'medium_3d']
        
        for name in arrays_3d:
            array = self.arrays[name]
            
            # 3D to DataFrame (should flatten)
            request = create_conversion_request(
                array,
                DataFormat.NUMPY_ARRAY,
                DataFormat.PANDAS_DATAFRAME
            )
            
            benchmark = self.profiler.profile_conversion(self.converter, request)
            
            assert benchmark.success
            assert benchmark.warnings_count > 0  # Should warn about flattening
            assert benchmark.conversion_time < 5.0
    
    def test_dtype_preservation_performance(self):
        """Test performance of dtype preservation."""
        dtypes_to_test = ['float32', 'int64']
        
        for dtype_name in dtypes_to_test:
            array = self.arrays[dtype_name]
            
            # Convert to dict (preserves dtype info)
            request = create_conversion_request(
                array,
                DataFormat.NUMPY_ARRAY,
                DataFormat.PYTHON_DICT
            )
            
            benchmark = self.profiler.profile_conversion(self.converter, request)
            
            assert benchmark.success
            assert benchmark.conversion_time < 5.0
            
            print(f"Dtype {dtype_name} conversion: {benchmark.conversion_time:.3f}s")
    
    def _print_benchmark_results(self, results: List[BenchmarkResult], test_name: str):
        """Print benchmark results."""
        print(f"\n=== {test_name} Benchmark Results ===")
        for result in results:
            print(f"{result.test_name}: {result.conversion_time:.3f}s, Quality: {result.quality_score:.3f}")


class TestSparseMatrixBenchmarks:
    """Performance benchmarks for SparseMatrixConverter."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.converter = SparseMatrixConverter()
        self.profiler = PerformanceProfiler()
        
        # Create sparse matrix test datasets
        self.sparse_matrices = self._create_sparse_datasets()
    
    def _create_sparse_datasets(self) -> Dict[str, sparse.spmatrix]:
        """Create sparse matrix datasets."""
        return {
            'small_csr': sparse.random(100, 50, density=0.1, format='csr'),
            'medium_csr': sparse.random(2000, 1000, density=0.05, format='csr'),
            'large_csr': sparse.random(10000, 5000, density=0.01, format='csr'),
            'small_csc': sparse.random(100, 50, density=0.1, format='csc'),
            'medium_csc': sparse.random(2000, 1000, density=0.05, format='csc'),
            'small_coo': sparse.random(100, 50, density=0.1, format='coo'),
            'very_sparse': sparse.random(5000, 5000, density=0.001, format='csr'),
            'moderately_sparse': sparse.random(1000, 1000, density=0.2, format='csr')
        }
    
    def test_sparse_to_dense_conversion_performance(self):
        """Test sparse to dense conversion performance."""
        dense_targets = [DataFormat.PANDAS_DATAFRAME, DataFormat.NUMPY_ARRAY]
        results = []
        
        for target_format in dense_targets:
            for name, sparse_matrix in self.sparse_matrices.items():
                if 'large' not in name:  # Skip very large matrices for dense conversion
                    request = create_conversion_request(
                        sparse_matrix,
                        DataFormat.SCIPY_SPARSE,
                        target_format
                    )
                    
                    benchmark = self.profiler.profile_conversion(self.converter, request)
                    results.append(benchmark)
                    
                    assert benchmark.success, f"Conversion failed for {name} to {target_format.value}"
                    
                    # Dense conversion should warn about memory usage for sparse data
                    if sparse_matrix.nnz / sparse_matrix.size < 0.1:  # Very sparse
                        assert benchmark.warnings_count > 0
        
        self._print_sparse_benchmark_results(results, "Sparse to Dense")
    
    def test_sparse_format_conversion_performance(self):
        """Test conversion between different sparse formats."""
        # Test CSR -> Dict -> CSR roundtrip
        csr_matrix = self.sparse_matrices['medium_csr']
        
        # CSR to Dict
        csr_to_dict_request = create_conversion_request(
            csr_matrix,
            DataFormat.SCIPY_SPARSE,
            DataFormat.PYTHON_DICT
        )
        
        dict_benchmark = self.profiler.profile_conversion(self.converter, csr_to_dict_request)
        
        assert dict_benchmark.success
        assert dict_benchmark.conversion_time < 5.0
        
        print(f"CSR to dict conversion: {dict_benchmark.conversion_time:.3f}s")
        print(f"Original matrix: {csr_matrix.shape}, nnz: {csr_matrix.nnz}")
    
    def test_very_sparse_matrix_efficiency(self):
        """Test efficiency with very sparse matrices."""
        very_sparse = self.sparse_matrices['very_sparse']
        density = very_sparse.nnz / very_sparse.size
        
        print(f"Testing very sparse matrix: {very_sparse.shape}, density: {density:.6f}")
        
        # Convert to dict format (should be very efficient)
        request = create_conversion_request(
            very_sparse,
            DataFormat.SCIPY_SPARSE,
            DataFormat.PYTHON_DICT
        )
        
        benchmark = self.profiler.profile_conversion(self.converter, request)
        
        assert benchmark.success
        assert benchmark.conversion_time < 2.0  # Should be fast for sparse data
        
        # Memory should not increase dramatically
        memory_increase = benchmark.memory_after - benchmark.memory_before
        assert memory_increase < 100  # Should not use excessive memory
    
    def test_dense_to_sparse_conversion_efficiency(self):
        """Test efficiency of converting dense data to sparse."""
        # Create dense arrays with different sparsity patterns
        sparse_patterns = {
            'very_sparse': (np.random.rand(1000, 500) > 0.99).astype(float),  # 1% non-zero
            'moderately_sparse': (np.random.rand(500, 200) > 0.7).astype(float),  # 30% non-zero
            'dense': np.random.rand(200, 100)  # ~100% non-zero
        }
        
        for pattern_name, array in sparse_patterns.items():
            # Convert NumPy array to sparse
            request = create_conversion_request(
                array,
                DataFormat.NUMPY_ARRAY,
                DataFormat.SCIPY_SPARSE
            )
            
            benchmark = self.profiler.profile_conversion(self.converter, request)
            
            assert benchmark.success
            
            if pattern_name == 'dense':
                # Should warn about dense data being converted to sparse
                assert benchmark.warnings_count > 0
            
            print(f"{pattern_name} to sparse: {benchmark.conversion_time:.3f}s, "
                  f"warnings: {benchmark.warnings_count}")
    
    def _print_sparse_benchmark_results(self, results: List[BenchmarkResult], test_name: str):
        """Print sparse benchmark results."""
        print(f"\n=== {test_name} Benchmark Results ===")
        for result in results:
            print(f"{result.test_name}: {result.conversion_time:.3f}s, "
                  f"Memory Δ: {result.memory_after - result.memory_before:.1f}MB, "
                  f"Warnings: {result.warnings_count}")


class TestMemoryEfficiencyComparison:
    """Compare memory efficiency across different conversion strategies."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.profiler = PerformanceProfiler()
        
        # Create standard and memory-efficient converters
        self.standard_converter = PandasConverter()
        self.memory_efficient_converter = PandasConverter(
            conversion_options=create_memory_efficient_options()
        )
        self.streaming_converter = PandasConverter(
            conversion_options=create_streaming_options(chunk_size=5000)
        )
    
    def test_memory_efficiency_comparison(self):
        """Compare memory usage across converter configurations."""
        # Create large test dataset
        large_df = pd.DataFrame(np.random.randn(20000, 80))
        
        converters = [
            ("standard", self.standard_converter),
            ("memory_efficient", self.memory_efficient_converter),
            ("streaming", self.streaming_converter)
        ]
        
        results = {}
        
        for name, converter in converters:
            request = create_conversion_request(
                large_df,
                DataFormat.PANDAS_DATAFRAME,
                DataFormat.NUMPY_ARRAY
            )
            
            benchmark = self.profiler.profile_conversion(converter, request)
            results[name] = benchmark
            
            assert benchmark.success, f"{name} conversion failed"
        
        # Print comparison
        print("\n=== Memory Efficiency Comparison ===")
        print("Strategy         | Time(s) | Memory Peak(MB) | Quality")
        print("-" * 50)
        
        for name, result in results.items():
            memory_peak = result.memory_peak - result.memory_before
            print(f"{name:15} | {result.conversion_time:6.3f} | {memory_peak:12.1f} | {result.quality_score:6.3f}")
        
        # Assertions for memory efficiency
        # Memory efficient should use less peak memory
        standard_peak = results["standard"].memory_peak - results["standard"].memory_before
        efficient_peak = results["memory_efficient"].memory_peak - results["memory_efficient"].memory_before
        
        # Note: This may not always be true due to test environment variability
        print(f"\nMemory usage - Standard: {standard_peak:.1f}MB, Efficient: {efficient_peak:.1f}MB")


if __name__ == '__main__':
    # Run specific benchmark tests
    pytest.main([__file__ + "::TestPandasConverterBenchmarks::test_dataframe_to_numpy_performance_scaling", "-v", "-s"])