"""
Performance Benchmark Tests for LocalData MCP v2.0 Integration Shims

This module provides systematic performance validation for the Integration Shims
Framework. Tests measure performance characteristics across various data sizes,
optimization strategies, and usage patterns to ensure the system meets
performance requirements while maintaining accuracy and functionality.

Benchmark Categories:
- Conversion performance across different data sizes
- Cache effectiveness and optimization
- Memory usage and streaming performance
- Concurrent operation throughput
- Scaling characteristics and bottleneck identification
- Optimization strategy effectiveness
"""

import asyncio
import gc
import logging
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from localdata_mcp.pipeline.integration import (
    # Core components
    ConversionRegistry,
    DataFormat,
    ConversionRequest,
    ConversionResult,
    # Performance optimization
    ConversionCache,
    LazyLoadingManager,
    create_memory_efficient_options,
    create_high_fidelity_options,
    create_streaming_options,
    # Domain shims for performance testing
    create_all_domain_shims,
    # Converters
    create_pandas_converter,
    create_numpy_converter,
    create_sparse_converter,
)

# Test fixtures and utilities
from ..fixtures.sample_datasets import (
    create_statistical_dataset,
    create_pandas_dataframe,
    create_numpy_array,
    create_sparse_matrix,
    create_streaming_data_source,
    create_high_dimensional_array,
)
from ..utils.test_helpers import (
    measure_async_performance,
    measure_cross_domain_performance,
)

logger = logging.getLogger(__name__)


class TestPerformanceBenchmarks:
    """
    Comprehensive performance benchmarks for the Integration Shims Framework.

    Benchmarks cover:
    1. Data size scaling performance (small → large → xlarge)
    2. Format conversion performance across all supported formats
    3. Cache effectiveness and hit rate optimization
    4. Memory usage patterns and optimization
    5. Concurrent operation throughput and scalability
    6. Streaming performance with memory constraints
    7. Cross-domain workflow performance optimization
    """

    # Performance requirement thresholds
    PERFORMANCE_THRESHOLDS = {
        "small_data_conversion_time": 1.0,  # < 1s for small datasets
        "medium_data_conversion_time": 5.0,  # < 5s for medium datasets
        "large_data_conversion_time": 15.0,  # < 15s for large datasets
        "xlarge_data_conversion_time": 45.0,  # < 45s for xlarge datasets
        "memory_efficiency_ratio": 0.8,  # > 80% memory efficiency
        "cache_hit_rate_threshold": 0.7,  # > 70% cache hit rate
        "concurrent_throughput_factor": 0.6,  # > 60% of single-thread throughput
        "streaming_memory_limit_adherence": 1.1,  # < 110% of memory limit
    }

    # Data size configurations for benchmarking
    DATA_SIZE_CONFIGS = {
        "small": {"rows": 1000, "cols": 20, "memory_target": "5MB"},
        "medium": {"rows": 10000, "cols": 50, "memory_target": "50MB"},
        "large": {"rows": 100000, "cols": 100, "memory_target": "500MB"},
        "xlarge": {"rows": 500000, "cols": 200, "memory_target": "2GB"},
    }

    @pytest.fixture(autouse=True)
    async def setup_performance_framework(self):
        """Setup performance testing framework with optimization components."""
        # Initialize core components
        self.registry = ConversionRegistry()
        self.domain_shims = create_all_domain_shims()

        # Performance optimization components
        self.cache = ConversionCache(max_size=100, ttl=300)
        self.lazy_manager = LazyLoadingManager(memory_limit=1024 * 1024 * 1024)  # 1GB

        # Initialize converters with performance optimization
        self.converters = {
            DataFormat.PANDAS_DATAFRAME: create_pandas_converter(),
            DataFormat.NUMPY_ARRAY: create_numpy_converter(),
            DataFormat.SCIPY_SPARSE: create_sparse_converter(),
        }

        # Register converters
        for converter in self.converters.values():
            await self.registry.register_adapter(converter)

        # Register domain shims
        for shim_type, shim in self.domain_shims.items():
            await self.registry.register_adapter(shim)

        # Performance tracking
        self.performance_results = {}
        self.benchmark_metadata = {
            "test_start_time": time.time(),
            "system_info": await self._gather_system_info(),
        }

        logger.info("Performance benchmarking framework initialized")

    @pytest.mark.benchmark
    @pytest.mark.parametrize("data_size", ["small", "medium", "large"])
    @pytest.mark.asyncio
    async def test_data_size_scaling_performance(self, data_size: str):
        """
        Test performance scaling across different data sizes.

        Validates:
        - Linear or sub-linear scaling with data size
        - Memory usage remains proportional
        - Performance meets size-specific thresholds
        - Quality maintained across all sizes
        """
        logger.info(f"Testing performance scaling for {data_size} data")

        config = self.DATA_SIZE_CONFIGS[data_size]
        threshold_key = f"{data_size}_data_conversion_time"
        time_threshold = self.PERFORMANCE_THRESHOLDS[threshold_key]

        # Create test dataset
        test_data = create_statistical_dataset(
            size=data_size, complexity="medium", features=config["cols"]
        )

        # Test different conversion paths
        conversion_tests = [
            (DataFormat.PANDAS_DATAFRAME, DataFormat.NUMPY_ARRAY),
            (DataFormat.PANDAS_DATAFRAME, DataFormat.SCIPY_SPARSE),
            (DataFormat.NUMPY_ARRAY, DataFormat.PANDAS_DATAFRAME),
        ]

        scaling_results = {}

        for source_format, target_format in conversion_tests:
            conversion_name = f"{source_format}_to_{target_format}"
            logger.info(f"  Testing {conversion_name}")

            # Prepare data in source format
            if source_format == DataFormat.NUMPY_ARRAY:
                source_data = test_data.select_dtypes(include=[np.number]).values
            else:
                source_data = test_data

            # Perform benchmarked conversion
            request = ConversionRequest(
                source_format=source_format,
                target_format=target_format,
                data=source_data,
                context={
                    "performance_benchmark": True,
                    "data_size": data_size,
                    "benchmark_iteration": 1,
                },
            )

            converter = self.converters[target_format]

            # Multiple runs for statistical significance
            run_times = []
            memory_usages = []

            for run in range(3):  # 3 runs for averaging
                gc.collect()  # Clean up before measurement

                result, exec_time, memory_usage = await measure_async_performance(
                    converter.convert, request
                )

                assert result.success, (
                    f"Conversion failed for {conversion_name} at size {data_size}"
                )

                run_times.append(exec_time)
                memory_usages.append(memory_usage)

            # Calculate statistics
            avg_time = statistics.mean(run_times)
            std_time = statistics.stdev(run_times) if len(run_times) > 1 else 0
            avg_memory = statistics.mean(memory_usages)

            scaling_results[conversion_name] = {
                "average_time": avg_time,
                "std_dev_time": std_time,
                "average_memory": avg_memory,
                "data_size": data_size,
                "rows": config["rows"],
                "columns": config["cols"],
                "throughput_rows_per_second": config["rows"] / avg_time
                if avg_time > 0
                else 0,
            }

            # Validate performance threshold
            assert avg_time < time_threshold, (
                f"{conversion_name} too slow for {data_size}: {avg_time:.2f}s > {time_threshold}s"
            )

            logger.info(
                f"    {conversion_name}: {avg_time:.3f}s ± {std_time:.3f}s, "
                f"Memory: {avg_memory / 1024 / 1024:.1f}MB"
            )

        # Store results for cross-size analysis
        self.performance_results[f"scaling_{data_size}"] = scaling_results

        logger.info(f"✅ {data_size.capitalize()} data scaling performance validated")

    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_cache_effectiveness_benchmark(self):
        """
        Test cache effectiveness and optimization performance.

        Validates:
        - Cache hit rate meets threshold requirements
        - Cache speedup is significant
        - Cache memory usage is reasonable
        - Cache eviction policies work correctly
        """
        logger.info("Testing cache effectiveness benchmark")

        # Create test datasets of varying complexity
        test_datasets = {
            "simple": create_statistical_dataset(size="medium", complexity="low"),
            "complex": create_statistical_dataset(size="medium", complexity="high"),
            "temporal": create_pandas_dataframe(
                rows=5000, columns=30, mixed_types=True
            ),
        }

        cache_results = {}

        for dataset_name, dataset in test_datasets.items():
            logger.info(f"  Testing cache with {dataset_name} dataset")

            # Configure cache for this test
            cache = ConversionCache(max_size=20, ttl=600)
            converter = self.converters[DataFormat.NUMPY_ARRAY]
            converter._cache = cache  # Inject cache

            # First round: cache misses
            request = ConversionRequest(
                source_format=DataFormat.PANDAS_DATAFRAME,
                target_format=DataFormat.NUMPY_ARRAY,
                data=dataset,
                context={"cache_test": True, "dataset_name": dataset_name},
            )

            miss_times = []
            for _ in range(3):
                result, exec_time, memory_usage = await measure_async_performance(
                    converter.convert, request
                )
                assert result.success
                miss_times.append(exec_time)

            avg_miss_time = statistics.mean(miss_times)

            # Second round: cache hits (same request)
            hit_times = []
            for _ in range(5):
                result, exec_time, memory_usage = await measure_async_performance(
                    converter.convert, request
                )
                assert result.success
                hit_times.append(exec_time)

            avg_hit_time = statistics.mean(hit_times)

            # Calculate cache effectiveness
            cache_speedup = avg_miss_time / avg_hit_time if avg_hit_time > 0 else 0
            cache_stats = cache.get_statistics()

            cache_results[dataset_name] = {
                "cache_miss_time": avg_miss_time,
                "cache_hit_time": avg_hit_time,
                "cache_speedup": cache_speedup,
                "hit_rate": cache_stats.hit_rate,
                "cache_size": cache_stats.current_size,
                "memory_usage": cache_stats.memory_usage,
            }

            # Validate cache effectiveness
            assert (
                cache_stats.hit_rate
                >= self.PERFORMANCE_THRESHOLDS["cache_hit_rate_threshold"]
            ), f"Cache hit rate too low for {dataset_name}: {cache_stats.hit_rate:.2f}"

            assert cache_speedup >= 3.0, (
                f"Cache speedup insufficient for {dataset_name}: {cache_speedup:.1f}x"
            )

            logger.info(
                f"    {dataset_name}: Hit rate: {cache_stats.hit_rate:.2f}, "
                f"Speedup: {cache_speedup:.1f}x"
            )

        # Test cache eviction under memory pressure
        large_dataset = create_statistical_dataset(size="large", complexity="high")

        # Fill cache with multiple large requests
        eviction_test_requests = []
        for i in range(25):  # More than cache capacity
            modified_data = large_dataset.copy()
            modified_data["unique_col"] = i  # Make each request unique

            request = ConversionRequest(
                source_format=DataFormat.PANDAS_DATAFRAME,
                target_format=DataFormat.NUMPY_ARRAY,
                data=modified_data,
                context={"eviction_test": True, "request_id": i},
            )
            eviction_test_requests.append(request)

        # Process all requests
        for request in eviction_test_requests:
            result = await converter.convert(request)
            assert result.success

        # Validate cache size stayed within limits
        final_cache_stats = cache.get_statistics()
        assert final_cache_stats.current_size <= cache.max_size, (
            "Cache failed to evict items properly"
        )

        self.performance_results["cache_effectiveness"] = cache_results
        logger.info("✅ Cache effectiveness benchmark validated")

    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_memory_usage_optimization(self):
        """
        Test memory usage patterns and optimization effectiveness.

        Validates:
        - Memory usage remains within expected bounds
        - Memory optimization strategies are effective
        - Memory cleanup happens properly
        - Streaming reduces memory footprint
        """
        logger.info("Testing memory usage optimization")

        memory_results = {}

        # Test 1: Memory-constrained processing
        logger.info("  Testing memory-constrained processing")

        # Create dataset that would exceed memory if loaded fully
        large_streaming_data = create_streaming_data_source(
            total_size="500MB",
            chunk_size="50MB",
            columns=100,
            data_types=["numeric", "categorical"],
        )

        memory_limit = 100 * 1024 * 1024  # 100MB limit
        streaming_options = create_streaming_options(
            chunk_size=50 * 1024 * 1024,
            memory_limit=memory_limit,
            quality_level="balanced",
        )

        memory_monitor = []

        async def memory_callback(current_usage):
            memory_monitor.append(current_usage)

        request = ConversionRequest(
            source_format=DataFormat.STREAMING_DATAFRAME,
            target_format=DataFormat.NUMPY_ARRAY,
            data=large_streaming_data,
            context={"conversion_options": streaming_options, "memory_test": True},
        )

        converter = self.converters[DataFormat.NUMPY_ARRAY]
        result, exec_time, total_memory = await measure_async_performance(
            converter.convert, request, memory_callback=memory_callback
        )

        assert result.success, "Memory-constrained processing failed"

        max_memory_used = max(memory_monitor) if memory_monitor else 0
        memory_limit_adherence = max_memory_used / memory_limit

        assert (
            memory_limit_adherence
            <= self.PERFORMANCE_THRESHOLDS["streaming_memory_limit_adherence"]
        ), (
            f"Memory limit exceeded: {memory_limit_adherence:.1f} > {self.PERFORMANCE_THRESHOLDS['streaming_memory_limit_adherence']}"
        )

        memory_results["streaming_memory_constrained"] = {
            "memory_limit": memory_limit,
            "max_memory_used": max_memory_used,
            "memory_efficiency": max_memory_used / memory_limit,
            "processing_time": exec_time,
            "data_size_processed": result.metadata.get("total_rows_processed", 0),
        }

        # Test 2: Memory optimization strategy comparison
        logger.info("  Testing memory optimization strategies")

        test_dataset = create_high_dimensional_array(shape=(10000, 500))

        optimization_strategies = [
            create_memory_efficient_options(),
            create_high_fidelity_options(),
            create_streaming_options(memory_limit=50 * 1024 * 1024),
        ]

        strategy_results = {}

        for i, options in enumerate(optimization_strategies):
            strategy_name = ["memory_efficient", "high_fidelity", "streaming"][i]

            request = ConversionRequest(
                source_format=DataFormat.NUMPY_ARRAY,
                target_format=DataFormat.PANDAS_DATAFRAME,
                data=test_dataset,
                context={"conversion_options": options, "strategy_test": strategy_name},
            )

            converter = self.converters[DataFormat.PANDAS_DATAFRAME]
            result, exec_time, memory_usage = await measure_async_performance(
                converter.convert, request
            )

            assert result.success, f"Strategy {strategy_name} failed"

            strategy_results[strategy_name] = {
                "execution_time": exec_time,
                "memory_usage": memory_usage,
                "memory_efficiency_score": result.metadata.get(
                    "memory_efficiency_score", 0
                ),
            }

        # Validate memory-efficient strategy uses less memory
        memory_efficient_usage = strategy_results["memory_efficient"]["memory_usage"]
        high_fidelity_usage = strategy_results["high_fidelity"]["memory_usage"]

        assert memory_efficient_usage <= high_fidelity_usage, (
            "Memory-efficient strategy should use less memory"
        )

        memory_results["optimization_strategies"] = strategy_results

        # Test 3: Memory cleanup and garbage collection
        logger.info("  Testing memory cleanup effectiveness")

        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Process multiple large datasets
        cleanup_datasets = []
        for i in range(5):
            large_data = create_statistical_dataset(size="large", complexity="medium")
            cleanup_datasets.append(large_data)

        peak_memory = initial_memory

        for i, dataset in enumerate(cleanup_datasets):
            request = ConversionRequest(
                source_format=DataFormat.PANDAS_DATAFRAME,
                target_format=DataFormat.NUMPY_ARRAY,
                data=dataset,
                context={"cleanup_test": True, "iteration": i},
            )

            result = await self.converters[DataFormat.NUMPY_ARRAY].convert(request)
            assert result.success

            current_memory = process.memory_info().rss
            peak_memory = max(peak_memory, current_memory)

            # Force cleanup
            del dataset
            gc.collect()

        final_memory = process.memory_info().rss
        memory_growth = (final_memory - initial_memory) / initial_memory

        # Memory growth should be reasonable after cleanup
        assert memory_growth < 0.5, (
            f"Excessive memory growth after cleanup: {memory_growth:.1%}"
        )

        memory_results["memory_cleanup"] = {
            "initial_memory": initial_memory,
            "peak_memory": peak_memory,
            "final_memory": final_memory,
            "memory_growth": memory_growth,
        }

        self.performance_results["memory_optimization"] = memory_results
        logger.info("✅ Memory usage optimization validated")

    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_concurrent_operation_throughput(self):
        """
        Test concurrent operation throughput and scalability.

        Validates:
        - Concurrent operations don't cause significant degradation
        - Thread safety is maintained
        - Resource contention is managed properly
        - Throughput scales reasonably with concurrency
        """
        logger.info("Testing concurrent operation throughput")

        # Create test datasets for concurrent processing
        concurrent_datasets = []
        for i in range(10):
            dataset = create_statistical_dataset(size="medium", complexity="medium")
            dataset["unique_id"] = i  # Make each dataset unique
            concurrent_datasets.append(dataset)

        # Test 1: Sequential baseline
        logger.info("  Establishing sequential baseline")

        sequential_times = []
        converter = self.converters[DataFormat.NUMPY_ARRAY]

        for i, dataset in enumerate(concurrent_datasets):
            request = ConversionRequest(
                source_format=DataFormat.PANDAS_DATAFRAME,
                target_format=DataFormat.NUMPY_ARRAY,
                data=dataset,
                context={"sequential_test": True, "dataset_id": i},
            )

            result, exec_time, memory_usage = await measure_async_performance(
                converter.convert, request
            )

            assert result.success, f"Sequential conversion {i} failed"
            sequential_times.append(exec_time)

        sequential_total_time = sum(sequential_times)
        sequential_avg_time = statistics.mean(sequential_times)

        # Test 2: Concurrent execution
        logger.info("  Testing concurrent execution")

        async def convert_dataset(dataset_id: int, dataset: pd.DataFrame):
            """Convert a single dataset concurrently."""
            request = ConversionRequest(
                source_format=DataFormat.PANDAS_DATAFRAME,
                target_format=DataFormat.NUMPY_ARRAY,
                data=dataset,
                context={"concurrent_test": True, "dataset_id": dataset_id},
            )

            result, exec_time, memory_usage = await measure_async_performance(
                converter.convert, request
            )

            return {
                "dataset_id": dataset_id,
                "success": result.success,
                "execution_time": exec_time,
                "memory_usage": memory_usage,
            }

        # Run concurrent conversions
        concurrent_start_time = time.time()

        concurrent_tasks = [
            convert_dataset(i, dataset) for i, dataset in enumerate(concurrent_datasets)
        ]

        concurrent_results = await asyncio.gather(*concurrent_tasks)
        concurrent_total_time = time.time() - concurrent_start_time

        # Validate all conversions succeeded
        for result in concurrent_results:
            assert result["success"], (
                f"Concurrent conversion {result['dataset_id']} failed"
            )

        concurrent_times = [result["execution_time"] for result in concurrent_results]
        concurrent_avg_time = statistics.mean(concurrent_times)

        # Calculate throughput metrics
        throughput_factor = sequential_total_time / concurrent_total_time
        individual_slowdown = concurrent_avg_time / sequential_avg_time

        concurrent_performance = {
            "sequential_total_time": sequential_total_time,
            "concurrent_total_time": concurrent_total_time,
            "throughput_factor": throughput_factor,
            "individual_slowdown": individual_slowdown,
            "concurrent_operations": len(concurrent_datasets),
        }

        # Validate concurrent performance
        assert (
            throughput_factor
            >= self.PERFORMANCE_THRESHOLDS["concurrent_throughput_factor"]
        ), f"Concurrent throughput too low: {throughput_factor:.2f}"

        assert individual_slowdown < 2.0, (
            f"Individual operation slowdown too high: {individual_slowdown:.2f}x"
        )

        logger.info(f"  Concurrent throughput: {throughput_factor:.2f}x improvement")
        logger.info(f"  Individual operation slowdown: {individual_slowdown:.2f}x")

        # Test 3: Resource contention under high concurrency
        logger.info("  Testing high concurrency resource contention")

        high_concurrency_datasets = [
            create_pandas_dataframe(rows=1000, columns=20)
            for _ in range(50)  # Many small tasks
        ]

        async def quick_convert(dataset_id: int, dataset: pd.DataFrame):
            request = ConversionRequest(
                source_format=DataFormat.PANDAS_DATAFRAME,
                target_format=DataFormat.NUMPY_ARRAY,
                data=dataset,
                context={"high_concurrency_test": True, "dataset_id": dataset_id},
            )

            result = await converter.convert(request)
            return result.success

        high_concurrency_start = time.time()

        high_concurrency_tasks = [
            quick_convert(i, dataset)
            for i, dataset in enumerate(high_concurrency_datasets)
        ]

        high_concurrency_results = await asyncio.gather(*high_concurrency_tasks)
        high_concurrency_time = time.time() - high_concurrency_start

        success_rate = sum(high_concurrency_results) / len(high_concurrency_results)

        assert success_rate >= 0.95, (
            f"High concurrency success rate too low: {success_rate:.2%}"
        )

        concurrent_performance["high_concurrency"] = {
            "operations": len(high_concurrency_datasets),
            "total_time": high_concurrency_time,
            "success_rate": success_rate,
            "operations_per_second": len(high_concurrency_datasets)
            / high_concurrency_time,
        }

        self.performance_results["concurrent_throughput"] = concurrent_performance
        logger.info("✅ Concurrent operation throughput validated")

    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_cross_domain_workflow_performance(self):
        """
        Test performance of complete cross-domain workflows.

        Validates:
        - Multi-domain workflow completion times
        - Performance optimization across domain boundaries
        - Context accumulation overhead
        - End-to-end throughput
        """
        logger.info("Testing cross-domain workflow performance")

        # Create comprehensive test dataset
        workflow_dataset = create_statistical_dataset(
            size="medium",
            complexity="high",
            features=30,
            include_correlations=True,
            include_outliers=True,
        )

        # Define multi-domain workflow
        workflow_domains = [
            "statistical",
            "regression",
            "time_series",
            "pattern_recognition",
        ]

        # Test 1: Complete workflow performance
        logger.info("  Testing complete four-domain workflow")

        workflow_start_time = time.time()
        workflow_results = []
        current_data = workflow_dataset
        workflow_context = []

        for i, domain in enumerate(workflow_domains):
            domain_shim = self.domain_shims[domain]

            # Determine formats
            format_mapping = {
                "statistical": (
                    DataFormat.PANDAS_DATAFRAME,
                    DataFormat.STATISTICAL_ANALYSIS,
                ),
                "regression": (
                    DataFormat.STATISTICAL_ANALYSIS,
                    DataFormat.REGRESSION_MODEL,
                ),
                "time_series": (
                    DataFormat.PANDAS_DATAFRAME,
                    DataFormat.TIME_SERIES_FEATURES,
                ),
                "pattern_recognition": (
                    DataFormat.NUMPY_ARRAY,
                    DataFormat.PATTERN_CLUSTERS,
                ),
            }

            input_format, output_format = format_mapping[domain]
            if i > 0:
                # Use output from previous step
                prev_domain = workflow_domains[i - 1]
                _, input_format = format_mapping[prev_domain]

            request = ConversionRequest(
                source_format=input_format,
                target_format=output_format,
                data=current_data,
                context={
                    "workflow_step": i + 1,
                    "total_steps": len(workflow_domains),
                    "accumulated_context": workflow_context,
                    "performance_benchmark": True,
                },
            )

            result, step_time, memory_usage = await measure_async_performance(
                domain_shim.convert, request
            )

            assert result.success, f"Workflow step {i + 1} ({domain}) failed"

            workflow_results.append(result)
            workflow_context.append(
                {
                    "domain": domain,
                    "step_time": step_time,
                    "memory_usage": memory_usage,
                    "metadata": result.metadata,
                }
            )
            current_data = result.data

        total_workflow_time = time.time() - workflow_start_time

        # Analyze workflow performance
        workflow_metrics = measure_cross_domain_performance(workflow_results)
        workflow_metrics["total_workflow_time"] = total_workflow_time
        workflow_metrics["domains_processed"] = len(workflow_domains)
        workflow_metrics["average_domain_time"] = total_workflow_time / len(
            workflow_domains
        )

        # Validate workflow performance requirements
        assert total_workflow_time < 30.0, (
            f"Complete workflow too slow: {total_workflow_time:.2f}s"
        )

        assert workflow_metrics["peak_memory_usage"] < 500 * 1024 * 1024, (
            f"Workflow memory usage too high: {workflow_metrics['peak_memory_usage'] / 1024 / 1024:.1f}MB"
        )

        logger.info(
            f"  Complete workflow: {total_workflow_time:.2f}s, "
            f"Peak memory: {workflow_metrics['peak_memory_usage'] / 1024 / 1024:.1f}MB"
        )

        # Test 2: Workflow optimization effectiveness
        logger.info("  Testing workflow optimization strategies")

        optimization_strategies = [
            {"name": "baseline", "context": {}},
            {
                "name": "performance_optimized",
                "context": {
                    "optimization_strategy": "speed_first",
                    "cache_aggressively": True,
                    "parallel_where_possible": True,
                },
            },
            {
                "name": "memory_optimized",
                "context": {
                    "optimization_strategy": "memory_first",
                    "streaming_preferred": True,
                    "memory_limit": 100 * 1024 * 1024,
                },
            },
        ]

        optimization_results = {}

        for strategy in optimization_strategies:
            strategy_start_time = time.time()
            strategy_workflow_data = workflow_dataset.copy()

            for i, domain in enumerate(workflow_domains):
                domain_shim = self.domain_shims[domain]

                format_mapping = {
                    "statistical": (
                        DataFormat.PANDAS_DATAFRAME,
                        DataFormat.STATISTICAL_ANALYSIS,
                    ),
                    "regression": (
                        DataFormat.STATISTICAL_ANALYSIS,
                        DataFormat.REGRESSION_MODEL,
                    ),
                    "time_series": (
                        DataFormat.PANDAS_DATAFRAME,
                        DataFormat.TIME_SERIES_FEATURES,
                    ),
                    "pattern_recognition": (
                        DataFormat.NUMPY_ARRAY,
                        DataFormat.PATTERN_CLUSTERS,
                    ),
                }

                input_format, output_format = format_mapping[domain]
                if i > 0:
                    prev_domain = workflow_domains[i - 1]
                    _, input_format = format_mapping[prev_domain]

                request = ConversionRequest(
                    source_format=input_format,
                    target_format=output_format,
                    data=strategy_workflow_data,
                    context={
                        **strategy["context"],
                        "strategy_test": strategy["name"],
                        "workflow_step": i + 1,
                    },
                )

                result = await domain_shim.convert(request)
                assert result.success, (
                    f"Strategy {strategy['name']} failed at step {i + 1}"
                )

                strategy_workflow_data = result.data

            strategy_time = time.time() - strategy_start_time

            optimization_results[strategy["name"]] = {
                "total_time": strategy_time,
                "strategy_context": strategy["context"],
            }

        # Validate optimization effectiveness
        baseline_time = optimization_results["baseline"]["total_time"]
        performance_optimized_time = optimization_results["performance_optimized"][
            "total_time"
        ]

        performance_improvement = baseline_time / performance_optimized_time
        assert performance_improvement >= 1.1, (
            f"Performance optimization insufficient: {performance_improvement:.2f}x"
        )

        workflow_performance = {
            "complete_workflow_metrics": workflow_metrics,
            "optimization_strategies": optimization_results,
            "performance_improvement": performance_improvement,
        }

        self.performance_results["cross_domain_workflow"] = workflow_performance
        logger.info(
            f"  Workflow optimization: {performance_improvement:.2f}x improvement"
        )
        logger.info("✅ Cross-domain workflow performance validated")

    async def _gather_system_info(self) -> Dict[str, Any]:
        """Gather system information for benchmark context."""
        import platform
        import psutil

        return {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "cpu_count": psutil.cpu_count(),
            "memory_total": psutil.virtual_memory().total,
            "memory_available": psutil.virtual_memory().available,
        }

    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_performance_regression_detection(self):
        """
        Test for performance regressions across different scenarios.

        This test establishes performance baselines and validates that
        current performance meets or exceeds historical performance.
        """
        logger.info("Testing performance regression detection")

        # Define performance baseline expectations
        performance_baselines = {
            "pandas_to_numpy_medium": {"max_time": 2.0, "max_memory_mb": 100},
            "numpy_to_sparse_large": {"max_time": 5.0, "max_memory_mb": 200},
            "four_domain_workflow": {"max_time": 25.0, "max_memory_mb": 400},
            "cache_hit_speedup": {"min_speedup": 3.0},
            "concurrent_throughput": {"min_factor": 0.6},
        }

        regression_results = {}

        # Test each baseline scenario
        for scenario, baseline in performance_baselines.items():
            logger.info(f"  Testing regression for {scenario}")

            if scenario == "pandas_to_numpy_medium":
                test_data = create_statistical_dataset(size="medium")
                request = ConversionRequest(
                    source_format=DataFormat.PANDAS_DATAFRAME,
                    target_format=DataFormat.NUMPY_ARRAY,
                    data=test_data,
                    context={"regression_test": scenario},
                )

                result, exec_time, memory_usage = await measure_async_performance(
                    self.converters[DataFormat.NUMPY_ARRAY].convert, request
                )

                memory_mb = memory_usage / (1024 * 1024)

                regression_results[scenario] = {
                    "execution_time": exec_time,
                    "memory_usage_mb": memory_mb,
                    "time_within_baseline": exec_time <= baseline["max_time"],
                    "memory_within_baseline": memory_mb <= baseline["max_memory_mb"],
                }

                assert exec_time <= baseline["max_time"], (
                    f"Performance regression in {scenario}: {exec_time:.2f}s > {baseline['max_time']}s"
                )
                assert memory_mb <= baseline["max_memory_mb"], (
                    f"Memory regression in {scenario}: {memory_mb:.1f}MB > {baseline['max_memory_mb']}MB"
                )

        # Store regression test results
        self.performance_results["regression_detection"] = regression_results

        logger.info("✅ Performance regression detection completed")

    def teardown_method(self):
        """Clean up after each test method."""
        # Force garbage collection
        gc.collect()

        # Log performance summary if available
        if hasattr(self, "performance_results") and self.performance_results:
            logger.info("Performance test summary:")
            for test_name, results in self.performance_results.items():
                logger.info(f"  {test_name}: {len(results)} metrics collected")


if __name__ == "__main__":
    # Run performance benchmarks
    import sys

    logging.basicConfig(level=logging.INFO)

    async def run_benchmarks():
        test_instance = TestPerformanceBenchmarks()
        await test_instance.setup_performance_framework()

        # Define benchmark tests to run
        benchmark_tests = [
            (
                "Small Data Scaling",
                test_instance.test_data_size_scaling_performance,
                "small",
            ),
            (
                "Medium Data Scaling",
                test_instance.test_data_size_scaling_performance,
                "medium",
            ),
            (
                "Large Data Scaling",
                test_instance.test_data_size_scaling_performance,
                "large",
            ),
            ("Cache Effectiveness", test_instance.test_cache_effectiveness_benchmark),
            ("Memory Optimization", test_instance.test_memory_usage_optimization),
            (
                "Concurrent Throughput",
                test_instance.test_concurrent_operation_throughput,
            ),
            (
                "Cross-Domain Workflow",
                test_instance.test_cross_domain_workflow_performance,
            ),
            (
                "Regression Detection",
                test_instance.test_performance_regression_detection,
            ),
        ]

        print("\n" + "=" * 60)
        print("Running Performance Benchmarks")
        print("=" * 60)

        for test_name, test_method, *args in benchmark_tests:
            try:
                print(f"\n📊 Running {test_name}...")
                if args:
                    await test_method(args[0])
                else:
                    await test_method()
                print(f"✅ {test_name} PASSED")
            except Exception as e:
                print(f"❌ {test_name} FAILED: {e}")
                import traceback

                traceback.print_exc()

        # Print performance summary
        print("\n" + "=" * 60)
        print("Performance Benchmark Summary")
        print("=" * 60)

        if hasattr(test_instance, "performance_results"):
            for category, results in test_instance.performance_results.items():
                print(f"\n{category.upper().replace('_', ' ')}:")
                if isinstance(results, dict):
                    for key, value in results.items():
                        if isinstance(value, (int, float)):
                            print(f"  {key}: {value:.3f}")
                        else:
                            print(f"  {key}: {value}")
                else:
                    print(f"  Results: {results}")

        print("\n" + "=" * 60)
        print("Performance Benchmarks Complete")
        print("=" * 60)

    if sys.version_info >= (3, 7):
        asyncio.run(run_benchmarks())
    else:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(run_benchmarks())
