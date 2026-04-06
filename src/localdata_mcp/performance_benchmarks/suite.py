"""Main benchmark orchestration suite.

This module contains the PerformanceBenchmarkSuite class that coordinates
all benchmark categories (token, streaming, memory, regression) and the
top-level run_performance_benchmark entry point.
"""

import logging
import shutil
import tempfile
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from sqlalchemy import create_engine

from ..config_manager import PerformanceConfig
from ..logging_manager import get_logger, get_logging_manager
from ..streaming import StreamingQueryExecutor, create_streaming_source
from .benchmarks import StreamingBenchmark, TokenBenchmark
from .comparison import generate_comparisons, run_regression_tests
from .generators import DatasetGenerator
from .models import BenchmarkConfig, BenchmarkResult, ComparisonResult
from .monitoring import PerformanceMonitor
from .reporting import (
    calculate_success_rate,
    generate_reports,
    generate_summary,
    save_detailed_results,
)

logger = get_logger(__name__)


class PerformanceBenchmarkSuite:
    """Main benchmarking suite orchestrator."""

    def __init__(self, config: Optional[BenchmarkConfig] = None):
        """Initialize benchmark suite.

        Args:
            config: Benchmark configuration. Uses defaults if None.
        """
        self.config = config or BenchmarkConfig()
        self.results: List[BenchmarkResult] = []
        self.comparisons: List[ComparisonResult] = []

        # Initialize benchmarking components
        self.streaming_benchmark = StreamingBenchmark()
        self.token_benchmark = TokenBenchmark()

        # Setup results directory
        self.config.results_dir.mkdir(exist_ok=True)

        # Initialize logging
        self.logging_manager = get_logging_manager()

        with self.logging_manager.context(
            operation="benchmark_suite_init", component="performance_benchmarks"
        ):
            logger.info(
                "Performance benchmark suite initialized", config=asdict(self.config)
            )

    def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """Run comprehensive performance benchmarking suite.

        Returns:
            Summary of all benchmark results
        """
        start_time = time.time()

        with self.logging_manager.context(
            operation="comprehensive_benchmark", component="performance_benchmarks"
        ):
            logger.info("Starting comprehensive performance benchmark")

        try:
            self._run_token_benchmarks()
            self._run_streaming_benchmarks()
            self._run_memory_benchmarks()
            run_regression_tests(self.config, self.results, self.comparisons)
            generate_comparisons(self.results, self.comparisons)

            if self.config.save_detailed_metrics:
                save_detailed_results(self.config, self.results, self.comparisons)

            if self.config.generate_reports:
                generate_reports(self.config, self.results, self.comparisons)

            execution_time = time.time() - start_time

            logger.info(
                "Comprehensive benchmark completed",
                total_tests=len(self.results),
                execution_time=execution_time,
                success_rate=calculate_success_rate(self.results),
            )

            return generate_summary(self.results, self.comparisons, self.config)

        except Exception as e:
            logger.error(f"Comprehensive benchmark failed: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    def _run_token_benchmarks(self):
        """Run token counting and estimation benchmarks."""
        logger.info("Running token performance benchmarks")

        test_scenarios = [
            (self.config.small_dataset_rows, False, False, "small_numeric"),
            (self.config.small_dataset_rows, True, False, "small_text_heavy"),
            (self.config.medium_dataset_rows, False, False, "medium_numeric"),
            (self.config.medium_dataset_rows, True, False, "medium_text_heavy"),
            (self.config.medium_dataset_rows, False, True, "medium_wide_table"),
        ]

        for rows, text_heavy, wide_table, scenario_name in test_scenarios:
            logger.debug(f"Running token benchmark: {scenario_name}")

            df = DatasetGenerator.create_test_dataframe(rows, text_heavy, wide_table)

            result = self.token_benchmark.benchmark_token_estimation(
                df, iterations=self.config.test_iterations
            )
            result.test_name = f"token_{scenario_name}"
            result.metadata["scenario"] = scenario_name
            result.metadata["text_heavy"] = text_heavy
            result.metadata["wide_table"] = wide_table

            self.results.append(result)
            logger.debug(
                f"Token benchmark completed: {scenario_name}, "
                f"Rate: {result.processing_rate_rows_per_second:.0f} rows/sec"
            )

    def _run_streaming_benchmarks(self):
        """Run streaming vs batch processing benchmarks."""
        logger.info("Running streaming vs batch benchmarks")

        temp_dir = Path(tempfile.mkdtemp())

        try:
            test_scenarios = [
                (self.config.small_dataset_rows, "small", False, False),
                (self.config.medium_dataset_rows, "medium", False, False),
                (self.config.medium_dataset_rows, "medium", True, False),
                (self.config.large_dataset_rows, "large", False, False),
            ]

            for rows, size_category, text_heavy, wide_table in test_scenarios:
                db_path = temp_dir / f"test_{size_category}_{rows}.db"
                scenario_name = f"{size_category}_{'text' if text_heavy else 'numeric'}"

                logger.debug(f"Creating test database: {scenario_name}")
                connection_string = DatasetGenerator.create_test_database(
                    str(db_path), rows, text_heavy=text_heavy, wide_table=wide_table
                )

                query = "SELECT * FROM test_data"

                streaming_result, batch_result = (
                    self.streaming_benchmark.benchmark_streaming_vs_batch(
                        connection_string,
                        query,
                        size_category,
                        self.config.test_iterations,
                    )
                )

                streaming_result.test_name = f"streaming_{scenario_name}"
                batch_result.test_name = f"batch_{scenario_name}"

                streaming_result.metadata["scenario"] = scenario_name
                batch_result.metadata["scenario"] = scenario_name

                self.results.extend([streaming_result, batch_result])

                logger.debug(
                    f"Streaming benchmark: {scenario_name}, "
                    f"Streaming: {streaming_result.processing_rate_rows_per_second:.0f} rows/sec, "
                    f"Batch: {batch_result.processing_rate_rows_per_second:.0f} rows/sec"
                )

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def _run_memory_benchmarks(self):
        """Run memory usage pattern benchmarks."""
        logger.info("Running memory usage benchmarks")

        temp_dir = Path(tempfile.mkdtemp())

        try:
            db_path = temp_dir / "memory_test.db"
            connection_string = DatasetGenerator.create_test_database(
                str(db_path), self.config.medium_dataset_rows
            )

            query = "SELECT * FROM test_data"

            for chunk_size in self.config.chunk_sizes:
                logger.debug(f"Testing chunk size: {chunk_size}")

                config = PerformanceConfig()
                config.chunk_size = chunk_size
                executor = StreamingQueryExecutor(config)

                monitor = PerformanceMonitor()
                monitor.start_monitoring()

                start_time = time.time()

                try:
                    engine = create_engine(connection_string)
                    source = create_streaming_source(engine=engine, query=query)

                    result_df, metadata = executor.execute_streaming(
                        source=source,
                        query_id=f"memory_test_{chunk_size}",
                        initial_chunk_size=chunk_size,
                    )

                    execution_time = time.time() - start_time

                except Exception as e:
                    logger.warning(
                        f"Memory benchmark failed for chunk size {chunk_size}: {e}"
                    )
                    continue
                finally:
                    performance_data = monitor.stop_monitoring()

                result = BenchmarkResult(
                    test_name=f"memory_chunk_{chunk_size}",
                    test_category="memory_efficiency",
                    dataset_size="medium",
                    database_type="sqlite",
                    processing_mode="streaming",
                    execution_time_seconds=execution_time,
                    peak_memory_mb=performance_data["peak_memory_mb"],
                    average_memory_mb=performance_data["average_memory_mb"],
                    memory_efficiency=metadata.get("memory_status", {}).get(
                        "final_available_gb", 0
                    )
                    / 8.0,  # Assume 8GB system
                    rows_processed=metadata.get("total_rows_processed", 0),
                    processing_rate_rows_per_second=metadata.get(
                        "total_rows_processed", 0
                    )
                    / execution_time,
                    chunk_count=metadata.get("chunks_processed", 0),
                    average_chunk_size=chunk_size,
                    memory_samples=performance_data["memory_samples"],
                    memory_timestamps=performance_data["timestamps"],
                    metadata={"chunk_size": chunk_size},
                )

                self.results.append(result)

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


def run_performance_benchmark(
    config: Optional[BenchmarkConfig] = None,
) -> Dict[str, Any]:
    """Main entry point for running performance benchmarks.

    Args:
        config: Benchmark configuration. Uses defaults if None.

    Returns:
        Summary of benchmark results
    """
    suite = PerformanceBenchmarkSuite(config)
    return suite.run_comprehensive_benchmark()


if __name__ == "__main__":
    # Run benchmarks when script is executed directly
    logging.basicConfig(level=logging.INFO)

    config = BenchmarkConfig()
    config.test_iterations = 2  # Reduce for quick testing

    results = run_performance_benchmark(config)

    print("\n" + "=" * 60)
    print("PERFORMANCE BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"Total Tests: {results['total_tests']}")
    print(f"Success Rate: {results['success_rate']:.1%}")
    print(
        f"Average Processing Rate: {results.get('average_processing_rate', 0):.0f} rows/sec"
    )
    print(f"Average Memory Usage: {results.get('average_memory_usage_mb', 0):.1f} MB")
    print(f"Performance Improvements: {results.get('performance_improvements', 0)}")
    print(f"Performance Regressions: {results.get('performance_regressions', 0)}")
    print(f"Results Directory: {results['results_directory']}")
    print("=" * 60)
