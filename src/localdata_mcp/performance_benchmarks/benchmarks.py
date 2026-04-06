"""Token and streaming benchmark implementations.

This module contains the TokenBenchmark and StreamingBenchmark classes
that measure token estimation performance and streaming vs batch
processing throughput respectively.
"""

import gc
import time
from typing import Tuple

import pandas as pd
from sqlalchemy import create_engine

from ..logging_manager import get_logger
from ..streaming import StreamingQueryExecutor, create_streaming_source
from ..token_manager import get_token_manager
from .models import BenchmarkResult
from .monitoring import PerformanceMonitor

logger = get_logger(__name__)


class TokenBenchmark:
    """Benchmark token counting and estimation performance."""

    def __init__(self):
        self.token_manager = get_token_manager()

    def benchmark_token_estimation(
        self, df: pd.DataFrame, iterations: int = 5
    ) -> BenchmarkResult:
        """Benchmark token estimation performance on a DataFrame.

        Args:
            df: DataFrame to analyze
            iterations: Number of iterations for averaging

        Returns:
            Benchmark results for token estimation
        """
        monitor = PerformanceMonitor()

        execution_times = []
        token_counts = []

        for i in range(iterations):
            monitor.start_monitoring()
            start_time = time.time()

            try:
                # Estimate tokens for the DataFrame result
                estimation = self.token_manager.estimate_tokens_for_query_result(
                    total_rows=len(df), sample_df=df.head(min(100, len(df)))
                )

                execution_time = time.time() - start_time
                execution_times.append(execution_time)
                token_counts.append(estimation.total_tokens)

            except Exception as e:
                logger.error(f"Token estimation failed: {e}")
                return BenchmarkResult(
                    test_name="token_estimation",
                    test_category="token_performance",
                    dataset_size=self._classify_dataset_size(len(df)),
                    database_type="memory",
                    processing_mode="estimation",
                    execution_time_seconds=0.0,
                    peak_memory_mb=0.0,
                    average_memory_mb=0.0,
                    memory_efficiency=0.0,
                    rows_processed=len(df),
                    processing_rate_rows_per_second=0.0,
                    chunk_count=1,
                    average_chunk_size=len(df),
                    success=False,
                    error_message=str(e),
                )
            finally:
                performance_data = monitor.stop_monitoring()

        # Calculate average metrics
        avg_execution_time = sum(execution_times) / len(execution_times)
        avg_token_count = sum(token_counts) / len(token_counts)
        processing_rate = len(df) / avg_execution_time if avg_execution_time > 0 else 0

        return BenchmarkResult(
            test_name="token_estimation",
            test_category="token_performance",
            dataset_size=self._classify_dataset_size(len(df)),
            database_type="memory",
            processing_mode="estimation",
            execution_time_seconds=avg_execution_time,
            peak_memory_mb=performance_data["peak_memory_mb"],
            average_memory_mb=performance_data["average_memory_mb"],
            memory_efficiency=1.0,  # Token estimation is memory-efficient
            rows_processed=len(df),
            processing_rate_rows_per_second=processing_rate,
            chunk_count=1,
            average_chunk_size=len(df),
            token_count=int(avg_token_count),
            token_estimation_time_seconds=avg_execution_time,
            memory_samples=performance_data["memory_samples"],
            memory_timestamps=performance_data["timestamps"],
        )

    def _classify_dataset_size(self, rows: int) -> str:
        """Classify dataset size category."""
        if rows < 10000:
            return "small"
        elif rows < 100000:
            return "medium"
        else:
            return "large"


class StreamingBenchmark:
    """Benchmark streaming vs batch processing performance."""

    def __init__(self):
        self.streaming_executor = StreamingQueryExecutor()
        self.token_benchmark = TokenBenchmark()

    def benchmark_streaming_vs_batch(
        self, connection_string: str, query: str, dataset_size: str, iterations: int = 3
    ) -> Tuple[BenchmarkResult, BenchmarkResult]:
        """Compare streaming vs batch processing performance.

        Args:
            connection_string: Database connection string
            query: SQL query to execute
            dataset_size: Size category of dataset
            iterations: Number of test iterations

        Returns:
            Tuple of (streaming_result, batch_result)
        """
        streaming_result = self._benchmark_streaming(
            connection_string, query, dataset_size, iterations
        )
        batch_result = self._benchmark_batch(
            connection_string, query, dataset_size, iterations
        )
        return streaming_result, batch_result

    def _benchmark_streaming(
        self, connection_string: str, query: str, dataset_size: str, iterations: int
    ) -> BenchmarkResult:
        """Benchmark streaming query execution."""
        monitor = PerformanceMonitor()

        execution_times = []
        memory_efficiencies = []
        processing_rates = []
        chunk_counts = []

        engine = create_engine(connection_string)

        for i in range(iterations):
            gc.collect()  # Clean memory before test

            monitor.start_monitoring()
            start_time = time.time()

            try:
                source = create_streaming_source(engine=engine, query=query)
                result_df, metadata = self.streaming_executor.execute_streaming(
                    source=source,
                    query_id=f"benchmark_streaming_{i}",
                    database_name="benchmark_db",
                )

                execution_time = time.time() - start_time
                execution_times.append(execution_time)

                rows_processed = metadata.get("total_rows_processed", len(result_df))
                processing_rate = (
                    rows_processed / execution_time if execution_time > 0 else 0
                )
                processing_rates.append(processing_rate)

                chunk_count = metadata.get("chunks_processed", 1)
                chunk_counts.append(chunk_count)

                memory_status = metadata.get("memory_status", {})
                initial_memory = memory_status.get("initial_available_gb", 0)
                final_memory = memory_status.get("final_available_gb", 0)
                memory_efficiency = (
                    final_memory / initial_memory if initial_memory > 0 else 1.0
                )
                memory_efficiencies.append(memory_efficiency)

            except Exception as e:
                logger.error(f"Streaming benchmark failed: {e}")
                return BenchmarkResult(
                    test_name="streaming_query",
                    test_category="execution_performance",
                    dataset_size=dataset_size,
                    database_type=self._extract_db_type(connection_string),
                    processing_mode="streaming",
                    execution_time_seconds=0.0,
                    peak_memory_mb=0.0,
                    average_memory_mb=0.0,
                    memory_efficiency=0.0,
                    rows_processed=0,
                    processing_rate_rows_per_second=0.0,
                    chunk_count=0,
                    average_chunk_size=0,
                    success=False,
                    error_message=str(e),
                )
            finally:
                performance_data = monitor.stop_monitoring()
                self.streaming_executor.manage_memory_bounds()

        avg_execution_time = sum(execution_times) / len(execution_times)
        avg_memory_efficiency = sum(memory_efficiencies) / len(memory_efficiencies)
        avg_processing_rate = sum(processing_rates) / len(processing_rates)
        avg_chunk_count = sum(chunk_counts) / len(chunk_counts)

        return BenchmarkResult(
            test_name="streaming_query",
            test_category="execution_performance",
            dataset_size=dataset_size,
            database_type=self._extract_db_type(connection_string),
            processing_mode="streaming",
            execution_time_seconds=avg_execution_time,
            peak_memory_mb=performance_data["peak_memory_mb"],
            average_memory_mb=performance_data["average_memory_mb"],
            memory_efficiency=avg_memory_efficiency,
            rows_processed=int(avg_processing_rate * avg_execution_time),
            processing_rate_rows_per_second=avg_processing_rate,
            chunk_count=int(avg_chunk_count),
            average_chunk_size=(
                int(avg_processing_rate * avg_execution_time / avg_chunk_count)
                if avg_chunk_count > 0
                else 0
            ),
            memory_samples=performance_data["memory_samples"],
            memory_timestamps=performance_data["timestamps"],
        )

    def _benchmark_batch(
        self, connection_string: str, query: str, dataset_size: str, iterations: int
    ) -> BenchmarkResult:
        """Benchmark traditional batch query execution."""
        monitor = PerformanceMonitor()

        execution_times = []
        processing_rates = []

        engine = create_engine(connection_string)

        for i in range(iterations):
            gc.collect()  # Clean memory before test

            monitor.start_monitoring()
            start_time = time.time()

            try:
                result_df = pd.read_sql(query, engine)

                execution_time = time.time() - start_time
                execution_times.append(execution_time)

                rows_processed = len(result_df)
                processing_rate = (
                    rows_processed / execution_time if execution_time > 0 else 0
                )
                processing_rates.append(processing_rate)

            except Exception as e:
                logger.error(f"Batch benchmark failed: {e}")
                return BenchmarkResult(
                    test_name="batch_query",
                    test_category="execution_performance",
                    dataset_size=dataset_size,
                    database_type=self._extract_db_type(connection_string),
                    processing_mode="batch",
                    execution_time_seconds=0.0,
                    peak_memory_mb=0.0,
                    average_memory_mb=0.0,
                    memory_efficiency=0.0,
                    rows_processed=0,
                    processing_rate_rows_per_second=0.0,
                    chunk_count=1,
                    average_chunk_size=0,
                    success=False,
                    error_message=str(e),
                )
            finally:
                performance_data = monitor.stop_monitoring()

        avg_execution_time = sum(execution_times) / len(execution_times)
        avg_processing_rate = sum(processing_rates) / len(processing_rates)

        return BenchmarkResult(
            test_name="batch_query",
            test_category="execution_performance",
            dataset_size=dataset_size,
            database_type=self._extract_db_type(connection_string),
            processing_mode="batch",
            execution_time_seconds=avg_execution_time,
            peak_memory_mb=performance_data["peak_memory_mb"],
            average_memory_mb=performance_data["average_memory_mb"],
            memory_efficiency=0.8,  # Batch processing typically less memory efficient
            rows_processed=int(avg_processing_rate * avg_execution_time),
            processing_rate_rows_per_second=avg_processing_rate,
            chunk_count=1,
            average_chunk_size=int(avg_processing_rate * avg_execution_time),
            memory_samples=performance_data["memory_samples"],
            memory_timestamps=performance_data["timestamps"],
        )

    def _extract_db_type(self, connection_string: str) -> str:
        """Extract database type from connection string."""
        if connection_string.startswith("sqlite"):
            return "sqlite"
        elif connection_string.startswith("postgresql"):
            return "postgresql"
        elif connection_string.startswith("mysql"):
            return "mysql"
        else:
            return "unknown"
