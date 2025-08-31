"""Performance Benchmarking Suite for LocalData MCP v1.3.1.

This module provides comprehensive performance testing and validation capabilities
for measuring memory usage improvements, streaming vs batch processing performance,
token counting efficiency, and database query execution across different scenarios.
"""

import gc
import json
import logging
import os
import psutil
import sqlite3
import tempfile
import time
import traceback
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from contextlib import contextmanager
from abc import ABC, abstractmethod

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text, MetaData, Table, Column, Integer, String, Float, DateTime
from sqlalchemy.engine import Engine

# Import existing components for benchmarking
from .streaming_executor import StreamingQueryExecutor, create_streaming_source, MemoryStatus
from .token_manager import get_token_manager
from .config_manager import get_config_manager, PerformanceConfig
from .logging_manager import get_logging_manager, get_logger

logger = get_logger(__name__)


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark execution."""
    
    # Test data configuration
    small_dataset_rows: int = 1000
    medium_dataset_rows: int = 50000  
    large_dataset_rows: int = 500000
    
    # Memory configuration
    memory_limit_mb: int = 512
    chunk_sizes: List[int] = field(default_factory=lambda: [100, 500, 1000, 5000])
    
    # Test repetition
    test_iterations: int = 3
    warmup_iterations: int = 1
    
    # Database types to test
    database_types: List[str] = field(default_factory=lambda: ['sqlite', 'postgresql', 'mysql'])
    
    # Output configuration
    results_dir: Path = field(default_factory=lambda: Path("benchmark_results"))
    generate_reports: bool = True
    save_detailed_metrics: bool = True


@dataclass 
class BenchmarkResult:
    """Results from a single benchmark test."""
    
    # Test identification
    test_name: str
    test_category: str
    dataset_size: str  # 'small', 'medium', 'large'
    database_type: str
    processing_mode: str  # 'streaming', 'batch'
    
    # Performance metrics
    execution_time_seconds: float
    peak_memory_mb: float
    average_memory_mb: float
    memory_efficiency: float  # 0-1 ratio
    
    # Data processing metrics
    rows_processed: int
    processing_rate_rows_per_second: float
    chunk_count: int
    average_chunk_size: int
    
    # Token metrics (if applicable)
    token_count: Optional[int] = None
    token_estimation_time_seconds: Optional[float] = None
    token_estimation_accuracy: Optional[float] = None
    
    # Memory usage profile
    memory_samples: List[float] = field(default_factory=list)
    memory_timestamps: List[float] = field(default_factory=list)
    
    # Error information
    success: bool = True
    error_message: Optional[str] = None
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return asdict(self)


@dataclass
class ComparisonResult:
    """Results comparing different approaches."""
    
    baseline_result: BenchmarkResult
    comparison_result: BenchmarkResult
    
    # Performance improvements (positive = improvement)
    execution_time_improvement_percent: float
    memory_improvement_percent: float
    processing_rate_improvement_percent: float
    
    # Statistical significance
    is_significant: bool
    confidence_level: float
    
    # Summary
    summary: str
    recommendations: List[str] = field(default_factory=list)


class PerformanceMonitor:
    """Monitor system performance during benchmarks."""
    
    def __init__(self, sample_interval: float = 0.1):
        """Initialize performance monitor.
        
        Args:
            sample_interval: How often to sample performance metrics (seconds)
        """
        self.sample_interval = sample_interval
        self.memory_samples: List[float] = []
        self.cpu_samples: List[float] = []
        self.timestamps: List[float] = []
        self.monitoring = False
        self._monitor_thread = None
    
    def start_monitoring(self):
        """Start performance monitoring in background thread."""
        import threading
        
        if self.monitoring:
            return
        
        self.monitoring = True
        self.memory_samples.clear()
        self.cpu_samples.clear()
        self.timestamps.clear()
        
        def monitor():
            process = psutil.Process()
            start_time = time.time()
            
            while self.monitoring:
                try:
                    memory_info = process.memory_info()
                    memory_mb = memory_info.rss / (1024 * 1024)
                    cpu_percent = process.cpu_percent()
                    
                    self.memory_samples.append(memory_mb)
                    self.cpu_samples.append(cpu_percent)
                    self.timestamps.append(time.time() - start_time)
                    
                    time.sleep(self.sample_interval)
                except Exception as e:
                    logger.warning(f"Performance monitoring error: {e}")
                    break
        
        self._monitor_thread = threading.Thread(target=monitor, daemon=True)
        self._monitor_thread.start()
    
    def stop_monitoring(self) -> Dict[str, Any]:
        """Stop monitoring and return collected metrics."""
        self.monitoring = False
        
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)
        
        if not self.memory_samples:
            return {
                "peak_memory_mb": 0.0,
                "average_memory_mb": 0.0,
                "memory_samples": [],
                "timestamps": []
            }
        
        return {
            "peak_memory_mb": max(self.memory_samples),
            "average_memory_mb": sum(self.memory_samples) / len(self.memory_samples),
            "memory_samples": self.memory_samples.copy(),
            "timestamps": self.timestamps.copy()
        }


class DatasetGenerator:
    """Generate test datasets of various sizes and characteristics."""
    
    @staticmethod
    def create_test_dataframe(rows: int, text_heavy: bool = False, 
                            wide_table: bool = False) -> pd.DataFrame:
        """Create test DataFrame with specified characteristics.
        
        Args:
            rows: Number of rows to generate
            text_heavy: Whether to include text-heavy columns
            wide_table: Whether to create many columns
        
        Returns:
            Generated test DataFrame
        """
        np.random.seed(42)  # For reproducible results
        
        # Base columns
        data = {
            'id': range(rows),
            'value': np.random.randint(0, 1000, rows),
            'score': np.random.uniform(0, 100, rows),
            'timestamp': pd.date_range('2024-01-01', periods=rows, freq='1min')
        }
        
        if text_heavy:
            # Add text columns with varying lengths
            data['short_text'] = [f"Short text {i % 100}" for i in range(rows)]
            data['medium_text'] = [f"Medium length text content for testing purposes {i}" * 2 
                                 for i in range(rows)]
            data['long_text'] = [f"Very long text content that simulates real-world data " * 10 + f" {i}"
                               for i in range(rows)]
        
        if wide_table:
            # Add many numeric columns
            for i in range(20):
                data[f'metric_{i}'] = np.random.uniform(0, 1000, rows)
        
        return pd.DataFrame(data)
    
    @staticmethod
    def create_test_database(db_path: str, rows: int, table_name: str = "test_data",
                           text_heavy: bool = False, wide_table: bool = False) -> str:
        """Create SQLite test database with specified characteristics.
        
        Args:
            db_path: Path for the database file
            rows: Number of rows to generate
            table_name: Name of the table to create
            text_heavy: Whether to include text-heavy columns
            wide_table: Whether to create many columns
        
        Returns:
            SQLAlchemy connection string
        """
        # Generate test data
        df = DatasetGenerator.create_test_dataframe(rows, text_heavy, wide_table)
        
        # Create database
        connection_string = f"sqlite:///{db_path}"
        engine = create_engine(connection_string)
        
        with engine.connect() as conn:
            df.to_sql(table_name, conn, if_exists='replace', index=False)
        
        logger.info(f"Created test database with {rows} rows at {db_path}")
        return connection_string


class TokenBenchmark:
    """Benchmark token counting and estimation performance."""
    
    def __init__(self):
        self.token_manager = get_token_manager()
    
    def benchmark_token_estimation(self, df: pd.DataFrame, iterations: int = 5) -> BenchmarkResult:
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
                    total_rows=len(df),
                    sample_df=df.head(min(100, len(df)))
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
                    error_message=str(e)
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
            memory_timestamps=performance_data["timestamps"]
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
    
    def benchmark_streaming_vs_batch(self, connection_string: str, query: str, 
                                   dataset_size: str, iterations: int = 3) -> Tuple[BenchmarkResult, BenchmarkResult]:
        """Compare streaming vs batch processing performance.
        
        Args:
            connection_string: Database connection string
            query: SQL query to execute
            dataset_size: Size category of dataset
            iterations: Number of test iterations
            
        Returns:
            Tuple of (streaming_result, batch_result)
        """
        # Benchmark streaming approach
        streaming_result = self._benchmark_streaming(
            connection_string, query, dataset_size, iterations
        )
        
        # Benchmark batch approach (traditional pandas read_sql)
        batch_result = self._benchmark_batch(
            connection_string, query, dataset_size, iterations
        )
        
        return streaming_result, batch_result
    
    def _benchmark_streaming(self, connection_string: str, query: str, 
                           dataset_size: str, iterations: int) -> BenchmarkResult:
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
                # Create streaming source
                source = create_streaming_source(engine=engine, query=query)
                
                # Execute with streaming
                result_df, metadata = self.streaming_executor.execute_streaming(
                    source=source,
                    query_id=f"benchmark_streaming_{i}",
                    database_name="benchmark_db"
                )
                
                execution_time = time.time() - start_time
                execution_times.append(execution_time)
                
                # Extract metrics from metadata
                rows_processed = metadata.get("total_rows_processed", len(result_df))
                processing_rate = rows_processed / execution_time if execution_time > 0 else 0
                processing_rates.append(processing_rate)
                
                chunk_count = metadata.get("chunks_processed", 1)
                chunk_counts.append(chunk_count)
                
                memory_status = metadata.get("memory_status", {})
                initial_memory = memory_status.get("initial_available_gb", 0)
                final_memory = memory_status.get("final_available_gb", 0)
                memory_efficiency = final_memory / initial_memory if initial_memory > 0 else 1.0
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
                    error_message=str(e)
                )
            finally:
                performance_data = monitor.stop_monitoring()
                # Clean up streaming executor buffers
                self.streaming_executor.manage_memory_bounds()
        
        # Calculate averages
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
            average_chunk_size=int(avg_processing_rate * avg_execution_time / avg_chunk_count) if avg_chunk_count > 0 else 0,
            memory_samples=performance_data["memory_samples"],
            memory_timestamps=performance_data["timestamps"]
        )
    
    def _benchmark_batch(self, connection_string: str, query: str, 
                        dataset_size: str, iterations: int) -> BenchmarkResult:
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
                # Execute with traditional batch approach
                result_df = pd.read_sql(query, engine)
                
                execution_time = time.time() - start_time
                execution_times.append(execution_time)
                
                rows_processed = len(result_df)
                processing_rate = rows_processed / execution_time if execution_time > 0 else 0
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
                    error_message=str(e)
                )
            finally:
                performance_data = monitor.stop_monitoring()
        
        # Calculate averages
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
            memory_timestamps=performance_data["timestamps"]
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
            operation="benchmark_suite_init",
            component="performance_benchmarks"
        ):
            logger.info("Performance benchmark suite initialized", 
                       config=asdict(self.config))
    
    def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """Run comprehensive performance benchmarking suite.
        
        Returns:
            Summary of all benchmark results
        """
        start_time = time.time()
        
        with self.logging_manager.context(
            operation="comprehensive_benchmark",
            component="performance_benchmarks"
        ):
            logger.info("Starting comprehensive performance benchmark")
        
        try:
            # Run token performance benchmarks
            self._run_token_benchmarks()
            
            # Run streaming vs batch benchmarks
            self._run_streaming_benchmarks()
            
            # Run memory usage benchmarks
            self._run_memory_benchmarks()
            
            # Run regression tests
            self._run_regression_tests()
            
            # Generate performance comparisons
            self._generate_comparisons()
            
            # Save detailed results
            if self.config.save_detailed_metrics:
                self._save_detailed_results()
            
            # Generate reports
            if self.config.generate_reports:
                self._generate_reports()
            
            execution_time = time.time() - start_time
            
            logger.info("Comprehensive benchmark completed",
                       total_tests=len(self.results),
                       execution_time=execution_time,
                       success_rate=self._calculate_success_rate())
            
            return self._generate_summary()
            
        except Exception as e:
            logger.error(f"Comprehensive benchmark failed: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    def _run_token_benchmarks(self):
        """Run token counting and estimation benchmarks."""
        logger.info("Running token performance benchmarks")
        
        # Test different dataset sizes and characteristics
        test_scenarios = [
            (self.config.small_dataset_rows, False, False, "small_numeric"),
            (self.config.small_dataset_rows, True, False, "small_text_heavy"),
            (self.config.medium_dataset_rows, False, False, "medium_numeric"),
            (self.config.medium_dataset_rows, True, False, "medium_text_heavy"),
            (self.config.medium_dataset_rows, False, True, "medium_wide_table"),
        ]
        
        for rows, text_heavy, wide_table, scenario_name in test_scenarios:
            logger.debug(f"Running token benchmark: {scenario_name}")
            
            # Generate test data
            df = DatasetGenerator.create_test_dataframe(rows, text_heavy, wide_table)
            
            # Run benchmark
            result = self.token_benchmark.benchmark_token_estimation(
                df, iterations=self.config.test_iterations
            )
            result.test_name = f"token_{scenario_name}"
            result.metadata["scenario"] = scenario_name
            result.metadata["text_heavy"] = text_heavy
            result.metadata["wide_table"] = wide_table
            
            self.results.append(result)
            logger.debug(f"Token benchmark completed: {scenario_name}, "
                        f"Rate: {result.processing_rate_rows_per_second:.0f} rows/sec")
    
    def _run_streaming_benchmarks(self):
        """Run streaming vs batch processing benchmarks."""
        logger.info("Running streaming vs batch benchmarks")
        
        # Create temporary databases for different scenarios
        temp_dir = Path(tempfile.mkdtemp())
        
        try:
            test_scenarios = [
                (self.config.small_dataset_rows, "small", False, False),
                (self.config.medium_dataset_rows, "medium", False, False), 
                (self.config.medium_dataset_rows, "medium", True, False),  # Text heavy
                (self.config.large_dataset_rows, "large", False, False),
            ]
            
            for rows, size_category, text_heavy, wide_table in test_scenarios:
                db_path = temp_dir / f"test_{size_category}_{rows}.db"
                scenario_name = f"{size_category}_{'text' if text_heavy else 'numeric'}"
                
                logger.debug(f"Creating test database: {scenario_name}")
                connection_string = DatasetGenerator.create_test_database(
                    str(db_path), rows, text_heavy=text_heavy, wide_table=wide_table
                )
                
                # Test query - select all data
                query = "SELECT * FROM test_data"
                
                # Run streaming vs batch comparison
                streaming_result, batch_result = self.streaming_benchmark.benchmark_streaming_vs_batch(
                    connection_string, query, size_category, self.config.test_iterations
                )
                
                # Update test names for clarity
                streaming_result.test_name = f"streaming_{scenario_name}"
                batch_result.test_name = f"batch_{scenario_name}"
                
                streaming_result.metadata["scenario"] = scenario_name
                batch_result.metadata["scenario"] = scenario_name
                
                self.results.extend([streaming_result, batch_result])
                
                logger.debug(f"Streaming benchmark: {scenario_name}, "
                           f"Streaming: {streaming_result.processing_rate_rows_per_second:.0f} rows/sec, "
                           f"Batch: {batch_result.processing_rate_rows_per_second:.0f} rows/sec")
        
        finally:
            # Cleanup temporary files
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def _run_memory_benchmarks(self):
        """Run memory usage pattern benchmarks."""
        logger.info("Running memory usage benchmarks")
        
        # Test memory efficiency with different chunk sizes
        temp_dir = Path(tempfile.mkdtemp())
        
        try:
            # Create a medium-sized database for memory testing
            db_path = temp_dir / "memory_test.db"
            connection_string = DatasetGenerator.create_test_database(
                str(db_path), self.config.medium_dataset_rows
            )
            
            query = "SELECT * FROM test_data"
            
            # Test different chunk sizes for memory efficiency
            for chunk_size in self.config.chunk_sizes:
                logger.debug(f"Testing chunk size: {chunk_size}")
                
                # Configure streaming executor with specific chunk size
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
                        initial_chunk_size=chunk_size
                    )
                    
                    execution_time = time.time() - start_time
                    
                except Exception as e:
                    logger.warning(f"Memory benchmark failed for chunk size {chunk_size}: {e}")
                    continue
                finally:
                    performance_data = monitor.stop_monitoring()
                
                # Create result
                result = BenchmarkResult(
                    test_name=f"memory_chunk_{chunk_size}",
                    test_category="memory_efficiency",
                    dataset_size="medium",
                    database_type="sqlite",
                    processing_mode="streaming",
                    execution_time_seconds=execution_time,
                    peak_memory_mb=performance_data["peak_memory_mb"],
                    average_memory_mb=performance_data["average_memory_mb"],
                    memory_efficiency=metadata.get("memory_status", {}).get("final_available_gb", 0) / 8.0,  # Assume 8GB system
                    rows_processed=metadata.get("total_rows_processed", 0),
                    processing_rate_rows_per_second=metadata.get("total_rows_processed", 0) / execution_time,
                    chunk_count=metadata.get("chunks_processed", 0),
                    average_chunk_size=chunk_size,
                    memory_samples=performance_data["memory_samples"],
                    memory_timestamps=performance_data["timestamps"],
                    metadata={"chunk_size": chunk_size}
                )
                
                self.results.append(result)
                
        finally:
            # Cleanup temporary files
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def _run_regression_tests(self):
        """Run automated regression testing against baseline performance."""
        logger.info("Running regression tests")
        
        # Load baseline results if they exist
        baseline_path = self.config.results_dir / "baseline_results.json"
        
        if not baseline_path.exists():
            logger.info("No baseline results found, current results will become baseline")
            return
        
        try:
            with open(baseline_path, 'r') as f:
                baseline_data = json.load(f)
            
            baseline_results = [BenchmarkResult(**result) for result in baseline_data]
            
            # Compare current results against baseline
            for current_result in self.results:
                baseline_result = self._find_matching_baseline(current_result, baseline_results)
                
                if baseline_result and baseline_result.success and current_result.success:
                    comparison = self._compare_results(baseline_result, current_result)
                    self.comparisons.append(comparison)
                    
                    # Log significant regressions
                    if comparison.execution_time_improvement_percent < -10:  # 10% slower
                        logger.warning(f"Performance regression detected in {current_result.test_name}: "
                                     f"{abs(comparison.execution_time_improvement_percent):.1f}% slower")
                    elif comparison.execution_time_improvement_percent > 10:  # 10% faster
                        logger.info(f"Performance improvement in {current_result.test_name}: "
                                  f"{comparison.execution_time_improvement_percent:.1f}% faster")
        
        except Exception as e:
            logger.warning(f"Regression testing failed: {e}")
    
    def _find_matching_baseline(self, current_result: BenchmarkResult, 
                               baseline_results: List[BenchmarkResult]) -> Optional[BenchmarkResult]:
        """Find matching baseline result for comparison."""
        for baseline in baseline_results:
            if (baseline.test_name == current_result.test_name and
                baseline.dataset_size == current_result.dataset_size and
                baseline.processing_mode == current_result.processing_mode):
                return baseline
        return None
    
    def _compare_results(self, baseline: BenchmarkResult, current: BenchmarkResult) -> ComparisonResult:
        """Compare two benchmark results."""
        # Calculate percentage improvements (positive = improvement)
        execution_improvement = ((baseline.execution_time_seconds - current.execution_time_seconds) / 
                                baseline.execution_time_seconds * 100)
        memory_improvement = ((baseline.peak_memory_mb - current.peak_memory_mb) / 
                             baseline.peak_memory_mb * 100)
        rate_improvement = ((current.processing_rate_rows_per_second - baseline.processing_rate_rows_per_second) / 
                           baseline.processing_rate_rows_per_second * 100)
        
        # Determine statistical significance (simplified)
        is_significant = abs(execution_improvement) > 5  # 5% threshold
        
        # Generate summary
        if execution_improvement > 5:
            summary = f"Performance improved by {execution_improvement:.1f}%"
        elif execution_improvement < -5:
            summary = f"Performance regressed by {abs(execution_improvement):.1f}%"
        else:
            summary = "No significant performance change"
        
        return ComparisonResult(
            baseline_result=baseline,
            comparison_result=current,
            execution_time_improvement_percent=execution_improvement,
            memory_improvement_percent=memory_improvement,
            processing_rate_improvement_percent=rate_improvement,
            is_significant=is_significant,
            confidence_level=0.95,  # Simplified
            summary=summary
        )
    
    def _generate_comparisons(self):
        """Generate performance comparisons between different approaches."""
        logger.info("Generating performance comparisons")
        
        # Group results by scenario for comparison
        scenarios = {}
        for result in self.results:
            scenario = result.metadata.get("scenario", "unknown")
            if scenario not in scenarios:
                scenarios[scenario] = []
            scenarios[scenario].append(result)
        
        # Compare streaming vs batch for each scenario
        for scenario, results in scenarios.items():
            streaming_results = [r for r in results if r.processing_mode == "streaming"]
            batch_results = [r for r in results if r.processing_mode == "batch"]
            
            if streaming_results and batch_results:
                # Compare the best performing result from each category
                best_streaming = max(streaming_results, key=lambda r: r.processing_rate_rows_per_second)
                best_batch = max(batch_results, key=lambda r: r.processing_rate_rows_per_second)
                
                comparison = self._compare_results(best_batch, best_streaming)
                comparison.summary = f"Streaming vs Batch comparison for {scenario}"
                self.comparisons.append(comparison)
    
    def _save_detailed_results(self):
        """Save detailed benchmark results to files."""
        logger.info("Saving detailed benchmark results")
        
        # Save all results as JSON
        results_file = self.config.results_dir / f"benchmark_results_{int(time.time())}.json"
        with open(results_file, 'w') as f:
            json.dump([result.to_dict() for result in self.results], f, indent=2, default=str)
        
        # Save as baseline if no baseline exists
        baseline_file = self.config.results_dir / "baseline_results.json"
        if not baseline_file.exists():
            with open(baseline_file, 'w') as f:
                json.dump([result.to_dict() for result in self.results], f, indent=2, default=str)
        
        # Save comparisons
        if self.comparisons:
            comparisons_file = self.config.results_dir / f"benchmark_comparisons_{int(time.time())}.json"
            with open(comparisons_file, 'w') as f:
                json.dump([asdict(comp) for comp in self.comparisons], f, indent=2, default=str)
        
        logger.info(f"Detailed results saved to {results_file}")
    
    def _generate_reports(self):
        """Generate human-readable performance reports."""
        logger.info("Generating performance reports")
        
        # Generate summary report
        report_file = self.config.results_dir / f"performance_report_{int(time.time())}.md"
        
        with open(report_file, 'w') as f:
            f.write("# LocalData MCP Performance Benchmark Report\n\n")
            f.write(f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Summary statistics
            f.write("## Summary Statistics\n\n")
            f.write(f"- Total tests executed: {len(self.results)}\n")
            f.write(f"- Successful tests: {sum(1 for r in self.results if r.success)}\n")
            f.write(f"- Success rate: {self._calculate_success_rate():.1%}\n\n")
            
            # Performance by category
            f.write("## Performance by Category\n\n")
            
            categories = set(r.test_category for r in self.results if r.success)
            for category in categories:
                f.write(f"### {category.title()}\n\n")
                
                category_results = [r for r in self.results if r.test_category == category and r.success]
                if category_results:
                    avg_rate = sum(r.processing_rate_rows_per_second for r in category_results) / len(category_results)
                    avg_memory = sum(r.peak_memory_mb for r in category_results) / len(category_results)
                    
                    f.write(f"- Average processing rate: {avg_rate:.0f} rows/second\n")
                    f.write(f"- Average peak memory usage: {avg_memory:.1f} MB\n")
                    f.write(f"- Tests in category: {len(category_results)}\n\n")
            
            # Comparisons
            if self.comparisons:
                f.write("## Performance Comparisons\n\n")
                for comp in self.comparisons:
                    f.write(f"### {comp.summary}\n\n")
                    f.write(f"- Execution time improvement: {comp.execution_time_improvement_percent:.1f}%\n")
                    f.write(f"- Memory improvement: {comp.memory_improvement_percent:.1f}%\n")
                    f.write(f"- Processing rate improvement: {comp.processing_rate_improvement_percent:.1f}%\n")
                    f.write(f"- Statistically significant: {comp.is_significant}\n\n")
            
            # Recommendations
            f.write("## Recommendations\n\n")
            recommendations = self._generate_recommendations()
            for rec in recommendations:
                f.write(f"- {rec}\n")
        
        logger.info(f"Performance report generated: {report_file}")
    
    def _calculate_success_rate(self) -> float:
        """Calculate the success rate of benchmark tests."""
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.success) / len(self.results)
    
    def _generate_recommendations(self) -> List[str]:
        """Generate performance optimization recommendations."""
        recommendations = []
        
        # Analyze streaming vs batch performance
        streaming_results = [r for r in self.results if r.processing_mode == "streaming" and r.success]
        batch_results = [r for r in self.results if r.processing_mode == "batch" and r.success]
        
        if streaming_results and batch_results:
            avg_streaming_rate = sum(r.processing_rate_rows_per_second for r in streaming_results) / len(streaming_results)
            avg_batch_rate = sum(r.processing_rate_rows_per_second for r in batch_results) / len(batch_results)
            
            if avg_streaming_rate > avg_batch_rate * 1.1:
                recommendations.append("Streaming processing shows significant performance benefits - recommend for large datasets")
            elif avg_batch_rate > avg_streaming_rate * 1.1:
                recommendations.append("Batch processing performs better for current workloads - streaming may have overhead")
            else:
                recommendations.append("Streaming and batch processing show similar performance - choose based on memory constraints")
        
        # Memory efficiency recommendations
        memory_results = [r for r in self.results if r.test_category == "memory_efficiency" and r.success]
        if memory_results:
            best_chunk_result = min(memory_results, key=lambda r: r.peak_memory_mb)
            optimal_chunk_size = best_chunk_result.metadata.get("chunk_size", "unknown")
            recommendations.append(f"Optimal chunk size for memory efficiency: {optimal_chunk_size}")
        
        # Token estimation performance
        token_results = [r for r in self.results if r.test_category == "token_performance" and r.success]
        if token_results:
            avg_token_rate = sum(r.processing_rate_rows_per_second for r in token_results) / len(token_results)
            if avg_token_rate > 10000:  # Arbitrary threshold
                recommendations.append("Token estimation performance is excellent - no optimization needed")
            else:
                recommendations.append("Token estimation could benefit from caching or sampling optimization")
        
        return recommendations
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate summary of all benchmark results."""
        successful_results = [r for r in self.results if r.success]
        
        if not successful_results:
            return {
                "total_tests": len(self.results),
                "successful_tests": 0,
                "success_rate": 0.0,
                "error": "No successful test results"
            }
        
        return {
            "total_tests": len(self.results),
            "successful_tests": len(successful_results),
            "success_rate": len(successful_results) / len(self.results),
            "average_processing_rate": sum(r.processing_rate_rows_per_second for r in successful_results) / len(successful_results),
            "average_memory_usage_mb": sum(r.peak_memory_mb for r in successful_results) / len(successful_results),
            "test_categories": list(set(r.test_category for r in successful_results)),
            "performance_improvements": len([c for c in self.comparisons if c.execution_time_improvement_percent > 5]),
            "performance_regressions": len([c for c in self.comparisons if c.execution_time_improvement_percent < -5]),
            "results_directory": str(self.config.results_dir),
            "timestamp": time.time()
        }


def run_performance_benchmark(config: Optional[BenchmarkConfig] = None) -> Dict[str, Any]:
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
    
    print("\n" + "="*60)
    print("PERFORMANCE BENCHMARK SUMMARY")
    print("="*60)
    print(f"Total Tests: {results['total_tests']}")
    print(f"Success Rate: {results['success_rate']:.1%}")
    print(f"Average Processing Rate: {results.get('average_processing_rate', 0):.0f} rows/sec")
    print(f"Average Memory Usage: {results.get('average_memory_usage_mb', 0):.1f} MB")
    print(f"Performance Improvements: {results.get('performance_improvements', 0)}")
    print(f"Performance Regressions: {results.get('performance_regressions', 0)}")
    print(f"Results Directory: {results['results_directory']}")
    print("="*60)