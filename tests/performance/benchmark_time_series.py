"""
Time Series Performance Benchmarking Suite

This comprehensive benchmarking suite validates the performance characteristics
of the Time Series Analysis Domain under various data loads and scenarios.

Benchmarks Include:
- High-frequency data processing (10k-100k points)
- Streaming architecture effectiveness
- Memory usage optimization  
- Scalability across different data sizes
- Tool discovery and initialization performance
- Cross-domain operation performance
- Concurrent analysis performance

Performance Standards:
- High-frequency data (10k+ points): < 30 seconds processing
- Memory usage: within 16-64GB bounds under load  
- Tool discovery: < 100ms
- Forecast generation: < 60 seconds for typical datasets
- Streaming operations: maintain consistent performance
"""

import time
import psutil
import os
import json
import numpy as np
import pandas as pd
import sqlite3
import tempfile
import threading
import concurrent.futures
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, asdict
from contextlib import contextmanager

# Import test utilities
import sys
sys.path.append(str(Path(__file__).parent.parent))
from test_utils import (
    create_synthetic_time_series,
    generate_high_frequency_data,
    create_multivariate_time_series,
    measure_performance,
    generate_performance_report
)


@dataclass
class BenchmarkResult:
    """Data class for benchmark results."""
    test_name: str
    data_size: int
    execution_time: float
    memory_used: float  
    memory_peak: float
    success: bool
    error_message: str = ""
    additional_metrics: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.additional_metrics is None:
            self.additional_metrics = {}


@dataclass  
class BenchmarkConfig:
    """Configuration for benchmark runs."""
    data_sizes: List[int]
    time_limits: Dict[str, float]  # Maximum allowed time per operation
    memory_limits: Dict[str, float]  # Maximum allowed memory increase  
    iterations: int = 3  # Number of iterations to average
    warmup_iterations: int = 1  # Warmup iterations
    

class TimeSeriesPerformanceBenchmark:
    """Comprehensive performance benchmarking for time series analysis."""
    
    def __init__(self, benchmark_config: BenchmarkConfig):
        self.config = benchmark_config
        self.results: List[BenchmarkResult] = []
        self.baseline_memory = 0
        self.test_databases = {}
        
    def setup_benchmark_environment(self):
        """Set up the benchmark environment with test data."""
        print("🔧 Setting up benchmark environment...")
        
        # Record baseline memory
        process = psutil.Process(os.getpid())
        self.baseline_memory = process.memory_info().rss / (1024 * 1024)  # MB
        
        # Create temporary directory for test databases
        self.temp_dir = tempfile.mkdtemp(prefix="timeseries_benchmark_")
        
        # Generate test datasets of various sizes
        self.create_benchmark_datasets()
        
        print(f"✅ Benchmark environment ready (baseline memory: {self.baseline_memory:.1f}MB)")
    
    def create_benchmark_datasets(self):
        """Create benchmark datasets of various sizes."""
        print("📊 Creating benchmark datasets...")
        
        for size in self.config.data_sizes:
            print(f"  Creating dataset with {size:,} points...")
            
            # Determine frequency based on size for realistic data
            if size <= 1000:
                freq, periods = 'D', size
                start_date = '2023-01-01'
            elif size <= 10000:
                freq, periods = 'H', size
                start_date = '2023-01-01'
            else:
                freq, periods = '5min', size  
                start_date = '2023-01-01'
            
            # Create end date based on periods and frequency
            start = pd.to_datetime(start_date)
            if freq == 'D':
                end_date = start + pd.Timedelta(days=periods-1)
            elif freq == 'H':  
                end_date = start + pd.Timedelta(hours=periods-1)
            else:  # 5min
                end_date = start + pd.Timedelta(minutes=5*(periods-1))
            
            # Generate synthetic time series
            ts_data = create_synthetic_time_series(
                start_date=start_date,
                end_date=str(end_date.date()),
                freq=freq,
                trend=0.05,
                seasonality=True,
                seasonal_periods=min(365, size//10) if freq == 'D' else min(24, size//100),
                noise_level=0.1,
                anomalies=0.01
            )
            
            # Ensure we have the right size
            if len(ts_data) != size:
                ts_data = ts_data.iloc[:size] if len(ts_data) > size else ts_data
            
            # Create database
            db_path = Path(self.temp_dir) / f"benchmark_data_{size}.db"
            with sqlite3.connect(db_path) as conn:
                ts_data.to_sql('timeseries_data', conn, index=False, if_exists='replace')
            
            self.test_databases[size] = str(db_path)
        
        print(f"✅ Created {len(self.test_databases)} benchmark datasets")
    
    @contextmanager
    def memory_monitor(self):
        """Context manager for monitoring memory usage."""
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / (1024 * 1024)
        peak_memory = initial_memory
        
        # Start monitoring thread
        monitoring = True
        
        def monitor():
            nonlocal peak_memory
            while monitoring:
                current_memory = process.memory_info().rss / (1024 * 1024)
                peak_memory = max(peak_memory, current_memory)
                time.sleep(0.1)
        
        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()
        
        try:
            yield lambda: (
                process.memory_info().rss / (1024 * 1024) - initial_memory,
                peak_memory - initial_memory
            )
        finally:
            monitoring = False
            monitor_thread.join(timeout=1)
    
    def run_benchmark_operation(self, operation_name: str, operation_func, data_size: int) -> BenchmarkResult:
        """Run a single benchmark operation with monitoring."""
        
        # Warmup iterations
        for _ in range(self.config.warmup_iterations):
            try:
                operation_func()
            except Exception:
                pass  # Ignore warmup errors
        
        # Actual benchmark iterations
        execution_times = []
        memory_increases = []
        memory_peaks = []
        success = True
        error_message = ""
        
        for iteration in range(self.config.iterations):
            try:
                with self.memory_monitor() as get_memory:
                    start_time = time.time()
                    result = operation_func()
                    execution_time = time.time() - start_time
                    
                    memory_used, memory_peak = get_memory()
                    
                    execution_times.append(execution_time)
                    memory_increases.append(memory_used)
                    memory_peaks.append(memory_peak)
                    
                    # Validate result if it's JSON
                    if isinstance(result, str):
                        try:
                            json.loads(result)
                        except json.JSONDecodeError:
                            success = False
                            error_message = "Invalid JSON result"
                            break
                    
            except Exception as e:
                success = False
                error_message = str(e)
                break
        
        # Calculate averages
        avg_execution_time = np.mean(execution_times) if execution_times else float('inf')
        avg_memory_used = np.mean(memory_increases) if memory_increases else 0
        avg_memory_peak = np.mean(memory_peaks) if memory_peaks else 0
        
        return BenchmarkResult(
            test_name=operation_name,
            data_size=data_size,
            execution_time=avg_execution_time,
            memory_used=avg_memory_used,
            memory_peak=avg_memory_peak,
            success=success,
            error_message=error_message,
            additional_metrics={
                'execution_times': execution_times,
                'memory_increases': memory_increases,
                'iterations': len(execution_times)
            }
        )
    
    def benchmark_basic_analysis(self):
        """Benchmark basic time series analysis across data sizes."""
        print("\n📊 Benchmarking Basic Time Series Analysis...")
        
        from src.localdata_mcp.localdata_mcp import _db_manager
        
        for data_size in self.config.data_sizes:
            print(f"  Testing with {data_size:,} data points...")
            
            # Connect to database
            db_path = self.test_databases[data_size]
            conn_result = _db_manager.connect_database(
                name=f"benchmark_{data_size}",
                db_type="sqlite",
                conn_string=db_path
            )
            
            if not json.loads(conn_result)["success"]:
                print(f"    ❌ Failed to connect to database for size {data_size}")
                continue
            
            def run_basic_analysis():
                return _db_manager.analyze_time_series_basic(
                    name=f"benchmark_{data_size}",
                    table_name="timeseries_data",
                    date_column="date",
                    value_column="value"
                )
            
            result = self.run_benchmark_operation(
                f"basic_analysis_{data_size}",
                run_basic_analysis,
                data_size
            )
            
            self.results.append(result)
            
            # Check against time limits
            time_limit = self.config.time_limits.get('basic_analysis', 30.0)
            status = "✅" if result.success and result.execution_time <= time_limit else "❌"
            
            print(f"    {status} {data_size:,} points: {result.execution_time:.2f}s "
                  f"(mem: +{result.memory_used:.1f}MB)")
            
            if not result.success:
                print(f"      Error: {result.error_message}")
    
    def benchmark_forecasting(self):
        """Benchmark ARIMA and exponential smoothing forecasting."""
        print("\n🔮 Benchmarking Forecasting Operations...")
        
        from src.localdata_mcp.localdata_mcp import _db_manager
        
        forecasting_methods = [
            ('arima', lambda name: _db_manager.forecast_arima(
                name=name, table_name="timeseries_data", 
                date_column="date", value_column="value",
                forecast_steps=30, auto_arima=True
            )),
            ('exponential_smoothing', lambda name: _db_manager.forecast_exponential_smoothing(
                name=name, table_name="timeseries_data",
                date_column="date", value_column="value", 
                method="auto", forecast_steps=30
            ))
        ]
        
        # Test on smaller datasets for forecasting (it's computationally intensive)
        forecast_sizes = [size for size in self.config.data_sizes if size <= 10000]
        
        for method_name, method_func in forecasting_methods:
            print(f"  📈 Testing {method_name}...")
            
            for data_size in forecast_sizes:
                print(f"    Testing with {data_size:,} data points...")
                
                # Ensure database connection
                db_path = self.test_databases[data_size]
                conn_result = _db_manager.connect_database(
                    name=f"forecast_benchmark_{data_size}",
                    db_type="sqlite",
                    conn_string=db_path
                )
                
                if not json.loads(conn_result)["success"]:
                    continue
                
                def run_forecast():
                    return method_func(f"forecast_benchmark_{data_size}")
                
                result = self.run_benchmark_operation(
                    f"forecast_{method_name}_{data_size}",
                    run_forecast,
                    data_size
                )
                
                self.results.append(result)
                
                # Check against time limits  
                time_limit = self.config.time_limits.get('forecasting', 60.0)
                status = "✅" if result.success and result.execution_time <= time_limit else "❌"
                
                print(f"      {status} {data_size:,} points: {result.execution_time:.2f}s "
                      f"(mem: +{result.memory_used:.1f}MB)")
                
                if not result.success:
                    print(f"        Error: {result.error_message}")
    
    def benchmark_anomaly_detection(self):
        """Benchmark time series anomaly detection."""
        print("\n🔍 Benchmarking Anomaly Detection...")
        
        from src.localdata_mcp.localdata_mcp import _db_manager
        
        detection_methods = ['statistical', 'isolation_forest']
        
        for method in detection_methods:
            print(f"  🕵️  Testing {method} anomaly detection...")
            
            for data_size in self.config.data_sizes:
                print(f"    Testing with {data_size:,} data points...")
                
                # Connect to database
                db_path = self.test_databases[data_size]
                conn_result = _db_manager.connect_database(
                    name=f"anomaly_benchmark_{data_size}_{method}",
                    db_type="sqlite",
                    conn_string=db_path
                )
                
                if not json.loads(conn_result)["success"]:
                    continue
                
                def run_anomaly_detection():
                    return _db_manager.detect_time_series_anomalies(
                        name=f"anomaly_benchmark_{data_size}_{method}",
                        table_name="timeseries_data",
                        date_column="date",
                        value_column="value",
                        method=method,
                        contamination=0.05
                    )
                
                result = self.run_benchmark_operation(
                    f"anomaly_{method}_{data_size}",
                    run_anomaly_detection,
                    data_size
                )
                
                self.results.append(result)
                
                # Check performance
                time_limit = self.config.time_limits.get('anomaly_detection', 30.0)
                status = "✅" if result.success and result.execution_time <= time_limit else "❌"
                
                print(f"      {status} {data_size:,} points: {result.execution_time:.2f}s "
                      f"(mem: +{result.memory_used:.1f}MB)")
                
                if not result.success:
                    print(f"        Error: {result.error_message}")
    
    def benchmark_streaming_operations(self):
        """Benchmark streaming-specific operations."""
        print("\n📡 Benchmarking Streaming Operations...")
        
        from src.localdata_mcp.localdata_mcp import _db_manager
        
        # Tool discovery benchmark
        print("  ⚡ Testing tool discovery performance...")
        
        def tool_discovery():
            return _db_manager.get_streaming_status()
        
        result = self.run_benchmark_operation(
            "tool_discovery",
            tool_discovery,
            0  # No data size for tool discovery
        )
        
        self.results.append(result)
        
        discovery_limit = 0.1  # 100ms limit
        status = "✅" if result.success and result.execution_time <= discovery_limit else "❌"
        print(f"    {status} Tool discovery: {result.execution_time*1000:.1f}ms "
              f"(limit: {discovery_limit*1000:.0f}ms)")
        
        # Memory management benchmark
        print("  🧹 Testing memory management performance...")
        
        def memory_management():
            return _db_manager.manage_memory_bounds()
        
        result = self.run_benchmark_operation(
            "memory_management",
            memory_management,
            0
        )
        
        self.results.append(result)
        
        mgmt_limit = 1.0  # 1 second limit
        status = "✅" if result.success and result.execution_time <= mgmt_limit else "❌"
        print(f"    {status} Memory management: {result.execution_time:.3f}s "
              f"(limit: {mgmt_limit:.1f}s)")
    
    def benchmark_concurrent_operations(self):
        """Benchmark concurrent time series operations."""
        print("\n🔄 Benchmarking Concurrent Operations...")
        
        from src.localdata_mcp.localdata_mcp import _db_manager
        
        # Use medium-sized dataset for concurrency test
        test_size = 1000 if 1000 in self.config.data_sizes else self.config.data_sizes[0]
        db_path = self.test_databases[test_size]
        
        # Setup connections for concurrent access
        connection_names = [f"concurrent_test_{i}" for i in range(3)]
        
        for name in connection_names:
            conn_result = _db_manager.connect_database(
                name=name,
                db_type="sqlite",
                conn_string=db_path
            )
            if not json.loads(conn_result)["success"]:
                print(f"    ❌ Failed to setup concurrent connection {name}")
                return
        
        def concurrent_analysis(connection_name: str) -> Dict[str, Any]:
            """Run analysis on a specific connection."""
            start_time = time.time()
            
            # Basic analysis
            basic_result = _db_manager.analyze_time_series_basic(
                name=connection_name,
                table_name="timeseries_data",
                date_column="date",
                value_column="value"
            )
            
            # Anomaly detection
            anomaly_result = _db_manager.detect_time_series_anomalies(
                name=connection_name,
                table_name="timeseries_data",
                date_column="date",
                value_column="value",
                method="statistical"
            )
            
            execution_time = time.time() - start_time
            
            return {
                'connection': connection_name,
                'execution_time': execution_time,
                'basic_success': 'error' not in json.loads(basic_result),
                'anomaly_success': 'error' not in json.loads(anomaly_result)
            }
        
        # Run concurrent operations
        print(f"  🚀 Running 3 concurrent analyses on {test_size:,} point dataset...")
        
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(concurrent_analysis, name) for name in connection_names]
            concurrent_results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        total_concurrent_time = time.time() - start_time
        
        # Analyze concurrent results
        all_successful = all(
            r['basic_success'] and r['anomaly_success'] for r in concurrent_results
        )
        
        avg_individual_time = np.mean([r['execution_time'] for r in concurrent_results])
        
        # Create benchmark result
        result = BenchmarkResult(
            test_name="concurrent_operations",
            data_size=test_size,
            execution_time=total_concurrent_time,
            memory_used=0,  # Not measured for concurrent test
            memory_peak=0,
            success=all_successful,
            additional_metrics={
                'individual_times': [r['execution_time'] for r in concurrent_results],
                'avg_individual_time': avg_individual_time,
                'parallelization_efficiency': avg_individual_time / total_concurrent_time
            }
        )
        
        self.results.append(result)
        
        status = "✅" if all_successful else "❌"
        efficiency = avg_individual_time / total_concurrent_time
        
        print(f"    {status} Concurrent operations: {total_concurrent_time:.2f}s total")
        print(f"      Average individual time: {avg_individual_time:.2f}s")
        print(f"      Parallelization efficiency: {efficiency:.1f}x")
    
    def benchmark_memory_stress_test(self):
        """Stress test memory usage with large datasets."""
        print("\n🧠 Memory Stress Testing...")
        
        from src.localdata_mcp.localdata_mcp import _db_manager
        
        # Use the largest available dataset
        max_size = max(self.config.data_sizes)
        
        if max_size < 10000:
            print("  ⚠️  Skipping memory stress test (largest dataset < 10k points)")
            return
        
        print(f"  💾 Testing memory usage with {max_size:,} point dataset...")
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / (1024 * 1024)
        
        # Connect to largest database
        db_path = self.test_databases[max_size]
        conn_result = _db_manager.connect_database(
            name="memory_stress_test",
            db_type="sqlite", 
            conn_string=db_path
        )
        
        if not json.loads(conn_result)["success"]:
            print("    ❌ Failed to connect for memory stress test")
            return
        
        # Perform multiple memory-intensive operations
        operations = [
            ("Basic Analysis", lambda: _db_manager.analyze_time_series_basic(
                name="memory_stress_test", table_name="timeseries_data",
                date_column="date", value_column="value")),
            ("Change Point Detection", lambda: _db_manager.detect_change_points(
                name="memory_stress_test", table_name="timeseries_data",
                date_column="date", value_column="value")),
            ("Anomaly Detection", lambda: _db_manager.detect_time_series_anomalies(
                name="memory_stress_test", table_name="timeseries_data",
                date_column="date", value_column="value", method="isolation_forest"))
        ]
        
        peak_memory = initial_memory
        operation_results = []
        
        for operation_name, operation_func in operations:
            print(f"    🔍 {operation_name}...")
            
            before_memory = process.memory_info().rss / (1024 * 1024)
            
            start_time = time.time()
            try:
                result = operation_func()
                success = 'error' not in json.loads(result)
                execution_time = time.time() - start_time
            except Exception as e:
                success = False
                execution_time = time.time() - start_time
                
            after_memory = process.memory_info().rss / (1024 * 1024)
            peak_memory = max(peak_memory, after_memory)
            
            operation_results.append({
                'operation': operation_name,
                'success': success,
                'execution_time': execution_time,
                'memory_before': before_memory,
                'memory_after': after_memory,
                'memory_increase': after_memory - before_memory
            })
            
            print(f"      Memory: {before_memory:.1f}MB → {after_memory:.1f}MB "
                  f"(Δ{after_memory-before_memory:+.1f}MB) in {execution_time:.2f}s")
        
        total_memory_increase = peak_memory - initial_memory
        memory_limit = self.config.memory_limits.get('stress_test', 1000.0)  # 1GB limit
        
        result = BenchmarkResult(
            test_name="memory_stress_test",
            data_size=max_size,
            execution_time=sum(r['execution_time'] for r in operation_results),
            memory_used=total_memory_increase,
            memory_peak=peak_memory,
            success=total_memory_increase <= memory_limit and all(r['success'] for r in operation_results),
            additional_metrics={
                'operation_results': operation_results,
                'memory_limit': memory_limit,
                'initial_memory': initial_memory
            }
        )
        
        self.results.append(result)
        
        status = "✅" if result.success else "❌"
        print(f"    {status} Memory stress test: peak +{total_memory_increase:.1f}MB "
              f"(limit: {memory_limit:.0f}MB)")
    
    def run_full_benchmark_suite(self) -> Dict[str, Any]:
        """Run the complete benchmark suite."""
        print("🚀 STARTING TIME SERIES PERFORMANCE BENCHMARK SUITE")
        print("=" * 70)
        
        suite_start_time = time.time()
        
        # Setup
        self.setup_benchmark_environment()
        
        try:
            # Run all benchmarks
            self.benchmark_basic_analysis()
            self.benchmark_forecasting()
            self.benchmark_anomaly_detection() 
            self.benchmark_streaming_operations()
            self.benchmark_concurrent_operations()
            self.benchmark_memory_stress_test()
            
            suite_execution_time = time.time() - suite_start_time
            
            # Generate comprehensive report
            report = self.generate_benchmark_report(suite_execution_time)
            
            print(f"\n🎉 BENCHMARK SUITE COMPLETED")
            print("=" * 70)
            print(f"Total execution time: {suite_execution_time:.2f} seconds")
            
            return report
            
        finally:
            # Cleanup
            self.cleanup_benchmark_environment()
    
    def generate_benchmark_report(self, total_time: float) -> Dict[str, Any]:
        """Generate comprehensive benchmark report."""
        
        # Organize results by category
        categories = {}
        for result in self.results:
            category = result.test_name.split('_')[0]
            if category not in categories:
                categories[category] = []
            categories[category].append(result)
        
        # Calculate summary statistics
        total_tests = len(self.results)
        successful_tests = sum(1 for r in self.results if r.success)
        
        performance_summary = {
            'total_tests': total_tests,
            'successful_tests': successful_tests,
            'success_rate': (successful_tests / total_tests) * 100,
            'total_execution_time': total_time,
            'categories': {}
        }
        
        # Per-category analysis
        for category, results in categories.items():
            category_stats = {
                'test_count': len(results),
                'success_count': sum(1 for r in results if r.success),
                'avg_execution_time': np.mean([r.execution_time for r in results if r.success]),
                'max_execution_time': max([r.execution_time for r in results if r.success], default=0),
                'avg_memory_usage': np.mean([r.memory_used for r in results if r.success]),
                'max_memory_usage': max([r.memory_used for r in results if r.success], default=0),
                'results': [asdict(r) for r in results]
            }
            
            performance_summary['categories'][category] = category_stats
        
        # Performance standards validation
        standards_check = self.validate_performance_standards()
        performance_summary['standards_validation'] = standards_check
        
        # Print summary
        print(f"\n📊 PERFORMANCE SUMMARY")
        print("-" * 40)
        print(f"Tests: {successful_tests}/{total_tests} passed ({(successful_tests/total_tests)*100:.1f}%)")
        
        for category, stats in performance_summary['categories'].items():
            success_rate = (stats['success_count'] / stats['test_count']) * 100
            print(f"{category.title()}: {stats['success_count']}/{stats['test_count']} "
                  f"({success_rate:.1f}%) - avg: {stats['avg_execution_time']:.2f}s")
        
        print(f"\n📋 Standards Validation:")
        for standard, passed in standards_check.items():
            status = "✅" if passed else "❌"
            print(f"  {status} {standard}")
        
        return performance_summary
    
    def validate_performance_standards(self) -> Dict[str, bool]:
        """Validate against performance standards."""
        standards = {}
        
        # High-frequency data processing (< 30s for 10k+ points)
        hf_results = [r for r in self.results if r.data_size >= 10000 and 'basic' in r.test_name]
        if hf_results:
            max_hf_time = max(r.execution_time for r in hf_results if r.success)
            standards['high_frequency_processing'] = max_hf_time <= 30.0
        
        # Tool discovery (< 100ms)
        discovery_results = [r for r in self.results if r.test_name == 'tool_discovery']
        if discovery_results:
            discovery_time = discovery_results[0].execution_time
            standards['tool_discovery_speed'] = discovery_time <= 0.1
        
        # Forecast generation (< 60s for typical datasets)
        forecast_results = [r for r in self.results if 'forecast' in r.test_name and r.success]
        if forecast_results:
            max_forecast_time = max(r.execution_time for r in forecast_results)
            standards['forecast_generation_speed'] = max_forecast_time <= 60.0
        
        # Memory usage (< 1GB increase for stress test)
        memory_results = [r for r in self.results if r.test_name == 'memory_stress_test']
        if memory_results:
            memory_increase = memory_results[0].memory_used
            standards['memory_usage_control'] = memory_increase <= 1000.0
        
        # Concurrent operations success
        concurrent_results = [r for r in self.results if r.test_name == 'concurrent_operations']
        if concurrent_results:
            standards['concurrent_operations'] = concurrent_results[0].success
        
        return standards
    
    def cleanup_benchmark_environment(self):
        """Clean up benchmark environment."""
        import shutil
        
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)


def create_standard_benchmark_config() -> BenchmarkConfig:
    """Create standard benchmark configuration."""
    return BenchmarkConfig(
        data_sizes=[100, 1000, 5000, 10000, 25000],
        time_limits={
            'basic_analysis': 30.0,
            'forecasting': 60.0,
            'anomaly_detection': 30.0,
            'tool_discovery': 0.1,
            'memory_management': 1.0
        },
        memory_limits={
            'stress_test': 1000.0,  # 1GB
            'operation': 500.0      # 500MB per operation
        },
        iterations=3,
        warmup_iterations=1
    )


def create_intensive_benchmark_config() -> BenchmarkConfig:
    """Create intensive benchmark configuration for thorough testing."""
    return BenchmarkConfig(
        data_sizes=[100, 1000, 5000, 10000, 25000, 50000],
        time_limits={
            'basic_analysis': 45.0,
            'forecasting': 90.0, 
            'anomaly_detection': 45.0,
            'tool_discovery': 0.1,
            'memory_management': 2.0
        },
        memory_limits={
            'stress_test': 2000.0,  # 2GB
            'operation': 1000.0     # 1GB per operation
        },
        iterations=5,
        warmup_iterations=2
    )


def run_benchmark_suite(config_type: str = 'standard') -> Dict[str, Any]:
    """Run the time series benchmark suite."""
    
    if config_type == 'standard':
        config = create_standard_benchmark_config()
    elif config_type == 'intensive':
        config = create_intensive_benchmark_config()
    else:
        raise ValueError(f"Unknown config type: {config_type}")
    
    benchmark = TimeSeriesPerformanceBenchmark(config)
    return benchmark.run_full_benchmark_suite()


def save_benchmark_results(results: Dict[str, Any], output_path: str):
    """Save benchmark results to file."""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"📄 Benchmark results saved to: {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Time Series Performance Benchmark")
    parser.add_argument('--config', choices=['standard', 'intensive'], default='standard',
                       help='Benchmark configuration type')
    parser.add_argument('--output', type=str, default='benchmark_results.json',
                       help='Output file for results')
    
    args = parser.parse_args()
    
    # Run benchmark
    results = run_benchmark_suite(args.config)
    
    # Save results
    save_benchmark_results(results, args.output)
    
    # Exit with appropriate code
    success_rate = results['success_rate']
    standards_passed = all(results['standards_validation'].values())
    
    if success_rate >= 80.0 and standards_passed:
        print("✅ Benchmark suite PASSED")
        exit(0)
    else:
        print("❌ Benchmark suite FAILED")
        exit(1)