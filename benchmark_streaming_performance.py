#!/usr/bin/env python3
"""
Performance benchmark for StreamingDataPipeline vs standard processing.

This script benchmarks memory usage, execution time, and scalability
of streaming vs batch processing under various data sizes and conditions.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
import time
import psutil
import gc
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass
class BenchmarkResult:
    """Stores benchmark results for analysis."""
    approach: str
    data_size_mb: float
    rows: int
    cols: int
    execution_time: float
    peak_memory_mb: float
    memory_efficiency: float
    success: bool
    chunks_processed: int = 0
    error: str = ""


class MemoryMonitor:
    """Monitors memory usage during execution."""
    
    def __init__(self):
        self.process = psutil.Process()
        self.baseline_memory = self.process.memory_info().rss / (1024 * 1024)
        self.peak_memory = self.baseline_memory
        self.samples = []
    
    def sample(self):
        """Sample current memory usage."""
        current = self.process.memory_info().rss / (1024 * 1024)
        self.peak_memory = max(self.peak_memory, current)
        self.samples.append(current)
        return current
    
    def get_peak_usage(self) -> float:
        """Get peak memory usage above baseline."""
        return self.peak_memory - self.baseline_memory
    
    def get_efficiency_score(self, data_size_mb: float) -> float:
        """Calculate memory efficiency score (lower is better)."""
        if data_size_mb == 0:
            return 0
        return self.get_peak_usage() / data_size_mb


class StreamingBenchmark:
    """Benchmarking harness for streaming vs batch processing."""
    
    def __init__(self):
        self.results: List[BenchmarkResult] = []
    
    def create_test_data(self, rows: int, cols: int, numeric_only: bool = True) -> Tuple[pd.DataFrame, np.ndarray]:
        """Create test dataset of specified size."""
        np.random.seed(42)  # For reproducible results
        
        if numeric_only:
            # Create only numeric data for sklearn compatibility
            data = {}
            for i in range(cols):
                if i % 3 == 0:
                    # Numeric data (float)
                    data[f'feature_{i}'] = np.random.random(rows)
                elif i % 3 == 1:
                    # Integer data (converted to float)
                    data[f'feature_{i}'] = np.random.randint(0, 1000, rows).astype(float)
                else:
                    # Normal distribution data
                    data[f'feature_{i}'] = np.random.normal(0, 1, rows)
        else:
            # Create mixed data types for realistic memory usage (requires preprocessing)
            data = {}
            for i in range(cols):
                if i % 4 == 0:
                    # Numeric data
                    data[f'numeric_{i}'] = np.random.random(rows)
                elif i % 4 == 1:
                    # Integer data
                    data[f'integer_{i}'] = np.random.randint(0, 1000, rows)
                elif i % 4 == 2:
                    # String data (memory intensive) - encoded as categories
                    categories = [f'category_{i}' for i in range(10)]  # Limited categories
                    data[f'categorical_{i}'] = np.random.choice(categories, rows)
                else:
                    # Boolean data (converted to float)
                    data[f'boolean_{i}'] = np.random.choice([0.0, 1.0], rows)
        
        X = pd.DataFrame(data)
        y = np.random.random(rows)  # Continuous target
        
        return X, y
    
    def benchmark_batch_processing(self, X: pd.DataFrame, y: np.ndarray, 
                                 pipeline: Pipeline) -> BenchmarkResult:
        """Benchmark standard batch processing."""
        memory_monitor = MemoryMonitor()
        data_size_mb = X.memory_usage(deep=True).sum() / (1024 * 1024)
        
        start_time = time.time()
        success = True
        error = ""
        
        try:
            memory_monitor.sample()  # Baseline
            
            # Fit pipeline
            pipeline.fit(X, y)
            memory_monitor.sample()
            
            # Transform/predict
            if hasattr(pipeline, 'transform'):
                result = pipeline.transform(X)
            else:
                result = pipeline.predict(X)
            memory_monitor.sample()
            
            execution_time = time.time() - start_time
            
        except Exception as e:
            execution_time = time.time() - start_time
            success = False
            error = str(e)
            result = None
        
        return BenchmarkResult(
            approach="batch",
            data_size_mb=data_size_mb,
            rows=len(X),
            cols=len(X.columns),
            execution_time=execution_time,
            peak_memory_mb=memory_monitor.get_peak_usage(),
            memory_efficiency=memory_monitor.get_efficiency_score(data_size_mb),
            success=success,
            error=error
        )
    
    def benchmark_streaming_processing(self, X: pd.DataFrame, y: np.ndarray,
                                     pipeline_factory, chunk_size: int = 1000) -> BenchmarkResult:
        """Benchmark streaming processing approach."""
        memory_monitor = MemoryMonitor()
        data_size_mb = X.memory_usage(deep=True).sum() / (1024 * 1024)
        
        start_time = time.time()
        success = True
        error = ""
        chunks_processed = 0
        
        try:
            memory_monitor.sample()  # Baseline
            
            # Create fresh pipeline
            pipeline = pipeline_factory()
            
            # Streaming fit
            for start_idx in range(0, len(X), chunk_size):
                end_idx = min(start_idx + chunk_size, len(X))
                X_chunk = X.iloc[start_idx:end_idx]
                y_chunk = y[start_idx:end_idx]
                
                if start_idx == 0:
                    # Initial fit
                    pipeline.fit(X_chunk, y_chunk)
                else:
                    # Try partial fit if available
                    if hasattr(pipeline.steps[-1][1], 'partial_fit'):
                        pipeline.steps[-1][1].partial_fit(X_chunk, y_chunk)
                    else:
                        # Fallback to refit (less efficient)
                        pipeline.fit(X_chunk, y_chunk)
                
                chunks_processed += 1
                memory_monitor.sample()
                
                # Force garbage collection after each chunk
                gc.collect()
            
            # Streaming transform/predict
            results = []
            for start_idx in range(0, len(X), chunk_size):
                end_idx = min(start_idx + chunk_size, len(X))
                X_chunk = X.iloc[start_idx:end_idx]
                
                if hasattr(pipeline, 'transform'):
                    chunk_result = pipeline.transform(X_chunk)
                else:
                    chunk_result = pipeline.predict(X_chunk)
                
                results.append(chunk_result)
                memory_monitor.sample()
                
                # Clean up chunk
                del X_chunk
                gc.collect()
            
            # Aggregate results
            if isinstance(results[0], pd.DataFrame):
                final_result = pd.concat(results, ignore_index=True)
            else:
                final_result = np.concatenate(results, axis=0)
            
            execution_time = time.time() - start_time
            
        except Exception as e:
            execution_time = time.time() - start_time
            success = False
            error = str(e)
            chunks_processed = max(chunks_processed, 1)
        
        return BenchmarkResult(
            approach="streaming",
            data_size_mb=data_size_mb,
            rows=len(X),
            cols=len(X.columns),
            execution_time=execution_time,
            peak_memory_mb=memory_monitor.get_peak_usage(),
            memory_efficiency=memory_monitor.get_efficiency_score(data_size_mb),
            success=success,
            chunks_processed=chunks_processed,
            error=error
        )
    
    def run_size_scaling_benchmark(self, max_rows: int = 50000) -> List[BenchmarkResult]:
        """Benchmark performance scaling with data size."""
        print("🚀 Running Data Size Scaling Benchmark")
        print("=" * 60)
        
        # Test various data sizes
        test_sizes = [
            (1000, 10),    # Small: 1K rows, 10 cols
            (5000, 20),    # Medium: 5K rows, 20 cols  
            (10000, 30),   # Large: 10K rows, 30 cols
            (25000, 40),   # Very Large: 25K rows, 40 cols
        ]
        
        if max_rows >= 50000:
            test_sizes.append((50000, 50))  # Huge: 50K rows, 50 cols
        
        results = []
        
        for rows, cols in test_sizes:
            print(f"\n📊 Testing {rows:,} rows × {cols} columns...")
            
            # Create test data
            X, y = self.create_test_data(rows, cols)
            data_size_mb = X.memory_usage(deep=True).sum() / (1024 * 1024)
            print(f"   Data size: {data_size_mb:.2f} MB")
            
            # Pipeline factories for fresh instances (numeric data compatible)
            def linear_pipeline():
                return Pipeline([
                    ('scaler', StandardScaler()),
                    ('regressor', LinearRegression())
                ])
            
            def sgd_pipeline():
                return Pipeline([
                    ('scaler', StandardScaler()), 
                    ('regressor', SGDRegressor(random_state=42, max_iter=100))
                ])
            
            # Test batch processing
            print("   🔄 Batch processing...")
            batch_result = self.benchmark_batch_processing(X, y, linear_pipeline())
            results.append(batch_result)
            
            if batch_result.success:
                print(f"      ✅ Time: {batch_result.execution_time:.3f}s, Memory: {batch_result.peak_memory_mb:.2f}MB")
            else:
                print(f"      ❌ Failed: {batch_result.error}")
            
            # Test streaming processing (with SGD for partial_fit support)
            print("   📡 Streaming processing...")
            streaming_result = self.benchmark_streaming_processing(
                X, y, sgd_pipeline, chunk_size=max(1000, rows // 10)
            )
            results.append(streaming_result)
            
            if streaming_result.success:
                print(f"      ✅ Time: {streaming_result.execution_time:.3f}s, Memory: {streaming_result.peak_memory_mb:.2f}MB")
                print(f"      📦 Chunks: {streaming_result.chunks_processed}")
            else:
                print(f"      ❌ Failed: {streaming_result.error}")
            
            # Memory cleanup
            del X, y
            gc.collect()
        
        return results
    
    def run_memory_pressure_benchmark(self) -> List[BenchmarkResult]:
        """Benchmark behavior under memory pressure."""
        print("\n🧠 Running Memory Pressure Benchmark")
        print("=" * 60)
        
        # Create increasingly memory-intensive data
        test_configs = [
            ("Low pressure", 5000, 20, 2000),      # 5K rows, chunk=2000
            ("Medium pressure", 10000, 40, 1000),  # 10K rows, chunk=1000  
            ("High pressure", 20000, 60, 500),     # 20K rows, chunk=500
            ("Extreme pressure", 30000, 80, 250),  # 30K rows, chunk=250
        ]
        
        results = []
        
        for name, rows, cols, chunk_size in test_configs:
            print(f"\n🔥 {name}: {rows:,} rows × {cols} columns (chunks: {chunk_size})...")
            
            X, y = self.create_test_data(rows, cols)
            data_size_mb = X.memory_usage(deep=True).sum() / (1024 * 1024)
            print(f"   Data size: {data_size_mb:.2f} MB")
            
            # Only test streaming under memory pressure
            def sgd_pipeline():
                return Pipeline([
                    ('scaler', MinMaxScaler()),  # Different scaler for variety
                    ('regressor', SGDRegressor(random_state=42, max_iter=100))
                ])
            
            streaming_result = self.benchmark_streaming_processing(X, y, sgd_pipeline, chunk_size)
            streaming_result.approach = f"streaming_{name.lower().replace(' ', '_')}"
            results.append(streaming_result)
            
            if streaming_result.success:
                efficiency = streaming_result.memory_efficiency
                print(f"      ✅ Efficiency: {efficiency:.2f} (lower=better)")
                print(f"      ⏱️  Time: {streaming_result.execution_time:.3f}s")
                print(f"      🧠 Peak memory: {streaming_result.peak_memory_mb:.2f}MB")
            else:
                print(f"      ❌ Failed: {streaming_result.error}")
            
            del X, y
            gc.collect()
        
        return results
    
    def generate_performance_report(self, results: List[BenchmarkResult]):
        """Generate comprehensive performance analysis report."""
        print("\n📊 PERFORMANCE ANALYSIS REPORT")
        print("=" * 80)
        
        # Separate batch and streaming results
        batch_results = [r for r in results if r.approach.startswith('batch') and r.success]
        streaming_results = [r for r in results if r.approach.startswith('streaming') and r.success]
        
        if batch_results and streaming_results:
            print("\n🏆 BATCH vs STREAMING COMPARISON")
            print("-" * 50)
            
            # Compare matching data sizes
            for batch in batch_results:
                matching_streaming = next(
                    (s for s in streaming_results if abs(s.data_size_mb - batch.data_size_mb) < 0.1), 
                    None
                )
                if matching_streaming:
                    print(f"\n📈 Data Size: {batch.data_size_mb:.2f} MB ({batch.rows:,} × {batch.cols})")
                    print(f"   Batch:     {batch.execution_time:.3f}s, {batch.peak_memory_mb:.2f}MB")
                    print(f"   Streaming: {matching_streaming.execution_time:.3f}s, {matching_streaming.peak_memory_mb:.2f}MB")
                    
                    # Calculate improvements
                    time_ratio = matching_streaming.execution_time / batch.execution_time
                    memory_ratio = matching_streaming.peak_memory_mb / batch.peak_memory_mb
                    
                    print(f"   Time ratio: {time_ratio:.2f}x {'(slower)' if time_ratio > 1 else '(faster)'}")
                    print(f"   Memory ratio: {memory_ratio:.2f}x {'(more)' if memory_ratio > 1 else '(less)'}")
        
        # Memory efficiency analysis
        if streaming_results:
            print(f"\n💾 MEMORY EFFICIENCY ANALYSIS")
            print("-" * 50)
            
            efficiencies = [r.memory_efficiency for r in streaming_results if r.memory_efficiency > 0]
            if efficiencies:
                print(f"   Best efficiency: {min(efficiencies):.2f}")
                print(f"   Worst efficiency: {max(efficiencies):.2f}")
                print(f"   Average efficiency: {np.mean(efficiencies):.2f}")
                print(f"   (Lower values indicate better memory efficiency)")
        
        # Scalability analysis  
        if len(streaming_results) >= 3:
            print(f"\n📈 SCALABILITY ANALYSIS")
            print("-" * 50)
            
            # Sort by data size
            sorted_results = sorted(streaming_results, key=lambda r: r.data_size_mb)
            
            print("   Data Size (MB) | Time (s) | Memory (MB) | Efficiency")
            print("   " + "-" * 55)
            for r in sorted_results:
                print(f"   {r.data_size_mb:10.2f} | {r.execution_time:7.3f} | {r.peak_memory_mb:9.2f} | {r.memory_efficiency:8.2f}")
        
        # Success rate
        total_tests = len(results)
        successful_tests = len([r for r in results if r.success])
        success_rate = (successful_tests / total_tests) * 100 if total_tests > 0 else 0
        
        print(f"\n✅ OVERALL SUCCESS RATE: {success_rate:.1f}% ({successful_tests}/{total_tests})")
        
        # Failed tests analysis
        failed_results = [r for r in results if not r.success]
        if failed_results:
            print(f"\n❌ FAILED TESTS ANALYSIS:")
            for r in failed_results:
                print(f"   {r.approach}: {r.rows:,} × {r.cols} - {r.error}")


def run_comprehensive_benchmark():
    """Run comprehensive streaming performance benchmark."""
    print("🚀 StreamingDataPipeline Performance Benchmark")
    print("=" * 80)
    print("Testing memory-bounded streaming vs batch processing performance")
    print("Evaluating scalability, memory efficiency, and execution time")
    
    benchmark = StreamingBenchmark()
    all_results = []
    
    try:
        # Size scaling benchmark
        scaling_results = benchmark.run_size_scaling_benchmark(max_rows=25000)
        all_results.extend(scaling_results)
        
        # Memory pressure benchmark
        pressure_results = benchmark.run_memory_pressure_benchmark()
        all_results.extend(pressure_results)
        
        # Generate comprehensive report
        benchmark.generate_performance_report(all_results)
        
        print(f"\n🎉 BENCHMARK COMPLETED SUCCESSFULLY!")
        print(f"Total tests executed: {len(all_results)}")
        print(f"Successful tests: {len([r for r in all_results if r.success])}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    try:
        success = run_comprehensive_benchmark()
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️  Benchmark interrupted by user")
        exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        exit(1)