"""
Performance Optimization System for LocalData MCP v2.0 Integration Shims - Complete Implementation.

This is the complete implementation including all remaining performance optimization components:
- Performance Benchmarking System
- Automatic Optimization Selection
- Data Characteristics Analysis
- Factory Functions and Integration Points
"""

import time
import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Callable
from abc import ABC, abstractmethod
from enum import Enum

from .interfaces import ShimAdapter, ConversionRequest, DataFormat
from ...logging_manager import get_logger

logger = get_logger(__name__)


@dataclass
class BenchmarkResult:
    """Result of performance benchmark testing."""
    operation_name: str
    adapter_id: str
    execution_time_seconds: float
    memory_used_mb: float
    throughput_ops_per_second: float
    success_rate: float
    error_count: int
    test_data_size_mb: float
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryProfile:
    """Memory usage profile for an operation."""
    operation_name: str
    peak_memory_mb: float
    average_memory_mb: float
    memory_growth_rate: float
    gc_triggered: int
    memory_efficiency_score: float
    timeline: List[Tuple[float, float]] = field(default_factory=list)  # (timestamp, memory_mb)


@dataclass
class ThroughputMetrics:
    """Throughput measurements for streaming operations."""
    converter_id: str
    rows_per_second: float
    mb_per_second: float
    chunks_per_second: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    error_rate: float
    measurement_duration_seconds: float


@dataclass
class ComparisonResult:
    """Result of comparing optimization strategies."""
    strategy_comparisons: Dict[str, BenchmarkResult]
    winner: str
    performance_improvement: float
    recommendation: str
    confidence_score: float


class PerformanceBenchmark:
    """
    Comprehensive performance measurement and profiling system.
    """
    
    def __init__(self, 
                 warmup_iterations: int = 3,
                 test_iterations: int = 10,
                 memory_sampling_interval: float = 0.1):
        """
        Initialize performance benchmark.
        
        Args:
            warmup_iterations: Number of warmup iterations before measurement
            test_iterations: Number of test iterations for measurement
            memory_sampling_interval: Interval for memory sampling in seconds
        """
        self.warmup_iterations = warmup_iterations
        self.test_iterations = test_iterations
        self.memory_sampling_interval = memory_sampling_interval
        
        # Results storage
        self._benchmark_results: Dict[str, List[BenchmarkResult]] = {}
        self._memory_profiles: Dict[str, MemoryProfile] = {}
        
        logger.info("PerformanceBenchmark initialized",
                   warmup_iterations=warmup_iterations,
                   test_iterations=test_iterations)
    
    def benchmark_conversion(self, 
                           converter: ShimAdapter, 
                           test_data: Any,
                           test_name: Optional[str] = None) -> BenchmarkResult:
        """
        Benchmark conversion performance with multiple iterations.
        
        Args:
            converter: Adapter to benchmark
            test_data: Test data for benchmarking
            test_name: Optional name for the test
            
        Returns:
            Benchmark result with performance metrics
        """
        test_name = test_name or f"{converter.adapter_id}_benchmark"
        
        logger.debug(f"Starting benchmark: {test_name}")
        
        # Estimate test data size
        test_data_size = self._estimate_data_size(test_data)
        
        # Create conversion request
        request = ConversionRequest(
            source_data=test_data,
            source_format=DataFormat.AUTO_DETECT,  # Will be detected by adapter
            target_format=DataFormat.AUTO_DETECT,
            request_id=f"benchmark_{test_name}"
        )
        
        # Warmup iterations
        logger.debug(f"Performing {self.warmup_iterations} warmup iterations")
        for i in range(self.warmup_iterations):
            try:
                converter.convert(request)
            except Exception as e:
                logger.warning(f"Warmup iteration {i+1} failed: {e}")
        
        # Benchmark iterations
        execution_times = []
        memory_usages = []
        success_count = 0
        error_count = 0
        
        logger.debug(f"Performing {self.test_iterations} benchmark iterations")
        
        for i in range(self.test_iterations):
            try:
                # Start memory profiling
                initial_memory = self._get_memory_usage()
                
                # Time the conversion
                start_time = time.time()
                result = converter.convert(request)
                execution_time = time.time() - start_time
                
                # Measure memory usage
                final_memory = self._get_memory_usage()
                memory_used = final_memory - initial_memory
                
                execution_times.append(execution_time)
                memory_usages.append(memory_used)
                
                if result.success:
                    success_count += 1
                else:
                    error_count += 1
                    
            except Exception as e:
                logger.warning(f"Benchmark iteration {i+1} failed: {e}")
                error_count += 1
        
        # Calculate metrics
        if execution_times:
            avg_execution_time = statistics.mean(execution_times)
            throughput = 1.0 / avg_execution_time if avg_execution_time > 0 else 0.0
        else:
            avg_execution_time = float('inf')
            throughput = 0.0
        
        avg_memory_used = statistics.mean(memory_usages) if memory_usages else 0.0
        success_rate = success_count / self.test_iterations
        
        benchmark_result = BenchmarkResult(
            operation_name=test_name,
            adapter_id=converter.adapter_id,
            execution_time_seconds=avg_execution_time,
            memory_used_mb=avg_memory_used,
            throughput_ops_per_second=throughput,
            success_rate=success_rate,
            error_count=error_count,
            test_data_size_mb=test_data_size,
            metadata={
                'execution_times': execution_times,
                'memory_usages': memory_usages,
                'warmup_iterations': self.warmup_iterations,
                'test_iterations': self.test_iterations
            }
        )
        
        # Store result
        if test_name not in self._benchmark_results:
            self._benchmark_results[test_name] = []
        self._benchmark_results[test_name].append(benchmark_result)
        
        logger.info(f"Benchmark completed: {test_name}",
                   execution_time=avg_execution_time,
                   throughput=throughput,
                   success_rate=success_rate)
        
        return benchmark_result
    
    def profile_memory_usage(self, operation: Callable[[], Any]) -> MemoryProfile:
        """
        Profile memory usage of an operation with detailed timeline.
        
        Args:
            operation: Operation to profile
            
        Returns:
            Memory profile with usage timeline
        """
        operation_name = getattr(operation, '__name__', 'anonymous_operation')
        
        logger.debug(f"Starting memory profiling: {operation_name}")
        
        # Memory timeline tracking
        memory_timeline = []
        peak_memory = 0.0
        initial_memory = self._get_memory_usage()
        gc_count_before = self._get_gc_count()
        
        start_time = time.time()
        
        try:
            # Sample memory during operation
            def memory_monitor():
                while True:
                    current_memory = self._get_memory_usage()
                    timestamp = time.time() - start_time
                    memory_timeline.append((timestamp, current_memory))
                    nonlocal peak_memory
                    peak_memory = max(peak_memory, current_memory)
                    time.sleep(self.memory_sampling_interval)
            
            import threading
            monitor_thread = threading.Thread(target=memory_monitor, daemon=True)
            monitor_thread.start()
            
            # Execute operation
            result = operation()
            
            execution_time = time.time() - start_time
            
            # Stop monitoring
            # (daemon thread will stop when main thread exits)
            
        except Exception as e:
            logger.error(f"Memory profiling failed: {e}")
            raise
        
        final_memory = self._get_memory_usage()
        gc_count_after = self._get_gc_count()
        
        # Calculate metrics
        if memory_timeline:
            memory_values = [mem for _, mem in memory_timeline]
            average_memory = statistics.mean(memory_values)
            memory_growth = (final_memory - initial_memory) / max(initial_memory, 1.0)
        else:
            average_memory = (initial_memory + final_memory) / 2
            memory_growth = 0.0
        
        # Memory efficiency score (inverse of memory growth relative to operation time)
        efficiency_score = 1.0 / (1.0 + abs(memory_growth) + (execution_time / 10.0))
        
        memory_profile = MemoryProfile(
            operation_name=operation_name,
            peak_memory_mb=peak_memory,
            average_memory_mb=average_memory,
            memory_growth_rate=memory_growth,
            gc_triggered=gc_count_after - gc_count_before,
            memory_efficiency_score=efficiency_score,
            timeline=memory_timeline
        )
        
        self._memory_profiles[operation_name] = memory_profile
        
        logger.info(f"Memory profiling completed: {operation_name}",
                   peak_memory=peak_memory,
                   average_memory=average_memory,
                   efficiency_score=efficiency_score)
        
        return memory_profile
    
    def measure_throughput(self, streaming_converter) -> ThroughputMetrics:
        """
        Measure throughput for streaming converter.
        
        Args:
            streaming_converter: Streaming converter to measure
            
        Returns:
            Throughput metrics
        """
        converter_id = getattr(streaming_converter, 'conversion_id', 'unknown_converter')
        
        logger.debug(f"Measuring throughput for {converter_id}")
        
        # Get progress information
        progress = streaming_converter.get_progress()
        
        if not progress:
            logger.warning("No progress information available for throughput measurement")
            return ThroughputMetrics(
                converter_id=converter_id,
                rows_per_second=0.0,
                mb_per_second=0.0,
                chunks_per_second=0.0,
                latency_p50_ms=0.0,
                latency_p95_ms=0.0,
                latency_p99_ms=0.0,
                error_rate=0.0,
                measurement_duration_seconds=0.0
            )
        
        # Extract performance data from converter
        chunk_times = getattr(streaming_converter, '_chunk_times', [])\n        memory_history = getattr(streaming_converter, '_memory_usage_history', [])
        
        # Calculate throughput metrics
        elapsed_time = time.time() - getattr(streaming_converter, '_start_time', time.time())
        
        rows_per_second = progress.processing_rate_rows_per_sec
        
        # Estimate MB/s from memory usage pattern
        if memory_history and elapsed_time > 0:
            total_memory_processed = sum(memory_history)
            mb_per_second = total_memory_processed / elapsed_time
        else:
            mb_per_second = 0.0
        
        chunks_per_second = progress.chunks_processed / max(elapsed_time, 1)
        
        # Calculate latency percentiles from chunk times
        if chunk_times:
            chunk_times_ms = [t * 1000 for t in chunk_times]  # Convert to milliseconds
            chunk_times_ms.sort()
            
            p50_idx = int(len(chunk_times_ms) * 0.5)
            p95_idx = int(len(chunk_times_ms) * 0.95)
            p99_idx = int(len(chunk_times_ms) * 0.99)
            
            latency_p50 = chunk_times_ms[p50_idx] if p50_idx < len(chunk_times_ms) else 0.0
            latency_p95 = chunk_times_ms[p95_idx] if p95_idx < len(chunk_times_ms) else 0.0
            latency_p99 = chunk_times_ms[p99_idx] if p99_idx < len(chunk_times_ms) else 0.0
        else:
            latency_p50 = latency_p95 = latency_p99 = 0.0
        
        # Calculate error rate
        total_chunks = progress.chunks_processed
        failed_chunks = getattr(progress, 'failed_chunks', 0)
        error_rate = failed_chunks / max(total_chunks, 1)
        
        throughput_metrics = ThroughputMetrics(
            converter_id=converter_id,
            rows_per_second=rows_per_second,
            mb_per_second=mb_per_second,
            chunks_per_second=chunks_per_second,
            latency_p50_ms=latency_p50,
            latency_p95_ms=latency_p95,
            latency_p99_ms=latency_p99,
            error_rate=error_rate,
            measurement_duration_seconds=elapsed_time
        )
        
        logger.info(f"Throughput measurement completed: {converter_id}",
                   rows_per_sec=rows_per_second,
                   mb_per_sec=mb_per_second,
                   chunks_per_sec=chunks_per_second)
        
        return throughput_metrics
    
    def compare_optimization_strategies(self, 
                                      strategies: List[str],
                                      test_operation: Callable[[str], BenchmarkResult]) -> ComparisonResult:
        """
        Compare multiple optimization strategies.
        
        Args:
            strategies: List of strategy names to compare
            test_operation: Function that takes strategy name and returns benchmark result
            
        Returns:
            Comparison result with winner and performance improvement
        """
        logger.info(f"Comparing {len(strategies)} optimization strategies: {strategies}")
        
        strategy_results = {}
        
        for strategy in strategies:
            try:
                logger.debug(f"Testing strategy: {strategy}")
                result = test_operation(strategy)
                strategy_results[strategy] = result
                
                logger.debug(f"Strategy {strategy} completed",
                            execution_time=result.execution_time_seconds,
                            throughput=result.throughput_ops_per_second)
                
            except Exception as e:
                logger.error(f"Strategy {strategy} failed: {e}")
                # Create a failed result
                strategy_results[strategy] = BenchmarkResult(
                    operation_name=f"{strategy}_failed",
                    adapter_id="unknown",
                    execution_time_seconds=float('inf'),
                    memory_used_mb=float('inf'),
                    throughput_ops_per_second=0.0,
                    success_rate=0.0,
                    error_count=1,
                    test_data_size_mb=0.0
                )
        
        # Find the best strategy
        if not strategy_results:
            raise ValueError("No strategies produced valid results")\n        \n        # Rank by composite score (weighted combination of throughput and success rate)
        def calculate_score(result: BenchmarkResult) -> float:
            if result.success_rate == 0:
                return 0.0
            
            # Normalize throughput (higher is better)
            throughput_score = result.throughput_ops_per_second
            
            # Penalize high memory usage
            memory_penalty = 1.0 / (1.0 + result.memory_used_mb / 100.0)
            
            # Combine metrics
            return throughput_score * result.success_rate * memory_penalty
        
        ranked_strategies = sorted(
            strategy_results.items(),
            key=lambda item: calculate_score(item[1]),
            reverse=True
        )
        
        winner = ranked_strategies[0][0]
        winner_result = ranked_strategies[0][1]
        
        # Calculate improvement over baseline (first strategy or worst performing)
        if len(ranked_strategies) > 1:
            baseline_result = ranked_strategies[-1][1]  # Worst performing
            if baseline_result.throughput_ops_per_second > 0:
                improvement = ((winner_result.throughput_ops_per_second - baseline_result.throughput_ops_per_second) 
                             / baseline_result.throughput_ops_per_second * 100)
            else:
                improvement = float('inf')
        else:
            improvement = 0.0
        
        # Generate recommendation
        recommendation = self._generate_strategy_recommendation(winner, winner_result, improvement)
        
        # Calculate confidence based on performance gap
        if len(ranked_strategies) > 1:
            winner_score = calculate_score(winner_result)
            second_best_score = calculate_score(ranked_strategies[1][1])
            confidence = min(1.0, (winner_score - second_best_score) / max(winner_score, 0.01))
        else:
            confidence = 1.0
        
        comparison_result = ComparisonResult(
            strategy_comparisons=strategy_results,
            winner=winner,
            performance_improvement=improvement,
            recommendation=recommendation,
            confidence_score=confidence
        )
        
        logger.info(f"Strategy comparison completed",
                   winner=winner,
                   improvement=f"{improvement:.1f}%",
                   confidence=f"{confidence:.2f}")
        
        return comparison_result
    
    def _estimate_data_size(self, data: Any) -> float:
        """Estimate data size in MB."""
        try:
            import sys
            if hasattr(data, 'memory_usage'):
                return data.memory_usage(deep=True).sum() / (1024 * 1024)
            elif hasattr(data, 'nbytes'):
                return data.nbytes / (1024 * 1024)
            else:
                return sys.getsizeof(data) / (1024 * 1024)
        except Exception:
            return 1.0  # Default estimate
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except ImportError:
            return 0.0
    
    def _get_gc_count(self) -> int:
        """Get garbage collection count."""
        try:
            import gc
            return sum(gc.get_count())
        except Exception:
            return 0
    
    def _generate_strategy_recommendation(self, 
                                        winner: str, 
                                        result: BenchmarkResult, 
                                        improvement: float) -> str:
        """Generate recommendation text for strategy selection."""
        if improvement > 50:
            performance_desc = "significant"
        elif improvement > 20:
            performance_desc = "moderate"
        elif improvement > 5:
            performance_desc = "small"
        else:
            performance_desc = "minimal"
        
        return (f"Recommended strategy: {winner}. "
                f"Shows {performance_desc} performance improvement ({improvement:.1f}%) "
                f"with {result.success_rate:.1%} success rate and "
                f"{result.throughput_ops_per_second:.1f} ops/sec throughput.")


@dataclass
class DataCharacteristics:
    """Analysis of data characteristics for optimization selection."""
    data_size_mb: float
    row_count: int
    column_count: int
    numeric_columns: int
    text_columns: int
    null_percentage: float
    data_density: float  # Ratio of non-null to total values
    memory_intensity: float  # Memory per row in KB
    complexity_score: float  # Overall data complexity (0-1)
    dominant_types: List[str]
    sparsity: float  # Percentage of zero/null values
    estimated_processing_time: float  # Rough estimate in seconds


@dataclass
class OptimizationStrategy:
    """Definition of an optimization strategy with applicability conditions."""
    name: str
    description: str
    applicability_conditions: Dict[str, Any]
    expected_benefits: List[str]
    overhead_cost: float  # Relative overhead (0-1)
    memory_efficiency: float  # Memory efficiency score (0-1)
    processing_speed: float  # Processing speed score (0-1)
    complexity: float  # Implementation complexity (0-1)


@dataclass
class OptimizedConversionPath:
    """Optimized conversion path with strategy selection."""
    source_format: DataFormat
    target_format: DataFormat
    selected_strategy: OptimizationStrategy
    estimated_performance: Dict[str, float]
    fallback_strategies: List[OptimizationStrategy]
    optimization_metadata: Dict[str, Any]


class OptimizationSelector:
    """
    Intelligent optimization strategy selector based on data characteristics and performance feedback.
    """
    
    def __init__(self):
        """Initialize optimization selector with predefined strategies."""
        self.strategies = self._initialize_optimization_strategies()
        self._performance_history: Dict[str, List[float]] = {}
        self._strategy_effectiveness: Dict[str, float] = {}
        
        logger.info("OptimizationSelector initialized", 
                   strategies=len(self.strategies))
    
    def analyze_data_characteristics(self, data: Any) -> DataCharacteristics:
        """
        Analyze data to determine optimization strategy.
        
        Args:
            data: Data to analyze
            
        Returns:
            Data characteristics analysis
        """
        logger.debug("Analyzing data characteristics for optimization selection")
        
        try:
            # Basic size and shape analysis
            if hasattr(data, 'shape'):
                row_count, column_count = data.shape if len(data.shape) > 1 else (data.shape[0], 1)
            elif hasattr(data, '__len__'):
                row_count = len(data)
                column_count = 1
            else:
                row_count = column_count = 1
            
            # Memory analysis
            data_size_mb = self._estimate_data_size(data)
            memory_per_row = (data_size_mb * 1024) / max(row_count, 1)  # KB per row
            
            # Type analysis
            numeric_cols = text_cols = 0
            null_percentage = 0.0
            dominant_types = []
            
            if hasattr(data, 'dtypes'):  # pandas DataFrame
                type_counts = data.dtypes.value_counts()
                numeric_cols = sum(1 for dtype in data.dtypes if dtype.kind in 'biufc')
                text_cols = sum(1 for dtype in data.dtypes if dtype.kind in 'OSU')
                null_percentage = data.isnull().sum().sum() / (row_count * column_count) * 100
                dominant_types = [str(dtype) for dtype in type_counts.head(3).index]
                
                # Calculate sparsity for numeric columns
                numeric_data = data.select_dtypes(include=['number'])
                if not numeric_data.empty:
                    zero_count = (numeric_data == 0).sum().sum()
                    null_count = numeric_data.isnull().sum().sum()
                    sparsity = (zero_count + null_count) / (numeric_data.shape[0] * numeric_data.shape[1]) * 100
                else:
                    sparsity = null_percentage
            else:
                # Fallback analysis for other data types
                sparsity = 0.0
                if hasattr(data, 'dtype'):  # numpy array
                    if data.dtype.kind in 'biufc':
                        numeric_cols = 1
                    else:
                        text_cols = 1
                    dominant_types = [str(data.dtype)]
                else:
                    # Generic analysis
                    text_cols = 1
                    dominant_types = [type(data).__name__]
            
            # Calculate complexity metrics
            data_density = 1.0 - (null_percentage / 100.0)
            
            # Complexity score based on multiple factors
            size_complexity = min(1.0, data_size_mb / 1000.0)  # Normalize to 1GB
            type_complexity = min(1.0, (text_cols / max(column_count, 1)) * 2)  # Text is more complex
            sparsity_complexity = sparsity / 100.0  # Higher sparsity = higher complexity
            
            complexity_score = (size_complexity + type_complexity + sparsity_complexity) / 3.0
            
            # Estimate processing time (rough heuristic)
            base_time = data_size_mb * 0.01  # 10ms per MB baseline
            complexity_multiplier = 1.0 + complexity_score
            estimated_processing_time = base_time * complexity_multiplier
            
            characteristics = DataCharacteristics(
                data_size_mb=data_size_mb,
                row_count=row_count,
                column_count=column_count,
                numeric_columns=numeric_cols,
                text_columns=text_cols,
                null_percentage=null_percentage,
                data_density=data_density,
                memory_intensity=memory_per_row,
                complexity_score=complexity_score,
                dominant_types=dominant_types,
                sparsity=sparsity,
                estimated_processing_time=estimated_processing_time
            )
            
            logger.debug("Data characteristics analysis completed",
                        size_mb=data_size_mb,
                        rows=row_count,
                        complexity=complexity_score)
            
            return characteristics
            
        except Exception as e:
            logger.error(f"Data characteristics analysis failed: {e}")
            # Return default characteristics
            return DataCharacteristics(
                data_size_mb=1.0,
                row_count=1000,
                column_count=10,
                numeric_columns=5,
                text_columns=5,
                null_percentage=0.0,
                data_density=1.0,
                memory_intensity=1.0,
                complexity_score=0.5,
                dominant_types=['unknown'],
                sparsity=0.0,
                estimated_processing_time=1.0
            )
    
    def select_optimization_strategy(self, characteristics: DataCharacteristics) -> OptimizationStrategy:
        """
        Select best optimization strategy based on data characteristics.
        
        Args:
            characteristics: Data characteristics from analysis
            
        Returns:
            Selected optimization strategy
        """
        logger.debug("Selecting optimization strategy", 
                    data_size_mb=characteristics.data_size_mb,
                    complexity=characteristics.complexity_score)
        
        strategy_scores = {}
        
        for strategy in self.strategies:
            score = self._calculate_strategy_score(strategy, characteristics)
            strategy_scores[strategy.name] = (strategy, score)
        
        # Sort by score (higher is better)
        ranked_strategies = sorted(strategy_scores.items(), key=lambda x: x[1][1], reverse=True)
        
        selected_strategy = ranked_strategies[0][1][0]
        
        logger.info(f"Selected optimization strategy: {selected_strategy.name}",
                   score=ranked_strategies[0][1][1],
                   data_size_mb=characteristics.data_size_mb)
        
        return selected_strategy
    
    def recommend_conversion_path(self, 
                                request: ConversionRequest,
                                characteristics: Optional[DataCharacteristics] = None) -> OptimizedConversionPath:
        """
        Recommend optimized conversion path for request.
        
        Args:
            request: Conversion request
            characteristics: Optional pre-analyzed characteristics
            
        Returns:
            Optimized conversion path with strategy and fallbacks
        """
        if characteristics is None:
            characteristics = self.analyze_data_characteristics(request.source_data)
        
        # Select primary strategy
        primary_strategy = self.select_optimization_strategy(characteristics)
        
        # Select fallback strategies
        fallback_strategies = []
        for strategy in self.strategies:
            if (strategy.name != primary_strategy.name and
                self._calculate_strategy_score(strategy, characteristics) > 0.3):
                fallback_strategies.append(strategy)
        
        # Sort fallbacks by score
        fallback_strategies.sort(
            key=lambda s: self._calculate_strategy_score(s, characteristics),
            reverse=True
        )
        fallback_strategies = fallback_strategies[:3]  # Keep top 3 fallbacks
        
        # Estimate performance
        estimated_performance = {
            'execution_time_seconds': characteristics.estimated_processing_time * (1.0 - primary_strategy.processing_speed),
            'memory_usage_mb': characteristics.data_size_mb * (1.0 - primary_strategy.memory_efficiency),
            'success_probability': 0.95 - (primary_strategy.complexity * 0.1),
            'optimization_overhead': primary_strategy.overhead_cost
        }
        
        path = OptimizedConversionPath(
            source_format=request.source_format,
            target_format=request.target_format,
            selected_strategy=primary_strategy,
            estimated_performance=estimated_performance,
            fallback_strategies=fallback_strategies,
            optimization_metadata={
                'data_characteristics': characteristics,
                'selection_reasoning': f"Selected {primary_strategy.name} due to {primary_strategy.expected_benefits[0] if primary_strategy.expected_benefits else 'general suitability'}",
                'confidence_score': self._strategy_effectiveness.get(primary_strategy.name, 0.8)
            }
        )
        
        logger.info(f"Recommended conversion path",
                   primary_strategy=primary_strategy.name,
                   fallbacks=len(fallback_strategies),
                   estimated_time=estimated_performance['execution_time_seconds'])
        
        return path
    
    def monitor_performance_feedback(self, 
                                   strategy_id: str, 
                                   performance: Dict[str, float]) -> None:
        """
        Monitor and record performance feedback for strategy improvement.
        
        Args:
            strategy_id: Strategy identifier
            performance: Performance metrics dictionary
        """
        if strategy_id not in self._performance_history:
            self._performance_history[strategy_id] = []
        
        # Calculate composite performance score
        execution_score = 1.0 / (1.0 + performance.get('execution_time_seconds', 1.0))
        memory_score = 1.0 / (1.0 + performance.get('memory_usage_mb', 100.0) / 100.0)
        success_score = performance.get('success_rate', 0.8)
        
        composite_score = (execution_score + memory_score + success_score) / 3.0
        
        self._performance_history[strategy_id].append(composite_score)
        
        # Keep only recent history
        if len(self._performance_history[strategy_id]) > 100:
            self._performance_history[strategy_id] = self._performance_history[strategy_id][-50:]
        
        # Update effectiveness score
        self._strategy_effectiveness[strategy_id] = statistics.mean(self._performance_history[strategy_id])
        
        logger.debug(f"Updated performance feedback for {strategy_id}",
                    composite_score=composite_score,
                    effectiveness=self._strategy_effectiveness[strategy_id])
    
    def _initialize_optimization_strategies(self) -> List[OptimizationStrategy]:
        """Initialize predefined optimization strategies."""
        strategies = [
            OptimizationStrategy(
                name="cache_first",
                description="Prioritize cache hits for repeated operations",
                applicability_conditions={
                    'min_data_size_mb': 1.0,
                    'max_complexity': 0.7,
                    'repeatable_operations': True
                },
                expected_benefits=["Reduced computation time", "Lower CPU usage"],
                overhead_cost=0.1,
                memory_efficiency=0.7,
                processing_speed=0.9,
                complexity=0.2
            ),
            
            OptimizationStrategy(
                name="lazy_loading",
                description="Defer conversion until data is actually needed",
                applicability_conditions={
                    'min_data_size_mb': 10.0,
                    'memory_constrained': True,
                    'selective_access': True
                },
                expected_benefits=["Reduced memory usage", "Faster startup"],
                overhead_cost=0.2,
                memory_efficiency=0.9,
                processing_speed=0.6,
                complexity=0.4
            ),
            
            OptimizationStrategy(
                name="streaming",
                description="Use chunked processing for large datasets",
                applicability_conditions={
                    'min_data_size_mb': 50.0,
                    'sequential_processing': True
                },
                expected_benefits=["Memory bounded processing", "Scalable to large data"],
                overhead_cost=0.3,
                memory_efficiency=0.95,
                processing_speed=0.7,
                complexity=0.6
            ),
            
            OptimizationStrategy(
                name="memory_pool",
                description="Reuse allocated memory for similar operations",
                applicability_conditions={
                    'frequent_allocations': True,
                    'similar_data_sizes': True
                },
                expected_benefits=["Reduced GC pressure", "Consistent performance"],
                overhead_cost=0.15,
                memory_efficiency=0.8,
                processing_speed=0.8,
                complexity=0.3
            ),
            
            OptimizationStrategy(
                name="parallel",
                description="Use multiple threads/processes for independent conversions",
                applicability_conditions={
                    'parallelizable': True,
                    'min_data_size_mb': 20.0,
                    'cpu_intensive': True
                },
                expected_benefits=["Higher throughput", "Better CPU utilization"],
                overhead_cost=0.4,
                memory_efficiency=0.6,
                processing_speed=0.9,
                complexity=0.8
            ),
            
            OptimizationStrategy(
                name="adaptive",
                description="Dynamically select best strategy based on runtime conditions",
                applicability_conditions={
                    'variable_conditions': True,
                    'performance_monitoring': True
                },
                expected_benefits=["Optimal performance", "Handles varying conditions"],
                overhead_cost=0.2,
                memory_efficiency=0.8,
                processing_speed=0.8,
                complexity=0.9
            )
        ]
        
        return strategies
    
    def _calculate_strategy_score(self, 
                                strategy: OptimizationStrategy, 
                                characteristics: DataCharacteristics) -> float:
        """
        Calculate fitness score for strategy given data characteristics.
        
        Args:
            strategy: Strategy to evaluate
            characteristics: Data characteristics
            
        Returns:
            Fitness score (0-1, higher is better)
        """
        score = 0.0
        conditions = strategy.applicability_conditions
        
        # Size-based scoring
        if 'min_data_size_mb' in conditions:
            if characteristics.data_size_mb >= conditions['min_data_size_mb']:
                score += 0.2
            else:
                score -= 0.1
        
        if 'max_data_size_mb' in conditions:
            if characteristics.data_size_mb <= conditions['max_data_size_mb']:
                score += 0.2
            else:
                score -= 0.1
        
        # Complexity-based scoring
        if 'max_complexity' in conditions:
            if characteristics.complexity_score <= conditions['max_complexity']:
                score += 0.2
            else:
                score -= 0.1
        
        # Memory intensity scoring
        if characteristics.memory_intensity > 10.0:  # High memory per row
            if 'memory_constrained' in conditions:
                score += 0.2
            if strategy.memory_efficiency > 0.8:
                score += 0.1
        
        # Processing characteristics
        if characteristics.estimated_processing_time > 5.0:  # Long processing time
            if strategy.processing_speed > 0.8:
                score += 0.2
        
        # Historical performance
        historical_effectiveness = self._strategy_effectiveness.get(strategy.name, 0.5)
        score += historical_effectiveness * 0.3
        
        # Penalize high overhead and complexity for simple cases
        if characteristics.complexity_score < 0.3:
            score -= strategy.overhead_cost * 0.2
            score -= strategy.complexity * 0.1
        
        return max(0.0, min(1.0, score))
    
    def _estimate_data_size(self, data: Any) -> float:
        """Estimate data size in MB."""
        try:
            if hasattr(data, 'memory_usage'):
                return data.memory_usage(deep=True).sum() / (1024 * 1024)
            elif hasattr(data, 'nbytes'):
                return data.nbytes / (1024 * 1024)
            else:
                import sys
                return sys.getsizeof(data) / (1024 * 1024)
        except Exception:
            return 1.0


# Factory Functions and Integration Points


def create_performance_optimizer(cache_size: int = 1000,
                               memory_limit_mb: float = 500.0,
                               lazy_threshold_mb: float = 50.0,
                               streaming_threshold_mb: float = 100.0) -> Dict[str, Any]:
    """
    Factory function to create a complete performance optimization suite.
    
    Args:
        cache_size: Maximum cache entries
        memory_limit_mb: Memory limit for caching and buffers
        lazy_threshold_mb: Threshold for enabling lazy loading
        streaming_threshold_mb: Threshold for enabling streaming
        
    Returns:
        Dictionary containing all optimization components
    """
    from .performance_optimization import ConversionCache, LazyLoadingManager
    
    cache = ConversionCache(
        max_size=cache_size,
        max_memory_mb=memory_limit_mb / 2  # Use half of limit for cache
    )
    
    lazy_manager = LazyLoadingManager(
        default_threshold_mb=lazy_threshold_mb
    )
    
    # Note: StreamingConversionEngine and MemoryPoolManager would be imported similarly
    # streaming_engine = StreamingConversionEngine(
    #     default_memory_limit_mb=streaming_threshold_mb
    # )
    
    # memory_pool = MemoryPoolManager(
    #     default_buffer_size_mb=memory_limit_mb / 10
    # )
    
    benchmark = PerformanceBenchmark()
    selector = OptimizationSelector()
    
    logger.info("Performance optimization suite created",
               cache_size=cache_size,
               memory_limit_mb=memory_limit_mb)
    
    return {
        'cache': cache,
        'lazy_manager': lazy_manager,
        # 'streaming_engine': streaming_engine,
        # 'memory_pool': memory_pool,
        'benchmark': benchmark,
        'selector': selector
    }


def optimize_conversion_request(request: ConversionRequest,
                              optimizer_suite: Dict[str, Any]) -> ConversionRequest:
    """
    Apply automatic optimization to a conversion request.
    
    Args:
        request: Original conversion request
        optimizer_suite: Suite of optimization components
        
    Returns:
        Optimized conversion request with performance hints
    """
    selector = optimizer_suite['selector']
    
    # Analyze data characteristics
    characteristics = selector.analyze_data_characteristics(request.source_data)
    
    # Get optimization recommendation
    optimization_path = selector.recommend_conversion_path(request, characteristics)
    
    # Update request with optimization hints
    optimized_request = ConversionRequest(
        source_data=request.source_data,
        source_format=request.source_format,
        target_format=request.target_format,
        metadata=request.metadata.copy(),
        context=request.context,
        format_spec=request.format_spec,
        request_id=request.request_id,
        timestamp=request.timestamp
    )
    
    # Add optimization metadata
    optimized_request.metadata.update({
        'optimization_strategy': optimization_path.selected_strategy.name,
        'data_characteristics': {
            'size_mb': characteristics.data_size_mb,
            'complexity': characteristics.complexity_score,
            'processing_estimate': characteristics.estimated_processing_time
        },
        'performance_hints': {
            'enable_caching': optimization_path.selected_strategy.name in ['cache_first', 'adaptive'],
            'enable_lazy_loading': optimization_path.selected_strategy.name in ['lazy_loading', 'adaptive'],
            'enable_streaming': optimization_path.selected_strategy.name in ['streaming', 'adaptive'],
            'enable_parallel': optimization_path.selected_strategy.name in ['parallel', 'adaptive']
        }
    })
    
    logger.debug(f"Optimized conversion request with {optimization_path.selected_strategy.name} strategy")
    
    return optimized_request


def monitor_conversion_performance(result: Any,
                                 optimizer_suite: Dict[str, Any],
                                 strategy_used: str) -> None:
    """
    Monitor and record conversion performance for optimization feedback.
    
    Args:
        result: Conversion result with performance metrics
        optimizer_suite: Suite of optimization components
        strategy_used: Name of strategy that was used
    """
    selector = optimizer_suite['selector']
    
    # Extract performance metrics from result
    performance_metrics = {}
    
    if hasattr(result, 'execution_time'):
        performance_metrics['execution_time_seconds'] = result.execution_time
    
    if hasattr(result, 'performance_metrics'):
        performance_metrics.update(result.performance_metrics)
    
    if hasattr(result, 'success'):
        performance_metrics['success_rate'] = 1.0 if result.success else 0.0
    
    # Record feedback
    selector.monitor_performance_feedback(strategy_used, performance_metrics)
    
    logger.debug(f"Recorded performance feedback for {strategy_used} strategy")


# Integration with ShimAdapter


class OptimizedShimAdapter:
    """
    Mixin class to add optimization capabilities to ShimAdapters.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._optimizer_suite = None
        self._optimization_enabled = True
    
    def set_optimization_suite(self, optimizer_suite: Dict[str, Any]) -> None:
        """Set the optimization suite for this adapter."""
        self._optimizer_suite = optimizer_suite
        logger.debug(f"Optimization suite set for adapter {self.adapter_id}")
    
    def convert_optimized(self, request: ConversionRequest) -> Any:
        """
        Perform optimized conversion with automatic strategy selection.
        
        Args:
            request: Conversion request
            
        Returns:
            Conversion result
        """
        if not self._optimization_enabled or not self._optimizer_suite:
            # Fall back to standard conversion
            return self.convert(request)
        
        # Check cache first
        cache = self._optimizer_suite.get('cache')
        if cache:
            cached_result = cache.get(request)
            if cached_result:
                logger.debug(f"Cache hit for conversion {request.request_id}")
                return cached_result.converted_data
        
        # Optimize request
        optimized_request = optimize_conversion_request(request, self._optimizer_suite)
        
        # Perform conversion with monitoring
        start_time = time.time()
        result = self.convert(optimized_request)
        execution_time = time.time() - start_time
        
        # Add execution time to result if possible
        if hasattr(result, 'execution_time'):
            result.execution_time = execution_time
        
        # Store in cache if successful
        if cache and hasattr(result, 'success') and result.success:
            cache.put(request, result)
        
        # Monitor performance
        strategy_used = optimized_request.metadata.get('optimization_strategy', 'default')
        monitor_conversion_performance(result, self._optimizer_suite, strategy_used)
        
        return result


# Export main components

__all__ = [
    'ConversionCache', 'LazyLoadingManager', 'StreamingConversionEngine',
    'MemoryPoolManager', 'PerformanceBenchmark', 'OptimizationSelector',
    'OptimizationStrategy', 'DataCharacteristics', 'OptimizedConversionPath',
    'create_performance_optimizer', 'optimize_conversion_request',
    'monitor_conversion_performance', 'OptimizedShimAdapter'
]