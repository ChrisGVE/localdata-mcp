# Performance Benchmarking Suite for LocalData MCP v1.3.1

This comprehensive benchmarking suite validates performance improvements in v1.3.1, particularly the new streaming architecture vs the previous batch processing approach. The suite provides automated testing, regression detection, and detailed performance analysis.

## Overview

The benchmarking suite consists of several components:

- **Performance Benchmarks Module** (`performance_benchmarks.py`) - Core benchmarking framework
- **Command Line Runner** (`run_benchmarks.py`) - Easy-to-use CLI for running benchmarks
- **Version Comparison** (`compare_versions.py`) - Compare v1.3.0 vs v1.3.1 performance
- **Test Validation** (`test_benchmarks.py`) - Validate benchmarking suite functionality
- **CI/CD Integration** (`.github/workflows/performance-benchmarks.yml`) - Automated testing

## Quick Start

### 1. Install Dependencies

```bash
# Install core requirements
pip install -r requirements.txt

# Install additional benchmarking dependencies  
pip install memory_profiler psutil
```

### 2. Run Basic Benchmarks

```bash
# Run quick benchmarks with default settings
python scripts/run_benchmarks.py --quick

# Run comprehensive benchmarks
python scripts/run_benchmarks.py

# Run specific category of benchmarks
python scripts/run_benchmarks.py --category token
python scripts/run_benchmarks.py --category streaming
python scripts/run_benchmarks.py --category memory
```

### 3. Compare Versions

```bash
# Compare v1.3.0 vs v1.3.1 performance
python scripts/compare_versions.py

# Custom dataset sizes
python scripts/compare_versions.py --dataset-sizes "5000,25000,100000"
```

## Benchmark Categories

### Token Performance Benchmarks

Tests the efficiency of token counting and estimation:

- **Small datasets** (< 1MB) - Basic token counting performance
- **Medium datasets** (1-100MB) - Scalability of token estimation
- **Text-heavy datasets** - Performance with high text content
- **Wide tables** - Performance with many columns

**Key Metrics:**
- Tokens estimated per second
- Estimation accuracy
- Memory usage during estimation
- Processing time for different data characteristics

### Streaming vs Batch Benchmarks

Compares the new streaming architecture (v1.3.1) with traditional batch processing (v1.3.0):

- **Small datasets** - Overhead comparison
- **Medium datasets** - Performance crossover point
- **Large datasets** - Memory efficiency gains
- **Text-heavy datasets** - Real-world performance

**Key Metrics:**
- Processing rate (rows/second)
- Peak memory usage
- Memory efficiency ratio
- Chunk processing performance

### Memory Usage Benchmarks

Tests memory efficiency with different chunk sizes and configurations:

- **Adaptive chunking** - Performance with different chunk sizes
- **Memory bounds** - Behavior under memory pressure
- **Buffer management** - Efficiency of result buffering
- **Garbage collection** - Memory cleanup performance

**Key Metrics:**
- Peak memory usage
- Average memory usage
- Memory efficiency ratio
- Optimal chunk sizes

### Regression Testing

Automated comparison against baseline performance:

- **Baseline establishment** - Store reference performance metrics
- **Regression detection** - Identify performance degradations
- **Improvement tracking** - Monitor performance gains
- **Statistical significance** - Validate changes are meaningful

## Command Line Usage

### run_benchmarks.py

```bash
# Basic usage
python scripts/run_benchmarks.py

# Configuration options
python scripts/run_benchmarks.py \
  --small-dataset 2000 \
  --medium-dataset 100000 \
  --large-dataset 1000000 \
  --iterations 5 \
  --chunk-sizes 100,500,1000,2000,5000 \
  --category all \
  --output-dir my_benchmarks

# Quick smoke test
python scripts/run_benchmarks.py --quick

# Extensive testing
python scripts/run_benchmarks.py --extensive

# Memory-focused testing
python scripts/run_benchmarks.py \
  --category memory \
  --chunk-sizes 50,100,200,500,1000,2000,5000

# Token performance only
python scripts/run_benchmarks.py --category token --iterations 10
```

### compare_versions.py

```bash
# Basic version comparison
python scripts/compare_versions.py

# Custom configuration
python scripts/compare_versions.py \
  --dataset-sizes "1000,10000,50000,100000,500000" \
  --iterations 5 \
  --output-dir version_analysis

# Quick comparison
python scripts/compare_versions.py --dataset-sizes "5000,25000"
```

## Configuration Options

### BenchmarkConfig Parameters

```python
BenchmarkConfig(
    # Dataset sizes
    small_dataset_rows=1000,      # Small dataset size
    medium_dataset_rows=50000,    # Medium dataset size  
    large_dataset_rows=500000,    # Large dataset size
    
    # Memory configuration
    memory_limit_mb=512,          # Memory limit for tests
    chunk_sizes=[100,500,1000,5000],  # Chunk sizes to test
    
    # Test repetition
    test_iterations=3,            # Iterations for averaging
    warmup_iterations=1,          # Warmup iterations
    
    # Database types
    database_types=['sqlite', 'postgresql', 'mysql'],
    
    # Output configuration
    results_dir=Path('benchmark_results'),
    generate_reports=True,        # Generate human-readable reports
    save_detailed_metrics=True   # Save detailed JSON results
)
```

## Output Files

### Benchmark Results Directory Structure

```
benchmark_results/
├── benchmark_results_[timestamp].json    # Detailed results
├── benchmark_summary.json                # Summary statistics
├── performance_report_[timestamp].md     # Human-readable report
├── baseline_results.json                 # Baseline for regression testing
└── benchmark_comparisons_[timestamp].json # Comparison results
```

### Version Comparison Directory Structure

```
version_comparison/
├── version_comparison_[timestamp].json    # Detailed comparison
└── version_comparison_report_[timestamp].md # Comparison report
```

## Integration with CI/CD

### GitHub Actions Workflow

The included workflow (`.github/workflows/performance-benchmarks.yml`) provides:

- **Automated testing** on push/PR
- **Scheduled regression testing** (weekly)
- **Manual workflow dispatch** with configuration options
- **Multi-Python version testing** (3.9, 3.10, 3.11)
- **Performance regression alerts**

### Workflow Triggers

1. **Push to main/develop** - Quick benchmarks
2. **Pull requests** - Quick benchmarks with PR comments
3. **Weekly schedule** - Full regression testing
4. **Manual dispatch** - Configurable benchmarks

### Environment Variables

```yaml
env:
  BENCHMARK_MEMORY_LIMIT: 1024    # Memory limit in MB
  POSTGRES_DB: benchmark_test     # Test database name
  POSTGRES_USER: ${{ env.USER }}  # Database user
```

## Interpreting Results

### Performance Metrics

- **Processing Rate** - Rows processed per second (higher is better)
- **Memory Usage** - Peak memory consumption in MB (lower is better)
- **Memory Efficiency** - Ratio of memory reused (higher is better)
- **Execution Time** - Total time for operation (lower is better)

### Success Indicators

✅ **Excellent Performance**: 
- Streaming 20%+ faster than batch
- Memory usage 15%+ lower
- Consistent improvements across dataset sizes

⚠️ **Acceptable Performance**:
- Streaming 5-20% faster than batch  
- Memory usage 5-15% lower
- Some improvements with mixed results

🚨 **Performance Issues**:
- Streaming slower than batch
- Higher memory usage
- Regressions detected

### Regression Analysis

The suite automatically detects performance regressions by comparing current results against stored baselines:

- **Significant Regression**: >10% performance decrease
- **Minor Regression**: 5-10% performance decrease  
- **Improvement**: >5% performance increase
- **Stable**: <5% change

## Troubleshooting

### Common Issues

1. **ModuleNotFoundError**: Install missing dependencies
   ```bash
   pip install -r requirements.txt
   pip install memory_profiler psutil
   ```

2. **Database Connection Errors**: Ensure test databases are accessible
   ```bash
   # For PostgreSQL testing
   sudo apt-get install postgresql
   sudo -u postgres createdb benchmark_test
   ```

3. **Memory Errors**: Reduce dataset sizes or increase memory limits
   ```bash
   python scripts/run_benchmarks.py --small-dataset 500 --medium-dataset 5000
   ```

4. **Timeout Issues**: Increase timeout limits or reduce test scope
   ```bash
   python scripts/run_benchmarks.py --quick --category token
   ```

### Debugging

Enable verbose logging for detailed information:

```bash
python scripts/run_benchmarks.py --verbose
python scripts/compare_versions.py --verbose
```

Check log output in the structured logging system for performance metrics and error details.

## Advanced Usage

### Custom Benchmarking

```python
from localdata_mcp.performance_benchmarks import (
    PerformanceBenchmarkSuite, BenchmarkConfig
)

# Custom configuration
config = BenchmarkConfig(
    small_dataset_rows=5000,
    test_iterations=10,
    chunk_sizes=[250, 500, 750, 1000, 1500, 2000]
)

# Run custom benchmark
suite = PerformanceBenchmarkSuite(config)
results = suite.run_comprehensive_benchmark()

print(f"Success rate: {results['success_rate']:.1%}")
print(f"Average processing rate: {results['average_processing_rate']:.0f} rows/sec")
```

### Integration with Monitoring

The benchmarking suite integrates with the structured logging system for monitoring:

```python
from localdata_mcp.logging_manager import get_logging_manager

logging_manager = get_logging_manager()

# Performance monitoring context
with logging_manager.context(
    operation="performance_benchmark",
    component="benchmarking_suite"
):
    results = run_performance_benchmark(config)
```

## Performance Targets

### v1.3.1 Improvement Goals

- **Streaming Performance**: 15-30% faster than batch processing
- **Memory Efficiency**: 20-40% reduction in peak memory usage  
- **Scalability**: Consistent performance across dataset sizes
- **Token Estimation**: <100ms for medium datasets

### Acceptable Performance Ranges

| Metric | Small Datasets | Medium Datasets | Large Datasets |
|--------|---------------|-----------------|----------------|
| Processing Rate | >1,000 rows/sec | >5,000 rows/sec | >10,000 rows/sec |
| Memory Usage | <50 MB | <200 MB | <500 MB |
| Token Estimation | <10ms | <100ms | <1s |

## Contributing

When adding new benchmarks:

1. **Follow the pattern** - Use existing benchmark classes as templates
2. **Add comprehensive tests** - Include validation in `test_benchmarks.py`
3. **Update documentation** - Document new metrics and configuration options
4. **Test CI integration** - Ensure new benchmarks work in automated testing

### Adding New Benchmark Categories

```python
class CustomBenchmark:
    """Custom benchmark for specific performance aspects."""
    
    def benchmark_custom_feature(self, config: BenchmarkConfig) -> BenchmarkResult:
        # Implement custom benchmarking logic
        # Return BenchmarkResult with standardized metrics
        pass
```

## License

This benchmarking suite is part of LocalData MCP and follows the same licensing terms.