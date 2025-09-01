# LocalData MCP Benchmarking & Reporting Framework

Comprehensive benchmarking system that validates LocalData MCP v1.3.0 architecture and provides baseline metrics for v1.4.0+ development.

## Overview

This framework integrates all stress testing components into a unified benchmarking system:

- **Dataset Generators**: E-commerce (5GB), IoT (12GB), Social Media (6GB)
- **Performance Testing**: Memory safety and concurrent usage validation
- **Metrics Collection**: Comprehensive performance data collection
- **Baseline Management**: Version-based performance baselines
- **Regression Detection**: Advanced statistical regression analysis
- **Reporting**: Interactive HTML and structured JSON reports

## Quick Start

### Run Complete Benchmark Suite

```bash
# From the tests/benchmarks directory
python benchmark_cli.py run --full

# Or run specific datasets
python benchmark_cli.py run --datasets ecommerce iot
```

### Create Performance Baseline

```bash
# After running a benchmark
python benchmark_cli.py baseline --create --benchmark-result benchmark_results/reports/benchmark_20231201_120000_results.json
```

### Compare Against Baseline

```bash
python benchmark_cli.py compare --benchmark-result results.json --baseline v1.3.0
```

### Generate Custom Reports

```bash
python benchmark_cli.py report --benchmark-result results.json --formats html json csv
```

## Architecture

### Core Components

1. **BenchmarkOrchestrator**: Main coordination system
   - Orchestrates all benchmarking phases
   - Manages parallel execution and error handling
   - Integrates with all testing frameworks

2. **DatasetBenchmark**: Individual dataset benchmark runner
   - Measures generation performance and throughput
   - Validates data integrity and streaming architecture
   - Executes query performance tests

3. **PerformanceCollector**: Comprehensive metrics collection
   - Real-time system resource monitoring
   - Application-level performance tracking
   - Statistical analysis and trend detection

4. **ReportGenerator**: Multi-format reporting system
   - Interactive HTML reports with visualizations
   - Structured JSON for automation
   - CSV exports for analysis
   - Executive summaries

5. **BaselineEstablisher**: Performance baseline management
   - Creates version-specific baselines
   - Supports multiple baseline types (dev, staging, production)
   - Statistical baseline comparison

6. **RegressionDetector**: Advanced regression analysis
   - Multiple detection algorithms (statistical, trend, anomaly, pattern)
   - Confidence-based scoring
   - Automated severity assessment

### Integration Points

```python
from tests.benchmarks import (
    BenchmarkOrchestrator, BaselineEstablisher, 
    RegressionDetector, ReportGenerator
)

# Initialize orchestrator
orchestrator = BenchmarkOrchestrator(output_dir="results")

# Run comprehensive benchmark
result = await orchestrator.run_comprehensive_benchmark()

# Establish baseline
establisher = BaselineEstablisher(Path("baselines"))
baseline = establisher.establish_baseline(result)

# Detect regressions
detector = RegressionDetector(establisher)
regression_report = detector.detect_regressions(result)
```

## Configuration

### Default Configuration

The framework uses sensible defaults that can be overridden via configuration file:

```json
{
  "benchmark": {
    "timeout_seconds": 7200,
    "parallel_datasets": 2,
    "cleanup_temp_files": true
  },
  "datasets": {
    "ecommerce": {"enabled": true, "size_gb": 5},
    "iot": {"enabled": true, "size_gb": 12},
    "social_media": {"enabled": true, "size_gb": 6}
  },
  "performance_testing": {
    "memory_testing": {"enabled": true, "max_memory_gb": 16},
    "concurrent_testing": {"enabled": true, "max_threads": 50}
  }
}
```

### CLI Usage

```bash
# Use custom configuration
python benchmark_cli.py run --config custom_config.json

# Run with specific parameters
python benchmark_cli.py run --datasets iot social_media --no-cleanup

# Generate specific report formats
python benchmark_cli.py report --benchmark-result results.json --formats html csv
```

## Dataset Integration

### Available Datasets

1. **E-commerce Dataset (5GB)**
   - Products, customers, orders, reviews, transactions
   - Complex relational queries
   - High write throughput validation

2. **IoT Dataset (12GB)**
   - Devices, sensors, telemetry, network topology
   - Time-series data patterns
   - Streaming architecture stress testing

3. **Social Media Dataset (6GB)**
   - Users, posts, interactions, graph relationships
   - Text-heavy processing
   - Complex graph queries

### Dataset Performance Metrics

- **Generation Time**: Time to generate full dataset
- **Throughput**: MB/second generation rate
- **Query Performance**: Average execution time for test queries
- **Memory Efficiency**: Memory usage during operations
- **Streaming Activation**: Whether streaming architecture engaged

## Performance Testing Integration

### Memory Safety Testing

```python
# Validates streaming architecture behavior
memory_results = {
    "streaming_activation_tests": "passed",
    "memory_leak_tests": "passed",
    "max_memory_usage_mb": 2048,
    "streaming_threshold_mb": 1024
}
```

### Concurrent Usage Testing

```python
# Validates concurrent operation handling  
concurrent_results = {
    "max_concurrent_operations": 25,
    "deadlock_tests": "passed",
    "race_condition_tests": "passed",
    "avg_response_time_ms": 85
}
```

## Baseline Management

### Creating Baselines

```bash
# Development baseline (default)
python benchmark_cli.py baseline --create --benchmark-result results.json

# Production baseline
python benchmark_cli.py baseline --create --benchmark-result results.json --baseline-type production
```

### Baseline Types

- **Development**: For ongoing development validation
- **Staging**: For pre-production verification
- **Production**: For production performance validation
- **Regression**: For regression test suites

### Baseline Comparison

The system automatically compares current results against appropriate baselines:

```python
comparison = establisher.compare_to_baseline(benchmark_result, "v1.3.0")
```

## Regression Detection

### Detection Algorithms

1. **Statistical Detection**: Statistical significance analysis
2. **Trend Detection**: Time-series trend analysis
3. **Anomaly Detection**: Outlier and spike detection
4. **Pattern Detection**: Multi-dimensional pattern recognition

### Severity Levels

- **Critical**: >30% degradation or system failure
- **High**: 15-30% degradation
- **Medium**: 5-15% degradation
- **Low**: 2-5% degradation
- **Negligible**: <2% degradation

### Confidence Scoring

Each regression detection includes confidence score (0.0-1.0) based on:
- Statistical significance
- Historical patterns
- Detection method reliability
- Data quality

## Reporting System

### HTML Reports

Interactive reports with:
- Executive summary with key metrics
- System information and configuration
- Dataset performance analysis
- Memory and concurrency validation results
- Performance visualizations (charts/graphs)
- Actionable recommendations

### JSON Reports

Structured data for automation:
- Complete benchmark results
- Performance metrics
- Baseline comparisons
- Regression analysis
- Metadata and timestamps

### CSV Exports

Tabular data for analysis:
- Key performance metrics
- Time-series data
- Comparison results
- Trend analysis

## System Requirements

### Minimum Requirements

- **OS**: Linux, macOS, or Windows
- **Python**: 3.8+
- **Memory**: 4GB+ available RAM
- **Storage**: 50GB+ free space (for all datasets)

### Recommended Configuration

- **Memory**: 16GB+ RAM for optimal performance
- **Storage**: SSD with 100GB+ free space
- **CPU**: Multi-core processor for parallel execution

### Dependencies

```bash
pip install psutil numpy faker
```

## Usage Examples

### Complete Validation Pipeline

```bash
# 1. Run comprehensive benchmark
python benchmark_cli.py run --full --output-dir validation_results

# 2. Create baseline for current version
python benchmark_cli.py baseline --create \
  --benchmark-result validation_results/reports/benchmark_*_results.json \
  --baseline-type production

# 3. Compare future runs to baseline
python benchmark_cli.py compare \
  --benchmark-result new_results.json \
  --baseline v1.3.0 \
  --baseline-type production

# 4. Generate detailed report
python benchmark_cli.py report \
  --benchmark-result new_results.json \
  --formats html json csv
```

### Continuous Integration Integration

```yaml
# .github/workflows/performance.yml
- name: Run Performance Benchmark
  run: |
    python tests/benchmarks/benchmark_cli.py run --datasets ecommerce iot
    
- name: Compare Against Baseline  
  run: |
    python tests/benchmarks/benchmark_cli.py compare \
      --benchmark-result benchmark_results/reports/*_results.json \
      --baseline v1.3.0
```

### Custom Dataset Testing

```bash
# Test specific dataset only
python benchmark_cli.py run --datasets social_media

# Skip cleanup for debugging
python benchmark_cli.py run --datasets ecommerce --no-cleanup

# Generate reports without running benchmark
python benchmark_cli.py report --benchmark-result existing_results.json
```

## Output Structure

```
benchmark_results/
├── datasets/           # Generated test datasets
│   ├── ecommerce_benchmark/
│   ├── iot_benchmark/
│   └── social_media_benchmark/
├── reports/           # Generated reports
│   ├── benchmark_*_report.html
│   ├── benchmark_*_results.json
│   └── benchmark_*_metrics.csv
├── baselines/         # Performance baselines  
│   ├── v1.3.0_development_baseline.json
│   └── v1.3.0_production_baseline.json
└── logs/             # Execution logs
    └── benchmark_*.log
```

## Performance Optimization

### For Large Datasets

- Use parallel dataset generation (`parallel_datasets: 4`)
- Increase timeout for large datasets (`timeout_seconds: 14400`)
- Enable cleanup to manage disk space (`cleanup_temp_files: true`)

### For Memory-Constrained Systems

- Reduce concurrent threads (`max_threads: 25`)
- Lower memory limits (`max_memory_gb: 8`)
- Disable raw data in reports (`include_raw_data: false`)

### For CI/CD Environments

- Skip visualizations (`include_visualizations: false`)
- Use minimal datasets configuration
- Enable automated cleanup
- Focus on JSON reports for automation

## Troubleshooting

### Common Issues

1. **Insufficient Disk Space**
   ```bash
   # Check available space
   df -h
   # Enable cleanup
   python benchmark_cli.py run --cleanup-temp-files
   ```

2. **Memory Errors**
   ```bash
   # Reduce parallel execution
   python benchmark_cli.py run --config reduced_memory_config.json
   ```

3. **Permission Errors**
   ```bash
   # Ensure write permissions
   chmod -R 755 benchmark_results/
   ```

### Debug Mode

```bash
# Enable verbose logging
python benchmark_cli.py run --verbose --datasets ecommerce
```

### Validation

```bash
# List available benchmarks and baselines
python benchmark_cli.py list

# Validate configuration
python benchmark_cli.py run --config benchmark_config.json --datasets ecommerce
```

## Contributing

### Adding New Datasets

1. Create generator in `stress_testing/dataset_generators/`
2. Add dataset spec to `BenchmarkOrchestrator.initialize_dataset_specs()`
3. Update configuration schema
4. Add integration tests

### Extending Reports

1. Modify `ReportGenerator` for new formats
2. Add visualization components
3. Update CLI to support new options
4. Test with various benchmark results

### Custom Regression Detection

1. Implement detection algorithm in `RegressionDetector`
2. Add to detection pipeline
3. Define severity calculations
4. Provide actionable recommendations

## License

LocalData MCP Benchmarking Framework
Copyright (c) 2024 LocalData MCP Team