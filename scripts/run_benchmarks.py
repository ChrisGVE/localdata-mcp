#!/usr/bin/env python3
"""Command-line interface for running LocalData MCP performance benchmarks.

This script provides an easy way to execute comprehensive performance benchmarks
with various configuration options for testing different scenarios.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from localdata_mcp.performance_benchmarks import (
    PerformanceBenchmarkSuite, BenchmarkConfig, run_performance_benchmark
)


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run LocalData MCP performance benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all benchmarks with default settings
  python run_benchmarks.py

  # Run with larger datasets and more iterations
  python run_benchmarks.py --large-dataset 1000000 --iterations 5

  # Run only token performance benchmarks
  python run_benchmarks.py --category token

  # Run streaming vs batch comparison only
  python run_benchmarks.py --category streaming --output-dir /tmp/benchmarks

  # Run with custom chunk sizes for memory testing
  python run_benchmarks.py --chunk-sizes 100,500,1000,2000,5000

  # Quick smoke test
  python run_benchmarks.py --quick
        """
    )
    
    # Dataset size configuration
    parser.add_argument(
        '--small-dataset', type=int, default=1000,
        help='Number of rows in small dataset (default: 1000)'
    )
    parser.add_argument(
        '--medium-dataset', type=int, default=50000,
        help='Number of rows in medium dataset (default: 50000)'
    )
    parser.add_argument(
        '--large-dataset', type=int, default=500000,
        help='Number of rows in large dataset (default: 500000)'
    )
    
    # Test configuration
    parser.add_argument(
        '--iterations', type=int, default=3,
        help='Number of test iterations for averaging (default: 3)'
    )
    parser.add_argument(
        '--warmup-iterations', type=int, default=1,
        help='Number of warmup iterations (default: 1)'
    )
    parser.add_argument(
        '--chunk-sizes', type=str, default='100,500,1000,5000',
        help='Comma-separated list of chunk sizes to test (default: 100,500,1000,5000)'
    )
    
    # Test category filtering
    parser.add_argument(
        '--category', choices=['all', 'token', 'streaming', 'memory', 'regression'],
        default='all', help='Category of benchmarks to run (default: all)'
    )
    
    # Memory configuration
    parser.add_argument(
        '--memory-limit', type=int, default=512,
        help='Memory limit in MB for tests (default: 512)'
    )
    
    # Output configuration
    parser.add_argument(
        '--output-dir', type=Path, default=Path('benchmark_results'),
        help='Directory to save benchmark results (default: benchmark_results)'
    )
    parser.add_argument(
        '--no-reports', action='store_true',
        help='Skip generating human-readable reports'
    )
    parser.add_argument(
        '--no-detailed', action='store_true',
        help='Skip saving detailed metrics'
    )
    
    # Database configuration
    parser.add_argument(
        '--databases', type=str, default='sqlite',
        help='Comma-separated list of database types to test (default: sqlite)'
    )
    
    # Convenience options
    parser.add_argument(
        '--quick', action='store_true',
        help='Run quick benchmarks with reduced dataset sizes and iterations'
    )
    parser.add_argument(
        '--extensive', action='store_true',
        help='Run extensive benchmarks with larger datasets and more iterations'
    )
    
    # Logging
    parser.add_argument(
        '--verbose', '-v', action='store_true',
        help='Enable verbose logging'
    )
    parser.add_argument(
        '--quiet', '-q', action='store_true',
        help='Suppress most output'
    )
    
    return parser.parse_args()


def create_config_from_args(args) -> BenchmarkConfig:
    """Create benchmark configuration from command line arguments."""
    
    # Handle convenience options
    if args.quick:
        small_rows = 500
        medium_rows = 5000
        large_rows = 25000
        iterations = 2
        warmup = 0
    elif args.extensive:
        small_rows = args.small_dataset * 2
        medium_rows = args.medium_dataset * 2
        large_rows = args.large_dataset * 2
        iterations = args.iterations * 2
        warmup = args.warmup_iterations
    else:
        small_rows = args.small_dataset
        medium_rows = args.medium_dataset
        large_rows = args.large_dataset
        iterations = args.iterations
        warmup = args.warmup_iterations
    
    # Parse chunk sizes
    chunk_sizes = [int(x.strip()) for x in args.chunk_sizes.split(',')]
    
    # Parse database types
    database_types = [x.strip() for x in args.databases.split(',')]
    
    return BenchmarkConfig(
        small_dataset_rows=small_rows,
        medium_dataset_rows=medium_rows,
        large_dataset_rows=large_rows,
        memory_limit_mb=args.memory_limit,
        chunk_sizes=chunk_sizes,
        test_iterations=iterations,
        warmup_iterations=warmup,
        database_types=database_types,
        results_dir=args.output_dir,
        generate_reports=not args.no_reports,
        save_detailed_metrics=not args.no_detailed
    )


def filter_benchmarks_by_category(suite: PerformanceBenchmarkSuite, category: str):
    """Filter which benchmark methods to run based on category."""
    if category == 'all':
        return  # Run all benchmarks
    
    # Override the benchmark runner methods based on category
    original_methods = {}
    
    if category != 'token':
        original_methods['_run_token_benchmarks'] = suite._run_token_benchmarks
        suite._run_token_benchmarks = lambda: None
    
    if category != 'streaming':
        original_methods['_run_streaming_benchmarks'] = suite._run_streaming_benchmarks
        suite._run_streaming_benchmarks = lambda: None
    
    if category != 'memory':
        original_methods['_run_memory_benchmarks'] = suite._run_memory_benchmarks
        suite._run_memory_benchmarks = lambda: None
    
    if category != 'regression':
        original_methods['_run_regression_tests'] = suite._run_regression_tests
        suite._run_regression_tests = lambda: None
    
    return original_methods


def print_summary_table(results: dict):
    """Print a formatted summary table of benchmark results."""
    print("\n" + "="*80)
    print("PERFORMANCE BENCHMARK SUMMARY")
    print("="*80)
    
    print(f"Total Tests Executed: {results['total_tests']}")
    print(f"Successful Tests: {results['successful_tests']}")
    print(f"Success Rate: {results['success_rate']:.1%}")
    
    if results['success_rate'] > 0:
        print(f"Average Processing Rate: {results.get('average_processing_rate', 0):.0f} rows/second")
        print(f"Average Memory Usage: {results.get('average_memory_usage_mb', 0):.1f} MB")
        
        improvements = results.get('performance_improvements', 0)
        regressions = results.get('performance_regressions', 0)
        
        print(f"Performance Improvements: {improvements}")
        print(f"Performance Regressions: {regressions}")
        
        if improvements > regressions:
            print("✅ Overall performance is improving!")
        elif regressions > improvements:
            print("⚠️ Performance regressions detected - investigation recommended")
        else:
            print("ℹ️ Performance is stable")
    
    print(f"Results saved to: {results['results_directory']}")
    print("="*80)


def main():
    """Main entry point for the benchmark runner."""
    args = parse_arguments()
    
    # Setup logging
    if not args.quiet:
        setup_logging(args.verbose)
    
    logger = logging.getLogger(__name__)
    
    if not args.quiet:
        logger.info("Starting LocalData MCP performance benchmarks")
        logger.info(f"Configuration: Category={args.category}, Iterations={args.iterations}")
    
    try:
        # Create benchmark configuration
        config = create_config_from_args(args)
        
        # Create benchmark suite
        suite = PerformanceBenchmarkSuite(config)
        
        # Filter benchmarks by category if specified
        if args.category != 'all':
            filter_benchmarks_by_category(suite, args.category)
        
        # Run benchmarks
        if not args.quiet:
            logger.info("Executing benchmark suite...")
        
        results = suite.run_comprehensive_benchmark()
        
        # Print results summary
        if not args.quiet:
            print_summary_table(results)
        
        # Save summary as JSON for programmatic access
        summary_file = config.results_dir / "benchmark_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        if not args.quiet:
            logger.info(f"Summary saved to: {summary_file}")
        
        # Exit with appropriate code based on results
        if results['success_rate'] < 0.8:  # Less than 80% success rate
            logger.error("Low success rate detected - some benchmarks failed")
            sys.exit(1)
        elif results.get('performance_regressions', 0) > results.get('performance_improvements', 0):
            logger.warning("Performance regressions detected")
            sys.exit(2)
        else:
            if not args.quiet:
                logger.info("All benchmarks completed successfully!")
            sys.exit(0)
            
    except KeyboardInterrupt:
        logger.info("Benchmarks interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Benchmark execution failed: {e}")
        if args.verbose:
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)


if __name__ == '__main__':
    main()