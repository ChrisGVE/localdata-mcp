#!/usr/bin/env python3
"""Test script to validate the performance benchmarking suite.

This script runs a minimal set of benchmarks to ensure the benchmarking
framework is working correctly without running the full comprehensive suite.
"""

import logging
import sys
import tempfile
from pathlib import Path

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from localdata_mcp.performance_benchmarks import (
    PerformanceBenchmarkSuite, BenchmarkConfig, DatasetGenerator, 
    TokenBenchmark, StreamingBenchmark
)


def setup_logging():
    """Setup basic logging for testing."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def test_dataset_generator():
    """Test the dataset generator functionality."""
    print("Testing dataset generator...")
    
    try:
        # Test DataFrame generation
        df = DatasetGenerator.create_test_dataframe(100, text_heavy=True)
        assert len(df) == 100
        assert 'id' in df.columns
        assert 'long_text' in df.columns
        print("✅ Dataset generator working correctly")
        return True
    except Exception as e:
        print(f"❌ Dataset generator failed: {e}")
        return False


def test_token_benchmark():
    """Test the token benchmarking functionality."""
    print("Testing token benchmark...")
    
    try:
        # Create small test dataset
        df = DatasetGenerator.create_test_dataframe(50, text_heavy=True)
        
        # Test token benchmark
        token_benchmark = TokenBenchmark()
        result = token_benchmark.benchmark_token_estimation(df, iterations=1)
        
        assert result.success
        assert result.token_count > 0
        assert result.processing_rate_rows_per_second > 0
        
        print(f"✅ Token benchmark working correctly")
        print(f"   - Processed {result.rows_processed} rows")
        print(f"   - Rate: {result.processing_rate_rows_per_second:.0f} rows/sec")
        print(f"   - Tokens: {result.token_count}")
        return True
    except Exception as e:
        print(f"❌ Token benchmark failed: {e}")
        return False


def test_streaming_benchmark():
    """Test the streaming benchmark functionality."""
    print("Testing streaming benchmark...")
    
    try:
        # Create temporary test database
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            
            # Generate small test database
            connection_string = DatasetGenerator.create_test_database(
                str(db_path), 100, text_heavy=False
            )
            
            # Test streaming benchmark
            streaming_benchmark = StreamingBenchmark()
            streaming_result, batch_result = streaming_benchmark.benchmark_streaming_vs_batch(
                connection_string, "SELECT * FROM test_data", "small", iterations=1
            )
            
            assert streaming_result.success
            assert batch_result.success
            assert streaming_result.rows_processed > 0
            assert batch_result.rows_processed > 0
            
            print(f"✅ Streaming benchmark working correctly")
            print(f"   - Streaming: {streaming_result.processing_rate_rows_per_second:.0f} rows/sec")
            print(f"   - Batch: {batch_result.processing_rate_rows_per_second:.0f} rows/sec")
            return True
    except Exception as e:
        print(f"❌ Streaming benchmark failed: {e}")
        import traceback
        print(f"   Error details: {traceback.format_exc()}")
        return False


def test_benchmark_suite():
    """Test the complete benchmark suite with minimal configuration."""
    print("Testing benchmark suite...")
    
    try:
        # Create minimal configuration
        config = BenchmarkConfig(
            small_dataset_rows=50,
            medium_dataset_rows=100,
            large_dataset_rows=200,
            chunk_sizes=[50, 100],
            test_iterations=1,
            warmup_iterations=0,
            database_types=['sqlite'],
            generate_reports=False,
            save_detailed_metrics=False
        )
        
        # Create benchmark suite
        suite = PerformanceBenchmarkSuite(config)
        
        # Override methods to run minimal tests
        def minimal_token_benchmarks():
            """Run minimal token benchmarks."""
            df = DatasetGenerator.create_test_dataframe(50, text_heavy=False)
            result = suite.token_benchmark.benchmark_token_estimation(df, iterations=1)
            result.test_name = "minimal_token_test"
            suite.results.append(result)
        
        def minimal_streaming_benchmarks():
            """Run minimal streaming benchmarks.""" 
            # Skip streaming benchmarks in minimal test to avoid complexity
            pass
        
        def minimal_memory_benchmarks():
            """Run minimal memory benchmarks."""
            # Skip memory benchmarks in minimal test
            pass
        
        def minimal_regression_tests():
            """Skip regression tests in minimal test."""
            pass
        
        # Replace methods with minimal versions
        suite._run_token_benchmarks = minimal_token_benchmarks
        suite._run_streaming_benchmarks = minimal_streaming_benchmarks
        suite._run_memory_benchmarks = minimal_memory_benchmarks
        suite._run_regression_tests = minimal_regression_tests
        
        # Run the suite
        results = suite.run_comprehensive_benchmark()
        
        assert results['total_tests'] > 0
        assert results['success_rate'] > 0
        
        print(f"✅ Benchmark suite working correctly")
        print(f"   - Tests: {results['total_tests']}")
        print(f"   - Success rate: {results['success_rate']:.1%}")
        return True
    except Exception as e:
        print(f"❌ Benchmark suite failed: {e}")
        import traceback
        print(f"   Error details: {traceback.format_exc()}")
        return False


def main():
    """Run all validation tests."""
    setup_logging()
    
    print("🧪 Validating Performance Benchmarking Suite")
    print("=" * 50)
    
    tests = [
        test_dataset_generator,
        test_token_benchmark,
        test_streaming_benchmark,
        test_benchmark_suite
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except KeyboardInterrupt:
            print("\n❌ Testing interrupted by user")
            sys.exit(130)
        except Exception as e:
            print(f"❌ Test failed with unexpected error: {e}")
        print()
    
    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ All tests passed! Benchmarking suite is ready to use.")
        sys.exit(0)
    else:
        print(f"❌ {total - passed} tests failed. Please check the implementation.")
        sys.exit(1)


if __name__ == "__main__":
    main()