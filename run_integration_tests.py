#!/usr/bin/env python3
"""
Time Series Integration Test Runner

This script runs the comprehensive Time Series Integration Test Suite
and Performance Benchmarking for LocalData MCP v2.0.

Usage:
    python run_integration_tests.py [options]

Options:
    --test-type: Type of tests to run ('integration', 'performance', 'all')
    --config: Performance test configuration ('standard', 'intensive')
    --output-dir: Directory to save results
    --verbose: Enable verbose output
"""

import argparse
import sys
import time
from pathlib import Path

# Add tests to path
sys.path.append(str(Path(__file__).parent / "tests"))


def run_integration_tests(verbose=False):
    """Run integration tests."""
    print("🧪 RUNNING TIME SERIES INTEGRATION TESTS")
    print("=" * 50)
    
    try:
        from tests.integration.test_time_series_integration import run_integration_test_suite
        
        success = run_integration_test_suite()
        
        if success:
            print("\n✅ INTEGRATION TESTS PASSED")
            return True
        else:
            print("\n❌ INTEGRATION TESTS FAILED")
            return False
            
    except ImportError as e:
        print(f"❌ Failed to import integration tests: {e}")
        return False
    except Exception as e:
        print(f"❌ Integration tests failed with error: {e}")
        return False


def run_performance_benchmarks(config='standard', output_dir='benchmark_results'):
    """Run performance benchmarks."""
    print("⚡ RUNNING TIME SERIES PERFORMANCE BENCHMARKS")
    print("=" * 50)
    
    try:
        from tests.performance.benchmark_time_series import run_benchmark_suite, save_benchmark_results
        
        # Run benchmark suite
        results = run_benchmark_suite(config)
        
        # Save results
        output_path = Path(output_dir) / f"benchmark_results_{config}.json"
        output_path.parent.mkdir(exist_ok=True)
        save_benchmark_results(results, str(output_path))
        
        # Check if benchmarks passed
        success_rate = results.get('success_rate', 0)
        standards_passed = all(results.get('standards_validation', {}).values())
        
        if success_rate >= 80.0 and standards_passed:
            print("\n✅ PERFORMANCE BENCHMARKS PASSED")
            return True
        else:
            print("\n❌ PERFORMANCE BENCHMARKS FAILED")
            print(f"Success rate: {success_rate:.1f}% (minimum: 80%)")
            print(f"Standards validation: {standards_passed}")
            return False
            
    except ImportError as e:
        print(f"❌ Failed to import performance benchmarks: {e}")
        return False
    except Exception as e:
        print(f"❌ Performance benchmarks failed with error: {e}")
        return False


def run_example_demonstration(output_dir='example_results'):
    """Run the complete example demonstration."""
    print("🎬 RUNNING TIME SERIES EXAMPLE DEMONSTRATION")
    print("=" * 50)
    
    try:
        from examples.time_series_complete_example import TimeSeriesExampleRunner
        
        runner = TimeSeriesExampleRunner(output_dir)
        success = runner.run_complete_example()
        
        if success:
            print("\n✅ EXAMPLE DEMONSTRATION PASSED")
            return True
        else:
            print("\n❌ EXAMPLE DEMONSTRATION FAILED")
            return False
            
    except ImportError as e:
        print(f"❌ Failed to import example demonstration: {e}")
        return False
    except Exception as e:
        print(f"❌ Example demonstration failed with error: {e}")
        return False


def generate_comprehensive_report(results, output_dir):
    """Generate comprehensive test report."""
    report_path = Path(output_dir) / "comprehensive_test_report.md"
    report_path.parent.mkdir(exist_ok=True)
    
    with open(report_path, 'w') as f:
        f.write("# Time Series Analysis Domain - Test Report\n\n")
        f.write(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Overall results
        total_tests = len(results)
        passed_tests = sum(1 for result in results.values() if result)
        
        f.write("## Overall Results\n\n")
        f.write(f"- **Total Test Categories:** {total_tests}\n")
        f.write(f"- **Passed Categories:** {passed_tests}\n")
        f.write(f"- **Success Rate:** {(passed_tests/total_tests)*100:.1f}%\n\n")
        
        # Individual results
        f.write("## Test Category Results\n\n")
        for category, passed in results.items():
            status = "✅ PASSED" if passed else "❌ FAILED"
            f.write(f"- **{category.title().replace('_', ' ')}:** {status}\n")
        
        f.write("\n## Summary\n\n")
        if all(results.values()):
            f.write("🎉 **ALL TESTS PASSED** - Time Series Analysis Domain is ready for production use.\n\n")
            f.write("The domain has been thoroughly validated with:\n")
            f.write("- Comprehensive integration testing across all tools\n")
            f.write("- Performance benchmarking with various data sizes\n")
            f.write("- Real-world usage example demonstrations\n")
            f.write("- Cross-domain integration validation\n")
            f.write("- Error handling and edge case testing\n")
        else:
            f.write("⚠️ **SOME TESTS FAILED** - Review individual test results for details.\n\n")
            failed_categories = [cat for cat, passed in results.items() if not passed]
            f.write(f"Failed categories: {', '.join(failed_categories)}\n")
        
        f.write(f"\n---\n\n*Report generated by LocalData MCP v2.0 Time Series Integration Test Suite*\n")
    
    print(f"📄 Comprehensive report saved to: {report_path}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Time Series Integration Test Runner for LocalData MCP v2.0"
    )
    
    parser.add_argument(
        '--test-type',
        choices=['integration', 'performance', 'example', 'all'],
        default='all',
        help='Type of tests to run (default: all)'
    )
    
    parser.add_argument(
        '--config',
        choices=['standard', 'intensive'],
        default='standard',
        help='Performance test configuration (default: standard)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='test_results',
        help='Directory to save results (default: test_results)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(exist_ok=True)
    
    print("🚀 TIME SERIES ANALYSIS DOMAIN - COMPREHENSIVE TEST SUITE")
    print("=" * 70)
    print(f"Test type: {args.test_type}")
    print(f"Output directory: {output_path}")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("")
    
    # Track results
    results = {}
    
    # Run requested tests
    if args.test_type in ['integration', 'all']:
        results['integration_tests'] = run_integration_tests(args.verbose)
    
    if args.test_type in ['performance', 'all']:
        perf_output = output_path / 'performance'
        results['performance_benchmarks'] = run_performance_benchmarks(args.config, str(perf_output))
    
    if args.test_type in ['example', 'all']:
        example_output = output_path / 'example'
        results['example_demonstration'] = run_example_demonstration(str(example_output))
    
    # Generate comprehensive report
    if results:
        generate_comprehensive_report(results, args.output_dir)
    
    # Final summary
    total_categories = len(results)
    passed_categories = sum(1 for passed in results.values() if passed)
    
    print(f"\n🏁 COMPREHENSIVE TEST SUITE COMPLETED")
    print("=" * 70)
    print(f"Categories tested: {total_categories}")
    print(f"Categories passed: {passed_categories}")
    print(f"Overall success rate: {(passed_categories/total_categories)*100:.1f}%")
    
    if all(results.values()):
        print(f"\n🎉 ALL TESTS PASSED!")
        print(f"✅ Time Series Analysis Domain is ready for production use.")
        exit_code = 0
    else:
        print(f"\n❌ SOME TESTS FAILED!")
        failed_categories = [cat for cat, passed in results.items() if not passed]
        print(f"Failed: {', '.join(failed_categories)}")
        exit_code = 1
    
    print(f"\nResults saved to: {output_path}")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()