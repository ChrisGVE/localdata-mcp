#!/usr/bin/env python3
"""
LocalData MCP Benchmark CLI

Command-line interface for the comprehensive LocalData MCP benchmarking system.
Provides easy access to all benchmarking functionality with configurable options.

Usage:
    python benchmark_cli.py run --full                    # Run complete benchmark suite
    python benchmark_cli.py run --datasets ecommerce iot  # Run specific datasets
    python benchmark_cli.py baseline --create             # Create new baseline
    python benchmark_cli.py compare --baseline v1.3.0     # Compare to baseline
    python benchmark_cli.py report --benchmark-id <id>    # Generate report for existing benchmark
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

# Add parent directories to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from tests.benchmarks import (
    BenchmarkOrchestrator, BaselineEstablisher, RegressionDetector,
    ReportGenerator, BaselineType, ReportConfiguration
)


def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Reduce noise from external libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)


def create_default_config() -> dict:
    """Create default benchmark configuration"""
    return {
        "benchmark": {
            "timeout_seconds": 7200,
            "retry_attempts": 3,
            "parallel_datasets": 2,
            "cleanup_temp_files": True,
            "generate_html_report": True,
            "generate_json_report": True,
            "establish_baseline": True,
            "detect_regressions": True
        },
        "datasets": {
            "ecommerce": {"enabled": True, "size_gb": 5},
            "iot": {"enabled": True, "size_gb": 12}, 
            "social_media": {"enabled": True, "size_gb": 6},
            "financial": {"enabled": False, "size_gb": 8}  # Missing generator
        },
        "performance_testing": {
            "memory_testing": {"enabled": True, "max_memory_gb": 16},
            "concurrent_testing": {"enabled": True, "max_threads": 50},
            "stress_duration_minutes": 30
        },
        "reporting": {
            "include_visualizations": True,
            "include_recommendations": True,
            "comparison_thresholds": {
                "memory_increase_warning": 20,
                "performance_degradation_warning": 15,
                "failure_rate_warning": 5
            }
        }
    }


async def run_benchmark(args):
    """Run benchmark suite"""
    print("🚀 Starting LocalData MCP Comprehensive Benchmark")
    print("=" * 60)
    
    # Setup output directory
    output_dir = Path(args.output_dir) if args.output_dir else Path("benchmark_results")
    
    # Load or create configuration
    config = create_default_config()
    if args.config:
        with open(args.config, 'r') as f:
            user_config = json.load(f)
            # Simple merge (user config overrides defaults)
            config.update(user_config)
    
    # Apply CLI overrides
    if args.datasets:
        # Disable all datasets first
        for dataset in config["datasets"]:
            config["datasets"][dataset]["enabled"] = False
        
        # Enable specified datasets
        for dataset in args.datasets:
            if dataset in config["datasets"]:
                config["datasets"][dataset]["enabled"] = True
            else:
                print(f"⚠️  Warning: Unknown dataset '{dataset}' - skipping")
    
    if args.no_cleanup:
        config["benchmark"]["cleanup_temp_files"] = False
    
    if args.skip_reports:
        config["benchmark"]["generate_html_report"] = False
        config["benchmark"]["generate_json_report"] = False
    
    # Initialize orchestrator
    orchestrator = BenchmarkOrchestrator(
        output_dir=str(output_dir),
        config_file=None  # We're passing config directly
    )
    orchestrator.config = config
    
    try:
        # Run comprehensive benchmark
        result = await orchestrator.run_comprehensive_benchmark()
        
        print("\n" + "=" * 60)
        print("✅ BENCHMARK COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print(f"📊 Benchmark ID: {result.benchmark_id}")
        print(f"⏱️  Duration: {result.total_duration_seconds:.1f} seconds")
        print(f"✅ Success Rate: {result.success_rate:.1f}%")
        print(f"📈 Performance Score: {result.performance_summary.get('overall_performance_score', 'N/A')}")
        print(f"📁 Reports Directory: {output_dir / 'reports'}")
        
        # Show recommendations if any
        if result.recommendations:
            print(f"\n📋 Top Recommendations:")
            for i, rec in enumerate(result.recommendations[:3], 1):
                print(f"  {i}. {rec}")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ BENCHMARK FAILED")
        print(f"Error: {str(e)}")
        return 1


def create_baseline(args):
    """Create performance baseline"""
    print("📏 Creating Performance Baseline")
    print("=" * 60)
    
    baselines_dir = Path(args.baselines_dir) if args.baselines_dir else Path("benchmark_results/baselines")
    
    if not args.benchmark_result:
        print("❌ Error: --benchmark-result required to create baseline")
        return 1
    
    result_file = Path(args.benchmark_result)
    if not result_file.exists():
        print(f"❌ Error: Benchmark result file not found: {result_file}")
        return 1
    
    try:
        # Load benchmark result
        with open(result_file, 'r') as f:
            result_data = json.load(f)
        
        # Convert to BenchmarkResult object (simplified)
        class MockBenchmarkResult:
            def __init__(self, data):
                self.__dict__.update(data)
        
        benchmark_result = MockBenchmarkResult(result_data)
        
        # Initialize baseline establisher
        establisher = BaselineEstablisher(baselines_dir)
        
        # Determine baseline type
        baseline_type = BaselineType.DEVELOPMENT
        if args.baseline_type:
            try:
                baseline_type = BaselineType(args.baseline_type)
            except ValueError:
                print(f"❌ Error: Invalid baseline type '{args.baseline_type}'. Valid types: {[t.value for t in BaselineType]}")
                return 1
        
        # Create baseline
        baseline = establisher.establish_baseline(benchmark_result, baseline_type)
        
        print(f"✅ Baseline created successfully")
        print(f"📝 Version: {baseline.version}")
        print(f"🏷️  Type: {baseline.baseline_type.value}")
        print(f"📁 Saved to: {baselines_dir}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error creating baseline: {str(e)}")
        return 1


def compare_to_baseline(args):
    """Compare benchmark results to baseline"""
    print("📊 Comparing to Baseline")
    print("=" * 60)
    
    if not args.benchmark_result:
        print("❌ Error: --benchmark-result required for comparison")
        return 1
    
    baselines_dir = Path(args.baselines_dir) if args.baselines_dir else Path("benchmark_results/baselines")
    result_file = Path(args.benchmark_result)
    
    if not result_file.exists():
        print(f"❌ Error: Benchmark result file not found: {result_file}")
        return 1
    
    try:
        # Load benchmark result
        with open(result_file, 'r') as f:
            result_data = json.load(f)
        
        class MockBenchmarkResult:
            def __init__(self, data):
                self.__dict__.update(data)
        
        benchmark_result = MockBenchmarkResult(result_data)
        
        # Initialize components
        establisher = BaselineEstablisher(baselines_dir)
        detector = RegressionDetector(establisher)
        
        # Compare to baseline
        if args.baseline_version:
            comparison = establisher.compare_to_baseline(
                benchmark_result, 
                args.baseline_version,
                BaselineType(args.baseline_type) if args.baseline_type else BaselineType.DEVELOPMENT
            )
        else:
            comparison = establisher.compare_to_baseline(benchmark_result)
        
        # Run regression detection
        regression_report = detector.detect_regressions(benchmark_result, args.baseline_version)
        
        # Display results
        print(f"📊 Comparison Results")
        print(f"🔄 Current: v{comparison.current_version}")  
        print(f"📏 Baseline: v{comparison.baseline_version}")
        print(f"📈 Overall Assessment: {comparison.overall_assessment.get('overall_trend', 'unknown')}")
        
        # Show regressions
        if regression_report.regressions_detected:
            print(f"\n⚠️  Regressions Detected: {len(regression_report.regressions_detected)}")
            
            for regression in regression_report.regressions_detected[:5]:  # Top 5
                severity_emoji = "🚨" if regression.severity.value == "critical" else "⚠️" 
                print(f"  {severity_emoji} {regression.metric_name}: {regression.percentage_change:+.1f}% ({regression.severity.value})")
        else:
            print("✅ No significant regressions detected")
        
        # Show top recommendations
        if regression_report.recommendations:
            print(f"\n💡 Recommendations:")
            for i, rec in enumerate(regression_report.recommendations[:3], 1):
                print(f"  {i}. {rec}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error in comparison: {str(e)}")
        return 1


def generate_report(args):
    """Generate report for existing benchmark"""
    print("📄 Generating Benchmark Report")
    print("=" * 60)
    
    if not args.benchmark_result:
        print("❌ Error: --benchmark-result required for report generation")
        return 1
    
    result_file = Path(args.benchmark_result)
    output_dir = Path(args.output_dir) if args.output_dir else result_file.parent
    
    try:
        # Load benchmark result
        with open(result_file, 'r') as f:
            result_data = json.load(f)
        
        class MockBenchmarkResult:
            def __init__(self, data):
                self.__dict__.update(data)
        
        benchmark_result = MockBenchmarkResult(result_data)
        
        # Configure report generation
        config = ReportConfiguration(
            include_visualizations=not args.no_charts,
            include_raw_data=args.include_raw_data,
            export_formats=args.formats or ["html", "json"]
        )
        
        # Initialize report generator
        generator = ReportGenerator(output_dir / "reports", config)
        
        # Generate reports
        generated_files = generator.generate_comprehensive_report(benchmark_result)
        
        print(f"✅ Reports generated successfully")
        for format_type, file_path in generated_files.items():
            print(f"  📄 {format_type.upper()}: {file_path}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error generating report: {str(e)}")
        return 1


def list_benchmarks(args):
    """List available benchmarks and baselines"""
    print("📋 Available Benchmarks and Baselines")
    print("=" * 60)
    
    # List benchmark results
    results_dir = Path(args.output_dir) if args.output_dir else Path("benchmark_results")
    
    if (results_dir / "reports").exists():
        result_files = list((results_dir / "reports").glob("*_results.json"))
        
        if result_files:
            print(f"📊 Benchmark Results ({len(result_files)}):")
            for result_file in sorted(result_files)[-5:]:  # Latest 5
                try:
                    with open(result_file, 'r') as f:
                        data = json.load(f)
                    
                    benchmark_id = data.get("benchmark_id", result_file.stem)
                    timestamp = data.get("timestamp", "unknown")
                    success_rate = data.get("success_rate", 0)
                    
                    print(f"  📄 {benchmark_id} ({timestamp}) - Success: {success_rate:.1f}%")
                except:
                    print(f"  📄 {result_file.name} (could not read)")
        else:
            print("📊 No benchmark results found")
    
    # List baselines
    baselines_dir = Path(args.baselines_dir) if args.baselines_dir else Path("benchmark_results/baselines")
    
    if baselines_dir.exists():
        try:
            establisher = BaselineEstablisher(baselines_dir)
            baselines = establisher.list_available_baselines()
            
            if baselines:
                print(f"\n📏 Baselines ({sum(len(types) for types in baselines.values())}):")
                for version, baseline_types in baselines.items():
                    types_str = ", ".join(baseline_types)
                    print(f"  📌 v{version}: {types_str}")
            else:
                print("\n📏 No baselines found")
        except Exception as e:
            print(f"\n📏 Error reading baselines: {str(e)}")
    
    return 0


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="LocalData MCP Comprehensive Benchmarking System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s run --full                           # Run complete benchmark suite
  %(prog)s run --datasets ecommerce iot        # Run specific datasets only  
  %(prog)s run --output-dir ./my_results       # Custom output directory
  %(prog)s baseline --create --benchmark-result results.json
  %(prog)s compare --benchmark-result results.json --baseline v1.3.0
  %(prog)s report --benchmark-result results.json --formats html json
        """
    )
    
    # Global options
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    parser.add_argument("--output-dir", help="Output directory for results")
    
    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Run benchmark command
    run_parser = subparsers.add_parser("run", help="Run benchmark suite")
    run_parser.add_argument("--full", action="store_true", help="Run full benchmark suite")
    run_parser.add_argument("--datasets", nargs="+", choices=["ecommerce", "iot", "social_media", "financial"], 
                           help="Specific datasets to benchmark")
    run_parser.add_argument("--config", help="Configuration file path")
    run_parser.add_argument("--no-cleanup", action="store_true", help="Don't cleanup temporary files")
    run_parser.add_argument("--skip-reports", action="store_true", help="Skip report generation")
    
    # Baseline command
    baseline_parser = subparsers.add_parser("baseline", help="Baseline management")
    baseline_parser.add_argument("--create", action="store_true", help="Create new baseline")
    baseline_parser.add_argument("--benchmark-result", help="Benchmark result file to use")
    baseline_parser.add_argument("--baseline-type", choices=[t.value for t in BaselineType], 
                                help="Type of baseline to create")
    baseline_parser.add_argument("--baselines-dir", help="Baselines directory")
    
    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare to baseline")
    compare_parser.add_argument("--benchmark-result", required=True, help="Benchmark result file")
    compare_parser.add_argument("--baseline-version", help="Baseline version to compare against")
    compare_parser.add_argument("--baseline-type", choices=[t.value for t in BaselineType],
                               help="Type of baseline to compare against")
    compare_parser.add_argument("--baselines-dir", help="Baselines directory")
    
    # Report command
    report_parser = subparsers.add_parser("report", help="Generate reports")
    report_parser.add_argument("--benchmark-result", required=True, help="Benchmark result file")
    report_parser.add_argument("--formats", nargs="+", choices=["html", "json", "csv"],
                              help="Report formats to generate")
    report_parser.add_argument("--no-charts", action="store_true", help="Exclude charts from reports")
    report_parser.add_argument("--include-raw-data", action="store_true", help="Include raw data in reports")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List benchmarks and baselines")
    list_parser.add_argument("--baselines-dir", help="Baselines directory")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Handle commands
    if args.command == "run":
        return asyncio.run(run_benchmark(args))
    elif args.command == "baseline":
        if args.create:
            return create_baseline(args)
        else:
            print("❌ Error: --create required for baseline command")
            return 1
    elif args.command == "compare":
        return compare_to_baseline(args)
    elif args.command == "report":
        return generate_report(args)
    elif args.command == "list":
        return list_benchmarks(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())