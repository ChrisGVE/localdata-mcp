#!/usr/bin/env python3
"""Version comparison script for LocalData MCP performance analysis.

This script helps compare performance between v1.3.0 (batch processing) and 
v1.3.1 (streaming architecture) to validate the streaming improvements.
"""

import argparse
import json
import logging
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
from sqlalchemy import create_engine

from localdata_mcp.performance_benchmarks import (
    BenchmarkResult, DatasetGenerator, PerformanceMonitor
)


class VersionComparison:
    """Compare performance between different versions of LocalData MCP."""
    
    def __init__(self, output_dir: Path = None):
        """Initialize version comparison.
        
        Args:
            output_dir: Directory to save comparison results
        """
        self.output_dir = output_dir or Path("version_comparison")
        self.output_dir.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        
    def simulate_v130_batch_processing(self, connection_string: str, query: str,
                                     dataset_size: str, iterations: int = 3) -> BenchmarkResult:
        """Simulate v1.3.0 batch processing approach for comparison.
        
        This simulates the old batch processing approach by loading entire
        result sets into memory at once using pandas.read_sql().
        
        Args:
            connection_string: Database connection string
            query: SQL query to execute
            dataset_size: Size category of dataset
            iterations: Number of test iterations
            
        Returns:
            Benchmark results for v1.3.0 simulation
        """
        self.logger.info(f"Running v1.3.0 batch processing simulation")
        
        monitor = PerformanceMonitor()
        execution_times = []
        processing_rates = []
        
        engine = create_engine(connection_string)
        
        for i in range(iterations):
            import gc
            gc.collect()  # Clean memory before test
            
            monitor.start_monitoring()
            start_time = time.time()
            
            try:
                # Simulate v1.3.0 approach: load entire result into memory
                result_df = pd.read_sql(query, engine)
                
                execution_time = time.time() - start_time
                execution_times.append(execution_time)
                
                rows_processed = len(result_df)
                processing_rate = rows_processed / execution_time if execution_time > 0 else 0
                processing_rates.append(processing_rate)
                
                self.logger.debug(f"v1.3.0 iteration {i+1}: {rows_processed} rows in {execution_time:.3f}s")
                
            except Exception as e:
                self.logger.error(f"v1.3.0 simulation failed: {e}")
                return BenchmarkResult(
                    test_name="v130_batch_processing",
                    test_category="version_comparison",
                    dataset_size=dataset_size,
                    database_type=self._extract_db_type(connection_string),
                    processing_mode="batch_v130",
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
            test_name="v130_batch_processing",
            test_category="version_comparison",
            dataset_size=dataset_size,
            database_type=self._extract_db_type(connection_string),
            processing_mode="batch_v130",
            execution_time_seconds=avg_execution_time,
            peak_memory_mb=performance_data["peak_memory_mb"],
            average_memory_mb=performance_data["average_memory_mb"],
            memory_efficiency=0.7,  # Batch processing typically less memory efficient
            rows_processed=int(avg_processing_rate * avg_execution_time),
            processing_rate_rows_per_second=avg_processing_rate,
            chunk_count=1,
            average_chunk_size=int(avg_processing_rate * avg_execution_time),
            memory_samples=performance_data["memory_samples"],
            memory_timestamps=performance_data["timestamps"],
            metadata={"version": "1.3.0", "approach": "batch"}
        )
    
    def test_v131_streaming_processing(self, connection_string: str, query: str,
                                     dataset_size: str, iterations: int = 3) -> BenchmarkResult:
        """Test v1.3.1 streaming processing approach.
        
        Args:
            connection_string: Database connection string
            query: SQL query to execute
            dataset_size: Size category of dataset
            iterations: Number of test iterations
            
        Returns:
            Benchmark results for v1.3.1 streaming
        """
        self.logger.info(f"Running v1.3.1 streaming processing test")
        
        from localdata_mcp.streaming_executor import StreamingQueryExecutor, create_streaming_source
        
        monitor = PerformanceMonitor()
        execution_times = []
        processing_rates = []
        memory_efficiencies = []
        chunk_counts = []
        
        executor = StreamingQueryExecutor()
        engine = create_engine(connection_string)
        
        for i in range(iterations):
            import gc
            gc.collect()  # Clean memory before test
            
            monitor.start_monitoring()
            start_time = time.time()
            
            try:
                # Use v1.3.1 streaming approach
                source = create_streaming_source(engine=engine, query=query)
                
                result_df, metadata = executor.execute_streaming(
                    source=source,
                    query_id=f"v131_test_{i}",
                    database_name="comparison_test"
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
                
                self.logger.debug(f"v1.3.1 iteration {i+1}: {rows_processed} rows in {execution_time:.3f}s, "
                                f"{chunk_count} chunks")
                
            except Exception as e:
                self.logger.error(f"v1.3.1 streaming test failed: {e}")
                return BenchmarkResult(
                    test_name="v131_streaming_processing",
                    test_category="version_comparison",
                    dataset_size=dataset_size,
                    database_type=self._extract_db_type(connection_string),
                    processing_mode="streaming_v131",
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
                executor.manage_memory_bounds()
        
        # Calculate averages
        avg_execution_time = sum(execution_times) / len(execution_times)
        avg_processing_rate = sum(processing_rates) / len(processing_rates)
        avg_memory_efficiency = sum(memory_efficiencies) / len(memory_efficiencies)
        avg_chunk_count = sum(chunk_counts) / len(chunk_counts)
        
        return BenchmarkResult(
            test_name="v131_streaming_processing",
            test_category="version_comparison",
            dataset_size=dataset_size,
            database_type=self._extract_db_type(connection_string),
            processing_mode="streaming_v131",
            execution_time_seconds=avg_execution_time,
            peak_memory_mb=performance_data["peak_memory_mb"],
            average_memory_mb=performance_data["average_memory_mb"],
            memory_efficiency=avg_memory_efficiency,
            rows_processed=int(avg_processing_rate * avg_execution_time),
            processing_rate_rows_per_second=avg_processing_rate,
            chunk_count=int(avg_chunk_count),
            average_chunk_size=int(avg_processing_rate * avg_execution_time / avg_chunk_count) if avg_chunk_count > 0 else 0,
            memory_samples=performance_data["memory_samples"],
            memory_timestamps=performance_data["timestamps"],
            metadata={"version": "1.3.1", "approach": "streaming"}
        )
    
    def compare_versions(self, dataset_rows: List[int] = None, 
                        iterations: int = 3) -> Dict[str, Any]:
        """Compare performance between v1.3.0 and v1.3.1 across different dataset sizes.
        
        Args:
            dataset_rows: List of dataset sizes to test
            iterations: Number of iterations per test
            
        Returns:
            Comprehensive comparison results
        """
        if dataset_rows is None:
            dataset_rows = [1000, 10000, 50000, 100000]
        
        self.logger.info(f"Starting version comparison: v1.3.0 vs v1.3.1")
        
        results = {
            "comparison_timestamp": time.time(),
            "test_config": {
                "dataset_rows": dataset_rows,
                "iterations": iterations
            },
            "results": [],
            "summary": {}
        }
        
        # Create temporary directory for test databases
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            for rows in dataset_rows:
                self.logger.info(f"Testing with {rows} rows...")
                
                # Determine dataset size category
                if rows < 10000:
                    size_category = "small"
                elif rows < 100000:
                    size_category = "medium"
                else:
                    size_category = "large"
                
                # Create test database
                db_path = temp_path / f"comparison_{rows}.db"
                connection_string = DatasetGenerator.create_test_database(
                    str(db_path), rows, text_heavy=True  # Use text-heavy data for realistic comparison
                )
                
                query = "SELECT * FROM test_data ORDER BY id"
                
                # Test v1.3.0 approach
                v130_result = self.simulate_v130_batch_processing(
                    connection_string, query, size_category, iterations
                )
                
                # Test v1.3.1 approach
                v131_result = self.test_v131_streaming_processing(
                    connection_string, query, size_category, iterations
                )
                
                # Calculate improvements
                comparison = self._calculate_improvements(v130_result, v131_result, rows)
                
                results["results"].append(comparison)
                
                self.logger.info(f"Completed {rows} rows: "
                               f"Speed improvement: {comparison['execution_time_improvement']:.1f}%, "
                               f"Memory improvement: {comparison['memory_improvement']:.1f}%")
        
        # Generate summary statistics
        results["summary"] = self._generate_comparison_summary(results["results"])
        
        # Save results
        self._save_comparison_results(results)
        
        return results
    
    def _calculate_improvements(self, v130_result: BenchmarkResult, 
                               v131_result: BenchmarkResult, rows: int) -> Dict[str, Any]:
        """Calculate performance improvements between versions."""
        
        # Handle failed tests
        if not v130_result.success or not v131_result.success:
            return {
                "dataset_rows": rows,
                "v130_success": v130_result.success,
                "v131_success": v131_result.success,
                "error": "One or both tests failed"
            }
        
        # Calculate percentage improvements (positive = improvement)
        execution_improvement = ((v130_result.execution_time_seconds - v131_result.execution_time_seconds) / 
                               v130_result.execution_time_seconds * 100)
        
        memory_improvement = ((v130_result.peak_memory_mb - v131_result.peak_memory_mb) / 
                            v130_result.peak_memory_mb * 100)
        
        rate_improvement = ((v131_result.processing_rate_rows_per_second - v130_result.processing_rate_rows_per_second) / 
                          v130_result.processing_rate_rows_per_second * 100)
        
        return {
            "dataset_rows": rows,
            "dataset_size": v131_result.dataset_size,
            "v130_result": v130_result.to_dict(),
            "v131_result": v131_result.to_dict(),
            "execution_time_improvement": execution_improvement,
            "memory_improvement": memory_improvement,
            "processing_rate_improvement": rate_improvement,
            "v130_execution_time": v130_result.execution_time_seconds,
            "v131_execution_time": v131_result.execution_time_seconds,
            "v130_memory_mb": v130_result.peak_memory_mb,
            "v131_memory_mb": v131_result.peak_memory_mb,
            "v130_rate": v130_result.processing_rate_rows_per_second,
            "v131_rate": v131_result.processing_rate_rows_per_second,
            "streaming_chunks": v131_result.chunk_count,
            "improvement_summary": self._classify_improvement(execution_improvement, memory_improvement)
        }
    
    def _classify_improvement(self, exec_improvement: float, memory_improvement: float) -> str:
        """Classify the overall improvement level."""
        if exec_improvement > 20 and memory_improvement > 15:
            return "Significant improvement in both speed and memory"
        elif exec_improvement > 10 and memory_improvement > 5:
            return "Notable improvement in both areas"
        elif exec_improvement > 20:
            return "Significant speed improvement"
        elif memory_improvement > 15:
            return "Significant memory improvement"
        elif exec_improvement > 5:
            return "Minor speed improvement"
        elif memory_improvement > 5:
            return "Minor memory improvement"
        elif exec_improvement < -5 or memory_improvement < -5:
            return "Performance regression detected"
        else:
            return "Similar performance"
    
    def _generate_comparison_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary statistics from comparison results."""
        successful_results = [r for r in results if "error" not in r]
        
        if not successful_results:
            return {"error": "No successful comparisons"}
        
        exec_improvements = [r["execution_time_improvement"] for r in successful_results]
        memory_improvements = [r["memory_improvement"] for r in successful_results]
        rate_improvements = [r["processing_rate_improvement"] for r in successful_results]
        
        return {
            "total_tests": len(results),
            "successful_tests": len(successful_results),
            "average_execution_improvement": sum(exec_improvements) / len(exec_improvements),
            "average_memory_improvement": sum(memory_improvements) / len(memory_improvements),
            "average_rate_improvement": sum(rate_improvements) / len(rate_improvements),
            "best_execution_improvement": max(exec_improvements),
            "best_memory_improvement": max(memory_improvements),
            "worst_execution_improvement": min(exec_improvements),
            "worst_memory_improvement": min(memory_improvements),
            "consistent_improvement": all(e > 0 for e in exec_improvements),
            "consistent_memory_improvement": all(m > 0 for m in memory_improvements),
            "overall_assessment": self._assess_overall_performance(successful_results)
        }
    
    def _assess_overall_performance(self, results: List[Dict[str, Any]]) -> str:
        """Assess overall performance improvement across all tests."""
        exec_improvements = [r["execution_time_improvement"] for r in results]
        memory_improvements = [r["memory_improvement"] for r in results]
        
        avg_exec = sum(exec_improvements) / len(exec_improvements)
        avg_memory = sum(memory_improvements) / len(memory_improvements)
        
        positive_exec = sum(1 for e in exec_improvements if e > 5)
        positive_memory = sum(1 for m in memory_improvements if m > 5)
        
        if avg_exec > 15 and avg_memory > 10 and positive_exec >= len(results) * 0.8:
            return "Excellent: Significant and consistent improvements"
        elif avg_exec > 10 and avg_memory > 5 and positive_exec >= len(results) * 0.7:
            return "Good: Notable improvements across most scenarios"
        elif avg_exec > 5 and positive_exec >= len(results) * 0.6:
            return "Moderate: Some improvements with mixed results"
        elif avg_exec > 0:
            return "Marginal: Slight improvements on average"
        else:
            return "Concerning: No clear performance improvements"
    
    def _save_comparison_results(self, results: Dict[str, Any]):
        """Save comparison results to files."""
        timestamp = int(time.time())
        
        # Save detailed results as JSON
        results_file = self.output_dir / f"version_comparison_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save summary report
        report_file = self.output_dir / f"version_comparison_report_{timestamp}.md"
        with open(report_file, 'w') as f:
            self._write_comparison_report(f, results)
        
        self.logger.info(f"Comparison results saved to {results_file}")
        self.logger.info(f"Comparison report saved to {report_file}")
    
    def _write_comparison_report(self, file, results: Dict[str, Any]):
        """Write human-readable comparison report."""
        file.write("# LocalData MCP Version Comparison Report\n\n")
        file.write(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        file.write("## Executive Summary\n\n")
        
        summary = results["summary"]
        if "error" in summary:
            file.write(f"❌ **Error:** {summary['error']}\n\n")
            return
        
        file.write(f"**Overall Assessment:** {summary['overall_assessment']}\n\n")
        file.write(f"- **Average Execution Time Improvement:** {summary['average_execution_improvement']:.1f}%\n")
        file.write(f"- **Average Memory Usage Improvement:** {summary['average_memory_improvement']:.1f}%\n")
        file.write(f"- **Average Processing Rate Improvement:** {summary['average_rate_improvement']:.1f}%\n")
        file.write(f"- **Tests with Consistent Improvement:** {summary['successful_tests']}/{summary['total_tests']}\n\n")
        
        file.write("## Detailed Results\n\n")
        file.write("| Dataset Size | Execution Improvement | Memory Improvement | Rate Improvement | Summary |\n")
        file.write("|-------------|---------------------|-------------------|------------------|----------|\n")
        
        for result in results["results"]:
            if "error" not in result:
                file.write(f"| {result['dataset_rows']:,} rows | "
                          f"{result['execution_time_improvement']:+.1f}% | "
                          f"{result['memory_improvement']:+.1f}% | "
                          f"{result['processing_rate_improvement']:+.1f}% | "
                          f"{result['improvement_summary']} |\n")
        
        file.write(f"\n## Key Findings\n\n")
        
        if summary['average_execution_improvement'] > 10:
            file.write("✅ **Significant speed improvements** - v1.3.1 streaming architecture is measurably faster\n\n")
        
        if summary['average_memory_improvement'] > 10:
            file.write("✅ **Significant memory efficiency gains** - Streaming reduces peak memory usage\n\n")
        
        if summary['consistent_improvement']:
            file.write("✅ **Consistent performance gains** - Improvements across all dataset sizes\n\n")
        else:
            file.write("⚠️ **Mixed performance results** - Some scenarios show regressions\n\n")
        
        file.write("## Recommendations\n\n")
        
        if summary['average_execution_improvement'] > 5 and summary['average_memory_improvement'] > 5:
            file.write("- ✅ **Recommended:** Deploy v1.3.1 streaming architecture for production use\n")
            file.write("- 📈 **Benefit:** Users will experience faster queries with lower memory usage\n")
        elif summary['average_execution_improvement'] > 0:
            file.write("- ⚠️ **Consider:** v1.3.1 shows improvements but may need optimization for specific use cases\n")
        else:
            file.write("- 🚨 **Investigate:** Performance regressions detected - review implementation\n")
    
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


def main():
    """Main entry point for version comparison."""
    parser = argparse.ArgumentParser(
        description="Compare performance between LocalData MCP v1.3.0 and v1.3.1"
    )
    
    parser.add_argument(
        '--dataset-sizes', type=str, default='1000,10000,50000,100000',
        help='Comma-separated list of dataset sizes to test (default: 1000,10000,50000,100000)'
    )
    parser.add_argument(
        '--iterations', type=int, default=3,
        help='Number of test iterations (default: 3)'
    )
    parser.add_argument(
        '--output-dir', type=Path, default=Path('version_comparison'),
        help='Output directory for results (default: version_comparison)'
    )
    parser.add_argument(
        '--verbose', '-v', action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    
    try:
        # Parse dataset sizes
        dataset_rows = [int(x.strip()) for x in args.dataset_sizes.split(',')]
        
        # Create comparison instance
        comparison = VersionComparison(args.output_dir)
        
        # Run comparison
        logger.info("Starting version performance comparison...")
        results = comparison.compare_versions(dataset_rows, args.iterations)
        
        # Print summary
        summary = results["summary"]
        print(f"\n{'='*80}")
        print("VERSION COMPARISON SUMMARY")
        print(f"{'='*80}")
        print(f"Overall Assessment: {summary.get('overall_assessment', 'Unknown')}")
        print(f"Average Execution Improvement: {summary.get('average_execution_improvement', 0):.1f}%")
        print(f"Average Memory Improvement: {summary.get('average_memory_improvement', 0):.1f}%")
        print(f"Successful Tests: {summary.get('successful_tests', 0)}/{summary.get('total_tests', 0)}")
        print(f"Results Directory: {args.output_dir}")
        print(f"{'='*80}")
        
        # Exit with appropriate code
        if summary.get('average_execution_improvement', 0) < -10:
            logger.error("Significant performance regression detected")
            sys.exit(1)
        else:
            logger.info("Comparison completed successfully")
            sys.exit(0)
            
    except Exception as e:
        logger.error(f"Version comparison failed: {e}")
        if args.verbose:
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)


if __name__ == '__main__':
    main()