"""Benchmark result persistence, report generation, and summarization.

This module provides functions for saving detailed benchmark results to
JSON files, generating human-readable Markdown reports, and producing
summary dictionaries and optimization recommendations.
"""

import json
import time
from dataclasses import asdict
from typing import Any, Dict, List

from ..logging_manager import get_logger
from .models import BenchmarkConfig, BenchmarkResult, ComparisonResult

logger = get_logger(__name__)


def save_detailed_results(
    config: BenchmarkConfig,
    results: List[BenchmarkResult],
    comparisons: List[ComparisonResult],
) -> None:
    """Save detailed benchmark results to files.

    Args:
        config: Benchmark configuration with results directory.
        results: All benchmark results to persist.
        comparisons: All comparison results to persist.
    """
    logger.info("Saving detailed benchmark results")

    results_file = config.results_dir / f"benchmark_results_{int(time.time())}.json"
    with open(results_file, "w") as f:
        json.dump([result.to_dict() for result in results], f, indent=2, default=str)

    baseline_file = config.results_dir / "baseline_results.json"
    if not baseline_file.exists():
        with open(baseline_file, "w") as f:
            json.dump(
                [result.to_dict() for result in results],
                f,
                indent=2,
                default=str,
            )

    if comparisons:
        comparisons_file = (
            config.results_dir / f"benchmark_comparisons_{int(time.time())}.json"
        )
        with open(comparisons_file, "w") as f:
            json.dump(
                [asdict(comp) for comp in comparisons],
                f,
                indent=2,
                default=str,
            )

    logger.info(f"Detailed results saved to {results_file}")


def calculate_success_rate(results: List[BenchmarkResult]) -> float:
    """Calculate the success rate of benchmark tests.

    Args:
        results: All benchmark results.

    Returns:
        Success rate as a float between 0.0 and 1.0.
    """
    if not results:
        return 0.0
    return sum(1 for r in results if r.success) / len(results)


def generate_recommendations(results: List[BenchmarkResult]) -> List[str]:
    """Generate performance optimization recommendations.

    Args:
        results: All benchmark results to analyze.

    Returns:
        List of recommendation strings.
    """
    recommendations: List[str] = []

    streaming_results = [
        r for r in results if r.processing_mode == "streaming" and r.success
    ]
    batch_results = [r for r in results if r.processing_mode == "batch" and r.success]

    if streaming_results and batch_results:
        avg_streaming_rate = sum(
            r.processing_rate_rows_per_second for r in streaming_results
        ) / len(streaming_results)
        avg_batch_rate = sum(
            r.processing_rate_rows_per_second for r in batch_results
        ) / len(batch_results)

        if avg_streaming_rate > avg_batch_rate * 1.1:
            recommendations.append(
                "Streaming processing shows significant performance benefits"
                " - recommend for large datasets"
            )
        elif avg_batch_rate > avg_streaming_rate * 1.1:
            recommendations.append(
                "Batch processing performs better for current workloads"
                " - streaming may have overhead"
            )
        else:
            recommendations.append(
                "Streaming and batch processing show similar performance"
                " - choose based on memory constraints"
            )

    memory_results = [
        r for r in results if r.test_category == "memory_efficiency" and r.success
    ]
    if memory_results:
        best_chunk_result = min(memory_results, key=lambda r: r.peak_memory_mb)
        optimal_chunk_size = best_chunk_result.metadata.get("chunk_size", "unknown")
        recommendations.append(
            f"Optimal chunk size for memory efficiency: {optimal_chunk_size}"
        )

    token_results = [
        r for r in results if r.test_category == "token_performance" and r.success
    ]
    if token_results:
        avg_token_rate = sum(
            r.processing_rate_rows_per_second for r in token_results
        ) / len(token_results)
        if avg_token_rate > 10000:
            recommendations.append(
                "Token estimation performance is excellent - no optimization needed"
            )
        else:
            recommendations.append(
                "Token estimation could benefit from caching or sampling optimization"
            )

    return recommendations


def generate_reports(
    config: BenchmarkConfig,
    results: List[BenchmarkResult],
    comparisons: List[ComparisonResult],
) -> None:
    """Generate human-readable performance reports.

    Args:
        config: Benchmark configuration with results directory.
        results: All benchmark results.
        comparisons: All comparison results.
    """
    logger.info("Generating performance reports")

    report_file = config.results_dir / f"performance_report_{int(time.time())}.md"

    with open(report_file, "w") as f:
        f.write("# LocalData MCP Performance Benchmark Report\n\n")
        f.write(f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Summary Statistics\n\n")
        f.write(f"- Total tests executed: {len(results)}\n")
        f.write(f"- Successful tests: {sum(1 for r in results if r.success)}\n")
        f.write(f"- Success rate: {calculate_success_rate(results):.1%}\n\n")

        f.write("## Performance by Category\n\n")

        categories = set(r.test_category for r in results if r.success)
        for category in categories:
            f.write(f"### {category.title()}\n\n")

            category_results = [
                r for r in results if r.test_category == category and r.success
            ]
            if category_results:
                avg_rate = sum(
                    r.processing_rate_rows_per_second for r in category_results
                ) / len(category_results)
                avg_memory = sum(r.peak_memory_mb for r in category_results) / len(
                    category_results
                )

                f.write(f"- Average processing rate: {avg_rate:.0f} rows/second\n")
                f.write(f"- Average peak memory usage: {avg_memory:.1f} MB\n")
                f.write(f"- Tests in category: {len(category_results)}\n\n")

        if comparisons:
            f.write("## Performance Comparisons\n\n")
            for comp in comparisons:
                f.write(f"### {comp.summary}\n\n")
                f.write(
                    f"- Execution time improvement: "
                    f"{comp.execution_time_improvement_percent:.1f}%\n"
                )
                f.write(
                    f"- Memory improvement: {comp.memory_improvement_percent:.1f}%\n"
                )
                f.write(
                    f"- Processing rate improvement: "
                    f"{comp.processing_rate_improvement_percent:.1f}%\n"
                )
                f.write(f"- Statistically significant: {comp.is_significant}\n\n")

        f.write("## Recommendations\n\n")
        recommendations = generate_recommendations(results)
        for rec in recommendations:
            f.write(f"- {rec}\n")

    logger.info(f"Performance report generated: {report_file}")


def generate_summary(
    results: List[BenchmarkResult],
    comparisons: List[ComparisonResult],
    config: BenchmarkConfig,
) -> Dict[str, Any]:
    """Generate summary of all benchmark results.

    Args:
        results: All benchmark results.
        comparisons: All comparison results.
        config: Benchmark configuration.

    Returns:
        Summary dictionary with aggregate metrics.
    """
    successful_results = [r for r in results if r.success]

    if not successful_results:
        return {
            "total_tests": len(results),
            "successful_tests": 0,
            "success_rate": 0.0,
            "error": "No successful test results",
        }

    return {
        "total_tests": len(results),
        "successful_tests": len(successful_results),
        "success_rate": len(successful_results) / len(results),
        "average_processing_rate": sum(
            r.processing_rate_rows_per_second for r in successful_results
        )
        / len(successful_results),
        "average_memory_usage_mb": sum(r.peak_memory_mb for r in successful_results)
        / len(successful_results),
        "test_categories": list(set(r.test_category for r in successful_results)),
        "performance_improvements": len(
            [c for c in comparisons if c.execution_time_improvement_percent > 5]
        ),
        "performance_regressions": len(
            [c for c in comparisons if c.execution_time_improvement_percent < -5]
        ),
        "results_directory": str(config.results_dir),
        "timestamp": time.time(),
    }
