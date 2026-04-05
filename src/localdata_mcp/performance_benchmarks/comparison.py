"""Benchmark comparison and regression testing logic.

This module provides functions for comparing benchmark results against
baselines, detecting regressions, and generating cross-approach comparisons
(e.g., streaming vs batch).
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

from ..logging_manager import get_logger
from .models import BenchmarkConfig, BenchmarkResult, ComparisonResult

logger = get_logger(__name__)


def find_matching_baseline(
    current_result: BenchmarkResult, baseline_results: List[BenchmarkResult]
) -> Optional[BenchmarkResult]:
    """Find matching baseline result for comparison."""
    for baseline in baseline_results:
        if (
            baseline.test_name == current_result.test_name
            and baseline.dataset_size == current_result.dataset_size
            and baseline.processing_mode == current_result.processing_mode
        ):
            return baseline
    return None


def compare_results(
    baseline: BenchmarkResult, current: BenchmarkResult
) -> ComparisonResult:
    """Compare two benchmark results."""
    execution_improvement = (
        (baseline.execution_time_seconds - current.execution_time_seconds)
        / baseline.execution_time_seconds
        * 100
    )
    memory_improvement = (
        (baseline.peak_memory_mb - current.peak_memory_mb)
        / baseline.peak_memory_mb
        * 100
    )
    rate_improvement = (
        (
            current.processing_rate_rows_per_second
            - baseline.processing_rate_rows_per_second
        )
        / baseline.processing_rate_rows_per_second
        * 100
    )

    is_significant = abs(execution_improvement) > 5

    if execution_improvement > 5:
        summary = f"Performance improved by {execution_improvement:.1f}%"
    elif execution_improvement < -5:
        summary = f"Performance regressed by {abs(execution_improvement):.1f}%"
    else:
        summary = "No significant performance change"

    return ComparisonResult(
        baseline_result=baseline,
        comparison_result=current,
        execution_time_improvement_percent=execution_improvement,
        memory_improvement_percent=memory_improvement,
        processing_rate_improvement_percent=rate_improvement,
        is_significant=is_significant,
        confidence_level=0.95,
        summary=summary,
    )


def run_regression_tests(
    config: BenchmarkConfig,
    results: List[BenchmarkResult],
    comparisons: List[ComparisonResult],
) -> None:
    """Run automated regression testing against baseline performance.

    Args:
        config: Benchmark configuration with results directory.
        results: Current benchmark results to compare.
        comparisons: List to append new comparisons to (mutated in place).
    """
    logger.info("Running regression tests")

    baseline_path = config.results_dir / "baseline_results.json"

    if not baseline_path.exists():
        logger.info("No baseline results found, current results will become baseline")
        return

    try:
        with open(baseline_path, "r") as f:
            baseline_data = json.load(f)

        baseline_results = [BenchmarkResult(**result) for result in baseline_data]

        for current_result in results:
            baseline_result = find_matching_baseline(current_result, baseline_results)

            if baseline_result and baseline_result.success and current_result.success:
                comparison = compare_results(baseline_result, current_result)
                comparisons.append(comparison)

                if comparison.execution_time_improvement_percent < -10:
                    logger.warning(
                        f"Performance regression detected in {current_result.test_name}: "
                        f"{abs(comparison.execution_time_improvement_percent):.1f}% slower"
                    )
                elif comparison.execution_time_improvement_percent > 10:
                    logger.info(
                        f"Performance improvement in {current_result.test_name}: "
                        f"{comparison.execution_time_improvement_percent:.1f}% faster"
                    )

    except Exception as e:
        logger.warning(f"Regression testing failed: {e}")


def generate_comparisons(
    results: List[BenchmarkResult],
    comparisons: List[ComparisonResult],
) -> None:
    """Generate performance comparisons between different approaches.

    Args:
        results: All benchmark results to group and compare.
        comparisons: List to append new comparisons to (mutated in place).
    """
    logger.info("Generating performance comparisons")

    scenarios: Dict[str, List[BenchmarkResult]] = {}
    for result in results:
        scenario = result.metadata.get("scenario", "unknown")
        if scenario not in scenarios:
            scenarios[scenario] = []
        scenarios[scenario].append(result)

    for scenario, scenario_results in scenarios.items():
        streaming_results = [
            r for r in scenario_results if r.processing_mode == "streaming"
        ]
        batch_results = [r for r in scenario_results if r.processing_mode == "batch"]

        if streaming_results and batch_results:
            best_streaming = max(
                streaming_results, key=lambda r: r.processing_rate_rows_per_second
            )
            best_batch = max(
                batch_results, key=lambda r: r.processing_rate_rows_per_second
            )

            comparison = compare_results(best_batch, best_streaming)
            comparison.summary = f"Streaming vs Batch comparison for {scenario}"
            comparisons.append(comparison)
