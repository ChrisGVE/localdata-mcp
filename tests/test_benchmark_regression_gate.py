"""Tests for the benchmark regression gate.

The CI gate fails the build when a benchmark run reports more regressions
than improvements. A regression means "this build is slower than the
recorded baseline" — it must never mean "streaming is slower than batch",
because streaming is *designed* to trade throughput for bounded memory
(project First Principle 4, streaming-first). These tests pin that
distinction: baseline comparisons feed the pass/fail counts, and
streaming-vs-batch comparisons are reported as information only.
"""

import json

import pytest

from localdata_mcp.performance_benchmarks.comparison import (
    compare_results,
    generate_mode_comparisons,
    run_regression_tests,
)
from localdata_mcp.performance_benchmarks.models import (
    BenchmarkConfig,
    BenchmarkResult,
)
from localdata_mcp.performance_benchmarks.reporting import (
    generate_summary,
    save_detailed_results,
)


def make_result(
    *,
    test_name: str = "scan",
    processing_mode: str = "batch",
    execution_time_seconds: float = 1.0,
    scenario: str = "csv_scan",
    dataset_size: str = "small",
    success: bool = True,
) -> BenchmarkResult:
    """Build a benchmark result with only the fields these tests care about."""
    return BenchmarkResult(
        test_name=test_name,
        test_category="streaming_performance",
        dataset_size=dataset_size,
        database_type="sqlite",
        processing_mode=processing_mode,
        execution_time_seconds=execution_time_seconds,
        peak_memory_mb=100.0,
        average_memory_mb=80.0,
        memory_efficiency=0.8,
        rows_processed=1000,
        processing_rate_rows_per_second=1000.0 / execution_time_seconds,
        chunk_count=10,
        average_chunk_size=100,
        success=success,
        metadata={"scenario": scenario},
    )


@pytest.fixture
def config(tmp_path) -> BenchmarkConfig:
    return BenchmarkConfig(results_dir=tmp_path / "benchmark_results")


class TestModeComparisonsAreInformational:
    """Streaming-vs-batch must never be counted as a regression."""

    def test_slower_streaming_does_not_count_as_regression(self):
        batch = make_result(processing_mode="batch", execution_time_seconds=1.0)
        streaming = make_result(processing_mode="streaming", execution_time_seconds=2.0)
        results = [batch, streaming]

        mode_comparisons = []
        generate_mode_comparisons(results, mode_comparisons)
        assert len(mode_comparisons) == 1, "expected one streaming-vs-batch comparison"
        assert mode_comparisons[0].execution_time_improvement_percent < -5, (
            "precondition: streaming is more than 5% slower than batch"
        )

        summary = generate_summary(
            results, [], BenchmarkConfig(), mode_comparisons=mode_comparisons
        )

        assert summary["performance_regressions"] == 0
        assert summary["performance_improvements"] == 0

    def test_mode_comparisons_are_reported_as_information(self):
        results = [
            make_result(processing_mode="batch", execution_time_seconds=1.0),
            make_result(processing_mode="streaming", execution_time_seconds=2.0),
        ]
        mode_comparisons = []
        generate_mode_comparisons(results, mode_comparisons)

        summary = generate_summary(
            results, [], BenchmarkConfig(), mode_comparisons=mode_comparisons
        )

        assert summary["streaming_vs_batch"], "mode comparisons must be surfaced"
        entry = summary["streaming_vs_batch"][0]
        assert entry["scenario"] == "csv_scan"
        assert entry["streaming_slower_percent"] == pytest.approx(100.0, rel=1e-3)

    def test_mode_comparisons_do_not_land_in_the_baseline_list(self):
        results = [
            make_result(processing_mode="batch", execution_time_seconds=1.0),
            make_result(processing_mode="streaming", execution_time_seconds=2.0),
        ]
        baseline_comparisons = []
        mode_comparisons = []

        generate_mode_comparisons(results, mode_comparisons)

        assert baseline_comparisons == []
        assert len(mode_comparisons) == 1


class TestBaselineComparisonsDriveTheGate:
    """Only baseline comparisons decide pass/fail."""

    def test_regression_against_baseline_is_counted(self):
        baseline = make_result(execution_time_seconds=1.0)
        current = make_result(execution_time_seconds=2.0)

        comparisons = [compare_results(baseline, current)]
        summary = generate_summary([current], comparisons, BenchmarkConfig())

        assert summary["performance_regressions"] == 1
        assert summary["performance_improvements"] == 0

    def test_improvement_against_baseline_is_counted(self):
        baseline = make_result(execution_time_seconds=2.0)
        current = make_result(execution_time_seconds=1.0)

        comparisons = [compare_results(baseline, current)]
        summary = generate_summary([current], comparisons, BenchmarkConfig())

        assert summary["performance_improvements"] == 1
        assert summary["performance_regressions"] == 0

    def test_summary_reports_whether_a_baseline_was_available(self):
        current = make_result()

        without = generate_summary([current], [], BenchmarkConfig())
        assert without["baseline_comparisons"] == 0

        with_baseline = generate_summary(
            [current],
            [compare_results(make_result(execution_time_seconds=1.0), current)],
            BenchmarkConfig(),
        )
        assert with_baseline["baseline_comparisons"] == 1


class TestBaselinePersistence:
    """A baseline the run itself wrote must not be compared against itself."""

    def test_regression_tests_use_a_restored_baseline(self, config):
        config.results_dir.mkdir(parents=True, exist_ok=True)
        baseline = make_result(execution_time_seconds=1.0)
        (config.results_dir / "baseline_results.json").write_text(
            json.dumps([baseline.to_dict()], default=str)
        )

        current = make_result(execution_time_seconds=2.0)
        comparisons = []
        run_regression_tests(config, [current], comparisons)

        assert len(comparisons) == 1
        assert comparisons[0].execution_time_improvement_percent == pytest.approx(
            -100.0
        )

    def test_first_run_writes_a_baseline_for_the_next_run(self, config):
        config.results_dir.mkdir(parents=True, exist_ok=True)
        current = make_result()

        save_detailed_results(config, [current], [], [])

        baseline_path = config.results_dir / "baseline_results.json"
        assert baseline_path.exists()
        stored = json.loads(baseline_path.read_text())
        assert stored[0]["test_name"] == current.test_name

    def test_an_existing_baseline_is_not_overwritten(self, config):
        config.results_dir.mkdir(parents=True, exist_ok=True)
        baseline_path = config.results_dir / "baseline_results.json"
        original = make_result(test_name="original")
        baseline_path.write_text(json.dumps([original.to_dict()], default=str))

        save_detailed_results(config, [make_result(test_name="newer")], [], [])

        stored = json.loads(baseline_path.read_text())
        assert stored[0]["test_name"] == "original"
