"""tests/v3/testbench/test_emit.py — RunCollector per-worker persistence.

Covers: grouping by (battery_name, run_mode) into one battery_runs row
apiece, seed recorded only for the randomized mode, numeric outputs and
durations round-tripped, the written file being a valid mergeable store,
and software_versions carrying the interpreter version.
"""

from pathlib import Path
from sqlite3 import Connection

from localdata_mcp.testbench.results_store import emit, merge, store

from .conftest import BUSY_TIMEOUT_MS, MAX_WRITE_ATTEMPTS, open_store

RETENTION = (
    2  # explicit test value; the config home is testbench.results_retention_runs
)


def _collector() -> emit.RunCollector:
    return emit.RunCollector(run_id_base="ci-42")


def _persist(collector: emit.RunCollector, path: Path) -> Path:
    return collector.persist(
        path,
        dataset_hash="sha256:fixtures",
        git_sha="deadbeef",
        busy_timeout_ms=BUSY_TIMEOUT_MS,
        max_write_attempts=MAX_WRITE_ATTEMPTS,
    )


class TestGroupingAndProvenance:
    def test_one_run_row_per_battery_mode_group(self, tmp_path: Path) -> None:
        collector = _collector()
        collector.record(
            battery_name="base",
            run_mode="deterministic",
            test_id="base.kv.roundtrip",
            passed=True,
        )
        collector.record(
            battery_name="base",
            run_mode="randomized",
            test_id="base.kv.roundtrip",
            passed=True,
        )
        collector.record(
            battery_name="security",
            run_mode="deterministic",
            test_id="security.path.escape",
            passed=True,
        )
        worker = _persist(collector, tmp_path / "worker.db")

        connection = open_store(worker)
        run_ids = [
            row[0]
            for row in connection.execute(
                "SELECT run_id FROM battery_runs ORDER BY run_id"
            )
        ]
        connection.close()
        assert run_ids == [
            "ci-42-base-deterministic",
            "ci-42-base-randomized",
            "ci-42-security-deterministic",
        ]

    def test_seed_only_recorded_for_randomized(self, tmp_path: Path) -> None:
        collector = _collector()
        collector.record(
            battery_name="base",
            run_mode="deterministic",
            test_id="d",
            passed=True,
        )
        collector.record(
            battery_name="base", run_mode="randomized", test_id="r", passed=True
        )
        worker = _persist(collector, tmp_path / "worker.db")
        connection = open_store(worker)
        seeds = dict(
            connection.execute("SELECT run_mode, seed FROM battery_runs").fetchall()
        )
        connection.close()
        assert seeds["deterministic"] is None
        assert seeds["randomized"] == 42

    def test_numeric_output_and_duration_round_trip(self, tmp_path: Path) -> None:
        collector = _collector()
        collector.record(
            battery_name="pipeline",
            run_mode="deterministic",
            test_id="pipeline.a->b",
            passed=True,
            numeric_output={"legal": True, "engine_rejected": False},
            duration_ms=12.5,
        )
        worker = _persist(collector, tmp_path / "worker.db")
        connection = open_store(worker)
        results = store.read_results(connection, "ci-42-pipeline-deterministic")
        connection.close()
        assert len(results) == 1
        assert results[0].numeric_output == {"legal": True, "engine_rejected": False}
        assert results[0].duration_ms == 12.5

    def test_software_versions_carries_interpreter(self) -> None:
        versions = emit.software_versions()
        assert "python" in versions
        assert versions["python"].count(".") >= 2


class TestWorkerFileMerges:
    def test_written_file_merges_into_canonical(
        self, canonical: Connection, tmp_path: Path
    ) -> None:
        collector = _collector()
        collector.record(
            battery_name="domain",
            run_mode="deterministic",
            test_id="domain.stats.anova",
            passed=False,
        )
        worker = _persist(collector, tmp_path / "worker.db")
        merge.merge_worker_file(
            canonical,
            worker,
            retention_runs=RETENTION,
            max_write_attempts=MAX_WRITE_ATTEMPTS,
        )
        results = store.read_results(canonical, "ci-42-domain-deterministic")
        assert len(results) == 1
        assert results[0].passed is False

    def test_empty_collector_reports_empty(self) -> None:
        assert _collector().is_empty()
