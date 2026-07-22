"""tests/v3/testbench/test_merge.py — merge.py union, refusals, retention.

Covers: per-worker merge into the canonical store, idempotent re-merge
(insert-or-ignore under the (run_id, test_id) key), schema-version
mismatch refusal, structurally-incomplete refusal, and the post-merge
retention prune leaving zero orphaned battery_results rows.
"""

import sqlite3
from pathlib import Path
from sqlite3 import Connection
from typing import List

import pytest

from localdata_mcp.testbench.results_store import merge, schema, store

from .conftest import (
    BUSY_TIMEOUT_MS,
    MAX_WRITE_ATTEMPTS,
    make_results,
    make_run,
    make_worker_file,
    open_store,
)

RETENTION = 2  # explicit test value; the config home arrives with E1


def merge_file(canonical: Connection, worker_path: Path) -> None:
    merge.merge_worker_file(
        canonical,
        worker_path,
        retention_runs=RETENTION,
        max_write_attempts=MAX_WRITE_ATTEMPTS,
    )


def count_rows(connection: Connection, table: str) -> int:
    return int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])


def orphaned_results(connection: Connection) -> List[str]:
    """battery_results rows whose parent battery_runs row is gone."""
    rows = connection.execute(
        "SELECT r.test_id FROM battery_results AS r"
        " LEFT JOIN battery_runs AS b ON b.run_id = r.run_id"
        " WHERE b.run_id IS NULL"
    ).fetchall()
    return [row[0] for row in rows]


class TestUnionAndIdempotence:
    def test_worker_results_land_in_canonical(
        self, canonical: Connection, tmp_path: Path
    ) -> None:
        run = make_run()
        results = make_results(run.run_id, count=3)
        worker = make_worker_file(tmp_path / "worker-1.db", run, results)
        merge_file(canonical, worker)
        assert store.read_run(canonical, run.run_id) == run
        assert len(store.read_results(canonical, run.run_id)) == 3

    def test_remerge_is_idempotent(self, canonical: Connection, tmp_path: Path) -> None:
        run = make_run()
        worker = make_worker_file(
            tmp_path / "worker-1.db", run, make_results(run.run_id, count=3)
        )
        merge_file(canonical, worker)
        merge_file(canonical, worker)  # a normal CI retry
        assert count_rows(canonical, "battery_runs") == 1
        assert count_rows(canonical, "battery_results") == 3

    def test_two_workers_one_logical_run(
        self, canonical: Connection, tmp_path: Path
    ) -> None:
        """Both workers carry the coordinator-minted run id; one run row."""
        run = make_run()
        first = make_results(run.run_id, count=2)
        second = [
            store.BatteryResult(run.run_id, "statistical.sleep.anova.case_9", True)
        ]
        merge_file(canonical, make_worker_file(tmp_path / "w1.db", run, first))
        merge_file(canonical, make_worker_file(tmp_path / "w2.db", run, second))
        assert count_rows(canonical, "battery_runs") == 1
        assert count_rows(canonical, "battery_results") == 3


class TestRefusals:
    def test_schema_version_mismatch_is_refused(
        self, canonical: Connection, tmp_path: Path
    ) -> None:
        run = make_run()
        worker = make_worker_file(
            tmp_path / "worker-1.db", run, make_results(run.run_id, count=1)
        )
        tamper = sqlite3.connect(worker)
        tamper.execute("UPDATE meta SET schema_version = 999")
        tamper.commit()
        tamper.close()
        with pytest.raises(merge.MergeRefusedError, match="schema_version"):
            merge_file(canonical, worker)

    def test_structurally_incomplete_file_is_refused(
        self, canonical: Connection, tmp_path: Path
    ) -> None:
        broken_path = tmp_path / "broken.db"
        connection = open_store(broken_path)
        connection.execute("DROP TABLE battery_results")
        connection.commit()
        connection.close()
        with pytest.raises(merge.MergeRefusedError, match="battery_results"):
            merge_file(canonical, broken_path)

    def test_refused_merge_leaves_canonical_untouched(
        self, canonical: Connection, tmp_path: Path
    ) -> None:
        broken_path = tmp_path / "broken.db"
        connection = open_store(broken_path)
        connection.execute("DROP TABLE battery_results")
        connection.commit()
        connection.close()
        with pytest.raises(merge.MergeRefusedError):
            merge_file(canonical, broken_path)
        assert count_rows(canonical, "battery_runs") == 0

    def test_retention_must_be_positive(
        self, canonical: Connection, tmp_path: Path
    ) -> None:
        run = make_run()
        worker = make_worker_file(
            tmp_path / "w.db", run, make_results(run.run_id, count=1)
        )
        with pytest.raises(ValueError):
            merge.merge_worker_file(
                canonical,
                worker,
                retention_runs=0,
                max_write_attempts=MAX_WRITE_ATTEMPTS,
            )


class TestRetentionPrune:
    def test_prune_keeps_most_recent_per_battery_and_mode(
        self, canonical: Connection, tmp_path: Path
    ) -> None:
        for index in range(RETENTION + 3):
            run = make_run(index=index)
            worker = make_worker_file(
                tmp_path / f"w{index}.db", run, make_results(run.run_id, count=2)
            )
            merge_file(canonical, worker)
        kept = [
            row[0]
            for row in canonical.execute(
                "SELECT run_id FROM battery_runs ORDER BY started_at"
            )
        ]
        expected = [make_run(index=i).run_id for i in (3, 4)]
        assert kept == expected

    def test_prune_leaves_zero_orphaned_results(
        self, canonical: Connection, tmp_path: Path
    ) -> None:
        for index in range(RETENTION + 4):
            run = make_run(index=index)
            worker = make_worker_file(
                tmp_path / f"w{index}.db", run, make_results(run.run_id, count=3)
            )
            merge_file(canonical, worker)
        assert orphaned_results(canonical) == []
        assert count_rows(canonical, "battery_results") == RETENTION * 3

    def test_prune_is_scoped_per_battery_and_mode(
        self, canonical: Connection, tmp_path: Path
    ) -> None:
        """Another (battery_name, run_mode) group keeps its own budget."""
        for index in range(RETENTION + 1):
            run = make_run(index=index)
            merge_file(
                canonical,
                make_worker_file(
                    tmp_path / f"a{index}.db", run, make_results(run.run_id, 1)
                ),
            )
        other = make_run(index=0, battery_name="security", run_mode="randomized")
        merge_file(
            canonical,
            make_worker_file(tmp_path / "b0.db", other, make_results(other.run_id, 1)),
        )
        assert store.read_run(canonical, other.run_id) == other
        assert count_rows(canonical, "battery_runs") == RETENTION + 1
