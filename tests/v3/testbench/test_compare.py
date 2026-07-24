"""tests/v3/testbench/test_compare.py — NFR-508 longitudinal comparison.

Covers latest_runs recency ordering and diff_runs / diff_latest
classification: regressions, fixes, added/removed tests, numeric-output
changes, and the fewer-than-two-runs None case.
"""

from sqlite3 import Connection
from typing import List, Optional

from localdata_mcp.testbench.results_store import compare, store

from .conftest import MAX_WRITE_ATTEMPTS


def _run(
    connection: Connection,
    run_id: str,
    *,
    started_at: str,
    outcomes: List[tuple],
    battery: str = "base",
    mode: str = "deterministic",
) -> None:
    run = store.BatteryRun(
        run_id=run_id,
        battery_name=battery,
        run_mode=mode,
        seed=None,
        dataset_hash="sha256:fx",
        software_versions={"python": "3.12"},
        started_at=started_at,
        finished_at=started_at,
        git_sha="abc",
    )
    results = [
        store.BatteryResult(run_id, test_id, passed=passed, numeric_output=numeric)
        for (test_id, passed, numeric) in outcomes
    ]
    with store.write_transaction(connection, max_attempts=MAX_WRITE_ATTEMPTS):
        store.write_run(connection, run)
        store.write_results(connection, results)


class TestLatestRuns:
    def test_returns_newest_first(self, canonical: Connection) -> None:
        _run(
            canonical,
            "r0",
            started_at="2026-07-22T10:00:00Z",
            outcomes=[("t", True, None)],
        )
        _run(
            canonical,
            "r1",
            started_at="2026-07-22T11:00:00Z",
            outcomes=[("t", True, None)],
        )
        _run(
            canonical,
            "r2",
            started_at="2026-07-22T12:00:00Z",
            outcomes=[("t", True, None)],
        )
        assert compare.latest_runs(
            canonical, battery_name="base", run_mode="deterministic"
        ) == ["r2", "r1"]


class TestDiffRuns:
    def test_classifies_every_change(self, canonical: Connection) -> None:
        _run(
            canonical,
            "base",
            started_at="2026-07-22T10:00:00Z",
            outcomes=[
                ("stable", True, {"v": 1}),
                ("regressed", True, None),
                ("fixed", False, None),
                ("gone", True, None),
                ("num", True, {"v": 1}),
            ],
        )
        _run(
            canonical,
            "head",
            started_at="2026-07-22T11:00:00Z",
            outcomes=[
                ("stable", True, {"v": 1}),
                ("regressed", False, None),
                ("fixed", True, None),
                ("new", True, None),
                ("num", True, {"v": 2}),
            ],
        )
        diff = compare.diff_runs(canonical, base_run_id="base", head_run_id="head")
        assert diff.regressions == ["regressed"]
        assert diff.fixes == ["fixed"]
        assert diff.added == ["new"]
        assert diff.removed == ["gone"]
        assert diff.numeric_changes == ["num"]
        assert diff.has_regressions()


class TestDiffLatest:
    def test_diffs_two_most_recent(self, canonical: Connection) -> None:
        _run(
            canonical,
            "old",
            started_at="2026-07-22T10:00:00Z",
            outcomes=[("t", True, None)],
        )
        _run(
            canonical,
            "new",
            started_at="2026-07-22T11:00:00Z",
            outcomes=[("t", False, None)],
        )
        diff: Optional[compare.RunDiff] = compare.diff_latest(
            canonical, battery_name="base", run_mode="deterministic"
        )
        assert diff is not None
        assert diff.head_run_id == "new"
        assert diff.regressions == ["t"]

    def test_single_run_is_none(self, canonical: Connection) -> None:
        _run(
            canonical,
            "only",
            started_at="2026-07-22T10:00:00Z",
            outcomes=[("t", True, None)],
        )
        assert (
            compare.diff_latest(
                canonical, battery_name="base", run_mode="deterministic"
            )
            is None
        )
