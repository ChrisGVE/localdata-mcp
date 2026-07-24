"""localdata_mcp/testbench/results_store/compare.py — NFR-508 longitudinal diff.

The read side of the results store's purpose: given two runs of one
battery, show what changed. `latest_runs` finds the most-recent runs of a
(battery_name, run_mode) group (the recency index schema.py declares
serves this); `diff_runs` classifies every test_id across the two runs
into regressions (was passing, now failing), fixes (the reverse), added
and removed tests, and numeric-output changes. A regression is the
signal the nightly comparison surfaces (NFR-508: "querying two runs shows
a diffable comparison"). Read-only; neighbors: store.py supplies the
run/result rows this reads.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Dict, List, Optional

from . import store


def latest_runs(
    connection: sqlite3.Connection,
    *,
    battery_name: str,
    run_mode: str,
    limit: int = 2,
) -> List[str]:
    """The `limit` most-recent run ids of one group, newest first."""
    if limit < 1:
        raise ValueError(f"limit must be >= 1, got {limit}")
    rows = connection.execute(
        "SELECT run_id FROM battery_runs"
        " WHERE battery_name = ? AND run_mode = ?"
        " ORDER BY started_at DESC, run_id DESC LIMIT ?",
        (battery_name, run_mode, limit),
    ).fetchall()
    return [row[0] for row in rows]


@dataclass(frozen=True)
class RunDiff:
    """The classified difference between an older and a newer run."""

    base_run_id: str
    head_run_id: str
    regressions: List[str]  # passing in base, failing in head
    fixes: List[str]  # failing in base, passing in head
    added: List[str]  # present only in head
    removed: List[str]  # present only in base
    numeric_changes: List[str]  # present in both, numeric_output differs

    def has_regressions(self) -> bool:
        return bool(self.regressions)


def _outcomes(
    connection: sqlite3.Connection, run_id: str
) -> Dict[str, store.BatteryResult]:
    return {r.test_id: r for r in store.read_results(connection, run_id)}


def diff_runs(
    connection: sqlite3.Connection, *, base_run_id: str, head_run_id: str
) -> RunDiff:
    """Classify every test_id's change between the base and head runs."""
    base = _outcomes(connection, base_run_id)
    head = _outcomes(connection, head_run_id)
    shared = base.keys() & head.keys()

    regressions = sorted(t for t in shared if base[t].passed and not head[t].passed)
    fixes = sorted(t for t in shared if not base[t].passed and head[t].passed)
    numeric_changes = sorted(
        t for t in shared if base[t].numeric_output != head[t].numeric_output
    )
    return RunDiff(
        base_run_id=base_run_id,
        head_run_id=head_run_id,
        regressions=regressions,
        fixes=fixes,
        added=sorted(head.keys() - base.keys()),
        removed=sorted(base.keys() - head.keys()),
        numeric_changes=numeric_changes,
    )


def diff_latest(
    connection: sqlite3.Connection, *, battery_name: str, run_mode: str
) -> Optional[RunDiff]:
    """Diff the two most-recent runs of a group; None if fewer than two."""
    runs = latest_runs(
        connection, battery_name=battery_name, run_mode=run_mode, limit=2
    )
    if len(runs) < 2:
        return None
    head_run_id, base_run_id = runs[0], runs[1]
    return diff_runs(connection, base_run_id=base_run_id, head_run_id=head_run_id)
