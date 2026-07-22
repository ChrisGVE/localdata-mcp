"""localdata_mcp/testbench/results_store/store.py — parameterized store I/O.

Read/write path for the results provenance store (ARCHITECTURE.md
section 5). Every statement is parameterized — no string-interpolated
SQL, even in this trusted, non-LLM-facing test path. Writers go through
`write_transaction`, the bounded-retry wrapper that serializes write
bursts on top of the busy timeout schema.connect applies (WAL allows
one writer at a time). Neighbors: schema.py owns the DDL this module
writes through; merge.py builds on these primitives.
"""

from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterable, Iterator, List, Optional

_RETRY_BACKOFF_SECONDS = 0.05  # brief pause between write-lock attempts


@dataclass(frozen=True)
class BatteryRun:
    """One logical battery execution — exactly one row per coordinator run."""

    run_id: str
    battery_name: str
    run_mode: str  # 'deterministic' | 'randomized' (enforced by the schema)
    seed: Optional[int]
    dataset_hash: str
    software_versions: dict
    started_at: str  # ISO-8601 UTC
    finished_at: Optional[str]
    git_sha: Optional[str]


@dataclass(frozen=True)
class BatteryResult:
    """One test outcome within a run, unique per (run_id, test_id)."""

    run_id: str
    test_id: str
    passed: bool
    numeric_output: Optional[Any] = None  # JSON-serializable, nullable
    duration_ms: Optional[float] = None


@contextmanager
def write_transaction(
    connection: sqlite3.Connection, *, max_attempts: int
) -> Iterator[sqlite3.Connection]:
    """One immediate write transaction; lock acquisition retried, bounded."""
    if max_attempts < 1:
        raise ValueError(f"max_attempts must be >= 1, got {max_attempts}")
    _begin_immediate_with_retry(connection, max_attempts)
    try:
        yield connection
        connection.execute("COMMIT")
    except BaseException:
        connection.execute("ROLLBACK")
        raise


def _begin_immediate_with_retry(
    connection: sqlite3.Connection, max_attempts: int
) -> None:
    """Acquire the write lock, retrying a busy database a bounded number of times."""
    for attempt in range(1, max_attempts + 1):
        try:
            connection.execute("BEGIN IMMEDIATE")
            return
        except sqlite3.OperationalError as error:
            if "locked" not in str(error) and "busy" not in str(error):
                raise
            if attempt == max_attempts:
                raise
            time.sleep(_RETRY_BACKOFF_SECONDS * attempt)


def write_run(connection: sqlite3.Connection, run: BatteryRun) -> None:
    """Insert one battery_runs row (coordinator-minted run_id)."""
    connection.execute(
        "INSERT INTO battery_runs (run_id, battery_name, run_mode, seed,"
        " dataset_hash, software_versions, started_at, finished_at, git_sha)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            run.run_id,
            run.battery_name,
            run.run_mode,
            run.seed,
            run.dataset_hash,
            json.dumps(run.software_versions, sort_keys=True),
            run.started_at,
            run.finished_at,
            run.git_sha,
        ),
    )


def write_results(
    connection: sqlite3.Connection, results: Iterable[BatteryResult]
) -> None:
    """Insert battery_results rows; duplicates of (run_id, test_id) refuse."""
    connection.executemany(
        "INSERT INTO battery_results"
        " (run_id, test_id, pass, numeric_output, duration_ms)"
        " VALUES (?, ?, ?, ?, ?)",
        (
            (
                result.run_id,
                result.test_id,
                int(result.passed),
                None
                if result.numeric_output is None
                else json.dumps(result.numeric_output, sort_keys=True),
                result.duration_ms,
            )
            for result in results
        ),
    )


def read_run(connection: sqlite3.Connection, run_id: str) -> Optional[BatteryRun]:
    """Fetch one run by id, or None."""
    row = connection.execute(
        "SELECT run_id, battery_name, run_mode, seed, dataset_hash,"
        " software_versions, started_at, finished_at, git_sha"
        " FROM battery_runs WHERE run_id = ?",
        (run_id,),
    ).fetchone()
    if row is None:
        return None
    return BatteryRun(
        run_id=row[0],
        battery_name=row[1],
        run_mode=row[2],
        seed=row[3],
        dataset_hash=row[4],
        software_versions=json.loads(row[5]),
        started_at=row[6],
        finished_at=row[7],
        git_sha=row[8],
    )


def read_results(connection: sqlite3.Connection, run_id: str) -> List[BatteryResult]:
    """Fetch all results of one run, ordered by test_id."""
    rows = connection.execute(
        "SELECT run_id, test_id, pass, numeric_output, duration_ms"
        " FROM battery_results WHERE run_id = ? ORDER BY test_id",
        (run_id,),
    ).fetchall()
    return [
        BatteryResult(
            run_id=row[0],
            test_id=row[1],
            passed=bool(row[2]),
            numeric_output=None if row[3] is None else json.loads(row[3]),
            duration_ms=row[4],
        )
        for row in rows
    ]
