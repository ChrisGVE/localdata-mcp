"""localdata_mcp/testbench/results_store/merge.py — per-worker file merge.

Merges one per-worker results file into the canonical store
(ARCHITECTURE.md section 5): CI battery workers on isolated jobs each
write their own SQLite file, and a post-job step folds them in here.
The merge refuses on schema-version mismatch or a structurally
incomplete file, unions battery_results under the (run_id, test_id)
unique key with insert-or-ignore semantics (a re-run of the post-job
step is idempotent), keeps exactly one battery_runs row per
coordinator-minted run_id, and applies the retention prune in the SAME
transaction. Neighbors: schema.py owns versioning; store.py provides
the bounded-retry write transaction used here.

`retention_runs` (most-recent runs kept per (battery_name, run_mode))
is a required parameter: its configuration home is the config nexus's
testbench section, wired in E1 — this module declares no default.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from . import schema, store

_REQUIRED_TABLES = frozenset({"meta", "battery_runs", "battery_results"})


class MergeRefusedError(Exception):
    """The worker file cannot be merged safely; nothing was written."""


def merge_worker_file(
    canonical: sqlite3.Connection,
    worker_path: Path | str,
    *,
    retention_runs: int,
    max_write_attempts: int,
) -> None:
    """Fold one worker file into the canonical store, then prune, atomically."""
    if retention_runs < 1:
        raise ValueError(f"retention_runs must be >= 1, got {retention_runs}")
    canonical.execute("ATTACH DATABASE ? AS worker", (str(worker_path),))
    try:
        _assert_mergeable(canonical)
        with store.write_transaction(canonical, max_attempts=max_write_attempts):
            _union_worker_rows(canonical)
            prune_stale_runs(canonical, retention_runs=retention_runs)
    finally:
        canonical.execute("DETACH DATABASE worker")


def _assert_mergeable(canonical: sqlite3.Connection) -> None:
    """Refuse a structurally incomplete or version-mismatched worker file."""
    tables = {
        row[0]
        for row in canonical.execute(
            "SELECT name FROM worker.sqlite_master WHERE type = 'table'"
        )
    }
    missing = _REQUIRED_TABLES - tables
    if missing:
        raise MergeRefusedError(
            f"worker file is structurally incomplete: missing {sorted(missing)}"
        )
    versions = canonical.execute("SELECT schema_version FROM worker.meta").fetchall()
    if len(versions) != 1:
        raise MergeRefusedError(
            f"worker meta must carry exactly one row, found {len(versions)}"
        )
    worker_version = int(versions[0][0])
    canonical_version = schema.schema_version(canonical)
    if worker_version != canonical_version:
        raise MergeRefusedError(
            f"schema_version mismatch: worker file is at {worker_version},"
            f" canonical store at {canonical_version}"
        )


def _union_worker_rows(canonical: sqlite3.Connection) -> None:
    """Union runs then results; the (run_id, test_id) key makes it idempotent."""
    # Workers carry the coordinator-minted run row so their own FK holds;
    # OR IGNORE keeps the canonical store at one row per logical run.
    canonical.execute(
        "INSERT OR IGNORE INTO battery_runs"
        " (run_id, battery_name, run_mode, seed, dataset_hash,"
        "  software_versions, started_at, finished_at, git_sha)"
        " SELECT run_id, battery_name, run_mode, seed, dataset_hash,"
        "  software_versions, started_at, finished_at, git_sha"
        " FROM worker.battery_runs"
    )
    canonical.execute(
        "INSERT OR IGNORE INTO battery_results"
        " (run_id, test_id, pass, numeric_output, duration_ms)"
        " SELECT run_id, test_id, pass, numeric_output, duration_ms"
        " FROM worker.battery_results"
    )


def prune_stale_runs(canonical: sqlite3.Connection, *, retention_runs: int) -> None:
    """Keep the most-recent `retention_runs` runs per (battery_name, run_mode).

    Deleting a battery_runs row cascades to its battery_results children
    (schema.py declares ON DELETE CASCADE; connections enforce foreign
    keys), so no orphaned child row can survive a prune.
    """
    canonical.execute(
        """
        DELETE FROM battery_runs WHERE run_id IN (
            SELECT run_id FROM (
                SELECT run_id,
                       ROW_NUMBER() OVER (
                           PARTITION BY battery_name, run_mode
                           ORDER BY started_at DESC, run_id DESC
                       ) AS recency_rank
                FROM battery_runs
            )
            WHERE recency_rank > ?
        )
        """,
        (retention_runs,),
    )
