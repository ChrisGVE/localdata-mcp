"""localdata_mcp/testbench/results_store/trailer.py — NFR-506 trailer check.

The mechanically-decidable core of the batch-then-diagnose gate. A fix
commit touching bench-covered code carries a `Battery-Run:` trailer
citing the antecedent full-battery run in the results store; this module
parses that trailer and verifies against the store what a machine CAN
verify: the cited run exists, it PREDATES the commit, it recorded at
least one failure, and a NEWER run of the same battery — initiated after
the commit — is present (NFR-506's two-condition merge block). The one
clause a machine cannot decide — that the diff addresses that specific
failure — stays with the human reviewer; the store carries no
source-file-to-test-id map, and this module claims only what it checks.
Neighbors: store.py reads the runs/results this inspects; the CLI
wrapper (scripts/check_battery_run_trailer.py) supplies git metadata.
"""

from __future__ import annotations

import re
import sqlite3
from typing import List, Optional

from . import store

# Git trailer form `Battery-Run: <run-id>` — the linking convention
# NFR-506 fixes so a CI job can parse the reference from the message.
_TRAILER = re.compile(r"^Battery-Run:[ \t]*(\S+)[ \t]*$", re.MULTILINE)


def parse_trailer(commit_message: str) -> Optional[str]:
    """The cited run id from the LAST `Battery-Run:` trailer, or None."""
    matches = _TRAILER.findall(commit_message)
    return matches[-1] if matches else None


def _failure_count(connection: sqlite3.Connection, run_id: str) -> int:
    """How many results of a run failed (pass = 0)."""
    row = connection.execute(
        "SELECT COUNT(*) FROM battery_results WHERE run_id = ? AND pass = 0",
        (run_id,),
    ).fetchone()
    return int(row[0])


def _newer_run_exists(
    connection: sqlite3.Connection, *, battery_name: str, after_iso: str
) -> bool:
    """A later run of the same battery, started strictly after the commit."""
    row = connection.execute(
        "SELECT 1 FROM battery_runs WHERE battery_name = ? AND started_at > ? LIMIT 1",
        (battery_name, after_iso),
    ).fetchone()
    return row is not None


def verify(
    connection: sqlite3.Connection, *, cited_run_id: str, commit_iso: str
) -> List[str]:
    """The mechanical two-condition check; empty list means it passes.

    `commit_iso` is the commit's authored time in the store's ISO-8601
    UTC shape (`YYYY-MM-DDThh:mm:ssZ`), so the ordering comparisons are
    lexicographic on a fixed-width format.
    """
    violations: List[str] = []
    run = store.read_run(connection, cited_run_id)
    if run is None:
        return [f"cited Battery-Run id {cited_run_id!r} is not in the results store"]

    # Condition (a): predates the commit AND recorded a failure.
    if run.started_at >= commit_iso:
        violations.append(
            f"cited run {cited_run_id!r} started at {run.started_at}, not before"
            f" the commit at {commit_iso} — the antecedent run must predate the fix"
        )
    if _failure_count(connection, cited_run_id) == 0:
        violations.append(
            f"cited run {cited_run_id!r} recorded no failures — a fix must cite a"
            " run that actually failed (batch-then-diagnose, NFR-506)"
        )

    # Condition (b): a new full-battery run after the commit.
    if not _newer_run_exists(
        connection, battery_name=run.battery_name, after_iso=commit_iso
    ):
        violations.append(
            f"no {run.battery_name!r} run started after the commit at {commit_iso}"
            " — the whole battery must be re-run after the fix, not left piecemeal"
        )
    return violations
