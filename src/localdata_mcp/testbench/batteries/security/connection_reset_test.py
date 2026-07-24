"""testbench/batteries/security/connection_reset_test.py — NFR-112 at the L3 seam.

An exception on a connection path must roll back or reset the connection to
a defined, non-mutating state before it is reused — no failed operation may
carry an undefined side-effect forward. NFR-301 governs the error *wire
shape* the client sees; this is the distinct data-side property, proven
through the wire the agent uses.

A mutation is forced to fail partway (a multi-row write whose later row
collides on the primary key, aborting the statement). The battery then
proves the connection came back to a defined state along three axes: the
follow-up read succeeds (the connection was not left wedged in a broken
transaction), it sees no partial mutation from the failed write, and a
subsequent valid mutation lands (the connection is genuinely reusable, not
merely readable). The SQLAlchemy transaction-and-pool reset this exercises
is backend-agnostic; the DSN-gated networked backends extend the same path.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from . import _seam


@pytest.fixture()
def endpoint(tmp_path: Path):
    """A writable SQLite endpoint over a seeded, unique-keyed table."""
    db = tmp_path / "reset.db"
    connection = sqlite3.connect(db)
    connection.executescript(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT); INSERT INTO t VALUES (1, 'a');"
    )
    connection.close()
    declarations = _seam.declare(sql=(f"sqlite:///{db}", "read_write"))
    with _seam.booted(allowed_paths=(str(tmp_path),), declarations=declarations):
        yield


def _ids(rows) -> list:
    return [row[0] for row in rows]


def test_connection_recovers_to_a_defined_state_after_a_failed_mutation(
    endpoint,
) -> None:
    # Force the fault: (2,'b') is valid but (1,'c') collides on the PK, so
    # the statement aborts as a unit.
    _seam.expect_refused(
        _seam.call_envelope(
            "write_query",
            {"endpoint": "sql", "sql": "INSERT INTO t VALUES (2, 'b'), (1, 'c')"},
        )
    )

    # 1) The connection is not wedged — a read succeeds.
    after_fault = _seam.expect_ok(
        _seam.call_envelope("query", {"endpoint": "sql", "sql": "SELECT id FROM t"})
    )
    # 2) It carried no partial mutation forward — 'b' never landed.
    assert _ids(after_fault["rows"]) == [1]

    # 3) The connection is genuinely reusable — a fresh valid write lands.
    _seam.expect_ok(
        _seam.call_envelope(
            "write_query", {"endpoint": "sql", "sql": "INSERT INTO t VALUES (7, 'g')"}
        )
    )
    final = _seam.expect_ok(
        _seam.call_envelope(
            "query", {"endpoint": "sql", "sql": "SELECT id FROM t ORDER BY id"}
        )
    )
    assert _ids(final["rows"]) == [1, 7]
