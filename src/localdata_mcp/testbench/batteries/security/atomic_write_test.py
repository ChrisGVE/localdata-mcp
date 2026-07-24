"""testbench/batteries/security/atomic_write_test.py — NFR-111 at the L3 seam.

A failure on any data-writing path leaves the target fully written or
untouched — never a partial artifact. Proven through the wire the agent
uses, across both writing surfaces NFR-111 names:

- **export format (file target)** — every export renders through NX-8's
  one `write_atomic` seam (temp file in the target's own directory,
  `os.replace` onto the target). A fault injected at the rename step, per
  format, leaves an existing target byte-for-byte its original self and
  leaves no stray temp file behind; the same export without the fault is
  the positive control that the write really does land.
- **mutation tool (backend transaction)** — a multi-row write whose later
  row violates a constraint fails as one statement: the earlier,
  individually-valid row is rolled back with it, so the table is exactly
  as it was. A valid single-row write is the positive control.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Callable

import pytest

from . import _seam

_ORIGINAL = b"the original bytes - must survive a failed overwrite\n"


@pytest.fixture()
def export_target(tmp_path: Path):
    """A contained root with an existing target file carrying known bytes;
    the fault-injected overwrite must leave those bytes intact."""
    target = tmp_path / "out.dat"
    target.write_bytes(_ORIGINAL)
    with _seam.booted(allowed_paths=(str(tmp_path),)):
        yield tmp_path, target


@pytest.mark.parametrize("fmt", ["csv", "json", "markdown", "parquet"])
def test_export_fault_at_rename_leaves_target_and_dir_clean(
    export_target, fmt: str, monkeypatch
) -> None:
    root, target = export_target
    real_replace: Callable[..., None] = os.replace

    def fault(src, dst, *args, **kwargs):
        # Fault only the final rename onto our target — the mid-write fault.
        if Path(dst) == target:
            raise OSError("injected mid-write fault at rename")
        return real_replace(src, dst, *args, **kwargs)

    monkeypatch.setattr(os, "replace", fault)
    envelope = _seam.call_envelope(
        "export_result",
        {"format": fmt, "path": str(target), "source": [{"x": 9}], "overwrite": True},
    )
    _seam.expect_refused(envelope)
    # Fully unmodified — not a partial artifact.
    assert target.read_bytes() == _ORIGINAL
    # And no orphaned temp file left in the directory.
    leftovers = [name for name in os.listdir(root) if name.startswith(".out.dat.")]
    assert leftovers == [], leftovers


def test_export_without_fault_lands(export_target) -> None:
    _root, target = export_target
    _seam.expect_ok(
        _seam.call_envelope(
            "export_result",
            {
                "format": "csv",
                "path": str(target),
                "source": [{"x": 9}],
                "overwrite": True,
            },
        )
    )
    assert target.read_bytes() != _ORIGINAL
    assert target.read_text(encoding="utf-8").splitlines()[0] == "x"


# -- mutation tool: a failing multi-row write rolls back whole ----------


@pytest.fixture()
def writable_sql(tmp_path: Path):
    """A writable SQLite endpoint over a seeded, unique-keyed table."""
    db = tmp_path / "w.db"
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


def test_failed_multirow_mutation_leaves_table_unchanged(writable_sql) -> None:
    # Row (2,'b') is valid but row (1,'c') collides on the primary key —
    # the whole statement aborts, so 'b' must NOT survive.
    envelope = _seam.call_envelope(
        "write_query",
        {"endpoint": "sql", "sql": "INSERT INTO t VALUES (2, 'b'), (1, 'c')"},
    )
    _seam.expect_refused(envelope)
    data = _seam.expect_ok(
        _seam.call_envelope("query", {"endpoint": "sql", "sql": "SELECT id FROM t"})
    )
    assert _ids(data["rows"]) == [1]


def test_valid_mutation_lands(writable_sql) -> None:
    _seam.expect_ok(
        _seam.call_envelope(
            "write_query", {"endpoint": "sql", "sql": "INSERT INTO t VALUES (3, 'c')"}
        )
    )
    data = _seam.expect_ok(
        _seam.call_envelope(
            "query", {"endpoint": "sql", "sql": "SELECT id FROM t ORDER BY id"}
        )
    )
    assert _ids(data["rows"]) == [1, 3]
