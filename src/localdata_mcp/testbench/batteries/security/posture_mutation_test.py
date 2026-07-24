"""testbench/batteries/security/posture_mutation_test.py — NFR-113 + NFR-106 at L3.

Two coupled controls proven through the wire the agent uses:

- **NFR-113** — each endpoint carries an operator-set read-only/read-write
  posture that the chokepoint enforces; a read-only endpoint refuses every
  mutation construct, SQL and non-SQL alike.
- **NFR-106** — mutation tools (SQL `write_query`, and the graph/tree/
  key-value `set`/`delete`/`move` operations) cross the *same* chokepoint
  as reads; there is no separate, weaker gate for mutations.

The battery seeds each backend while it is still writable, re-boots the
endpoints read-only over the seeded files, and drives one mutation per
mutating tool — all refused as structured FR-403 errors — plus one read
per store kind as the positive control that the read-only endpoint is
genuinely usable (not blanket-dead).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from . import _seam


@pytest.fixture()
def readonly_bench(tmp_path: Path):
    """SQL + kv + tree + graph endpoints, seeded writable then re-declared
    read-only over the same files."""
    sql_path = tmp_path / "ro.db"
    sqlite3.connect(sql_path).executescript(
        "CREATE TABLE t (id INTEGER, label TEXT); INSERT INTO t VALUES (1, 'a');"
    )
    kv_dsn = f"kv+sqlite:///{tmp_path / 'kv.db'}"
    tree_dsn = f"tree+sqlite:///{tmp_path / 'tree.db'}"
    graph_dsn = f"graph+sqlite:///{tmp_path / 'graph.db'}"

    writable = _seam.declare(
        kv=(kv_dsn, "read_write"),
        tree=(tree_dsn, "read_write"),
        graph=(graph_dsn, "read_write"),
    )
    with _seam.booted(allowed_paths=(str(tmp_path),), declarations=writable):
        _seam.expect_ok(
            _seam.call_envelope(
                "set_value",
                {"endpoint": "kv", "path": "root.n", "key": "k", "value": "1"},
            )
        )
        _seam.expect_ok(
            _seam.call_envelope("set_node", {"endpoint": "tree", "path": "root.child"})
        )
        _seam.expect_ok(
            _seam.call_envelope(
                "add_edge",
                {"endpoint": "graph", "source": "a", "target": "b", "label": "x"},
            )
        )

    readonly = _seam.declare(
        sql=(f"sqlite:///{sql_path}", "read_only"),
        kv=(kv_dsn, "read_only"),
        tree=(tree_dsn, "read_only"),
        graph=(graph_dsn, "read_only"),
    )
    with _seam.booted(allowed_paths=(str(tmp_path),), declarations=readonly) as guard:
        yield guard


# One mutation per mutating tool, spanning SQL DML and every store kind.
_MUTATIONS: tuple[tuple[str, str, dict], ...] = (
    (
        "sql_insert",
        "write_query",
        {"endpoint": "sql", "sql": "INSERT INTO t VALUES (2, 'b')"},
    ),
    (
        "sql_update",
        "write_query",
        {"endpoint": "sql", "sql": "UPDATE t SET label = 'z'"},
    ),
    ("sql_delete", "write_query", {"endpoint": "sql", "sql": "DELETE FROM t"}),
    (
        "kv_set",
        "set_value",
        {"endpoint": "kv", "path": "root.n", "key": "k2", "value": "9"},
    ),
    ("kv_delete", "delete_key", {"endpoint": "kv", "path": "root.n", "key": "k"}),
    ("tree_set", "set_node", {"endpoint": "tree", "path": "root.new"}),
    (
        "tree_move",
        "move_node",
        {"endpoint": "tree", "path": "root.child", "new_parent": "root"},
    ),
    ("tree_delete", "delete_node", {"endpoint": "tree", "path": "root.child"}),
    (
        "graph_add",
        "add_edge",
        {"endpoint": "graph", "source": "c", "target": "d", "label": "y"},
    ),
    (
        "graph_remove",
        "remove_edge",
        {"endpoint": "graph", "source": "a", "target": "b", "label": "x"},
    ),
)


@pytest.mark.parametrize(
    ("label", "tool", "arguments"),
    _MUTATIONS,
    ids=[row[0] for row in _MUTATIONS],
)
def test_mutation_on_readonly_endpoint_is_refused(
    readonly_bench, label: str, tool: str, arguments: dict
) -> None:
    envelope = _seam.call_envelope(tool, arguments)
    error = _seam.expect_refused(envelope)
    # Pin the cause: refused for POSTURE, not some incidental validation.
    assert "read_only" in error["message"], error["message"]


# -- positive controls: a read-only endpoint still serves reads ---------
_READS: tuple[tuple[str, str, dict], ...] = (
    ("sql_select", "query", {"endpoint": "sql", "sql": "SELECT id FROM t"}),
    ("kv_get", "get_value", {"endpoint": "kv", "path": "root.n", "key": "k"}),
    ("graph_stats", "get_graph_stats", {"endpoint": "graph"}),
)


@pytest.mark.parametrize(
    ("label", "tool", "arguments"), _READS, ids=[row[0] for row in _READS]
)
def test_read_on_readonly_endpoint_succeeds(
    readonly_bench, label: str, tool: str, arguments: dict
) -> None:
    _seam.expect_ok(_seam.call_envelope(tool, arguments))
