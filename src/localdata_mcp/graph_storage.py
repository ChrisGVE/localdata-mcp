"""Graph storage schema, dataclasses, and helpers for graph-structured data.

Stores directed or undirected graphs as nodes and edges with attached
key-value properties, using three SQL tables: ``graph_nodes`` (vertices),
``graph_edges`` (connections), and ``graph_properties`` (typed key-value
data on nodes or edges).

The :class:`GraphStorageManager` class lives in :mod:`graph_manager` but
is re-exported here for backward compatibility.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from sqlalchemy import text
from sqlalchemy.engine import Engine

from localdata_mcp.tree_storage import ValueType

# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------

GRAPH_NODES_TABLE_SQL = """\
CREATE TABLE IF NOT EXISTS graph_nodes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    node_id TEXT NOT NULL UNIQUE,
    label TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
)"""

GRAPH_EDGES_TABLE_SQL = """\
CREATE TABLE IF NOT EXISTS graph_edges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id TEXT NOT NULL REFERENCES graph_nodes(node_id),
    target_id TEXT NOT NULL REFERENCES graph_nodes(node_id),
    label TEXT,
    weight REAL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    UNIQUE(source_id, target_id, label)
)"""

GRAPH_PROPERTIES_TABLE_SQL = """\
CREATE TABLE IF NOT EXISTS graph_properties (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    owner_type TEXT NOT NULL CHECK(owner_type IN ('node', 'edge')),
    owner_id TEXT NOT NULL,
    key TEXT NOT NULL,
    value TEXT,
    value_type TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    UNIQUE(owner_type, owner_id, key)
)"""

GRAPH_INDEX_STATEMENTS = [
    "CREATE INDEX IF NOT EXISTS idx_graph_nodes_node_id ON graph_nodes(node_id)",
    "CREATE INDEX IF NOT EXISTS idx_graph_edges_source ON graph_edges(source_id)",
    "CREATE INDEX IF NOT EXISTS idx_graph_edges_target ON graph_edges(target_id)",
    (
        "CREATE INDEX IF NOT EXISTS idx_graph_properties_owner "
        "ON graph_properties(owner_type, owner_id)"
    ),
]


def create_graph_schema(engine: Engine) -> None:
    """Create the graph tables and indexes (idempotent)."""
    with engine.connect() as conn:
        conn.execute(text("PRAGMA foreign_keys = ON"))
        conn.execute(text(GRAPH_NODES_TABLE_SQL))
        conn.execute(text(GRAPH_EDGES_TABLE_SQL))
        conn.execute(text(GRAPH_PROPERTIES_TABLE_SQL))
        for stmt in GRAPH_INDEX_STATEMENTS:
            conn.execute(text(stmt))
        conn.commit()


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class GraphNode:
    """A vertex in the graph."""

    id: int
    node_id: str
    label: Optional[str]
    created_at: float = 0.0
    updated_at: float = 0.0

    @classmethod
    def from_row(cls, row: Any) -> "GraphNode":
        m = row._mapping if hasattr(row, "_mapping") else row
        return cls(
            id=m["id"],
            node_id=m["node_id"],
            label=m["label"],
            created_at=m["created_at"],
            updated_at=m["updated_at"],
        )


@dataclass
class GraphEdge:
    """A directed edge in the graph."""

    id: int
    source_id: str
    target_id: str
    label: Optional[str] = None
    weight: Optional[float] = None
    created_at: float = 0.0
    updated_at: float = 0.0

    @classmethod
    def from_row(cls, row: Any) -> "GraphEdge":
        m = row._mapping if hasattr(row, "_mapping") else row
        return cls(
            id=m["id"],
            source_id=m["source_id"],
            target_id=m["target_id"],
            label=m["label"],
            weight=m["weight"],
            created_at=m["created_at"],
            updated_at=m["updated_at"],
        )


@dataclass
class GraphProperty:
    """A typed key-value pair attached to a node or edge."""

    id: int
    owner_type: str
    owner_id: str
    key: str
    value: Optional[str]
    value_type: ValueType
    created_at: float = 0.0
    updated_at: float = 0.0

    @classmethod
    def from_row(cls, row: Any) -> "GraphProperty":
        m = row._mapping if hasattr(row, "_mapping") else row
        return cls(
            id=m["id"],
            owner_type=m["owner_type"],
            owner_id=m["owner_id"],
            key=m["key"],
            value=m["value"],
            value_type=ValueType(m["value_type"]),
            created_at=m["created_at"],
            updated_at=m["updated_at"],
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _edge_label_clause(label: Optional[str]) -> Tuple[str, Dict[str, Any]]:
    """Return (SQL fragment, params) for matching an edge label that may be NULL."""
    if label is None:
        return "AND label IS NULL", {}
    return "AND label = :lbl", {"lbl": label}


def _find_edge(conn, source: str, target: str, label: Optional[str]):
    """Fetch a single edge row by source/target/label within *conn*."""
    lbl_sql, lbl_params = _edge_label_clause(label)
    return conn.execute(
        text(
            f"SELECT * FROM graph_edges "
            f"WHERE source_id = :src AND target_id = :tgt {lbl_sql}"
        ),
        {"src": source, "tgt": target, **lbl_params},
    ).fetchone()
