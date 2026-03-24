"""Graph storage model for graph-structured data files (DOT, GML, GraphML).

Stores directed or undirected graphs as nodes and edges with attached
key-value properties, using three SQL tables: ``graph_nodes`` (vertices),
``graph_edges`` (connections), and ``graph_properties`` (typed key-value
data on nodes or edges).
"""

from dataclasses import dataclass
from typing import Optional

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
    """Create the graph_nodes, graph_edges, and graph_properties tables (idempotent)."""
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
    def from_row(cls, row) -> "GraphNode":
        """Create from a SQLAlchemy Row or dict-like mapping."""
        mapping = row._mapping if hasattr(row, "_mapping") else row
        return cls(
            id=mapping["id"],
            node_id=mapping["node_id"],
            label=mapping["label"],
            created_at=mapping["created_at"],
            updated_at=mapping["updated_at"],
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
    def from_row(cls, row) -> "GraphEdge":
        """Create from a SQLAlchemy Row or dict-like mapping."""
        mapping = row._mapping if hasattr(row, "_mapping") else row
        return cls(
            id=mapping["id"],
            source_id=mapping["source_id"],
            target_id=mapping["target_id"],
            label=mapping["label"],
            weight=mapping["weight"],
            created_at=mapping["created_at"],
            updated_at=mapping["updated_at"],
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
    def from_row(cls, row) -> "GraphProperty":
        """Create from a SQLAlchemy Row or dict-like mapping."""
        mapping = row._mapping if hasattr(row, "_mapping") else row
        return cls(
            id=mapping["id"],
            owner_type=mapping["owner_type"],
            owner_id=mapping["owner_id"],
            key=mapping["key"],
            value=mapping["value"],
            value_type=ValueType(mapping["value_type"]),
            created_at=mapping["created_at"],
            updated_at=mapping["updated_at"],
        )
