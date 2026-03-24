"""Graph storage model for graph-structured data files (DOT, GML, GraphML).

Stores directed or undirected graphs as nodes and edges with attached
key-value properties, using three SQL tables: ``graph_nodes`` (vertices),
``graph_edges`` (connections), and ``graph_properties`` (typed key-value
data on nodes or edges).
"""

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import text
from sqlalchemy.engine import Engine

from localdata_mcp.tree_storage import ValueType, infer_value_type, serialize_value

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


# ---------------------------------------------------------------------------
# Storage manager
# ---------------------------------------------------------------------------


class GraphStorageManager:
    """Manage a directed multigraph backed by SQLite."""

    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        create_graph_schema(engine)

    # -- node operations ----------------------------------------------------

    def create_node(self, node_id: str, label: Optional[str] = None) -> GraphNode:
        """Create a node or update its label if it already exists (upsert)."""
        now = time.time()
        with self.engine.connect() as conn:
            conn.execute(text("PRAGMA foreign_keys = ON"))
            existing = conn.execute(
                text("SELECT 1 FROM graph_nodes WHERE node_id = :nid"),
                {"nid": node_id},
            ).fetchone()
            if existing:
                conn.execute(
                    text(
                        "UPDATE graph_nodes SET label = :lbl, updated_at = :ua "
                        "WHERE node_id = :nid"
                    ),
                    {"lbl": label, "ua": now, "nid": node_id},
                )
            else:
                conn.execute(
                    text(
                        "INSERT INTO graph_nodes (node_id, label, created_at, updated_at) "
                        "VALUES (:nid, :lbl, :ca, :ua)"
                    ),
                    {"nid": node_id, "lbl": label, "ca": now, "ua": now},
                )
            conn.commit()
            row = conn.execute(
                text("SELECT * FROM graph_nodes WHERE node_id = :nid"),
                {"nid": node_id},
            ).fetchone()
            return GraphNode.from_row(row)

    def get_node(self, node_id: str) -> Optional[GraphNode]:
        """Get a node by its node_id, or None."""
        with self.engine.connect() as conn:
            row = conn.execute(
                text("SELECT * FROM graph_nodes WHERE node_id = :nid"),
                {"nid": node_id},
            ).fetchone()
            return GraphNode.from_row(row) if row else None

    def node_exists(self, node_id: str) -> bool:
        """Return True if a node with the given node_id exists."""
        with self.engine.connect() as conn:
            row = conn.execute(
                text("SELECT 1 FROM graph_nodes WHERE node_id = :nid"),
                {"nid": node_id},
            ).fetchone()
            return row is not None

    def delete_node(self, node_id: str) -> Tuple[int, int, int]:
        """Delete a node, cascading to edges and properties.

        Returns (nodes_deleted, edges_deleted, properties_deleted).
        """
        with self.engine.connect() as conn:
            conn.execute(text("PRAGMA foreign_keys = ON"))
            edge_count = conn.execute(
                text(
                    "SELECT COUNT(*) FROM graph_edges "
                    "WHERE source_id = :nid OR target_id = :nid"
                ),
                {"nid": node_id},
            ).fetchone()[0]
            conn.execute(
                text(
                    "DELETE FROM graph_edges WHERE source_id = :nid OR target_id = :nid"
                ),
                {"nid": node_id},
            )
            prop_count = conn.execute(
                text(
                    "SELECT COUNT(*) FROM graph_properties "
                    "WHERE owner_type = 'node' AND owner_id = :nid"
                ),
                {"nid": node_id},
            ).fetchone()[0]
            conn.execute(
                text(
                    "DELETE FROM graph_properties "
                    "WHERE owner_type = 'node' AND owner_id = :nid"
                ),
                {"nid": node_id},
            )
            result = conn.execute(
                text("DELETE FROM graph_nodes WHERE node_id = :nid"),
                {"nid": node_id},
            )
            conn.commit()
            return (result.rowcount, edge_count, prop_count)

    def list_nodes(self, offset: int = 0, limit: int = 50) -> List[GraphNode]:
        """List nodes with pagination."""
        with self.engine.connect() as conn:
            rows = conn.execute(
                text(
                    "SELECT * FROM graph_nodes ORDER BY node_id LIMIT :lim OFFSET :off"
                ),
                {"lim": limit, "off": offset},
            ).fetchall()
            return [GraphNode.from_row(r) for r in rows]

    def get_node_count(self) -> int:
        """Return the total number of nodes."""
        with self.engine.connect() as conn:
            return conn.execute(text("SELECT COUNT(*) FROM graph_nodes")).fetchone()[0]

    # -- edge operations ----------------------------------------------------

    def _ensure_node(self, conn, node_id: str) -> None:
        """Create a node if it does not already exist (within a connection)."""
        if not conn.execute(
            text("SELECT 1 FROM graph_nodes WHERE node_id = :nid"), {"nid": node_id}
        ).fetchone():
            now = time.time()
            conn.execute(
                text(
                    "INSERT INTO graph_nodes (node_id, label, created_at, updated_at) "
                    "VALUES (:nid, NULL, :ca, :ua)"
                ),
                {"nid": node_id, "ca": now, "ua": now},
            )

    def add_edge(
        self,
        source: str,
        target: str,
        label: Optional[str] = None,
        weight: Optional[float] = None,
    ) -> GraphEdge:
        """Add an edge, auto-creating source/target nodes if needed."""
        now = time.time()
        with self.engine.connect() as conn:
            conn.execute(text("PRAGMA foreign_keys = ON"))
            self._ensure_node(conn, source)
            self._ensure_node(conn, target)
            existing = _find_edge(conn, source, target, label)
            if existing:
                conn.execute(
                    text(
                        "UPDATE graph_edges SET weight = :w, updated_at = :ua WHERE id = :eid"
                    ),
                    {"w": weight, "ua": now, "eid": existing[0]},
                )
            else:
                conn.execute(
                    text(
                        "INSERT INTO graph_edges "
                        "(source_id, target_id, label, weight, created_at, updated_at) "
                        "VALUES (:src, :tgt, :lbl, :w, :ca, :ua)"
                    ),
                    {
                        "src": source,
                        "tgt": target,
                        "lbl": label,
                        "w": weight,
                        "ca": now,
                        "ua": now,
                    },
                )
            conn.commit()
            row = _find_edge(conn, source, target, label)
            return GraphEdge.from_row(row)

    def remove_edge(
        self, source: str, target: str, label: Optional[str] = None
    ) -> bool:
        """Remove an edge. Returns True if it existed."""
        lbl_sql, lbl_params = _edge_label_clause(label)
        with self.engine.connect() as conn:
            conn.execute(text("PRAGMA foreign_keys = ON"))
            result = conn.execute(
                text(
                    f"DELETE FROM graph_edges "
                    f"WHERE source_id = :src AND target_id = :tgt {lbl_sql}"
                ),
                {"src": source, "tgt": target, **lbl_params},
            )
            conn.commit()
            return result.rowcount > 0

    def get_edge(
        self,
        source: str,
        target: str,
        label: Optional[str] = None,
    ) -> Optional[GraphEdge]:
        """Get a single edge by source, target, and optional label."""
        with self.engine.connect() as conn:
            row = _find_edge(conn, source, target, label)
            return GraphEdge.from_row(row) if row else None

    def list_edges(
        self,
        node_id: Optional[str] = None,
        offset: int = 0,
        limit: int = 50,
    ) -> List[GraphEdge]:
        """List edges, optionally filtered by a node (as source or target)."""
        with self.engine.connect() as conn:
            if node_id is None:
                rows = conn.execute(
                    text(
                        "SELECT * FROM graph_edges "
                        "ORDER BY source_id, target_id LIMIT :lim OFFSET :off"
                    ),
                    {"lim": limit, "off": offset},
                ).fetchall()
            else:
                rows = conn.execute(
                    text(
                        "SELECT * FROM graph_edges "
                        "WHERE source_id = :nid OR target_id = :nid "
                        "ORDER BY source_id, target_id LIMIT :lim OFFSET :off"
                    ),
                    {"nid": node_id, "lim": limit, "off": offset},
                ).fetchall()
            return [GraphEdge.from_row(r) for r in rows]

    def get_edge_count(self) -> int:
        """Return the total number of edges."""
        with self.engine.connect() as conn:
            return conn.execute(text("SELECT COUNT(*) FROM graph_edges")).fetchone()[0]

    # -- neighbor operations ------------------------------------------------

    def get_successors(
        self, node_id: str, offset: int = 0, limit: int = 50
    ) -> List[str]:
        """Return node_ids reachable via outgoing edges from node_id."""
        with self.engine.connect() as conn:
            rows = conn.execute(
                text(
                    "SELECT DISTINCT target_id FROM graph_edges "
                    "WHERE source_id = :nid ORDER BY target_id LIMIT :lim OFFSET :off"
                ),
                {"nid": node_id, "lim": limit, "off": offset},
            ).fetchall()
            return [r[0] for r in rows]

    def get_predecessors(
        self, node_id: str, offset: int = 0, limit: int = 50
    ) -> List[str]:
        """Return node_ids with incoming edges to node_id."""
        with self.engine.connect() as conn:
            rows = conn.execute(
                text(
                    "SELECT DISTINCT source_id FROM graph_edges "
                    "WHERE target_id = :nid ORDER BY source_id LIMIT :lim OFFSET :off"
                ),
                {"nid": node_id, "lim": limit, "off": offset},
            ).fetchall()
            return [r[0] for r in rows]

    def get_neighbors(
        self,
        node_id: str,
        direction: str = "both",
        offset: int = 0,
        limit: int = 50,
    ) -> List[str]:
        """Return neighbor node_ids by direction ('out', 'in', or 'both')."""
        if direction == "out":
            return self.get_successors(node_id, offset, limit)
        if direction == "in":
            return self.get_predecessors(node_id, offset, limit)
        with self.engine.connect() as conn:
            rows = conn.execute(
                text(
                    "SELECT DISTINCT nid FROM ("
                    "  SELECT target_id AS nid FROM graph_edges WHERE source_id = :nid "
                    "  UNION "
                    "  SELECT source_id AS nid FROM graph_edges WHERE target_id = :nid"
                    ") ORDER BY nid LIMIT :lim OFFSET :off"
                ),
                {"nid": node_id, "lim": limit, "off": offset},
            ).fetchall()
            return [r[0] for r in rows]

    # -- property operations ------------------------------------------------

    def set_property(
        self,
        owner_type: str,
        owner_id: str,
        key: str,
        value: Any,
        value_type: Optional[ValueType] = None,
    ) -> GraphProperty:
        """Set a property on a node or edge (upsert)."""
        if value_type is None:
            value_type = infer_value_type(value)
        serialized, _ = serialize_value(value, value_type)
        now = time.time()
        with self.engine.connect() as conn:
            conn.execute(text("PRAGMA foreign_keys = ON"))
            existing = conn.execute(
                text(
                    "SELECT id FROM graph_properties "
                    "WHERE owner_type = :ot AND owner_id = :oid AND key = :key"
                ),
                {"ot": owner_type, "oid": owner_id, "key": key},
            ).fetchone()
            if existing:
                conn.execute(
                    text(
                        "UPDATE graph_properties "
                        "SET value = :val, value_type = :vt, updated_at = :ua WHERE id = :pid"
                    ),
                    {
                        "val": serialized,
                        "vt": value_type.value,
                        "ua": now,
                        "pid": existing[0],
                    },
                )
            else:
                conn.execute(
                    text(
                        "INSERT INTO graph_properties "
                        "(owner_type, owner_id, key, value, value_type, created_at, updated_at) "
                        "VALUES (:ot, :oid, :key, :val, :vt, :ca, :ua)"
                    ),
                    {
                        "ot": owner_type,
                        "oid": owner_id,
                        "key": key,
                        "val": serialized,
                        "vt": value_type.value,
                        "ca": now,
                        "ua": now,
                    },
                )
            conn.commit()
            row = conn.execute(
                text(
                    "SELECT * FROM graph_properties "
                    "WHERE owner_type = :ot AND owner_id = :oid AND key = :key"
                ),
                {"ot": owner_type, "oid": owner_id, "key": key},
            ).fetchone()
            return GraphProperty.from_row(row)

    def get_property(
        self,
        owner_type: str,
        owner_id: str,
        key: str,
    ) -> Optional[GraphProperty]:
        """Get a single property by owner and key."""
        with self.engine.connect() as conn:
            row = conn.execute(
                text(
                    "SELECT * FROM graph_properties "
                    "WHERE owner_type = :ot AND owner_id = :oid AND key = :key"
                ),
                {"ot": owner_type, "oid": owner_id, "key": key},
            ).fetchone()
            return GraphProperty.from_row(row) if row else None

    def list_properties(
        self,
        owner_type: str,
        owner_id: str,
        offset: int = 0,
        limit: int = 50,
    ) -> List[GraphProperty]:
        """List properties for an owner with pagination."""
        with self.engine.connect() as conn:
            rows = conn.execute(
                text(
                    "SELECT * FROM graph_properties "
                    "WHERE owner_type = :ot AND owner_id = :oid "
                    "ORDER BY key LIMIT :lim OFFSET :off"
                ),
                {"ot": owner_type, "oid": owner_id, "lim": limit, "off": offset},
            ).fetchall()
            return [GraphProperty.from_row(r) for r in rows]

    def delete_property(self, owner_type: str, owner_id: str, key: str) -> bool:
        """Delete a property. Returns True if it existed."""
        with self.engine.connect() as conn:
            conn.execute(text("PRAGMA foreign_keys = ON"))
            result = conn.execute(
                text(
                    "DELETE FROM graph_properties "
                    "WHERE owner_type = :ot AND owner_id = :oid AND key = :key"
                ),
                {"ot": owner_type, "oid": owner_id, "key": key},
            )
            conn.commit()
            return result.rowcount > 0

    def get_property_count(self, owner_type: str, owner_id: str) -> int:
        """Return the number of properties for an owner."""
        with self.engine.connect() as conn:
            row = conn.execute(
                text(
                    "SELECT COUNT(*) FROM graph_properties "
                    "WHERE owner_type = :ot AND owner_id = :oid"
                ),
                {"ot": owner_type, "oid": owner_id},
            ).fetchone()
            return row[0] if row else 0

    # -- statistics ---------------------------------------------------------

    def get_graph_stats(self) -> Dict[str, Any]:
        """Get summary statistics about the graph."""
        with self.engine.connect() as conn:
            nc = conn.execute(text("SELECT COUNT(*) FROM graph_nodes")).fetchone()[0]
            ec = conn.execute(text("SELECT COUNT(*) FROM graph_edges")).fetchone()[0]
            pc = conn.execute(text("SELECT COUNT(*) FROM graph_properties")).fetchone()[
                0
            ]
            density = ec / (nc * (nc - 1)) if nc > 1 else 0.0
            return {
                "node_count": nc,
                "edge_count": ec,
                "property_count": pc,
                "is_directed": True,
                "density": density,
            }
