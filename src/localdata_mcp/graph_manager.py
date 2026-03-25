"""Graph storage manager for graph-structured data.

Houses the :class:`GraphStorageManager` class which manages a directed
multigraph backed by SQLite via three tables: ``graph_nodes``,
``graph_edges``, and ``graph_properties``.
"""

import time
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import text
from sqlalchemy.engine import Engine

from .graph_storage import (
    GraphEdge,
    GraphNode,
    GraphProperty,
    _edge_label_clause,
    _find_edge,
    create_graph_schema,
)
from .tree_storage import ValueType, infer_value_type, serialize_value


class GraphStorageManager:
    """Manage a directed multigraph backed by SQLite.

    Uses manual cascade for node deletion because the schema relies on
    TEXT foreign-key references and a polymorphic ``owner_type``/``owner_id``
    pattern in ``graph_properties``, which SQLite cannot cascade
    automatically.
    """

    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        create_graph_schema(engine)

    # -- node operations ----------------------------------------------------

    def create_node(self, node_id: str, label: Optional[str] = None) -> GraphNode:
        """Create a node or update its label if it already exists (upsert)."""
        if not node_id or not node_id.strip():
            raise ValueError("node_id must be a non-empty string")
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

        Manual cascade is required here because ``graph_edges`` references
        ``graph_nodes(node_id)`` via TEXT columns and
        ``graph_properties`` uses a polymorphic ``owner_type``/``owner_id``
        pattern — neither of which SQLite can cascade automatically.

        Returns (nodes_deleted, edges_deleted, properties_deleted).
        """
        with self.engine.connect() as conn:
            conn.execute(text("PRAGMA foreign_keys = ON"))

            # Collect edge IDs that will be deleted (for property cleanup)
            edge_ids = [
                row[0]
                for row in conn.execute(
                    text(
                        "SELECT id FROM graph_edges "
                        "WHERE source_id = :nid OR target_id = :nid"
                    ),
                    {"nid": node_id},
                ).fetchall()
            ]

            edge_count = len(edge_ids)

            # Delete edge properties
            edge_prop_count = 0
            if edge_ids:
                for eid in edge_ids:
                    edge_prop_count += conn.execute(
                        text(
                            "DELETE FROM graph_properties "
                            "WHERE owner_type = 'edge' AND owner_id = :oid"
                        ),
                        {"oid": str(eid)},
                    ).rowcount

            # Delete edges
            conn.execute(
                text(
                    "DELETE FROM graph_edges WHERE source_id = :nid OR target_id = :nid"
                ),
                {"nid": node_id},
            )

            # Delete node properties
            node_prop_count = conn.execute(
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

            # Delete node
            result = conn.execute(
                text("DELETE FROM graph_nodes WHERE node_id = :nid"),
                {"nid": node_id},
            )
            conn.commit()
            return (result.rowcount, edge_count, node_prop_count + edge_prop_count)

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
        if not source or not source.strip():
            raise ValueError("source must be a non-empty string")
        if not target or not target.strip():
            raise ValueError("target must be a non-empty string")
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
            # Find edge ID for property cleanup
            edge_row = _find_edge(conn, source, target, label)
            if edge_row:
                conn.execute(
                    text(
                        "DELETE FROM graph_properties "
                        "WHERE owner_type = 'edge' AND owner_id = :oid"
                    ),
                    {"oid": str(edge_row[0])},
                )
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
            # Density for simple directed graphs; clamped for multigraphs
            density = ec / (nc * (nc - 1)) if nc > 1 else 0.0
            density = min(density, 1.0)
            return {
                "node_count": nc,
                "edge_count": ec,
                "property_count": pc,
                "is_directed": True,
                "density": density,
            }
