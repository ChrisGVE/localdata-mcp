"""Graph storage manager for graph-structured data.

Houses the :class:`GraphStorageManager` class which manages a directed
multigraph backed by SQLite via three tables: ``graph_nodes``,
``graph_edges``, and ``graph_properties``.
"""

import time
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import text
from sqlalchemy.engine import Engine

from .graph_edge_ops import (
    add_edge as _add_edge,
    get_edge as _get_edge,
    get_edge_count as _get_edge_count,
    get_neighbors as _get_neighbors,
    get_predecessors as _get_predecessors,
    get_successors as _get_successors,
    list_edges as _list_edges,
    remove_edge as _remove_edge,
)
from .graph_properties import (
    delete_property as _delete_property,
    get_graph_stats as _get_graph_stats,
    get_property as _get_property,
    get_property_count as _get_property_count,
    list_properties as _list_properties,
    set_property as _set_property,
)
from .graph_storage import (
    GraphEdge,
    GraphNode,
    GraphProperty,
    create_graph_schema,
)
from .tree_storage import ValueType


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

    def _upsert_node(
        self, conn, node_id: str, label: Optional[str], now: float
    ) -> None:
        """Insert or update a node row within a connection."""
        existing = conn.execute(
            text("SELECT 1 FROM graph_nodes WHERE node_id = :nid"),
            {"nid": node_id},
        ).fetchone()
        if existing:
            conn.execute(
                text(
                    "UPDATE graph_nodes SET label = :lbl, updated_at = :ua WHERE node_id = :nid"
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

    def create_node(self, node_id: str, label: Optional[str] = None) -> GraphNode:
        """Create a node or update its label if it already exists (upsert)."""
        if not node_id or not node_id.strip():
            raise ValueError("node_id must be a non-empty string")
        now = time.time()
        with self.engine.connect() as conn:
            conn.execute(text("PRAGMA foreign_keys = ON"))
            self._upsert_node(conn, node_id, label, now)
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
            edge_ids = self._collect_edge_ids(conn, node_id)
            edge_prop_count = self._delete_edge_properties(conn, edge_ids)
            conn.execute(
                text(
                    "DELETE FROM graph_edges WHERE source_id = :nid OR target_id = :nid"
                ),
                {"nid": node_id},
            )
            node_prop_count = self._delete_node_properties(conn, node_id)
            result = conn.execute(
                text("DELETE FROM graph_nodes WHERE node_id = :nid"),
                {"nid": node_id},
            )
            conn.commit()
            return (result.rowcount, len(edge_ids), node_prop_count + edge_prop_count)

    def _collect_edge_ids(self, conn, node_id: str) -> List[int]:
        """Return edge IDs touching *node_id*."""
        return [
            row[0]
            for row in conn.execute(
                text(
                    "SELECT id FROM graph_edges "
                    "WHERE source_id = :nid OR target_id = :nid"
                ),
                {"nid": node_id},
            ).fetchall()
        ]

    def _delete_edge_properties(self, conn, edge_ids: List[int]) -> int:
        """Delete all properties for the given edge IDs."""
        count = 0
        for eid in edge_ids:
            count += conn.execute(
                text(
                    "DELETE FROM graph_properties "
                    "WHERE owner_type = 'edge' AND owner_id = :oid"
                ),
                {"oid": str(eid)},
            ).rowcount
        return count

    def _delete_node_properties(self, conn, node_id: str) -> int:
        """Delete all properties for a node; return count deleted."""
        count = conn.execute(
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
        return count

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

    # -- edge operations (delegated) ----------------------------------------

    def add_edge(
        self,
        source: str,
        target: str,
        label: Optional[str] = None,
        weight: Optional[float] = None,
    ) -> GraphEdge:
        """Add an edge, auto-creating source/target nodes if needed."""
        return _add_edge(self.engine, source, target, label, weight)

    def remove_edge(
        self, source: str, target: str, label: Optional[str] = None
    ) -> bool:
        """Remove an edge. Returns True if it existed."""
        return _remove_edge(self.engine, source, target, label)

    def get_edge(
        self, source: str, target: str, label: Optional[str] = None
    ) -> Optional[GraphEdge]:
        """Get a single edge by source, target, and optional label."""
        return _get_edge(self.engine, source, target, label)

    def list_edges(
        self,
        node_id: Optional[str] = None,
        offset: int = 0,
        limit: int = 50,
    ) -> List[GraphEdge]:
        """List edges, optionally filtered by a node."""
        return _list_edges(self.engine, node_id, offset, limit)

    def get_edge_count(self) -> int:
        """Return the total number of edges."""
        return _get_edge_count(self.engine)

    # -- neighbor operations (delegated) ------------------------------------

    def get_successors(
        self, node_id: str, offset: int = 0, limit: int = 50
    ) -> List[str]:
        """Return node_ids reachable via outgoing edges."""
        return _get_successors(self.engine, node_id, offset, limit)

    def get_predecessors(
        self, node_id: str, offset: int = 0, limit: int = 50
    ) -> List[str]:
        """Return node_ids with incoming edges to node_id."""
        return _get_predecessors(self.engine, node_id, offset, limit)

    def get_neighbors(
        self,
        node_id: str,
        direction: str = "both",
        offset: int = 0,
        limit: int = 50,
    ) -> List[str]:
        """Return neighbor node_ids by direction."""
        return _get_neighbors(self.engine, node_id, direction, offset, limit)

    # -- property operations (delegated) ------------------------------------

    def set_property(
        self,
        owner_type: str,
        owner_id: str,
        key: str,
        value: Any,
        value_type: Optional[ValueType] = None,
    ) -> GraphProperty:
        """Set a property on a node or edge (upsert)."""
        return _set_property(self.engine, owner_type, owner_id, key, value, value_type)

    def get_property(
        self, owner_type: str, owner_id: str, key: str
    ) -> Optional[GraphProperty]:
        """Get a single property by owner and key."""
        return _get_property(self.engine, owner_type, owner_id, key)

    def list_properties(
        self,
        owner_type: str,
        owner_id: str,
        offset: int = 0,
        limit: int = 50,
    ) -> List[GraphProperty]:
        """List properties for an owner with pagination."""
        return _list_properties(self.engine, owner_type, owner_id, offset, limit)

    def delete_property(self, owner_type: str, owner_id: str, key: str) -> bool:
        """Delete a property. Returns True if it existed."""
        return _delete_property(self.engine, owner_type, owner_id, key)

    def get_property_count(self, owner_type: str, owner_id: str) -> int:
        """Return the number of properties for an owner."""
        return _get_property_count(self.engine, owner_type, owner_id)

    # -- statistics (delegated) ---------------------------------------------

    def get_graph_stats(self) -> Dict[str, Any]:
        """Get summary statistics about the graph."""
        return _get_graph_stats(self.engine)
