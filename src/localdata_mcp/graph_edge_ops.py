"""Edge and neighbor operations for graph storage.

Standalone functions that operate on a SQLAlchemy engine directly,
called by :class:`~localdata_mcp.graph_manager.GraphStorageManager`.
"""

import time
from typing import List, Optional

from sqlalchemy import text
from sqlalchemy.engine import Engine

from .graph_storage import GraphEdge, _edge_label_clause, _find_edge


def _ensure_node(conn, node_id: str) -> None:
    """Create a node if it does not already exist (within a connection)."""
    if not conn.execute(
        text("SELECT 1 FROM graph_nodes WHERE node_id = :nid"),
        {"nid": node_id},
    ).fetchone():
        now = time.time()
        conn.execute(
            text(
                "INSERT INTO graph_nodes (node_id, label, created_at, updated_at) "
                "VALUES (:nid, NULL, :ca, :ua)"
            ),
            {"nid": node_id, "ca": now, "ua": now},
        )


def _upsert_edge(conn, source: str, target: str, label, weight, now: float) -> None:
    """Insert or update an edge row within an existing connection."""
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


def add_edge(
    engine: Engine,
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
    with engine.connect() as conn:
        conn.execute(text("PRAGMA foreign_keys = ON"))
        _ensure_node(conn, source)
        _ensure_node(conn, target)
        _upsert_edge(conn, source, target, label, weight, now)
        conn.commit()
        row = _find_edge(conn, source, target, label)
        return GraphEdge.from_row(row)


def remove_edge(
    engine: Engine,
    source: str,
    target: str,
    label: Optional[str] = None,
) -> bool:
    """Remove an edge. Returns True if it existed."""
    lbl_sql, lbl_params = _edge_label_clause(label)
    with engine.connect() as conn:
        conn.execute(text("PRAGMA foreign_keys = ON"))
        edge_row = _find_edge(conn, source, target, label)
        if edge_row:
            conn.execute(
                text(
                    "DELETE FROM graph_properties WHERE owner_type = 'edge' AND owner_id = :oid"
                ),
                {"oid": str(edge_row[0])},
            )
        result = conn.execute(
            text(
                f"DELETE FROM graph_edges WHERE source_id = :src AND target_id = :tgt {lbl_sql}"
            ),
            {"src": source, "tgt": target, **lbl_params},
        )
        conn.commit()
        return result.rowcount > 0


def get_edge(
    engine: Engine,
    source: str,
    target: str,
    label: Optional[str] = None,
) -> Optional[GraphEdge]:
    """Get a single edge by source, target, and optional label."""
    with engine.connect() as conn:
        row = _find_edge(conn, source, target, label)
        return GraphEdge.from_row(row) if row else None


def list_edges(
    engine: Engine,
    node_id: Optional[str] = None,
    offset: int = 0,
    limit: int = 50,
) -> List[GraphEdge]:
    """List edges, optionally filtered by a node."""
    with engine.connect() as conn:
        if node_id is None:
            rows = conn.execute(
                text(
                    "SELECT * FROM graph_edges ORDER BY source_id, target_id "
                    "LIMIT :lim OFFSET :off"
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


def get_edge_count(engine: Engine) -> int:
    """Return the total number of edges."""
    with engine.connect() as conn:
        return conn.execute(text("SELECT COUNT(*) FROM graph_edges")).fetchone()[0]


# -- neighbor operations ----------------------------------------------------


def get_successors(
    engine: Engine, node_id: str, offset: int = 0, limit: int = 50
) -> List[str]:
    """Return node_ids reachable via outgoing edges from node_id."""
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT DISTINCT target_id FROM graph_edges "
                "WHERE source_id = :nid ORDER BY target_id LIMIT :lim OFFSET :off"
            ),
            {"nid": node_id, "lim": limit, "off": offset},
        ).fetchall()
        return [r[0] for r in rows]


def get_predecessors(
    engine: Engine, node_id: str, offset: int = 0, limit: int = 50
) -> List[str]:
    """Return node_ids with incoming edges to node_id."""
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT DISTINCT source_id FROM graph_edges "
                "WHERE target_id = :nid ORDER BY source_id LIMIT :lim OFFSET :off"
            ),
            {"nid": node_id, "lim": limit, "off": offset},
        ).fetchall()
        return [r[0] for r in rows]


def get_neighbors(
    engine: Engine,
    node_id: str,
    direction: str = "both",
    offset: int = 0,
    limit: int = 50,
) -> List[str]:
    """Return neighbor node_ids by direction ('out', 'in', or 'both')."""
    if direction == "out":
        return get_successors(engine, node_id, offset, limit)
    if direction == "in":
        return get_predecessors(engine, node_id, offset, limit)
    with engine.connect() as conn:
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
