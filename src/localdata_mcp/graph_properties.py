"""Property and statistics operations for graph storage.

Standalone functions that operate on a SQLAlchemy engine directly,
called by :class:`~localdata_mcp.graph_manager.GraphStorageManager`
to keep the manager file under the size limit.
"""

import time
from typing import Any, Dict, List, Optional

from sqlalchemy import text
from sqlalchemy.engine import Engine

from .graph_storage import GraphProperty
from .tree_storage import ValueType, infer_value_type, serialize_value

_PROP_WHERE = "WHERE owner_type = :ot AND owner_id = :oid AND key = :key"


def _update_existing(
    conn, pid: int, serialized: str, vt: ValueType, now: float
) -> None:
    """Update an existing property row."""
    conn.execute(
        text(
            "UPDATE graph_properties "
            "SET value = :val, value_type = :vt, updated_at = :ua "
            "WHERE id = :pid"
        ),
        {"val": serialized, "vt": vt.value, "ua": now, "pid": pid},
    )


def _insert_new(
    conn,
    owner_type: str,
    owner_id: str,
    key: str,
    serialized: str,
    vt: ValueType,
    now: float,
) -> None:
    """Insert a new property row."""
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
            "vt": vt.value,
            "ca": now,
            "ua": now,
        },
    )


def _upsert_property(
    engine: Engine,
    owner_type: str,
    owner_id: str,
    key: str,
    serialized: str,
    value_type: ValueType,
) -> GraphProperty:
    """Insert or update a property and return the result."""
    now = time.time()
    params = {"ot": owner_type, "oid": owner_id, "key": key}
    with engine.connect() as conn:
        conn.execute(text("PRAGMA foreign_keys = ON"))
        existing = conn.execute(
            text(f"SELECT id FROM graph_properties {_PROP_WHERE}"),
            params,
        ).fetchone()
        if existing:
            _update_existing(conn, existing[0], serialized, value_type, now)
        else:
            _insert_new(conn, owner_type, owner_id, key, serialized, value_type, now)
        conn.commit()
        row = conn.execute(
            text(f"SELECT * FROM graph_properties {_PROP_WHERE}"),
            params,
        ).fetchone()
        return GraphProperty.from_row(row)


def set_property(
    engine: Engine,
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
    return _upsert_property(engine, owner_type, owner_id, key, serialized, value_type)


def get_property(
    engine: Engine,
    owner_type: str,
    owner_id: str,
    key: str,
) -> Optional[GraphProperty]:
    """Get a single property by owner and key."""
    with engine.connect() as conn:
        row = conn.execute(
            text(f"SELECT * FROM graph_properties {_PROP_WHERE}"),
            {"ot": owner_type, "oid": owner_id, "key": key},
        ).fetchone()
        return GraphProperty.from_row(row) if row else None


def list_properties(
    engine: Engine,
    owner_type: str,
    owner_id: str,
    offset: int = 0,
    limit: int = 50,
) -> List[GraphProperty]:
    """List properties for an owner with pagination."""
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT * FROM graph_properties "
                "WHERE owner_type = :ot AND owner_id = :oid "
                "ORDER BY key LIMIT :lim OFFSET :off"
            ),
            {"ot": owner_type, "oid": owner_id, "lim": limit, "off": offset},
        ).fetchall()
        return [GraphProperty.from_row(r) for r in rows]


def delete_property(engine: Engine, owner_type: str, owner_id: str, key: str) -> bool:
    """Delete a property. Returns True if it existed."""
    with engine.connect() as conn:
        conn.execute(text("PRAGMA foreign_keys = ON"))
        result = conn.execute(
            text(f"DELETE FROM graph_properties {_PROP_WHERE}"),
            {"ot": owner_type, "oid": owner_id, "key": key},
        )
        conn.commit()
        return result.rowcount > 0


def get_property_count(engine: Engine, owner_type: str, owner_id: str) -> int:
    """Return the number of properties for an owner."""
    with engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT COUNT(*) FROM graph_properties "
                "WHERE owner_type = :ot AND owner_id = :oid"
            ),
            {"ot": owner_type, "oid": owner_id},
        ).fetchone()
        return row[0] if row else 0


def get_graph_stats(engine: Engine) -> Dict[str, Any]:
    """Get summary statistics about the graph."""
    with engine.connect() as conn:
        nc = conn.execute(text("SELECT COUNT(*) FROM graph_nodes")).fetchone()[0]
        ec = conn.execute(text("SELECT COUNT(*) FROM graph_edges")).fetchone()[0]
        pc = conn.execute(text("SELECT COUNT(*) FROM graph_properties")).fetchone()[0]
        density = ec / (nc * (nc - 1)) if nc > 1 else 0.0
        density = min(density, 1.0)
        return {
            "node_count": nc,
            "edge_count": ec,
            "property_count": pc,
            "is_directed": True,
            "density": density,
        }
