"""localdata_mcp/ingest/connectors/tree_props.py — tree properties (E8.3).

The property half of the harvested tree-schema access (split from
tree_store.py per the code-size discipline): typed key-value rows on a
tree node, every statement parameterized SQL through the guard.
`set_property` auto-creates the addressed node (tree_store.ensure_node
— `main`'s harvested semantics). Neighbors: kv/tools.py is the tool
surface; tree_store.py owns nodes; values.py owns the value round-trip.
"""

from __future__ import annotations

import time
from typing import Any, Optional

from localdata_mcp.nexus.chokepoint.guard import Result

from .store_dispatch import one_row, one_value
from .tree_store import _NO_LIMIT, _mutate, _query, ensure_node
from .values import (
    ValueType,
    deserialize_value,
    infer_value_type,
    serialize_value,
)


def get_property(endpoint: str, path: str, key: str) -> Optional[dict[str, Any]]:
    """One property as {key, value, value_type}, or None."""
    result = _query(
        endpoint,
        "SELECT p.key, p.value, p.value_type, p.original_repr "
        "FROM properties p JOIN nodes n ON p.node_id = n.id "
        "WHERE n.path = :path AND p.key = :key",
        {"path": path, "key": key},
    )
    row = one_row(result)
    if row is None:
        return None
    return {
        "key": row[0],
        "value": deserialize_value(row[1], ValueType(row[2]), row[3]),
        "value_type": row[2],
    }


def set_property(
    endpoint: str,
    path: str,
    key: str,
    value: Any,
    value_type: Optional[ValueType],
) -> dict[str, Any]:
    """Upsert a property; auto-creates the node (harvested semantics)."""
    if value_type is None:
        value_type = infer_value_type(value)
    stored, original = serialize_value(value, value_type)
    node_id, _created = ensure_node(endpoint, path)
    now = time.time()
    existing = one_row(
        _query(
            endpoint,
            "SELECT id FROM properties WHERE node_id = :nid AND key = :key",
            {"nid": node_id, "key": key},
        )
    )
    if existing is not None:
        _mutate(
            endpoint,
            "UPDATE properties SET value = :val, value_type = :vt, "
            "original_repr = :orig, updated_at = :now WHERE id = :pid",
            {
                "val": stored,
                "vt": value_type.value,
                "orig": original,
                "now": now,
                "pid": existing[0],
            },
        )
    else:
        _mutate(
            endpoint,
            "INSERT INTO properties (node_id, key, value, value_type, "
            "original_repr, created_at, updated_at) "
            "VALUES (:nid, :key, :val, :vt, :orig, :now, :now)",
            {
                "nid": node_id,
                "key": key,
                "val": stored,
                "vt": value_type.value,
                "orig": original,
                "now": now,
            },
        )
    return {
        "path": path,
        "key": key,
        "value": deserialize_value(stored, value_type, original),
        "value_type": value_type.value,
    }


def delete_property(endpoint: str, path: str, key: str) -> bool:
    """Delete a property; True when it existed."""
    outcome = _mutate(
        endpoint,
        "DELETE FROM properties WHERE id IN ("
        "  SELECT p.id FROM properties p JOIN nodes n ON p.node_id = n.id"
        "  WHERE n.path = :path AND p.key = :key)",
        {"path": path, "key": key},
    )
    return bool(outcome.affected_rows)


def list_properties(
    endpoint: str, path: str, offset: int, limit: Optional[int]
) -> Result:
    """The node's properties as a tabular result, key-ordered."""
    return _query(
        endpoint,
        "SELECT p.key, p.value, p.value_type, p.original_repr "
        "FROM properties p JOIN nodes n ON p.node_id = n.id "
        "WHERE n.path = :path ORDER BY p.key LIMIT :lim OFFSET :off",
        {
            "path": path,
            "lim": _NO_LIMIT if limit is None else limit,
            "off": offset,
        },
    )


def property_count(endpoint: str, path: str) -> int:
    return int(
        one_value(
            _query(
                endpoint,
                "SELECT COUNT(*) FROM properties p "
                "JOIN nodes n ON p.node_id = n.id WHERE n.path = :path",
                {"path": path},
            )
        )
    )
