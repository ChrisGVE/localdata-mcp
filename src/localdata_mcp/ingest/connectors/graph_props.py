"""localdata_mcp/ingest/connectors/graph_props.py — graph node properties (E8.3).

The property half of the harvested graph-schema access (split from
graph_store.py per the code-size discipline): typed key-value rows on
a graph node (`owner_type = 'node'`; the harvested shape carries no
`original_repr`), every statement parameterized SQL through the guard.
`set_property` auto-creates the addressed node. Neighbors:
kv/tools.py is the tool surface; graph_store.py owns nodes and edges;
values.py owns the value round-trip.
"""

from __future__ import annotations

import time
from typing import Any, Optional

from localdata_mcp.nexus.chokepoint.guard import Result

from .graph_store import _NO_LIMIT, _mutate, _query, ensure_node
from .store_dispatch import one_row, one_value
from .values import (
    ValueType,
    deserialize_value,
    infer_value_type,
    serialize_value,
)


def get_property(endpoint: str, node_id: str, key: str) -> Optional[dict[str, Any]]:
    result = _query(
        endpoint,
        "SELECT key, value, value_type FROM graph_properties "
        "WHERE owner_type = 'node' AND owner_id = :nid AND key = :key",
        {"nid": node_id, "key": key},
    )
    row = one_row(result)
    if row is None:
        return None
    return {
        "key": row[0],
        "value": deserialize_value(row[1], ValueType(row[2])),
        "value_type": row[2],
    }


def set_property(
    endpoint: str,
    node_id: str,
    key: str,
    value: Any,
    value_type: Optional[ValueType],
) -> dict[str, Any]:
    """Upsert a node property; auto-creates the node (harvested)."""
    if value_type is None:
        value_type = infer_value_type(value)
    stored, _original = serialize_value(value, value_type)
    ensure_node(endpoint, node_id)
    now = time.time()
    existing = one_row(
        _query(
            endpoint,
            "SELECT id FROM graph_properties "
            "WHERE owner_type = 'node' AND owner_id = :nid AND key = :key",
            {"nid": node_id, "key": key},
        )
    )
    if existing is not None:
        _mutate(
            endpoint,
            "UPDATE graph_properties SET value = :val, value_type = :vt, "
            "updated_at = :now WHERE id = :pid",
            {"val": stored, "vt": value_type.value, "now": now, "pid": existing[0]},
        )
    else:
        _mutate(
            endpoint,
            "INSERT INTO graph_properties (owner_type, owner_id, key, value, "
            "value_type, created_at, updated_at) "
            "VALUES ('node', :nid, :key, :val, :vt, :now, :now)",
            {
                "nid": node_id,
                "key": key,
                "val": stored,
                "vt": value_type.value,
                "now": now,
            },
        )
    return {
        "node_id": node_id,
        "key": key,
        "value": deserialize_value(stored, value_type),
        "value_type": value_type.value,
    }


def delete_property(endpoint: str, node_id: str, key: str) -> bool:
    outcome = _mutate(
        endpoint,
        "DELETE FROM graph_properties "
        "WHERE owner_type = 'node' AND owner_id = :nid AND key = :key",
        {"nid": node_id, "key": key},
    )
    return bool(outcome.affected_rows)


def list_properties(
    endpoint: str, node_id: str, offset: int, limit: Optional[int]
) -> Result:
    return _query(
        endpoint,
        "SELECT key, value, value_type FROM graph_properties "
        "WHERE owner_type = 'node' AND owner_id = :nid "
        "ORDER BY key LIMIT :lim OFFSET :off",
        {
            "nid": node_id,
            "lim": _NO_LIMIT if limit is None else limit,
            "off": offset,
        },
    )


def property_count(endpoint: str, node_id: str) -> int:
    return int(
        one_value(
            _query(
                endpoint,
                "SELECT COUNT(*) FROM graph_properties "
                "WHERE owner_type = 'node' AND owner_id = :nid",
                {"nid": node_id},
            )
        )
    )
