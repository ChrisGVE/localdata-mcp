"""localdata_mcp/ingest/connectors/graph_store.py — graph-schema access (E8.3).

The harvested successor of `graph_manager.py` + `graph_edge_ops.py` +
`graph_properties.py`, rebuilt on the chokepoint: parameterized SQL
over the `graph_nodes`/`graph_edges`/`graph_properties` tables
(store_schemas.py), every statement crossing
`guarded_query`/`guarded_mutation`. Node deletion keeps `main`'s
MANUAL cascade (TEXT references and the polymorphic owner pattern
SQLite cannot cascade) as a sequence of individually-screened guarded
mutations — the same atomicity-for-discipline trade tree_store.py
declares. Properties carry no `original_repr` (the graph schema's
harvested shape). Neighbors: kv/tools.py (node properties) and
graph_tree/tools.py (structure, edges, paths) compose these;
values.py owns the value round-trip.
"""

from __future__ import annotations

import time
from typing import Any, Mapping, Optional

from localdata_mcp.nexus.chokepoint.guard import QueryRequest, Result

from ..runtime import chokepoint
from .store_dispatch import one_row, one_value, rows_as_mappings
from .values import (
    ValueType,
    deserialize_value,
    infer_value_type,
    serialize_value,
)

_NO_LIMIT = -1  # SQLite's "no limit" binding for a None limit.


def _query(endpoint: str, sql: str, params: Mapping[str, Any]) -> Result:
    return chokepoint().guarded_query(
        endpoint, QueryRequest(text=sql, parameters=params)
    )


def _mutate(endpoint: str, sql: str, params: Mapping[str, Any]) -> Result:
    return chokepoint().guarded_mutation(
        endpoint, QueryRequest(text=sql, parameters=params)
    )


# -- nodes ------------------------------------------------------------


def node_row(endpoint: str, node_id: str) -> Optional[Mapping[str, Any]]:
    result = _query(
        endpoint,
        "SELECT id, node_id, label FROM graph_nodes WHERE node_id = :nid",
        {"nid": node_id},
    )
    mappings = rows_as_mappings(result)
    return mappings[0] if mappings else None


def node_exists(endpoint: str, node_id: str) -> bool:
    return node_row(endpoint, node_id) is not None


def upsert_node(endpoint: str, node_id: str, label: Optional[str]) -> Mapping[str, Any]:
    """Create a node or update its label (harvested create_node)."""
    if not node_id or not node_id.strip():
        raise ValueError("node_id must be a non-empty string")
    now = time.time()
    if node_exists(endpoint, node_id):
        _mutate(
            endpoint,
            "UPDATE graph_nodes SET label = :lbl, updated_at = :now "
            "WHERE node_id = :nid",
            {"lbl": label, "now": now, "nid": node_id},
        )
    else:
        _mutate(
            endpoint,
            "INSERT INTO graph_nodes (node_id, label, created_at, updated_at) "
            "VALUES (:nid, :lbl, :now, :now)",
            {"nid": node_id, "lbl": label, "now": now},
        )
    refreshed = node_row(endpoint, node_id)
    assert refreshed is not None
    return refreshed


def ensure_node(endpoint: str, node_id: str) -> None:
    """Create a bare node when missing (edge auto-creation semantics)."""
    if not node_exists(endpoint, node_id):
        now = time.time()
        _mutate(
            endpoint,
            "INSERT INTO graph_nodes (node_id, label, created_at, updated_at) "
            "VALUES (:nid, NULL, :now, :now)",
            {"nid": node_id, "now": now},
        )


def delete_node_cascade(endpoint: str, node_id: str) -> tuple[int, int, int]:
    """Delete a node, cascading edges and properties manually
    (harvested delete_node). Returns (nodes, edges, properties) deleted."""
    edge_ids = [
        int(row[0])
        for row in _query(
            endpoint,
            "SELECT id FROM graph_edges WHERE source_id = :nid OR target_id = :nid",
            {"nid": node_id},
        ).rows
    ]
    properties_deleted = 0
    for edge_id in edge_ids:
        outcome = _mutate(
            endpoint,
            "DELETE FROM graph_properties "
            "WHERE owner_type = 'edge' AND owner_id = :oid",
            {"oid": str(edge_id)},
        )
        properties_deleted += outcome.affected_rows or 0
    _mutate(
        endpoint,
        "DELETE FROM graph_edges WHERE source_id = :nid OR target_id = :nid",
        {"nid": node_id},
    )
    node_props = _mutate(
        endpoint,
        "DELETE FROM graph_properties WHERE owner_type = 'node' AND owner_id = :nid",
        {"nid": node_id},
    )
    properties_deleted += node_props.affected_rows or 0
    nodes = _mutate(
        endpoint,
        "DELETE FROM graph_nodes WHERE node_id = :nid",
        {"nid": node_id},
    )
    return (nodes.affected_rows or 0, len(edge_ids), properties_deleted)


def degrees(endpoint: str, node_id: str) -> dict[str, int]:
    in_degree = one_value(
        _query(
            endpoint,
            "SELECT COUNT(*) FROM graph_edges WHERE target_id = :nid",
            {"nid": node_id},
        )
    )
    out_degree = one_value(
        _query(
            endpoint,
            "SELECT COUNT(*) FROM graph_edges WHERE source_id = :nid",
            {"nid": node_id},
        )
    )
    return {"in_degree": int(in_degree), "out_degree": int(out_degree)}


# -- edges ------------------------------------------------------------


def _edge_row(
    endpoint: str, source: str, target: str, label: Optional[str]
) -> Optional[Mapping[str, Any]]:
    if label is None:
        return next(
            iter(
                rows_as_mappings(
                    _query(
                        endpoint,
                        "SELECT id, source_id, target_id, label, weight "
                        "FROM graph_edges WHERE source_id = :src "
                        "AND target_id = :tgt AND label IS NULL",
                        {"src": source, "tgt": target},
                    )
                )
            ),
            None,
        )
    return next(
        iter(
            rows_as_mappings(
                _query(
                    endpoint,
                    "SELECT id, source_id, target_id, label, weight "
                    "FROM graph_edges WHERE source_id = :src "
                    "AND target_id = :tgt AND label = :lbl",
                    {"src": source, "tgt": target, "lbl": label},
                )
            )
        ),
        None,
    )


def add_edge(
    endpoint: str,
    source: str,
    target: str,
    label: Optional[str],
    weight: Optional[float],
) -> tuple[Mapping[str, Any], list[str]]:
    """Upsert an edge, auto-creating endpoints (harvested add_edge).
    Returns (edge_row, nodes_created)."""
    if not source or not source.strip():
        raise ValueError("source must be a non-empty string")
    if not target or not target.strip():
        raise ValueError("target must be a non-empty string")
    nodes_created = [
        node_id
        for node_id in dict.fromkeys((source, target))
        if not node_exists(endpoint, node_id)
    ]
    for node_id in nodes_created:
        ensure_node(endpoint, node_id)
    now = time.time()
    existing = _edge_row(endpoint, source, target, label)
    if existing is not None:
        _mutate(
            endpoint,
            "UPDATE graph_edges SET weight = :w, updated_at = :now WHERE id = :eid",
            {"w": weight, "now": now, "eid": int(existing["id"])},
        )
    else:
        _mutate(
            endpoint,
            "INSERT INTO graph_edges (source_id, target_id, label, weight, "
            "created_at, updated_at) VALUES (:src, :tgt, :lbl, :w, :now, :now)",
            {
                "src": source,
                "tgt": target,
                "lbl": label,
                "w": weight,
                "now": now,
            },
        )
    refreshed = _edge_row(endpoint, source, target, label)
    assert refreshed is not None
    return refreshed, nodes_created


def remove_edge(endpoint: str, source: str, target: str, label: Optional[str]) -> bool:
    """Remove an edge and its properties; True when it existed."""
    existing = _edge_row(endpoint, source, target, label)
    if existing is None:
        return False
    _mutate(
        endpoint,
        "DELETE FROM graph_properties WHERE owner_type = 'edge' AND owner_id = :oid",
        {"oid": str(int(existing["id"]))},
    )
    outcome = _mutate(
        endpoint,
        "DELETE FROM graph_edges WHERE id = :eid",
        {"eid": int(existing["id"])},
    )
    return bool(outcome.affected_rows)


def edges(
    endpoint: str,
    node_id: Optional[str],
    offset: int,
    limit: Optional[int],
) -> Result:
    bound = _NO_LIMIT if limit is None else limit
    if node_id is None:
        return _query(
            endpoint,
            "SELECT source_id, target_id, label, weight FROM graph_edges "
            "ORDER BY source_id, target_id LIMIT :lim OFFSET :off",
            {"lim": bound, "off": offset},
        )
    return _query(
        endpoint,
        "SELECT source_id, target_id, label, weight FROM graph_edges "
        "WHERE source_id = :nid OR target_id = :nid "
        "ORDER BY source_id, target_id LIMIT :lim OFFSET :off",
        {"nid": node_id, "lim": bound, "off": offset},
    )


def edge_count(endpoint: str, node_id: Optional[str]) -> int:
    if node_id is None:
        result = _query(endpoint, "SELECT COUNT(*) FROM graph_edges", {})
    else:
        result = _query(
            endpoint,
            "SELECT COUNT(*) FROM graph_edges "
            "WHERE source_id = :nid OR target_id = :nid",
            {"nid": node_id},
        )
    return int(one_value(result))


def neighbors(
    endpoint: str,
    node_id: str,
    direction: str,
    offset: int,
    limit: Optional[int],
) -> Result:
    """Neighbor edges by direction ('out', 'in', 'both') with edge info."""
    bound = _NO_LIMIT if limit is None else limit
    if direction == "out":
        return _query(
            endpoint,
            "SELECT target_id AS neighbor_id, label, weight "
            "FROM graph_edges WHERE source_id = :nid "
            "ORDER BY target_id LIMIT :lim OFFSET :off",
            {"nid": node_id, "lim": bound, "off": offset},
        )
    if direction == "in":
        return _query(
            endpoint,
            "SELECT source_id AS neighbor_id, label, weight "
            "FROM graph_edges WHERE target_id = :nid "
            "ORDER BY source_id LIMIT :lim OFFSET :off",
            {"nid": node_id, "lim": bound, "off": offset},
        )
    return _query(
        endpoint,
        "SELECT source_id, target_id, label, weight FROM graph_edges "
        "WHERE source_id = :nid OR target_id = :nid "
        "ORDER BY source_id, target_id LIMIT :lim OFFSET :off",
        {"nid": node_id, "lim": bound, "off": offset},
    )


def graph_stats(endpoint: str) -> dict[str, Any]:
    """Summary statistics (harvested get_graph_stats)."""
    node_count = int(
        one_value(_query(endpoint, "SELECT COUNT(*) FROM graph_nodes", {}))
    )
    edge_total = int(
        one_value(_query(endpoint, "SELECT COUNT(*) FROM graph_edges", {}))
    )
    property_count = int(
        one_value(_query(endpoint, "SELECT COUNT(*) FROM graph_properties", {}))
    )
    density = edge_total / (node_count * (node_count - 1)) if node_count > 1 else 0.0
    return {
        "node_count": node_count,
        "edge_count": edge_total,
        "property_count": property_count,
        "is_directed": True,
        "density": min(density, 1.0),
    }


# -- node properties --------------------------------------------------


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
