"""localdata_mcp/ingest/connectors/tree_store.py — tree-schema access (E8.3).

The harvested successor of `tree_storage/node_ops.py` +
`property_ops.py`, rebuilt on the chokepoint: every statement here is
parameterized SQL over the `nodes`/`properties` tables
(store_schemas.py) crossing `guarded_query`/`guarded_mutation` — no
engine, no connection, no security logic (NFR-106/113 enforced by the
guard per statement). Multi-step operations (ancestor creation, a
subtree move) are SEQUENCES of individually-screened guarded calls:
cross-statement atomicity is traded for chokepoint discipline — the
declared E8.3 design decision, acceptable because each step is
idempotent-shaped (upserts, existence-checked inserts) against a
local store. Shared at the connectors level: the kv family
auto-creates nodes on set_value and the graph/tree family owns
structure — one schema, one access layer (NFR-402). Neighbors:
kv/tools.py and graph_tree/tools.py compose these; values.py owns the
property value round-trip; treepaths.py owns the path grammar.
"""

from __future__ import annotations

import time
from typing import Any, Mapping, Optional

from localdata_mcp.nexus.chokepoint.guard import QueryRequest, Result

from ..runtime import chokepoint
from .store_dispatch import one_row, one_value, rows_as_mappings
from .treepaths import build_path, parse_path
from .values import (
    ValueType,
    deserialize_value,
    infer_value_type,
    serialize_value,
)

# SQLite treats LIMIT -1 as "no limit" — the None-limit binding.
_NO_LIMIT = -1


def _query(endpoint: str, sql: str, params: Mapping[str, Any]) -> Result:
    return chokepoint().guarded_query(
        endpoint, QueryRequest(text=sql, parameters=params)
    )


def _mutate(endpoint: str, sql: str, params: Mapping[str, Any]) -> Result:
    return chokepoint().guarded_mutation(
        endpoint, QueryRequest(text=sql, parameters=params)
    )


# -- nodes ------------------------------------------------------------


def node_row(endpoint: str, path: str) -> Optional[Mapping[str, Any]]:
    """The node at `path`, or None."""
    result = _query(
        endpoint,
        "SELECT id, parent_id, name, path, depth, is_array_item "
        "FROM nodes WHERE path = :path",
        {"path": path},
    )
    mappings = rows_as_mappings(result)
    return mappings[0] if mappings else None


def node_exists(endpoint: str, path: str) -> bool:
    return node_row(endpoint, path) is not None


def ensure_node(endpoint: str, path: str) -> tuple[int, list[str]]:
    """The node id at `path`, creating it and any missing ancestors
    (harvested create_node semantics). Returns (node_id, created_paths)."""
    segments = parse_path(path)
    if not segments:
        raise ValueError("Path must not be empty")
    now = time.time()
    created: list[str] = []
    parent_id: Optional[int] = None
    node_id: Optional[int] = None
    for depth in range(len(segments)):
        partial = build_path(segments[: depth + 1])
        existing = node_row(endpoint, partial)
        if existing is None:
            _mutate(
                endpoint,
                "INSERT INTO nodes (parent_id, name, path, depth, "
                "is_array_item, created_at, updated_at) "
                "VALUES (:pid, :name, :path, :depth, FALSE, :now, :now)",
                {
                    "pid": parent_id,
                    "name": segments[depth],
                    "path": partial,
                    "depth": depth,
                    "now": now,
                },
            )
            created.append(partial)
            existing = node_row(endpoint, partial)
        assert existing is not None
        parent_id = int(existing["id"])
        node_id = parent_id
    assert node_id is not None
    return node_id, created


def children(
    endpoint: str,
    parent_path: Optional[str],
    offset: int,
    limit: Optional[int],
) -> Result:
    """Direct children of a node (root nodes when parent is None)."""
    bound = _NO_LIMIT if limit is None else limit
    if parent_path is None:
        return _query(
            endpoint,
            "SELECT name, path, depth FROM nodes WHERE parent_id IS NULL "
            "ORDER BY name LIMIT :lim OFFSET :off",
            {"lim": bound, "off": offset},
        )
    return _query(
        endpoint,
        "SELECT c.name, c.path, c.depth FROM nodes c "
        "JOIN nodes p ON c.parent_id = p.id WHERE p.path = :path "
        "ORDER BY c.name LIMIT :lim OFFSET :off",
        {"path": parent_path, "lim": bound, "off": offset},
    )


def children_count(endpoint: str, parent_path: Optional[str]) -> int:
    if parent_path is None:
        result = _query(
            endpoint,
            "SELECT COUNT(*) FROM nodes WHERE parent_id IS NULL",
            {},
        )
    else:
        result = _query(
            endpoint,
            "SELECT COUNT(*) FROM nodes c JOIN nodes p ON c.parent_id = p.id "
            "WHERE p.path = :path",
            {"path": parent_path},
        )
    return int(one_value(result))


def delete_subtree(endpoint: str, path: str) -> tuple[int, int]:
    """Delete a node and all descendants; properties cascade via the
    schema's FK (engines.py keeps `foreign_keys = ON` per connect).
    Returns (nodes_deleted, properties_deleted)."""
    like = path + ".%"
    properties_deleted = int(
        one_value(
            _query(
                endpoint,
                "SELECT COUNT(*) FROM properties WHERE node_id IN "
                "(SELECT id FROM nodes WHERE path = :p OR path LIKE :prefix)",
                {"p": path, "prefix": like},
            )
        )
    )
    outcome = _mutate(
        endpoint,
        "DELETE FROM nodes WHERE path = :p OR path LIKE :prefix",
        {"p": path, "prefix": like},
    )
    return (outcome.affected_rows or 0, properties_deleted)


def move_subtree(
    endpoint: str, path: str, new_parent: Optional[str]
) -> tuple[int, str]:
    """Move a node and its subtree under a new parent (root when None) —
    harvested move_node semantics, each row update its own guarded
    mutation. Returns (nodes_moved, new_path)."""
    node = node_row(endpoint, path)
    if node is None:
        raise ValueError(f"Node not found: {path}")
    if new_parent is not None:
        if new_parent == path or new_parent.startswith(path + "."):
            raise ValueError(
                f"Cannot move '{path}' under its own subtree '{new_parent}'."
            )
        if not node_exists(endpoint, new_parent):
            raise ValueError(f"Target parent not found: {new_parent}")

    segments = parse_path(path)
    node_name = segments[-1]
    if new_parent is not None:
        parent_node = node_row(endpoint, new_parent)
        assert parent_node is not None
        new_path = build_path(parse_path(new_parent) + [node_name])
        new_parent_id: Optional[int] = int(parent_node["id"])
        new_depth_base = int(parent_node["depth"]) + 1
    else:
        new_path = build_path([node_name])
        new_parent_id = None
        new_depth_base = 0

    if node_exists(endpoint, new_path) and new_path != path:
        raise ValueError(f"A node already exists at '{new_path}'.")

    old_depth = int(node["depth"])
    now = time.time()
    rows = rows_as_mappings(
        _query(
            endpoint,
            "SELECT id, path, depth FROM nodes "
            "WHERE path = :p OR path LIKE :prefix ORDER BY depth",
            {"p": path, "prefix": path + ".%"},
        )
    )
    for row in rows:
        old_row_path = str(row["path"])
        if old_row_path == path:
            updated_path = new_path
            updated_parent_id = new_parent_id
        else:
            updated_path = new_path + old_row_path[len(path) :]
            parent_path = build_path(parse_path(updated_path)[:-1])
            parent_row = node_row(endpoint, parent_path)
            updated_parent_id = None if parent_row is None else int(parent_row["id"])
        _mutate(
            endpoint,
            "UPDATE nodes SET path = :new_path, parent_id = :pid, "
            "depth = :depth, updated_at = :now WHERE id = :nid",
            {
                "new_path": updated_path,
                "pid": updated_parent_id,
                "depth": int(row["depth"]) - old_depth + new_depth_base,
                "now": now,
                "nid": int(row["id"]),
            },
        )
    return (len(rows), new_path)


def tree_stats(endpoint: str) -> dict[str, Any]:
    """Summary statistics (harvested get_tree_stats, sans sampled names
    — get_children is the browse surface)."""
    return {
        "total_nodes": int(
            one_value(_query(endpoint, "SELECT COUNT(*) FROM nodes", {}))
        ),
        "total_properties": int(
            one_value(_query(endpoint, "SELECT COUNT(*) FROM properties", {}))
        ),
        "max_depth": int(
            one_value(
                _query(endpoint, "SELECT COALESCE(MAX(depth), -1) FROM nodes", {})
            )
        ),
        "root_count": int(
            one_value(
                _query(
                    endpoint,
                    "SELECT COUNT(*) FROM nodes WHERE parent_id IS NULL",
                    {},
                )
            )
        ),
    }


