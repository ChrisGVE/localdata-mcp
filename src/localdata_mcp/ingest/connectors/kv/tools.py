"""localdata_mcp/ingest/connectors/kv/tools.py — the kv family (E8.3).

I-3's key-value tool set (names carried from `main`, semantics intact
— DR GP2): property CRUD on a node of any declared store endpoint —
kv/tree kinds speak the tree schema, graph kinds the graph schema
(`path` addresses the node; for graph stores it is the node_id, the
harvested dual). Every mutation crosses `guarded_mutation` (NFR-106)
and respects posture (NFR-113) inside the access layers; this file is
tool surface and kind dispatch ONLY (§3's connector boundary — no
SQL, no security logic). `set_value` infers a string value's type
(`"42"` → integer) unless `value_type` names one explicitly, matching
`main`. Neighbors: tree_store.py/graph_store.py execute;
store_dispatch.py resolves kinds; refusals.py shapes the misses.
"""

from __future__ import annotations

from typing import Any

from localdata_mcp.nexus.chokepoint.guard import Result
from localdata_mcp.nexus.contract.spec import Param, TypeShape, tool_spec

from ...refusals import missing_entity_refusal
from .. import graph_store, tree_store
from ..store_dispatch import PROPERTY_KINDS, resolve_store_kind
from ..values import ValueType, deserialize_value, infer_value_type_from_string

_LIST_KEYS_HINT = (
    "Call list_keys(endpoint, path) to browse the node's properties, or "
    "get_children / get_graph_stats to discover the store's structure."
)


def _typed(value: Any, value_type: str | None) -> tuple[Any, ValueType | None]:
    """`main`'s set_value coercion: an explicit value_type converts a
    string value; a bare string infers its type from content."""
    if value_type is not None:
        declared = ValueType(value_type)
        if isinstance(value, str):
            return deserialize_value(value, declared), declared
        return value, declared
    if isinstance(value, str):
        inferred, converted = infer_value_type_from_string(value)
        return converted, inferred
    return value, None


@tool_spec(
    name="get_value",
    summary=(
        "Get one property value from a node of a declared kv, tree, or "
        "graph store endpoint (path addresses the node; node_id for "
        "graph stores)."
    ),
    params=(
        Param("endpoint", str, "The operator-declared store endpoint name."),
        Param("path", str, "The node's dot-path (or graph node_id)."),
        Param("key", str, "The property key."),
    ),
    input_shape=TypeShape.NONE,
    output_shape=TypeShape.SCALAR,
    domain="ingest",
)
def get_value(endpoint: str, path: str, key: str) -> Any:
    kind = resolve_store_kind(endpoint, PROPERTY_KINDS)
    if kind == "graph":
        found = graph_store.get_property(endpoint, path, key)
    else:
        found = tree_store.get_property(endpoint, path, key)
    if found is None:
        raise missing_entity_refusal(
            f"Property {key!r} not found on node {path!r}.", _LIST_KEYS_HINT
        )
    return found


@tool_spec(
    name="set_value",
    summary=(
        "Set (upsert) one property on a node of a declared read-write "
        "store endpoint, auto-creating the node; string values infer "
        "their type unless value_type names one."
    ),
    params=(
        Param("endpoint", str, "The operator-declared store endpoint name."),
        Param("path", str, "The node's dot-path (or graph node_id)."),
        Param("key", str, "The property key."),
        Param("value", str, "The value to store."),
        Param(
            "value_type",
            str,
            "Optional explicit type: string, integer, float, boolean, "
            "array, null, or datetime.",
        ),
    ),
    input_shape=TypeShape.NONE,
    output_shape=TypeShape.SCALAR,
    domain="ingest",
)
def set_value(
    endpoint: str,
    path: str,
    key: str,
    value: Any,
    value_type: str | None = None,
) -> Any:
    kind = resolve_store_kind(endpoint, PROPERTY_KINDS)
    python_value, declared = _typed(value, value_type)
    if kind == "graph":
        return graph_store.set_property(endpoint, path, key, python_value, declared)
    return tree_store.set_property(endpoint, path, key, python_value, declared)


@tool_spec(
    name="delete_key",
    summary=(
        "Delete one property from a node of a declared read-write store endpoint."
    ),
    params=(
        Param("endpoint", str, "The operator-declared store endpoint name."),
        Param("path", str, "The node's dot-path (or graph node_id)."),
        Param("key", str, "The property key."),
    ),
    input_shape=TypeShape.NONE,
    output_shape=TypeShape.SCALAR,
    domain="ingest",
)
def delete_key(endpoint: str, path: str, key: str) -> Any:
    kind = resolve_store_kind(endpoint, PROPERTY_KINDS)
    if kind == "graph":
        deleted = graph_store.delete_property(endpoint, path, key)
    else:
        deleted = tree_store.delete_property(endpoint, path, key)
    if not deleted:
        raise missing_entity_refusal(
            f"Property {key!r} not found on node {path!r}.", _LIST_KEYS_HINT
        )
    return {"path": path, "key": key, "deleted": True}


@tool_spec(
    name="list_keys",
    summary=(
        "List a node's properties (key, value, value_type) from a "
        "declared kv, tree, or graph store endpoint, key-ordered."
    ),
    params=(
        Param("endpoint", str, "The operator-declared store endpoint name."),
        Param("path", str, "The node's dot-path (or graph node_id)."),
        Param("offset", int, "Pagination offset (default 0)."),
        Param("limit", int, "Optional page size; omitted serves all rows."),
    ),
    input_shape=TypeShape.NONE,
    output_shape=TypeShape.TABULAR,
    domain="ingest",
)
def list_keys(
    endpoint: str,
    path: str,
    offset: int = 0,
    limit: int | None = None,
) -> Any:
    kind = resolve_store_kind(endpoint, PROPERTY_KINDS)
    if kind == "graph":
        if not graph_store.node_exists(endpoint, path):
            raise missing_entity_refusal(f"Node not found: {path}", _LIST_KEYS_HINT)
        raw = graph_store.list_properties(endpoint, path, offset, limit)
        rows = [
            (row[0], deserialize_value(row[1], ValueType(row[2])), row[2])
            for row in raw.rows
        ]
    else:
        if not tree_store.node_exists(endpoint, path):
            raise missing_entity_refusal(f"Node not found: {path}", _LIST_KEYS_HINT)
        raw = tree_store.list_properties(endpoint, path, offset, limit)
        rows = [
            (row[0], deserialize_value(row[1], ValueType(row[2]), row[3]), row[2])
            for row in raw.rows
        ]
    return Result(
        columns=("key", "value", "value_type"),
        rows=tuple(rows),
        category="query",
    )
