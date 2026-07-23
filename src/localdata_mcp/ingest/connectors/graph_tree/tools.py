"""localdata_mcp/ingest/connectors/graph_tree/tools.py — node tools (E8.3).

I-3's node/structure half (names carried from `main`, semantics
intact — DR GP2): get_node / set_node / delete_node dispatch on the
declared endpoint kind (kv/tree speak the tree schema, graph the
graph schema — the harvested dual), get_children and move_node are
tree-shaped operations (a graph has neighbors, not children). Node
detail returns COUNTS plus the list_keys hint, never an inline
property dump — progressive disclosure, and the property surface has
one home. Mutations cross `guarded_mutation` inside the access layers
(NFR-106/113); this file is tool surface and dispatch only (§3).
Neighbors: tree_store.py/graph_store.py execute; graph_tools.py owns
the edge/path half; warnings.py supplies the harvested signals.
"""

from __future__ import annotations

from typing import Any

from localdata_mcp.nexus.contract.spec import Param, TypeShape, tool_spec

from ...refusals import missing_entity_refusal
from .. import graph_props, graph_store, tree_props, tree_store
from ..store_dispatch import PROPERTY_KINDS, TREE_KINDS, resolve_store_kind
from ..treepaths import build_path, parse_path
from .warnings import casing_warnings

_BROWSE_HINT = (
    "Call get_children(endpoint) to browse tree roots, get_node(endpoint) "
    "for the store summary, or list_keys(endpoint, path) for a node's "
    "properties."
)


def _tree_node_detail(endpoint: str, path: str) -> dict[str, Any]:
    node = tree_store.node_row(endpoint, path)
    if node is None:
        raise missing_entity_refusal(f"Node not found: {path}", _BROWSE_HINT)
    segments = parse_path(path)
    return {
        "path": path,
        "name": node["name"],
        "depth": node["depth"],
        "is_array_item": bool(node["is_array_item"]),
        "parent_path": build_path(segments[:-1]) if len(segments) > 1 else None,
        "children_count": tree_store.children_count(endpoint, path),
        "property_count": tree_props.property_count(endpoint, path),
        "hint": "Use list_keys to browse properties.",
    }


def _graph_node_detail(endpoint: str, node_id: str) -> dict[str, Any]:
    node = graph_store.node_row(endpoint, node_id)
    if node is None:
        raise missing_entity_refusal(f"Node not found: {node_id}", _BROWSE_HINT)
    detail: dict[str, Any] = {
        "node_id": node["node_id"],
        "label": node["label"],
        "property_count": graph_props.property_count(endpoint, node_id),
        "hint": "Use list_keys to browse properties.",
    }
    detail.update(graph_store.degrees(endpoint, node_id))
    return detail


@tool_spec(
    name="get_node",
    summary=(
        "Get node details from a declared tree or graph store endpoint "
        "(counts and addressing; properties via list_keys); omit path "
        "for the store-level summary."
    ),
    params=(
        Param("endpoint", str, "The operator-declared store endpoint name."),
        Param(
            "path",
            str,
            "The node's dot-path (or graph node_id); omit for a summary.",
        ),
    ),
    input_shape=TypeShape.NONE,
    output_shape=TypeShape.SCALAR,
    domain="ingest",
)
def get_node(endpoint: str, path: str | None = None) -> Any:
    kind = resolve_store_kind(endpoint, PROPERTY_KINDS)
    if kind == "graph":
        if path is None:
            return graph_store.graph_stats(endpoint)
        return _graph_node_detail(endpoint, path)
    if path is None:
        return tree_store.tree_stats(endpoint)
    return _tree_node_detail(endpoint, path)


@tool_spec(
    name="set_node",
    summary=(
        "Create a node on a declared read-write store endpoint: tree "
        "kinds create the path (and missing ancestors), graph kinds "
        "upsert the node with an optional label."
    ),
    params=(
        Param("endpoint", str, "The operator-declared store endpoint name."),
        Param("path", str, "The node's dot-path (or graph node_id)."),
        Param("label", str, "Optional label (graph stores only)."),
    ),
    input_shape=TypeShape.NONE,
    output_shape=TypeShape.SCALAR,
    domain="ingest",
)
def set_node(endpoint: str, path: str, label: str | None = None) -> Any:
    kind = resolve_store_kind(endpoint, PROPERTY_KINDS)
    if kind == "graph":
        existed = graph_store.node_exists(endpoint, path)
        node = graph_store.upsert_node(endpoint, path, label)
        result: dict[str, Any] = {
            "created": not existed,
            "node_id": node["node_id"],
            "label": node["label"],
        }
        warned = casing_warnings(endpoint, path)
        if warned:
            result["warnings"] = warned
        return result
    if label is not None:
        raise missing_entity_refusal(
            "Tree nodes carry no label; store it as a property instead.",
            "Use set_value(endpoint, path, key, value) for node data.",
        )
    existed = tree_store.node_exists(endpoint, path)
    node_id, created_paths = tree_store.ensure_node(endpoint, path)
    # created_paths ends with the node itself when it was created; the
    # harvested result names only the ANCESTORS brought into being.
    return {
        "created": not existed,
        "node_id": node_id,
        "path": path,
        "ancestors_created": created_paths[:-1] if not existed else [],
    }


@tool_spec(
    name="delete_node",
    summary=(
        "Delete a node from a declared read-write store endpoint: tree "
        "kinds delete the whole subtree (properties cascade), graph "
        "kinds cascade the node's edges and properties."
    ),
    params=(
        Param("endpoint", str, "The operator-declared store endpoint name."),
        Param("path", str, "The node's dot-path (or graph node_id)."),
    ),
    input_shape=TypeShape.NONE,
    output_shape=TypeShape.SCALAR,
    domain="ingest",
)
def delete_node(endpoint: str, path: str) -> Any:
    kind = resolve_store_kind(endpoint, PROPERTY_KINDS)
    if kind == "graph":
        if not graph_store.node_exists(endpoint, path):
            raise missing_entity_refusal(f"Node not found: {path}", _BROWSE_HINT)
        nodes, edges, properties = graph_store.delete_node_cascade(endpoint, path)
        return {
            "node_id": path,
            "nodes_deleted": nodes,
            "edges_deleted": edges,
            "properties_deleted": properties,
        }
    if not tree_store.node_exists(endpoint, path):
        raise missing_entity_refusal(f"Node not found: {path}", _BROWSE_HINT)
    nodes, properties = tree_store.delete_subtree(endpoint, path)
    return {
        "path": path,
        "nodes_deleted": nodes,
        "properties_deleted": properties,
    }


@tool_spec(
    name="get_children",
    summary=(
        "List direct children of a tree-store node (root nodes when "
        "path is omitted), name-ordered with counts."
    ),
    params=(
        Param("endpoint", str, "The operator-declared store endpoint name."),
        Param("path", str, "The parent's dot-path; omit for root nodes."),
        Param("offset", int, "Pagination offset (default 0)."),
        Param("limit", int, "Optional page size; omitted serves all rows."),
    ),
    input_shape=TypeShape.NONE,
    output_shape=TypeShape.TABULAR,
    domain="ingest",
)
def get_children(
    endpoint: str,
    path: str | None = None,
    offset: int = 0,
    limit: int | None = None,
) -> Any:
    resolve_store_kind(endpoint, TREE_KINDS)
    if path is not None and not tree_store.node_exists(endpoint, path):
        raise missing_entity_refusal(f"Node not found: {path}", _BROWSE_HINT)
    return tree_store.children(endpoint, path, offset, limit)


@tool_spec(
    name="move_node",
    summary=(
        "Move a tree-store node and its whole subtree under a new "
        "parent (or to root level when new_parent is omitted)."
    ),
    params=(
        Param("endpoint", str, "The operator-declared store endpoint name."),
        Param("path", str, "The node's dot-path."),
        Param("new_parent", str, "Target parent path; omit for root."),
    ),
    input_shape=TypeShape.NONE,
    output_shape=TypeShape.SCALAR,
    domain="ingest",
)
def move_node(endpoint: str, path: str, new_parent: str | None = None) -> Any:
    resolve_store_kind(endpoint, TREE_KINDS)
    try:
        nodes_moved, new_path = tree_store.move_subtree(endpoint, path, new_parent)
    except ValueError as invalid:
        raise missing_entity_refusal(str(invalid), _BROWSE_HINT) from None
    return {
        "old_path": path,
        "new_path": new_path,
        "new_parent": new_parent,
        "nodes_moved": nodes_moved,
    }
