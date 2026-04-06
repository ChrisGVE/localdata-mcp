"""MCP mutation and property tool functions for graphs.

Contains set_node, delete_node, add_edge, remove_edge, and property
tools (get/set/delete/list_keys).  Separated from
:mod:`graph_tools` to keep files under the size limit.
"""

from typing import Any, Dict, List, Optional

from sqlalchemy import text

from .graph_manager import GraphStorageManager
from .graph_storage import GraphProperty
from .tree_storage import ValueType, deserialize_value, infer_value_type_from_string

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _property_to_dict(prop: GraphProperty) -> Dict[str, Any]:
    """Convert a GraphProperty to a serialisable dict."""
    return {
        "key": prop.key,
        "value": deserialize_value(prop.value, prop.value_type),
        "value_type": prop.value_type.value,
    }


def _edge_warnings(
    manager: GraphStorageManager,
    source: str,
    target: str,
    label: Optional[str],
) -> List[Dict[str, Any]]:
    """Targeted warnings after adding an edge."""
    warnings: List[Dict[str, Any]] = []
    if source == target:
        warnings.append({"code": "self_loop", "message": f"Self-loop on '{source}'"})
    if not label:
        warnings.append(
            {
                "code": "missing_edge_labels",
                "message": f"Edge {source}→{target} has no label",
            }
        )
    with manager.engine.connect() as conn:
        dup = conn.execute(
            text(
                "SELECT COUNT(*) FROM graph_edges "
                "WHERE source_id = :s AND target_id = :t "
                "AND (label = :l OR (label IS NULL AND :l IS NULL))"
            ),
            {"s": source, "t": target, "l": label},
        ).fetchone()[0]
        if dup > 1:
            warnings.append(
                {
                    "code": "duplicate_edges",
                    "message": f"Edge {source}→{target}"
                    + (f" [{label}]" if label else "")
                    + f" now exists {dup} times",
                }
            )
        if label:
            rev = conn.execute(
                text(
                    "SELECT COUNT(*) FROM graph_edges "
                    "WHERE source_id = :t AND target_id = :s AND label = :l"
                ),
                {"s": source, "t": target, "l": label},
            ).fetchone()[0]
            if rev > 0:
                warnings.append(
                    {
                        "code": "contradictory_edges",
                        "message": f"Both {source}→{target} and "
                        f"{target}→{source} labeled '{label}'",
                    }
                )
    return warnings


def _orphan_warnings(
    manager: GraphStorageManager, *node_ids: str
) -> List[Dict[str, Any]]:
    """Check if any of the given nodes became orphans."""
    warnings: List[Dict[str, Any]] = []
    for nid in node_ids:
        if not manager.node_exists(nid):
            continue
        with manager.engine.connect() as conn:
            has = conn.execute(
                text(
                    "SELECT EXISTS(SELECT 1 FROM graph_edges "
                    "WHERE source_id = :n OR target_id = :n)"
                ),
                {"n": nid},
            ).fetchone()[0]
        if not has:
            warnings.append(
                {
                    "code": "orphan_nodes",
                    "message": f"Node '{nid}' is now an orphan (no edges)",
                }
            )
    return warnings


def _casing_warnings(
    manager: GraphStorageManager, node_id: str
) -> List[Dict[str, Any]]:
    """Check for node ID casing conflicts."""
    with manager.engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT node_id FROM graph_nodes "
                "WHERE LOWER(node_id) = LOWER(:nid) AND node_id != :nid"
            ),
            {"nid": node_id},
        ).fetchall()
    if not rows:
        return []
    others = [r[0] for r in rows]
    return [
        {
            "code": "duplicate_casing",
            "message": f"'{node_id}' has casing variants: {', '.join(others)}",
        }
    ]


# ---------------------------------------------------------------------------
# Mutation tools
# ---------------------------------------------------------------------------


def tool_set_node_graph(
    manager: GraphStorageManager,
    name: str,
    node_id: str,
    label: Optional[str] = None,
) -> Dict[str, Any]:
    """Create or update a node."""
    existed = manager.node_exists(node_id)
    node = manager.create_node(node_id, label=label)
    result: Dict[str, Any] = {
        "created": not existed,
        "node_id": node.node_id,
        "label": node.label,
    }
    warnings = _casing_warnings(manager, node_id)
    if warnings:
        result["warnings"] = warnings
    return result


def tool_delete_node_graph(
    manager: GraphStorageManager,
    name: str,
    node_id: str,
) -> Dict[str, Any]:
    """Delete a node and cascade edges and properties."""
    if not manager.node_exists(node_id):
        return {"error": f"Node not found: {node_id}"}

    nodes_del, edges_del, props_del = manager.delete_node(node_id)
    return {
        "node_id": node_id,
        "nodes_deleted": nodes_del,
        "edges_deleted": edges_del,
        "properties_deleted": props_del,
    }


def tool_add_edge(
    manager: GraphStorageManager,
    name: str,
    source: str,
    target: str,
    label: Optional[str] = None,
    weight: Optional[float] = None,
) -> Dict[str, Any]:
    """Create an edge, auto-creating source/target nodes if needed."""
    src_existed = manager.node_exists(source)
    tgt_existed = manager.node_exists(target)

    edge = manager.add_edge(source, target, label=label, weight=weight)

    nodes_created: List[str] = []
    if not src_existed:
        nodes_created.append(source)
    if not tgt_existed:
        nodes_created.append(target)

    result: Dict[str, Any] = {
        "source": edge.source_id,
        "target": edge.target_id,
        "label": edge.label,
        "weight": edge.weight,
        "nodes_created": nodes_created,
    }
    warnings = _edge_warnings(manager, source, target, label)
    if warnings:
        result["warnings"] = warnings
    return result


def tool_remove_edge(
    manager: GraphStorageManager,
    name: str,
    source: str,
    target: str,
    label: Optional[str] = None,
) -> Dict[str, Any]:
    """Remove an edge."""
    removed = manager.remove_edge(source, target, label=label)
    result: Dict[str, Any] = {
        "source": source,
        "target": target,
        "label": label,
        "removed": removed,
    }
    if removed:
        warnings = _orphan_warnings(manager, source, target)
        if warnings:
            result["warnings"] = warnings
    return result


# ---------------------------------------------------------------------------
# Property tools
# ---------------------------------------------------------------------------


def tool_get_value_graph(
    manager: GraphStorageManager,
    name: str,
    node_id: str,
    key: str,
) -> Dict[str, Any]:
    """Get a property value from a node."""
    prop = manager.get_property("node", node_id, key)
    if prop is None:
        return {"error": f"Property '{key}' not found on node '{node_id}'."}
    return _property_to_dict(prop)


def tool_set_value_graph(
    manager: GraphStorageManager,
    name: str,
    node_id: str,
    key: str,
    value: Any,
    value_type: Optional[str] = None,
) -> Dict[str, Any]:
    """Set a property on a node with optional explicit type."""
    vt: Optional[ValueType] = None
    python_value: Any = value

    if value_type is not None:
        vt = ValueType(value_type)
        if isinstance(value, str):
            python_value = deserialize_value(value, vt)
    elif isinstance(value, str):
        vt, python_value = infer_value_type_from_string(value)

    if not manager.node_exists(node_id):
        manager.create_node(node_id)

    prop = manager.set_property("node", node_id, key, python_value, vt)
    return {
        "node_id": node_id,
        "key": prop.key,
        "value": deserialize_value(prop.value, prop.value_type),
        "value_type": prop.value_type.value,
    }


def tool_delete_key_graph(
    manager: GraphStorageManager,
    name: str,
    node_id: str,
    key: str,
) -> Dict[str, Any]:
    """Delete a property from a node."""
    deleted = manager.delete_property("node", node_id, key)
    if not deleted:
        return {"error": f"Property '{key}' not found on node '{node_id}'."}
    return {"node_id": node_id, "key": key, "deleted": True}


def tool_list_keys_graph(
    manager: GraphStorageManager,
    name: str,
    node_id: str,
    offset: int = 0,
    limit: int = 50,
) -> Dict[str, Any]:
    """List properties on a node with pagination."""
    if not manager.node_exists(node_id):
        return {"error": f"Node not found: {node_id}"}

    props = manager.list_properties("node", node_id, offset=offset, limit=limit)
    total = manager.get_property_count("node", node_id)
    return {
        "node_id": node_id,
        "keys": [_property_to_dict(p) for p in props],
        "total": total,
        "offset": offset,
        "limit": limit,
        "has_more": (offset + limit) < total,
    }
