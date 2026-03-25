"""MCP mutation and property tool functions for graphs.

Contains set_node, delete_node, add_edge, remove_edge, and property
tools (get/set/delete/list_keys).  Separated from
:mod:`graph_tools` to keep files under the size limit.
"""

from typing import Any, Dict, List, Optional

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
    return {
        "created": not existed,
        "node_id": node.node_id,
        "label": node.label,
    }


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

    return {
        "source": edge.source_id,
        "target": edge.target_id,
        "label": edge.label,
        "weight": edge.weight,
        "nodes_created": nodes_created,
    }


def tool_remove_edge(
    manager: GraphStorageManager,
    name: str,
    source: str,
    target: str,
    label: Optional[str] = None,
) -> Dict[str, Any]:
    """Remove an edge."""
    removed = manager.remove_edge(source, target, label=label)
    return {
        "source": source,
        "target": target,
        "label": label,
        "removed": removed,
    }


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
