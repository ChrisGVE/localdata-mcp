"""MCP tool functions for tree navigation and mutation.

Each function takes a :class:`TreeStorageManager` plus tool-specific
parameters and returns a dict (FastMCP v3 handles serialization).
"""

from typing import Any, Dict, List, Optional

from .tree_storage import (
    TreeStorageManager,
    ValueType,
    build_path,
    deserialize_value,
    infer_value_type_from_string,
    parse_path,
)


def _node_summary(manager: TreeStorageManager, path: str) -> Dict[str, Any]:
    """Build a compact summary dict for a single node."""
    return {
        "path": path,
        "name": parse_path(path)[-1] if path else "",
        "property_count": manager.get_property_count(path),
        "children_count": manager.get_children_count(path),
    }


def _property_to_dict(prop) -> Dict[str, Any]:
    """Convert a NodeProperty to a serialisable dict."""
    return {
        "key": prop.key,
        "value": deserialize_value(prop.value, prop.value_type, prop.original_repr),
        "value_type": prop.value_type.value,
    }


# -- Navigation tools -------------------------------------------------------


def tool_get_node(
    manager: TreeStorageManager,
    name: str,
    path: Optional[str] = None,
) -> Dict[str, Any]:
    """Return node details, or root summary when *path* is ``None``."""
    if path is None:
        stats = manager.get_tree_stats()
        return {
            "root_nodes": stats["root_nodes"],
            "total_nodes": stats["total_nodes"],
            "total_properties": stats["total_properties"],
            "max_depth": stats["max_depth"],
        }

    node = manager.get_node(path)
    if node is None:
        return {"error": f"Node not found: {path}"}

    prop_count = manager.get_property_count(path)
    children_count = manager.get_children_count(path)
    segments = parse_path(path)
    parent_path = build_path(segments[:-1]) if len(segments) > 1 else None

    result: Dict[str, Any] = {
        "path": path,
        "name": node.name,
        "depth": node.depth,
        "is_array_item": node.is_array_item,
        "parent_path": parent_path,
        "children_count": children_count,
        "property_count": prop_count,
    }

    if prop_count <= 100:
        props = manager.list_properties(path, offset=0, limit=100)
        result["properties"] = [_property_to_dict(p) for p in props]
    else:
        result["properties_truncated"] = True
        result["hint"] = "Use list_keys with pagination to browse properties."

    return result


def tool_get_children(
    manager: TreeStorageManager,
    name: str,
    path: Optional[str] = None,
    offset: int = 0,
    limit: int = 50,
) -> Dict[str, Any]:
    """Return paginated children of a node (or root nodes)."""
    children = manager.get_children(parent_path=path, offset=offset, limit=limit)
    total = manager.get_children_count(parent_path=path)
    items = [_node_summary(manager, c.path) for c in children]
    return {
        "parent_path": path,
        "children": items,
        "total": total,
        "offset": offset,
        "limit": limit,
        "has_more": (offset + limit) < total,
    }


# -- Structure mutation tools ------------------------------------------------


def tool_set_node(
    manager: TreeStorageManager,
    name: str,
    path: str,
) -> Dict[str, Any]:
    """Create a node (and any missing ancestors). Idempotent."""
    segments = parse_path(path)
    if not segments:
        return {"error": "Path must not be empty."}

    existing_paths: set[str] = set()
    for depth in range(len(segments)):
        partial = build_path(segments[: depth + 1])
        if manager.node_exists(partial):
            existing_paths.add(partial)

    node = manager.create_node(path)

    ancestors_created: List[str] = []
    for depth in range(len(segments) - 1):
        partial = build_path(segments[: depth + 1])
        if partial not in existing_paths:
            ancestors_created.append(partial)

    return {
        "created": path not in existing_paths,
        "node_id": node.id,
        "path": node.path,
        "ancestors_created": ancestors_created,
    }


def tool_delete_node(
    manager: TreeStorageManager,
    name: str,
    path: str,
) -> Dict[str, Any]:
    """Delete a node and all its descendants."""
    if not manager.node_exists(path):
        return {"error": f"Node not found: {path}"}

    nodes_deleted, properties_deleted = manager.delete_node(path)
    return {
        "path": path,
        "nodes_deleted": nodes_deleted,
        "properties_deleted": properties_deleted,
    }


# -- Property mutation tools -------------------------------------------------


def tool_list_keys(
    manager: TreeStorageManager,
    name: str,
    path: str,
    offset: int = 0,
    limit: int = 50,
) -> Dict[str, Any]:
    """Return paginated property keys for a node."""
    if not manager.node_exists(path):
        return {"error": f"Node not found: {path}"}

    props = manager.list_properties(path, offset=offset, limit=limit)
    total = manager.get_property_count(path)
    return {
        "path": path,
        "keys": [_property_to_dict(p) for p in props],
        "total": total,
        "offset": offset,
        "limit": limit,
        "has_more": (offset + limit) < total,
    }


def tool_get_value(
    manager: TreeStorageManager,
    name: str,
    path: str,
    key: str,
) -> Dict[str, Any]:
    """Return a single property value."""
    prop = manager.get_property(path, key)
    if prop is None:
        return {"error": f"Property '{key}' not found on node '{path}'."}
    return _property_to_dict(prop)


def tool_set_value(
    manager: TreeStorageManager,
    name: str,
    path: str,
    key: str,
    value: Any,
    value_type: Optional[str] = None,
) -> Dict[str, Any]:
    """Set a property on a node with optional explicit type.

    When *value_type* is ``None`` and *value* is a string, type inference
    from the string content is applied (e.g. ``"42"`` becomes an integer).
    """
    vt: Optional[ValueType] = None
    python_value: Any = value

    if value_type is not None:
        vt = ValueType(value_type)
    elif isinstance(value, str):
        vt, python_value = infer_value_type_from_string(value)

    prop = manager.set_property(path, key, python_value, vt)
    return {
        "path": path,
        "key": prop.key,
        "value": deserialize_value(prop.value, prop.value_type, prop.original_repr),
        "value_type": prop.value_type.value,
    }


def tool_delete_key(
    manager: TreeStorageManager,
    name: str,
    path: str,
    key: str,
) -> Dict[str, Any]:
    """Delete a property from a node."""
    deleted = manager.delete_property(path, key)
    if not deleted:
        return {"error": f"Property '{key}' not found on node '{path}'."}
    return {"path": path, "key": key, "deleted": True}
