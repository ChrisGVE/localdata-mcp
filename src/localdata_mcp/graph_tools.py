"""MCP tool functions for graph navigation, mutation, and properties.

Each function takes a :class:`GraphStorageManager` plus tool-specific
parameters and returns a dict (FastMCP v3 handles serialization).

Algorithm tools (find_path, get_graph_stats, export_graph) live in
:mod:`graph_algorithms` but are re-exported here for backward
compatibility.
"""

from typing import Any, Dict, List, Optional

from sqlalchemy import text

from .graph_manager import GraphStorageManager
from .graph_storage import GraphProperty
from .tree_storage import ValueType, deserialize_value, infer_value_type_from_string

# Re-export algorithm tools and MAX_EXPORT_BYTES for backward compatibility
from .graph_algorithms import (  # noqa: F401
    MAX_EXPORT_BYTES,
    _storage_to_networkx,
    tool_export_graph,
    tool_find_path,
    tool_get_graph_stats,
)


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


def _get_degree_info(manager: GraphStorageManager, node_id: str) -> Dict[str, int]:
    """Return in_degree and out_degree for a node."""
    with manager.engine.connect() as conn:
        in_deg = conn.execute(
            text("SELECT COUNT(*) FROM graph_edges WHERE target_id = :nid"),
            {"nid": node_id},
        ).fetchone()[0]
        out_deg = conn.execute(
            text("SELECT COUNT(*) FROM graph_edges WHERE source_id = :nid"),
            {"nid": node_id},
        ).fetchone()[0]
    return {"in_degree": in_deg, "out_degree": out_deg}


def _edge_count_for_node(manager: GraphStorageManager, node_id: str) -> int:
    """Return total edges involving a specific node."""
    with manager.engine.connect() as conn:
        return conn.execute(
            text(
                "SELECT COUNT(*) FROM graph_edges "
                "WHERE source_id = :nid OR target_id = :nid"
            ),
            {"nid": node_id},
        ).fetchone()[0]


# ---------------------------------------------------------------------------
# Navigation tools
# ---------------------------------------------------------------------------


def tool_get_node_graph(
    manager: GraphStorageManager,
    name: str,
    node_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Return graph summary or node details.

    When *node_id* is None, returns a graph-level summary.
    When provided, returns details about the specific node.
    """
    if node_id is None:
        stats = manager.get_graph_stats()
        first_nodes = manager.list_nodes(offset=0, limit=20)
        return {
            "node_count": stats["node_count"],
            "edge_count": stats["edge_count"],
            "density": stats["density"],
            "is_directed": stats["is_directed"],
            "first_node_ids": [n.node_id for n in first_nodes],
        }

    node = manager.get_node(node_id)
    if node is None:
        return {"error": f"Node not found: {node_id}"}

    degrees = _get_degree_info(manager, node_id)
    prop_count = manager.get_property_count("node", node_id)

    result: Dict[str, Any] = {
        "node_id": node.node_id,
        "label": node.label,
        "in_degree": degrees["in_degree"],
        "out_degree": degrees["out_degree"],
        "property_count": prop_count,
    }

    if prop_count <= 10:
        props = manager.list_properties("node", node_id, offset=0, limit=10)
        result["properties"] = [_property_to_dict(p) for p in props]
    else:
        first_props = manager.list_properties("node", node_id, offset=0, limit=10)
        result["properties"] = [_property_to_dict(p) for p in first_props]
        result["properties_truncated"] = True
        result["hint"] = "Use list_keys with pagination to browse all properties."

    return result


def tool_get_neighbors(
    manager: GraphStorageManager,
    name: str,
    node_id: str,
    direction: str = "both",
    offset: int = 0,
    limit: int = 50,
) -> Dict[str, Any]:
    """Return neighbor list with edge info, paginated."""
    if not manager.node_exists(node_id):
        return {"error": f"Node not found: {node_id}"}

    if direction not in ("in", "out", "both"):
        return {"error": f"Invalid direction: {direction}. Use 'in', 'out', or 'both'."}

    # Query edges with direction info
    total = 0
    items: List[Dict[str, Any]] = []
    with manager.engine.connect() as conn:
        if direction == "out":
            rows = conn.execute(
                text(
                    "SELECT target_id, label, weight FROM graph_edges "
                    "WHERE source_id = :nid "
                    "ORDER BY target_id LIMIT :lim OFFSET :off"
                ),
                {"nid": node_id, "lim": limit, "off": offset},
            ).fetchall()
            total = conn.execute(
                text("SELECT COUNT(*) FROM graph_edges WHERE source_id = :nid"),
                {"nid": node_id},
            ).fetchone()[0]
            for r in rows:
                m = r._mapping
                items.append(
                    {
                        "neighbor_id": m["target_id"],
                        "edge_label": m["label"],
                        "edge_weight": m["weight"],
                        "direction": "out",
                    }
                )
        elif direction == "in":
            rows = conn.execute(
                text(
                    "SELECT source_id, label, weight FROM graph_edges "
                    "WHERE target_id = :nid "
                    "ORDER BY source_id LIMIT :lim OFFSET :off"
                ),
                {"nid": node_id, "lim": limit, "off": offset},
            ).fetchall()
            total = conn.execute(
                text("SELECT COUNT(*) FROM graph_edges WHERE target_id = :nid"),
                {"nid": node_id},
            ).fetchone()[0]
            for r in rows:
                m = r._mapping
                items.append(
                    {
                        "neighbor_id": m["source_id"],
                        "edge_label": m["label"],
                        "edge_weight": m["weight"],
                        "direction": "in",
                    }
                )
        else:
            # Both directions
            rows = conn.execute(
                text(
                    "SELECT source_id, target_id, label, weight FROM graph_edges "
                    "WHERE source_id = :nid OR target_id = :nid "
                    "ORDER BY source_id, target_id LIMIT :lim OFFSET :off"
                ),
                {"nid": node_id, "lim": limit, "off": offset},
            ).fetchall()
            total = conn.execute(
                text(
                    "SELECT COUNT(*) FROM graph_edges "
                    "WHERE source_id = :nid OR target_id = :nid"
                ),
                {"nid": node_id},
            ).fetchone()[0]
            for r in rows:
                m = r._mapping
                if m["source_id"] == m["target_id"] == node_id:
                    # Self-loop
                    items.append(
                        {
                            "neighbor_id": node_id,
                            "edge_label": m["label"],
                            "edge_weight": m["weight"],
                            "direction": "self",
                        }
                    )
                elif m["source_id"] == node_id:
                    items.append(
                        {
                            "neighbor_id": m["target_id"],
                            "edge_label": m["label"],
                            "edge_weight": m["weight"],
                            "direction": "out",
                        }
                    )
                else:
                    items.append(
                        {
                            "neighbor_id": m["source_id"],
                            "edge_label": m["label"],
                            "edge_weight": m["weight"],
                            "direction": "in",
                        }
                    )

    return {
        "node_id": node_id,
        "direction": direction,
        "neighbors": items,
        "total": total,
        "offset": offset,
        "limit": limit,
        "has_more": (offset + limit) < total,
    }


def tool_get_edges(
    manager: GraphStorageManager,
    name: str,
    node_id: Optional[str] = None,
    offset: int = 0,
    limit: int = 50,
) -> Dict[str, Any]:
    """List edges, optionally filtered by a node."""
    edges = manager.list_edges(node_id=node_id, offset=offset, limit=limit)

    # Get total count
    if node_id is None:
        total = manager.get_edge_count()
    else:
        total = _edge_count_for_node(manager, node_id)

    items = [
        {
            "source": e.source_id,
            "target": e.target_id,
            "label": e.label,
            "weight": e.weight,
        }
        for e in edges
    ]

    return {
        "node_id": node_id,
        "edges": items,
        "total": total,
        "offset": offset,
        "limit": limit,
        "has_more": (offset + limit) < total,
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

    nodes_deleted, edges_deleted, properties_deleted = manager.delete_node(node_id)
    return {
        "node_id": node_id,
        "nodes_deleted": nodes_deleted,
        "edges_deleted": edges_deleted,
        "properties_deleted": properties_deleted,
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

    # Ensure the node exists
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
