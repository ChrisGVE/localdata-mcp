"""MCP tool functions for graph navigation.

Navigation tools (get_node, get_neighbors, get_edges) live here.
Mutation and property tools live in :mod:`graph_mutation_tools`.
Algorithm tools live in :mod:`graph_algorithms`.
Everything is re-exported from this module for backward compatibility.
"""

from typing import Any, Dict, List, Optional

from sqlalchemy import text

# Re-export algorithm tools for backward compatibility
from .graph_algorithms import (  # noqa: F401
    MAX_EXPORT_BYTES,
    _storage_to_networkx,
    tool_export_graph,
    tool_find_path,
    tool_get_graph_stats,
)
from .graph_manager import GraphStorageManager

# Re-export mutation/property tools for backward compatibility
from .graph_mutation_tools import (  # noqa: F401
    tool_add_edge,
    tool_delete_key_graph,
    tool_delete_node_graph,
    tool_get_value_graph,
    tool_list_keys_graph,
    tool_remove_edge,
    tool_set_node_graph,
    tool_set_value_graph,
)
from .graph_storage import GraphProperty
from .tree_storage import deserialize_value


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


def tool_get_node_graph(
    manager: GraphStorageManager,
    name: str,
    node_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Return graph summary or node details."""
    if node_id is None:
        return _graph_summary(manager)
    return _node_detail(manager, node_id)


def _graph_summary(manager: GraphStorageManager) -> Dict[str, Any]:
    """Return graph-level summary statistics."""
    stats = manager.get_graph_stats()
    first_nodes = manager.list_nodes(offset=0, limit=20)
    return {
        "node_count": stats["node_count"],
        "edge_count": stats["edge_count"],
        "density": stats["density"],
        "is_directed": stats["is_directed"],
        "first_node_ids": [n.node_id for n in first_nodes],
    }


def _node_detail(manager: GraphStorageManager, node_id: str) -> Dict[str, Any]:
    """Return detailed information about a single node."""
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
    props = manager.list_properties("node", node_id, offset=0, limit=10)
    result["properties"] = [_property_to_dict(p) for p in props]
    if prop_count > 10:
        result["properties_truncated"] = True
        result["hint"] = "Use list_keys with pagination to browse all properties."
    return result


def _query_neighbors(
    manager: GraphStorageManager,
    node_id: str,
    direction: str,
    offset: int,
    limit: int,
) -> tuple:
    """Core neighbor query returning (items, total)."""
    with manager.engine.connect() as conn:
        if direction == "out":
            return _query_directional(conn, node_id, offset, limit, outgoing=True)
        if direction == "in":
            return _query_directional(conn, node_id, offset, limit, outgoing=False)
        return _query_both_neighbors(conn, node_id, offset, limit)


def _query_directional(
    conn,
    node_id: str,
    offset: int,
    limit: int,
    outgoing: bool,
) -> tuple:
    """Query neighbors in a single direction (out or in)."""
    if outgoing:
        col, where_col, dir_label = "target_id", "source_id", "out"
    else:
        col, where_col, dir_label = "source_id", "target_id", "in"
    rows = conn.execute(
        text(
            f"SELECT {col}, label, weight FROM graph_edges "
            f"WHERE {where_col} = :nid "
            f"ORDER BY {col} LIMIT :lim OFFSET :off"
        ),
        {"nid": node_id, "lim": limit, "off": offset},
    ).fetchall()
    total = conn.execute(
        text(f"SELECT COUNT(*) FROM graph_edges WHERE {where_col} = :nid"),
        {"nid": node_id},
    ).fetchone()[0]
    items = [
        {
            "neighbor_id": r._mapping[col],
            "edge_label": r._mapping["label"],
            "edge_weight": r._mapping["weight"],
            "direction": dir_label,
        }
        for r in rows
    ]
    return items, total


def _classify_neighbor(m, node_id: str) -> Dict[str, Any]:
    """Classify a single edge row for the 'both' direction query."""
    if m["source_id"] == m["target_id"] == node_id:
        nid, d = node_id, "self"
    elif m["source_id"] == node_id:
        nid, d = m["target_id"], "out"
    else:
        nid, d = m["source_id"], "in"
    return {
        "neighbor_id": nid,
        "edge_label": m["label"],
        "edge_weight": m["weight"],
        "direction": d,
    }


def _query_both_neighbors(conn, node_id: str, offset: int, limit: int) -> tuple:
    """Query neighbors in both directions."""
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
    return [_classify_neighbor(r._mapping, node_id) for r in rows], total


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
    items, total = _query_neighbors(manager, node_id, direction, offset, limit)
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
    total = (
        manager.get_edge_count()
        if node_id is None
        else _edge_count_for_node(manager, node_id)
    )
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
