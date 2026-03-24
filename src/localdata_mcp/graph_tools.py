"""MCP tool functions for graph navigation, mutation, and export.

Each function takes a :class:`GraphStorageManager` plus tool-specific
parameters and returns a dict (FastMCP v3 handles serialization).
"""

from typing import Any, Dict, List, Optional

import networkx as nx
from sqlalchemy import text

from .graph_storage import GraphStorageManager, GraphProperty
from .tree_storage import ValueType, deserialize_value, infer_value_type_from_string

# ---------------------------------------------------------------------------
# Maximum export payload size (bytes)
# ---------------------------------------------------------------------------

MAX_EXPORT_BYTES = 100 * 1024  # 100 KB


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


def _storage_to_networkx(manager: GraphStorageManager) -> nx.DiGraph:
    """Reconstruct a NetworkX DiGraph from graph storage.

    Iterates over all nodes and edges in the storage manager and
    builds a directed graph suitable for NetworkX algorithms.
    """
    G = nx.DiGraph()

    # Load all nodes
    offset = 0
    limit = 500
    while True:
        nodes = manager.list_nodes(offset=offset, limit=limit)
        for node in nodes:
            attrs: Dict[str, Any] = {}
            if node.label:
                attrs["label"] = node.label
            G.add_node(node.node_id, **attrs)
        if len(nodes) < limit:
            break
        offset += limit

    # Load all edges
    offset = 0
    while True:
        edges = manager.list_edges(offset=offset, limit=limit)
        for edge in edges:
            attrs = {}
            if edge.label:
                attrs["label"] = edge.label
            if edge.weight is not None:
                attrs["weight"] = edge.weight
            G.add_edge(edge.source_id, edge.target_id, **attrs)
        if len(edges) < limit:
            break
        offset += limit

    return G


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
                if m["source_id"] == node_id:
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
# Algorithm tools
# ---------------------------------------------------------------------------


def tool_find_path(
    manager: GraphStorageManager,
    name: str,
    source: str,
    target: str,
    algorithm: str = "shortest",
) -> Dict[str, Any]:
    """Find path(s) between two nodes using NetworkX.

    Supports "shortest" (single shortest path) and "all" (up to 10
    simple paths with max depth 20).
    """
    if algorithm not in ("shortest", "all"):
        return {"error": (f"Unknown algorithm: {algorithm}. Use 'shortest' or 'all'.")}

    if not manager.node_exists(source):
        return {"error": f"Source node not found: {source}"}
    if not manager.node_exists(target):
        return {"error": f"Target node not found: {target}"}

    G = _storage_to_networkx(manager)

    if algorithm == "shortest":
        try:
            path = nx.shortest_path(G, source, target)
        except nx.NetworkXNoPath:
            return {
                "source": source,
                "target": target,
                "algorithm": algorithm,
                "path": None,
                "path_length": None,
                "message": "No path exists between source and target.",
            }
        return {
            "source": source,
            "target": target,
            "algorithm": algorithm,
            "path": path,
            "path_length": len(path) - 1,
        }

    # algorithm == "all"
    try:
        all_paths = list(nx.all_simple_paths(G, source, target, cutoff=20))
    except nx.NetworkXNoPath:
        all_paths = []

    # Limit to 10 paths
    paths = all_paths[:10]
    return {
        "source": source,
        "target": target,
        "algorithm": algorithm,
        "paths": paths,
        "paths_count": len(paths),
        "total_paths_found": len(all_paths),
        "truncated": len(all_paths) > 10,
    }


def tool_get_graph_stats(
    manager: GraphStorageManager,
    name: str,
) -> Dict[str, Any]:
    """Compute advanced graph statistics using NetworkX."""
    basic = manager.get_graph_stats()
    node_count = basic["node_count"]
    edge_count = basic["edge_count"]

    result: Dict[str, Any] = {
        "node_count": node_count,
        "edge_count": edge_count,
        "density": basic["density"],
    }

    # Skip expensive NetworkX calculations for large graphs
    if node_count > 10000:
        result["warning"] = (
            f"Graph has {node_count} nodes; skipping expensive "
            "NetworkX calculations (is_dag, components, degree stats)."
        )
        return result

    G = _storage_to_networkx(manager)

    result["is_dag"] = nx.is_directed_acyclic_graph(G)

    # Connected components on undirected view
    undirected = G.to_undirected()
    result["connected_components"] = nx.number_connected_components(undirected)

    # Degree statistics
    if node_count > 0:
        in_degrees = dict(G.in_degree())
        out_degrees = dict(G.out_degree())
        total_degree = sum(in_degrees.values()) + sum(out_degrees.values())
        result["average_degree"] = total_degree / node_count

        max_in_node = max(in_degrees, key=in_degrees.get)
        max_out_node = max(out_degrees, key=out_degrees.get)
        result["max_in_degree"] = {
            "node_id": max_in_node,
            "in_degree": in_degrees[max_in_node],
        }
        result["max_out_degree"] = {
            "node_id": max_out_node,
            "out_degree": out_degrees[max_out_node],
        }
    else:
        result["average_degree"] = 0.0

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


# ---------------------------------------------------------------------------
# Graph export
# ---------------------------------------------------------------------------


def _export_dot(G: nx.DiGraph) -> str:
    """Export graph to DOT format via pydot."""
    import pydot

    P = nx.drawing.nx_pydot.to_pydot(G)
    return P.to_string()


def _export_gml(G: nx.DiGraph) -> str:
    """Export graph to GML format."""
    return "\n".join(nx.generate_gml(G))


def _export_graphml(G: nx.DiGraph) -> str:
    """Export graph to GraphML format."""
    return "\n".join(nx.generate_graphml(G))


_GRAPH_EXPORTERS = {
    "dot": _export_dot,
    "gml": _export_gml,
    "graphml": _export_graphml,
}


def tool_export_graph(
    manager: GraphStorageManager,
    name: str,
    format: str,
    node_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Export graph data in the requested format (dot, gml, graphml).

    When *node_id* is provided, exports only the ego graph (the node
    and its immediate neighbors).
    """
    fmt = format.lower()
    if fmt not in _GRAPH_EXPORTERS:
        return {"error": (f"Unsupported format '{format}'. Use dot, gml, or graphml.")}

    G = _storage_to_networkx(manager)

    if node_id is not None:
        if node_id not in G:
            return {"error": f"Node not found: {node_id}"}
        G = nx.ego_graph(G, node_id)

    try:
        output = _GRAPH_EXPORTERS[fmt](G)
    except Exception as exc:
        return {"error": f"Export failed: {exc}"}

    if len(output.encode("utf-8")) > MAX_EXPORT_BYTES:
        truncated = output[: MAX_EXPORT_BYTES // 2]
        return {
            "format": fmt,
            "truncated": True,
            "content": truncated,
            "notice": (
                f"Output exceeded {MAX_EXPORT_BYTES // 1024}KB limit. "
                "Use node_id to export a smaller subgraph."
            ),
        }

    return {"format": fmt, "content": output}
