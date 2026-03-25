"""Graph algorithm tools using NetworkX.

Contains ``_storage_to_networkx``, ``tool_find_path``,
``tool_get_graph_stats``, and ``tool_export_graph``.
"""

from typing import Any, Dict, List, Optional

import networkx as nx

from .graph_manager import GraphStorageManager

MAX_EXPORT_BYTES = 100 * 1024  # 100 KB


def _storage_to_networkx(manager: GraphStorageManager) -> nx.MultiDiGraph:
    """Reconstruct a NetworkX MultiDiGraph from graph storage."""
    G = nx.MultiDiGraph()
    _load_nodes(G, manager)
    _load_edges(G, manager)
    return G


def _load_nodes(G: nx.MultiDiGraph, manager: GraphStorageManager) -> None:
    """Load all nodes from storage into *G*."""
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


def _load_edges(G: nx.MultiDiGraph, manager: GraphStorageManager) -> None:
    """Load all edges from storage into *G*."""
    offset = 0
    limit = 500
    while True:
        edges = manager.list_edges(offset=offset, limit=limit)
        for edge in edges:
            attrs: Dict[str, Any] = {}
            if edge.label:
                attrs["label"] = edge.label
            if edge.weight is not None:
                attrs["weight"] = edge.weight
            G.add_edge(edge.source_id, edge.target_id, **attrs)
        if len(edges) < limit:
            break
        offset += limit


def _find_shortest_path(G: nx.MultiDiGraph, source: str, target: str) -> Dict[str, Any]:
    """Find single shortest path between source and target."""
    try:
        path = nx.shortest_path(G, source, target)
    except nx.NetworkXNoPath:
        return {
            "source": source,
            "target": target,
            "algorithm": "shortest",
            "path": None,
            "path_length": None,
            "message": "No path exists between source and target.",
        }
    return {
        "source": source,
        "target": target,
        "algorithm": "shortest",
        "path": path,
        "path_length": len(path) - 1,
    }


def _find_all_paths(G: nx.MultiDiGraph, source: str, target: str) -> Dict[str, Any]:
    """Find all simple paths (up to 10, cutoff 20)."""
    try:
        all_paths = list(nx.all_simple_paths(G, source, target, cutoff=20))
    except nx.NetworkXNoPath:
        all_paths = []
    paths = all_paths[:10]
    return {
        "source": source,
        "target": target,
        "algorithm": "all",
        "paths": paths,
        "paths_count": len(paths),
        "total_paths_found": len(all_paths),
        "truncated": len(all_paths) > 10,
    }


def tool_find_path(
    manager: GraphStorageManager,
    name: str,
    source: str,
    target: str,
    algorithm: str = "shortest",
) -> Dict[str, Any]:
    """Find path(s) between two nodes using NetworkX."""
    if algorithm not in ("shortest", "all"):
        return {"error": f"Unknown algorithm: {algorithm}. Use 'shortest' or 'all'."}
    if not manager.node_exists(source):
        return {"error": f"Source node not found: {source}"}
    if not manager.node_exists(target):
        return {"error": f"Target node not found: {target}"}
    G = _storage_to_networkx(manager)
    if algorithm == "shortest":
        return _find_shortest_path(G, source, target)
    return _find_all_paths(G, source, target)


def tool_get_graph_stats(manager: GraphStorageManager, name: str) -> Dict[str, Any]:
    """Compute advanced graph statistics using NetworkX."""
    basic = manager.get_graph_stats()
    node_count = basic["node_count"]
    result: Dict[str, Any] = {
        "node_count": node_count,
        "edge_count": basic["edge_count"],
        "density": basic["density"],
    }
    if node_count > 10000:
        result["warning"] = (
            f"Graph has {node_count} nodes; skipping expensive "
            "NetworkX calculations (is_dag, components, degree stats)."
        )
        return result
    G = _storage_to_networkx(manager)
    _compute_nx_stats(G, result, node_count)
    return result


def _compute_nx_stats(
    G: nx.MultiDiGraph, result: Dict[str, Any], node_count: int
) -> None:
    """Populate *result* with NetworkX-derived statistics."""
    result["is_dag"] = nx.is_directed_acyclic_graph(G)
    result["connected_components"] = nx.number_weakly_connected_components(G)
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


# ---------------------------------------------------------------------------
# Graph export
# ---------------------------------------------------------------------------


def _export_dot(G: nx.MultiDiGraph) -> str:
    """Export graph to DOT format via pydot."""
    import pydot  # noqa: F811

    P = nx.drawing.nx_pydot.to_pydot(G)
    return P.to_string()


def _export_gml(G: nx.MultiDiGraph) -> str:
    """Export graph to GML format."""
    return "\n".join(nx.generate_gml(G))


def _export_graphml(G: nx.MultiDiGraph) -> str:
    """Export graph to GraphML format."""
    return "\n".join(nx.generate_graphml(G))


def _truncate_output(output: str, fmt: str) -> Optional[Dict[str, Any]]:
    """Return a truncated result dict if output exceeds MAX_EXPORT_BYTES, else None."""
    if len(output.encode("utf-8")) <= MAX_EXPORT_BYTES:
        return None
    return {
        "format": fmt,
        "truncated": True,
        "content": output[: MAX_EXPORT_BYTES // 2],
        "notice": (
            f"Output exceeded {MAX_EXPORT_BYTES // 1024}KB limit. "
            "Use node_id to export a smaller subgraph."
        ),
    }


def _resolve_format(format: str) -> tuple:
    """Normalize format string and return (fmt, exporters) or (None, error_dict)."""
    from .mermaid_export import export_mermaid

    exporters = {
        "dot": _export_dot,
        "gml": _export_gml,
        "graphml": _export_graphml,
        "mermaid": export_mermaid,
    }
    fmt = format.lower()
    if fmt == "mmd":
        fmt = "mermaid"
    if fmt not in exporters:
        return None, {
            "error": f"Unsupported format '{format}'. Use dot, gml, graphml, or mermaid."
        }
    return fmt, exporters


def tool_export_graph(
    manager: GraphStorageManager,
    name: str,
    format: str,
    node_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Export graph data in the requested format."""
    fmt, exporters = _resolve_format(format)
    if fmt is None:
        return exporters  # error dict

    G = _storage_to_networkx(manager)
    if node_id is not None:
        if node_id not in G:
            return {"error": f"Node not found: {node_id}"}
        G = nx.ego_graph(G, node_id)

    try:
        output = exporters[fmt](G)
    except Exception as exc:
        return {"error": f"Export failed: {exc}"}

    truncated = _truncate_output(output, fmt)
    return truncated if truncated else {"format": fmt, "content": output}
