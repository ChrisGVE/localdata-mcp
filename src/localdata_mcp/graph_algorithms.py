"""Graph algorithm tools using NetworkX.

Contains ``_storage_to_networkx``, ``tool_find_path``,
``tool_get_graph_stats``, and ``tool_export_graph``.
"""

from typing import Any, Dict, List, Optional

import networkx as nx

from .graph_manager import GraphStorageManager

# ---------------------------------------------------------------------------
# Maximum export payload size (bytes)
# ---------------------------------------------------------------------------

MAX_EXPORT_BYTES = 100 * 1024  # 100 KB


# ---------------------------------------------------------------------------
# Storage → NetworkX conversion
# ---------------------------------------------------------------------------


def _storage_to_networkx(manager: GraphStorageManager) -> nx.MultiDiGraph:
    """Reconstruct a NetworkX MultiDiGraph from graph storage.

    Iterates over all nodes and edges in the storage manager and
    builds a directed multigraph suitable for NetworkX algorithms.
    """
    G = nx.MultiDiGraph()

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
    result["connected_components"] = nx.number_weakly_connected_components(G)

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


def _export_mermaid(G: nx.MultiDiGraph) -> str:
    """Export graph to Mermaid flowchart syntax."""
    # Reverse shape mapping: shape name → Mermaid syntax wrappers
    _SHAPE_WRAP = {
        "rectangle": ("[", "]"),
        "rounded": ("(", ")"),
        "diamond": ("{", "}"),
        "circle": ("((", "))"),
        "stadium": ("([", "])"),
        "subroutine": ("[[", "]]"),
        "database": ("[(", ")]"),
        "asymmetric": (">", "]"),
    }
    # Reverse edge style mapping: (style, directed) → Mermaid operator
    _STYLE_OP = {
        ("solid", True): "-->",
        ("solid", False): "---",
        ("thick", True): "==>",
        ("dotted", True): "-.->",
    }

    direction = G.graph.get("direction", "TD")
    lines = [f"graph {direction}"]

    # Emit node declarations with shapes and labels
    for node_id in sorted(G.nodes()):
        attrs = G.nodes[node_id]
        label = attrs.get("label", "")
        shape = attrs.get("shape", "rectangle")
        left, right = _SHAPE_WRAP.get(shape, ("[", "]"))
        if label and label != node_id:
            lines.append(f"    {node_id}{left}{label}{right}")
        elif shape != "rectangle":
            lines.append(f"    {node_id}{left}{node_id}{right}")

    # Group nodes by subgraph
    subgraphs: Dict[str, List[str]] = {}
    for node_id in G.nodes():
        sg = G.nodes[node_id].get("subgraph")
        if sg:
            subgraphs.setdefault(sg, []).append(node_id)

    # Emit subgraph blocks
    for sg_name, members in sorted(subgraphs.items()):
        lines.append(f"    subgraph {sg_name}")
        for m in sorted(members):
            lines.append(f"        {m}")
        lines.append("    end")

    # Emit edges
    for u, v, data in G.edges(data=True):
        style = data.get("edge_style", "solid")
        directed = data.get("directed", True)
        op = _STYLE_OP.get((style, directed), "-->")
        label = data.get("label", "")
        if label:
            lines.append(f"    {u} {op}|{label}| {v}")
        else:
            lines.append(f"    {u} {op} {v}")

    return "\n".join(lines) + "\n"


_GRAPH_EXPORTERS = {
    "dot": _export_dot,
    "gml": _export_gml,
    "graphml": _export_graphml,
    "mermaid": _export_mermaid,
}


def tool_export_graph(
    manager: GraphStorageManager,
    name: str,
    format: str,
    node_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Export graph data in the requested format (dot, gml, graphml, mermaid).

    When *node_id* is provided, exports only the ego graph (the node
    and its immediate neighbors).
    """
    fmt = format.lower()
    if fmt == "mmd":
        fmt = "mermaid"
    if fmt not in _GRAPH_EXPORTERS:
        return {
            "error": (
                f"Unsupported format '{format}'. Use dot, gml, graphml, or mermaid."
            )
        }

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
