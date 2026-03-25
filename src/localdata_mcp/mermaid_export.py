"""Mermaid flowchart export from NetworkX graphs.

Converts a NetworkX MultiDiGraph (or DiGraph with edge attributes)
back into Mermaid flowchart syntax.
"""

from typing import Dict, List

import networkx as nx

# Reverse shape mapping: shape name -> Mermaid syntax wrappers
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

# Reverse edge style mapping: (style, directed) -> Mermaid operator
_STYLE_OP = {
    ("solid", True): "-->",
    ("solid", False): "---",
    ("thick", True): "==>",
    ("dotted", True): "-.->",
}


def _escape_mermaid_label(label: str) -> str:
    """Remove characters that would break Mermaid syntax."""
    return label.replace("[", "(").replace("]", ")").replace("|", "/")


def _emit_mermaid_nodes(G: nx.MultiDiGraph) -> List[str]:
    """Build Mermaid node declaration lines."""
    lines: List[str] = []
    for node_id in sorted(G.nodes()):
        attrs = G.nodes[node_id]
        raw_label = attrs.get("label", "")
        label = _escape_mermaid_label(raw_label) if raw_label else ""
        shape = attrs.get("shape", "rectangle")
        left, right = _SHAPE_WRAP.get(shape, ("[", "]"))
        if label and label != node_id:
            lines.append(f"    {node_id}{left}{label}{right}")
        elif shape != "rectangle":
            lines.append(f"    {node_id}{left}{node_id}{right}")
    return lines


def _emit_mermaid_subgraphs(G: nx.MultiDiGraph) -> List[str]:
    """Build Mermaid subgraph block lines."""
    subgraphs: Dict[str, List[str]] = {}
    for node_id in G.nodes():
        sg = G.nodes[node_id].get("subgraph")
        if sg:
            subgraphs.setdefault(sg, []).append(node_id)

    lines: List[str] = []
    for sg_name, members in sorted(subgraphs.items()):
        lines.append(f"    subgraph {sg_name}")
        for m in sorted(members):
            lines.append(f"        {m}")
        lines.append("    end")
    return lines


def _emit_mermaid_edges(G: nx.MultiDiGraph) -> List[str]:
    """Build Mermaid edge lines, deduplicating undirected reverse edges."""
    lines: List[str] = []
    seen_undirected: set = set()

    for u, v, data in G.edges(data=True):
        style = data.get("style", "solid")
        directed = data.get("directed", True)

        if not directed:
            pair = tuple(sorted((u, v)))
            if pair in seen_undirected:
                continue
            seen_undirected.add(pair)

        op = _STYLE_OP.get((style, directed), "-->")
        raw_label = data.get("label", "")
        label = _escape_mermaid_label(raw_label) if raw_label else ""
        if label:
            lines.append(f"    {u} {op}|{label}| {v}")
        else:
            lines.append(f"    {u} {op} {v}")
    return lines


def export_mermaid(G: nx.MultiDiGraph) -> str:
    """Export graph to Mermaid flowchart syntax."""
    direction = G.graph.get("direction", "TD")
    lines = [f"graph {direction}"]
    lines.extend(_emit_mermaid_nodes(G))
    lines.extend(_emit_mermaid_subgraphs(G))
    lines.extend(_emit_mermaid_edges(G))
    return "\n".join(lines) + "\n"
