"""Graph-specific markdown export with summary, tables, and Mermaid.

Builds on the shared table generator in ``markdown_export`` to produce
LLM-friendly markdown for graph structures.
"""

from collections import defaultdict
from typing import Any, Dict, List, Optional

from .markdown_export import MAX_EXPORT_BYTES, generate_markdown_table


def _generate_graph_summary_markdown(stats: Dict[str, Any]) -> str:
    """Generate markdown summary for a graph."""
    lines = ["## Graph Summary", ""]
    lines.append(f"- **Nodes**: {stats.get('node_count', 0)}")
    lines.append(f"- **Edges**: {stats.get('edge_count', 0)}")
    if "density" in stats:
        lines.append(f"- **Density**: {stats['density']:.4f}")
    if "is_dag" in stats:
        lines.append(f"- **DAG**: {'Yes' if stats['is_dag'] else 'No'}")
    if "connected_components" in stats:
        lines.append(f"- **Connected components**: {stats['connected_components']}")
    return "\n".join(lines)


def _generate_node_table_markdown(
    nodes: List[Dict[str, Any]], max_rows: int = 50
) -> str:
    """Generate markdown table of graph nodes."""
    if not nodes:
        return "*No nodes*"
    rows = []
    for node in nodes:
        props = node.get("properties", {})
        prop_str = ", ".join(f"{k}={v}" for k, v in props.items()) if props else "-"
        rows.append(
            [
                str(node.get("id", "")),
                str(node.get("label", node.get("name", ""))),
                prop_str,
            ]
        )
    table_md, _, _ = generate_markdown_table(
        ["ID", "Label", "Properties"], rows, max_rows=max_rows
    )
    return f"## Nodes\n\n{table_md}"


def _generate_edge_table_markdown(
    edges: List[Dict[str, Any]], max_rows: int = 50
) -> str:
    """Generate markdown table of graph edges."""
    if not edges:
        return "*No edges*"
    rows = []
    for edge in edges:
        rows.append(
            [
                str(edge.get("source", "")),
                str(edge.get("target", "")),
                str(edge.get("label", "-")),
                str(edge.get("weight", "-")),
            ]
        )
    table_md, _, _ = generate_markdown_table(
        ["Source", "Target", "Label", "Weight"], rows, max_rows=max_rows
    )
    return f"## Edges\n\n{table_md}"


def _generate_mermaid_block(
    nodes: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
    max_nodes: int = 50,
) -> Optional[str]:
    """Generate optional Mermaid visualization block."""
    if len(nodes) > max_nodes:
        return None
    lines = ["## Visualization", "", "```mermaid", "graph TD"]
    for node in nodes:
        nid = str(node.get("id", "")).replace(" ", "_")
        label = str(node.get("label", node.get("name", nid)))
        lines.append(f"    {nid}[{label}]")
    for edge in edges:
        src = str(edge.get("source", "")).replace(" ", "_")
        tgt = str(edge.get("target", "")).replace(" ", "_")
        elabel = edge.get("label", "")
        if elabel and elabel != "-":
            lines.append(f"    {src} -->|{elabel}| {tgt}")
        else:
            lines.append(f"    {src} --> {tgt}")
    lines.append("```")
    return "\n".join(lines)


def generate_hierarchy_markdown(
    nodes: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
    max_depth: int = 10,
) -> str:
    """Render DAG as indented tree with multi-parent annotations."""
    children_map: Dict[str, List[str]] = defaultdict(list)
    parents_map: Dict[str, List[str]] = defaultdict(list)
    node_lookup = {str(n.get("id", "")): n for n in nodes}

    for e in edges:
        src, tgt = str(e["source"]), str(e["target"])
        children_map[src].append(tgt)
        parents_map[tgt].append(src)

    all_ids = set(node_lookup.keys())
    child_ids = {str(e["target"]) for e in edges}
    roots = sorted(all_ids - child_ids) or sorted(all_ids)[:1]

    def has_cycle(node: str, path: set) -> bool:
        if node in path:
            return True
        path.add(node)
        for child in children_map.get(node, []):
            if has_cycle(child, path):
                return True
        path.discard(node)
        return False

    if any(has_cycle(r, set()) for r in roots):
        return generate_adjacency_markdown(nodes, edges)

    lines = ["## Graph Hierarchy\n"]
    rendered: set = set()

    def render(nid: str, depth: int) -> None:
        if depth > max_depth:
            return
        node = node_lookup.get(nid, {})
        label = node.get("label", node.get("name", nid))
        indent = "  " * depth
        other_parents = [p for p in parents_map.get(nid, []) if p != nid]
        if len(other_parents) > 1 and nid in rendered:
            lines.append(f"{indent}- **{label}** *(see above)*")
            return
        annotation = ""
        if len(other_parents) > 1:
            names = [
                node_lookup.get(p, {}).get("label", p)
                for p in other_parents[1:]
            ]
            annotation = f" *(also child of: {', '.join(names[:3])})*"
        lines.append(f"{indent}- **{label}**{annotation}")
        rendered.add(nid)
        for k, v in list(node.get("properties", {}).items())[:5]:
            lines.append(f"{indent}  - {k}: {v}")
        for child in children_map.get(nid, []):
            render(child, depth + 1)

    for root in roots:
        render(root, 0)
    return "\n".join(lines)


def generate_adjacency_markdown(
    nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]
) -> str:
    """Render graph as compact adjacency list."""
    node_lookup = {str(n.get("id", "")): n for n in nodes}
    lines = ["## Adjacency List\n"]
    for e in sorted(edges, key=lambda e: str(e.get("source", ""))):
        src = node_lookup.get(str(e["source"]), {}).get("label", e["source"])
        tgt = node_lookup.get(str(e["target"]), {}).get("label", e["target"])
        label = e.get("label", "")
        if label and label != "-":
            lines.append(f"{src} -> {tgt} [{label}]")
        else:
            lines.append(f"{src} -> {tgt}")
    return "\n".join(lines)


def generate_detailed_markdown(
    nodes: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
    max_nodes: int = 50,
) -> str:
    """Render full properties per node with outgoing edges."""
    lines = ["## Node Details\n"]
    for node in nodes[:max_nodes]:
        nid = str(node.get("id", ""))
        label = node.get("label", node.get("name", nid))
        lines.append(f"### {label} (id: {nid})")
        for k, v in node.get("properties", {}).items():
            lines.append(f"- **{k}**: {v}")
        out = [e for e in edges if str(e["source"]) == nid]
        if out:
            lines.append("\n**Edges out:**")
            for e in out:
                tgt = e.get("target", "")
                lbl = e.get("label", "")
                lines.append(f"- -> {tgt}" + (f" [{lbl}]" if lbl else ""))
        lines.append("")
    return "\n".join(lines)


def export_graph_markdown(
    nodes: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
    stats: Optional[Dict[str, Any]] = None,
    title: Optional[str] = None,
    include_mermaid: bool = True,
    max_rows: int = 50,
    style: str = "summary",
) -> Dict[str, Any]:
    """Export graph as markdown with summary, tables, and optional Mermaid.

    Args:
        style: One of "summary" (default), "hierarchy", "adjacency",
            or "detailed".
    """
    if style == "hierarchy":
        content = generate_hierarchy_markdown(nodes, edges)
    elif style == "adjacency":
        content = generate_adjacency_markdown(nodes, edges)
    elif style == "detailed":
        content = generate_detailed_markdown(nodes, edges, max_nodes=max_rows)
    else:
        parts: List[str] = []
        if title:
            parts.append(f"# {title}\n")
        if stats:
            parts.append(_generate_graph_summary_markdown(stats))
        parts.append("")
        parts.append(_generate_node_table_markdown(nodes, max_rows=max_rows))
        parts.append("")
        parts.append(_generate_edge_table_markdown(edges, max_rows=max_rows))
        if include_mermaid:
            mermaid = _generate_mermaid_block(nodes, edges)
            if mermaid:
                parts.append("")
                parts.append(mermaid)
        content = "\n".join(parts)
    truncated = False
    if len(content.encode("utf-8")) > MAX_EXPORT_BYTES:
        encoded = content.encode("utf-8")[:MAX_EXPORT_BYTES]
        content = encoded.decode("utf-8", errors="ignore").rsplit("\n", 1)[0]
        content += "\n\n*... output truncated due to size limits*"
        truncated = True
    return {"format": "markdown", "content": content, "truncated": truncated}
