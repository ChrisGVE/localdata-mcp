"""Graph-specific markdown export with summary, tables, and Mermaid.

Builds on the shared table generator in ``markdown_export`` to produce
LLM-friendly markdown for graph structures.
"""

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


def export_graph_markdown(
    nodes: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
    stats: Optional[Dict[str, Any]] = None,
    title: Optional[str] = None,
    include_mermaid: bool = True,
    max_rows: int = 50,
) -> Dict[str, Any]:
    """Export graph as markdown with summary, tables, and optional Mermaid."""
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
