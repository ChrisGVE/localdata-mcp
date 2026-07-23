"""renderers/graph.py — a graph payload as a Markdown document (FR-902).

The harvested `graph_markdown_export` + `mermaid_export` design over a
plain payload `{"nodes": [{id, label?}, ...], "edges": [{source,
target, label?}, ...]}`: summary line, node/edge tables, and a mermaid
block for renderers that support it — one consolidated graph artifact
instead of `main`'s two parallel modules.
"""

from __future__ import annotations

import re
from typing import Any, Mapping

from ..interface import ExportError

FORMAT = "graph"

_UNSAFE_MERMAID = re.compile(r"[\"\[\]{}()<>]")


def render(payload: Any) -> bytes:
    nodes, edges = _validated(payload)
    parts = [
        f"# Graph — {len(nodes)} nodes, {len(edges)} edges",
        "",
        _node_table(nodes),
        "",
        _edge_table(edges),
        "",
        _mermaid_block(nodes, edges),
    ]
    return ("\n".join(parts) + "\n").encode("utf-8")


def _validated(
    payload: Any,
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    if not isinstance(payload, Mapping):
        raise ExportError(
            f"graph payload must be a mapping, got {type(payload).__name__}"
        )
    nodes = payload.get("nodes")
    edges = payload.get("edges")
    if not isinstance(nodes, (list, tuple)) or not isinstance(edges, (list, tuple)):
        raise ExportError("graph payload needs 'nodes' and 'edges' lists")
    for node in nodes:
        if not isinstance(node, Mapping) or "id" not in node:
            raise ExportError("every graph node needs an 'id'")
    for edge in edges:
        if (
            not isinstance(edge, Mapping)
            or "source" not in edge
            or ("target" not in edge)
        ):
            raise ExportError("every graph edge needs 'source' and 'target'")
    return list(nodes), list(edges)


def _node_table(nodes: list[Mapping[str, Any]]) -> str:
    lines = ["| id | label |", "| --- | --- |"]
    lines.extend(f"| {node['id']} | {node.get('label', '')} |" for node in nodes)
    return "\n".join(lines)


def _edge_table(edges: list[Mapping[str, Any]]) -> str:
    lines = ["| source | target | label |", "| --- | --- | --- |"]
    lines.extend(
        f"| {edge['source']} | {edge['target']} | {edge.get('label', '')} |"
        for edge in edges
    )
    return "\n".join(lines)


def _safe_label(value: Any) -> str:
    """Mermaid labels with structural characters stripped — the
    harvested `_escape_mermaid_label` posture: a label cannot inject
    mermaid syntax."""
    return _UNSAFE_MERMAID.sub("", str(value))


def _mermaid_block(
    nodes: list[Mapping[str, Any]], edges: list[Mapping[str, Any]]
) -> str:
    lines = ["```mermaid", "graph TD"]
    for node in nodes:
        identifier = _safe_label(node["id"])
        label = _safe_label(node.get("label", node["id"]))
        lines.append(f'    {identifier}["{label}"]')
    for edge in edges:
        source = _safe_label(edge["source"])
        target = _safe_label(edge["target"])
        label = _safe_label(edge.get("label", ""))
        arrow = f" -->|{label}| " if label else " --> "
        lines.append(f"    {source}{arrow}{target}")
    lines.append("```")
    return "\n".join(lines)
