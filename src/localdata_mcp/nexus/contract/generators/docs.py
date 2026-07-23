"""localdata_mcp/nexus/contract/generators/docs.py — artifact 3's renderer.

Renders docs/tools/*.md: one Markdown table per domain group (tools
with domain=None group under "core") — the human-docs view of the
registry, a RENDERED VIEW never a consumed source (section 7.2's rule
applied to tools). Neighbors: generate.py writes the outputs;
docstring.py renders the sibling caller-facing prose.
"""

from __future__ import annotations

from localdata_mcp.nexus.contract.registry import ToolRegistry
from localdata_mcp.nexus.contract.spec import ToolSpec

GENERATOR_NAME = "localdata_mcp.nexus.contract.generators.docs"

_BANNER = (
    f"<!-- MACHINE-WRITTEN by {GENERATOR_NAME} — DO NOT EDIT; regenerate "
    "via `python -m localdata_mcp.nexus.contract.generate` -->\n"
)
_TABLE_HEAD = (
    "| Tool | Summary | Input shape | Output shape | Streaming | Params |\n"
    "|---|---|---|---|---|---|\n"
)


def _row(spec: ToolSpec) -> str:
    params = (
        ", ".join(
            f"`{param.name}`" if param.required else f"`{param.name}?`"
            for param in spec.params
        )
        or "—"
    )
    streaming = "yes" if spec.streaming_capable else "no"
    return (
        f"| `{spec.name}` | {spec.summary} | {spec.input_shape.name} "
        f"| {spec.output_shape.name} | {streaming} | {params} |\n"
    )


def render_docs(registry: ToolRegistry) -> dict[str, str]:
    """Per-group markdown files: `{filename: content}`, groups sorted."""
    groups: dict[str, list[ToolSpec]] = {}
    for spec in registry:
        groups.setdefault(spec.domain or "core", []).append(spec)
    files: dict[str, str] = {}
    for group in sorted(groups):
        rows = "".join(_row(spec) for spec in groups[group])
        files[f"{group}.md"] = _BANNER + f"\n# Tools — {group}\n\n" + _TABLE_HEAD + rows
    return files
