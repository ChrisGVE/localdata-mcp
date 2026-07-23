"""localdata_mcp/nexus/contract/generators/docstring.py — artifact 2's renderer.

Renders the caller-facing docstring for one ToolSpec. The docstring
rides the GENERATED wrapper (wrapper.py embeds it), never the
hand-authored implementation function — closing #40's residual
structural cause and FR-704: the prose has exactly one source, the
spec. Neighbors: wrapper.py is the sole consumer; docs.py renders the
sibling human-docs view from the same fields.
"""

from __future__ import annotations

from localdata_mcp.nexus.contract.spec import ToolSpec, TypeShape

_SHAPE_NOTES = {
    TypeShape.NONE: "NONE (chain endpoint — composes with nothing)",
    TypeShape.DYNAMIC: "DYNAMIC (validated per submitted dag_spec)",
}


def _shape_label(shape: TypeShape) -> str:
    return _SHAPE_NOTES.get(shape, shape.name)


def render_docstring(spec: ToolSpec) -> str:
    """The full docstring text for `spec`'s generated wrapper."""
    lines = [
        spec.summary,
        "",
        f"Input shape: {_shape_label(spec.input_shape)}.",
        f"Output shape: {_shape_label(spec.output_shape)}.",
        f"Streaming-capable: {'yes' if spec.streaming_capable else 'no'}.",
    ]
    if spec.domain is not None:
        lines.append(f"Domain: {spec.domain}.")
    if spec.params:
        lines.extend(["", "Args:"])
        lines.extend(
            f"    {param.name}"
            + ("" if param.required else " (optional)")
            + f": {param.description}"
            for param in spec.params
        )
    return "\n".join(lines)
