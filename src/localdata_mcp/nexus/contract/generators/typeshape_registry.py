"""localdata_mcp/nexus/contract/generators/typeshape_registry.py — artifact 5.

Renders nexus/contract/generated_shapes.py: the FR-606 type-shape
registry entry per tool — name -> (input shape, output shape,
streaming flag, domain) — that the composition engine (E11) consults
when validating a dag_spec's edges against compatibility.py's
adjacency table. Neighbors: generate.py writes the output.
"""

from __future__ import annotations

from localdata_mcp.nexus.contract.registry import ToolRegistry

GENERATOR_NAME = "localdata_mcp.nexus.contract.generators.typeshape_registry"

_HEADER = f'''"""MACHINE-WRITTEN by {GENERATOR_NAME} — DO NOT EDIT.

Per-tool type-shape declarations (ARCHITECTURE.md 6.1 artifact 5).
Regenerate via `python -m localdata_mcp.nexus.contract.generate`;
hand edits fail CI through nexus/contract/check_drift.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Mapping

from localdata_mcp.nexus.contract.spec import TypeShape


@dataclass(frozen=True)
class ShapeEntry:
    """One tool's declared composition facts."""

    input_shape: TypeShape
    output_shape: TypeShape
    streaming_capable: bool
    domain: "str | None"


TOOL_SHAPES: Final[Mapping[str, ShapeEntry]] = {{
'''


def render_shapes_module(registry: ToolRegistry) -> str:
    """The complete generated_shapes.py text for `registry`."""
    entries = "".join(
        f'    "{spec.name}": ShapeEntry(\n'
        f"        input_shape=TypeShape.{spec.input_shape.name},\n"
        f"        output_shape=TypeShape.{spec.output_shape.name},\n"
        f"        streaming_capable={spec.streaming_capable},\n"
        f"        domain={spec.domain!r},\n"
        f"    ),\n"
        for spec in registry
    )
    return _HEADER + entries + "}\n"
