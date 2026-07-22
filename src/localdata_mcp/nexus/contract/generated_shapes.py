"""MACHINE-WRITTEN by localdata_mcp.nexus.contract.generators.typeshape_registry — DO NOT EDIT.

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


TOOL_SHAPES: Final[Mapping[str, ShapeEntry]] = {
    "ping": ShapeEntry(
        input_shape=TypeShape.NONE,
        output_shape=TypeShape.SCALAR,
        streaming_capable=False,
        domain=None,
    ),
}
