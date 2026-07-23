"""localdata_mcp/nexus/contract/spec.py — the ToolSpec declaration vocabulary.

NX-1's one-declaration model (ARCHITECTURE.md section 6.1): a tool is
authored as exactly one @tool_spec declaration, and every generated
artifact — FastMCP wrapper, docstring, docs row, L3 contract-test stub,
type-shape registry entry — derives from it. TypeShape is the CLOSED
enumeration FR-606's compatibility mechanism draws from; adding a
member is a reviewed policy change (compatibility.py must gain rows in
the same change-set). Neighbors: registry.py holds registered specs;
errors.py defines the refusal taxonomy raised here.
"""

from __future__ import annotations

import enum
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from localdata_mcp.nexus.contract.errors import SpecValidationError


class TypeShape(enum.Enum):
    """Closed vocabulary of composable data shapes (section 6.1).

    NONE marks a chain endpoint: as input_shape the tool is a source
    (chain-initial only), as output_shape a terminal sink (leaf only);
    NONE appears in no compatibility-table row. DYNAMIC belongs to
    compose_pipeline exclusively — excluded from adjacency checks and
    barred as a dag_spec stage; the bar is enforced at dag_spec
    validation (E11), not at registration.
    """

    TABULAR = "tabular"
    SCALAR = "scalar"
    VECTOR = "vector"
    MATRIX = "matrix"
    FITTED_MODEL = "fitted_model"
    GRAPH = "graph"
    GEO = "geo"
    CHART_SPEC = "chart_spec"
    NONE = "none"
    DYNAMIC = "dynamic"


@dataclass(frozen=True)
class Param:
    """One declared tool parameter: name, Python annotation, prose.

    `required=False` declares a caller-omittable parameter (E8.5): the
    generated wrapper exposes it as `type | None = None` and forwards
    it ONLY when supplied, so the implementation's own default governs
    — progressive disclosure at the MCP schema itself (FP #3), never
    a wrapper-invented default value (one default site, NFR-403)."""

    name: str
    annotation: type
    description: str
    required: bool = True


@dataclass(frozen=True)
class ToolSpec:
    """The single authoritative declaration of one tool's contract."""

    name: str
    summary: str
    params: tuple[Param, ...]
    input_shape: TypeShape
    output_shape: TypeShape
    streaming_capable: bool = False
    domain: str | None = None
    func: Callable[..., Any] = field(default=lambda: None, compare=False)

    def __post_init__(self) -> None:
        if not self.name:
            raise SpecValidationError("ToolSpec.name must be non-empty")
        if not self.summary:
            raise SpecValidationError(
                "ToolSpec.summary must be non-empty", tool_name=self.name
            )
        object.__setattr__(self, "params", tuple(self.params))
        for entry in self.params:
            if not isinstance(entry, Param):
                raise SpecValidationError(
                    f"ToolSpec.params entry {entry!r} is not a Param",
                    tool_name=self.name,
                )


def tool_spec(
    *,
    name: str,
    summary: str,
    params: Sequence[Param],
    input_shape: TypeShape,
    output_shape: TypeShape,
    streaming_capable: bool = False,
    domain: str | None = None,
    registry: Any = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Declare a tool: build its ToolSpec, register it, tag the function.

    Returns the implementation function unchanged (the generated wrapper
    carries the caller-facing docstring, section 6.1 — the hand-authored
    function stays plain). `registry` defaults to the process-wide
    registry (registry.py); tests pass their own ToolRegistry.
    """

    def decorate(func: Callable[..., Any]) -> Callable[..., Any]:
        spec = ToolSpec(
            name=name,
            summary=summary,
            params=tuple(params),
            input_shape=input_shape,
            output_shape=output_shape,
            streaming_capable=streaming_capable,
            domain=domain,
            func=func,
        )
        target = registry
        if target is None:
            from localdata_mcp.nexus.contract.registry import default_registry

            target = default_registry()
        target.register(spec)
        func.__tool_spec__ = spec  # type: ignore[attr-defined]
        return func

    return decorate
