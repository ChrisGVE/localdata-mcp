"""localdata_mcp/nexus/response/metadata.py — harvested composition shapes (E7.1).

The `CompositionMetadata`/`PipelineResult` data model, harvested
UNMODIFIED IN SHAPE from `pipeline/base.py` (§8 NX-7 — read in full:
well-designed on `main`, only unreachable from the live path, T12):
the same field set with v3's immutability discipline (frozen, tuples).
Plus the one derivation O-1 adds: `applicable_next_steps` reads the
FR-606 registry + adjacency table (NX-1 owns both halves, §6.3) to
name the registered tools this result's shape may feed — derived per
call from the live registry, never a hand-kept list. Neighbors:
envelope.py embeds `CompositionMetadata` in every success envelope;
the E11 composition engine returns `PipelineResult` per stage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from localdata_mcp.nexus.contract.compatibility import may_feed
from localdata_mcp.nexus.contract.registry import ToolRegistry
from localdata_mcp.nexus.contract.spec import TypeShape


@dataclass(frozen=True)
class CompositionMetadata:
    """Metadata for downstream tool composition and workflow chaining
    (harvested shape, `pipeline/base.py:73-98`)."""

    # Core composition information
    domain: str
    analysis_type: str
    result_type: str

    # Downstream compatibility
    compatible_tools: tuple[str, ...] = ()
    suggested_compositions: tuple[Mapping[str, Any], ...] = ()
    data_artifacts: Mapping[str, Any] = field(default_factory=dict)

    # Pipeline chaining context
    input_schema: Mapping[str, Any] = field(default_factory=dict)
    output_schema: Mapping[str, Any] = field(default_factory=dict)
    transformation_summary: Mapping[str, Any] = field(default_factory=dict)

    # Quality and confidence
    confidence_level: float = 0.0
    quality_score: float = 0.0
    limitations: tuple[str, ...] = ()

    # Next step recommendations
    recommended_next_steps: tuple[Mapping[str, Any], ...] = ()
    alternative_approaches: tuple[Mapping[str, Any], ...] = ()

    def to_wire(self) -> dict[str, Any]:
        """The JSON-serializable form the envelope embeds."""
        return {
            "domain": self.domain,
            "analysis_type": self.analysis_type,
            "result_type": self.result_type,
            "compatible_tools": list(self.compatible_tools),
            "suggested_compositions": [dict(s) for s in self.suggested_compositions],
            "data_artifacts": dict(self.data_artifacts),
            "input_schema": dict(self.input_schema),
            "output_schema": dict(self.output_schema),
            "transformation_summary": dict(self.transformation_summary),
            "confidence_level": self.confidence_level,
            "quality_score": self.quality_score,
            "limitations": list(self.limitations),
            "recommended_next_steps": [dict(s) for s in self.recommended_next_steps],
            "alternative_approaches": [dict(s) for s in self.alternative_approaches],
        }


@dataclass(frozen=True)
class PipelineResult:
    """Standardized pipeline execution result (harvested shape,
    `pipeline/base.py:119-138`) — the per-stage form the E11 engine
    returns under the multi-leaf response contract (§4d)."""

    # Core result data
    success: bool
    data: Any
    metadata: Mapping[str, Any]

    # Execution information
    execution_time_seconds: float
    memory_used_mb: float
    pipeline_stage: str

    # Composition context
    composition_metadata: CompositionMetadata | None = None

    # Error information (if success=False)
    error: Mapping[str, Any] | None = None
    partial_results: Any = None
    recovery_options: tuple[Mapping[str, Any], ...] = ()


def applicable_next_steps(
    output_shape: TypeShape, registry: ToolRegistry
) -> tuple[str, ...]:
    """Registered tool names this shape may feed, per the declared
    adjacency table (FR-606) — derived from the live registry at call
    time, so the recommendation set and the composition validator can
    never disagree (one home for the relation, §6.3).

    A terminal shape (NONE) feeds nothing; DYNAMIC is excluded from
    adjacency by contract; candidate sources (input NONE) and DYNAMIC
    stages are never recommendation targets.
    """
    if output_shape in (TypeShape.NONE, TypeShape.DYNAMIC):
        return ()
    return tuple(
        spec.name
        for spec in registry
        if spec.input_shape not in (TypeShape.NONE, TypeShape.DYNAMIC)
        and may_feed(output_shape, spec.input_shape)
    )
