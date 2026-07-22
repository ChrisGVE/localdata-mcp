"""localdata_mcp/nexus/contract/compatibility.py — the FR-606 adjacency table.

Hand-authored DECLARED DATA over the closed TypeShape set (spec.py):
which output shapes may feed which input shapes. Deliberately not a
generators/ output — which shapes compose is a policy decision, not a
derivation from ToolSpecs, so authored policy and generated artifacts
never share a file (ARCHITECTURE.md section 6.3). The composition
engine (E11) consults may_feed() per dag_spec edge pre-execution;
NX-7 only carries the resulting metadata.

Policy rationale per row:
- TABULAR is the workhorse: it feeds any TABULAR-input stage.
- VECTOR additionally feeds TABULAR-input stages — an ordered series
  is trivially a one-column frame (legalizes cluster labels feeding a
  tabular chart stage, S3.6 cluster_then_chart).
- GEO feeds only GEO-input stages: section 6.1 keeps geometry-bearing
  data away from non-geo stages that would mangle it.
- FITTED_MODEL feeds only predict/transform stages declaring it.
- NONE has no row: nothing feeds a source, a sink feeds nothing.
- DYNAMIC has no row and may not be consulted: the engine validates a
  submitted dag_spec's internal edges against concrete stage shapes
  instead (section 6.1's compose_pipeline contract).
"""

from __future__ import annotations

from typing import Final, Mapping

from localdata_mcp.nexus.contract.errors import ToolContractError
from localdata_mcp.nexus.contract.spec import TypeShape

MAY_FEED: Final[Mapping[TypeShape, frozenset[TypeShape]]] = {
    TypeShape.TABULAR: frozenset({TypeShape.TABULAR}),
    TypeShape.SCALAR: frozenset({TypeShape.SCALAR}),
    TypeShape.VECTOR: frozenset({TypeShape.VECTOR, TypeShape.TABULAR}),
    TypeShape.MATRIX: frozenset({TypeShape.MATRIX}),
    TypeShape.FITTED_MODEL: frozenset({TypeShape.FITTED_MODEL}),
    TypeShape.GRAPH: frozenset({TypeShape.GRAPH}),
    TypeShape.GEO: frozenset({TypeShape.GEO}),
    TypeShape.CHART_SPEC: frozenset({TypeShape.CHART_SPEC}),
}


def may_feed(output_shape: TypeShape, input_shape: TypeShape) -> bool:
    """Whether a stage emitting `output_shape` may feed one consuming
    `input_shape`, per the declared table.

    NONE in either position is False by construction (no row). DYNAMIC
    in either position raises: adjacency checks must never be consulted
    for it — that is a caller defect, not a composition verdict.
    """
    if TypeShape.DYNAMIC in (output_shape, input_shape):
        raise ToolContractError(
            "DYNAMIC is excluded from adjacency checks; validate the "
            "dag_spec's concrete stage shapes instead"
        )
    return input_shape in MAY_FEED.get(output_shape, frozenset())
