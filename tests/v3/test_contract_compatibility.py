"""tests/v3/test_contract_compatibility.py — E3.2: the FR-606 adjacency table.

Pins the declared-data policy in nexus/contract/compatibility.py:
NONE appears in no row (chain endpoints compose with nothing), DYNAMIC
is excluded from adjacency checks entirely (consulting it is a
programming error), and the S3.6 wrapper chains — clean_then_profile,
clean_then_regress, cluster_then_chart — are legal edge-by-edge.
"""

from __future__ import annotations

import pytest

from localdata_mcp.nexus.contract.compatibility import MAY_FEED, may_feed
from localdata_mcp.nexus.contract.errors import ToolContractError
from localdata_mcp.nexus.contract.spec import TypeShape

# The S3.6 wrapper chains, expressed as the (output_shape, input_shape)
# edges the composition engine would validate. A chain's terminal output
# is not an edge — only stage-to-stage handoffs are checked.
WRAPPER_CHAIN_EDGES = {
    # preprocessing TABULAR->TABULAR, then profile consumes TABULAR
    "clean_then_profile": [
        (TypeShape.TABULAR, TypeShape.TABULAR),
        (TypeShape.TABULAR, TypeShape.TABULAR),
    ],
    # preprocessing TABULAR->TABULAR, then regress consumes TABULAR
    # (emitting FITTED_MODEL/SCALAR as terminal output)
    "clean_then_regress": [
        (TypeShape.TABULAR, TypeShape.TABULAR),
        (TypeShape.TABULAR, TypeShape.TABULAR),
    ],
    # assign_clusters (TABULAR: features + cluster column) -> render_chart
    # (TABULAR input, NONE terminal sink) — the one stage-to-stage edge
    "cluster_then_chart": [
        (TypeShape.TABULAR, TypeShape.TABULAR),
    ],
}


class TestTableShape:
    def test_none_appears_in_no_row(self) -> None:
        assert TypeShape.NONE not in MAY_FEED
        for accepted in MAY_FEED.values():
            assert TypeShape.NONE not in accepted

    def test_dynamic_appears_in_no_row(self) -> None:
        assert TypeShape.DYNAMIC not in MAY_FEED
        for accepted in MAY_FEED.values():
            assert TypeShape.DYNAMIC not in accepted

    def test_every_concrete_shape_has_a_row(self) -> None:
        concrete = set(TypeShape) - {TypeShape.NONE, TypeShape.DYNAMIC}
        assert set(MAY_FEED) == concrete

    def test_values_are_frozensets(self) -> None:
        for accepted in MAY_FEED.values():
            assert isinstance(accepted, frozenset)


class TestMayFeed:
    def test_none_feeds_nothing_and_is_fed_by_nothing(self) -> None:
        # DYNAMIC is skipped: consulting it raises (tested below), and
        # that refusal outranks NONE's silent False.
        for shape in set(TypeShape) - {TypeShape.DYNAMIC}:
            assert may_feed(TypeShape.NONE, shape) is False
            assert may_feed(shape, TypeShape.NONE) is False

    def test_dynamic_consultation_is_refused(self) -> None:
        with pytest.raises(ToolContractError):
            may_feed(TypeShape.DYNAMIC, TypeShape.TABULAR)
        with pytest.raises(ToolContractError):
            may_feed(TypeShape.TABULAR, TypeShape.DYNAMIC)

    def test_geo_isolation(self) -> None:
        # section 6.1: GEO is distinct from TABULAR so a non-geo stage
        # cannot silently receive geometries it would mangle.
        assert may_feed(TypeShape.GEO, TypeShape.TABULAR) is False
        assert may_feed(TypeShape.TABULAR, TypeShape.GEO) is False

    def test_fitted_model_feeds_only_fitted_model_consumers(self) -> None:
        assert may_feed(TypeShape.FITTED_MODEL, TypeShape.FITTED_MODEL) is True
        assert may_feed(TypeShape.FITTED_MODEL, TypeShape.TABULAR) is False

    def test_chart_spec_feeds_chart_spec_consumers(self) -> None:
        assert may_feed(TypeShape.CHART_SPEC, TypeShape.CHART_SPEC) is True

    @pytest.mark.parametrize("chain", sorted(WRAPPER_CHAIN_EDGES))
    def test_wrapper_chain_edges_are_legal(self, chain: str) -> None:
        for output_shape, input_shape in WRAPPER_CHAIN_EDGES[chain]:
            assert may_feed(output_shape, input_shape) is True, (
                f"{chain}: {output_shape.name} must feed {input_shape.name}"
            )
