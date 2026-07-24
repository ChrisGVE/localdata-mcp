"""tests/v3/test_pipeline_stretch_enumeration.py — stretch/longchain enum.

Fast, engine- and registry-free coverage of the enumeration additions:
the alternating dag_spec shape at lengths 2..6 and the seeded stretch
sampler (determinism, sample size, length bounds, argument validation).
Uses synthetic DomainLinks so no tool registry needs booting.
"""

from __future__ import annotations

from typing import Tuple

import pytest

from localdata_mcp.testbench.batteries.pipeline import enumeration
from localdata_mcp.testbench.batteries.pipeline.enumeration import DomainLink


def _links() -> Tuple[DomainLink, ...]:
    """A small synthetic link set; tool names are arbitrary labels here."""
    return (
        DomainLink("stats", "pattern", "analyze_x", "transform_y", legal=True),
        DomainLink("pattern", "stats", "transform_y", "analyze_x", legal=True),
        DomainLink("stats", "network", "analyze_x", "analyze_net", legal=False),
    )


class TestAlternatingDagSpec:
    def test_shape_alternates_and_lines_up(self) -> None:
        link = _links()[0]
        spec = enumeration.alternating_dag_spec(link, 4, "/tmp/x.csv")
        assert [s["stage"] for s in spec] == ["a", "b", "c", "d"]
        assert [s["tool"] for s in spec] == [
            link.source_tool,
            link.target_tool,
            link.source_tool,
            link.target_tool,
        ]
        assert spec[0]["params"] == {"path": "/tmp/x.csv"}
        assert spec[3]["depends_on"] == ["c"]

    def test_length_two_matches_legacy_helper(self) -> None:
        link = _links()[0]
        assert enumeration.alternating_dag_spec(
            link, 2, "/p.csv"
        ) == enumeration.length2_dag_spec(link, "/p.csv")

    @pytest.mark.parametrize("bad", [1, 7])
    def test_out_of_range_length_refused(self, bad: int) -> None:
        with pytest.raises(ValueError):
            enumeration.alternating_dag_spec(_links()[0], bad, "/p.csv")


class TestSampledStretchChains:
    def test_is_deterministic_for_a_seed(self) -> None:
        links = _links()
        first = enumeration.sampled_stretch_chains(
            links, max_length=6, sample_count=25, seed=42, source_path="/p.csv"
        )
        again = enumeration.sampled_stretch_chains(
            links, max_length=6, sample_count=25, seed=42, source_path="/p.csv"
        )
        assert [(c.link, c.length) for c in first] == [
            (c.link, c.length) for c in again
        ]

    def test_sample_size_and_length_bounds(self) -> None:
        chains = enumeration.sampled_stretch_chains(
            _links(), max_length=6, sample_count=50, seed=7, source_path="/p.csv"
        )
        assert len(chains) == 50
        assert all(2 <= c.length <= 6 for c in chains)
        assert all(len(c.dag_spec) == c.length for c in chains)

    def test_zero_sample_is_empty(self) -> None:
        chains = enumeration.sampled_stretch_chains(
            _links(), max_length=6, sample_count=0, seed=1, source_path="/p.csv"
        )
        assert chains == []

    def test_invalid_arguments_refused(self) -> None:
        links = _links()
        with pytest.raises(ValueError):
            enumeration.sampled_stretch_chains(
                links, max_length=1, sample_count=5, seed=1, source_path="/p.csv"
            )
        with pytest.raises(ValueError):
            enumeration.sampled_stretch_chains(
                links, max_length=6, sample_count=-1, seed=1, source_path="/p.csv"
            )
