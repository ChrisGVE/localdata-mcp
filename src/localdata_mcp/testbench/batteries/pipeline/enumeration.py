"""localdata_mcp/testbench/batteries/pipeline/enumeration.py — NFR-502d enumeration (E11.4).

The pipeline battery's capacity model as executable data (S7.3), one
home for both the runner and the count assertions. The launch domain
set is FR-301's nine analysis families (preprocessing and composition
are excluded — they are the stages and the engine, not coupling
subjects). Each domain has one PINNED representative tool (declared
here, its shapes read from the live NX-1 registry — never restated),
so a length-2 domain pair is a concrete two-stage dag_spec. A link's
validity is computed by enumeration against the FR-606 adjacency table
(may_feed on the representatives' shapes), NOT hardcoded. Length-3/4
totals derive from the length-2 link facts ALONE (an A-B-A extension
is valid iff A→B and B→A are both valid length-2 links), so no third
unverified function sits between the exhaustively-checked 72 and the
longer-chain totals. Neighbors: pipeline_battery_test.py executes this
and asserts the counts; dag_spec.py is the validator under test.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from typing import Mapping

from localdata_mcp.nexus.contract.compatibility import may_feed
from localdata_mcp.nexus.contract.registry import ToolRegistry
from localdata_mcp.nexus.contract.spec import TypeShape

# FR-301's nine domains, each mapped to its PINNED representative tool
# (S7.3's "domain-pair" made concrete). Highest-traffic / most
# characteristic tool per family; shapes are read from the registry,
# so this is a name choice, not a shape restatement.
DOMAIN_REPRESENTATIVES: Mapping[str, str] = {
    "statistical_analysis": "analyze_hypothesis_test",
    "regression_modeling": "analyze_regression",
    "pattern_recognition": "transform_data",
    "time_series": "forecast_time_series",
    "geospatial_analysis": "analyze_spatial_autocorrelation",
    "optimization": "solve_linear_program",
    "sampling_estimation": "generate_sample",
    "business_intelligence": "analyze_rfm",
    "network_graph": "analyze_network",
}


@dataclass(frozen=True)
class DomainLink:
    """One ordered length-2 domain pair and its FR-606 legality."""

    source_domain: str
    target_domain: str
    source_tool: str
    target_tool: str
    legal: bool


def ordered_links(registry: ToolRegistry) -> tuple[DomainLink, ...]:
    """All 9×8 ordered domain pairs with legality computed from the
    representatives' registry-declared shapes (the length-2 link
    facts). Legal iff the source's output may feed the target's input
    per the adjacency table — NONE endpoints never link."""
    specs = {
        domain: registry.lookup(tool) for domain, tool in DOMAIN_REPRESENTATIVES.items()
    }
    links: list[DomainLink] = []
    for source, target in permutations(DOMAIN_REPRESENTATIVES, 2):
        out_shape = specs[source].output_shape
        in_shape = specs[target].input_shape
        legal = (
            TypeShape.NONE not in (out_shape, in_shape)
            and TypeShape.DYNAMIC not in (out_shape, in_shape)
            and may_feed(out_shape, in_shape)
        )
        links.append(
            DomainLink(
                source_domain=source,
                target_domain=target,
                source_tool=specs[source].name,
                target_tool=specs[target].name,
                legal=legal,
            )
        )
    return tuple(links)


def length2_total() -> int:
    """The closed-form executed total: 9 domains × 8 = 72."""
    domains = len(DOMAIN_REPRESENTATIVES)
    return domains * (domains - 1)


def _link_index(links: "tuple[DomainLink, ...]") -> "dict[tuple[str, str], bool]":
    return {(link.source_domain, link.target_domain): link.legal for link in links}


def alternating_valid_total(links: "tuple[DomainLink, ...]", length: int) -> int:
    """The enumerated valid count for an alternating A-B-A(-B) chain at
    `length`, derived FROM THE LENGTH-2 LINK FACTS ALONE: valid iff
    every adjacent directed link in the alternation is a valid length-2
    link. length 3 = A-B-A (needs A→B, B→A); length 4 = A-B-A-B (adds
    A→B again — same two links)."""
    if length not in (3, 4):
        raise ValueError(f"alternating extension defined for lengths 3-4, got {length}")
    index = _link_index(links)
    valid = 0
    for source, target in permutations(DOMAIN_REPRESENTATIVES, 2):
        forward = index[(source, target)]
        backward = index[(target, source)]
        if forward and backward:
            valid += 1
    return valid


def length2_dag_spec(link: DomainLink, source_path: str) -> "list[dict[str, object]]":
    """The concrete two-stage dag_spec for one ordered pair: the source
    tool addresses the fixture, the target consumes its output. Params
    beyond the source path are omitted deliberately — a legal pair may
    still fail at the DOMAIN level (missing columns) which is the
    expected 'meaningless but correct' outcome (FR-302); only an
    engine-level rejection distinguishes an illegal pair."""
    return [
        {"stage": "a", "tool": link.source_tool, "params": {"path": source_path}},
        {"stage": "b", "tool": link.target_tool, "depends_on": ["a"]},
    ]
