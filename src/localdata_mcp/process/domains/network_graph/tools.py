"""localdata_mcp/process/domains/network_graph/tools.py — E10.i ToolSpec.

The network family's tool, carried by name from `main` (DR GP2):
`analyze_network` over a tabular edge list addressed by the X-2
contract — the I-3 graph-store tools (`get_edges`) feed it through
composition. Neighbors: network.py computes; spec_modules.py rosters
this module.
"""

from __future__ import annotations

from typing import Any

from localdata_mcp.nexus.contract.spec import Param, TypeShape, tool_spec

from ..support import addressed_frame, source_params
from .network import analyze_edge_list


@tool_spec(
    name="analyze_network",
    summary=(
        "Analyze a network stored as a tabular edge list (addressed "
        "source with source_column/target_column, optional "
        "weight_column, directed on request): density, connectivity, "
        "components, degree summary, clustering, and top centrality "
        "nodes."
    ),
    params=(
        *source_params(),
        Param("source_column", str, "The edge-source node column."),
        Param("target_column", str, "The edge-target node column."),
        Param("weight_column", str, "Optional edge-weight column.", required=False),
        Param(
            "directed",
            bool,
            "Treat edges as directed (implementation default false).",
            required=False,
        ),
        Param(
            "include_centrality",
            bool,
            "Compute centrality measures (implementation default true).",
            required=False,
        ),
    ),
    input_shape=TypeShape.TABULAR,
    output_shape=TypeShape.SCALAR,
    domain="process",
)
def analyze_network(
    source_column: str,
    target_column: str,
    endpoint: str | None = None,
    path: str | None = None,
    table: str | None = None,
    query: str | None = None,
    **knobs: Any,
) -> Any:
    frame, source = addressed_frame(endpoint, path, table, query)
    result = analyze_edge_list(frame, source_column, target_column, **knobs)
    result["source"] = source
    return result
