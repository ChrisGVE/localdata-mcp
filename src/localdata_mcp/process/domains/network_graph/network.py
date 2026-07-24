"""localdata_mcp/process/domains/network_graph/network.py — FR-301.

`analyze_network`'s computation, re-authored from `main`'s
`NetworkAnalyzer`: build the graph from the addressed edge list
(source/target columns, optional weight, directed on request) and
report the structural verdict — node/edge counts, density,
connectivity, components, degree summary, clustering (undirected),
and the three classic centrality measures' top nodes. The
heavyweight extras (max-flow, MST, TSP heuristic) stay behind:
they were optimization-flavored riders, and the optimization family
owns that ground. Neighbors: tools.py declares the ToolSpec.
"""

from __future__ import annotations

from typing import Any

import networkx as nx
import pandas as pd

from ..support import invalid_source_refusal, require_columns

# Centrality listings: the readable head, not the whole ranking.
_TOP_NODES = 5


def analyze_edge_list(
    frame: pd.DataFrame,
    source_column: str,
    target_column: str,
    weight_column: str | None = None,
    directed: bool = False,
    include_centrality: bool = True,
) -> dict[str, Any]:
    """The structural analysis of the addressed edge list."""
    graph = _build_graph(frame, source_column, target_column, weight_column, directed)
    if graph.number_of_nodes() == 0:
        raise invalid_source_refusal(
            "The addressed edge list is empty — nothing to analyze."
        )
    degrees = [int(degree) for _node, degree in graph.degree()]
    connected = nx.is_weakly_connected(graph) if directed else nx.is_connected(graph)
    components = (
        nx.number_weakly_connected_components(graph)
        if directed
        else nx.number_connected_components(graph)
    )
    result: dict[str, Any] = {
        "n_nodes": int(graph.number_of_nodes()),
        "n_edges": int(graph.number_of_edges()),
        "directed": directed,
        "weighted": weight_column is not None,
        "density": float(nx.density(graph)),
        "is_connected": bool(connected),
        "n_components": int(components),
        "degree_summary": {
            "min": min(degrees),
            "max": max(degrees),
            "mean": float(sum(degrees) / len(degrees)),
        },
    }
    if not directed:
        result["average_clustering"] = float(nx.average_clustering(graph))
    if include_centrality:
        result["centrality"] = _centrality_block(graph)
    return result


def _build_graph(
    frame: pd.DataFrame,
    source_column: str,
    target_column: str,
    weight_column: str | None,
    directed: bool,
) -> "nx.Graph[Any]":
    """Build the graph from the edge-list frame with the vectorized
    `nx.from_pandas_edgelist` constructor (never a per-row iterrows).
    Rows missing an endpoint are dropped; endpoints are stringified; a
    weight column is carried under the `weight` edge attribute, and a row
    whose weight is missing stays a plain (unweighted) edge — parity with
    the original row-wise builder."""
    require_columns(frame, source_column, target_column, weight_column)
    create_using = nx.DiGraph if directed else nx.Graph
    columns = [source_column, target_column]
    if weight_column is not None:
        columns.append(weight_column)
    edges = frame[columns].dropna(subset=[source_column, target_column]).copy()
    edges[source_column] = edges[source_column].astype(str)
    edges[target_column] = edges[target_column].astype(str)
    if weight_column is None:
        return nx.from_pandas_edgelist(
            edges, source_column, target_column, create_using=create_using()
        )
    edges = edges.rename(columns={weight_column: "weight"})
    weighted = edges[edges["weight"].notna()].copy()
    weighted["weight"] = weighted["weight"].astype(float)
    graph = nx.from_pandas_edgelist(
        weighted,
        source_column,
        target_column,
        edge_attr="weight",
        create_using=create_using(),
    )
    plain = edges[edges["weight"].isna()]
    graph.add_edges_from(zip(plain[source_column], plain[target_column]))
    return graph


def _centrality_block(graph: "nx.Graph[Any]") -> dict[str, Any]:
    """Degree, betweenness, closeness — each measure's top nodes."""

    def top(measure: dict[Any, float]) -> list[dict[str, Any]]:
        ranked = sorted(measure.items(), key=lambda pair: -pair[1])[:_TOP_NODES]
        return [{"node": str(node), "score": float(score)} for node, score in ranked]

    return {
        "degree": top(nx.degree_centrality(graph)),
        "betweenness": top(nx.betweenness_centrality(graph)),
        "closeness": top(nx.closeness_centrality(graph)),
    }
