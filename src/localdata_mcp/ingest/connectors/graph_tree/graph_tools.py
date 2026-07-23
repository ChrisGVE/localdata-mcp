"""localdata_mcp/ingest/connectors/graph_tree/graph_tools.py — edges (E8.3).

I-3's graph half (names carried from `main`, semantics intact — DR
GP2): neighbors, edges, edge mutation with the harvested advisory
warnings, path finding, and graph statistics — all against declared
graph-kind endpoints. `find_path` reads the FULL edge set through the
guard (admission-gated like every read) and runs NetworkX locally:
reads cross NX-6, computation is pure — no live handle ever reaches
this module. The all-paths search is depth- and count-bounded by the
named constants below (non-configurable exploration caps, not S8
resource defaults — the guard's admission gate owns memory). Mutations
cross `guarded_mutation` in the access layer (NFR-106/113).
Neighbors: graph_store.py executes; warnings.py supplies signals;
tools.py owns the node half.
"""

from __future__ import annotations

from itertools import islice
from typing import Any

import networkx as nx

from localdata_mcp.nexus.chokepoint.guard import Result
from localdata_mcp.nexus.contract.spec import Param, TypeShape, tool_spec

from ...refusals import missing_entity_refusal
from .. import graph_store
from ..store_dispatch import GRAPH_KINDS, resolve_store_kind
from .warnings import edge_warnings, orphan_warnings

# Exploration caps for the all-simple-paths search (harvested shape:
# `main` cut off at depth 20 and truncated the listing) — legibility
# bounds on an exponential enumeration, deliberately not operator
# config: the memory ceiling is the guard's job, these keep the answer
# readable.
_ALL_PATHS_DEPTH_CUTOFF = 20
_MAX_PATHS = 16

_GRAPH_HINT = (
    "Call get_graph_stats(endpoint) for the graph summary or "
    "get_edges(endpoint) to browse edges."
)


@tool_spec(
    name="get_neighbors",
    summary=(
        "List a graph node's neighbors with edge label/weight and "
        "direction ('in', 'out', or 'both')."
    ),
    params=(
        Param("endpoint", str, "The operator-declared graph endpoint name."),
        Param("node_id", str, "The node whose neighbors to list."),
        Param(
            "direction",
            str,
            "'in', 'out', or 'both' (default).",
            required=False,
        ),
        Param("offset", int, "Pagination offset (default 0).", required=False),
        Param(
            "limit",
            int,
            "Optional page size; omitted serves all rows.",
            required=False,
        ),
    ),
    input_shape=TypeShape.NONE,
    output_shape=TypeShape.TABULAR,
    domain="ingest",
)
def get_neighbors(
    endpoint: str,
    node_id: str,
    direction: str = "both",
    offset: int = 0,
    limit: int | None = None,
) -> Any:
    resolve_store_kind(endpoint, GRAPH_KINDS)
    if direction not in ("in", "out", "both"):
        raise missing_entity_refusal(
            f"Invalid direction: {direction}. Use 'in', 'out', or 'both'.",
            _GRAPH_HINT,
        )
    if not graph_store.node_exists(endpoint, node_id):
        raise missing_entity_refusal(f"Node not found: {node_id}", _GRAPH_HINT)
    raw = graph_store.neighbors(endpoint, node_id, direction, offset, limit)
    if direction in ("in", "out"):
        rows = tuple((row[0], row[1], row[2], direction) for row in raw.rows)
    else:
        classified = []
        for source, target, label, weight in raw.rows:
            if source == target == node_id:
                classified.append((node_id, label, weight, "self"))
            elif source == node_id:
                classified.append((target, label, weight, "out"))
            else:
                classified.append((source, label, weight, "in"))
        rows = tuple(classified)
    return Result(
        columns=("neighbor_id", "edge_label", "edge_weight", "direction"),
        rows=rows,
        category="query",
    )


@tool_spec(
    name="get_edges",
    summary=(
        "List a graph store's edges (source, target, label, weight), "
        "optionally filtered to those touching one node."
    ),
    params=(
        Param("endpoint", str, "The operator-declared graph endpoint name."),
        Param("node_id", str, "Optional node filter.", required=False),
        Param("offset", int, "Pagination offset (default 0).", required=False),
        Param(
            "limit",
            int,
            "Optional page size; omitted serves all rows.",
            required=False,
        ),
    ),
    input_shape=TypeShape.NONE,
    output_shape=TypeShape.TABULAR,
    domain="ingest",
)
def get_edges(
    endpoint: str,
    node_id: str | None = None,
    offset: int = 0,
    limit: int | None = None,
) -> Any:
    resolve_store_kind(endpoint, GRAPH_KINDS)
    if node_id is not None and not graph_store.node_exists(endpoint, node_id):
        raise missing_entity_refusal(f"Node not found: {node_id}", _GRAPH_HINT)
    return graph_store.edges(endpoint, node_id, offset, limit)


@tool_spec(
    name="add_edge",
    summary=(
        "Add (or re-weight) a directed edge on a declared read-write "
        "graph endpoint, auto-creating missing nodes; returns the "
        "harvested integrity warnings (self-loop, duplicates, "
        "contradictory reverse edge)."
    ),
    params=(
        Param("endpoint", str, "The operator-declared graph endpoint name."),
        Param("source", str, "The edge's source node id."),
        Param("target", str, "The edge's target node id."),
        Param("label", str, "Optional edge label.", required=False),
        Param("weight", float, "Optional edge weight.", required=False),
    ),
    input_shape=TypeShape.NONE,
    output_shape=TypeShape.SCALAR,
    domain="ingest",
)
def add_edge(
    endpoint: str,
    source: str,
    target: str,
    label: str | None = None,
    weight: float | None = None,
) -> Any:
    resolve_store_kind(endpoint, GRAPH_KINDS)
    edge, nodes_created = graph_store.add_edge(endpoint, source, target, label, weight)
    result: dict[str, Any] = {
        "source": edge["source_id"],
        "target": edge["target_id"],
        "label": edge["label"],
        "weight": edge["weight"],
        "nodes_created": nodes_created,
    }
    warned = edge_warnings(endpoint, source, target, label)
    if warned:
        result["warnings"] = warned
    return result


@tool_spec(
    name="remove_edge",
    summary=(
        "Remove a directed edge (and its properties) from a declared "
        "read-write graph endpoint; warns when a node becomes an orphan."
    ),
    params=(
        Param("endpoint", str, "The operator-declared graph endpoint name."),
        Param("source", str, "The edge's source node id."),
        Param("target", str, "The edge's target node id."),
        Param(
            "label",
            str,
            "Optional edge label (NULL-labeled when omitted).",
            required=False,
        ),
    ),
    input_shape=TypeShape.NONE,
    output_shape=TypeShape.SCALAR,
    domain="ingest",
)
def remove_edge(
    endpoint: str,
    source: str,
    target: str,
    label: str | None = None,
) -> Any:
    resolve_store_kind(endpoint, GRAPH_KINDS)
    removed = graph_store.remove_edge(endpoint, source, target, label)
    result: dict[str, Any] = {
        "source": source,
        "target": target,
        "label": label,
        "removed": removed,
    }
    if removed:
        warned = orphan_warnings(endpoint, source, target)
        if warned:
            result["warnings"] = warned
    return result


def _graph_from_edges(endpoint: str, *anchor_nodes: str) -> nx.MultiDiGraph:
    """The store's topology as a NetworkX graph: the full edge set read
    through the guard (admission-gated), anchors added so isolated
    endpoints resolve to no-path instead of a missing-node error."""
    topology = nx.MultiDiGraph()
    for node_id in anchor_nodes:
        topology.add_node(node_id)
    for source, target, _label, weight in graph_store.edges(
        endpoint, None, 0, None
    ).rows:
        if weight is None:
            topology.add_edge(source, target)
        else:
            topology.add_edge(source, target, weight=weight)
    return topology


@tool_spec(
    name="find_path",
    summary=(
        "Find path(s) between two graph nodes: the shortest path, or "
        "all simple paths (bounded enumeration)."
    ),
    params=(
        Param("endpoint", str, "The operator-declared graph endpoint name."),
        Param("source", str, "The start node id."),
        Param("target", str, "The end node id."),
        Param(
            "algorithm", str, "'shortest' (default) or 'all'.", required=False
        ),
    ),
    input_shape=TypeShape.NONE,
    output_shape=TypeShape.SCALAR,
    domain="ingest",
)
def find_path(
    endpoint: str,
    source: str,
    target: str,
    algorithm: str = "shortest",
) -> Any:
    resolve_store_kind(endpoint, GRAPH_KINDS)
    if algorithm not in ("shortest", "all"):
        raise missing_entity_refusal(
            f"Unknown algorithm: {algorithm}. Use 'shortest' or 'all'.",
            _GRAPH_HINT,
        )
    for node_id in (source, target):
        if not graph_store.node_exists(endpoint, node_id):
            raise missing_entity_refusal(f"Node not found: {node_id}", _GRAPH_HINT)
    topology = _graph_from_edges(endpoint, source, target)
    if algorithm == "shortest":
        try:
            path = nx.shortest_path(topology, source, target)
        except nx.NetworkXNoPath:
            return {
                "source": source,
                "target": target,
                "algorithm": "shortest",
                "path": None,
                "path_length": None,
                "message": "No path exists between source and target.",
            }
        return {
            "source": source,
            "target": target,
            "algorithm": "shortest",
            "path": list(path),
            "path_length": len(path) - 1,
        }
    paths = [
        list(found)
        for found in islice(
            nx.all_simple_paths(
                topology, source, target, cutoff=_ALL_PATHS_DEPTH_CUTOFF
            ),
            _MAX_PATHS + 1,
        )
    ]
    truncated = len(paths) > _MAX_PATHS
    if truncated:
        paths = paths[:_MAX_PATHS]
    return {
        "source": source,
        "target": target,
        "algorithm": "all",
        "paths": paths,
        "paths_count": len(paths),
        "truncated": truncated,
    }


@tool_spec(
    name="get_graph_stats",
    summary=(
        "Summary statistics for a declared graph endpoint: node, edge, "
        "and property counts plus density."
    ),
    params=(Param("endpoint", str, "The operator-declared graph endpoint name."),),
    input_shape=TypeShape.NONE,
    output_shape=TypeShape.SCALAR,
    domain="ingest",
)
def get_graph_stats(endpoint: str) -> Any:
    resolve_store_kind(endpoint, GRAPH_KINDS)
    return graph_store.graph_stats(endpoint)
