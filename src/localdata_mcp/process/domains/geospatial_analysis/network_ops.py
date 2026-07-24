"""localdata_mcp/process/domains/geospatial_analysis/network_ops.py — FR-301.

The network-routing trio carried by name from `main`:
`optimize_route` (order waypoints greedily, then connect them by
shortest path), `analyze_accessibility` (score demand nodes by their
travel time to the nearest service node), and
`generate_service_isochrones` (the node set reachable from the
services within each time band, with its convex-hull footprint). The
X-2 contract addresses the node table (id, x, y); the edge list
arrives inline as [source, target, weight] triples, so the whole
network is one addressed source plus one inline argument. Neighbors:
capabilities.py guards; tools.py declares the ToolSpecs.
"""

from __future__ import annotations

from typing import Any

import networkx as nx
import pandas as pd

from ..support import invalid_source_refusal, require_columns
from .capabilities import require_geostack


def _network(
    frame: pd.DataFrame,
    edges: list[list[Any]],
    node_id_column: str,
    x_column: str,
    y_column: str,
) -> "nx.Graph[Any]":
    require_columns(frame, node_id_column, x_column, y_column)
    graph: "nx.Graph[Any]" = nx.Graph()
    # Vectorized node construction (never a per-row iterrows): each node
    # carries its x/y coordinates as attributes.
    graph.add_nodes_from(
        (str(node_id), {"x": float(x), "y": float(y)})
        for node_id, x, y in zip(
            frame[node_id_column], frame[x_column], frame[y_column]
        )
    )
    for edge in edges:
        if len(edge) < 2:
            raise invalid_source_refusal(
                "each edge must be [source, target] or [source, target, weight]."
            )
        weight = float(edge[2]) if len(edge) > 2 else 1.0
        graph.add_edge(str(edge[0]), str(edge[1]), weight=weight)
    return graph


def _require_nodes(graph: "nx.Graph[Any]", nodes: list[Any], label: str) -> list[str]:
    resolved = [str(node) for node in nodes]
    missing = [node for node in resolved if node not in graph]
    if missing:
        raise invalid_source_refusal(f"{label} not in the node set: {missing}.")
    return resolved


def optimize_route(
    frame: pd.DataFrame,
    edges: list[list[Any]],
    waypoints: list[Any],
    node_id_column: str = "id",
    x_column: str = "x",
    y_column: str = "y",
    return_to_start: bool = False,
) -> dict[str, Any]:
    """Greedy nearest-neighbour waypoint order plus the connecting paths."""
    require_geostack()
    graph = _network(frame, edges, node_id_column, x_column, y_column)
    stops = _require_nodes(graph, waypoints, "waypoints")
    if len(stops) < 2:
        raise invalid_source_refusal("A route needs at least two waypoints.")
    order = _greedy_order(graph, stops)
    if return_to_start:
        order = order + [order[0]]
    path: list[str] = []
    total = 0.0
    for start, end in zip(order, order[1:]):
        leg = nx.shortest_path(graph, start, end, weight="weight")
        total += float(nx.shortest_path_length(graph, start, end, weight="weight"))
        path.extend(leg if not path else leg[1:])
    return {
        "waypoint_order": order,
        "route_path": path,
        "total_distance": total,
        "return_to_start": return_to_start,
    }


def _greedy_order(graph: "nx.Graph[Any]", stops: list[str]) -> list[str]:
    remaining = list(stops[1:])
    order = [stops[0]]
    while remaining:
        current = order[-1]
        nearest = min(
            remaining,
            key=lambda node: nx.shortest_path_length(
                graph, current, node, weight="weight"
            ),
        )
        order.append(nearest)
        remaining.remove(nearest)
    return order


def analyze_accessibility(
    frame: pd.DataFrame,
    edges: list[list[Any]],
    service_locations: list[Any],
    demand_locations: list[Any],
    node_id_column: str = "id",
    x_column: str = "x",
    y_column: str = "y",
    max_travel_time: float | None = None,
) -> dict[str, Any]:
    """Each demand node's travel time to its nearest service node."""
    require_geostack()
    graph = _network(frame, edges, node_id_column, x_column, y_column)
    services = _require_nodes(graph, service_locations, "service_locations")
    demands = _require_nodes(graph, demand_locations, "demand_locations")
    scores: dict[str, float] = {}
    reachable: list[str] = []
    for demand in demands:
        best = min(
            (
                float(nx.shortest_path_length(graph, demand, service, weight="weight"))
                for service in services
                if nx.has_path(graph, demand, service)
            ),
            default=None,
        )
        if best is None or (max_travel_time is not None and best > max_travel_time):
            continue
        scores[demand] = best
        reachable.append(demand)
    return {
        "n_services": len(services),
        "n_demand": len(demands),
        "max_travel_time": max_travel_time,
        "reachable_count": len(reachable),
        "unreachable": [node for node in demands if node not in reachable],
        "travel_times": scores,
    }


def service_isochrones(
    frame: pd.DataFrame,
    edges: list[list[Any]],
    service_locations: list[Any],
    time_bands: list[float],
    node_id_column: str = "id",
    x_column: str = "x",
    y_column: str = "y",
) -> dict[str, Any]:
    """The node set reachable within each time band and its footprint."""
    require_geostack()
    from shapely.geometry import MultiPoint

    graph = _network(frame, edges, node_id_column, x_column, y_column)
    services = _require_nodes(graph, service_locations, "service_locations")
    if not time_bands:
        raise invalid_source_refusal("At least one time band is required.")
    lengths: dict[str, float] = {}
    for service in services:
        for node, distance in nx.shortest_path_length(
            graph, service, weight="weight"
        ).items():
            lengths[node] = min(lengths.get(node, float("inf")), float(distance))
    bands = []
    for band in sorted(time_bands):
        within = [node for node, distance in lengths.items() if distance <= band]
        hull = MultiPoint(
            [(graph.nodes[node]["x"], graph.nodes[node]["y"]) for node in within]
        ).convex_hull
        bands.append(
            {
                "time_band": band,
                "reachable_nodes": within,
                "reachable_count": len(within),
                "footprint_wkt": hull.wkt,
            }
        )
    return {"n_services": len(services), "isochrones": bands}
