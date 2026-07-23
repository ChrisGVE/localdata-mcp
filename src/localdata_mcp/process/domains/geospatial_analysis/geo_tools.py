"""localdata_mcp/process/domains/geospatial_analysis/geo_tools.py — E10.e ToolSpecs (ops).

The geometry-pair and network-routing tools' ToolSpecs (carried by
name from `main`): `perform_spatial_join`, `perform_spatial_overlay`,
`aggregate_points_in_polygons`, `optimize_route`,
`analyze_accessibility`, `generate_service_isochrones`. Each addresses
ONE source through the X-2 contract; the second geometry/edge set
arrives inline (WKT list or edge triples) so no second addressing
seam is invented. tools.py holds the capability probe and the
coordinate-statistics half (codesize split). Neighbors:
geometry_ops.py / network_ops.py compute.
"""

from __future__ import annotations

from typing import Any

from localdata_mcp.nexus.contract.spec import Param, TypeShape, tool_spec

from ..support import addressed_frame, source_params
from .geometry_ops import aggregate_in_polygons, spatial_join, spatial_overlay
from .network_ops import analyze_accessibility, optimize_route, service_isochrones

_GEOMETRY = Param(
    "geometry_column", str, "Column of WKT geometries in the addressed source."
)
_RIGHT = Param("right_geometries", list, "The second geometry set as WKT strings.")
_EDGES = Param(
    "edges", list, "Network edges as [source, target] or [source, target, weight]."
)
_NODE_ID = Param(
    "node_id_column", str, "The node-id column (default 'id').", required=False
)
_NX = Param("x_column", str, "The x column (default 'x').", required=False)
_NY = Param("y_column", str, "The y column (default 'y').", required=False)


@tool_spec(
    name="perform_spatial_join",
    summary=(
        "Attach the addressed source's rows to the inline right_"
        "geometries they spatially relate to (predicate intersects "
        "(default), within, or contains). The addressed source's "
        "geometry_column holds WKT."
    ),
    params=(
        *source_params(),
        _GEOMETRY,
        _RIGHT,
        Param(
            "predicate",
            str,
            "intersects (default), within, or contains.",
            required=False,
        ),
    ),
    input_shape=TypeShape.TABULAR,
    output_shape=TypeShape.SCALAR,
    domain="geospatial_analysis",
)
def perform_spatial_join(
    geometry_column: str,
    right_geometries: list[Any],
    endpoint: str | None = None,
    path: str | None = None,
    table: str | None = None,
    query: str | None = None,
    **knobs: Any,
) -> Any:
    frame, source = addressed_frame(endpoint, path, table, query)
    result = spatial_join(frame, geometry_column, right_geometries, **knobs)
    result["source"] = source
    return result


@tool_spec(
    name="perform_spatial_overlay",
    summary=(
        "Set operation (intersection (default), union, difference, "
        "symmetric_difference) between the addressed source's WKT "
        "geometry_column and the inline right_geometries."
    ),
    params=(
        *source_params(),
        _GEOMETRY,
        _RIGHT,
        Param(
            "operation",
            str,
            "intersection (default), union, difference, symmetric_difference.",
            required=False,
        ),
    ),
    input_shape=TypeShape.TABULAR,
    output_shape=TypeShape.GEO,
    domain="geospatial_analysis",
)
def perform_spatial_overlay(
    geometry_column: str,
    right_geometries: list[Any],
    endpoint: str | None = None,
    path: str | None = None,
    table: str | None = None,
    query: str | None = None,
    **knobs: Any,
) -> Any:
    frame, source = addressed_frame(endpoint, path, table, query)
    result = spatial_overlay(frame, geometry_column, right_geometries, **knobs)
    result["source"] = source
    return result


@tool_spec(
    name="aggregate_points_in_polygons",
    summary=(
        "Summarize an addressed point source's value_column inside each "
        "inline polygon (WKT list): mean/sum/count by default."
    ),
    params=(
        *source_params(),
        Param("value_column", str, "The point measurement column."),
        Param("polygons", list, "Containing polygons as WKT strings."),
        Param("x_column", str, "The x column (default 'x').", required=False),
        Param("y_column", str, "The y column (default 'y').", required=False),
        Param(
            "aggregations",
            list,
            "Aggregations to apply (default mean, sum, count).",
            required=False,
        ),
    ),
    input_shape=TypeShape.TABULAR,
    output_shape=TypeShape.SCALAR,
    domain="geospatial_analysis",
)
def aggregate_points_in_polygons(
    value_column: str,
    polygons: list[Any],
    endpoint: str | None = None,
    path: str | None = None,
    table: str | None = None,
    query: str | None = None,
    **knobs: Any,
) -> Any:
    frame, source = addressed_frame(endpoint, path, table, query)
    result = aggregate_in_polygons(frame, value_column, polygons, **knobs)
    result["source"] = source
    return result


@tool_spec(
    name="optimize_route",
    summary=(
        "Order waypoints greedily and connect them by shortest path "
        "over a network whose nodes are the addressed source (id, x, y) "
        "and whose edges arrive inline. Reports the order, path, and "
        "total distance."
    ),
    params=(
        *source_params(),
        _EDGES,
        Param("waypoints", list, "Node ids to visit."),
        _NODE_ID,
        _NX,
        _NY,
        Param(
            "return_to_start",
            bool,
            "Close the loop back to the first waypoint (default false).",
            required=False,
        ),
    ),
    input_shape=TypeShape.TABULAR,
    output_shape=TypeShape.SCALAR,
    domain="geospatial_analysis",
)
def optimize_route_tool(
    edges: list[Any],
    waypoints: list[Any],
    endpoint: str | None = None,
    path: str | None = None,
    table: str | None = None,
    query: str | None = None,
    **knobs: Any,
) -> Any:
    frame, source = addressed_frame(endpoint, path, table, query)
    result = optimize_route(frame, edges, waypoints, **knobs)
    result["source"] = source
    return result


@tool_spec(
    name="analyze_accessibility",
    summary=(
        "Score demand nodes by travel time to their nearest service "
        "node over a network (addressed nodes, inline edges), optionally "
        "capped at max_travel_time."
    ),
    params=(
        *source_params(),
        _EDGES,
        Param("service_locations", list, "Service node ids."),
        Param("demand_locations", list, "Demand node ids."),
        _NODE_ID,
        _NX,
        _NY,
        Param(
            "max_travel_time",
            float,
            "Reachability cap (default: unbounded).",
            required=False,
        ),
    ),
    input_shape=TypeShape.TABULAR,
    output_shape=TypeShape.SCALAR,
    domain="geospatial_analysis",
)
def analyze_accessibility_tool(
    edges: list[Any],
    service_locations: list[Any],
    demand_locations: list[Any],
    endpoint: str | None = None,
    path: str | None = None,
    table: str | None = None,
    query: str | None = None,
    **knobs: Any,
) -> Any:
    frame, source = addressed_frame(endpoint, path, table, query)
    result = analyze_accessibility(
        frame, edges, service_locations, demand_locations, **knobs
    )
    result["source"] = source
    return result


@tool_spec(
    name="generate_service_isochrones",
    summary=(
        "The node set reachable from the service nodes within each "
        "travel-time band, with its convex-hull footprint, over a "
        "network (addressed nodes, inline edges)."
    ),
    params=(
        *source_params(),
        _EDGES,
        Param("service_locations", list, "Service node ids."),
        Param("time_bands", list, "Travel-time bands to map."),
        _NODE_ID,
        _NX,
        _NY,
    ),
    input_shape=TypeShape.TABULAR,
    output_shape=TypeShape.GEO,
    domain="geospatial_analysis",
)
def generate_service_isochrones_tool(
    edges: list[Any],
    service_locations: list[Any],
    time_bands: list[Any],
    endpoint: str | None = None,
    path: str | None = None,
    table: str | None = None,
    query: str | None = None,
    **knobs: Any,
) -> Any:
    frame, source = addressed_frame(endpoint, path, table, query)
    result = service_isochrones(frame, edges, service_locations, time_bands, **knobs)
    result["source"] = source
    return result
