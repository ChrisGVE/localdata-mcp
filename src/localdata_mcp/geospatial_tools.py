"""MCP tool functions for the geospatial domain.

Thin adapters over :mod:`localdata_mcp.domains.geospatial_analysis`, mirroring
:mod:`datascience_tools`: take a SQLAlchemy engine and a query, obtain a
DataFrame, delegate to the domain, and return a JSON-serializable dict.

Two shapes of spatial input arrive from SQL, and each gets its own helper:

- **Point data** — a pair of numeric coordinate columns (``_coordinates``).
- **Geometry data** — a single text column holding WKT, which is how a geometry
  survives a round trip through a database that has no geometry type
  (``_wkt_frame``).

Network tools take two queries, one for nodes and one for edges, because that is
how a graph is stored relationally and what the domain's network builder wants.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sqlalchemy.engine import Engine

from .datascience_tools import _query_to_dataframe, _require_columns
from .logging_manager import get_logger

logger = get_logger(__name__)

# A pairwise distance matrix grows with the square of the row count, so the
# distance tool summarises by default and caps how many points it will accept.
MAX_DISTANCE_POINTS = 2_000

# Cap on individual pairs returned when a caller asks for them explicitly.
MAX_DISTANCE_PAIRS = 10_000


def _coordinates(df: pd.DataFrame, x_column: str, y_column: str) -> "pd.DataFrame":
    """Return the frame restricted to rows with usable numeric coordinates."""
    _require_columns(df, x_column, y_column)
    frame = df.copy()
    frame[x_column] = pd.to_numeric(frame[x_column], errors="coerce")
    frame[y_column] = pd.to_numeric(frame[y_column], errors="coerce")
    frame = frame.dropna(subset=[x_column, y_column])

    if frame.empty:
        raise ValueError(
            f"No rows with numeric values in both '{x_column}' and '{y_column}'."
        )
    return frame


def _wkt_frame(df: pd.DataFrame, geometry_column: str, crs: Optional[str]):
    """Parse a WKT text column into a GeoDataFrame.

    A database without a geometry type stores geometries as WKT text, so that is
    what the query returns and what has to be parsed back before geopandas can
    do anything spatial with it.
    """
    import geopandas as gpd
    from shapely import wkt as shapely_wkt

    _require_columns(df, geometry_column)

    try:
        geometries = df[geometry_column].map(
            lambda value: shapely_wkt.loads(value) if isinstance(value, str) else value
        )
    except Exception as exc:
        raise ValueError(
            f"Column '{geometry_column}' does not contain valid WKT geometries: {exc}"
        ) from exc

    frame = df.drop(columns=[geometry_column])
    return gpd.GeoDataFrame(frame, geometry=list(geometries), crs=crs or "EPSG:4326")


def _spatial_frame(
    df: pd.DataFrame,
    geometry_column: Optional[str],
    x_column: Optional[str],
    y_column: Optional[str],
    crs: Optional[str],
):
    """Build a GeoDataFrame from whichever spatial encoding the query produced.

    A geometry reaches us either as WKT in one column or as a pair of coordinate
    columns. Joining points to polygons is the commonest spatial join there is,
    and point tables rarely carry a WKT column, so both are accepted.
    """
    import geopandas as gpd
    from shapely.geometry import Point

    if geometry_column and geometry_column in df.columns:
        return _wkt_frame(df, geometry_column, crs)

    if x_column and y_column and {x_column, y_column} <= set(df.columns):
        frame = _coordinates(df, x_column, y_column)
        return gpd.GeoDataFrame(
            frame,
            geometry=[Point(x, y) for x, y in zip(frame[x_column], frame[y_column])],
            crs=crs or "EPSG:4326",
        )

    raise ValueError(
        f"No geometry found. Provide either a WKT column (looked for "
        f"'{geometry_column}') or a coordinate pair (looked for '{x_column}' and "
        f"'{y_column}'). Available columns: {list(df.columns)}"
    )


def _geometries_to_wkt(frame: pd.DataFrame) -> List[Dict[str, Any]]:
    """Render a (Geo)DataFrame as records, geometries written back out as WKT."""
    records = []
    for row in frame.to_dict(orient="records"):
        records.append(
            {
                key: (value.wkt if hasattr(value, "wkt") else value)
                for key, value in row.items()
            }
        )
    return records


def _network_data(
    engine: Engine,
    nodes_query: str,
    edges_query: str,
    node_id_column: str,
    x_column: str,
    y_column: str,
    source_column: str,
    target_column: str,
    weight_column: Optional[str],
) -> Dict[str, Any]:
    """Build the ``{"nodes": [...], "edges": [...]}`` mapping the domain expects."""
    nodes_df = _coordinates(
        _query_to_dataframe(engine, nodes_query), x_column, y_column
    )
    _require_columns(nodes_df, node_id_column)
    edges_df = _query_to_dataframe(engine, edges_query)
    _require_columns(edges_df, source_column, target_column, weight_column)

    nodes = [
        {
            "id": row[node_id_column],
            "x": float(row[x_column]),
            "y": float(row[y_column]),
        }
        for row in nodes_df.to_dict(orient="records")
    ]

    known_ids = {node["id"] for node in nodes}
    edges = []
    for row in edges_df.to_dict(orient="records"):
        source, target = row[source_column], row[target_column]
        if source not in known_ids or target not in known_ids:
            raise ValueError(
                f"Edge ({source} -> {target}) references a node absent from the "
                f"nodes query. Check that '{node_id_column}' and "
                f"'{source_column}'/'{target_column}' identify the same nodes."
            )
        edge = {"source": source, "target": target}
        if weight_column:
            edge["weight"] = float(row[weight_column])
        edges.append(edge)

    if not edges:
        raise ValueError("The edges query returned no rows; the network is empty.")

    return {"nodes": nodes, "edges": edges}


def _coerce_node_ids(node_ids: List[Any], network: Dict[str, Any]) -> List[Any]:
    """Match caller-supplied node ids to the type the nodes query produced.

    MCP arguments arrive as JSON, so an integer node id may reach us as the
    string "7". Comparing that against an int node id silently finds nothing.
    """
    known = {node["id"] for node in network["nodes"]}
    by_text = {str(node_id): node_id for node_id in known}

    resolved = []
    for node_id in node_ids:
        if node_id in known:
            resolved.append(node_id)
        elif str(node_id) in by_text:
            resolved.append(by_text[str(node_id)])
        else:
            raise ValueError(
                f"Node '{node_id}' is not in the network. "
                f"Known nodes include: {sorted(map(str, known))[:10]}"
            )
    return resolved


def tool_check_geospatial_capabilities() -> Dict[str, Any]:
    """Report which geospatial backends are installed and what they enable."""
    from .domains.geospatial_analysis import check_geospatial_capabilities

    capabilities = check_geospatial_capabilities()
    status = capabilities["dependency_status"]

    return {
        "available_libraries": {
            library.value: available
            for library, available in status.available_libraries.items()
        },
        "versions": {
            library.value: version for library, version in status.versions.items()
        },
        "missing_libraries": [
            library.value for library in status.get_missing_libraries()
        ],
        "has_core_geospatial": status.has_core_geospatial(),
        "available_features": capabilities["available_features"],
        "fallback_active": capabilities["fallback_active"],
    }


def tool_analyze_spatial_autocorrelation(
    engine: Engine,
    query: str,
    value_column: str,
    x_column: str = "x",
    y_column: str = "y",
    method: str = "moran",
    k_neighbors: int = 8,
) -> Dict[str, Any]:
    """Test whether nearby locations hold similar values."""
    from .domains.geospatial_analysis import analyze_spatial_autocorrelation

    df = _coordinates(_query_to_dataframe(engine, query), x_column, y_column)
    _require_columns(df, value_column)

    if len(df) <= k_neighbors:
        raise ValueError(
            f"Spatial autocorrelation over {k_neighbors} neighbours needs more than "
            f"{k_neighbors} points; the query returned {len(df)}."
        )

    return analyze_spatial_autocorrelation(
        df,
        value_column=value_column,
        coordinate_columns=[x_column, y_column],
        method=method,
        k_neighbors=k_neighbors,
    )


def tool_find_spatial_hotspots(
    engine: Engine,
    query: str,
    value_column: str,
    x_column: str = "x",
    y_column: str = "y",
    significance_level: float = 0.05,
) -> Dict[str, Any]:
    """Locate statistically significant hot and cold spots (Getis-Ord Gi*)."""
    from .domains.geospatial_analysis import perform_spatial_clustering

    df = _coordinates(_query_to_dataframe(engine, query), x_column, y_column)
    _require_columns(df, value_column)

    # The domain function analyses the first numeric non-coordinate column, so
    # the caller's choice is honoured by narrowing the frame to it.
    clustered = perform_spatial_clustering(
        df[[x_column, y_column, value_column]],
        coordinate_columns=[x_column, y_column],
        method="hotspot",
        significance_level=significance_level,
    )

    return {
        "method": "getis_ord_gi_star",
        "n_points": int(len(clustered)),
        "n_hotspots": int(clustered["is_hotspot"].sum()),
        "n_coldspots": int(clustered["is_coldspot"].sum()),
        "significance_level": significance_level,
        "value_column": value_column,
        "points": clustered.to_dict(orient="records"),
        "interpretation": (
            f"{int(clustered['is_hotspot'].sum())} of {len(clustered)} locations sit "
            f"in a cluster of unusually high '{value_column}', and "
            f"{int(clustered['is_coldspot'].sum())} in a cluster of unusually low "
            f"values, at the {significance_level} level."
        ),
    }


def tool_calculate_spatial_distances(
    engine: Engine,
    query: str,
    x_column: str = "x",
    y_column: str = "y",
    distance_type: str = "euclidean",
    reference_query: Optional[str] = None,
    include_pairs: bool = False,
) -> Dict[str, Any]:
    """Measure distances between points, summarised rather than returned whole."""
    from .domains.geospatial_analysis import calculate_spatial_distance

    df = _coordinates(_query_to_dataframe(engine, query), x_column, y_column)
    if len(df) > MAX_DISTANCE_POINTS:
        raise ValueError(
            f"Distance analysis is limited to {MAX_DISTANCE_POINTS} points because the "
            f"matrix grows with their square; the query returned {len(df)}. "
            "Narrow the query or aggregate first."
        )

    reference = None
    if reference_query:
        reference = _coordinates(
            _query_to_dataframe(engine, reference_query), x_column, y_column
        )

    matrix = calculate_spatial_distance(
        df,
        reference,
        coordinate_columns=[x_column, y_column],
        distance_type=distance_type,
        output_format="matrix",
    )
    matrix = np.asarray(matrix, dtype=float)

    # Self-distances are structurally zero and would drag every summary down.
    if reference is None:
        offdiag = matrix[~np.eye(len(matrix), dtype=bool)]
    else:
        offdiag = matrix.ravel()

    nearest_index = None
    if reference is None and len(matrix) > 1:
        # Mask by selection, not by adding `eye * inf`: off the diagonal that
        # product is 0 * inf = nan, which poisons every entry and makes argmin
        # return the first column rather than the nearest neighbour.
        masked = np.where(np.eye(len(matrix), dtype=bool), np.inf, matrix)
        nearest_index = masked.argmin(axis=1)

    result: Dict[str, Any] = {
        "distance_type": distance_type,
        "unit": "kilometres" if distance_type == "haversine" else "coordinate units",
        "n_points": int(matrix.shape[0]),
        "n_reference_points": int(matrix.shape[1]),
        "summary": {
            "min": float(offdiag.min()),
            "max": float(offdiag.max()),
            "mean": float(offdiag.mean()),
            "median": float(np.median(offdiag)),
        },
    }

    if nearest_index is not None:
        nearest_distance = matrix[np.arange(len(matrix)), nearest_index]
        result["nearest_neighbors"] = [
            {
                "point_index": int(i),
                "nearest_index": int(j),
                "distance": float(d),
            }
            for i, (j, d) in enumerate(zip(nearest_index, nearest_distance))
        ]
        result["summary"]["mean_nearest_neighbor_distance"] = float(
            nearest_distance.mean()
        )

    if include_pairs:
        n_pairs = matrix.shape[0] * matrix.shape[1]
        if n_pairs > MAX_DISTANCE_PAIRS:
            raise ValueError(
                f"Returning every pair would produce {n_pairs} rows, over the "
                f"{MAX_DISTANCE_PAIRS} limit. Narrow the query or drop include_pairs."
            )
        result["pairs"] = [
            {"point1_index": int(i), "point2_index": int(j), "distance": float(d)}
            for i, row in enumerate(matrix)
            for j, d in enumerate(row)
        ]

    return result


def tool_optimize_route(
    engine: Engine,
    nodes_query: str,
    edges_query: str,
    waypoints: List[Any],
    node_id_column: str = "id",
    x_column: str = "x",
    y_column: str = "y",
    source_column: str = "source",
    target_column: str = "target",
    weight_column: Optional[str] = None,
    return_to_start: bool = False,
    optimization_method: str = "greedy",
) -> Dict[str, Any]:
    """Order and connect waypoints into the cheapest route across a network."""
    from .domains.geospatial_analysis import optimize_route

    network = _network_data(
        engine,
        nodes_query,
        edges_query,
        node_id_column,
        x_column,
        y_column,
        source_column,
        target_column,
        weight_column,
    )
    if len(waypoints) < 2:
        raise ValueError("A route needs at least two waypoints.")

    route = optimize_route(
        network,
        _coerce_node_ids(waypoints, network),
        return_to_start=return_to_start,
        optimization_method=optimization_method,
    )

    return {
        "route_path": list(route.route_path),
        "route_coordinates": [list(coord) for coord in route.route_coordinates],
        "total_distance": route.total_distance,
        "total_time": route.total_time,
        "waypoints": list(route.waypoints),
        "path_geometry": route.path_geometry.wkt if route.path_geometry else None,
        "optimization_method": optimization_method,
        "return_to_start": return_to_start,
    }


def tool_analyze_accessibility(
    engine: Engine,
    nodes_query: str,
    edges_query: str,
    service_locations: List[Any],
    demand_locations: List[Any],
    node_id_column: str = "id",
    x_column: str = "x",
    y_column: str = "y",
    source_column: str = "source",
    target_column: str = "target",
    weight_column: Optional[str] = None,
    max_travel_time: Optional[float] = None,
    impedance_function: str = "linear",
) -> Dict[str, Any]:
    """Score how well a set of service points covers a set of demand points."""
    from .domains.geospatial_analysis import analyze_accessibility

    network = _network_data(
        engine,
        nodes_query,
        edges_query,
        node_id_column,
        x_column,
        y_column,
        source_column,
        target_column,
        weight_column,
    )

    result = analyze_accessibility(
        network,
        _coerce_node_ids(service_locations, network),
        _coerce_node_ids(demand_locations, network),
        max_travel_time=max_travel_time,
        impedance_function=impedance_function,
    )

    return {
        "accessibility_scores": dict(result.accessibility_scores),
        "reachable_locations": list(result.reachable_locations),
        "unreachable_locations": [
            node
            for node in _coerce_node_ids(demand_locations, network)
            if node not in result.reachable_locations
        ],
        "travel_times": dict(result.travel_times),
        "service_coverage": dict(result.service_coverage),
        "analysis_parameters": dict(result.analysis_parameters),
    }


def tool_generate_service_isochrones(
    engine: Engine,
    nodes_query: str,
    edges_query: str,
    service_locations: List[Any],
    time_bands: List[float],
    node_id_column: str = "id",
    x_column: str = "x",
    y_column: str = "y",
    source_column: str = "source",
    target_column: str = "target",
    weight_column: Optional[str] = None,
    resolution: int = 50,
) -> Dict[str, Any]:
    """Draw the area reachable from service points within each travel-time band."""
    from .domains.geospatial_analysis import generate_service_isochrones

    if not time_bands:
        raise ValueError("At least one travel-time band is required.")

    network = _network_data(
        engine,
        nodes_query,
        edges_query,
        node_id_column,
        x_column,
        y_column,
        source_column,
        target_column,
        weight_column,
    )

    result = generate_service_isochrones(
        network,
        _coerce_node_ids(service_locations, network),
        sorted(time_bands),
        resolution=resolution,
    )

    return {
        "time_bands": list(result.time_bands),
        "isochrones": [
            {
                "time_band": band,
                "geometry": polygon.wkt if polygon is not None else None,
                "area": result.coverage_areas.get(band, 0.0),
            }
            for band, polygon in zip(
                sorted(result.time_bands), result.isochrone_polygons
            )
        ],
        "coverage_areas": {str(k): v for k, v in result.coverage_areas.items()},
        "service_points": list(result.service_points),
    }


def tool_perform_spatial_join(
    engine: Engine,
    left_query: str,
    right_query: str,
    left_geometry_column: str = "geometry",
    right_geometry_column: str = "geometry",
    join_type: str = "intersects",
    how: str = "inner",
    x_column: str = "x",
    y_column: str = "y",
    crs: Optional[str] = None,
) -> Dict[str, Any]:
    """Attach the attributes of one geometry set to another by spatial relation."""
    from .domains.geospatial_analysis import perform_spatial_join

    left = _spatial_frame(
        _query_to_dataframe(engine, left_query),
        left_geometry_column,
        x_column,
        y_column,
        crs,
    )
    right = _spatial_frame(
        _query_to_dataframe(engine, right_query),
        right_geometry_column,
        x_column,
        y_column,
        crs,
    )

    result = perform_spatial_join(left, right, join_type=join_type, how=how)

    return {
        "join_type": result.join_type.value,
        "how": how,
        "match_counts": dict(result.match_counts),
        "row_count": int(len(result.joined_data)),
        "rows": _geometries_to_wkt(result.joined_data),
    }


def tool_perform_spatial_overlay(
    engine: Engine,
    left_query: str,
    right_query: str,
    operation: str = "intersection",
    left_geometry_column: str = "geometry",
    right_geometry_column: str = "geometry",
    crs: Optional[str] = None,
) -> Dict[str, Any]:
    """Combine two geometry sets with a set operation (intersection, union, ...)."""
    from .domains.geospatial_analysis import perform_spatial_overlay

    left = _wkt_frame(
        _query_to_dataframe(engine, left_query), left_geometry_column, crs
    )
    right = _wkt_frame(
        _query_to_dataframe(engine, right_query), right_geometry_column, crs
    )

    result = perform_spatial_overlay(left, right, operation=operation)

    return {
        "operation": result.operation.value,
        "input_counts": dict(result.input_counts),
        "output_count": int(result.output_count),
        "rows": _geometries_to_wkt(result.result_data),
    }


def tool_aggregate_points_in_polygons(
    engine: Engine,
    point_query: str,
    polygon_query: str,
    value_column: str,
    x_column: str = "x",
    y_column: str = "y",
    polygon_geometry_column: str = "geometry",
    aggregation_functions: Optional[List[str]] = None,
    crs: Optional[str] = None,
) -> Dict[str, Any]:
    """Summarise a point measurement within each polygon that contains it."""
    from .domains.geospatial_analysis import aggregate_points_in_polygons

    functions = aggregation_functions or ["mean", "sum", "count"]

    points_df = _query_to_dataframe(engine, point_query)
    _require_columns(points_df, value_column)

    polygons = _wkt_frame(
        _query_to_dataframe(engine, polygon_query), polygon_geometry_column, crs
    )
    points = _spatial_frame(points_df, None, x_column, y_column, str(polygons.crs))

    aggregated = aggregate_points_in_polygons(
        points, polygons, value_column, aggregation_functions=functions
    )

    return {
        "value_column": value_column,
        "aggregation_functions": functions,
        "n_points": int(len(points)),
        "n_polygons": int(len(polygons)),
        "polygons": _geometries_to_wkt(aggregated),
    }
