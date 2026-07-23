"""localdata_mcp/process/domains/geospatial_analysis/geometry_ops.py — FR-301.

The geometry-pair trio carried by name from `main`:
`perform_spatial_join` (attach the addressed frame's attributes to the
inline geometries they relate to), `perform_spatial_overlay` (a set
operation between the addressed geometries and the inline ones), and
`aggregate_points_in_polygons` (summarize an addressed point
measurement inside each inline polygon). The X-2 contract addresses
ONE source, so the second geometry set arrives inline as a WKT list —
no second addressing seam, no string-sniffing. Neighbors:
capabilities.py guards the stack; tools.py declares the ToolSpecs.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..support import invalid_source_refusal, require_columns
from .capabilities import require_geostack

JOIN_PREDICATES = ("intersects", "within", "contains")
OVERLAY_OPERATIONS = ("intersection", "union", "difference", "symmetric_difference")


def _geoframe_from_wkt(geometries: list[str], label: str) -> Any:
    import geopandas as gpd
    from shapely import wkt

    if not geometries:
        raise invalid_source_refusal(f"{label} must supply at least one WKT geometry.")
    try:
        shapes = [wkt.loads(text) for text in geometries]
    except Exception as failure:  # noqa: BLE001 — malformed WKT is a refusal
        raise invalid_source_refusal(f"{label}: invalid WKT — {failure}") from None
    return gpd.GeoDataFrame({"right_id": range(len(shapes))}, geometry=shapes)


def spatial_join(
    frame: pd.DataFrame,
    geometry_column: str,
    right_geometries: list[str],
    predicate: str = "intersects",
) -> dict[str, Any]:
    """Join the addressed frame's rows to the inline geometries."""
    require_geostack()
    if predicate not in JOIN_PREDICATES:
        raise invalid_source_refusal(
            f"Unknown predicate {predicate!r} — one of {list(JOIN_PREDICATES)}."
        )
    import geopandas as gpd
    from shapely import wkt

    require_columns(frame, geometry_column)
    try:
        left_shapes = [wkt.loads(text) for text in frame[geometry_column]]
    except Exception as failure:  # noqa: BLE001
        raise invalid_source_refusal(
            f"{geometry_column!r}: invalid WKT — {failure}"
        ) from None
    left = gpd.GeoDataFrame(frame.drop(columns=[geometry_column]), geometry=left_shapes)
    right = _geoframe_from_wkt(right_geometries, "right_geometries")
    joined = gpd.sjoin(left, right, predicate=predicate, how="inner")
    return {
        "predicate": predicate,
        "left_rows": int(len(left)),
        "right_geometries": len(right_geometries),
        "match_count": int(len(joined)),
        "matches": [
            {"left_index": int(index), "right_id": int(row["right_id"])}
            for index, row in joined.iterrows()
        ],
    }


def spatial_overlay(
    frame: pd.DataFrame,
    geometry_column: str,
    right_geometries: list[str],
    operation: str = "intersection",
) -> dict[str, Any]:
    """A set operation between the addressed and inline geometries."""
    require_geostack()
    if operation not in OVERLAY_OPERATIONS:
        raise invalid_source_refusal(
            f"Unknown operation {operation!r} — one of {list(OVERLAY_OPERATIONS)}."
        )
    import geopandas as gpd
    from shapely import wkt

    require_columns(frame, geometry_column)
    left = gpd.GeoDataFrame(
        {"left_id": range(len(frame))},
        geometry=[wkt.loads(text) for text in frame[geometry_column]],
    )
    right = _geoframe_from_wkt(right_geometries, "right_geometries")
    result = gpd.overlay(left, right, how=operation)
    return {
        "operation": operation,
        "input_counts": {"left": int(len(left)), "right": int(len(right))},
        "output_count": int(len(result)),
        "geometries": [geometry.wkt for geometry in result.geometry],
    }


def aggregate_in_polygons(
    frame: pd.DataFrame,
    value_column: str,
    polygons: list[str],
    x_column: str = "x",
    y_column: str = "y",
    aggregations: list[str] | None = None,
) -> dict[str, Any]:
    """Summarize the addressed point measurement inside each polygon."""
    require_geostack()
    import geopandas as gpd
    from shapely import wkt
    from shapely.geometry import Point

    require_columns(frame, value_column, x_column, y_column)
    functions = aggregations or ["mean", "sum", "count"]
    points = gpd.GeoDataFrame(
        frame[[value_column]].copy(),
        geometry=[Point(xy) for xy in zip(frame[x_column], frame[y_column])],
    )
    polygon_frame = _geoframe_from_wkt(polygons, "polygons")
    joined = gpd.sjoin(points, polygon_frame, predicate="within", how="inner")
    grouped = joined.groupby("right_id")[value_column].agg(functions)
    return {
        "value_column": value_column,
        "aggregations": functions,
        "n_polygons": len(polygons),
        "n_points_matched": int(len(joined)),
        "per_polygon": {
            str(polygon_id): {function: float(row[function]) for function in functions}
            for polygon_id, row in grouped.iterrows()
        },
    }
