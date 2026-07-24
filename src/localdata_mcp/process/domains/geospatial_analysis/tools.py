"""localdata_mcp/process/domains/geospatial_analysis/tools.py — E10.e ToolSpecs (stats).

The capability probe and the coordinate-statistics tools' ToolSpecs
(carried by name from `main`): `check_geospatial_capabilities`,
`analyze_spatial_autocorrelation`, `find_spatial_hotspots`,
`calculate_spatial_distances`. The geometry and network halves live
in geo_tools.py (codesize split). Thin over spatial_stats.py with the
X-2 addressing contract. The k-NN neighbour count and the significance
level default to the operator-configured process values fetched through
the guard's `process_defaults()` seam (the tool layer never reads NX-2,
section 6.2). Neighbors: spec_modules.py rosters both this module and
geo_tools.py.
"""

from __future__ import annotations

from typing import Any

from localdata_mcp.nexus.contract.spec import Param, TypeShape, tool_spec

from localdata_mcp.ingest.runtime import chokepoint
from ..support import addressed_frame, source_params
from .capabilities import check_capabilities
from .spatial_stats import (
    spatial_autocorrelation,
    spatial_distances,
    spatial_hotspots,
)

_X = Param("x_column", str, "The x/longitude column (default 'x').", required=False)
_Y = Param("y_column", str, "The y/latitude column (default 'y').", required=False)


@tool_spec(
    name="check_geospatial_capabilities",
    summary=(
        "Report which geospatial backend libraries are installed and "
        "what the geospatial extra enables — answers unconditionally."
    ),
    params=(),
    input_shape=TypeShape.NONE,
    output_shape=TypeShape.SCALAR,
    domain="geospatial_analysis",
)
def check_geospatial_capabilities() -> Any:
    return check_capabilities()


@tool_spec(
    name="analyze_spatial_autocorrelation",
    summary=(
        "Global Moran's I on an addressed point source: whether nearby "
        "locations hold similar value_column values, over a k-nearest-"
        "neighbour neighbourhood. Reports I, z-score, and p-value."
    ),
    params=(
        *source_params(),
        Param("value_column", str, "The measured value column."),
        _X,
        _Y,
        Param(
            "k_neighbors",
            int,
            "Neighbours per point (default: the configured process value).",
            required=False,
        ),
    ),
    input_shape=TypeShape.TABULAR,
    output_shape=TypeShape.SCALAR,
    domain="geospatial_analysis",
)
def analyze_spatial_autocorrelation(
    value_column: str,
    endpoint: str | None = None,
    path: str | None = None,
    table: str | None = None,
    query: str | None = None,
    **knobs: Any,
) -> Any:
    frame, source = addressed_frame(endpoint, path, table, query)
    defaults = chokepoint().process_defaults()
    result = spatial_autocorrelation(
        frame,
        value_column,
        default_k_neighbors=defaults.spatial_k_neighbors,
        **knobs,
    )
    result["source"] = source
    return result


@tool_spec(
    name="find_spatial_hotspots",
    summary=(
        "Getis-Ord Gi* hot- and cold-spots on an addressed point "
        "source: the statistically significant clusters of high or low "
        "value_column at the significance level."
    ),
    params=(
        *source_params(),
        Param("value_column", str, "The measured value column."),
        _X,
        _Y,
        Param(
            "significance_level",
            float,
            "Two-sided significance (default: the configured process value).",
            required=False,
        ),
    ),
    input_shape=TypeShape.TABULAR,
    output_shape=TypeShape.SCALAR,
    domain="geospatial_analysis",
)
def find_spatial_hotspots(
    value_column: str,
    endpoint: str | None = None,
    path: str | None = None,
    table: str | None = None,
    query: str | None = None,
    **knobs: Any,
) -> Any:
    frame, source = addressed_frame(endpoint, path, table, query)
    defaults = chokepoint().process_defaults()
    result = spatial_hotspots(
        frame,
        value_column,
        default_significance=defaults.significance_level,
        default_k_neighbors=defaults.spatial_k_neighbors,
        **knobs,
    )
    result["source"] = source
    return result


@tool_spec(
    name="calculate_spatial_distances",
    summary=(
        "Summarize pairwise Euclidean distances between the points of "
        "an addressed source (min/max/mean/median), bounded by point "
        "count to keep the matrix legible."
    ),
    params=(*source_params(), _X, _Y),
    input_shape=TypeShape.TABULAR,
    output_shape=TypeShape.SCALAR,
    domain="geospatial_analysis",
)
def calculate_spatial_distances(
    endpoint: str | None = None,
    path: str | None = None,
    table: str | None = None,
    query: str | None = None,
    **knobs: Any,
) -> Any:
    frame, source = addressed_frame(endpoint, path, table, query)
    result = spatial_distances(frame, **knobs)
    result["source"] = source
    return result
