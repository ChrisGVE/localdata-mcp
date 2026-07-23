"""MACHINE-WRITTEN by localdata_mcp.nexus.contract.generators.typeshape_registry — DO NOT EDIT.

Per-tool type-shape declarations (ARCHITECTURE.md 6.1 artifact 5).
Regenerate via `python -m localdata_mcp.nexus.contract.generate`;
hand edits fail CI through nexus/contract/check_drift.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Mapping

from localdata_mcp.nexus.contract.spec import TypeShape


@dataclass(frozen=True)
class ShapeEntry:
    """One tool's declared composition facts."""

    input_shape: TypeShape
    output_shape: TypeShape
    streaming_capable: bool
    domain: "str | None"


TOOL_SHAPES: Final[Mapping[str, ShapeEntry]] = {
    "ping": ShapeEntry(
        input_shape=TypeShape.NONE,
        output_shape=TypeShape.SCALAR,
        streaming_capable=False,
        domain=None,
    ),
    "probe_table": ShapeEntry(
        input_shape=TypeShape.NONE,
        output_shape=TypeShape.TABULAR,
        streaming_capable=False,
        domain=None,
    ),
    "probe_vector": ShapeEntry(
        input_shape=TypeShape.NONE,
        output_shape=TypeShape.VECTOR,
        streaming_capable=False,
        domain=None,
    ),
    "probe_matrix": ShapeEntry(
        input_shape=TypeShape.NONE,
        output_shape=TypeShape.MATRIX,
        streaming_capable=False,
        domain=None,
    ),
    "probe_model": ShapeEntry(
        input_shape=TypeShape.NONE,
        output_shape=TypeShape.FITTED_MODEL,
        streaming_capable=False,
        domain=None,
    ),
    "probe_graph": ShapeEntry(
        input_shape=TypeShape.NONE,
        output_shape=TypeShape.GRAPH,
        streaming_capable=False,
        domain=None,
    ),
    "probe_geo": ShapeEntry(
        input_shape=TypeShape.NONE,
        output_shape=TypeShape.GEO,
        streaming_capable=False,
        domain=None,
    ),
    "probe_chart": ShapeEntry(
        input_shape=TypeShape.NONE,
        output_shape=TypeShape.CHART_SPEC,
        streaming_capable=False,
        domain=None,
    ),
    "probe_sink": ShapeEntry(
        input_shape=TypeShape.TABULAR,
        output_shape=TypeShape.NONE,
        streaming_capable=False,
        domain=None,
    ),
    "list_endpoints": ShapeEntry(
        input_shape=TypeShape.NONE,
        output_shape=TypeShape.TABULAR,
        streaming_capable=False,
        domain='ingest',
    ),
    "fetch_chunk": ShapeEntry(
        input_shape=TypeShape.NONE,
        output_shape=TypeShape.TABULAR,
        streaming_capable=True,
        domain='ingest',
    ),
    "close_stream": ShapeEntry(
        input_shape=TypeShape.NONE,
        output_shape=TypeShape.SCALAR,
        streaming_capable=False,
        domain='ingest',
    ),
    "query": ShapeEntry(
        input_shape=TypeShape.NONE,
        output_shape=TypeShape.TABULAR,
        streaming_capable=True,
        domain='ingest',
    ),
    "write_query": ShapeEntry(
        input_shape=TypeShape.NONE,
        output_shape=TypeShape.TABULAR,
        streaming_capable=False,
        domain='ingest',
    ),
    "read_file": ShapeEntry(
        input_shape=TypeShape.NONE,
        output_shape=TypeShape.TABULAR,
        streaming_capable=True,
        domain='ingest',
    ),
    "query_file": ShapeEntry(
        input_shape=TypeShape.NONE,
        output_shape=TypeShape.TABULAR,
        streaming_capable=True,
        domain='ingest',
    ),
    "get_value": ShapeEntry(
        input_shape=TypeShape.NONE,
        output_shape=TypeShape.SCALAR,
        streaming_capable=False,
        domain='ingest',
    ),
    "set_value": ShapeEntry(
        input_shape=TypeShape.NONE,
        output_shape=TypeShape.SCALAR,
        streaming_capable=False,
        domain='ingest',
    ),
    "delete_key": ShapeEntry(
        input_shape=TypeShape.NONE,
        output_shape=TypeShape.SCALAR,
        streaming_capable=False,
        domain='ingest',
    ),
    "list_keys": ShapeEntry(
        input_shape=TypeShape.NONE,
        output_shape=TypeShape.TABULAR,
        streaming_capable=False,
        domain='ingest',
    ),
    "get_node": ShapeEntry(
        input_shape=TypeShape.NONE,
        output_shape=TypeShape.SCALAR,
        streaming_capable=False,
        domain='ingest',
    ),
    "set_node": ShapeEntry(
        input_shape=TypeShape.NONE,
        output_shape=TypeShape.SCALAR,
        streaming_capable=False,
        domain='ingest',
    ),
    "delete_node": ShapeEntry(
        input_shape=TypeShape.NONE,
        output_shape=TypeShape.SCALAR,
        streaming_capable=False,
        domain='ingest',
    ),
    "get_children": ShapeEntry(
        input_shape=TypeShape.NONE,
        output_shape=TypeShape.TABULAR,
        streaming_capable=False,
        domain='ingest',
    ),
    "move_node": ShapeEntry(
        input_shape=TypeShape.NONE,
        output_shape=TypeShape.SCALAR,
        streaming_capable=False,
        domain='ingest',
    ),
    "get_neighbors": ShapeEntry(
        input_shape=TypeShape.NONE,
        output_shape=TypeShape.TABULAR,
        streaming_capable=False,
        domain='ingest',
    ),
    "get_edges": ShapeEntry(
        input_shape=TypeShape.NONE,
        output_shape=TypeShape.TABULAR,
        streaming_capable=False,
        domain='ingest',
    ),
    "add_edge": ShapeEntry(
        input_shape=TypeShape.NONE,
        output_shape=TypeShape.SCALAR,
        streaming_capable=False,
        domain='ingest',
    ),
    "remove_edge": ShapeEntry(
        input_shape=TypeShape.NONE,
        output_shape=TypeShape.SCALAR,
        streaming_capable=False,
        domain='ingest',
    ),
    "find_path": ShapeEntry(
        input_shape=TypeShape.NONE,
        output_shape=TypeShape.SCALAR,
        streaming_capable=False,
        domain='ingest',
    ),
    "get_graph_stats": ShapeEntry(
        input_shape=TypeShape.NONE,
        output_shape=TypeShape.SCALAR,
        streaming_capable=False,
        domain='ingest',
    ),
    "describe_database": ShapeEntry(
        input_shape=TypeShape.NONE,
        output_shape=TypeShape.SCALAR,
        streaming_capable=False,
        domain='explore',
    ),
    "describe_table": ShapeEntry(
        input_shape=TypeShape.NONE,
        output_shape=TypeShape.SCALAR,
        streaming_capable=False,
        domain='explore',
    ),
    "find_table": ShapeEntry(
        input_shape=TypeShape.NONE,
        output_shape=TypeShape.SCALAR,
        streaming_capable=False,
        domain='explore',
    ),
    "profile_data": ShapeEntry(
        input_shape=TypeShape.TABULAR,
        output_shape=TypeShape.SCALAR,
        streaming_capable=False,
        domain='explore',
    ),
    "search_data": ShapeEntry(
        input_shape=TypeShape.TABULAR,
        output_shape=TypeShape.SCALAR,
        streaming_capable=False,
        domain='explore',
    ),
    "map_categories": ShapeEntry(
        input_shape=TypeShape.TABULAR,
        output_shape=TypeShape.SCALAR,
        streaming_capable=False,
        domain='explore',
    ),
    "analyze_hypothesis_test": ShapeEntry(
        input_shape=TypeShape.TABULAR,
        output_shape=TypeShape.SCALAR,
        streaming_capable=False,
        domain='statistical_analysis',
    ),
    "analyze_anova": ShapeEntry(
        input_shape=TypeShape.TABULAR,
        output_shape=TypeShape.SCALAR,
        streaming_capable=False,
        domain='statistical_analysis',
    ),
    "analyze_effect_sizes": ShapeEntry(
        input_shape=TypeShape.TABULAR,
        output_shape=TypeShape.SCALAR,
        streaming_capable=False,
        domain='statistical_analysis',
    ),
    "analyze_ab_test": ShapeEntry(
        input_shape=TypeShape.TABULAR,
        output_shape=TypeShape.SCALAR,
        streaming_capable=False,
        domain='statistical_analysis',
    ),
    "analyze_regression": ShapeEntry(
        input_shape=TypeShape.TABULAR,
        output_shape=TypeShape.FITTED_MODEL,
        streaming_capable=False,
        domain='regression_modeling',
    ),
    "evaluate_model_performance": ShapeEntry(
        input_shape=TypeShape.TABULAR,
        output_shape=TypeShape.SCALAR,
        streaming_capable=False,
        domain='regression_modeling',
    ),
    "analyze_clusters": ShapeEntry(
        input_shape=TypeShape.TABULAR,
        output_shape=TypeShape.SCALAR,
        streaming_capable=False,
        domain='pattern_recognition',
    ),
    "detect_anomalies": ShapeEntry(
        input_shape=TypeShape.TABULAR,
        output_shape=TypeShape.SCALAR,
        streaming_capable=False,
        domain='pattern_recognition',
    ),
    "reduce_dimensions": ShapeEntry(
        input_shape=TypeShape.TABULAR,
        output_shape=TypeShape.MATRIX,
        streaming_capable=False,
        domain='pattern_recognition',
    ),
    "transform_data": ShapeEntry(
        input_shape=TypeShape.TABULAR,
        output_shape=TypeShape.TABULAR,
        streaming_capable=False,
        domain='pattern_recognition',
    ),
    "analyze_time_series": ShapeEntry(
        input_shape=TypeShape.TABULAR,
        output_shape=TypeShape.SCALAR,
        streaming_capable=False,
        domain='time_series',
    ),
    "forecast_time_series": ShapeEntry(
        input_shape=TypeShape.TABULAR,
        output_shape=TypeShape.VECTOR,
        streaming_capable=False,
        domain='time_series',
    ),
    "generate_sample": ShapeEntry(
        input_shape=TypeShape.TABULAR,
        output_shape=TypeShape.TABULAR,
        streaming_capable=False,
        domain='sampling_estimation',
    ),
    "bootstrap_statistic": ShapeEntry(
        input_shape=TypeShape.TABULAR,
        output_shape=TypeShape.SCALAR,
        streaming_capable=False,
        domain='sampling_estimation',
    ),
    "monte_carlo_simulate": ShapeEntry(
        input_shape=TypeShape.TABULAR,
        output_shape=TypeShape.SCALAR,
        streaming_capable=False,
        domain='sampling_estimation',
    ),
    "bayesian_estimate": ShapeEntry(
        input_shape=TypeShape.TABULAR,
        output_shape=TypeShape.SCALAR,
        streaming_capable=False,
        domain='sampling_estimation',
    ),
    "analyze_rfm": ShapeEntry(
        input_shape=TypeShape.TABULAR,
        output_shape=TypeShape.SCALAR,
        streaming_capable=False,
        domain='business_intelligence',
    ),
    "calculate_clv": ShapeEntry(
        input_shape=TypeShape.TABULAR,
        output_shape=TypeShape.SCALAR,
        streaming_capable=False,
        domain='business_intelligence',
    ),
    "analyze_network": ShapeEntry(
        input_shape=TypeShape.TABULAR,
        output_shape=TypeShape.SCALAR,
        streaming_capable=False,
        domain='network_graph',
    ),
    "solve_linear_program": ShapeEntry(
        input_shape=TypeShape.TABULAR,
        output_shape=TypeShape.SCALAR,
        streaming_capable=False,
        domain='optimization',
    ),
    "optimize_constrained": ShapeEntry(
        input_shape=TypeShape.TABULAR,
        output_shape=TypeShape.SCALAR,
        streaming_capable=False,
        domain='optimization',
    ),
    "solve_assignment_problem": ShapeEntry(
        input_shape=TypeShape.TABULAR,
        output_shape=TypeShape.SCALAR,
        streaming_capable=False,
        domain='optimization',
    ),
    "check_geospatial_capabilities": ShapeEntry(
        input_shape=TypeShape.NONE,
        output_shape=TypeShape.SCALAR,
        streaming_capable=False,
        domain='geospatial_analysis',
    ),
    "analyze_spatial_autocorrelation": ShapeEntry(
        input_shape=TypeShape.TABULAR,
        output_shape=TypeShape.SCALAR,
        streaming_capable=False,
        domain='geospatial_analysis',
    ),
    "find_spatial_hotspots": ShapeEntry(
        input_shape=TypeShape.TABULAR,
        output_shape=TypeShape.SCALAR,
        streaming_capable=False,
        domain='geospatial_analysis',
    ),
    "calculate_spatial_distances": ShapeEntry(
        input_shape=TypeShape.TABULAR,
        output_shape=TypeShape.SCALAR,
        streaming_capable=False,
        domain='geospatial_analysis',
    ),
    "perform_spatial_join": ShapeEntry(
        input_shape=TypeShape.TABULAR,
        output_shape=TypeShape.SCALAR,
        streaming_capable=False,
        domain='geospatial_analysis',
    ),
    "perform_spatial_overlay": ShapeEntry(
        input_shape=TypeShape.TABULAR,
        output_shape=TypeShape.GEO,
        streaming_capable=False,
        domain='geospatial_analysis',
    ),
    "aggregate_points_in_polygons": ShapeEntry(
        input_shape=TypeShape.TABULAR,
        output_shape=TypeShape.SCALAR,
        streaming_capable=False,
        domain='geospatial_analysis',
    ),
    "optimize_route": ShapeEntry(
        input_shape=TypeShape.TABULAR,
        output_shape=TypeShape.SCALAR,
        streaming_capable=False,
        domain='geospatial_analysis',
    ),
    "analyze_accessibility": ShapeEntry(
        input_shape=TypeShape.TABULAR,
        output_shape=TypeShape.SCALAR,
        streaming_capable=False,
        domain='geospatial_analysis',
    ),
    "generate_service_isochrones": ShapeEntry(
        input_shape=TypeShape.TABULAR,
        output_shape=TypeShape.GEO,
        streaming_capable=False,
        domain='geospatial_analysis',
    ),
    "prepare_missing_values": ShapeEntry(
        input_shape=TypeShape.TABULAR,
        output_shape=TypeShape.TABULAR,
        streaming_capable=False,
        domain='preprocessing',
    ),
    "convert_types": ShapeEntry(
        input_shape=TypeShape.TABULAR,
        output_shape=TypeShape.TABULAR,
        streaming_capable=False,
        domain='preprocessing',
    ),
    "compose_pipeline": ShapeEntry(
        input_shape=TypeShape.DYNAMIC,
        output_shape=TypeShape.DYNAMIC,
        streaming_capable=False,
        domain='composition',
    ),
    "clean_then_profile": ShapeEntry(
        input_shape=TypeShape.DYNAMIC,
        output_shape=TypeShape.DYNAMIC,
        streaming_capable=False,
        domain='composition',
    ),
    "clean_then_regress": ShapeEntry(
        input_shape=TypeShape.DYNAMIC,
        output_shape=TypeShape.DYNAMIC,
        streaming_capable=False,
        domain='composition',
    ),
}
