"""MACHINE-WRITTEN by localdata_mcp.nexus.contract.generators.test_stub — DO NOT EDIT.

Parametrized L3 contract stubs (ARCHITECTURE.md 6.1 artifact 4): every
served tool answers a well-formed fastmcp.Client call on the served
app, every test_only walking-skeleton probe answers on a battery-local
app kept OFF the served surface (CR-012), and the coverage check fails
if any registered ToolSpec lacks an entry in either list. Regenerate
via `python -m localdata_mcp.nexus.contract.generate`; hand edits fail
CI through nexus/contract/check_drift.py.
"""

from __future__ import annotations

import json
from typing import Any

import anyio
import pytest
from fastmcp import Client, FastMCP

from localdata_mcp.nexus.contract.registry import default_registry
from localdata_mcp.nexus.contract.spec_modules import load_spec_modules
from localdata_mcp.server.mcp_app import app
from localdata_mcp.server.tools_generated import register_skeleton_tools

# The served product surface: exercised against the app mcp_app.py boots.
SERVED_TOOL_CALLS: "tuple[tuple[str, dict[str, Any]], ...]" = (
    ("list_endpoints", {}),
    ("fetch_chunk", {"stream_id": "probe"}),
    ("close_stream", {"stream_id": "probe"}),
    ("query", {"endpoint": "probe", "sql": "probe"}),
    ("write_query", {"endpoint": "probe", "sql": "probe"}),
    ("read_file", {"path": "probe"}),
    ("query_file", {"path": "probe", "sql": "probe"}),
    ("get_value", {"endpoint": "probe", "path": "probe", "key": "probe"}),
    ("set_value", {"endpoint": "probe", "path": "probe", "key": "probe", "value": "probe"}),
    ("delete_key", {"endpoint": "probe", "path": "probe", "key": "probe"}),
    ("list_keys", {"endpoint": "probe", "path": "probe"}),
    ("get_node", {"endpoint": "probe"}),
    ("set_node", {"endpoint": "probe", "path": "probe"}),
    ("delete_node", {"endpoint": "probe", "path": "probe"}),
    ("get_children", {"endpoint": "probe"}),
    ("move_node", {"endpoint": "probe", "path": "probe"}),
    ("get_neighbors", {"endpoint": "probe", "node_id": "probe"}),
    ("get_edges", {"endpoint": "probe"}),
    ("add_edge", {"endpoint": "probe", "source": "probe", "target": "probe"}),
    ("remove_edge", {"endpoint": "probe", "source": "probe", "target": "probe"}),
    ("find_path", {"endpoint": "probe", "source": "probe", "target": "probe"}),
    ("get_graph_stats", {"endpoint": "probe"}),
    ("describe_database", {"endpoint": "probe"}),
    ("describe_table", {"endpoint": "probe", "table": "probe"}),
    ("find_table", {"endpoint": "probe", "name_pattern": "probe"}),
    ("profile_data", {}),
    ("search_data", {"query": "probe"}),
    ("map_categories", {"column": "probe"}),
    ("analyze_hypothesis_test", {}),
    ("analyze_anova", {"dependent_var": "probe", "group_var": "probe"}),
    ("analyze_effect_sizes", {"column": "probe", "group_column": "probe"}),
    ("analyze_ab_test", {"metric_column": "probe", "variant_column": "probe"}),
    ("analyze_regression", {"target_column": "probe"}),
    ("evaluate_model_performance", {"target_column": "probe", "prediction_column": "probe"}),
    ("analyze_clusters", {}),
    ("assign_clusters", {}),
    ("detect_anomalies", {}),
    ("reduce_dimensions", {}),
    ("transform_data", {"column": "probe", "find": "probe", "replace": "probe"}),
    ("analyze_time_series", {"date_column": "probe", "value_column": "probe"}),
    ("forecast_time_series", {"date_column": "probe", "value_column": "probe"}),
    ("generate_sample", {}),
    ("bootstrap_statistic", {"column": "probe"}),
    ("monte_carlo_simulate", {"column": "probe"}),
    ("bayesian_estimate", {"column": "probe"}),
    ("analyze_rfm", {"customer_column": "probe", "date_column": "probe", "value_column": "probe"}),
    ("calculate_clv", {"customer_column": "probe", "date_column": "probe", "value_column": "probe"}),
    ("analyze_network", {"source_column": "probe", "target_column": "probe"}),
    ("solve_linear_program", {"objective_column": "probe"}),
    ("optimize_constrained", {"objective_expression": "probe", "initial_guess_column": "probe"}),
    ("solve_assignment_problem", {"cost_columns": ["probe"]}),
    ("check_geospatial_capabilities", {}),
    ("analyze_spatial_autocorrelation", {"value_column": "probe"}),
    ("find_spatial_hotspots", {"value_column": "probe"}),
    ("calculate_spatial_distances", {}),
    ("perform_spatial_join", {"geometry_column": "probe", "right_geometries": ["probe"]}),
    ("perform_spatial_overlay", {"geometry_column": "probe", "right_geometries": ["probe"]}),
    ("aggregate_points_in_polygons", {"value_column": "probe", "polygons": ["probe"]}),
    ("optimize_route", {"edges": ["probe"], "waypoints": ["probe"]}),
    ("analyze_accessibility", {"edges": ["probe"], "service_locations": ["probe"], "demand_locations": ["probe"]}),
    ("generate_service_isochrones", {"edges": ["probe"], "service_locations": ["probe"], "time_bands": ["probe"]}),
    ("prepare_missing_values", {}),
    ("convert_types", {"conversions": {}}),
    ("compose_pipeline", {"dag_spec": ["probe"]}),
    ("clean_then_profile", {}),
    ("clean_then_regress", {"target": "probe"}),
    ("cluster_then_chart", {}),
    ("render_chart", {"kind": "probe"}),
    ("export_result", {"format": "probe", "path": "probe"}),
)

# The test_only walking-skeleton probes: exercised against a battery-
# local app built here, so they never reach the served surface (CR-012)
# yet stay L3-proven at the seam (GP5).
SKELETON_TOOL_CALLS: "tuple[tuple[str, dict[str, Any]], ...]" = (
    ("ping", {}),
    ("probe_table", {"rows": 3}),
    ("probe_vector", {"length": 3}),
    ("probe_matrix", {"size": 3}),
    ("probe_model", {"points": 3}),
    ("probe_graph", {"nodes": 3}),
    ("probe_geo", {"points": 3}),
    ("probe_chart", {"points": 3}),
    ("probe_sink", {"text": "probe"}),
)

_skeleton_app = FastMCP("localdata-skeleton-probes")
register_skeleton_tools(_skeleton_app)

_ENVELOPE_REGIONS = {"inline", "data", "composition_metadata", "error"}


def _envelope_of(result: Any) -> "dict[str, Any]":
    """The wire envelope from a client result: structured content when
    the transport carries it, else the JSON text block."""
    if isinstance(result.structured_content, dict) and (
        set(result.structured_content) >= _ENVELOPE_REGIONS
    ):
        return result.structured_content
    payload = json.loads(result.content[0].text)
    assert isinstance(payload, dict)
    return payload


def _assert_well_formed(target: Any, name: str, arguments: "dict[str, Any]") -> None:
    """Call `name` on `target` and assert the FR-403 four-region
    envelope, with error exclusive of the other regions."""

    async def session() -> None:
        async with Client(target) as client:
            result = await client.call_tool(name, arguments)
            assert not result.is_error
            envelope = _envelope_of(result)
            assert set(envelope) >= _ENVELOPE_REGIONS
            if envelope["error"] is None:
                assert envelope["inline"] is not None
            else:
                assert envelope["inline"] is None
                assert envelope["data"] is None
                assert envelope["composition_metadata"] is None

    anyio.run(session)


@pytest.mark.parametrize(
    ("name", "arguments"),
    SERVED_TOOL_CALLS,
    ids=[name for name, _ in SERVED_TOOL_CALLS],
)
def test_served_tool_answers_well_formed(
    name: str, arguments: "dict[str, Any]"
) -> None:
    _assert_well_formed(app, name, arguments)


@pytest.mark.parametrize(
    ("name", "arguments"),
    SKELETON_TOOL_CALLS,
    ids=[name for name, _ in SKELETON_TOOL_CALLS],
)
def test_skeleton_probe_answers_well_formed(
    name: str, arguments: "dict[str, Any]"
) -> None:
    _assert_well_formed(_skeleton_app, name, arguments)


def test_every_registered_spec_has_a_generated_entry() -> None:
    load_spec_modules()
    registered = {spec.name for spec in default_registry()}
    generated = {name for name, _ in SERVED_TOOL_CALLS} | {
        name for name, _ in SKELETON_TOOL_CALLS
    }
    missing = registered - generated
    assert not missing, (
        f"registered ToolSpecs lacking generated L3 entries: {sorted(missing)}"
    )
