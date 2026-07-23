"""testbench/batteries/domain/geospatial_battery_test.py — E10.e slice.

The NFR-502c domain battery's geospatial rows (extras tier). The
capability probe row runs UNCONDITIONALLY — it must answer whether the
extra is installed. Every other row is gated on the geospatial stack
being importable (the pg/mysql-row precedent: SKIP with a named reason
when the opt-in extra is absent, so a base install stays green). The
oracle is the closed-form geometry: a checkerboard high/low field has
a strongly negative Moran's I; a corner-clustered field is positive;
the assignment/route/overlay answers are hand-derivable from the
fixture.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from typing import Any, Iterator

import anyio
import pandas as pd
import pytest
from fastmcp import Client

import localdata_mcp.ingest.runtime as runtime
from localdata_mcp.nexus.chokepoint.guard import Chokepoint
from localdata_mcp.nexus.config.models import ConfigModel, SecurityConfig
from localdata_mcp.nexus.contract.registry import default_registry
from localdata_mcp.nexus.response.shaping import configure_shaping
from localdata_mcp.server.mcp_app import app

_HAS_GEOSTACK = all(
    importlib.util.find_spec(name) is not None
    for name in ("geopandas", "shapely", "pyproj")
)
_geo = pytest.mark.skipif(not _HAS_GEOSTACK, reason="geospatial extra not installed")


@pytest.fixture()
def bench(tmp_path: Path) -> Iterator[Path]:
    config = ConfigModel(
        security=SecurityConfig(allowed_paths=(str(tmp_path),)),
    )
    guard = Chokepoint.boot(config, environ=dict(os.environ))
    configure_shaping(config, default_registry())
    runtime.configure_ingest(guard)
    yield tmp_path
    runtime._CHOKEPOINT = None
    configure_shaping(ConfigModel(), default_registry())
    guard.shutdown()


def _call(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    async def session() -> dict[str, Any]:
        async with Client(app) as client:
            result = await client.call_tool(name, arguments)
            assert not result.is_error
            if isinstance(result.structured_content, dict) and (
                "inline" in result.structured_content
            ):
                return result.structured_content
            payload = json.loads(result.content[0].text)
            assert isinstance(payload, dict)
            return payload

    return anyio.run(session)


def _data(envelope: dict[str, Any]) -> Any:
    assert envelope["error"] is None, envelope["error"]
    return envelope["data"]


def test_capability_probe_answers_unconditionally() -> None:
    """check_geospatial_capabilities always answers (its whole point)."""
    data = _data(_call("check_geospatial_capabilities", {}))
    assert "available_libraries" in data
    assert "has_core_geospatial" in data
    assert data["has_core_geospatial"] is _HAS_GEOSTACK


def _grid_csv(tmp_path: Path, checkerboard: bool) -> str:
    rows = []
    for x in range(6):
        for y in range(6):
            if checkerboard:
                value = 1.0 if (x + y) % 2 == 0 else 0.0
            else:
                value = 1.0 if (x < 3 and y < 3) else 0.0
            rows.append({"x": float(x), "y": float(y), "z": value})
    target = tmp_path / ("checker.csv" if checkerboard else "clustered.csv")
    pd.DataFrame(rows).to_csv(target, index=False)
    return str(target)


@_geo
def test_morans_i_sign_tracks_the_pattern(bench: Path) -> None:
    """analyze_spatial_autocorrelation — closed-form oracle on the
    sign: checkerboard disperses (I<0), corner-cluster clusters (I>0)."""
    checker = _data(
        _call(
            "analyze_spatial_autocorrelation",
            {"path": _grid_csv(bench, True), "value_column": "z", "k_neighbors": 4},
        )
    )
    assert checker["morans_i"] < 0
    clustered = _data(
        _call(
            "analyze_spatial_autocorrelation",
            {"path": _grid_csv(bench, False), "value_column": "z", "k_neighbors": 4},
        )
    )
    assert clustered["morans_i"] > 0
    assert clustered["p_value"] < 0.05


@_geo
def test_hotspots_find_the_high_corner(bench: Path) -> None:
    """find_spatial_hotspots — the high corner is the hot cluster."""
    data = _data(
        _call(
            "find_spatial_hotspots",
            {"path": _grid_csv(bench, False), "value_column": "z"},
        )
    )
    assert data["n_hotspots"] >= 1
    assert data["method"] == "getis_ord_gi_star"


@_geo
def test_distances_summary_is_exact(bench: Path) -> None:
    """calculate_spatial_distances — closed-form min/max on a unit square."""
    target = bench / "square.csv"
    pd.DataFrame({"x": [0.0, 1.0, 0.0, 1.0], "y": [0.0, 0.0, 1.0, 1.0]}).to_csv(
        target, index=False
    )
    data = _data(_call("calculate_spatial_distances", {"path": str(target)}))
    config = ConfigModel()
    rtol = config.testbench.tol_closed_form_rtol
    assert data["n_pairs"] == 6
    assert data["distance_summary"]["min"] == pytest.approx(1.0, rel=rtol)
    assert data["distance_summary"]["max"] == pytest.approx(2.0**0.5, rel=rtol)


@_geo
def test_spatial_join_matches_the_containing_polygon(bench: Path) -> None:
    """perform_spatial_join — a point-geometry row lands in its box."""
    target = bench / "pts.csv"
    pd.DataFrame(
        {"label": ["a", "b"], "geometry": ["POINT (1 1)", "POINT (9 9)"]}
    ).to_csv(target, index=False)
    data = _data(
        _call(
            "perform_spatial_join",
            {
                "path": str(target),
                "geometry_column": "geometry",
                "right_geometries": ["POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))"],
            },
        )
    )
    assert data["match_count"] == 1
    assert data["matches"][0]["left_index"] == 0


@_geo
def test_overlay_intersection_area(bench: Path) -> None:
    """perform_spatial_overlay — two overlapping unit squares."""
    target = bench / "poly.csv"
    pd.DataFrame({"geometry": ["POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))"]}).to_csv(
        target, index=False
    )
    data = _data(
        _call(
            "perform_spatial_overlay",
            {
                "path": str(target),
                "geometry_column": "geometry",
                "right_geometries": ["POLYGON ((1 1, 3 1, 3 3, 1 3, 1 1))"],
            },
        )
    )
    assert data["output_count"] == 1
    from shapely import wkt

    assert wkt.loads(data["geometries"][0]).area == pytest.approx(1.0)


@_geo
def test_aggregate_points_in_polygons(bench: Path) -> None:
    """aggregate_points_in_polygons — mean/sum inside one box."""
    target = bench / "measure.csv"
    pd.DataFrame(
        {"x": [1.0, 1.5, 9.0], "y": [1.0, 1.5, 9.0], "v": [10.0, 20.0, 99.0]}
    ).to_csv(target, index=False)
    data = _data(
        _call(
            "aggregate_points_in_polygons",
            {
                "path": str(target),
                "value_column": "v",
                "polygons": ["POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))"],
            },
        )
    )
    per_polygon = data["per_polygon"]["0"]
    assert per_polygon["count"] == 2
    assert per_polygon["mean"] == pytest.approx(15.0)


@_geo
def test_route_orders_and_connects_waypoints(bench: Path) -> None:
    """optimize_route — a 4-node path graph, greedy order end to end."""
    target = bench / "nodes.csv"
    pd.DataFrame(
        {"id": ["a", "b", "c", "d"], "x": [0.0, 1, 2, 3], "y": [0.0, 0, 0, 0]}
    ).to_csv(target, index=False)
    edges = [["a", "b", 1.0], ["b", "c", 1.0], ["c", "d", 1.0]]
    data = _data(
        _call(
            "optimize_route",
            {
                "path": str(target),
                "edges": edges,
                "waypoints": ["a", "d"],
            },
        )
    )
    assert data["route_path"] == ["a", "b", "c", "d"]
    assert data["total_distance"] == pytest.approx(3.0)


@_geo
def test_accessibility_scores_reachable_demand(bench: Path) -> None:
    """analyze_accessibility — nearest-service travel time per demand."""
    target = bench / "net.csv"
    pd.DataFrame({"id": ["s", "m", "d"], "x": [0.0, 1, 2], "y": [0.0, 0, 0]}).to_csv(
        target, index=False
    )
    data = _data(
        _call(
            "analyze_accessibility",
            {
                "path": str(target),
                "edges": [["s", "m", 1.0], ["m", "d", 1.0]],
                "service_locations": ["s"],
                "demand_locations": ["d"],
            },
        )
    )
    assert data["reachable_count"] == 1
    assert data["travel_times"]["d"] == pytest.approx(2.0)


@_geo
def test_isochrones_report_reachable_bands(bench: Path) -> None:
    """generate_service_isochrones — nested reachable node sets."""
    target = bench / "iso.csv"
    pd.DataFrame({"id": ["s", "a", "b"], "x": [0.0, 1, 2], "y": [0.0, 0, 0]}).to_csv(
        target, index=False
    )
    data = _data(
        _call(
            "generate_service_isochrones",
            {
                "path": str(target),
                "edges": [["s", "a", 1.0], ["a", "b", 1.0]],
                "service_locations": ["s"],
                "time_bands": [1.0, 2.0],
            },
        )
    )
    bands = {
        entry["time_band"]: entry["reachable_count"] for entry in data["isochrones"]
    }
    assert bands[1.0] == 2  # s and a
    assert bands[2.0] == 3  # s, a, b
