"""tests/v3/test_chart_spec.py — CHART_SPEC construction (E12.1, FR-503).

The declarative chart layer in isolation: every launch kind builds a
ChartSpec with the right mark token and extracted data from an
addressed frame + encoding, and every refusal path (unknown kind,
missing channel, absent column, non-numeric axis) returns the
structured NX-3 shape, never a bare error. No matplotlib here — this is
the "what chart" layer, tested independent of "how drawn".
"""

from __future__ import annotations

import pandas as pd
import pytest

from localdata_mcp.nexus.chokepoint.guard import GuardedExecutionError
from localdata_mcp.visualize.charts import CHART_KINDS, build_chart_spec


@pytest.fixture
def frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
            "y": [2.0, 4.1, 5.9, 8.2, 9.8],
            "z": [5.0, 3.0, 1.0, 4.0, 2.0],
            "group": ["a", "b", "a", "b", "a"],
            "lon": [10.0, 11.0, 12.0, 13.0, 14.0],
            "lat": [40.0, 41.0, 42.0, 43.0, 44.0],
            "src": ["n1", "n2", "n3", "n1", "n2"],
            "dst": ["n2", "n3", "n1", "n3", "n1"],
        }
    )


class TestKindRegistry:
    def test_six_launch_kinds(self) -> None:
        assert set(CHART_KINDS) == {
            "histogram",
            "heatmap",
            "scatter_fit",
            "line_timeseries",
            "geo_map",
            "network_layout",
        }

    def test_every_kind_carries_a_mark_token(self) -> None:
        assert all(kind.mark for kind in CHART_KINDS.values())


class TestHistogram:
    def test_extracts_values(self, frame: pd.DataFrame) -> None:
        spec = build_chart_spec("histogram", frame, {"value": "x"}, "Dist")
        assert spec.kind == "histogram"
        assert spec.mark == "bars"
        assert spec.title == "Dist"
        assert spec.data["values"] == [1.0, 2.0, 3.0, 4.0, 5.0]
        assert spec.data["label"] == "x"

    def test_missing_value_channel_refused(self, frame: pd.DataFrame) -> None:
        with pytest.raises(GuardedExecutionError):
            build_chart_spec("histogram", frame, {}, None)

    def test_non_numeric_column_refused(self, frame: pd.DataFrame) -> None:
        with pytest.raises(GuardedExecutionError):
            build_chart_spec("histogram", frame, {"value": "group"}, None)

    def test_absent_column_refused(self, frame: pd.DataFrame) -> None:
        with pytest.raises(GuardedExecutionError):
            build_chart_spec("histogram", frame, {"value": "nope"}, None)


class TestHeatmap:
    def test_correlation_matrix_over_numeric(self, frame: pd.DataFrame) -> None:
        spec = build_chart_spec("heatmap", frame, {"columns": ["x", "y", "z"]}, None)
        assert spec.mark == "cells"
        assert spec.data["labels"] == ["x", "y", "z"]
        matrix = spec.data["matrix"]
        assert len(matrix) == 3 and len(matrix[0]) == 3
        # diagonal is self-correlation 1.0
        assert matrix[0][0] == pytest.approx(1.0)
        # x and y are near-perfectly correlated
        assert matrix[0][1] == pytest.approx(1.0, abs=0.01)

    def test_default_uses_all_numeric_columns(self, frame: pd.DataFrame) -> None:
        spec = build_chart_spec("heatmap", frame, None, None)
        assert set(spec.data["labels"]) == {"x", "y", "z", "lon", "lat"}

    def test_single_numeric_column_refused(self) -> None:
        one = pd.DataFrame({"only": [1.0, 2.0], "label": ["a", "b"]})
        with pytest.raises(GuardedExecutionError):
            build_chart_spec("heatmap", one, None, None)


class TestScatterFit:
    def test_points_and_fit_line(self, frame: pd.DataFrame) -> None:
        spec = build_chart_spec("scatter_fit", frame, {"x": "x", "y": "y"}, None)
        assert spec.mark == "points"
        assert spec.data["x"] == [1.0, 2.0, 3.0, 4.0, 5.0]
        assert spec.data["fit"]["slope"] == pytest.approx(1.96, abs=0.05)
        assert spec.data["fit"]["intercept"] is not None

    def test_vertical_x_yields_none_fit(self) -> None:
        flat = pd.DataFrame({"x": [3.0, 3.0, 3.0], "y": [1.0, 2.0, 3.0]})
        spec = build_chart_spec("scatter_fit", flat, {"x": "x", "y": "y"}, None)
        assert spec.data["fit"]["slope"] is None
        assert spec.data["fit"]["intercept"] is None

    def test_auto_axes_when_encoding_omitted(self) -> None:
        # progressive disclosure: no x/y -> first two numeric columns
        auto = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [2.0, 4.0, 6.0]})
        spec = build_chart_spec("scatter_fit", auto, {}, None)
        assert spec.data["x_label"] == "a"
        assert spec.data["y_label"] == "b"

    def test_color_channel_carries_categories(self) -> None:
        clustered = pd.DataFrame(
            {
                "a": [1.0, 2.0, 3.0, 4.0],
                "b": [1.0, 2.0, 1.0, 2.0],
                "cluster": [0, 1, 0, 1],
            }
        )
        spec = build_chart_spec("scatter_fit", clustered, {"color": "cluster"}, None)
        # x/y auto-selected from the non-colour numeric columns
        assert spec.data["x_label"] == "a"
        assert spec.data["y_label"] == "b"
        assert spec.data["color"] == [0, 1, 0, 1]
        assert spec.data["color_label"] == "cluster"

    def test_too_few_numeric_columns_refused(self) -> None:
        one = pd.DataFrame({"a": [1.0, 2.0], "label": ["x", "y"]})
        with pytest.raises(GuardedExecutionError):
            build_chart_spec("scatter_fit", one, {}, None)


class TestLineTimeseries:
    def test_orders_by_x(self) -> None:
        unordered = pd.DataFrame({"t": [3, 1, 2], "v": [30.0, 10.0, 20.0]})
        spec = build_chart_spec(
            "line_timeseries", unordered, {"x": "t", "y": "v"}, None
        )
        assert spec.mark == "line"
        assert spec.data["x"] == [1.0, 2.0, 3.0]
        assert spec.data["y"] == [10.0, 20.0, 30.0]
        assert spec.data["x_is_time"] is False

    def test_datetime_axis_detected_and_iso(self) -> None:
        ts = pd.DataFrame(
            {
                "when": ["2020-01-03", "2020-01-01", "2020-01-02"],
                "v": [3.0, 1.0, 2.0],
            }
        )
        spec = build_chart_spec("line_timeseries", ts, {"x": "when", "y": "v"}, None)
        assert spec.data["x_is_time"] is True
        assert spec.data["x"][0].startswith("2020-01-01")
        assert spec.data["y"] == [1.0, 2.0, 3.0]


class TestGeoMap:
    def test_lon_lat_points(self, frame: pd.DataFrame) -> None:
        spec = build_chart_spec("geo_map", frame, {"lon": "lon", "lat": "lat"}, None)
        assert spec.mark == "points"
        assert spec.data["lon"] == [10.0, 11.0, 12.0, 13.0, 14.0]
        assert "value" not in spec.data

    def test_optional_value_channel(self, frame: pd.DataFrame) -> None:
        spec = build_chart_spec(
            "geo_map", frame, {"lon": "lon", "lat": "lat", "value": "z"}, None
        )
        assert spec.data["value"] == [5.0, 3.0, 1.0, 4.0, 2.0]

    def test_missing_lat_refused(self, frame: pd.DataFrame) -> None:
        with pytest.raises(GuardedExecutionError):
            build_chart_spec("geo_map", frame, {"lon": "lon"}, None)


class TestNetworkLayout:
    def test_nodes_edges_deterministic_layout(self, frame: pd.DataFrame) -> None:
        spec = build_chart_spec(
            "network_layout", frame, {"source": "src", "target": "dst"}, None
        )
        assert spec.mark == "edges"
        assert spec.data["nodes"] == ["n1", "n2", "n3"]
        # 5 edges, indices into the node list
        assert len(spec.data["edges"]) == 5
        assert spec.data["edges"][0] == [0, 1]  # n1->n2
        # circular layout: 3 nodes on the unit circle, first at (1,0)
        positions = spec.data["positions"]
        assert len(positions) == 3
        assert positions[0] == [1.0, 0.0]

    def test_layout_is_reproducible(self, frame: pd.DataFrame) -> None:
        first = build_chart_spec(
            "network_layout", frame, {"source": "src", "target": "dst"}, None
        )
        second = build_chart_spec(
            "network_layout", frame, {"source": "src", "target": "dst"}, None
        )
        assert first.data["positions"] == second.data["positions"]

    def test_missing_target_refused(self, frame: pd.DataFrame) -> None:
        with pytest.raises(GuardedExecutionError):
            build_chart_spec("network_layout", frame, {"source": "src"}, None)


class TestUnknownKind:
    def test_unknown_kind_refused(self, frame: pd.DataFrame) -> None:
        with pytest.raises(GuardedExecutionError):
            build_chart_spec("piechart", frame, {"value": "x"}, None)


class TestSpecToDict:
    def test_round_trips_to_plain_dict(self, frame: pd.DataFrame) -> None:
        spec = build_chart_spec("histogram", frame, {"value": "x"}, "T")
        wire = spec.to_dict()
        assert wire["kind"] == "histogram"
        assert wire["mark"] == "bars"
        assert wire["title"] == "T"
        assert wire["encoding"] == {"value": "x"}
        assert isinstance(wire["data"], dict)
