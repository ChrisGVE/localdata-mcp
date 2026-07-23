"""tests/v3/test_chart_render.py — the matplotlib OO renderer (E12.2).

The renderer in isolation: every kind's ChartSpec renders to SVG (with
its gid-tagged data marks) and to PNG (valid signature), an unsupported
format is refused, and the rendered SVG survives NX-8 sanitization with
its marks intact and no active content. Figure disposal is covered by
the NFR-204 leak loop (E12.4), not asserted per-render here.
"""

from __future__ import annotations

import pandas as pd
import pytest

from localdata_mcp.nexus.export.interface import render as export_render
from localdata_mcp.visualize.charts import CHART_KINDS, build_chart_spec
from localdata_mcp.visualize.render import render_spec

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


@pytest.fixture
def frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
            "y": [2.0, 4.1, 5.9, 8.2, 9.8],
            "z": [5.0, 3.0, 1.0, 4.0, 2.0],
            "lon": [10.0, 11.0, 12.0, 13.0, 14.0],
            "lat": [40.0, 41.0, 42.0, 43.0, 44.0],
            "src": ["a", "b", "c", "a", "b"],
            "dst": ["b", "c", "a", "c", "a"],
        }
    )


# (kind, encoding, the gid the renderer must emit for this kind's marks)
_KIND_CASES = [
    ("histogram", {"value": "x"}, "localdata-bars"),
    ("heatmap", {"columns": ["x", "y", "z"]}, "localdata-cells"),
    ("scatter_fit", {"x": "x", "y": "y"}, "localdata-points"),
    ("line_timeseries", {"x": "x", "y": "y"}, "localdata-line"),
    ("geo_map", {"lon": "lon", "lat": "lat"}, "localdata-points"),
    ("network_layout", {"source": "src", "target": "dst"}, "localdata-edges"),
]


class TestCoverageMatchesRegistry:
    def test_every_registered_kind_has_a_render_case(self) -> None:
        assert {case[0] for case in _KIND_CASES} == set(CHART_KINDS)


class TestSvgRender:
    @pytest.mark.parametrize("kind,encoding,gid", _KIND_CASES)
    def test_svg_carries_gid_marks(
        self, frame: pd.DataFrame, kind: str, encoding: dict, gid: str
    ) -> None:
        spec = build_chart_spec(kind, frame, encoding, f"T-{kind}")
        svg = render_spec(spec, "svg").decode("utf-8")
        assert svg.lstrip().startswith("<?xml") or svg.lstrip().startswith("<svg")
        assert f'id="{gid}"' in svg

    @pytest.mark.parametrize("kind,encoding,gid", _KIND_CASES)
    def test_svg_survives_nx8_sanitization_with_marks(
        self, frame: pd.DataFrame, kind: str, encoding: dict, gid: str
    ) -> None:
        spec = build_chart_spec(kind, frame, encoding, None)
        clean = export_render(render_spec(spec, "svg"), "svg").decode("utf-8")
        assert f'id="{gid}"' in clean  # marks survive the allow-list
        assert "<image" not in clean  # no embedded raster (imshow trap)
        assert "<script" not in clean


class TestPngRender:
    @pytest.mark.parametrize("kind,encoding,gid", _KIND_CASES)
    def test_png_signature(
        self, frame: pd.DataFrame, kind: str, encoding: dict, gid: str
    ) -> None:
        spec = build_chart_spec(kind, frame, encoding, None)
        png = render_spec(spec, "png")
        assert png.startswith(_PNG_SIGNATURE)
        # NX-8 png renderer accepts it (signature validation passes)
        assert export_render(png, "png") == png


class TestFormatRefusal:
    def test_unsupported_format_refused(self, frame: pd.DataFrame) -> None:
        spec = build_chart_spec("histogram", frame, {"value": "x"}, None)
        with pytest.raises(ValueError):
            render_spec(spec, "pdf")
