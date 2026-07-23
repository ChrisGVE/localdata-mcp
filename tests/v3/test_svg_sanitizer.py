"""tests/v3/test_svg_sanitizer.py — E7.4 FR-501 inertness battery.

The hostile-fixture tests the E7 exit gate names: active content
(script, event handlers, foreignObject, animate) dropped by the
allow-list; reference-class values (href, clip-path, style url())
refused when non-local; a REAL matplotlib-Agg-rendered chart passes
through structurally intact (the allow-list matches what the backend
legitimately emits — a sanitizer that mangles its own renderer's
output would be worse than none). Plus the PNG signature validation
on the raster half.
"""

from __future__ import annotations

import io

import pytest

from localdata_mcp.nexus.export.interface import ExportError, render
from localdata_mcp.nexus.export.renderers.svg import SvgRefusedError

_SVG_OPEN = '<svg xmlns="http://www.w3.org/2000/svg" '
_XLINK = 'xmlns:xlink="http://www.w3.org/1999/xlink" '


def sanitize(text: str) -> str:
    return render(text, "svg").decode("utf-8")


class TestActiveContentDropped:
    def test_script_element_and_subtree_dropped(self) -> None:
        hostile = (
            _SVG_OPEN + 'width="10" height="10">'
            '<script>alert("owned")</script>'
            '<rect x="0" y="0" width="5" height="5"/>'
            "</svg>"
        )
        clean = sanitize(hostile)
        assert "script" not in clean
        assert "alert" not in clean
        assert "rect" in clean

    def test_event_handler_attributes_dropped(self) -> None:
        hostile = (
            _SVG_OPEN + ">"
            '<rect onclick="alert(\'x\')" onload="evil()" width="5"/>'
            "</svg>"
        )
        clean = sanitize(hostile)
        assert "onclick" not in clean and "onload" not in clean
        assert 'width="5"' in clean

    def test_foreignobject_and_animate_dropped(self) -> None:
        hostile = (
            _SVG_OPEN + ">"
            "<foreignObject><body>html</body></foreignObject>"
            '<animate attributeName="href" to="javascript:evil()"/>'
            "<g/></svg>"
        )
        clean = sanitize(hostile)
        assert "foreignObject" not in clean
        assert "animate" not in clean

    def test_style_element_dropped_entirely(self) -> None:
        """<style> url() imports are part of the deny-by-omission set —
        the element is not on matplotlib's legitimate list."""
        hostile = (
            _SVG_OPEN + "><style>@import url(https://evil/x.css);</style><g/></svg>"
        )
        clean = sanitize(hostile)
        assert "import" not in clean


class TestReferenceValueRules:
    def test_external_href_refuses_the_artifact(self) -> None:
        hostile = (
            _SVG_OPEN + _XLINK + '><use xlink:href="https://evil/defs.svg#p"/></svg>'
        )
        with pytest.raises(SvgRefusedError):
            sanitize(hostile)

    def test_local_href_is_kept(self) -> None:
        clean = sanitize(
            _SVG_OPEN + _XLINK + '><defs><path id="p1" d="M0 0"/></defs>'
            '<use xlink:href="#p1"/></svg>'
        )
        assert 'href="#p1"' in clean

    def test_javascript_uri_refuses(self) -> None:
        hostile = _SVG_OPEN + _XLINK + '><use xlink:href="javascript:x()"/></svg>'
        with pytest.raises(SvgRefusedError):
            sanitize(hostile)

    def test_style_with_url_reference_refuses(self) -> None:
        hostile = _SVG_OPEN + '><rect style="fill: url(#grad); x: 1"/></svg>'
        with pytest.raises(SvgRefusedError):
            sanitize(hostile)

    def test_plain_style_values_pass(self) -> None:
        clean = sanitize(
            _SVG_OPEN + '><rect style="fill: #ffffff; stroke: none" width="5"/></svg>'
        )
        assert "fill: #ffffff" in clean

    def test_clip_path_must_be_a_local_url_fragment(self) -> None:
        clean = sanitize(
            _SVG_OPEN + '><g clip-path="url(#pane)"><path d="M0 0"/></g></svg>'
        )
        assert 'clip-path="url(#pane)"' in clean
        with pytest.raises(SvgRefusedError):
            sanitize(_SVG_OPEN + '><g clip-path="url(https://evil/c.svg#x)"/></svg>')


class TestParseHardening:
    def test_entity_expansion_is_refused_by_defusedxml(self) -> None:
        bomb = (
            '<?xml version="1.0"?>'
            '<!DOCTYPE svg [<!ENTITY a "aaaa"><!ENTITY b "&a;&a;&a;">]>'
            + _SVG_OPEN
            + "><text>&b;</text></svg>"
        )
        with pytest.raises(ExportError):
            sanitize(bomb)

    def test_non_svg_root_refused(self) -> None:
        with pytest.raises(ExportError):
            sanitize("<html><body/></html>")

    def test_unparseable_payload_refused(self) -> None:
        with pytest.raises(ExportError):
            sanitize("not xml at all <<<")


class TestRealMatplotlibOutputSurvives:
    def test_agg_rendered_chart_passes_structurally_intact(self) -> None:
        """The allow-list is derived from what the Agg backend emits —
        prove a real Figure round-trips with its drawing intact."""
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        figure = Figure(figsize=(2, 2))
        FigureCanvasAgg(figure)
        axes = figure.add_subplot()
        axes.plot([0, 1, 2], [1, 0, 2])
        axes.set_title("t")
        buffer = io.StringIO()
        figure.savefig(buffer, format="svg")
        clean = sanitize(buffer.getvalue())
        # The drawing survived: paths, defs/use text glyphs, clip refs.
        assert "<" in clean and "path" in clean
        assert "use" in clean or "text" in clean
        assert "clip-path" in clean


class TestPngValidation:
    def test_real_png_bytes_pass_through_unmodified(self) -> None:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        figure = Figure(figsize=(1, 1))
        FigureCanvasAgg(figure)
        buffer = io.BytesIO()
        figure.savefig(buffer, format="png")
        payload = buffer.getvalue()
        assert render(payload, "png") == payload

    def test_non_png_bytes_refused(self) -> None:
        with pytest.raises(ExportError):
            render(b"GIF89a not a png", "png")

    def test_non_bytes_payload_refused(self) -> None:
        with pytest.raises(ExportError):
            render({"not": "bytes"}, "png")
