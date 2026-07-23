"""localdata_mcp/visualize/render/backend.py — the matplotlib OO renderer (E12.2).

The one render path (§6c, FR-501/502): a ChartSpec in, raw image bytes
out. Uses the object-oriented `Figure` + `FigureCanvasSVG`/
`FigureCanvasAgg` API exclusively — never `pyplot` global state, so no
figure is retained in a global registry (the leak-class the NFR-204
loop guards against) — and disposes every Figure explicitly in a
`finally`, whatever the outcome. SVG bytes are UNSANITIZED here: NX-8's
allow-list sanitizer (nexus/export/renderers/svg.py) is the sole owner
of inertness and runs downstream; PNG is inert by construction. The
per-kind drawing lives in marks.py; this module owns only the figure
lifecycle and the format export. Neighbors: charts/ builds the
ChartSpec; tools.py hands the bytes to NX-8; marks.py draws.
"""

from __future__ import annotations

from io import BytesIO

import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_svg import FigureCanvasSVG
from matplotlib.figure import Figure

from localdata_mcp.visualize.charts import ChartSpec

from .marks import DRAWERS

_SUPPORTED_FORMATS = ("svg", "png")


def render_spec(spec: ChartSpec, image_format: str) -> bytes:
    """Raw `image_format` bytes for `spec` — the Figure is drawn, the
    kind's data marks are gid-tagged (marks.py), the title is set, and
    the Figure is disposed before returning."""
    if image_format not in _SUPPORTED_FORMATS:
        raise ValueError(
            f"unsupported image format {image_format!r} — supported: "
            f"{list(_SUPPORTED_FORMATS)}"
        )
    drawer = DRAWERS[spec.kind]
    figure = Figure(figsize=(6.4, 4.8), dpi=100)
    try:
        axes = figure.subplots()
        drawer(axes, spec.data)
        if spec.title:
            axes.set_title(spec.title)
        figure.tight_layout()
        return _export(figure, image_format)
    finally:
        figure.clear()


def _export(figure: Figure, image_format: str) -> bytes:
    """The figure as bytes via the format's OO canvas (no pyplot)."""
    buffer = BytesIO()
    if image_format == "svg":
        # svg.fonttype='none' keeps text as escaped <text> nodes (not
        # glyph paths), so matplotlib's XML-escaping of titles/labels/
        # data-derived values is FR-501's first defense layer and the
        # text is structurally present. rc_context scopes the setting —
        # no global pyplot state leaks (§6c). PNG is unaffected.
        with matplotlib.rc_context({"svg.fonttype": "none"}):
            FigureCanvasSVG(figure).print_svg(buffer)
    else:
        FigureCanvasAgg(figure).print_png(buffer)
    return buffer.getvalue()
