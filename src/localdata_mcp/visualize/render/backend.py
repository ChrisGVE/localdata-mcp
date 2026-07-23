"""localdata_mcp/visualize/render/backend.py — the matplotlib OO renderer (E12.2).

The one render path (§6c, FR-501/502/503-styling): a ChartSpec plus a
StyleSpec in, raw image bytes out. Uses the object-oriented `Figure` +
`FigureCanvasSVG`/`FigureCanvasAgg` API exclusively — never `pyplot`
global state, so no figure is retained in a global registry (the
leak-class the NFR-204 loop guards against) — and disposes every Figure
explicitly in a `finally`, whatever the outcome. All styling (the
palette color cycle, the grid, and the SVG determinism keys) is applied
through a single `matplotlib.rc_context` scoped to this render — never
seaborn's global `set_theme` — so the no-global-state and
byte-reproducibility invariants hold. SVG bytes are UNSANITIZED here:
NX-8's allow-list sanitizer (nexus/export/renderers/svg.py) is the sole
owner of inertness and runs downstream; PNG is inert by construction.
The per-kind drawing lives in marks.py; this module owns only the
figure lifecycle, the frame treatment, and the format export.
Neighbors: charts/ builds the ChartSpec; style/ builds the StyleSpec;
tools.py hands the bytes to NX-8; marks.py draws.
"""

from __future__ import annotations

from io import BytesIO
from typing import Any, cast

import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_svg import FigureCanvasSVG
from matplotlib.figure import Figure

from localdata_mcp.visualize.charts import ChartSpec
from localdata_mcp.visualize.style import StyleSpec

from .marks import DRAWERS

_SUPPORTED_FORMATS = ("svg", "png")

# SVG determinism, always applied (inert for PNG): svg.fonttype='none'
# keeps text as escaped <text> nodes (not glyph paths), so matplotlib's
# XML-escaping of titles/labels/data values is FR-501's first defense
# layer AND the text is structurally present; svg.hashsalt fixes the
# otherwise per-render element-id hashes so the SVG is byte-reproducible
# (the C-2 wrapper equality acceptance, and stable goldens).
_SVG_DETERMINISM = {"svg.fonttype": "none", "svg.hashsalt": "localdata"}


def render_spec(spec: ChartSpec, image_format: str, style: StyleSpec) -> bytes:
    """Raw `image_format` bytes for `spec` drawn under `style` — the
    palette and grid are applied via a scoped rc_context, the kind's
    data marks are gid-tagged (marks.py), the title is set, and the
    Figure is disposed before returning."""
    if image_format not in _SUPPORTED_FORMATS:
        raise ValueError(
            f"unsupported image format {image_format!r} — supported: "
            f"{list(_SUPPORTED_FORMATS)}"
        )
    drawer = DRAWERS[spec.kind]
    # A dynamically-merged str-keyed dict widens past matplotlib's
    # Literal-keyed rcParams stub; every key is a valid rcParam, so
    # bridge the third-party over-constraint rather than silence it.
    rc = {**style.rc_overrides(), **_SVG_DETERMINISM}
    with matplotlib.rc_context(cast(Any, rc)):
        figure = Figure(figsize=style.figsize, dpi=style.dpi)
        try:
            axes = figure.subplots()
            drawer(axes, spec.data, style)
            if style.despine:
                _despine(axes)
            if spec.title:
                axes.set_title(spec.title)
            figure.tight_layout()
            return _export(figure, image_format)
        finally:
            figure.clear()


def _despine(axes: "Any") -> None:
    """The Tufte frame treatment: drop the top/right spines — non-data
    ink. Safe for every kind (a no-op where the drawer turned the axis
    off)."""
    for side in ("top", "right"):
        spine = axes.spines.get(side)
        if spine is not None:
            spine.set_visible(False)


def _export(figure: Figure, image_format: str) -> bytes:
    """The figure as bytes via the format's OO canvas (no pyplot). The
    SVG determinism rcParams are already active in the caller's
    rc_context; `metadata={'Date': None}` drops matplotlib's creation
    timestamp so the raw SVG is byte-reproducible (not just the
    NX-8-sanitized form) — completing the determinism guarantee the C-2
    equality and the goldens rely on."""
    buffer = BytesIO()
    if image_format == "svg":
        FigureCanvasSVG(figure).print_svg(buffer, metadata={"Date": None})
    else:
        FigureCanvasAgg(figure).print_png(buffer)
    return buffer.getvalue()
