"""localdata_mcp/visualize/tools.py — the render_chart ToolSpec (E12.2).

Feature V-1/V-2 (FR-501…504): one tool that turns an addressed tabular
source into a chart artifact. The flow is four hand-offs, each owned by
its home: resolve the frame through the ONE addressing contract
(explore.addressing — so a composition stage's upstream output is
picked up automatically, FR-504's pipe leg), build the declarative
CHART_SPEC (charts/), render raw image bytes (render/), and hand those
bytes to NX-8 (nexus/export) which owns SVG inertness sanitization and
PNG validation. input_shape=TABULAR (chain-initial via its own
addressing, or fed by an upstream stage); output_shape=NONE — a chart
is a terminal sink, its artifact carried in the envelope for the export
surface (FR-402), never a further pipeline value. Neighbors:
charts/spec.py builds the spec; render/backend.py draws; spec_modules.py
rosters this module.
"""

from __future__ import annotations

import base64
from typing import Any

from localdata_mcp.explore.addressing import resolve_frame, source_params
from localdata_mcp.ingest.runtime import chokepoint
from localdata_mcp.nexus.contract.spec import Param, TypeShape, tool_spec
from localdata_mcp.nexus.export.interface import render as export_render

from .charts import build_chart_spec
from .render import render_spec
from .style import resolve_style

_MEDIA_TYPES = {"svg": "image/svg+xml", "png": "image/png"}


@tool_spec(
    name="render_chart",
    summary=(
        "Render an addressed tabular source as a chart image. kind is "
        "one of histogram, heatmap, scatter_fit, line_timeseries, "
        "geo_map, network_layout (one per analysis domain). encoding "
        "maps the kind's visual channels to columns "
        "(e.g. {'x': col, 'y': col}; heatmap defaults to all numeric "
        "columns). format is svg (default, sanitized to an inert "
        "document) or png. Styling is progressive: palette names a "
        "qualitative preset (deep, muted, pastel, bright, dark, "
        "colorblind — the default), colors supplies a custom color "
        "cycle instead, and style tunes figure size, dpi, grid, and the "
        "sequential colormap. The artifact is returned inline in the "
        "envelope, extractable through the Output surface."
    ),
    params=(
        *source_params(),
        Param(
            "kind",
            str,
            "Chart kind: histogram, heatmap, scatter_fit, "
            "line_timeseries, geo_map, or network_layout.",
        ),
        Param(
            "encoding",
            dict,
            "Visual-channel to column map for the kind "
            "(e.g. {'x': <col>, 'y': <col>}); heatmap accepts "
            "{'columns': [<col>, …]} or omits it for all numeric columns.",
            required=False,
        ),
        Param(
            "format",
            str,
            "Image format: svg (default, inert-sanitized) or png.",
            required=False,
        ),
        Param("title", str, "Chart title drawn above the plot.", required=False),
        Param(
            "palette",
            str,
            "Qualitative palette preset for categorical marks: deep, "
            "muted, pastel, bright, dark, or colorblind (the configured "
            "default). Ignored when colors is supplied.",
            required=False,
        ),
        Param(
            "colors",
            list,
            "Custom categorical color cycle (hex like '#1b9e77' or "
            "named matplotlib colors) — overrides palette when given.",
            required=False,
        ),
        Param(
            "style",
            dict,
            "Fine styling overrides: figure_width_inches, "
            "figure_height_inches, dpi, grid, despine, sequential_cmap "
            "(the continuous colormap), fit_color, edge_color.",
            required=False,
        ),
    ),
    input_shape=TypeShape.TABULAR,
    output_shape=TypeShape.NONE,
    domain="visualize",
)
def render_chart(
    kind: str,
    endpoint: str | None = None,
    path: str | None = None,
    table: str | None = None,
    query: str | None = None,
    encoding: "dict[str, Any] | None" = None,
    format: str | None = None,
    title: str | None = None,
    palette: str | None = None,
    colors: "list[str] | None" = None,
    style: "dict[str, Any] | None" = None,
) -> "dict[str, Any]":
    image_format = _resolve_format(format)
    frame, source = resolve_frame(endpoint, path, table, query)
    spec = build_chart_spec(kind, frame, encoding, title)
    # A custom color cycle overrides a named preset; either resolves the
    # StyleSpec together with the config-backed visualize defaults (read
    # through the NX-6 seam, never NX-2 directly).
    palette_choice = colors if colors is not None else palette
    spec_style = resolve_style(
        chokepoint().visualize_defaults(), palette=palette_choice, style=style
    )
    raw_bytes = render_spec(spec, image_format, spec_style)
    artifact_bytes = export_render(raw_bytes, image_format)  # NX-8 sanitize/validate
    return {
        "kind": spec.kind,
        "format": image_format,
        "source": source,
        "title": title,
        "mark": spec.mark,
        "encoding": spec.encoding,
        "artifact": _wire_artifact(artifact_bytes, image_format),
    }


def _resolve_format(requested: str | None) -> str:
    """The image format, defaulting to svg (the one default site) and
    refusing an unsupported value with the structured NX-3 shape."""
    image_format = requested if requested is not None else "svg"
    if image_format not in _MEDIA_TYPES:
        from localdata_mcp.ingest.refusals import invalid_source_refusal

        raise invalid_source_refusal(
            f"Unknown chart format {image_format!r} — supported: "
            f"{sorted(_MEDIA_TYPES)}."
        )
    return image_format


def _wire_artifact(payload: bytes, image_format: str) -> "dict[str, Any]":
    """The rendered artifact in a wire-safe envelope: SVG as inline
    UTF-8 text, PNG as base64 (a raster has no text form)."""
    media_type = _MEDIA_TYPES[image_format]
    if image_format == "svg":
        return {
            "media_type": media_type,
            "encoding": "utf-8",
            "content": payload.decode("utf-8"),
        }
    return {
        "media_type": media_type,
        "encoding": "base64",
        "content": base64.b64encode(payload).decode("ascii"),
    }
