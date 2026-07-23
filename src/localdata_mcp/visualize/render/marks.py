"""localdata_mcp/visualize/render/marks.py — per-kind drawing (E12.2).

The six draw functions, one per launch ChartKind: each receives a
matplotlib `Axes` and the ChartSpec's extracted data, draws the marks
via the OO API, and tags its data artist(s) with a stable SVG gid
(`localdata-<mark>` for the primary marks, named secondaries for fit
lines and network nodes) so the SVG structural test can locate and
count them (FR-501). No Figure lifecycle here — backend.py owns
creation, export, and disposal. Heatmap uses `pcolormesh` (paths), NOT
`imshow`: imshow embeds an `<image>` data-URI raster the NX-8 allow-list
sanitizer strips. Neighbors: backend.py dispatches here; charts/ built
the data these functions consume.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection


def draw_histogram(ax: Axes, data: Mapping[str, Any]) -> None:
    counts, edges = np.histogram(np.asarray(data["values"], dtype=float), bins="auto")
    bars = ax.stairs(counts, edges, fill=True)
    bars.set_gid("localdata-bars")
    ax.set_xlabel(str(data.get("label", "")))
    ax.set_ylabel("count")


def draw_heatmap(ax: Axes, data: Mapping[str, Any]) -> None:
    labels = list(data["labels"])
    matrix = np.array(
        [[np.nan if v is None else float(v) for v in row] for row in data["matrix"]],
        dtype=float,
    )
    mesh = ax.pcolormesh(matrix, cmap="viridis", vmin=-1.0, vmax=1.0)
    mesh.set_gid("localdata-cells")
    ax.set_xticks(np.arange(len(labels)) + 0.5, labels=labels)
    ax.set_yticks(np.arange(len(labels)) + 0.5, labels=labels)
    ax.invert_yaxis()


def draw_scatter_fit(ax: Axes, data: Mapping[str, Any]) -> None:
    xs = np.asarray(data["x"], dtype=float)
    ys = np.asarray(data["y"], dtype=float)
    points = ax.scatter(xs, ys)
    points.set_gid("localdata-points")
    fit = data["fit"]
    if fit["slope"] is not None and xs.size:
        line_x = np.array([xs.min(), xs.max()])
        line_y = fit["slope"] * line_x + fit["intercept"]
        (fit_line,) = ax.plot(line_x, line_y, color="crimson")
        fit_line.set_gid("localdata-fit")
    ax.set_xlabel(str(data.get("x_label", "")))
    ax.set_ylabel(str(data.get("y_label", "")))


def draw_line_timeseries(ax: Axes, data: Mapping[str, Any]) -> None:
    xs = _time_axis(data["x"], bool(data.get("x_is_time")))
    (line,) = ax.plot(xs, np.asarray(data["y"], dtype=float))
    line.set_gid("localdata-line")
    ax.set_xlabel(str(data.get("x_label", "")))
    ax.set_ylabel(str(data.get("y_label", "")))


def draw_geo_map(ax: Axes, data: Mapping[str, Any]) -> None:
    lon = np.asarray(data["lon"], dtype=float)
    lat = np.asarray(data["lat"], dtype=float)
    if "value" in data:
        points = ax.scatter(
            lon, lat, c=np.asarray(data["value"], dtype=float), cmap="viridis"
        )
    else:
        points = ax.scatter(lon, lat)
    points.set_gid("localdata-points")
    ax.set_xlabel("lon")
    ax.set_ylabel("lat")


def draw_network_layout(ax: Axes, data: Mapping[str, Any]) -> None:
    positions = [tuple(p) for p in data["positions"]]
    segments = [
        (positions[source], positions[target]) for source, target in data["edges"]
    ]
    edges = LineCollection(segments, colors="gray", linewidths=1.0)
    edges.set_gid("localdata-edges")
    ax.add_collection(edges)
    if positions:
        coords = np.asarray(positions, dtype=float)
        nodes = ax.scatter(coords[:, 0], coords[:, 1], zorder=2)
        nodes.set_gid("localdata-points")
    ax.set_axis_off()
    ax.autoscale_view()


def _time_axis(values: "list[Any]", is_time: bool) -> "np.ndarray":
    """The x values as a plottable axis — datetime64 when the spec
    flagged a time axis, float otherwise."""
    if is_time:
        return np.array(values, dtype="datetime64[ns]")
    return np.asarray(values, dtype=float)


DRAWERS = {
    "histogram": draw_histogram,
    "heatmap": draw_heatmap,
    "scatter_fit": draw_scatter_fit,
    "line_timeseries": draw_line_timeseries,
    "geo_map": draw_geo_map,
    "network_layout": draw_network_layout,
}
