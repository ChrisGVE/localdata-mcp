"""localdata_mcp/visualize/charts/relational.py — relational kinds (E12.1).

The two two-channel relational charts (FR-503's regression-fit and
time-series families): `scatter_fit` (x vs y points plus a
least-squares fit line) and `line_timeseries` (y ordered along x). Pure
extractors — the fit is computed here (numpy least squares) so the
renderer only draws. Neighbors: columns.py resolves channels; spec.py
registers these; render/ draws the points/line and the fit overlay.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from .base import ChartKind
from .columns import channel_column, numeric_frame


def _scatter_fit_data(
    frame: pd.DataFrame, encoding: Mapping[str, Any]
) -> "dict[str, Any]":
    """Aligned x/y points and the least-squares line over them. x and y
    are optional — omitted, they default to the first two numeric
    columns (excluding an optional `color` column), so a
    just-clustered relation charts with no column knowledge (the
    cluster_then_chart wrapper). An optional `color` channel colours the
    points by a category column (e.g. `cluster`)."""
    color_col = encoding.get("color")
    x_col, y_col = _resolve_axes(frame, encoding, color_col)
    aligned = numeric_frame(frame, [x_col, y_col])
    xs = aligned[x_col].to_numpy(dtype=float)
    ys = aligned[y_col].to_numpy(dtype=float)
    data: dict[str, Any] = {
        "x": [float(v) for v in xs],
        "y": [float(v) for v in ys],
        "fit": _least_squares(xs, ys),
        "x_label": x_col,
        "y_label": y_col,
    }
    if color_col is not None:
        from .columns import require_column

        require_column(frame, str(color_col))
        colors = frame.loc[aligned.index, str(color_col)]
        data["color"] = [None if pd.isna(v) else v for v in colors]
        data["color_label"] = str(color_col)
    return data


def _resolve_axes(
    frame: pd.DataFrame, encoding: Mapping[str, Any], color_col: Any
) -> "tuple[str, str]":
    """The x/y columns: the supplied channels, or the first two numeric
    columns (never the colour column) when omitted."""
    x_given, y_given = encoding.get("x"), encoding.get("y")
    if x_given is not None and y_given is not None:
        return str(x_given), str(y_given)
    excluded = {str(color_col)} if color_col is not None else set()
    numeric = [
        str(name)
        for name in frame.columns
        if str(name) not in excluded
        and pd.to_numeric(frame[name], errors="coerce").notna().any()
    ]
    if len(numeric) < 2:
        from localdata_mcp.ingest.refusals import invalid_source_refusal

        raise invalid_source_refusal(
            "scatter_fit needs two numeric columns — supply "
            "encoding={'x': <col>, 'y': <col>} or an addressed source "
            f"with two numeric columns (found {numeric})."
        )
    return numeric[0], numeric[1]


def _least_squares(xs: "np.ndarray", ys: "np.ndarray") -> "dict[str, Any]":
    """Slope/intercept of the least-squares line, or a None-pair when
    x has zero spread (a vertical fit is undefined — the sentinel
    contract keeps it None, never NaN)."""
    if xs.size < 2 or float(np.ptp(xs)) == 0.0:
        return {"slope": None, "intercept": None}
    slope, intercept = np.polyfit(xs, ys, 1)
    return {"slope": float(slope), "intercept": float(intercept)}


def _line_timeseries_data(
    frame: pd.DataFrame, encoding: Mapping[str, Any]
) -> "dict[str, Any]":
    """y values ordered along the x (time/ordering) channel."""
    x_col = channel_column(encoding, "x", "line_timeseries")
    y_col = channel_column(encoding, "y", "line_timeseries")
    ordering = _ordering_axis(frame, x_col)
    y_values = pd.to_numeric(frame[y_col], errors="coerce")
    ordered = pd.DataFrame({"x": ordering, "y": y_values}).dropna()
    ordered = ordered.sort_values("x", kind="stable")
    return {
        "x": [_axis_value(v) for v in ordered["x"]],
        "y": [float(v) for v in ordered["y"]],
        "x_label": x_col,
        "y_label": y_col,
        "x_is_time": bool(pd.api.types.is_datetime64_any_dtype(ordering)),
    }


def _ordering_axis(frame: pd.DataFrame, column: str) -> "pd.Series[Any]":
    """The x column as a sortable axis: a real datetime parse when it
    reads as time, else numeric — refuses neither (an unparseable axis
    falls back to numeric coercion, dropped rows handle the rest)."""
    from .columns import require_column

    require_column(frame, column)
    raw = frame[column]
    if pd.api.types.is_datetime64_any_dtype(raw):
        return raw
    if pd.api.types.is_numeric_dtype(raw):
        return pd.to_numeric(raw, errors="coerce")
    # A non-numeric column: prefer a datetime reading (a date-string
    # axis), falling back to numeric coercion when it does not parse as
    # time — a bare integer column never masquerades as epoch nanoseconds.
    parsed = pd.to_datetime(raw, errors="coerce")
    if parsed.notna().mean() >= 0.8:
        return parsed
    return pd.to_numeric(raw, errors="coerce")


def _axis_value(value: Any) -> Any:
    """One x-axis value as a wire-safe scalar: an ISO string for a
    timestamp, a float otherwise."""
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return float(value)


SCATTER_FIT = ChartKind(name="scatter_fit", mark="points", extract=_scatter_fit_data)
LINE_TIMESERIES = ChartKind(
    name="line_timeseries", mark="line", extract=_line_timeseries_data
)
