"""localdata_mcp/visualize/charts/distribution.py — distribution kinds (E12.1).

The two single-source distribution charts (FR-503's distribution and
correlation families): `histogram` over one numeric column, and
`heatmap` of the correlation matrix across numeric columns. Each is a
pure extractor returning the arrays the renderer plots — no matplotlib
here. Neighbors: columns.py resolves channels; spec.py registers these
ChartKinds; render/ draws them.
"""

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from localdata_mcp.ingest.refusals import invalid_source_refusal

from .base import ChartKind
from .columns import channel_column, numeric_column


def _histogram_data(
    frame: pd.DataFrame, encoding: Mapping[str, Any]
) -> "dict[str, Any]":
    """Values of the `value` channel column — the renderer bins them."""
    column = channel_column(encoding, "value", "histogram")
    values = numeric_column(frame, column)
    return {"values": [float(v) for v in values], "label": column}


def _heatmap_data(frame: pd.DataFrame, encoding: Mapping[str, Any]) -> "dict[str, Any]":
    """The Pearson correlation matrix across the `columns` channel (or
    every numeric column when omitted) — refuses fewer than two."""
    requested = encoding.get("columns")
    if requested is not None:
        columns = [str(c) for c in requested]
        for column in columns:
            numeric_column(frame, column)  # presence + numeric refusal
        numeric = frame[columns].apply(pd.to_numeric, errors="coerce")
    else:
        numeric = frame.apply(pd.to_numeric, errors="coerce").dropna(
            axis="columns", how="all"
        )
        columns = [str(c) for c in numeric.columns]
    if len(columns) < 2:
        raise invalid_source_refusal(
            "A correlation heatmap needs at least two numeric columns "
            f"(found {len(columns)}: {columns}). Supply "
            "encoding={'columns': [<col>, <col>, …]} or an addressed "
            "source with two or more numeric columns."
        )
    matrix = numeric[columns].corr()
    return {
        "matrix": [[_finite(v) for v in row] for row in matrix.to_numpy()],
        "labels": columns,
    }


def _finite(value: float) -> float | None:
    """A correlation cell as a plain float, NaN mapped to None (the
    sentinel contract: a missing cell is None, never NaN)."""
    number = float(value)
    return None if number != number else number


HISTOGRAM = ChartKind(name="histogram", mark="bars", extract=_histogram_data)
HEATMAP = ChartKind(name="heatmap", mark="cells", extract=_heatmap_data)
