"""localdata_mcp/visualize/charts/columns.py — encoding-channel column resolution (E12.1).

The shared column-access every chart kind leans on, declared once
(NFR-402): read a required encoding channel, refuse a missing channel
or absent column with the structured NX-3 shapes (ingest/refusals),
and coerce a channel to numeric. A chart's *encoding* names which
frame column fills each visual channel ("x", "value", "source"…); this
module turns that name into data or a structured refusal — a chart
kind never raises a bare error at a caller. Neighbors: distribution.py,
relational.py, spatial.py extract per-kind data through here; spec.py
dispatches to them.
"""

from __future__ import annotations

from typing import Mapping

import pandas as pd

from localdata_mcp.ingest.refusals import (
    invalid_source_refusal,
    missing_entity_refusal,
)


def channel_column(encoding: Mapping[str, object], channel: str, kind: str) -> str:
    """The column name bound to a required encoding channel — a missing
    channel is a structured refusal naming the kind and the channel."""
    value = encoding.get(channel)
    if value is None:
        raise invalid_source_refusal(
            f"Chart kind {kind!r} requires the {channel!r} encoding "
            f"channel (got encoding keys {sorted(encoding)}). Supply "
            f"encoding={{{channel!r}: <column>}}."
        )
    return str(value)


def require_column(frame: pd.DataFrame, column: str) -> None:
    """Refuse (structured, NX-3) a column absent from the frame."""
    if column not in frame.columns:
        raise missing_entity_refusal(
            f"Column {column!r} is not present in the addressed data "
            f"(available: {[str(c) for c in frame.columns]}).",
            "Call profile_data on the same source to inspect its columns.",
        )


def numeric_column(frame: pd.DataFrame, column: str) -> "pd.Series[float]":
    """The column coerced to numeric with missing values dropped;
    refuses a column with no numeric content (a chart axis needs
    numbers)."""
    require_column(frame, column)
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        raise invalid_source_refusal(
            f"Column {column!r} carries no numeric values — this chart "
            "channel needs numeric input."
        )
    return values


def numeric_frame(frame: pd.DataFrame, columns: "list[str]") -> pd.DataFrame:
    """The named columns coerced numeric and row-aligned (rows with any
    missing side dropped) — the multi-column chart input (heatmap,
    paired scatter)."""
    for column in columns:
        require_column(frame, column)
    coerced = frame[columns].apply(pd.to_numeric, errors="coerce").dropna()
    if coerced.empty:
        raise invalid_source_refusal(
            f"Columns {columns} share no rows with numeric values on "
            "every column — this chart needs aligned numeric input."
        )
    return coerced
