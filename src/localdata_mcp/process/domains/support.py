"""localdata_mcp/process/domains/support.py — shared domain substrate (E10).

The helpers every Process domain family leans on, declared once
(NFR-402): the addressed-frame entry (delegating to explore's
exactly-one-source contract — one addressing home, E9.2), column
presence checks, numeric coercion, and the two-group split the
comparison tools (hypothesis, effect sizes, A/B) share. All refusals
are the structured NX-3 shapes from ingest/refusals.py — a domain
tool never raises a bare ValueError at a caller. Neighbors: each
domains/<family>/ package imports through here; tools.py modules
declare the ToolSpecs.
"""

from __future__ import annotations

import pandas as pd

from localdata_mcp.explore.addressing import resolve_frame, source_params
from localdata_mcp.ingest.refusals import (
    invalid_source_refusal,
    missing_entity_refusal,
)

__all__ = [
    "addressed_frame",
    "source_params",
    "require_columns",
    "numeric_values",
    "two_groups",
    "paired_numeric",
    "comparison_groups",
    "invalid_source_refusal",
    "missing_entity_refusal",
]


def addressed_frame(
    endpoint: str | None,
    path: str | None,
    table: str | None,
    query: str | None,
) -> tuple[pd.DataFrame, str]:
    """The addressed tabular data plus its source label (X-2 contract)."""
    return resolve_frame(endpoint, path, table, query)


def require_columns(frame: pd.DataFrame, *columns: str | None) -> None:
    """Refuse (structured, NX-3) any named column absent from the frame."""
    missing = [
        name for name in columns if name is not None and name not in frame.columns
    ]
    if missing:
        raise missing_entity_refusal(
            f"Column(s) {missing} not present in the addressed data "
            f"(available: {[str(c) for c in frame.columns]}).",
            "Call profile_data on the same source to inspect its columns.",
        )


def numeric_values(frame: pd.DataFrame, column: str) -> "pd.Series[float]":
    """The column coerced to numeric with missing values dropped;
    refuses a column with no numeric content."""
    require_columns(frame, column)
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        raise invalid_source_refusal(
            f"Column {column!r} carries no numeric values — the "
            "statistical tools need numeric input."
        )
    return values


def paired_numeric(
    frame: pd.DataFrame, column: str | None, second_column: str | None
) -> tuple["pd.Series[float]", "pd.Series[float]"]:
    """The two columns coerced numeric and aligned row-wise (rows with
    a missing side dropped) — the paired-test/correlation input."""
    if column is None or second_column is None:
        raise invalid_source_refusal(
            "This computation needs both column= and second_column=."
        )
    require_columns(frame, column, second_column)
    paired = frame[[column, second_column]].apply(pd.to_numeric, errors="coerce")
    paired = paired.dropna()
    return paired[column], paired[second_column]


def comparison_groups(
    frame: pd.DataFrame, column: str | None, group_column: str | None
) -> tuple[str, "pd.Series[float]", str, "pd.Series[float]"]:
    """two_groups with the None-refusal the dispatchers share."""
    if column is None or group_column is None:
        raise invalid_source_refusal(
            "This two-sample comparison needs column= and group_column=."
        )
    return two_groups(frame, column, group_column)


def two_groups(
    frame: pd.DataFrame, value_column: str, group_column: str
) -> tuple[str, "pd.Series[float]", str, "pd.Series[float]"]:
    """The exactly-two-group split (label_a, values_a, label_b,
    values_b) the pairwise comparison tools share; any other group
    count is a structured refusal naming the groups found."""
    require_columns(frame, value_column, group_column)
    labels = frame[group_column].dropna().unique().tolist()
    if len(labels) != 2:
        raise invalid_source_refusal(
            f"Column {group_column!r} defines {len(labels)} group(s) "
            f"({[str(label) for label in labels]}) — this comparison "
            "needs exactly two; use analyze_anova for three or more."
        )
    first, second = sorted(labels, key=str)
    values_a = numeric_values(frame[frame[group_column] == first], value_column)
    values_b = numeric_values(frame[frame[group_column] == second], value_column)
    return str(first), values_a, str(second), values_b
