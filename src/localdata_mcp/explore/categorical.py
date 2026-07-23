"""localdata_mcp/explore/categorical.py — X-4 category mapping (E9.4).

FR-204's `map_categories`, report-only by decision (section 6(h)):
distinct values, frequencies, and a suggested encoding for one column
of any exactly-one-source address — label encoding for high-cardinality
or ordered-looking values, one-hot for small closed sets. The
suggestion is REPORTING, never a transform: transform-and-persist
stays deferred under FR-303's lifecycle (S10). Endpoint sources take
the X-2 second slot (table=/query=). Neighbors: addressing.py
resolves; quality.py profiles the same frames.
"""

from __future__ import annotations

from typing import Any

from localdata_mcp.nexus.contract.spec import Param, TypeShape, tool_spec

from ..ingest.refusals import missing_entity_refusal
from .addressing import resolve_frame, source_params

# The one-hot suggestion cutoff: above this many distinct values a
# one-hot expansion stops being a sane width (non-config legibility
# bound; the report states the cardinality either way).
_ONE_HOT_MAX_CATEGORIES = 16
# Value-listing cap for high-cardinality columns (explicit `values_
# truncated` flag — the distinct_count is always the full number).
_VALUE_LISTING_CAP = 64


@tool_spec(
    name="map_categories",
    summary=(
        "Map one column's categorical values: distinct values with "
        "frequencies and a suggested encoding (label vs one-hot) — a "
        "report only, nothing is transformed or persisted. Address "
        "with exactly one of endpoint= or path=."
    ),
    params=(
        *source_params(),
        Param("column", str, "The column whose categories to map."),
    ),
    input_shape=TypeShape.TABULAR,
    output_shape=TypeShape.SCALAR,
    domain="explore",
)
def map_categories(
    endpoint: str | None = None,
    path: str | None = None,
    table: str | None = None,
    query: str | None = None,
    column: str = "",
) -> Any:
    frame, source = resolve_frame(endpoint, path, table, query)
    if column not in frame.columns:
        raise missing_entity_refusal(
            f"Column {column!r} not in the source.",
            "Call profile_data on the same source to list its columns.",
        )
    series = frame[column]
    frequencies = series.value_counts(dropna=True)
    distinct = int(frequencies.size)
    suggested = "one-hot" if distinct <= _ONE_HOT_MAX_CATEGORIES else "label"
    return {
        "source": source,
        "column": column,
        "row_count": int(len(series)),
        "null_count": int(series.isna().sum()),
        "distinct_count": distinct,
        "values": [
            {
                "value": value.item() if hasattr(value, "item") else value,
                "count": int(count),
            }
            for value, count in list(frequencies.items())[:_VALUE_LISTING_CAP]
        ],
        "values_truncated": distinct > _VALUE_LISTING_CAP,
        "suggested_encoding": suggested,
        "note": (
            "Report only - transform-and-persist is deferred (FR-303 lifecycle, S10)."
        ),
    }
