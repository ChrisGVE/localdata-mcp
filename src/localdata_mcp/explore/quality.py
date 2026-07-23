"""localdata_mcp/explore/quality.py — X-2 data-quality profile (E9.2).

FR-202's `profile_data` (the renamed successor of `main`'s
`get_data_quality_report` — the rename marks the envelope/scope
change, DR GP2): per-column null counts, inferred types, numeric/
datetime ranges, and cardinality over any exactly-one-source address
(addressing.py). Pure computation over the guard-fetched frame —
floats are direct pandas arithmetic on the addressed data (the
testbench compares them at `testbench.tol_quality_rtol`; counts and
cardinalities compare exactly). Neighbors: addressing.py resolves;
categorical.py and search.py are the sibling X-tools.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from localdata_mcp.nexus.contract.spec import TypeShape, tool_spec

from .addressing import resolve_frame, source_params


def _column_profile(series: "pd.Series[Any]") -> dict[str, Any]:
    profile: dict[str, Any] = {
        "inferred_type": str(series.dtype),
        "null_count": int(series.isna().sum()),
        "cardinality": int(series.nunique(dropna=True)),
    }
    if pd.api.types.is_numeric_dtype(series) and series.notna().any():
        profile["min"] = series.min().item()
        profile["max"] = series.max().item()
        profile["mean"] = float(series.mean())
    elif pd.api.types.is_datetime64_any_dtype(series) and series.notna().any():
        profile["min"] = str(series.min())
        profile["max"] = str(series.max())
    return profile


@tool_spec(
    name="profile_data",
    summary=(
        "Profile a tabular source's data quality: per-column null "
        "counts, inferred types, numeric ranges, and cardinality. "
        "Address with exactly one of endpoint= or path=; endpoint "
        "sources take exactly one of table= or query=."
    ),
    params=source_params(),
    input_shape=TypeShape.TABULAR,
    output_shape=TypeShape.SCALAR,
    domain="explore",
)
def profile_data(
    endpoint: str | None = None,
    path: str | None = None,
    table: str | None = None,
    query: str | None = None,
) -> Any:
    frame, source = resolve_frame(endpoint, path, table, query)
    return {
        "source": source,
        "row_count": int(len(frame)),
        "column_count": int(frame.shape[1]),
        "columns": {str(name): _column_profile(frame[name]) for name in frame.columns},
    }
