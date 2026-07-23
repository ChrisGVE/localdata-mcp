"""localdata_mcp/process/preprocessing/stages.py — FR-303 data-prep stages.

Feature P-2: missing-value handling and type conversion as composable
pipeline stages, harvested from the dead preprocessing mass. Both are
TABULAR→TABULAR — callable standalone AND usable as `dag_spec` stages
(E11). `missing_strategy` is a CLOSED enum declared once in the
ToolSpec ({drop, mean, median, mode, forward_fill, constant}) with
**default drop** — the one strategy that fabricates no values, so a
caller who supplied nothing gets no silently-invented data (imputation
is an explicit analytical choice). `convert_types` coerces named
columns to a declared target type, reporting how many cells failed the
cast rather than crashing. Neighbors: tools.py declares the ToolSpecs;
process/domains/support.py owns the addressing.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..domains.support import invalid_source_refusal, require_columns

MISSING_STRATEGIES = ("drop", "mean", "median", "mode", "forward_fill", "constant")
TARGET_TYPES = ("numeric", "integer", "string", "datetime", "boolean")


def prepare_missing_values(
    frame: pd.DataFrame,
    columns: list[str] | None = None,
    missing_strategy: str = "drop",
    fill_value: Any = None,
) -> dict[str, Any]:
    """Apply the missing-value strategy, returning the cleaned relation."""
    if missing_strategy not in MISSING_STRATEGIES:
        raise invalid_source_refusal(
            f"Unknown missing_strategy {missing_strategy!r} — one of "
            f"{list(MISSING_STRATEGIES)}."
        )
    targets = list(columns) if columns else [str(c) for c in frame.columns]
    require_columns(frame, *targets)
    before = int(frame.isna().sum().sum())
    result = (
        _drop(frame, targets)
        if missing_strategy == "drop"
        else _impute(frame, targets, missing_strategy, fill_value)
    )
    cleaned = result.astype(object).where(pd.notna(result), None)
    return {
        "missing_strategy": missing_strategy,
        "columns": [str(name) for name in cleaned.columns],
        "rows": cleaned.to_numpy().tolist(),
        "total_rows": int(len(cleaned)),
        "missing_before": before,
        "missing_after": int(result.isna().sum().sum()),
    }


def _drop(frame: pd.DataFrame, targets: list[str]) -> pd.DataFrame:
    return frame.dropna(subset=targets).reset_index(drop=True)


def _impute(
    frame: pd.DataFrame, targets: list[str], strategy: str, fill_value: Any
) -> pd.DataFrame:
    result = frame.copy()
    for column in targets:
        series = result[column]
        if strategy == "mean":
            result[column] = series.fillna(
                pd.to_numeric(series, errors="coerce").mean()
            )
        elif strategy == "median":
            result[column] = series.fillna(
                pd.to_numeric(series, errors="coerce").median()
            )
        elif strategy == "mode":
            modes = series.mode(dropna=True)
            if not modes.empty:
                result[column] = series.fillna(modes.iloc[0])
        elif strategy == "forward_fill":
            result[column] = series.ffill()
        else:  # constant
            if fill_value is None:
                raise invalid_source_refusal(
                    "missing_strategy='constant' needs fill_value=."
                )
            result[column] = series.fillna(fill_value)
    return result


def convert_types(
    frame: pd.DataFrame,
    conversions: dict[str, str],
) -> dict[str, Any]:
    """Coerce each named column to its target type; report cast failures."""
    if not conversions:
        raise invalid_source_refusal("conversions must name at least one column.")
    require_columns(frame, *conversions.keys())
    result = frame.copy()
    report: dict[str, Any] = {}
    for column, target in conversions.items():
        if target not in TARGET_TYPES:
            raise invalid_source_refusal(
                f"Unknown target type {target!r} for {column!r} — one of "
                f"{list(TARGET_TYPES)}."
            )
        converted, failures = _coerce(result[column], target)
        result[column] = converted
        report[column] = {"target": target, "failed_cells": failures}
    cleaned = result.astype(object).where(pd.notna(result), None)
    return {
        "conversions": report,
        "columns": [str(name) for name in cleaned.columns],
        "rows": cleaned.to_numpy().tolist(),
        "total_rows": int(len(cleaned)),
    }


def _coerce(series: "pd.Series[Any]", target: str) -> "tuple[pd.Series[Any], int]":
    if target in ("numeric", "integer"):
        coerced = pd.to_numeric(series, errors="coerce")
        failures = int(coerced.isna().sum() - series.isna().sum())
        if target == "integer":
            coerced = coerced.astype("Int64")
        return coerced, max(0, failures)
    if target == "datetime":
        coerced = pd.to_datetime(series, errors="coerce")
        return coerced, max(0, int(coerced.isna().sum() - series.isna().sum()))
    if target == "boolean":
        return series.astype("boolean"), 0
    return series.astype("string"), 0
