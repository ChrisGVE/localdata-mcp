"""localdata_mcp/process/domains/regression_modeling/evaluation.py — FR-301.

`evaluate_model_performance`'s computation, ported from `main`'s
corrected wrapper: the tool scores an actual-vs-predicted column PAIR
already stored in the addressed data — it holds no fitted estimator,
so the metrics are computed from the two columns directly (which is
what the tool has always claimed to do). Regression metrics for a
numeric pair, weighted classification metrics otherwise. Neighbors:
tools.py declares the ToolSpec; regression.py fits models.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..support import invalid_source_refusal, require_columns

MODEL_TYPES = ("regression", "classification")


def evaluate_predictions(
    frame: pd.DataFrame,
    target_column: str,
    prediction_column: str,
    model_type: str = "regression",
) -> dict[str, Any]:
    """Metrics over the stored actual/predicted pair."""
    if model_type not in MODEL_TYPES:
        raise invalid_source_refusal(
            f"Unknown model_type {model_type!r} — one of {list(MODEL_TYPES)}."
        )
    require_columns(frame, target_column, prediction_column)
    paired = frame[[target_column, prediction_column]].dropna()
    if paired.empty:
        raise invalid_source_refusal(
            f"No rows with both {target_column!r} and {prediction_column!r} present."
        )
    metrics = (
        _classification_metrics(paired[target_column], paired[prediction_column])
        if model_type == "classification"
        else _regression_metrics(paired, target_column, prediction_column)
    )
    return {
        "model_type": model_type,
        "n_samples": int(len(paired)),
        "target_column": target_column,
        "prediction_column": prediction_column,
        "metrics": metrics,
    }


def _classification_metrics(
    actual: "pd.Series[Any]", predicted: "pd.Series[Any]"
) -> dict[str, float]:
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
    )

    return {
        "accuracy": float(accuracy_score(actual, predicted)),
        "precision": float(
            precision_score(actual, predicted, average="weighted", zero_division=0)
        ),
        "recall": float(
            recall_score(actual, predicted, average="weighted", zero_division=0)
        ),
        "f1": float(f1_score(actual, predicted, average="weighted", zero_division=0)),
    }


def _regression_metrics(
    paired: pd.DataFrame, target_column: str, prediction_column: str
) -> dict[str, float]:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    numeric = paired.apply(pd.to_numeric, errors="coerce").dropna()
    if numeric.empty:
        raise invalid_source_refusal(
            f"Columns {target_column!r} and {prediction_column!r} are not "
            "numeric — regression evaluation needs numeric values."
        )
    actual = numeric[target_column]
    predicted = numeric[prediction_column]
    residuals = actual - predicted
    mse = float(mean_squared_error(actual, predicted))
    return {
        "r2": float(r2_score(actual, predicted)),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(actual, predicted)),
        "mean_residual": float(residuals.mean()),
        "residual_std": float(residuals.std()),
    }
