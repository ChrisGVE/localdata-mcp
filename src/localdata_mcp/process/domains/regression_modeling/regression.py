"""localdata_mcp/process/domains/regression_modeling/regression.py — FR-301.

`analyze_regression`'s computation: assemble the design matrix
(numeric features, intercept column included), fit the resolved
estimator (estimators.py), and report coefficients plus fit quality.
The result carries the sentinel's class-4 keys BY DESIGN (S3.3): the
design matrix's `rank` (numpy `matrix_rank`), its `design_columns`,
and its `condition_number` (κ computed on the matrix exactly as the
solver sees it — intercept included, no re-scaling), so a
rank-deficient fit's full-shaped, NaN-free minimum-norm garbage
becomes a structured NX-3 error instead of a silent success. The
logistic path also emits the class-2 `converged` flag from the
solver's own iteration report. Neighbors: tools.py declares the
ToolSpec; evaluation.py scores stored predictions.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..support import invalid_source_refusal, require_columns
from .estimators import build_estimator, resolve_model_type

# Polynomial expansion default: quadratic — the smallest expansion
# that is actually polynomial.
_DEFAULT_DEGREE = 2


def fit_regression(
    frame: pd.DataFrame,
    target_column: str,
    feature_columns: list[str] | None = None,
    model_type: str = "linear",
    regularization: str | None = None,
    degree: int = _DEFAULT_DEGREE,
    algorithm_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fit the requested model; coefficients, metrics, and the
    sentinel's rank/condition keys on the solved design matrix."""
    model_type = resolve_model_type(model_type, regularization)
    target, features = _design_frame(frame, target_column, feature_columns)
    fit_matrix, names = _fit_matrix(features, model_type, degree)
    estimator = build_estimator(model_type, algorithm_params)
    estimator.fit(fit_matrix, target)
    result: dict[str, Any] = {
        "model_type": model_type,
        "n_samples": int(fit_matrix.shape[0]),
        "target_column": target_column,
        "feature_columns": list(features.columns),
        "coefficients": _coefficients(estimator, names),
        "intercept": _intercept(estimator),
        "metrics": _fit_metrics(estimator, fit_matrix, target, model_type),
        # Sentinel class-4 keys: κ and rank on [1 | X] — the intercept
        # the estimator fits internally IS a design column (S3.3).
        **_design_health(np.hstack([np.ones((fit_matrix.shape[0], 1)), fit_matrix])),
    }
    # Direct solvers (OLS, closed-form ridge) report n_iter_ as None —
    # only an actually-iterative fit carries a convergence verdict.
    iterations_report = getattr(estimator, "n_iter_", None)
    if iterations_report is not None:
        limit = estimator.get_params().get("max_iter")
        iterations = int(np.max(iterations_report))
        result["converged"] = bool(limit is None or iterations < limit)
    return result


def _design_frame(
    frame: pd.DataFrame, target_column: str, feature_columns: list[str] | None
) -> tuple["pd.Series[float]", pd.DataFrame]:
    require_columns(frame, target_column, *(feature_columns or ()))
    if feature_columns is None:
        # main's issue-#23 rule: without an explicit choice, every
        # OTHER numeric column — a text column must never reach the
        # estimator as an opaque conversion crash.
        feature_columns = [
            str(name)
            for name in frame.select_dtypes(include=[np.number]).columns
            if str(name) != target_column
        ]
        if not feature_columns:
            raise invalid_source_refusal(
                f"No numeric feature columns found besides {target_column!r} "
                f"(columns: {[str(c) for c in frame.columns]})."
            )
    selected = frame[[*feature_columns, target_column]].apply(
        pd.to_numeric, errors="coerce"
    )
    selected = selected.dropna()
    if selected.empty:
        raise invalid_source_refusal(
            "No complete numeric rows remain after dropping missing values."
        )
    return selected[target_column], selected[feature_columns]


def _fit_matrix(
    features: pd.DataFrame, model_type: str, degree: int
) -> tuple["np.ndarray[Any, Any]", list[str]]:
    """The feature matrix the estimator fits (its own intercept term
    on top): raw numerics, or the bias-free polynomial expansion."""
    if model_type == "polynomial":
        from sklearn.preprocessing import PolynomialFeatures

        expansion = PolynomialFeatures(degree=degree, include_bias=False)
        matrix = expansion.fit_transform(features.to_numpy(dtype=float))
        names = [
            str(name) for name in expansion.get_feature_names_out(features.columns)
        ]
        return matrix, names
    raw = features.to_numpy(dtype=float)
    return raw, [str(name) for name in features.columns]


def _coefficients(estimator: Any, names: list[str]) -> dict[str, float]:
    flat = np.asarray(estimator.coef_).reshape(-1)
    return {name: float(value) for name, value in zip(names, flat)}


def _intercept(estimator: Any) -> float:
    return float(np.asarray(estimator.intercept_).reshape(-1)[0])


def _fit_metrics(
    estimator: Any,
    design: "np.ndarray[Any, Any]",
    target: "pd.Series[float]",
    model_type: str,
) -> dict[str, float]:
    predicted = estimator.predict(design)
    if model_type == "logistic":
        from sklearn.metrics import accuracy_score

        return {"accuracy": float(accuracy_score(target, predicted))}
    from sklearn.metrics import mean_squared_error, r2_score

    mse = float(mean_squared_error(target, predicted))
    return {
        "r2": float(r2_score(target, predicted)),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
    }


def _design_health(design: "np.ndarray[Any, Any]") -> dict[str, Any]:
    """The class-4 sentinel inputs, computed on the solved matrix."""
    return {
        "rank": int(np.linalg.matrix_rank(design)),
        "design_columns": int(design.shape[1]),
        "condition_number": float(np.linalg.cond(design)),
    }
