"""localdata_mcp/process/domains/pattern_recognition/anomalies.py — FR-301.

`detect_anomalies`'s computation, re-authored from `main`'s
`AnomalyDetectionTransformer`: isolation_forest (default), lof (local
outlier factor), or zscore (the parameter-free baseline; a
zero-variance column drives its scores to NaN, which the shared
sentinel converts to a structured error — the degenerate path is the
sentinel's, not ad-hoc code here). `contamination` keeps `main`'s
0.1 default; `seed` pins the forest (S3.3). Neighbors: matrices.py
preps input; tools.py declares the ToolSpec.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..support import invalid_source_refusal
from .matrices import construct, numeric_matrix

METHODS = ("isolation_forest", "lof", "zscore")

# main's default expected anomaly share; the z-score cut at |z| >= 3
# is the conventional three-sigma rule.
_DEFAULT_CONTAMINATION = 0.1
_ZSCORE_CUT = 3.0


def find_anomalies(
    frame: pd.DataFrame,
    columns: list[str] | None = None,
    method: str = "isolation_forest",
    contamination: float = _DEFAULT_CONTAMINATION,
    seed: int | None = None,
    algorithm_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Anomaly indices and scores over the addressed data."""
    if method not in METHODS:
        raise invalid_source_refusal(
            f"Unknown method {method!r} — one of {list(METHODS)}."
        )
    matrix, names = numeric_matrix(frame, columns)
    if method == "zscore":
        flags, scores = _zscore_flags(matrix)
    else:
        flags, scores = _estimator_flags(
            matrix, method, contamination, seed, algorithm_params
        )
    indices = [int(index) for index in np.nonzero(flags)[0]]
    return {
        "method": method,
        "columns": names,
        "n_samples": int(matrix.shape[0]),
        "anomaly_count": len(indices),
        "anomaly_indices": indices,
        "anomaly_share": float(len(indices) / matrix.shape[0]),
        "score_summary": {
            "min": float(np.min(scores)),
            "max": float(np.max(scores)),
            "mean": float(np.mean(scores)),
        },
    }


def _estimator_flags(
    matrix: "np.ndarray[Any, Any]",
    method: str,
    contamination: float,
    seed: int | None,
    algorithm_params: dict[str, Any] | None,
) -> tuple["np.ndarray[Any, Any]", "np.ndarray[Any, Any]"]:
    if method == "isolation_forest":
        from sklearn.ensemble import IsolationForest

        model = construct(
            IsolationForest, algorithm_params, seed, contamination=contamination
        )
        verdicts = model.fit_predict(matrix)
        scores = model.score_samples(matrix)
    else:
        from sklearn.neighbors import LocalOutlierFactor

        model = construct(
            LocalOutlierFactor, algorithm_params, None, contamination=contamination
        )
        verdicts = model.fit_predict(matrix)
        scores = model.negative_outlier_factor_
    return verdicts == -1, np.asarray(scores)


def _zscore_flags(
    matrix: "np.ndarray[Any, Any]",
) -> tuple["np.ndarray[Any, Any]", "np.ndarray[Any, Any]"]:
    """Three-sigma rule on the per-column standardized values; the
    row score is its worst column."""
    scores = np.abs((matrix - matrix.mean(axis=0)) / matrix.std(axis=0, ddof=1)).max(
        axis=1
    )
    return scores >= _ZSCORE_CUT, scores
