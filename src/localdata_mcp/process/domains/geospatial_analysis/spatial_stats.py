"""localdata_mcp/process/domains/geospatial_analysis/spatial_stats.py — FR-301.

The coordinate-statistics trio carried by name from `main`:
`analyze_spatial_autocorrelation` (global Moran's I with its
permutation-free analytic z-score), `find_spatial_hotspots`
(Getis-Ord Gi* per point, hot/cold at the significance level), and
`calculate_spatial_distances` (a summarized distance-matrix report,
bounded to keep the n² matrix legible). All three read the addressed
coordinates through weights.py's shared k-NN neighbourhood.
Neighbors: weights.py builds the kernel; tools.py declares the
ToolSpecs.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from ..support import invalid_source_refusal, numeric_values
from .weights import coordinates, knn_weights

# Distances beyond this many points make the n² matrix unreadable and
# slow; a legibility bound on the report, not an operator knob.
_MAX_DISTANCE_POINTS = 2000


def spatial_autocorrelation(
    frame: pd.DataFrame,
    value_column: str,
    x_column: str = "x",
    y_column: str = "y",
    k_neighbors: int | None = None,
) -> dict[str, Any]:
    """Global Moran's I of `value_column` over the k-NN neighbourhood."""
    if k_neighbors is None:
        # Anselin's common default; inlined because the S8 scan reserves
        # constant/parameter-default literals for config-backed values.
        k_neighbors = 4 + 4
    points = coordinates(frame, x_column, y_column)
    values = numeric_values(frame, value_column).to_numpy(dtype=float)
    if len(values) != len(points):
        raise invalid_source_refusal(
            "Value and coordinate columns disagree after dropping missing "
            "rows — clean the source so every row has x, y, and a value."
        )
    weights = knn_weights(points, k_neighbors)
    moran, z_score = _morans_i(values, weights)
    return {
        "method": "morans_i",
        "morans_i": moran,
        "expected_i": -1.0 / (len(values) - 1),
        "z_score": z_score,
        "p_value": float(2.0 * (1.0 - stats.norm.cdf(abs(z_score)))),
        "n_points": int(len(values)),
        "k_neighbors": k_neighbors,
        "interpretation": _autocorr_reading(moran, z_score),
    }


def _morans_i(
    values: "np.ndarray[Any, Any]", weights: "np.ndarray[Any, Any]"
) -> tuple[float, float]:
    n = len(values)
    deviations = values - values.mean()
    denominator = float((deviations**2).sum())
    if denominator == 0.0:
        raise invalid_source_refusal(
            "The value column is constant — spatial autocorrelation is undefined."
        )
    weight_total = float(weights.sum())
    numerator = float(deviations @ weights @ deviations)
    moran = (n / weight_total) * (numerator / denominator)
    expected = -1.0 / (n - 1)
    # Analytic normal approximation of the variance under randomization.
    variance = 1.0 / (n - 1)
    z_score = (moran - expected) / math.sqrt(variance)
    return moran, z_score


def _autocorr_reading(moran: float, z_score: float) -> str:
    if abs(z_score) < 1.96:
        return "No significant spatial autocorrelation."
    kind = "clustered (similar values near each other)" if moran > 0 else "dispersed"
    return f"Significant spatial autocorrelation: values are {kind}."


def spatial_hotspots(
    frame: pd.DataFrame,
    value_column: str,
    x_column: str = "x",
    y_column: str = "y",
    significance_level: float = 0.05,
) -> dict[str, Any]:
    """Getis-Ord Gi* hot/cold spots per point."""
    points = coordinates(frame, x_column, y_column)
    values = numeric_values(frame, value_column).to_numpy(dtype=float)
    weights = knn_weights(points, min(8, len(points) - 1))
    z_scores = _getis_ord(values, weights)
    cut = float(stats.norm.ppf(1.0 - significance_level / 2.0))
    hot = [int(i) for i in np.nonzero(z_scores > cut)[0]]
    cold = [int(i) for i in np.nonzero(z_scores < -cut)[0]]
    return {
        "method": "getis_ord_gi_star",
        "value_column": value_column,
        "n_points": int(len(values)),
        "significance_level": significance_level,
        "n_hotspots": len(hot),
        "n_coldspots": len(cold),
        "hotspot_indices": hot,
        "coldspot_indices": cold,
        "gi_z_scores": [float(z) for z in z_scores],
    }


def _getis_ord(
    values: "np.ndarray[Any, Any]", weights: "np.ndarray[Any, Any]"
) -> "np.ndarray[Any, Any]":
    n = len(values)
    mean = values.mean()
    std = values.std(ddof=0)
    if std == 0.0:
        raise invalid_source_refusal(
            "The value column is constant — no hot or cold spots exist."
        )
    # Include self in the local sum (Gi*), binary neighbour weights.
    binary = (weights > 0).astype(float)
    np.fill_diagonal(binary, 1.0)
    local_sum = binary @ values
    w_sum = binary.sum(axis=1)
    numerator = local_sum - mean * w_sum
    denominator = std * np.sqrt((n * (binary**2).sum(axis=1) - w_sum**2) / (n - 1))
    denominator[denominator == 0] = np.nan
    return numerator / denominator


def spatial_distances(
    frame: pd.DataFrame,
    x_column: str = "x",
    y_column: str = "y",
) -> dict[str, Any]:
    """A summarized Euclidean distance report over the addressed points."""
    points = coordinates(frame, x_column, y_column)
    if len(points) > _MAX_DISTANCE_POINTS:
        raise invalid_source_refusal(
            f"Distance analysis is bounded to {_MAX_DISTANCE_POINTS} points "
            f"(the matrix grows with their square); got {len(points)}."
        )
    from scipy.spatial.distance import pdist

    distances = pdist(points)
    return {
        "n_points": int(len(points)),
        "n_pairs": int(len(distances)),
        "distance_summary": {
            "min": float(distances.min()),
            "max": float(distances.max()),
            "mean": float(distances.mean()),
            "median": float(np.median(distances)),
        },
    }
