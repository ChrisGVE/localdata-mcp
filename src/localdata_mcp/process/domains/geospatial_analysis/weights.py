"""localdata_mcp/process/domains/geospatial_analysis/weights.py — shared kernel.

The k-nearest-neighbour spatial weights the autocorrelation and
hotspot statistics both need, plus the coordinate extraction they
share. A row-standardized binary k-NN weights matrix is the standard
neighbourhood definition for point patterns (Anselin); building it
once here keeps Moran's I and Getis-Ord Gi* reading from the same
neighbourhood. Neighbors: spatial_stats.py consumes.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..support import invalid_source_refusal, require_columns


def coordinates(
    frame: pd.DataFrame, x_column: str, y_column: str
) -> "np.ndarray[Any, Any]":
    """The (n, 2) coordinate array, missing rows dropped and refused
    when nothing usable remains."""
    require_columns(frame, x_column, y_column)
    coords = frame[[x_column, y_column]].apply(pd.to_numeric, errors="coerce").dropna()
    if coords.empty:
        raise invalid_source_refusal(
            f"Columns {x_column!r}/{y_column!r} carry no numeric coordinates."
        )
    return coords.to_numpy(dtype=float)


def knn_weights(points: "np.ndarray[Any, Any]", k: int) -> "np.ndarray[Any, Any]":
    """A row-standardized binary k-NN weights matrix (self excluded)."""
    n = len(points)
    if k >= n:
        raise invalid_source_refusal(
            f"k_neighbors={k} needs more than {k} points (got {n})."
        )
    from scipy.spatial import cKDTree

    tree = cKDTree(points)
    # k+1 because the nearest neighbour of a point is itself.
    _distances, indices = tree.query(points, k=k + 1)
    weights = np.zeros((n, n))
    for row, neighbours in enumerate(indices):
        for neighbour in neighbours:
            if neighbour != row:
                weights[row, neighbour] = 1.0
    row_sums = weights.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return weights / row_sums
