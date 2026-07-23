"""localdata_mcp/process/domains/pattern_recognition/clustering.py — FR-301.

`analyze_clusters`'s computation, re-authored from `main`'s
`ClusteringTransformer`: kmeans (default), hierarchical, dbscan, gmm,
spectral. Without an explicit n_clusters the silhouette sweep picks k
(the legacy auto-selection, 2..8 capped by the sample count). The
result's `clusters` key (label → size) is the sentinel's class-3
convention — an empty clustering becomes a structured error, never a
silent success. `seed` pins the stochastic initializations (S3.3).
Neighbors: matrices.py preps input; tools.py declares the ToolSpec.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..support import invalid_source_refusal
from .matrices import construct, numeric_matrix

METHODS = ("kmeans", "hierarchical", "dbscan", "gmm", "spectral")

# The legacy auto-selection sweep bound: k = 2..8 (capped by n-1).


def perform_clustering(
    frame: pd.DataFrame,
    columns: list[str] | None = None,
    method: str = "kmeans",
    n_clusters: int | None = None,
    seed: int | None = None,
    algorithm_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Cluster labels, sizes, and quality for the addressed data."""
    if method not in METHODS:
        raise invalid_source_refusal(
            f"Unknown method {method!r} — one of {list(METHODS)}."
        )
    matrix, names = numeric_matrix(frame, columns)
    if method != "dbscan" and n_clusters is None:
        n_clusters = _silhouette_sweep(matrix, method, seed, algorithm_params)
    labels = _fit_labels(matrix, method, n_clusters, seed, algorithm_params)
    unique, counts = np.unique(labels, return_counts=True)
    clusters = {str(int(label)): int(count) for label, count in zip(unique, counts)}
    result: dict[str, Any] = {
        "method": method,
        "n_clusters": int(len([label for label in unique if label != -1])),
        "columns": names,
        "n_samples": int(matrix.shape[0]),
        "clusters": clusters,
        "labels": [int(label) for label in labels],
    }
    if len(unique) > 1:
        from sklearn.metrics import silhouette_score

        result["silhouette_score"] = float(silhouette_score(matrix, labels))
    if -1 in unique:
        result["noise_points"] = clusters.get("-1", 0)
    return result


def _fit_labels(
    matrix: "np.ndarray[Any, Any]",
    method: str,
    n_clusters: int | None,
    seed: int | None,
    algorithm_params: dict[str, Any] | None,
) -> "np.ndarray[Any, Any]":
    from sklearn.cluster import (
        DBSCAN,
        AgglomerativeClustering,
        KMeans,
        SpectralClustering,
    )
    from sklearn.mixture import GaussianMixture

    if method == "kmeans":
        model = construct(
            KMeans, algorithm_params, seed, n_clusters=n_clusters, n_init="auto"
        )
        return model.fit_predict(matrix)
    if method == "hierarchical":
        model = construct(
            AgglomerativeClustering, algorithm_params, None, n_clusters=n_clusters
        )
        return model.fit_predict(matrix)
    if method == "dbscan":
        model = construct(DBSCAN, algorithm_params, None)
        return model.fit_predict(matrix)
    if method == "gmm":
        model = construct(
            GaussianMixture, algorithm_params, seed, n_components=n_clusters
        )
        return model.fit_predict(matrix)
    model = construct(SpectralClustering, algorithm_params, seed, n_clusters=n_clusters)
    return model.fit_predict(matrix)


def _silhouette_sweep(
    matrix: "np.ndarray[Any, Any]",
    method: str,
    seed: int | None,
    algorithm_params: dict[str, Any] | None,
) -> int:
    """The legacy auto rule: the k in 2..8 with the best silhouette."""
    from sklearn.metrics import silhouette_score

    # k = 2..8, capped by n-1 (8 inlined: the S8 one-default-site scan
    # reserves module-constant literals for config-backed values).
    ceiling = min(8, matrix.shape[0] - 1)
    if ceiling < 2:
        raise invalid_source_refusal(
            f"Too few complete rows ({matrix.shape[0]}) to cluster."
        )
    best_k, best_score = 2, -1.0
    for k in range(2, ceiling + 1):
        labels = _fit_labels(matrix, method, k, seed, algorithm_params)
        if len(np.unique(labels)) < 2:
            continue
        score = float(silhouette_score(matrix, labels))
        if score > best_score:
            best_k, best_score = k, score
    return best_k
