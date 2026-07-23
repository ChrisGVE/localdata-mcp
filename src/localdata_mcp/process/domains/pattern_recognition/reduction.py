"""localdata_mcp/process/domains/pattern_recognition/reduction.py — FR-301/308.

`reduce_dimensions`'s computation, re-authored from `main`'s
`DimensionalityReductionTransformer`: pca (default), tsne. The FR-308
closure lives here: every PCA result carries
`explained_variance_ratio` (the omission was #29's register row).
t-SNE results carry `trustworthiness` measured against the original
high-dimensional data — the S8 row-15f quality the battery gates on.
`seed` pins the stochastic embeddings (S3.3). Neighbors: matrices.py
preps input; tools.py declares the ToolSpec.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..support import invalid_source_refusal
from .matrices import construct, numeric_matrix

METHODS = ("pca", "tsne")

# The conventional embedding target: two components for inspection.
_DEFAULT_COMPONENTS = 2


def reduce_to_components(
    frame: pd.DataFrame,
    columns: list[str] | None = None,
    method: str = "pca",
    n_components: int = _DEFAULT_COMPONENTS,
    seed: int | None = None,
    algorithm_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The embedded coordinates plus the method's quality measures."""
    if method not in METHODS:
        raise invalid_source_refusal(
            f"Unknown method {method!r} — one of {list(METHODS)}."
        )
    matrix, names = numeric_matrix(frame, columns)
    if not (1 <= n_components <= min(matrix.shape)):
        raise invalid_source_refusal(
            f"n_components={n_components} outside 1..{min(matrix.shape)} for this data."
        )
    result: dict[str, Any] = {
        "method": method,
        "columns": names,
        "n_samples": int(matrix.shape[0]),
        "n_components": n_components,
    }
    if method == "pca":
        from sklearn.decomposition import PCA

        model = construct(PCA, algorithm_params, seed, n_components=n_components)
        embedded = model.fit_transform(matrix)
        # FR-308: explained variance on EVERY successful PCA result.
        result["explained_variance_ratio"] = [
            float(ratio) for ratio in model.explained_variance_ratio_
        ]
    else:
        from sklearn.manifold import TSNE, trustworthiness

        # Perplexity must stay below the sample count; scale the
        # default down for small fixtures rather than crash opaquely.
        perplexity = min(30.0, (matrix.shape[0] - 1) / 3.0)
        model = construct(
            TSNE,
            algorithm_params,
            seed,
            n_components=n_components,
            perplexity=perplexity,
        )
        embedded = model.fit_transform(matrix)
        result["trustworthiness"] = float(trustworthiness(matrix, embedded))
    result["components"] = [[float(value) for value in row] for row in embedded]
    return result
