"""localdata_mcp/process/domains/pattern_recognition/matrices.py — shared input prep.

The numeric-matrix extraction the three estimator tools share
(`main`'s `_numeric_matrix` rule): the named columns — or every
numeric column when none are named — coerced to float with incomplete
rows dropped, refused when nothing numeric remains. Estimator
construction mirrors regression's FR-306 discipline: caller
`algorithm_params` land at construction, plus the stochastic tools'
`seed` mapped onto `random_state` (S3.3 caller-controllable
determinism). Neighbors: clustering/anomalies/reduction.py consume.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..support import invalid_source_refusal, require_columns


def numeric_matrix(
    frame: pd.DataFrame, columns: list[str] | None
) -> tuple["np.ndarray[Any, Any]", list[str]]:
    """(matrix, column names) — the named or every-numeric selection."""
    if columns:
        require_columns(frame, *columns)
        selected = frame[columns]
    else:
        selected = frame.select_dtypes(include=[np.number])
        if selected.shape[1] == 0:
            raise invalid_source_refusal(
                "No numeric columns in the addressed data — name columns= "
                f"explicitly (available: {[str(c) for c in frame.columns]})."
            )
    coerced = selected.apply(pd.to_numeric, errors="coerce").dropna()
    if coerced.empty:
        raise invalid_source_refusal(
            "No complete numeric rows remain after dropping missing values."
        )
    return coerced.to_numpy(dtype=float), [str(name) for name in coerced.columns]


def construct(
    constructor: Any,
    algorithm_params: dict[str, Any] | None,
    seed: int | None = None,
    **fixed: Any,
) -> Any:
    """One estimator with caller params at construction (the FR-306
    discipline) and seed mapped onto random_state when supplied."""
    params = dict(fixed)
    params.update(algorithm_params or {})
    if seed is not None:
        params["random_state"] = seed
    try:
        return constructor(**params)
    except TypeError as failure:
        raise invalid_source_refusal(
            f"algorithm_params not accepted by {constructor.__name__}: {failure}"
        ) from None
