"""localdata_mcp/process/domains/sampling_estimation/estimation.py — FR-301.

`bootstrap_statistic` and `bayesian_estimate`'s computations,
re-authored from `main`'s transformer pair. Bootstrap: percentile
confidence intervals (Efron & Tibshirani) over seeded resamples —
the resample count and the interval-coverage level arrive from the
caller or the operator-configured process defaults (fetched at the
tool layer through the guard seam, never read from NX-2 here).
Bayesian: the conjugate normal model with a noninformative prior,
whose posterior for the mean is the published Student-t result — the
battery pins the credible interval against scipy's t distribution
directly. Neighbors: tools.py declares the ToolSpecs; monte_carlo.py
is the sibling.
"""

from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np
import pandas as pd

from ..support import invalid_source_refusal, numeric_values

STATISTICS: dict[str, Callable[["np.ndarray[Any, Any]"], float]] = {
    "mean": lambda values: float(np.mean(values)),
    "median": lambda values: float(np.median(values)),
    "std": lambda values: float(np.std(values, ddof=1)),
    "var": lambda values: float(np.var(values, ddof=1)),
}

PRIORS = ("normal",)


def bootstrap(
    frame: pd.DataFrame,
    column: str,
    statistic: str = "mean",
    resamples: int | None = None,
    confidence_level: float | None = None,
    seed: int | None = None,
    default_resamples: int = 0,
    default_confidence: float = 0.0,
) -> dict[str, Any]:
    """The percentile-bootstrap estimate and interval for `statistic`."""
    if statistic not in STATISTICS:
        raise invalid_source_refusal(
            f"Unknown statistic {statistic!r} — one of {list(STATISTICS)}."
        )
    # The interval coverage arrives from the caller or the operator-
    # configured process default (the guard seam), never an inline literal.
    if confidence_level is None:
        confidence_level = default_confidence
    if not (0.0 < confidence_level < 1.0):
        raise invalid_source_refusal("confidence_level must be inside (0, 1).")
    count = resamples if resamples is not None else default_resamples
    if count < 1:
        raise invalid_source_refusal("resamples must be at least 1.")
    values = numeric_values(frame, column).to_numpy(dtype=float)
    compute = STATISTICS[statistic]
    rng = np.random.default_rng(seed)
    draws = np.array(
        [
            compute(values[rng.integers(0, len(values), size=len(values))])
            for _ in range(count)
        ]
    )
    tail = (1.0 - confidence_level) / 2.0
    return {
        "statistic": statistic,
        "estimate": compute(values),
        "n": int(len(values)),
        "resamples": count,
        "confidence_level": confidence_level,
        "confidence_interval": {
            "lower": float(np.quantile(draws, tail)),
            "upper": float(np.quantile(draws, 1.0 - tail)),
        },
        "standard_error": float(np.std(draws, ddof=1)),
    }


def bayesian_posterior(
    frame: pd.DataFrame,
    column: str,
    prior_distribution: str = "normal",
    credible_level: float | None = None,
    default_credible: float = 0.0,
) -> dict[str, Any]:
    """The conjugate-normal posterior of the mean under a
    noninformative prior: Student-t with n-1 degrees of freedom."""
    from scipy import stats

    if prior_distribution not in PRIORS:
        raise invalid_source_refusal(
            f"Unknown prior_distribution {prior_distribution!r} — one of "
            f"{list(PRIORS)}."
        )
    # The credible-interval coverage arrives from the caller or the
    # operator-configured process default (the guard seam).
    if credible_level is None:
        credible_level = default_credible
    if not (0.0 < credible_level < 1.0):
        raise invalid_source_refusal("credible_level must be inside (0, 1).")
    values = numeric_values(frame, column).to_numpy(dtype=float)
    if len(values) < 2:
        raise invalid_source_refusal(
            f"Column {column!r} needs at least two values for a posterior."
        )
    n = len(values)
    mean = float(np.mean(values))
    scale = float(np.std(values, ddof=1) / math.sqrt(n))
    lower, upper = stats.t.interval(credible_level, df=n - 1, loc=mean, scale=scale)
    return {
        "prior_distribution": prior_distribution,
        "n": n,
        "posterior_mean": mean,
        "posterior_scale": scale,
        "degrees_of_freedom": n - 1,
        "credible_level": credible_level,
        "credible_interval": {"lower": float(lower), "upper": float(upper)},
    }
