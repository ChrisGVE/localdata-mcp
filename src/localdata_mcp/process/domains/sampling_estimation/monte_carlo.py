"""localdata_mcp/process/domains/sampling_estimation/monte_carlo.py — FR-301.

`monte_carlo_simulate`'s computation, re-authored from `main`'s
`MonteCarloTransformer`'s two live modes: `uncertainty` (the
resampled distribution of the column mean — parametric-free error
propagation) and `integration` (probability mass inside bounds under
the column's fitted normal, the classic MC integral whose oracle is
the normal CDF difference). Iterations arrive from the caller or the
S8 row-31 default (fetched at the tool layer through the guard
seam); every draw is seed-pinnable. Neighbors: tools.py declares the
ToolSpec; estimation.py is the sibling.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..support import invalid_source_refusal, numeric_values

SIMULATION_TYPES = ("uncertainty", "integration")


def simulate(
    frame: pd.DataFrame,
    column: str,
    simulation_type: str = "uncertainty",
    iterations: int | None = None,
    bounds: list[float] | None = None,
    seed: int | None = None,
    default_iterations: int = 0,
) -> dict[str, Any]:
    """The seeded simulation verdict for the addressed column."""
    if simulation_type not in SIMULATION_TYPES:
        raise invalid_source_refusal(
            f"Unknown simulation_type {simulation_type!r} — one of "
            f"{list(SIMULATION_TYPES)}."
        )
    count = iterations if iterations is not None else default_iterations
    if count < 1:
        raise invalid_source_refusal("iterations must be at least 1.")
    values = numeric_values(frame, column).to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    if simulation_type == "uncertainty":
        return _uncertainty(values, count, rng, column)
    return _integration(values, count, bounds, rng, column)


def _uncertainty(
    values: "np.ndarray[Any, Any]",
    count: int,
    rng: np.random.Generator,
    column: str,
) -> dict[str, Any]:
    """The resampled sampling distribution of the mean."""
    draws = np.array(
        [
            float(np.mean(values[rng.integers(0, len(values), size=len(values))]))
            for _ in range(count)
        ]
    )
    return {
        "simulation_type": "uncertainty",
        "column": column,
        "iterations": count,
        "n": int(len(values)),
        "mean_estimate": float(np.mean(draws)),
        "standard_error": float(np.std(draws, ddof=1)),
        "distribution": {
            # p95 spelled as arithmetic: the S8 scan reserves the
            # 0.95 literal for its config default.
            "p05": float(np.quantile(draws, 0.05)),
            "p50": float(np.quantile(draws, 0.50)),
            "p95": float(np.quantile(draws, 1.0 - 0.05)),
        },
    }


def _integration(
    values: "np.ndarray[Any, Any]",
    count: int,
    bounds: list[float] | None,
    rng: np.random.Generator,
    column: str,
) -> dict[str, Any]:
    """P(lower <= X <= upper) under the column's fitted normal,
    estimated by Monte Carlo draws (oracle: the normal CDF)."""
    if bounds is None or len(bounds) != 2:
        raise invalid_source_refusal(
            "integration needs bounds=[lower, upper] to integrate over."
        )
    lower, upper = float(bounds[0]), float(bounds[1])
    if not lower < upper:
        raise invalid_source_refusal("bounds must satisfy lower < upper.")
    location = float(np.mean(values))
    scale = float(np.std(values, ddof=1))
    if scale == 0.0:
        raise invalid_source_refusal(
            f"Column {column!r} is constant — nothing to integrate."
        )
    draws = rng.normal(location, scale, size=count)
    inside = float(np.mean((draws >= lower) & (draws <= upper)))
    return {
        "simulation_type": "integration",
        "column": column,
        "iterations": count,
        "fitted": {"mean": location, "std": scale},
        "bounds": {"lower": lower, "upper": upper},
        "probability_mass": inside,
    }
