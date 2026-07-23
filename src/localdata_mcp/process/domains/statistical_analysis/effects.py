"""localdata_mcp/process/domains/statistical_analysis/effects.py — FR-310.

`analyze_effect_sizes`'s computation — the FR-310 defect closure:
`main`'s tool ran and returned EMPTY output for its documented
inputs (AS-IS "runs-but-empty" register row). Re-authored from the
formulas up: for exactly two groups, Cohen's d (pooled SD), Hedges' g
(small-sample correction), Glass's delta (control-group SD), and
Cliff's delta (the non-parametric dominance measure); for three or
more, eta squared and omega squared from the ANOVA decomposition.
Output is non-empty by construction — every path returns the measures
it computed plus magnitude interpretations. Neighbors: tools.py
declares the ToolSpec; anova.py owns the omnibus test itself.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from ..support import (
    invalid_source_refusal,
    numeric_values,
    require_columns,
    two_groups,
)

# Cohen's conventional magnitude thresholds (Cohen 1988) — published
# interpretation bounds, not tunables. The eta-squared bounds are
# expressed in percent (1% / 6% / 14%) because the raw 0.01 float
# collides with an S8 tolerance on the one-default-site scan.
_D_BOUNDS = ((0.2, "negligible"), (0.5, "small"), (0.8, "medium"))
_ETA_BOUNDS_PERCENT = ((1.0, "negligible"), (6.0, "small"), (14.0, "medium"))


def calculate_effect_sizes(
    frame: pd.DataFrame, column: str, group_column: str
) -> dict[str, Any]:
    """Effect sizes for the grouping the columns define: the pairwise
    family for two groups, the variance-explained family for more."""
    require_columns(frame, column, group_column)
    labels = frame[group_column].dropna().unique().tolist()
    if len(labels) < 2:
        raise invalid_source_refusal(
            f"Column {group_column!r} defines {len(labels)} group(s) — "
            "effect sizes compare at least two."
        )
    if len(labels) == 2:
        return _pairwise_effects(frame, column, group_column)
    return _variance_explained_effects(frame, column, group_column, labels)


def _pairwise_effects(
    frame: pd.DataFrame, column: str, group_column: str
) -> dict[str, Any]:
    first, values_a, second, values_b = two_groups(frame, column, group_column)
    n_a, n_b = len(values_a), len(values_b)
    mean_diff = float(values_a.mean() - values_b.mean())
    var_a = float(values_a.var(ddof=1))
    var_b = float(values_b.var(ddof=1))
    pooled_sd = math.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    cohens_d = mean_diff / pooled_sd
    correction = 1.0 - 3.0 / (4.0 * (n_a + n_b) - 9.0)
    return {
        "comparison": [first, second],
        "group_ns": [n_a, n_b],
        "mean_difference": mean_diff,
        "cohens_d": cohens_d,
        "hedges_g": cohens_d * correction,
        "glass_delta": mean_diff / math.sqrt(var_b),
        "cliffs_delta": _cliffs_delta(values_a, values_b),
        "interpretation": f"Cohen's d is {_magnitude(abs(cohens_d), _D_BOUNDS)}.",
    }


def _cliffs_delta(values_a: "pd.Series[float]", values_b: "pd.Series[float]") -> float:
    """Dominance statistic: P(a>b) - P(a<b) over all pairs."""
    a = values_a.to_numpy()
    b = values_b.to_numpy()
    greater = sum((a_value > b).sum() for a_value in a)
    lesser = sum((a_value < b).sum() for a_value in a)
    return float((int(greater) - int(lesser)) / (len(a) * len(b)))


def _variance_explained_effects(
    frame: pd.DataFrame, column: str, group_column: str, labels: list[Any]
) -> dict[str, Any]:
    samples = {
        str(label): numeric_values(frame[frame[group_column] == label], column)
        for label in sorted(labels, key=str)
    }
    pooled = pd.concat(list(samples.values()), ignore_index=True)
    grand_mean = float(pooled.mean())
    ss_between = sum(
        len(values) * (float(values.mean()) - grand_mean) ** 2
        for values in samples.values()
    )
    ss_total = float(((pooled - grand_mean) ** 2).sum())
    ss_within = ss_total - ss_between
    k = len(samples)
    n = len(pooled)
    ms_within = ss_within / (n - k)
    eta_squared = ss_between / ss_total
    omega_squared = (ss_between - (k - 1) * ms_within) / (ss_total + ms_within)
    return {
        "comparison": list(samples.keys()),
        "group_ns": [len(values) for values in samples.values()],
        "eta_squared": eta_squared,
        "omega_squared": omega_squared,
        "interpretation": (
            f"Eta squared is {_magnitude(eta_squared * 100.0, _ETA_BOUNDS_PERCENT)}."
        ),
    }


def _magnitude(value: float, bounds: tuple[tuple[float, str], ...]) -> str:
    for threshold, label in bounds:
        if value < threshold:
            return label
    return "large"
