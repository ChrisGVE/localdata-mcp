"""localdata_mcp/process/domains/statistical_analysis/anova.py — FR-301.

`analyze_anova`'s computation: one-way ANOVA (scipy `f_oneway`) with
the effect size (eta squared from the sums of squares) and a Tukey
HSD post-hoc pass (statsmodels) when the omnibus test is significant
— `main`'s `ANOVAAnalysisTransformer` algorithm without the
transformer scaffolding. The significance level arrives from the
caller or the operator-configured process default (the guard seam),
never an inline literal (CR-009). The per-group summary lands under
the `groups` key deliberately: an empty group is the sentinel's
class-3 degenerate-shape signal (E7.2), so a hollow grouping becomes a
structured error, never a silent success. Neighbors: tools.py
declares the ToolSpec; hypothesis.py covers the two-group case.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from scipy import stats

from ..support import invalid_source_refusal, numeric_values, require_columns


def perform_anova(
    frame: pd.DataFrame,
    dependent_var: str,
    group_var: str,
    alpha: float | None = None,
    default_alpha: float = 0.0,
) -> dict[str, Any]:
    """One-way ANOVA across every group of `group_var`."""
    require_columns(frame, dependent_var, group_var)
    effective_alpha = alpha if alpha is not None else default_alpha
    labels = sorted(frame[group_var].dropna().unique().tolist(), key=str)
    if len(labels) < 2:
        raise invalid_source_refusal(
            f"Column {group_var!r} defines {len(labels)} group(s) — "
            "ANOVA needs at least two."
        )
    samples = {
        str(label): numeric_values(frame[frame[group_var] == label], dependent_var)
        for label in labels
    }
    outcome = stats.f_oneway(*samples.values())
    significant = bool(outcome.pvalue < effective_alpha)
    result: dict[str, Any] = {
        "anova_type": "one_way",
        "f_statistic": float(outcome.statistic),
        "p_value": float(outcome.pvalue),
        "alpha": effective_alpha,
        "significant": significant,
        "eta_squared": _eta_squared(samples),
        "groups": {
            label: {
                "n": int(len(values)),
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)),
            }
            for label, values in samples.items()
        },
        "interpretation": (
            "The group means differ significantly."
            if significant
            else "No significant difference between the group means."
        ),
    }
    if significant:
        result["post_hoc"] = _tukey(samples)
    return result


def _eta_squared(samples: dict[str, "pd.Series[float]"]) -> float:
    """Effect size from the ANOVA decomposition: SS_between / SS_total."""
    pooled = pd.concat(list(samples.values()), ignore_index=True)
    grand_mean = float(pooled.mean())
    ss_between = sum(
        len(values) * (float(values.mean()) - grand_mean) ** 2
        for values in samples.values()
    )
    ss_total = float(((pooled - grand_mean) ** 2).sum())
    return ss_between / ss_total


def _tukey(samples: dict[str, "pd.Series[float]"]) -> list[dict[str, Any]]:
    """Tukey HSD pairwise comparisons (statsmodels)."""
    from statsmodels.stats.multicomp import pairwise_tukeyhsd

    values = pd.concat(list(samples.values()), ignore_index=True)
    labels = [label for label, group in samples.items() for _ in range(len(group))]
    tukey = pairwise_tukeyhsd(values, labels)
    rows = tukey.summary().data[1:]  # first row is the header
    return [
        {
            "group_a": str(row[0]),
            "group_b": str(row[1]),
            "mean_difference": float(row[2]),
            "p_value": float(row[3]),
            "reject_null": bool(row[6]),
        }
        for row in rows
    ]
