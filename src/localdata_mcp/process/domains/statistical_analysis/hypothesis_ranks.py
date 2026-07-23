"""localdata_mcp/process/domains/statistical_analysis/hypothesis_ranks.py — FR-301.

The rank-based and association half of `analyze_hypothesis_test`'s
dispatch table: Mann-Whitney U, Wilcoxon signed-rank, chi-square
independence, Shapiro-Wilk normality, and Pearson correlation.
hypothesis.py owns the entry point, auto selection, and the t-family;
this sibling exists purely to keep each file inside the codesize
discipline — every function here has the same dispatcher signature.
Neighbors: support.py owns the paired/grouped extraction both halves
share.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from scipy import stats

from ..support import (
    comparison_groups,
    invalid_source_refusal,
    numeric_values,
    paired_numeric,
    require_columns,
)

# Shapiro's implementation bounds (below 3 undefined; above 5000 the
# p-value is unreliable per scipy docs) — screen bounds, not tunables.
SHAPIRO_MIN = 3
SHAPIRO_MAX = 5000


def mann_whitney(
    frame: pd.DataFrame,
    column: str | None,
    second_column: str | None,
    group_column: str | None,
    popmean: float,
    alpha: float,
    alternative: str,
) -> dict[str, Any]:
    first, values_a, second, values_b = comparison_groups(frame, column, group_column)
    outcome = stats.mannwhitneyu(values_a, values_b, alternative=alternative)
    return {
        "statistic": float(outcome.statistic),
        "p_value": float(outcome.pvalue),
        "group_labels": [first, second],
        "group_ns": [int(len(values_a)), int(len(values_b))],
        "alternative": alternative,
        "interpretation": "Mann-Whitney U rank test between the two groups.",
    }


def wilcoxon(
    frame: pd.DataFrame,
    column: str | None,
    second_column: str | None,
    group_column: str | None,
    popmean: float,
    alpha: float,
    alternative: str,
) -> dict[str, Any]:
    before, after = paired_numeric(frame, column, second_column)
    outcome = stats.wilcoxon(before, after, alternative=alternative)
    return {
        "statistic": float(outcome.statistic),
        "p_value": float(outcome.pvalue),
        "n_pairs": int(len(before)),
        "alternative": alternative,
        "interpretation": "Wilcoxon signed-rank test on the paired columns.",
    }


def chi2(
    frame: pd.DataFrame,
    column: str | None,
    second_column: str | None,
    group_column: str | None,
    popmean: float,
    alpha: float,
    alternative: str,
) -> dict[str, Any]:
    if column is None or second_column is None:
        raise invalid_source_refusal("chi2 needs column= and second_column=.")
    require_columns(frame, column, second_column)
    table = pd.crosstab(frame[column], frame[second_column])
    outcome = stats.chi2_contingency(table)
    return {
        "statistic": float(outcome.statistic),
        "p_value": float(outcome.pvalue),
        "degrees_of_freedom": int(outcome.dof),
        "contingency_shape": [int(table.shape[0]), int(table.shape[1])],
        "interpretation": "Chi-square test of independence on the crosstab.",
    }


def normality(
    frame: pd.DataFrame,
    column: str | None,
    second_column: str | None,
    group_column: str | None,
    popmean: float,
    alpha: float,
    alternative: str,
) -> dict[str, Any]:
    if column is None:
        raise invalid_source_refusal("normality needs column=.")
    values = numeric_values(frame, column)
    if not (SHAPIRO_MIN <= len(values) <= SHAPIRO_MAX):
        raise invalid_source_refusal(
            f"Shapiro-Wilk needs between {SHAPIRO_MIN} and {SHAPIRO_MAX} "
            f"values (column {column!r} has {len(values)})."
        )
    outcome = stats.shapiro(values)
    return {
        "statistic": float(outcome.statistic),
        "p_value": float(outcome.pvalue),
        "n": int(len(values)),
        "interpretation": (
            "Shapiro-Wilk normality test — a small p rejects normality."
        ),
    }


def correlation(
    frame: pd.DataFrame,
    column: str | None,
    second_column: str | None,
    group_column: str | None,
    popmean: float,
    alpha: float,
    alternative: str,
) -> dict[str, Any]:
    if column is None or second_column is None:
        raise invalid_source_refusal("correlation needs column= and second_column=.")
    left, right = paired_numeric(frame, column, second_column)
    outcome = stats.pearsonr(left, right, alternative=alternative)
    return {
        "statistic": float(outcome.statistic),
        "p_value": float(outcome.pvalue),
        "n": int(len(left)),
        "alternative": alternative,
        "interpretation": "Pearson correlation between the two columns.",
    }
