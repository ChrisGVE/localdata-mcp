"""localdata_mcp/process/domains/statistical_analysis/ab_test.py — FR-301.

`analyze_ab_test`'s computation, carried by name from `main` (where it
lived in the business-intelligence package; v3 homes it with the
statistical family per PRD S3.3's family listing). Two variants,
one metric: a binary metric (or test_type='proportion') runs the
two-proportion z-test (statsmodels); a continuous one runs Welch's
t-test, or Mann-Whitney on request. The significance level arrives
from the caller or the operator-configured process default (the guard
seam), never an inline literal (CR-009). The result names the winner
and the relative lift so the agent can read the verdict without a
second computation. Neighbors: tools.py declares the ToolSpec;
support.py owns the exactly-two-group split.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from scipy import stats

from ..support import invalid_source_refusal, two_groups

TEST_TYPES = ("auto", "proportion", "t_test", "mann_whitney")


def perform_ab_test(
    frame: pd.DataFrame,
    metric_column: str,
    variant_column: str,
    test_type: str = "auto",
    alpha: float | None = None,
    alternative: str = "two-sided",
    default_alpha: float = 0.0,
) -> dict[str, Any]:
    """The A/B verdict between the exactly-two variants."""
    if test_type not in TEST_TYPES:
        raise invalid_source_refusal(
            f"Unknown test_type {test_type!r} — one of {list(TEST_TYPES)}."
        )
    effective_alpha = alpha if alpha is not None else default_alpha
    first, values_a, second, values_b = two_groups(frame, metric_column, variant_column)
    if test_type == "auto":
        test_type = "proportion" if _binary(values_a, values_b) else "t_test"
    if test_type == "proportion":
        statistic, p_value = _proportion_test(values_a, values_b, alternative)
    elif test_type == "t_test":
        outcome = stats.ttest_ind(
            values_a, values_b, equal_var=False, alternative=alternative
        )
        statistic, p_value = float(outcome.statistic), float(outcome.pvalue)
    else:
        outcome = stats.mannwhitneyu(values_a, values_b, alternative=alternative)
        statistic, p_value = float(outcome.statistic), float(outcome.pvalue)
    mean_a, mean_b = float(values_a.mean()), float(values_b.mean())
    significant = bool(p_value < effective_alpha)
    winner = (first if mean_a > mean_b else second) if significant else None
    result: dict[str, Any] = {
        "test_type": test_type,
        "statistic": statistic,
        "p_value": p_value,
        "alpha": effective_alpha,
        "significant": significant,
        "variants": {
            first: {"n": int(len(values_a)), "mean": mean_a},
            second: {"n": int(len(values_b)), "mean": mean_b},
        },
        "winner": winner,
        "interpretation": (
            f"Variant {winner!r} outperforms significantly."
            if winner is not None
            else "No significant difference between the variants."
        ),
    }
    if mean_b != 0.0:
        result["relative_lift"] = (mean_a - mean_b) / mean_b
    return result


def _binary(values_a: "pd.Series[float]", values_b: "pd.Series[float]") -> bool:
    observed = set(values_a.unique().tolist()) | set(values_b.unique().tolist())
    return observed <= {0, 1, 0.0, 1.0, True, False}


def _proportion_test(
    values_a: "pd.Series[float]", values_b: "pd.Series[float]", alternative: str
) -> tuple[float, float]:
    """Two-proportion z-test; refuses a non-binary metric."""
    if not _binary(values_a, values_b):
        raise invalid_source_refusal(
            "test_type='proportion' needs a binary (0/1) metric column — "
            "use t_test or mann_whitney for a continuous metric."
        )
    from statsmodels.stats.proportion import proportions_ztest

    # statsmodels spells the one-sided alternatives 'larger'/'smaller'.
    spelled = {"two-sided": "two-sided", "greater": "larger", "less": "smaller"}
    statistic, p_value = proportions_ztest(
        count=[int(values_a.sum()), int(values_b.sum())],
        nobs=[len(values_a), len(values_b)],
        alternative=spelled.get(alternative, alternative),
    )
    return float(statistic), float(p_value)
