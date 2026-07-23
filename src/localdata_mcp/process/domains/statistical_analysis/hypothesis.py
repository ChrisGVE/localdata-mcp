"""localdata_mcp/process/domains/statistical_analysis/hypothesis.py — FR-301.

`analyze_hypothesis_test`'s computation, re-authored from `main`'s
`HypothesisTestingTransformer` (the algorithmic choices survive; the
sklearn-transformer scaffolding does not — DR GP2 carries tool names,
not class hierarchies). The test vocabulary is `main`'s: auto,
ttest_1samp, ttest_ind, ttest_rel, mann_whitney, wilcoxon, chi2,
normality, correlation. `auto` answers the question the supplied
columns pose: a group column → two-sample comparison (parametric when
both groups pass Shapiro, else Mann-Whitney — the legacy selection
rule); two numeric columns → correlation; one column → normality.
Every result is a plain dict the NX-7 envelope shapes; degenerate
outputs (NaN statistics on constant data) trip the shared sentinel,
never a silent success. Neighbors: hypothesis_ranks.py carries the
rank/association half of the dispatch table (codesize split);
support.py owns column/group handling; tools.py declares the ToolSpec.
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
    two_groups,
)
from . import hypothesis_ranks as ranks

TEST_TYPES = (
    "auto",
    "ttest_1samp",
    "ttest_ind",
    "ttest_rel",
    "mann_whitney",
    "wilcoxon",
    "chi2",
    "normality",
    "correlation",
)


def run_hypothesis_test(
    frame: pd.DataFrame,
    test_type: str = "auto",
    column: str | None = None,
    second_column: str | None = None,
    group_column: str | None = None,
    popmean: float = 0.0,
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> dict[str, Any]:
    """The dispatched test result as a plain dict."""
    if test_type not in TEST_TYPES:
        raise invalid_source_refusal(
            f"Unknown test_type {test_type!r} — one of {list(TEST_TYPES)}."
        )
    if test_type == "auto":
        test_type = _select_test(frame, column, second_column, group_column)
    result = _DISPATCH[test_type](
        frame, column, second_column, group_column, popmean, alpha, alternative
    )
    result["test_type"] = test_type
    result["alpha"] = alpha
    result["significant"] = bool(result["p_value"] < alpha)
    return result


def _select_test(
    frame: pd.DataFrame,
    column: str | None,
    second_column: str | None,
    group_column: str | None,
) -> str:
    """The legacy auto rule: the supplied columns pose the question."""
    if group_column is not None and column is not None:
        first, values_a, second, values_b = two_groups(frame, column, group_column)
        if _both_normal(values_a, values_b):
            return "ttest_ind"
        return "mann_whitney"
    if column is not None and second_column is not None:
        return "correlation"
    if column is not None:
        return "normality"
    raise invalid_source_refusal(
        "test_type='auto' needs column= (plus group_column= for a group "
        "comparison, or second_column= for correlation) to select a test."
    )


def _both_normal(values_a: "pd.Series[float]", values_b: "pd.Series[float]") -> bool:
    """The legacy parametric screen: Shapiro on both groups at the
    conventional 5% level (a selection heuristic, not the verdict)."""
    for values in (values_a, values_b):
        n = len(values)
        if not (ranks.SHAPIRO_MIN <= n <= ranks.SHAPIRO_MAX):
            return False
        if stats.shapiro(values).pvalue < 0.05:
            return False
    return True


def _ttest_1samp(
    frame: pd.DataFrame,
    column: str | None,
    second_column: str | None,
    group_column: str | None,
    popmean: float,
    alpha: float,
    alternative: str,
) -> dict[str, Any]:
    if column is None:
        raise invalid_source_refusal("ttest_1samp needs column=.")
    values = numeric_values(frame, column)
    outcome = stats.ttest_1samp(values, popmean, alternative=alternative)
    return {
        "statistic": float(outcome.statistic),
        "p_value": float(outcome.pvalue),
        "n": int(len(values)),
        "popmean": popmean,
        "sample_mean": float(values.mean()),
        "alternative": alternative,
        "interpretation": "One-sample t-test of the column mean against popmean.",
    }


def _ttest_ind(
    frame: pd.DataFrame,
    column: str | None,
    second_column: str | None,
    group_column: str | None,
    popmean: float,
    alpha: float,
    alternative: str,
) -> dict[str, Any]:
    first, values_a, second, values_b = comparison_groups(frame, column, group_column)
    outcome = stats.ttest_ind(values_a, values_b, alternative=alternative)
    return {
        "statistic": float(outcome.statistic),
        "p_value": float(outcome.pvalue),
        "group_labels": [first, second],
        "group_ns": [int(len(values_a)), int(len(values_b))],
        "group_means": [float(values_a.mean()), float(values_b.mean())],
        "alternative": alternative,
        "interpretation": "Independent two-sample t-test between the two groups.",
    }


def _ttest_rel(
    frame: pd.DataFrame,
    column: str | None,
    second_column: str | None,
    group_column: str | None,
    popmean: float,
    alpha: float,
    alternative: str,
) -> dict[str, Any]:
    before, after = paired_numeric(frame, column, second_column)
    outcome = stats.ttest_rel(before, after, alternative=alternative)
    return {
        "statistic": float(outcome.statistic),
        "p_value": float(outcome.pvalue),
        "n_pairs": int(len(before)),
        "mean_difference": float((before - after).mean()),
        "alternative": alternative,
        "interpretation": "Paired t-test between the two columns row-wise.",
    }


_DISPATCH = {
    "ttest_1samp": _ttest_1samp,
    "ttest_ind": _ttest_ind,
    "ttest_rel": _ttest_rel,
    "mann_whitney": ranks.mann_whitney,
    "wilcoxon": ranks.wilcoxon,
    "chi2": ranks.chi2,
    "normality": ranks.normality,
    "correlation": ranks.correlation,
}
