"""
Statistical Analysis Domain - Assumption-driven group comparison.

File: _group_comparison.py
Folder: src/localdata_mcp/domains/statistical_analysis/

When a caller names a grouping column, the question is "do these groups
differ?". This module answers it by running the comparison the data's
assumptions actually support, rather than making the caller name a procedure:

- two groups, both plausibly normal      -> independent-samples t-test
  (Welch's variant when the variances differ)
- two groups, at least one non-normal    -> Mann-Whitney U
- three or more, normal + equal variance -> one-way ANOVA
- three or more, otherwise               -> Kruskal-Wallis H

``HypothesisTestingTransformer`` (``_hypothesis.py``) calls
:func:`compare_groups` from its automatic-selection path, and reuses
:func:`independent_ttest_result` for the explicitly requested ``ttest_ind``
so there is one implementation of the t-test in the domain.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from ...logging_manager import get_logger
from ._base import StatisticalTestResult

logger = get_logger(__name__)

# A group needs two observations before it has a variance to compare.
MIN_GROUP_SIZE = 2

# Shapiro-Wilk needs at least three points and is unreliable (and refused by
# scipy) well past five thousand.
MIN_NORMALITY_SAMPLES = 3
MAX_NORMALITY_SAMPLES = 5000


def compare_groups(
    data: pd.DataFrame,
    value_column: str,
    group_column: str,
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> List[StatisticalTestResult]:
    """Compare ``value_column`` across the levels of ``group_column``.

    Args:
        data: Frame holding both columns.
        value_column: Numeric column whose values are compared.
        group_column: Column whose distinct values define the groups.
        alpha: Significance level, also used for the assumption checks.
        alternative: Direction passed to the two-sample tests.

    Returns:
        The comparison results — usually one, empty when fewer than two
        groups carry enough observations to compare.
    """
    labels, samples = _split_groups(data, value_column, group_column)
    if len(samples) < 2:
        logger.warning(
            f"Group comparison skipped: '{group_column}' yields "
            f"{len(samples)} usable group(s) of '{value_column}'"
        )
        return []

    normal = _groups_look_normal(samples, alpha)
    equal_variance = _variances_look_equal(samples, alpha)

    if len(samples) == 2:
        if normal:
            result = independent_ttest_result(
                group1=samples[0],
                group2=samples[1],
                labels=(labels[0], labels[1]),
                value_column=value_column,
                group_column=group_column,
                alpha=alpha,
                equal_var=equal_variance,
                alternative=alternative,
            )
        else:
            result = _mann_whitney_result(
                group1=samples[0],
                group2=samples[1],
                labels=(labels[0], labels[1]),
                value_column=value_column,
                group_column=group_column,
                alpha=alpha,
                alternative=alternative,
            )
    elif normal and equal_variance:
        result = _one_way_anova_result(
            labels, samples, value_column, group_column, alpha
        )
    else:
        result = _kruskal_result(labels, samples, value_column, group_column, alpha)

    return [result] if result is not None else []


# ---------------------------------------------------------------------------
# Assumption checks
# ---------------------------------------------------------------------------


def _split_groups(
    data: pd.DataFrame,
    value_column: str,
    group_column: str,
) -> Tuple[List[str], List[pd.Series]]:
    """Return the group labels and their value samples, in label order."""
    if value_column not in data.columns or group_column not in data.columns:
        return [], []

    frame = data[[value_column, group_column]].dropna()
    labels: List[str] = []
    samples: List[pd.Series] = []
    for label, subset in frame.groupby(group_column, sort=True, observed=True):
        values = pd.to_numeric(subset[value_column], errors="coerce").dropna()
        if len(values) >= MIN_GROUP_SIZE:
            labels.append(str(label))
            samples.append(values)
    return labels, samples


def _groups_look_normal(samples: Sequence[pd.Series], alpha: float) -> bool:
    """Whether every group is compatible with a normal distribution.

    A group too large for Shapiro-Wilk is accepted on the strength of the
    central limit theorem; a group too small to test is not, because nothing
    supports the parametric assumption there.
    """
    for sample in samples:
        if len(sample) > MAX_NORMALITY_SAMPLES:
            continue
        if len(sample) < MIN_NORMALITY_SAMPLES:
            return False
        try:
            _, p_value = stats.shapiro(sample)
        except Exception as exc:  # pragma: no cover - scipy edge cases
            logger.warning(f"Normality check failed, assuming non-normal: {exc}")
            return False
        if p_value <= alpha:
            return False
    return True


def _variances_look_equal(samples: Sequence[pd.Series], alpha: float) -> bool:
    """Whether Levene's test fails to reject equal variances across groups."""
    try:
        _, p_value = stats.levene(*samples)
    except Exception as exc:  # pragma: no cover - scipy edge cases
        logger.warning(f"Variance check failed, assuming unequal variances: {exc}")
        return False
    return bool(p_value > alpha)


def _describe_magnitude(value: float, thresholds: Sequence[float]) -> str:
    """Label an effect size against (large, medium, small) cut-offs."""
    large, medium, small = thresholds
    magnitude = abs(value)
    if magnitude >= large:
        return "large"
    if magnitude >= medium:
        return "medium"
    if magnitude >= small:
        return "small"
    return "negligible"


def _verdict(p_value: float, alpha: float) -> str:
    return "Significant" if p_value <= alpha else "Non-significant"


# ---------------------------------------------------------------------------
# The individual comparisons
# ---------------------------------------------------------------------------


def independent_ttest_result(
    group1: pd.Series,
    group2: pd.Series,
    labels: Tuple[str, str],
    value_column: str,
    group_column: str,
    alpha: float = 0.05,
    equal_var: bool = True,
    alternative: str = "two-sided",
) -> Optional[StatisticalTestResult]:
    """Independent-samples t-test with Cohen's d as the effect size."""
    try:
        t_stat, p_value = stats.ttest_ind(
            group1, group2, equal_var=equal_var, alternative=alternative
        )
    except Exception as exc:
        logger.warning(
            f"Independent t-test failed for {value_column} by {group_column}: {exc}"
        )
        return None

    n1, n2 = len(group1), len(group2)
    pooled_std = np.sqrt(
        ((n1 - 1) * group1.var() + (n2 - 1) * group2.var()) / (n1 + n2 - 2)
    )
    cohens_d = (
        (group1.mean() - group2.mean()) / pooled_std if pooled_std > 0 else float("nan")
    )
    effect_desc = _describe_magnitude(cohens_d, (0.8, 0.5, 0.2))
    variant = "Student's" if equal_var else "Welch's"

    return StatisticalTestResult(
        test_name=f"Independent t-test ({value_column} by {group_column})",
        statistic=t_stat,
        p_value=p_value,
        degrees_of_freedom=n1 + n2 - 2,
        effect_size=abs(cohens_d),
        interpretation=(
            f"{_verdict(p_value, alpha)} difference between groups "
            f"({effect_desc} effect)"
        ),
        additional_info={
            "numeric_column": value_column,
            "grouping_column": group_column,
            "group1": labels[0],
            "group2": labels[1],
            "group1_mean": group1.mean(),
            "group2_mean": group2.mean(),
            "group1_size": n1,
            "group2_size": n2,
            "cohens_d": cohens_d,
            "effect_description": effect_desc,
            "equal_var_assumed": equal_var,
            "test_variant": f"{variant} t-test",
        },
    )


def _mann_whitney_result(
    group1: pd.Series,
    group2: pd.Series,
    labels: Tuple[str, str],
    value_column: str,
    group_column: str,
    alpha: float,
    alternative: str,
) -> Optional[StatisticalTestResult]:
    """Mann-Whitney U with the rank-biserial correlation as effect size."""
    try:
        u_stat, p_value = stats.mannwhitneyu(group1, group2, alternative=alternative)
    except Exception as exc:
        logger.warning(
            f"Mann-Whitney U failed for {value_column} by {group_column}: {exc}"
        )
        return None

    n1, n2 = len(group1), len(group2)
    # Rank-biserial correlation: the U statistic rescaled to [-1, 1]
    # (Kerby, 2014, "The simple difference formula").
    rank_biserial = 2.0 * u_stat / (n1 * n2) - 1.0
    effect_desc = _describe_magnitude(rank_biserial, (0.5, 0.3, 0.1))

    return StatisticalTestResult(
        test_name=f"Mann-Whitney U ({value_column} by {group_column})",
        statistic=u_stat,
        p_value=p_value,
        effect_size=abs(rank_biserial),
        interpretation=(
            f"{_verdict(p_value, alpha)} difference between group "
            f"distributions ({effect_desc} effect)"
        ),
        additional_info={
            "numeric_column": value_column,
            "grouping_column": group_column,
            "group1": labels[0],
            "group2": labels[1],
            "group1_median": group1.median(),
            "group2_median": group2.median(),
            "group1_size": n1,
            "group2_size": n2,
            "rank_biserial_correlation": rank_biserial,
            "effect_description": effect_desc,
            "selected_because": "at least one group is not normally distributed",
        },
    )


def _one_way_anova_result(
    labels: Sequence[str],
    samples: Sequence[pd.Series],
    value_column: str,
    group_column: str,
    alpha: float,
) -> Optional[StatisticalTestResult]:
    """One-way ANOVA across three or more groups, with eta squared."""
    try:
        f_stat, p_value = stats.f_oneway(*samples)
    except Exception as exc:
        logger.warning(
            f"One-way ANOVA failed for {value_column} by {group_column}: {exc}"
        )
        return None

    pooled = np.concatenate([sample.to_numpy() for sample in samples])
    grand_mean = pooled.mean()
    ss_between = sum(
        len(sample) * (sample.mean() - grand_mean) ** 2 for sample in samples
    )
    ss_total = float(((pooled - grand_mean) ** 2).sum())
    eta_squared = ss_between / ss_total if ss_total > 0 else float("nan")
    effect_desc = _describe_magnitude(eta_squared, (0.14, 0.06, 0.01))

    return StatisticalTestResult(
        test_name=f"One-way ANOVA ({value_column} by {group_column})",
        statistic=f_stat,
        p_value=p_value,
        degrees_of_freedom=len(samples) - 1,
        effect_size=eta_squared,
        interpretation=(
            f"{_verdict(p_value, alpha)} difference among "
            f"{len(samples)} group means ({effect_desc} effect)"
        ),
        additional_info=_group_summary(labels, samples, value_column, group_column)
        | {"eta_squared": eta_squared, "effect_description": effect_desc},
    )


def _kruskal_result(
    labels: Sequence[str],
    samples: Sequence[pd.Series],
    value_column: str,
    group_column: str,
    alpha: float,
) -> Optional[StatisticalTestResult]:
    """Kruskal-Wallis H across groups, with epsilon squared."""
    try:
        h_stat, p_value = stats.kruskal(*samples)
    except Exception as exc:
        logger.warning(
            f"Kruskal-Wallis failed for {value_column} by {group_column}: {exc}"
        )
        return None

    total = sum(len(sample) for sample in samples)
    # Epsilon squared for Kruskal-Wallis (Tomczak & Tomczak, 2014).
    epsilon_squared = (
        (h_stat - len(samples) + 1) / (total - len(samples))
        if total > len(samples)
        else float("nan")
    )
    effect_desc = _describe_magnitude(epsilon_squared, (0.14, 0.06, 0.01))

    return StatisticalTestResult(
        test_name=f"Kruskal-Wallis H ({value_column} by {group_column})",
        statistic=h_stat,
        p_value=p_value,
        degrees_of_freedom=len(samples) - 1,
        effect_size=epsilon_squared,
        interpretation=(
            f"{_verdict(p_value, alpha)} difference among "
            f"{len(samples)} group distributions ({effect_desc} effect)"
        ),
        additional_info=_group_summary(labels, samples, value_column, group_column)
        | {
            "epsilon_squared": epsilon_squared,
            "effect_description": effect_desc,
            "selected_because": (
                "groups are not normal or do not share a common variance"
            ),
        },
    )


def _group_summary(
    labels: Sequence[str],
    samples: Sequence[pd.Series],
    value_column: str,
    group_column: str,
) -> Dict[str, Any]:
    """Per-group sizes, means and medians, for a multi-group result."""
    return {
        "numeric_column": value_column,
        "grouping_column": group_column,
        "groups": list(labels),
        "group_sizes": {label: len(sample) for label, sample in zip(labels, samples)},
        "group_means": {
            label: float(sample.mean()) for label, sample in zip(labels, samples)
        },
        "group_medians": {
            label: float(sample.median()) for label, sample in zip(labels, samples)
        },
    }
