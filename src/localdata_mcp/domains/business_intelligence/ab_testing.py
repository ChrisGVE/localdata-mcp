"""
Business Intelligence Domain - A/B Testing and Power Analysis.

This module implements A/B test analysis with statistical significance testing,
effect size calculation, confidence intervals, and power analysis for
experiment design planning.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2_contingency, mannwhitneyu, ttest_ind
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from statsmodels.stats.power import tt_solve_power, ttest_power
from statsmodels.stats.proportion import proportion_confint, proportions_ztest

from ...logging_manager import get_logger
from .models import ABTestResult

logger = get_logger(__name__)


class ABTestAnalyzer(BaseEstimator, TransformerMixin):
    """
    sklearn-compatible transformer for A/B test analysis and statistical testing.

    Performs comprehensive A/B test analysis including statistical significance,
    effect size calculation, confidence intervals, and power analysis.

    Parameters:
    -----------
    alpha : float, default=0.05
        Significance level for hypothesis testing
    alternative : str, default='two-sided'
        Alternative hypothesis: 'two-sided', 'greater', 'less'
    test_type : str, default='proportion'
        Type of test: 'proportion', 'mean', 'conversion'
    """

    def __init__(self, alpha=0.05, alternative="two-sided", test_type="proportion"):
        self.alpha = alpha
        self.alternative = alternative
        self.test_type = test_type

    def fit(self, X, y=None):
        """Fit the A/B test analyzer."""
        self.is_fitted_ = True
        return self

    def transform(self, X):
        """Perform A/B test analysis."""
        check_is_fitted(self, "is_fitted_")

        logger.info(f"Performing A/B test analysis with {self.test_type} test")

        if self.test_type == "proportion":
            return self._analyze_proportion_test(X)
        elif self.test_type == "mean":
            return self._analyze_mean_test(X)
        elif self.test_type == "conversion":
            return self._analyze_conversion_test(X)
        else:
            raise ValueError(f"Unsupported test type: {self.test_type}")

    def _analyze_proportion_test(self, X):
        """Analyze A/B test for proportions (e.g., conversion rates)."""
        # Expect columns: 'group', 'converted'
        if "group" not in X.columns or "converted" not in X.columns:
            raise ValueError(
                "For proportion test, data must have 'group' and 'converted' columns"
            )

        # Calculate conversion rates by group
        summary_stats = (
            X.groupby("group").agg({"converted": ["count", "sum", "mean"]}).round(4)
        )

        summary_stats.columns = ["total", "conversions", "conversion_rate"]

        # Extract data for statistical test
        groups = summary_stats.index.tolist()
        if len(groups) != 2:
            raise ValueError("Currently supports only 2-group A/B tests")

        group_a, group_b = groups[0], groups[1]

        n_a = summary_stats.loc[group_a, "total"]
        x_a = summary_stats.loc[group_a, "conversions"]
        n_b = summary_stats.loc[group_b, "total"]
        x_b = summary_stats.loc[group_b, "conversions"]

        p_a = x_a / n_a
        p_b = x_b / n_b

        # Perform z-test for proportions
        counts = np.array([x_a, x_b])
        nobs = np.array([n_a, n_b])

        z_stat, p_value = proportions_ztest(counts, nobs, alternative=self.alternative)

        # Calculate confidence interval for difference
        p_diff = p_b - p_a
        se_diff = np.sqrt((p_a * (1 - p_a) / n_a) + (p_b * (1 - p_b) / n_b))
        z_critical = stats.norm.ppf(1 - self.alpha / 2)
        ci_lower = p_diff - z_critical * se_diff
        ci_upper = p_diff + z_critical * se_diff

        # Effect size (Cohen's h for proportions)
        h = 2 * (np.arcsin(np.sqrt(p_b)) - np.arcsin(np.sqrt(p_a)))

        # Power analysis
        pooled_p = (x_a + x_b) / (n_a + n_b)
        power = self._calculate_power_proportion(n_a, n_b, p_a, p_b, self.alpha)

        # Generate conclusion
        conclusion = self._generate_conclusion(p_value, p_diff, self.alpha)

        return ABTestResult(
            test_name=f"Proportion A/B Test ({group_a} vs {group_b})",
            test_statistic=z_stat,
            p_value=p_value,
            confidence_interval=(ci_lower, ci_upper),
            effect_size=h,
            power=power,
            conclusion=conclusion,
            sample_sizes={group_a: n_a, group_b: n_b},
            conversion_rates={group_a: p_a, group_b: p_b},
        )

    def _calculate_power_proportion(self, n1, n2, p1, p2, alpha):
        """Calculate statistical power for proportion test."""
        # Simplified power calculation
        pooled_p = ((n1 * p1) + (n2 * p2)) / (n1 + n2)
        pooled_se = np.sqrt(pooled_p * (1 - pooled_p) * (1 / n1 + 1 / n2))

        if pooled_se == 0:
            return 1.0

        z_alpha = stats.norm.ppf(1 - alpha / 2)
        z_beta = abs(p2 - p1) / pooled_se - z_alpha
        power = stats.norm.cdf(z_beta)

        return max(0.0, min(1.0, power))

    def _generate_conclusion(self, p_value, effect, alpha):
        """Generate human-readable conclusion."""
        if p_value < alpha:
            significance = "statistically significant"
            direction = "positive" if effect > 0 else "negative"
        else:
            significance = "not statistically significant"
            direction = "inconclusive"

        return f"Result is {significance} (p={p_value:.4f}). Effect direction: {direction}."


class PowerAnalysisTransformer(BaseEstimator, TransformerMixin):
    """
    sklearn-compatible transformer for statistical power analysis and experiment design.

    Calculates required sample sizes, detectable effect sizes, and power for
    experimental design planning.

    Parameters:
    -----------
    power : float, default=0.8
        Desired statistical power (1 - beta)
    alpha : float, default=0.05
        Type I error rate (significance level)
    effect_size : float, optional
        Expected effect size (Cohen's d for means, Cohen's h for proportions)
    """

    def __init__(self, power=0.8, alpha=0.05, effect_size=None):
        self.power = power
        self.alpha = alpha
        self.effect_size = effect_size

    def fit(self, X, y=None):
        """Fit the power analysis transformer."""
        self.is_fitted_ = True
        return self

    def transform(self, X):
        """Perform power analysis calculations."""
        check_is_fitted(self, "is_fitted_")

        logger.info("Performing statistical power analysis for experiment design")

        # For demonstration, calculate sample size requirements for different effect sizes
        effect_sizes = (
            [0.1, 0.2, 0.3, 0.5, 0.8]
            if self.effect_size is None
            else [self.effect_size]
        )

        results = []
        for es in effect_sizes:
            try:
                # Calculate required sample size per group
                n_required = tt_solve_power(
                    effect_size=es, power=self.power, alpha=self.alpha
                )

                results.append(
                    {
                        "effect_size": es,
                        "required_n_per_group": int(np.ceil(n_required)),
                        "total_required_n": int(np.ceil(n_required * 2)),
                        "power": self.power,
                        "alpha": self.alpha,
                    }
                )
            except:
                # Handle edge cases
                results.append(
                    {
                        "effect_size": es,
                        "required_n_per_group": "Unable to calculate",
                        "total_required_n": "Unable to calculate",
                        "power": self.power,
                        "alpha": self.alpha,
                    }
                )

        return pd.DataFrame(results)
