"""
Statistical Analysis Domain - Experimental Design Transformer.

Implements power analysis, sample size determination, effect size calculations,
and confidence interval construction for experimental research design.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from statsmodels.stats.power import tt_solve_power, ttest_power

from ...logging_manager import get_logger

logger = get_logger(__name__)


class ExperimentalDesignTransformer(BaseEstimator, TransformerMixin):
    """
    sklearn-compatible transformer for experimental design analysis.

    Performs power analysis, sample size determination, effect size calculations,
    and confidence interval construction for experimental research design.

    Parameters:
    -----------
    analysis_type : str, default='power_analysis'
        Type of analysis: 'power_analysis', 'sample_size', 'effect_size', 'confidence_intervals'
    effect_size : float, default=None
        Expected effect size (Cohen's d for t-tests, eta-squared for ANOVA)
    alpha : float, default=0.05
        Type I error rate (significance level)
    power : float, default=0.80
        Desired statistical power (1 - Type II error rate)
    test_type : str, default='ttest'
        Statistical test type: 'ttest', 'anova', 'correlation', 'proportion'
    alternative : str, default='two-sided'
        Alternative hypothesis: 'two-sided', 'larger', 'smaller'
    confidence_level : float, default=0.95
        Confidence level for interval estimation

    Attributes:
    -----------
    power_analysis_ : Dict[str, Any]
        Power analysis results
    sample_sizes_ : Dict[str, int]
        Calculated sample sizes
    effect_sizes_ : Dict[str, float]
        Calculated effect sizes
    confidence_intervals_ : Dict[str, Tuple[float, float]]
        Calculated confidence intervals
    """

    def __init__(
        self,
        analysis_type: str = "power_analysis",
        effect_size: Optional[float] = None,
        alpha: float = 0.05,
        power: float = 0.80,
        test_type: str = "ttest",
        alternative: str = "two-sided",
        confidence_level: float = 0.95,
    ):
        self.analysis_type = analysis_type
        self.effect_size = effect_size
        self.alpha = alpha
        self.power = power
        self.test_type = test_type
        self.alternative = alternative
        self.confidence_level = confidence_level
        self._validate_parameters()

    def fit(self, X, y=None):
        """Fit the transformer (no-op for experimental design)."""
        self._validate_parameters()
        self.is_fitted_ = True
        return self

    def transform(self, X):
        """Perform experimental design analysis on the input data."""
        check_is_fitted(self)

        if isinstance(X, pd.DataFrame):
            data = X
        else:
            data = pd.DataFrame(X)

        self.power_analysis_ = {}
        self.sample_sizes_ = {}
        self.effect_sizes_ = {}
        self.confidence_intervals_ = {}

        # Perform analysis based on type
        if self.analysis_type == "power_analysis":
            self._perform_power_analysis(data)
        elif self.analysis_type == "sample_size":
            self._calculate_sample_sizes(data)
        elif self.analysis_type == "effect_size":
            self._calculate_effect_sizes(data)
        elif self.analysis_type == "confidence_intervals":
            self._calculate_confidence_intervals(data)
        else:
            # Perform all analyses
            self._perform_power_analysis(data)
            self._calculate_sample_sizes(data)
            self._calculate_effect_sizes(data)
            self._calculate_confidence_intervals(data)

        # Create result summary
        result_summary = {
            "power_analysis": self.power_analysis_,
            "sample_sizes": self.sample_sizes_,
            "effect_sizes": self.effect_sizes_,
            "confidence_intervals": self.confidence_intervals_,
            "parameters": {
                "alpha": self.alpha,
                "power": self.power,
                "effect_size": self.effect_size,
                "test_type": self.test_type,
                "confidence_level": self.confidence_level,
            },
        }

        return pd.DataFrame([result_summary])

    def _validate_parameters(self):
        """Validate input parameters."""
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must be between 0 and 1")

        if not 0 < self.power < 1:
            raise ValueError("power must be between 0 and 1")

        if not 0 < self.confidence_level < 1:
            raise ValueError("confidence_level must be between 0 and 1")

        valid_tests = ["ttest", "anova", "correlation", "proportion"]
        if self.test_type not in valid_tests:
            raise ValueError(f"test_type must be one of {valid_tests}")

    def _perform_power_analysis(self, data: pd.DataFrame):
        """Perform statistical power analysis."""
        if self.test_type == "ttest":
            self._power_analysis_ttest(data)
        elif self.test_type == "anova":
            self._power_analysis_anova(data)
        elif self.test_type == "correlation":
            self._power_analysis_correlation(data)

    def _power_analysis_ttest(self, data: pd.DataFrame):
        """Power analysis for t-tests."""
        try:
            # Use provided effect size or estimate from data
            if self.effect_size is None:
                effect_size = 0.5  # Medium effect size
            else:
                effect_size = self.effect_size

            # Calculate power for different sample sizes
            sample_sizes = [10, 20, 30, 50, 100, 200, 500]
            powers = []

            for n in sample_sizes:
                try:
                    power = ttest_power(
                        effect_size, n, self.alpha, alternative=self.alternative
                    )
                    powers.append(power)
                except:
                    powers.append(None)

            # Calculate required sample size for desired power
            try:
                required_n = tt_solve_power(
                    effect_size=effect_size,
                    power=self.power,
                    alpha=self.alpha,
                    alternative=self.alternative,
                )
                required_n = int(np.ceil(required_n))
            except:
                required_n = None

            self.power_analysis_["ttest"] = {
                "effect_size": effect_size,
                "alpha": self.alpha,
                "desired_power": self.power,
                "required_sample_size": required_n,
                "power_curve": dict(zip(sample_sizes, powers)),
                "interpretation": self._interpret_power_analysis(
                    required_n, self.power
                ),
            }

        except Exception as e:
            logger.warning(f"T-test power analysis failed: {e}")

    def _power_analysis_anova(self, data: pd.DataFrame):
        """Power analysis for ANOVA (simplified version)."""
        try:
            # Estimate effect size from data if not provided
            if self.effect_size is None:
                effect_size = 0.25  # Medium effect size for ANOVA (eta-squared)
            else:
                effect_size = self.effect_size

            # Convert eta-squared to Cohen's f
            cohens_f = np.sqrt(effect_size / (1 - effect_size))

            # Assume 3 groups for power calculation
            num_groups = 3

            # Calculate power for different sample sizes per group
            sample_sizes_per_group = [10, 15, 20, 30, 50, 100]
            powers = []

            for n_per_group in sample_sizes_per_group:
                # Simplified power calculation using non-centrality parameter
                df1 = num_groups - 1
                df2 = num_groups * (n_per_group - 1)
                ncp = n_per_group * num_groups * (cohens_f**2)

                # Critical F value
                f_crit = stats.f.ppf(1 - self.alpha, df1, df2)

                # Power calculation
                power = 1 - stats.ncf.cdf(f_crit, df1, df2, ncp)
                powers.append(power)

            self.power_analysis_["anova"] = {
                "effect_size_eta_squared": effect_size,
                "cohens_f": cohens_f,
                "num_groups": num_groups,
                "alpha": self.alpha,
                "desired_power": self.power,
                "power_curve": dict(zip(sample_sizes_per_group, powers)),
                "interpretation": f"Power analysis for {num_groups}-group ANOVA with eta-squared = {effect_size:.3f}",
            }

        except Exception as e:
            logger.warning(f"ANOVA power analysis failed: {e}")

    def _power_analysis_correlation(self, data: pd.DataFrame):
        """Power analysis for correlation tests."""
        try:
            if self.effect_size is None:
                effect_size = 0.3  # Medium correlation
            else:
                effect_size = self.effect_size

            # Fisher's z-transformation
            z_effect = 0.5 * np.log((1 + effect_size) / (1 - effect_size))

            # Calculate power for different sample sizes
            sample_sizes = [20, 30, 50, 100, 200, 500]
            powers = []

            for n in sample_sizes:
                if n > 3:
                    # Standard error
                    se = 1 / np.sqrt(n - 3)

                    # Critical values
                    if self.alternative == "two-sided":
                        z_crit = stats.norm.ppf(1 - self.alpha / 2)
                    else:
                        z_crit = stats.norm.ppf(1 - self.alpha)

                    # Power calculation
                    if self.alternative == "two-sided":
                        power = (
                            1
                            - stats.norm.cdf(z_crit - z_effect / se)
                            + stats.norm.cdf(-z_crit - z_effect / se)
                        )
                    else:
                        power = 1 - stats.norm.cdf(z_crit - z_effect / se)

                    powers.append(power)
                else:
                    powers.append(None)

            # Calculate required sample size
            if self.alternative == "two-sided":
                z_crit = stats.norm.ppf(1 - self.alpha / 2)
            else:
                z_crit = stats.norm.ppf(1 - self.alpha)

            z_power = stats.norm.ppf(self.power)
            required_n = int(np.ceil(((z_crit + z_power) / z_effect) ** 2 + 3))

            self.power_analysis_["correlation"] = {
                "correlation_coefficient": effect_size,
                "alpha": self.alpha,
                "desired_power": self.power,
                "required_sample_size": required_n,
                "power_curve": dict(zip(sample_sizes, powers)),
                "interpretation": self._interpret_power_analysis(
                    required_n, self.power
                ),
            }

        except Exception as e:
            logger.warning(f"Correlation power analysis failed: {e}")

    def _calculate_sample_sizes(self, data: pd.DataFrame):
        """Calculate required sample sizes for different effect sizes."""
        if self.test_type == "ttest":
            effect_sizes = [0.2, 0.5, 0.8]  # Small, medium, large

            for es in effect_sizes:
                try:
                    n = tt_solve_power(
                        effect_size=es,
                        power=self.power,
                        alpha=self.alpha,
                        alternative=self.alternative,
                    )
                    self.sample_sizes_[f"ttest_cohens_d_{es}"] = int(np.ceil(n))
                except:
                    self.sample_sizes_[f"ttest_cohens_d_{es}"] = None

    def _calculate_effect_sizes(self, data: pd.DataFrame):
        """Calculate effect sizes from actual data."""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        categorical_cols = data.select_dtypes(include=["object", "category"]).columns

        # Cohen's d for two-group comparisons
        for num_col in numeric_cols:
            for cat_col in categorical_cols:
                if data[cat_col].nunique() == 2:
                    self._calculate_cohens_d(data, num_col, cat_col)

        # Correlation effect sizes
        if len(numeric_cols) >= 2:
            for i, col1 in enumerate(numeric_cols):
                for col2 in numeric_cols[i + 1 :]:
                    self._calculate_correlation_effect_size(data, col1, col2)

    def _calculate_cohens_d(self, data: pd.DataFrame, num_col: str, cat_col: str):
        """Calculate Cohen's d effect size."""
        try:
            categories = data[cat_col].value_counts().head(2).index
            group1 = data[data[cat_col] == categories[0]][num_col].dropna()
            group2 = data[data[cat_col] == categories[1]][num_col].dropna()

            if len(group1) < 2 or len(group2) < 2:
                return

            # Calculate pooled standard deviation
            pooled_std = np.sqrt(
                ((len(group1) - 1) * group1.var() + (len(group2) - 1) * group2.var())
                / (len(group1) + len(group2) - 2)
            )

            # Calculate Cohen's d
            cohens_d = (group1.mean() - group2.mean()) / pooled_std

            # Effect size interpretation
            abs_d = abs(cohens_d)
            if abs_d >= 0.8:
                effect_desc = "large"
            elif abs_d >= 0.5:
                effect_desc = "medium"
            elif abs_d >= 0.2:
                effect_desc = "small"
            else:
                effect_desc = "negligible"

            key = f"cohens_d_{num_col}_by_{cat_col}"
            self.effect_sizes_[key] = {
                "cohens_d": cohens_d,
                "absolute_effect_size": abs_d,
                "effect_description": effect_desc,
                "group1_mean": group1.mean(),
                "group2_mean": group2.mean(),
                "pooled_std": pooled_std,
                "group1_size": len(group1),
                "group2_size": len(group2),
            }

        except Exception as e:
            logger.warning(
                f"Cohen's d calculation failed for {num_col} by {cat_col}: {e}"
            )

    def _calculate_correlation_effect_size(
        self, data: pd.DataFrame, col1: str, col2: str
    ):
        """Calculate correlation effect size."""
        try:
            clean_data = data[[col1, col2]].dropna()
            if len(clean_data) < 3:
                return

            r, p_value = pearsonr(clean_data[col1], clean_data[col2])

            # Effect size interpretation for correlation
            abs_r = abs(r)
            if abs_r >= 0.5:
                effect_desc = "large"
            elif abs_r >= 0.3:
                effect_desc = "medium"
            elif abs_r >= 0.1:
                effect_desc = "small"
            else:
                effect_desc = "negligible"

            key = f"correlation_{col1}_vs_{col2}"
            self.effect_sizes_[key] = {
                "pearson_r": r,
                "absolute_correlation": abs_r,
                "effect_description": effect_desc,
                "p_value": p_value,
                "sample_size": len(clean_data),
                "r_squared": r**2,
            }

        except Exception as e:
            logger.warning(
                f"Correlation effect size calculation failed for {col1} vs {col2}: {e}"
            )

    def _calculate_confidence_intervals(self, data: pd.DataFrame):
        """Calculate confidence intervals for various statistics."""
        numeric_cols = data.select_dtypes(include=[np.number]).columns

        # Confidence intervals for means
        for col in numeric_cols:
            self._calculate_mean_ci(data[col], col)

        # Confidence intervals for correlations
        if len(numeric_cols) >= 2:
            for i, col1 in enumerate(numeric_cols):
                for col2 in numeric_cols[i + 1 :]:
                    self._calculate_correlation_ci(data, col1, col2)

    def _calculate_mean_ci(self, series: pd.Series, col_name: str):
        """Calculate confidence interval for mean."""
        try:
            clean_data = series.dropna()
            if len(clean_data) < 2:
                return

            mean = clean_data.mean()
            std_err = clean_data.std() / np.sqrt(len(clean_data))

            # t-distribution critical value
            alpha_ci = 1 - self.confidence_level
            t_crit = stats.t.ppf(1 - alpha_ci / 2, len(clean_data) - 1)

            # Confidence interval
            margin_error = t_crit * std_err
            ci_lower = mean - margin_error
            ci_upper = mean + margin_error

            key = f"mean_ci_{col_name}"
            self.confidence_intervals_[key] = {
                "mean": mean,
                "confidence_level": self.confidence_level,
                "lower_bound": ci_lower,
                "upper_bound": ci_upper,
                "margin_of_error": margin_error,
                "standard_error": std_err,
                "sample_size": len(clean_data),
            }

        except Exception as e:
            logger.warning(f"Mean CI calculation failed for {col_name}: {e}")

    def _calculate_correlation_ci(self, data: pd.DataFrame, col1: str, col2: str):
        """Calculate confidence interval for correlation coefficient."""
        try:
            clean_data = data[[col1, col2]].dropna()
            if len(clean_data) < 4:
                return

            r, _ = pearsonr(clean_data[col1], clean_data[col2])
            n = len(clean_data)

            # Fisher's z-transformation
            z_r = 0.5 * np.log((1 + r) / (1 - r))
            z_se = 1 / np.sqrt(n - 3)

            # Critical value
            alpha_ci = 1 - self.confidence_level
            z_crit = stats.norm.ppf(1 - alpha_ci / 2)

            # CI for z-transformed correlation
            z_lower = z_r - z_crit * z_se
            z_upper = z_r + z_crit * z_se

            # Transform back to correlation scale
            r_lower = (np.exp(2 * z_lower) - 1) / (np.exp(2 * z_lower) + 1)
            r_upper = (np.exp(2 * z_upper) - 1) / (np.exp(2 * z_upper) + 1)

            key = f"correlation_ci_{col1}_vs_{col2}"
            self.confidence_intervals_[key] = {
                "correlation": r,
                "confidence_level": self.confidence_level,
                "lower_bound": r_lower,
                "upper_bound": r_upper,
                "sample_size": n,
                "fisher_z": z_r,
                "z_standard_error": z_se,
            }

        except Exception as e:
            logger.warning(
                f"Correlation CI calculation failed for {col1} vs {col2}: {e}"
            )

    def _interpret_power_analysis(self, required_n: int, desired_power: float) -> str:
        """Generate interpretation for power analysis results."""
        if required_n is None:
            return "Power analysis calculation failed"

        if required_n <= 30:
            size_desc = "small"
        elif required_n <= 100:
            size_desc = "moderate"
        elif required_n <= 300:
            size_desc = "large"
        else:
            size_desc = "very large"

        return f"Requires {required_n} participants per group for {desired_power:.0%} power ({size_desc} sample size)"
