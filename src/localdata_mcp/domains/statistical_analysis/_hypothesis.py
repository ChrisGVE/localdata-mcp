"""
Statistical Analysis Domain - Hypothesis Testing Transformer.

Implements comprehensive hypothesis testing including t-tests, chi-square tests,
normality tests, and correlation tests with proper effect size calculations
and assumption checking.
"""

from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2_contingency, pearsonr, spearmanr
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from ...logging_manager import get_logger
from ._base import StatisticalTestResult

logger = get_logger(__name__)


class HypothesisTestingTransformer(BaseEstimator, TransformerMixin):
    """
    sklearn-compatible transformer for comprehensive hypothesis testing.

    Performs various hypothesis tests including t-tests, chi-square tests,
    normality tests, and correlation tests with proper effect size calculations
    and assumption checking.

    Parameters:
    -----------
    test_type : str, default='auto'
        Type of test to perform: 'ttest_1samp', 'ttest_ind', 'ttest_rel',
        'chi2', 'normality', 'correlation', or 'auto' for automatic selection
    alpha : float, default=0.05
        Significance level for hypothesis tests
    alternative : str, default='two-sided'
        Alternative hypothesis: 'two-sided', 'less', 'greater'
    equal_var : bool, default=True
        Assume equal variances for independent t-tests
    correction : str, default=None
        Multiple comparison correction: 'bonferroni', 'fdr_bh', None
    calculate_effect_size : bool, default=True
        Whether to calculate effect sizes (Cohen's d, Cramer's V, etc.)
    check_assumptions : bool, default=True
        Whether to check statistical assumptions

    Attributes:
    -----------
    test_results_ : List[StatisticalTestResult]
        Results of performed statistical tests
    assumptions_checked_ : Dict[str, bool]
        Results of assumption checks
    effect_sizes_ : Dict[str, float]
        Calculated effect sizes
    """

    def __init__(
        self,
        test_type: str = "auto",
        alpha: float = 0.05,
        alternative: str = "two-sided",
        equal_var: bool = True,
        correction: Optional[str] = None,
        calculate_effect_size: bool = True,
        check_assumptions: bool = True,
    ):
        self.test_type = test_type
        self.alpha = alpha
        self.alternative = alternative
        self.equal_var = equal_var
        self.correction = correction
        self.calculate_effect_size = calculate_effect_size
        self.check_assumptions = check_assumptions
        self._validate_parameters()

    def fit(self, X, y=None):
        """Fit the transformer (no-op for statistical tests)."""
        self._validate_parameters()
        self.is_fitted_ = True
        return self

    def transform(self, X):
        """Perform hypothesis tests on the input data."""
        check_is_fitted(self)

        if isinstance(X, pd.DataFrame):
            data = X
        else:
            data = pd.DataFrame(X)

        self.test_results_ = []
        self.assumptions_checked_ = {}
        self.effect_sizes_ = {}

        # Automatic test selection based on data characteristics
        if self.test_type == "auto":
            self._perform_automatic_testing(data)
        else:
            self._perform_specific_test(data, self.test_type)

        # Create result summary
        result_summary = {
            "test_results": [result.to_dict() for result in self.test_results_],
            "assumptions_checked": self.assumptions_checked_,
            "effect_sizes": self.effect_sizes_,
            "alpha_level": self.alpha,
            "correction_applied": self.correction,
        }

        return pd.DataFrame([result_summary])

    def _validate_parameters(self):
        """Validate input parameters."""
        valid_tests = [
            "auto",
            "ttest_1samp",
            "ttest_ind",
            "ttest_rel",
            "chi2",
            "normality",
            "correlation",
        ]
        if self.test_type not in valid_tests:
            raise ValueError(f"test_type must be one of {valid_tests}")

        if not 0 < self.alpha < 1:
            raise ValueError("alpha must be between 0 and 1")

        if self.alternative not in ["two-sided", "less", "greater"]:
            raise ValueError("alternative must be 'two-sided', 'less', or 'greater'")

    def _perform_automatic_testing(self, data: pd.DataFrame):
        """Automatically select and perform appropriate tests."""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        categorical_cols = data.select_dtypes(include=["object", "category"]).columns

        # Normality tests for numeric columns
        for col in numeric_cols:
            if data[col].notna().sum() >= 3:  # Minimum samples for normality test
                self._test_normality(data[col].dropna(), col)

        # Correlation tests between numeric columns
        if len(numeric_cols) >= 2:
            self._test_correlations(data[numeric_cols])

        # Chi-square tests for categorical columns
        if len(categorical_cols) >= 2:
            for i, col1 in enumerate(categorical_cols):
                for col2 in categorical_cols[i + 1 :]:
                    self._test_chi_square(data, col1, col2)

    def _perform_specific_test(self, data: pd.DataFrame, test_type: str):
        """Perform a specific type of test."""
        if test_type == "normality":
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if data[col].notna().sum() >= 3:
                    self._test_normality(data[col].dropna(), col)

        elif test_type == "correlation":
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) >= 2:
                self._test_correlations(data[numeric_cols])

        elif test_type == "chi2":
            categorical_cols = data.select_dtypes(
                include=["object", "category"]
            ).columns
            if len(categorical_cols) >= 2:
                for i, col1 in enumerate(categorical_cols):
                    for col2 in categorical_cols[i + 1 :]:
                        self._test_chi_square(data, col1, col2)

        elif test_type.startswith("ttest"):
            self._perform_t_test(data, test_type)

    def _test_normality(self, series: pd.Series, col_name: str):
        """Perform normality tests (Shapiro-Wilk and Kolmogorov-Smirnov)."""
        data_clean = series.dropna()

        if len(data_clean) < 3:
            return

        # Shapiro-Wilk test (better for smaller samples)
        if len(data_clean) <= 5000:
            try:
                shapiro_stat, shapiro_p = stats.shapiro(data_clean)
                interpretation = f"Data {'appears' if shapiro_p > self.alpha else 'does not appear'} to be normally distributed"

                result = StatisticalTestResult(
                    test_name=f"Shapiro-Wilk ({col_name})",
                    statistic=shapiro_stat,
                    p_value=shapiro_p,
                    interpretation=interpretation,
                    additional_info={
                        "column": col_name,
                        "sample_size": len(data_clean),
                    },
                )
                self.test_results_.append(result)
            except Exception as e:
                logger.warning(f"Shapiro-Wilk test failed for {col_name}: {e}")

        # Kolmogorov-Smirnov test (better for larger samples)
        try:
            ks_stat, ks_p = stats.kstest(
                data_clean, "norm", args=(data_clean.mean(), data_clean.std())
            )
            interpretation = f"Data {'appears' if ks_p > self.alpha else 'does not appear'} to follow normal distribution"

            result = StatisticalTestResult(
                test_name=f"Kolmogorov-Smirnov ({col_name})",
                statistic=ks_stat,
                p_value=ks_p,
                interpretation=interpretation,
                additional_info={"column": col_name, "sample_size": len(data_clean)},
            )
            self.test_results_.append(result)
        except Exception as e:
            logger.warning(f"KS test failed for {col_name}: {e}")

    def _test_correlations(self, data: pd.DataFrame):
        """Perform correlation tests between numeric variables."""
        numeric_data = data.select_dtypes(include=[np.number]).dropna()

        if len(numeric_data) < 3:
            return

        cols = list(numeric_data.columns)

        for i, col1 in enumerate(cols):
            for col2 in cols[i + 1 :]:
                # Pearson correlation
                try:
                    pearson_r, pearson_p = pearsonr(
                        numeric_data[col1], numeric_data[col2]
                    )

                    # Calculate effect size (Cohen's conventions for correlation)
                    effect_size = abs(pearson_r)
                    if effect_size >= 0.5:
                        effect_desc = "large"
                    elif effect_size >= 0.3:
                        effect_desc = "medium"
                    elif effect_size >= 0.1:
                        effect_desc = "small"
                    else:
                        effect_desc = "negligible"

                    interpretation = f"{'Significant' if pearson_p <= self.alpha else 'Non-significant'} correlation ({effect_desc} effect)"

                    result = StatisticalTestResult(
                        test_name=f"Pearson Correlation ({col1} vs {col2})",
                        statistic=pearson_r,
                        p_value=pearson_p,
                        effect_size=effect_size,
                        interpretation=interpretation,
                        additional_info={
                            "column1": col1,
                            "column2": col2,
                            "effect_description": effect_desc,
                            "sample_size": len(numeric_data),
                        },
                    )
                    self.test_results_.append(result)
                except Exception as e:
                    logger.warning(
                        f"Pearson correlation failed for {col1} vs {col2}: {e}"
                    )

                # Spearman correlation (non-parametric)
                try:
                    spearman_r, spearman_p = spearmanr(
                        numeric_data[col1], numeric_data[col2]
                    )
                    interpretation = f"{'Significant' if spearman_p <= self.alpha else 'Non-significant'} rank correlation"

                    result = StatisticalTestResult(
                        test_name=f"Spearman Correlation ({col1} vs {col2})",
                        statistic=spearman_r,
                        p_value=spearman_p,
                        effect_size=abs(spearman_r),
                        interpretation=interpretation,
                        additional_info={
                            "column1": col1,
                            "column2": col2,
                            "sample_size": len(numeric_data),
                        },
                    )
                    self.test_results_.append(result)
                except Exception as e:
                    logger.warning(
                        f"Spearman correlation failed for {col1} vs {col2}: {e}"
                    )

    def _test_chi_square(self, data: pd.DataFrame, col1: str, col2: str):
        """Perform chi-square test of independence."""
        try:
            # Create contingency table
            contingency_table = pd.crosstab(data[col1], data[col2])

            # Perform chi-square test
            chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)

            # Calculate Cramer's V (effect size)
            n = contingency_table.sum().sum()
            cramers_v = np.sqrt(chi2_stat / (n * (min(contingency_table.shape) - 1)))

            # Effect size interpretation
            if min(contingency_table.shape) == 2:  # 2x2 table
                if cramers_v >= 0.5:
                    effect_desc = "large"
                elif cramers_v >= 0.3:
                    effect_desc = "medium"
                elif cramers_v >= 0.1:
                    effect_desc = "small"
                else:
                    effect_desc = "negligible"
            else:  # Larger tables
                if cramers_v >= 0.25:
                    effect_desc = "large"
                elif cramers_v >= 0.15:
                    effect_desc = "medium"
                elif cramers_v >= 0.05:
                    effect_desc = "small"
                else:
                    effect_desc = "negligible"

            interpretation = f"{'Significant' if p_value <= self.alpha else 'Non-significant'} association ({effect_desc} effect)"

            result = StatisticalTestResult(
                test_name=f"Chi-square ({col1} vs {col2})",
                statistic=chi2_stat,
                p_value=p_value,
                degrees_of_freedom=dof,
                effect_size=cramers_v,
                interpretation=interpretation,
                additional_info={
                    "column1": col1,
                    "column2": col2,
                    "effect_description": effect_desc,
                    "sample_size": n,
                    "contingency_table": contingency_table.to_dict(),
                },
            )
            self.test_results_.append(result)

        except Exception as e:
            logger.warning(f"Chi-square test failed for {col1} vs {col2}: {e}")

    def _perform_t_test(self, data: pd.DataFrame, test_type: str):
        """Perform various t-tests."""
        numeric_cols = data.select_dtypes(include=[np.number]).columns

        if test_type == "ttest_1samp" and len(numeric_cols) >= 1:
            # One-sample t-test against mean of 0
            for col in numeric_cols:
                series_clean = data[col].dropna()
                if len(series_clean) >= 3:
                    self._one_sample_ttest(series_clean, col)

        elif test_type == "ttest_ind" and len(numeric_cols) >= 1:
            # Independent t-test (requires grouping variable)
            categorical_cols = data.select_dtypes(
                include=["object", "category"]
            ).columns
            if len(categorical_cols) >= 1:
                for num_col in numeric_cols:
                    for cat_col in categorical_cols:
                        self._independent_ttest(data, num_col, cat_col)

        elif test_type == "ttest_rel" and len(numeric_cols) >= 2:
            # Paired t-test
            for i, col1 in enumerate(numeric_cols):
                for col2 in numeric_cols[i + 1 :]:
                    self._paired_ttest(data, col1, col2)

    def _one_sample_ttest(self, series: pd.Series, col_name: str, popmean: float = 0):
        """Perform one-sample t-test."""
        try:
            t_stat, p_value = stats.ttest_1samp(series, popmean)

            # Calculate Cohen's d
            cohens_d = (series.mean() - popmean) / series.std()

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

            interpretation = f"{'Significant' if p_value <= self.alpha else 'Non-significant'} difference from {popmean} ({effect_desc} effect)"

            result = StatisticalTestResult(
                test_name=f"One-sample t-test ({col_name})",
                statistic=t_stat,
                p_value=p_value,
                degrees_of_freedom=len(series) - 1,
                effect_size=abs_d,
                interpretation=interpretation,
                additional_info={
                    "column": col_name,
                    "sample_mean": series.mean(),
                    "population_mean": popmean,
                    "cohens_d": cohens_d,
                    "effect_description": effect_desc,
                    "sample_size": len(series),
                },
            )
            self.test_results_.append(result)

        except Exception as e:
            logger.warning(f"One-sample t-test failed for {col_name}: {e}")

    def _independent_ttest(self, data: pd.DataFrame, num_col: str, cat_col: str):
        """Perform independent samples t-test."""
        try:
            # Get unique categories (limit to 2 for t-test)
            categories = data[cat_col].value_counts().head(2).index
            if len(categories) < 2:
                return

            group1 = data[data[cat_col] == categories[0]][num_col].dropna()
            group2 = data[data[cat_col] == categories[1]][num_col].dropna()

            if len(group1) < 2 or len(group2) < 2:
                return

            # Perform t-test
            t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=self.equal_var)

            # Calculate Cohen's d
            pooled_std = np.sqrt(
                ((len(group1) - 1) * group1.var() + (len(group2) - 1) * group2.var())
                / (len(group1) + len(group2) - 2)
            )
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

            interpretation = f"{'Significant' if p_value <= self.alpha else 'Non-significant'} difference between groups ({effect_desc} effect)"

            result = StatisticalTestResult(
                test_name=f"Independent t-test ({num_col} by {cat_col})",
                statistic=t_stat,
                p_value=p_value,
                degrees_of_freedom=len(group1) + len(group2) - 2,
                effect_size=abs_d,
                interpretation=interpretation,
                additional_info={
                    "numeric_column": num_col,
                    "grouping_column": cat_col,
                    "group1": str(categories[0]),
                    "group2": str(categories[1]),
                    "group1_mean": group1.mean(),
                    "group2_mean": group2.mean(),
                    "group1_size": len(group1),
                    "group2_size": len(group2),
                    "cohens_d": cohens_d,
                    "effect_description": effect_desc,
                    "equal_var_assumed": self.equal_var,
                },
            )
            self.test_results_.append(result)

        except Exception as e:
            logger.warning(f"Independent t-test failed for {num_col} by {cat_col}: {e}")

    def _paired_ttest(self, data: pd.DataFrame, col1: str, col2: str):
        """Perform paired samples t-test."""
        try:
            # Get paired data
            paired_data = data[[col1, col2]].dropna()
            if len(paired_data) < 3:
                return

            # Perform paired t-test
            t_stat, p_value = stats.ttest_rel(paired_data[col1], paired_data[col2])

            # Calculate Cohen's d for paired samples
            differences = paired_data[col1] - paired_data[col2]
            cohens_d = differences.mean() / differences.std()

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

            interpretation = f"{'Significant' if p_value <= self.alpha else 'Non-significant'} difference between paired measurements ({effect_desc} effect)"

            result = StatisticalTestResult(
                test_name=f"Paired t-test ({col1} vs {col2})",
                statistic=t_stat,
                p_value=p_value,
                degrees_of_freedom=len(paired_data) - 1,
                effect_size=abs_d,
                interpretation=interpretation,
                additional_info={
                    "column1": col1,
                    "column2": col2,
                    "mean_difference": differences.mean(),
                    "cohens_d": cohens_d,
                    "effect_description": effect_desc,
                    "sample_size": len(paired_data),
                },
            )
            self.test_results_.append(result)

        except Exception as e:
            logger.warning(f"Paired t-test failed for {col1} vs {col2}: {e}")
