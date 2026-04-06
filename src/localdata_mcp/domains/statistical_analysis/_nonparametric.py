"""
Statistical Analysis Domain - Non-Parametric Test Transformer.

Implements Mann-Whitney U, Wilcoxon signed-rank, Kruskal-Wallis H, and Friedman tests
as alternatives to parametric tests when assumptions are violated.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from ...logging_manager import get_logger
from ._base import StatisticalTestResult

logger = get_logger(__name__)


class NonParametricTestTransformer(BaseEstimator, TransformerMixin):
    """
    sklearn-compatible transformer for non-parametric statistical tests.

    Performs Mann-Whitney U, Wilcoxon signed-rank, Kruskal-Wallis H, and Friedman tests
    as alternatives to parametric tests when assumptions are violated.

    Parameters:
    -----------
    test_type : str, default='auto'
        Type of test: 'mann_whitney', 'wilcoxon', 'kruskal_wallis', 'friedman', 'auto'
    alpha : float, default=0.05
        Significance level for hypothesis tests
    alternative : str, default='two-sided'
        Alternative hypothesis: 'two-sided', 'less', 'greater'
    correction : str, default=None
        Multiple comparison correction: 'bonferroni', 'fdr_bh', None
    calculate_effect_size : bool, default=True
        Whether to calculate effect sizes (rank-biserial correlation, etc.)

    Attributes:
    -----------
    test_results_ : List[StatisticalTestResult]
        Results of performed non-parametric tests
    effect_sizes_ : Dict[str, float]
        Calculated effect sizes for non-parametric tests
    """

    def __init__(
        self,
        test_type: str = "auto",
        alpha: float = 0.05,
        alternative: str = "two-sided",
        correction: Optional[str] = None,
        calculate_effect_size: bool = True,
    ):
        self.test_type = test_type
        self.alpha = alpha
        self.alternative = alternative
        self.correction = correction
        self.calculate_effect_size = calculate_effect_size
        self._validate_parameters()

    def fit(self, X, y=None):
        """Fit the transformer (no-op for non-parametric tests)."""
        self._validate_parameters()
        self.is_fitted_ = True
        return self

    def transform(self, X):
        """Perform non-parametric tests on the input data."""
        check_is_fitted(self)

        if isinstance(X, pd.DataFrame):
            data = X
        else:
            data = pd.DataFrame(X)

        self.test_results_ = []
        self.effect_sizes_ = {}

        # Perform tests based on type
        if self.test_type == "auto":
            self._perform_automatic_nonparametric_tests(data)
        else:
            self._perform_specific_nonparametric_test(data, self.test_type)

        # Create result summary
        result_summary = {
            "test_results": [result.to_dict() for result in self.test_results_],
            "effect_sizes": self.effect_sizes_,
            "alpha_level": self.alpha,
            "correction_applied": self.correction,
        }

        return pd.DataFrame([result_summary])

    def _validate_parameters(self):
        """Validate input parameters."""
        valid_tests = ["auto", "mann_whitney", "wilcoxon", "kruskal_wallis", "friedman"]
        if self.test_type not in valid_tests:
            raise ValueError(f"test_type must be one of {valid_tests}")

        if not 0 < self.alpha < 1:
            raise ValueError("alpha must be between 0 and 1")

    def _perform_automatic_nonparametric_tests(self, data: pd.DataFrame):
        """Automatically select and perform appropriate non-parametric tests."""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        categorical_cols = data.select_dtypes(include=["object", "category"]).columns

        # Mann-Whitney U tests for numeric vs binary categorical
        for num_col in numeric_cols:
            for cat_col in categorical_cols:
                unique_cats = data[cat_col].nunique()
                if unique_cats == 2:
                    self._mann_whitney_test(data, num_col, cat_col)
                elif unique_cats > 2:
                    self._kruskal_wallis_test(data, num_col, cat_col)

        # Wilcoxon tests for paired numeric columns
        if len(numeric_cols) >= 2:
            for i, col1 in enumerate(numeric_cols):
                for col2 in numeric_cols[i + 1 :]:
                    self._wilcoxon_test(data, col1, col2)

    def _perform_specific_nonparametric_test(self, data: pd.DataFrame, test_type: str):
        """Perform a specific non-parametric test."""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        categorical_cols = data.select_dtypes(include=["object", "category"]).columns

        if test_type == "mann_whitney":
            for num_col in numeric_cols:
                for cat_col in categorical_cols:
                    if data[cat_col].nunique() == 2:
                        self._mann_whitney_test(data, num_col, cat_col)

        elif test_type == "wilcoxon":
            if len(numeric_cols) >= 2:
                for i, col1 in enumerate(numeric_cols):
                    for col2 in numeric_cols[i + 1 :]:
                        self._wilcoxon_test(data, col1, col2)

        elif test_type == "kruskal_wallis":
            for num_col in numeric_cols:
                for cat_col in categorical_cols:
                    if data[cat_col].nunique() > 2:
                        self._kruskal_wallis_test(data, num_col, cat_col)

        elif test_type == "friedman":
            if len(numeric_cols) >= 3:
                # Friedman test requires repeated measures design
                # For simplicity, we'll test all numeric columns as if they're repeated measures
                numeric_data = data[numeric_cols].dropna()
                if len(numeric_data) >= 3:
                    self._friedman_test(numeric_data, numeric_cols.tolist())

    def _mann_whitney_test(self, data: pd.DataFrame, num_col: str, cat_col: str):
        """Perform Mann-Whitney U test."""
        try:
            categories = data[cat_col].value_counts().head(2).index
            group1 = data[data[cat_col] == categories[0]][num_col].dropna()
            group2 = data[data[cat_col] == categories[1]][num_col].dropna()

            if len(group1) < 3 or len(group2) < 3:
                return

            # Perform Mann-Whitney U test
            statistic, p_value = stats.mannwhitneyu(
                group1, group2, alternative=self.alternative, use_continuity=True
            )

            # Calculate effect size (rank-biserial correlation)
            n1, n2 = len(group1), len(group2)
            r = 1 - (2 * statistic) / (n1 * n2)  # rank-biserial correlation

            # Effect size interpretation
            abs_r = abs(r)
            if abs_r >= 0.5:
                effect_desc = "large"
            elif abs_r >= 0.3:
                effect_desc = "medium"
            elif abs_r >= 0.1:
                effect_desc = "small"
            else:
                effect_desc = "negligible"

            interpretation = f"{'Significant' if p_value <= self.alpha else 'Non-significant'} difference between groups ({effect_desc} effect)"

            result = StatisticalTestResult(
                test_name=f"Mann-Whitney U ({num_col} by {cat_col})",
                statistic=statistic,
                p_value=p_value,
                effect_size=abs_r,
                interpretation=interpretation,
                additional_info={
                    "numeric_column": num_col,
                    "grouping_column": cat_col,
                    "group1": str(categories[0]),
                    "group2": str(categories[1]),
                    "group1_median": group1.median(),
                    "group2_median": group2.median(),
                    "group1_size": n1,
                    "group2_size": n2,
                    "rank_biserial_correlation": r,
                    "effect_description": effect_desc,
                },
            )
            self.test_results_.append(result)

        except Exception as e:
            logger.warning(
                f"Mann-Whitney U test failed for {num_col} by {cat_col}: {e}"
            )

    def _wilcoxon_test(self, data: pd.DataFrame, col1: str, col2: str):
        """Perform Wilcoxon signed-rank test."""
        try:
            paired_data = data[[col1, col2]].dropna()
            if len(paired_data) < 6:
                return

            # Perform Wilcoxon signed-rank test
            statistic, p_value = stats.wilcoxon(
                paired_data[col1],
                paired_data[col2],
                alternative=self.alternative,
                zero_method="wilcox",
            )

            # Calculate effect size (rank-biserial correlation for paired data)
            differences = paired_data[col1] - paired_data[col2]
            n_pos = (differences > 0).sum()
            n_neg = (differences < 0).sum()
            r = (n_pos - n_neg) / len(differences) if len(differences) > 0 else 0

            # Effect size interpretation
            abs_r = abs(r)
            if abs_r >= 0.5:
                effect_desc = "large"
            elif abs_r >= 0.3:
                effect_desc = "medium"
            elif abs_r >= 0.1:
                effect_desc = "small"
            else:
                effect_desc = "negligible"

            interpretation = f"{'Significant' if p_value <= self.alpha else 'Non-significant'} difference between paired measurements ({effect_desc} effect)"

            result = StatisticalTestResult(
                test_name=f"Wilcoxon Signed-Rank ({col1} vs {col2})",
                statistic=statistic,
                p_value=p_value,
                effect_size=abs_r,
                interpretation=interpretation,
                additional_info={
                    "column1": col1,
                    "column2": col2,
                    "median_difference": differences.median(),
                    "positive_ranks": n_pos,
                    "negative_ranks": n_neg,
                    "rank_biserial_correlation": r,
                    "effect_description": effect_desc,
                    "sample_size": len(paired_data),
                },
            )
            self.test_results_.append(result)

        except Exception as e:
            logger.warning(f"Wilcoxon test failed for {col1} vs {col2}: {e}")

    def _kruskal_wallis_test(self, data: pd.DataFrame, num_col: str, cat_col: str):
        """Perform Kruskal-Wallis H test."""
        try:
            groups = [
                group[num_col].dropna().values for name, group in data.groupby(cat_col)
            ]
            group_names = list(data[cat_col].unique())

            # Filter out groups with insufficient data
            valid_groups = [g for g in groups if len(g) >= 3]
            if len(valid_groups) < 2:
                return

            # Perform Kruskal-Wallis test
            statistic, p_value = stats.kruskal(*valid_groups)
            df = len(valid_groups) - 1

            # Calculate effect size (eta-squared analog for Kruskal-Wallis)
            n_total = sum(len(g) for g in valid_groups)
            eta_squared = (statistic - len(valid_groups) + 1) / (
                n_total - len(valid_groups)
            )
            eta_squared = max(0, min(1, eta_squared))  # Bound between 0 and 1

            # Effect size interpretation
            if eta_squared >= 0.14:
                effect_desc = "large"
            elif eta_squared >= 0.06:
                effect_desc = "medium"
            elif eta_squared >= 0.01:
                effect_desc = "small"
            else:
                effect_desc = "negligible"

            interpretation = f"{'Significant' if p_value <= self.alpha else 'Non-significant'} difference between groups ({effect_desc} effect)"

            result = StatisticalTestResult(
                test_name=f"Kruskal-Wallis H ({num_col} by {cat_col})",
                statistic=statistic,
                p_value=p_value,
                degrees_of_freedom=df,
                effect_size=eta_squared,
                interpretation=interpretation,
                additional_info={
                    "numeric_column": num_col,
                    "grouping_column": cat_col,
                    "group_names": group_names[: len(valid_groups)],
                    "group_medians": [np.median(g) for g in valid_groups],
                    "group_sizes": [len(g) for g in valid_groups],
                    "eta_squared": eta_squared,
                    "effect_description": effect_desc,
                },
            )
            self.test_results_.append(result)

        except Exception as e:
            logger.warning(
                f"Kruskal-Wallis test failed for {num_col} by {cat_col}: {e}"
            )

    def _friedman_test(self, data: pd.DataFrame, columns: List[str]):
        """Perform Friedman test for repeated measures."""
        try:
            if len(data) < 3:
                return

            # Perform Friedman test
            statistic, p_value = stats.friedmanchisquare(
                *[data[col].values for col in columns]
            )
            df = len(columns) - 1

            # Calculate Kendall's W (effect size for Friedman test)
            n = len(data)
            k = len(columns)
            kendalls_w = statistic / (n * (k - 1))

            # Effect size interpretation for Kendall's W
            if kendalls_w >= 0.5:
                effect_desc = "large"
            elif kendalls_w >= 0.3:
                effect_desc = "medium"
            elif kendalls_w >= 0.1:
                effect_desc = "small"
            else:
                effect_desc = "negligible"

            interpretation = f"{'Significant' if p_value <= self.alpha else 'Non-significant'} difference between repeated measures ({effect_desc} effect)"

            result = StatisticalTestResult(
                test_name=f"Friedman Test ({', '.join(columns)})",
                statistic=statistic,
                p_value=p_value,
                degrees_of_freedom=df,
                effect_size=kendalls_w,
                interpretation=interpretation,
                additional_info={
                    "columns": columns,
                    "sample_size": n,
                    "kendalls_w": kendalls_w,
                    "effect_description": effect_desc,
                    "column_medians": [data[col].median() for col in columns],
                },
            )
            self.test_results_.append(result)

        except Exception as e:
            logger.warning(f"Friedman test failed: {e}")
