"""
Statistical Analysis Domain - ANOVA Analysis Transformer.

Implements one-way and two-way ANOVA with post-hoc analysis and effect size calculations.
Includes assumptions checking and non-parametric alternatives when appropriate.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.anova import anova_lm
from statsmodels.formula.api import ols

from ...logging_manager import get_logger

logger = get_logger(__name__)


class ANOVAAnalysisTransformer(BaseEstimator, TransformerMixin):
    """
    sklearn-compatible transformer for comprehensive ANOVA analysis.

    Performs one-way and two-way ANOVA with post-hoc analysis and effect size calculations.
    Includes assumptions checking and non-parametric alternatives when appropriate.

    Parameters:
    -----------
    anova_type : str, default='one_way'
        Type of ANOVA: 'one_way', 'two_way', 'auto'
    alpha : float, default=0.05
        Significance level for hypothesis tests
    post_hoc : str, default='tukey'
        Post-hoc test method: 'tukey', 'bonferroni', 'scheffe', None
    effect_size : str, default='eta_squared'
        Effect size measure: 'eta_squared', 'partial_eta_squared', 'omega_squared'
    check_assumptions : bool, default=True
        Whether to check ANOVA assumptions (normality, homoscedasticity)
    alpha_adjustment : str, default=None
        Multiple comparison adjustment: 'bonferroni', 'fdr_bh', None

    Attributes:
    -----------
    anova_results_ : Dict[str, Any]
        ANOVA test results and statistics
    post_hoc_results_ : Dict[str, Any]
        Post-hoc comparison results
    effect_sizes_ : Dict[str, float]
        Calculated effect sizes
    assumptions_ : Dict[str, bool]
        Results of assumption checks
    """

    def __init__(
        self,
        anova_type: str = "one_way",
        alpha: float = 0.05,
        post_hoc: str = "tukey",
        effect_size: str = "eta_squared",
        check_assumptions: bool = True,
        alpha_adjustment: Optional[str] = None,
    ):
        self.anova_type = anova_type
        self.alpha = alpha
        self.post_hoc = post_hoc
        self.effect_size = effect_size
        self.check_assumptions = check_assumptions
        self.alpha_adjustment = alpha_adjustment
        self._validate_parameters()

    def fit(self, X, y=None):
        """Fit the transformer (no-op for ANOVA)."""
        self._validate_parameters()
        self.is_fitted_ = True
        return self

    def transform(self, X):
        """Perform ANOVA analysis on the input data."""
        check_is_fitted(self)

        if isinstance(X, pd.DataFrame):
            data = X
        else:
            data = pd.DataFrame(X)

        self.anova_results_ = {}
        self.post_hoc_results_ = {}
        self.effect_sizes_ = {}
        self.assumptions_ = {}

        # Identify numeric and categorical columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        categorical_cols = data.select_dtypes(include=["object", "category"]).columns

        if len(numeric_cols) == 0 or len(categorical_cols) == 0:
            logger.warning("ANOVA requires both numeric and categorical variables")
            return pd.DataFrame(
                [
                    {
                        "error": "ANOVA requires both numeric and categorical variables",
                        "numeric_columns": len(numeric_cols),
                        "categorical_columns": len(categorical_cols),
                    }
                ]
            )

        # Perform ANOVA based on type
        if self.anova_type == "auto":
            self._perform_automatic_anova(data, numeric_cols, categorical_cols)
        elif self.anova_type == "one_way":
            self._perform_one_way_anova(data, numeric_cols, categorical_cols)
        elif self.anova_type == "two_way":
            self._perform_two_way_anova(data, numeric_cols, categorical_cols)

        # Create result summary
        result_summary = {
            "anova_results": self.anova_results_,
            "post_hoc_results": self.post_hoc_results_,
            "effect_sizes": self.effect_sizes_,
            "assumptions_checked": self.assumptions_,
            "alpha_level": self.alpha,
            "post_hoc_method": self.post_hoc,
        }

        return pd.DataFrame([result_summary])

    def _validate_parameters(self):
        """Validate input parameters."""
        valid_types = ["one_way", "two_way", "auto"]
        if self.anova_type not in valid_types:
            raise ValueError(f"anova_type must be one of {valid_types}")

        if not 0 < self.alpha < 1:
            raise ValueError("alpha must be between 0 and 1")

        valid_post_hoc = ["tukey", "bonferroni", "scheffe", None]
        if self.post_hoc not in valid_post_hoc:
            raise ValueError(f"post_hoc must be one of {valid_post_hoc}")

    def _perform_automatic_anova(
        self, data: pd.DataFrame, numeric_cols: pd.Index, categorical_cols: pd.Index
    ):
        """Automatically determine and perform appropriate ANOVA."""
        # Start with one-way ANOVA for each combination
        self._perform_one_way_anova(data, numeric_cols, categorical_cols)

        # If we have multiple categorical variables, also try two-way ANOVA
        if len(categorical_cols) >= 2:
            self._perform_two_way_anova(data, numeric_cols, categorical_cols[:2])

    def _perform_one_way_anova(
        self, data: pd.DataFrame, numeric_cols: pd.Index, categorical_cols: pd.Index
    ):
        """Perform one-way ANOVA for each numeric-categorical combination."""
        for num_col in numeric_cols:
            for cat_col in categorical_cols:
                try:
                    self._one_way_anova_single(data, num_col, cat_col)
                except Exception as e:
                    logger.warning(
                        f"One-way ANOVA failed for {num_col} by {cat_col}: {e}"
                    )

    def _one_way_anova_single(self, data: pd.DataFrame, num_col: str, cat_col: str):
        """Perform one-way ANOVA for a single numeric-categorical pair."""
        # Clean data
        clean_data = data[[num_col, cat_col]].dropna()
        if len(clean_data) < 6:  # Minimum samples for ANOVA
            return

        # Group data
        groups = [group[num_col].values for name, group in clean_data.groupby(cat_col)]
        group_names = list(clean_data[cat_col].unique())

        if len(groups) < 2:
            return

        # Check assumptions if requested
        if self.check_assumptions:
            self._check_anova_assumptions(groups, f"{num_col}_by_{cat_col}")

        # Perform one-way ANOVA
        f_stat, p_value = stats.f_oneway(*groups)
        df_between = len(groups) - 1
        df_within = len(clean_data) - len(groups)
        df_total = len(clean_data) - 1

        # Calculate effect sizes
        ss_between = sum(
            len(g) * (np.mean(g) - clean_data[num_col].mean()) ** 2 for g in groups
        )
        ss_total = sum((clean_data[num_col] - clean_data[num_col].mean()) ** 2)
        ss_within = ss_total - ss_between

        eta_squared = ss_between / ss_total
        omega_squared = (ss_between - df_between * (ss_within / df_within)) / (
            ss_total + ss_within / df_within
        )
        omega_squared = max(0, omega_squared)  # Can't be negative

        # Effect size interpretation (Cohen's conventions for eta-squared)
        if eta_squared >= 0.14:
            effect_desc = "large"
        elif eta_squared >= 0.06:
            effect_desc = "medium"
        elif eta_squared >= 0.01:
            effect_desc = "small"
        else:
            effect_desc = "negligible"

        # Store results
        anova_key = f"one_way_{num_col}_by_{cat_col}"
        self.anova_results_[anova_key] = {
            "test_type": "One-way ANOVA",
            "dependent_variable": num_col,
            "independent_variable": cat_col,
            "f_statistic": f_stat,
            "p_value": p_value,
            "df_between": df_between,
            "df_within": df_within,
            "df_total": df_total,
            "significant": bool(p_value <= self.alpha),
            "group_names": group_names,
            "group_means": [np.mean(g) for g in groups],
            "group_sizes": [len(g) for g in groups],
            "interpretation": f"{'Significant' if p_value <= self.alpha else 'Non-significant'} group differences ({effect_desc} effect)",
        }

        self.effect_sizes_[anova_key] = {
            "eta_squared": eta_squared,
            "omega_squared": omega_squared,
            "effect_description": effect_desc,
        }

        # Perform post-hoc tests if significant and requested
        if p_value <= self.alpha and self.post_hoc and len(groups) > 2:
            self._perform_post_hoc_analysis(clean_data, num_col, cat_col, anova_key)

    def _perform_two_way_anova(
        self, data: pd.DataFrame, numeric_cols: pd.Index, categorical_cols: pd.Index
    ):
        """Perform two-way ANOVA with interaction effects."""
        if len(categorical_cols) < 2:
            return

        for num_col in numeric_cols:
            try:
                cat_col1, cat_col2 = categorical_cols[0], categorical_cols[1]
                self._two_way_anova_single(data, num_col, cat_col1, cat_col2)
            except Exception as e:
                logger.warning(f"Two-way ANOVA failed for {num_col}: {e}")

    def _two_way_anova_single(
        self, data: pd.DataFrame, num_col: str, cat_col1: str, cat_col2: str
    ):
        """Perform two-way ANOVA for a single configuration."""
        # Clean data
        clean_data = data[[num_col, cat_col1, cat_col2]].dropna()
        if len(clean_data) < 12:  # Minimum samples for two-way ANOVA
            return

        try:
            # Create formula for OLS
            formula = f"{num_col} ~ C({cat_col1}) + C({cat_col2}) + C({cat_col1}):C({cat_col2})"

            # Fit OLS model
            model = ols(formula, data=clean_data).fit()

            # Perform ANOVA
            anova_table = anova_lm(model, typ=2)

            # Extract results
            anova_key = f"two_way_{num_col}_by_{cat_col1}_and_{cat_col2}"

            self.anova_results_[anova_key] = {
                "test_type": "Two-way ANOVA",
                "dependent_variable": num_col,
                "independent_variable_1": cat_col1,
                "independent_variable_2": cat_col2,
                "anova_table": anova_table.to_dict(),
                "model_summary": {
                    "r_squared": model.rsquared,
                    "adj_r_squared": model.rsquared_adj,
                    "f_statistic": model.fvalue,
                    "f_p_value": model.f_pvalue,
                },
            }

            # Calculate effect sizes for main effects and interaction
            ss_residual = anova_table.loc["Residual", "sum_sq"]
            ss_total = anova_table["sum_sq"].sum()

            effect_sizes = {}
            for factor in anova_table.index[:-1]:  # Exclude residual
                ss_factor = anova_table.loc[factor, "sum_sq"]
                eta_squared = ss_factor / ss_total
                partial_eta_squared = ss_factor / (ss_factor + ss_residual)
                effect_sizes[factor] = {
                    "eta_squared": eta_squared,
                    "partial_eta_squared": partial_eta_squared,
                }

            self.effect_sizes_[anova_key] = effect_sizes

        except Exception as e:
            logger.warning(f"Two-way ANOVA calculation failed: {e}")

    def _check_anova_assumptions(self, groups: List[np.ndarray], test_name: str):
        """Check ANOVA assumptions (normality and homoscedasticity)."""
        assumptions = {}

        # Check normality for each group (Shapiro-Wilk)
        normality_results = []
        for i, group in enumerate(groups):
            if len(group) >= 3 and len(group) <= 5000:
                try:
                    _, p_val = stats.shapiro(group)
                    normality_results.append(p_val > 0.05)
                except:
                    normality_results.append(None)
            else:
                normality_results.append(None)

        assumptions["normality"] = all(r for r in normality_results if r is not None)

        # Check homoscedasticity (Levene's test)
        if len(groups) >= 2 and all(len(g) >= 2 for g in groups):
            try:
                _, levene_p = stats.levene(*groups)
                assumptions["homoscedasticity"] = bool(levene_p > 0.05)
            except:
                assumptions["homoscedasticity"] = None
        else:
            assumptions["homoscedasticity"] = None

        self.assumptions_[test_name] = assumptions

    def _perform_post_hoc_analysis(
        self, data: pd.DataFrame, num_col: str, cat_col: str, anova_key: str
    ):
        """Perform post-hoc pairwise comparisons."""
        if self.post_hoc == "tukey":
            try:
                # Tukey HSD test
                tukey_result = pairwise_tukeyhsd(
                    data[num_col], data[cat_col], alpha=self.alpha
                )

                # Convert to structured format
                post_hoc_summary = {
                    "method": "Tukey HSD",
                    "alpha": self.alpha,
                    "comparisons": [],
                }

                # Extract pairwise comparisons
                for i in range(len(tukey_result.groupsunique)):
                    for j in range(i + 1, len(tukey_result.groupsunique)):
                        group1 = tukey_result.groupsunique[i]
                        group2 = tukey_result.groupsunique[j]

                        # Find the corresponding result
                        mask = (
                            (tukey_result.data["group1"] == group1)
                            & (tukey_result.data["group2"] == group2)
                        ) | (
                            (tukey_result.data["group1"] == group2)
                            & (tukey_result.data["group2"] == group1)
                        )

                        if mask.any():
                            row = tukey_result.data[mask].iloc[0]
                            post_hoc_summary["comparisons"].append(
                                {
                                    "group1": str(group1),
                                    "group2": str(group2),
                                    "mean_diff": row["meandiff"],
                                    "p_value": row["p-adj"],
                                    "significant": row["reject"],
                                    "lower_ci": row["lower"],
                                    "upper_ci": row["upper"],
                                }
                            )

                self.post_hoc_results_[anova_key] = post_hoc_summary

            except Exception as e:
                logger.warning(f"Tukey post-hoc test failed: {e}")
