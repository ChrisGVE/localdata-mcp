"""
Sampling & Estimation Domain - BootstrapTransformer.

sklearn-compatible transformer for bootstrap methods and confidence intervals
including parametric/non-parametric bootstrap, bias correction, and BCa intervals.
"""

from typing import Any, Callable, Dict, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from ._results import BootstrapResult, get_logger

logger = get_logger(__name__)


class BootstrapTransformer(BaseEstimator, TransformerMixin):
    """
    sklearn-compatible transformer for bootstrap methods and confidence intervals.

    Implements parametric and non-parametric bootstrap, confidence interval construction,
    bias correction, and bootstrap hypothesis testing.

    Parameters:
    -----------
    statistic_func : callable or str, default='mean'
        Function to bootstrap or string name of common statistics
    n_bootstrap : int, default=1000
        Number of bootstrap samples
    confidence_level : float, default=0.95
        Confidence level for intervals
    method : str, default='percentile'
        Bootstrap method: 'percentile', 'bca', 'basic', 'studentized'
    random_state : int, default=None
        Random seed for reproducibility

    Attributes:
    -----------
    bootstrap_results_ : List[BootstrapResult]
        Results of bootstrap analysis for each column
    """

    def __init__(
        self,
        statistic_func: Union[Callable, str] = "mean",
        n_bootstrap: int = 1000,
        confidence_level: float = 0.95,
        method: str = "percentile",
        random_state: Optional[int] = None,
    ):
        self.statistic_func = statistic_func
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
        self.method = method
        self.random_state = random_state

    def fit(self, X, y=None):
        """Fit the transformer (no-op for bootstrap)."""
        self._validate_parameters()
        self.is_fitted_ = True
        return self

    def transform(self, X):
        """Perform bootstrap analysis on the input data."""
        check_is_fitted(self)

        if isinstance(X, pd.DataFrame):
            data = X
        else:
            data = pd.DataFrame(X)

        self.bootstrap_results_ = []

        # Get statistic function
        stat_func = self._get_statistic_function()

        # Perform bootstrap for each numeric column
        numeric_cols = data.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            try:
                col_data = data[col].dropna()
                if len(col_data) >= 2:
                    bootstrap_result = self._bootstrap_single_column(
                        col_data, col, stat_func
                    )
                    self.bootstrap_results_.append(bootstrap_result)
            except Exception as e:
                logger.warning(f"Bootstrap failed for column {col}: {e}")

        # Create result summary
        result_summary = {
            "bootstrap_results": [
                result.to_dict() for result in self.bootstrap_results_
            ],
            "n_bootstrap": self.n_bootstrap,
            "confidence_level": self.confidence_level,
            "method": self.method,
        }

        return pd.DataFrame([result_summary])

    def _validate_parameters(self):
        """Validate input parameters."""
        if not 0 < self.confidence_level < 1:
            raise ValueError("confidence_level must be between 0 and 1")

        if self.n_bootstrap <= 0:
            raise ValueError("n_bootstrap must be positive")

        valid_methods = ["percentile", "bca", "basic", "studentized"]
        if self.method not in valid_methods:
            raise ValueError(f"method must be one of {valid_methods}")

    def _get_statistic_function(self) -> Callable:
        """Get the statistic function to bootstrap."""
        if isinstance(self.statistic_func, str):
            if self.statistic_func == "mean":
                return np.mean
            elif self.statistic_func == "median":
                return np.median
            elif self.statistic_func == "std":
                return np.std
            elif self.statistic_func == "var":
                return np.var
            else:
                raise ValueError(f"Unknown statistic function: {self.statistic_func}")
        else:
            return self.statistic_func

    def _bootstrap_single_column(
        self, data: pd.Series, col_name: str, stat_func: Callable
    ) -> BootstrapResult:
        """Perform bootstrap analysis for a single column."""
        # Original statistic
        original_stat = stat_func(data.values)

        # Generate bootstrap samples
        np.random.seed(self.random_state)
        bootstrap_stats = []

        for _ in range(self.n_bootstrap):
            bootstrap_sample = np.random.choice(
                data.values, size=len(data), replace=True
            )
            bootstrap_stat = stat_func(bootstrap_sample)
            bootstrap_stats.append(bootstrap_stat)

        bootstrap_stats = np.array(bootstrap_stats)

        # Calculate bias and variance
        bias_estimate = np.mean(bootstrap_stats) - original_stat
        variance_estimate = np.var(bootstrap_stats)
        standard_error = np.sqrt(variance_estimate)

        # Calculate confidence intervals
        confidence_intervals = {}
        alpha = 1 - self.confidence_level

        if self.method == "percentile":
            lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
            upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
            confidence_intervals["percentile"] = (lower, upper)

        elif self.method == "basic":
            lower = 2 * original_stat - np.percentile(
                bootstrap_stats, 100 * (1 - alpha / 2)
            )
            upper = 2 * original_stat - np.percentile(bootstrap_stats, 100 * alpha / 2)
            confidence_intervals["basic"] = (lower, upper)

        elif self.method == "bca":
            # BCa (Bias-Corrected and Accelerated) method
            # This is a simplified implementation
            z0 = stats.norm.ppf((bootstrap_stats < original_stat).mean())

            # Jackknife to estimate acceleration
            n = len(data)
            jackknife_stats = []
            for i in range(n):
                jackknife_sample = np.concatenate(
                    [data.values[:i], data.values[i + 1 :]]
                )
                jackknife_stat = stat_func(jackknife_sample)
                jackknife_stats.append(jackknife_stat)

            jackknife_stats = np.array(jackknife_stats)
            jackknife_mean = np.mean(jackknife_stats)

            # Acceleration parameter
            numerator = np.sum((jackknife_mean - jackknife_stats) ** 3)
            denominator = 6 * (np.sum((jackknife_mean - jackknife_stats) ** 2)) ** 1.5

            if denominator != 0:
                acceleration = numerator / denominator
            else:
                acceleration = 0

            # Adjusted percentiles
            z_alpha_2 = stats.norm.ppf(alpha / 2)
            z_1_alpha_2 = stats.norm.ppf(1 - alpha / 2)

            alpha1 = stats.norm.cdf(
                z0 + (z0 + z_alpha_2) / (1 - acceleration * (z0 + z_alpha_2))
            )
            alpha2 = stats.norm.cdf(
                z0 + (z0 + z_1_alpha_2) / (1 - acceleration * (z0 + z_1_alpha_2))
            )

            lower = np.percentile(bootstrap_stats, 100 * alpha1)
            upper = np.percentile(bootstrap_stats, 100 * alpha2)
            confidence_intervals["bca"] = (lower, upper)

        # Bias-corrected estimate
        bias_corrected_estimate = original_stat - bias_estimate

        return BootstrapResult(
            statistic_name=f"{self.statistic_func}_{col_name}",
            original_statistic=original_stat,
            bootstrap_method=self.method,
            n_bootstrap=self.n_bootstrap,
            bootstrap_distribution=bootstrap_stats,
            confidence_intervals=confidence_intervals,
            bias_estimate=bias_estimate,
            bias_corrected_estimate=bias_corrected_estimate,
            variance_estimate=variance_estimate,
            standard_error=standard_error,
            bootstrap_params={
                "confidence_level": self.confidence_level,
                "random_state": self.random_state,
            },
        )
