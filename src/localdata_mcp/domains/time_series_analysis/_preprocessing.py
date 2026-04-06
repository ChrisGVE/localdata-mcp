"""
Time Series Analysis - Preprocessing transformers.

Contains resampling and imputation transformers for time series data preparation.
"""

from typing import Any, Dict, Optional

import pandas as pd
from sklearn.utils.validation import check_is_fitted

from ...logging_manager import get_logger
from ._transformer import TimeSeriesTransformer

logger = get_logger(__name__)


class TimeSeriesResamplingTransformer(TimeSeriesTransformer):
    """
    sklearn-compatible transformer for time series resampling operations.

    Resamples time series data to a specified frequency with aggregation functions.
    """

    def __init__(
        self,
        target_frequency: str,
        aggregation_method="mean",
        interpolate_missing=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.target_frequency = target_frequency
        self.aggregation_method = aggregation_method
        self.interpolate_missing = interpolate_missing
        self.original_frequency_ = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fit the resampling transformer."""
        if self.validate_input:
            X, y = self._validate_time_series(X, y)

        if self.infer_frequency:
            self.original_frequency_ = self._infer_frequency(X)

        logger.debug(
            f"Fitted resampling transformer: "
            f"{self.original_frequency_} -> {self.target_frequency}"
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Resample the time series data."""
        check_is_fitted(self, ["original_frequency_"])

        if self.validate_input:
            X, _ = self._validate_time_series(X, None)

        try:
            X_resampled = self._apply_resampling(X)

            if self.interpolate_missing and X_resampled.isnull().any().any():
                X_resampled = X_resampled.interpolate(method="time")

            logger.debug(f"Resampled from {len(X)} to {len(X_resampled)} observations")
            return X_resampled

        except Exception as e:
            logger.error(f"Error in resampling transformation: {e}")
            raise

    def _apply_resampling(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the configured resampling strategy."""
        if isinstance(self.aggregation_method, dict):
            return X.resample(self.target_frequency).agg(self.aggregation_method)

        resampler = X.resample(self.target_frequency)
        agg_methods = {
            "mean": resampler.mean,
            "sum": resampler.sum,
            "min": resampler.min,
            "max": resampler.max,
            "first": resampler.first,
            "last": resampler.last,
        }
        method_fn = agg_methods.get(self.aggregation_method)
        if method_fn is None:
            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")
        return method_fn()


class TimeSeriesImputationTransformer(TimeSeriesTransformer):
    """
    sklearn-compatible transformer for time series missing value imputation.

    Provides imputation methods that respect temporal structure.
    """

    def __init__(
        self,
        method="interpolate",
        interpolation_method="time",
        limit_direction="forward",
        seasonal_period=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.method = method
        self.interpolation_method = interpolation_method
        self.limit_direction = limit_direction
        self.seasonal_period = seasonal_period
        self.seasonal_patterns_: Dict[str, Any] = {}

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fit the imputation transformer."""
        if self.validate_input:
            X, y = self._validate_time_series(X, y)

        if self.method == "seasonal":
            self._learn_seasonal_patterns(X)

        self.is_fitted_ = True
        return self

    def _learn_seasonal_patterns(self, X: pd.DataFrame) -> None:
        """Learn seasonal patterns from the data for seasonal imputation."""
        for col in X.columns:
            series = X[col].dropna()
            if len(series) == 0:
                continue

            if self.seasonal_period is None:
                seasonality_info = self._detect_seasonality(series)
                period = (
                    seasonality_info["dominant_period"]
                    if seasonality_info["has_seasonality"]
                    else min(12, len(series) // 4)
                )
            else:
                period = self.seasonal_period

            seasonal_means = {}
            for i in range(period):
                season_values = series.iloc[i::period]
                seasonal_means[i] = (
                    season_values.mean() if len(season_values) > 0 else series.mean()
                )

            self.seasonal_patterns_[col] = {
                "period": period,
                "seasonal_means": seasonal_means,
            }

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values in time series data."""
        if self.validate_input:
            X, _ = self._validate_time_series(X, None)

        X_imputed = X.copy()

        try:
            impute_methods = {
                "forward_fill": lambda df: df.fillna(method="ffill"),
                "backward_fill": lambda df: df.fillna(method="bfill"),
                "interpolate": lambda df: df.interpolate(
                    method=self.interpolation_method
                ),
                "mean": lambda df: df.fillna(df.mean()),
                "median": lambda df: df.fillna(df.median()),
            }

            if self.method in impute_methods:
                X_imputed = impute_methods[self.method](X_imputed)
            elif self.method == "seasonal":
                X_imputed = self._apply_seasonal_imputation(X, X_imputed)

            original_missing = X.isnull().sum().sum()
            final_missing = X_imputed.isnull().sum().sum()
            imputed_values = original_missing - final_missing

            if imputed_values > 0:
                logger.debug(
                    f"Imputed {imputed_values} missing values using {self.method}"
                )

            return X_imputed

        except Exception as e:
            logger.error(f"Error in imputation: {e}")
            raise

    def _apply_seasonal_imputation(
        self, X: pd.DataFrame, X_imputed: pd.DataFrame
    ) -> pd.DataFrame:
        """Apply seasonal pattern-based imputation."""
        check_is_fitted(self, ["seasonal_patterns_"])

        for col in X.columns:
            if col not in self.seasonal_patterns_:
                continue
            pattern = self.seasonal_patterns_[col]
            period = pattern["period"]
            seasonal_means = pattern["seasonal_means"]

            missing_mask = X_imputed[col].isnull()
            for idx in missing_mask[missing_mask].index:
                position_in_period = len(X_imputed.loc[:idx]) % period
                if position_in_period in seasonal_means:
                    X_imputed.loc[idx, col] = seasonal_means[position_in_period]

        return X_imputed
