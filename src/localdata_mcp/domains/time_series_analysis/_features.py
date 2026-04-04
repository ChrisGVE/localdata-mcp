"""
Time Series Analysis - Feature extraction transformer.

Contains the TimeSeriesFeatureExtractor for creating temporal features
including lag variables, rolling statistics, and datetime components.
"""

from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.utils.validation import check_is_fitted

from ...logging_manager import get_logger
from ._transformer import TimeSeriesTransformer

logger = get_logger(__name__)


class TimeSeriesFeatureExtractor(TimeSeriesTransformer):
    """
    sklearn-compatible transformer for extracting temporal features.

    Creates lag variables, rolling statistics, datetime components,
    and seasonality indicators.
    """

    def __init__(
        self,
        lag_features=[1, 7, 30],
        rolling_windows=[7, 30],
        datetime_features=["hour", "dayofweek", "month", "quarter"],
        seasonal_features=True,
        cyclical_encoding=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.lag_features = lag_features
        self.rolling_windows = rolling_windows
        self.datetime_features = datetime_features
        self.seasonal_features = seasonal_features
        self.cyclical_encoding = cyclical_encoding
        self.feature_names_: List[str] = []

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fit the feature extraction transformer."""
        if self.validate_input:
            X, y = self._validate_time_series(X, y)

        self.feature_names_ = list(X.columns)
        self._build_feature_names(X)

        logger.debug(f"Will create {len(self.feature_names_)} features")
        return self

    def _build_feature_names(self, X: pd.DataFrame) -> None:
        """Build the list of feature names that will be created."""
        for col in X.columns:
            for lag in self.lag_features:
                self.feature_names_.append(f"{col}_lag_{lag}")

        for col in X.columns:
            for window in self.rolling_windows:
                self.feature_names_.extend(
                    [
                        f"{col}_rolling_mean_{window}",
                        f"{col}_rolling_std_{window}",
                        f"{col}_rolling_min_{window}",
                        f"{col}_rolling_max_{window}",
                    ]
                )

        for dt_feature in self.datetime_features:
            if self.cyclical_encoding and dt_feature in ["hour", "dayofweek", "month"]:
                self.feature_names_.extend([f"{dt_feature}_sin", f"{dt_feature}_cos"])
            else:
                self.feature_names_.append(dt_feature)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Extract temporal features from time series data."""
        check_is_fitted(self, ["feature_names_"])

        if self.validate_input:
            X, _ = self._validate_time_series(X, None)

        try:
            X_features = X.copy()
            self._add_lag_features(X, X_features)
            self._add_rolling_features(X, X_features)
            self._add_datetime_features(X, X_features)
            self._add_seasonal_features(X, X_features)

            logger.debug(f"Created {len(X_features.columns)} total features")
            return X_features

        except Exception as e:
            logger.error(f"Error in feature extraction: {e}")
            raise

    def _add_lag_features(self, X: pd.DataFrame, X_features: pd.DataFrame) -> None:
        """Add lag features to the feature DataFrame."""
        for col in X.columns:
            for lag in self.lag_features:
                X_features[f"{col}_lag_{lag}"] = X[col].shift(lag)

    def _add_rolling_features(self, X: pd.DataFrame, X_features: pd.DataFrame) -> None:
        """Add rolling window statistics to the feature DataFrame."""
        for col in X.columns:
            for window in self.rolling_windows:
                rolling = X[col].rolling(window=window, min_periods=1)
                X_features[f"{col}_rolling_mean_{window}"] = rolling.mean()
                X_features[f"{col}_rolling_std_{window}"] = rolling.std()
                X_features[f"{col}_rolling_min_{window}"] = rolling.min()
                X_features[f"{col}_rolling_max_{window}"] = rolling.max()

    def _add_datetime_features(self, X: pd.DataFrame, X_features: pd.DataFrame) -> None:
        """Add datetime component features to the feature DataFrame."""
        cyclical_periods = {"hour": 24, "dayofweek": 7, "month": 12}
        simple_extractors = {
            "quarter": lambda idx: idx.quarter,
            "year": lambda idx: idx.year,
            "dayofyear": lambda idx: idx.dayofyear,
            "weekofyear": lambda idx: idx.isocalendar().week,
        }

        for dt_feature in self.datetime_features:
            if dt_feature in cyclical_periods:
                values = getattr(X.index, dt_feature)
                if self.cyclical_encoding:
                    period = cyclical_periods[dt_feature]
                    X_features[f"{dt_feature}_sin"] = np.sin(
                        2 * np.pi * values / period
                    )
                    X_features[f"{dt_feature}_cos"] = np.cos(
                        2 * np.pi * values / period
                    )
                else:
                    X_features[dt_feature] = values
            elif dt_feature in simple_extractors:
                X_features[dt_feature] = simple_extractors[dt_feature](X.index)

    def _add_seasonal_features(self, X: pd.DataFrame, X_features: pd.DataFrame) -> None:
        """Add seasonal position features if seasonality is detected."""
        if not self.seasonal_features:
            return

        seasonality_info = self._detect_seasonality(X.iloc[:, 0])
        if seasonality_info["has_seasonality"]:
            period = seasonality_info["dominant_period"]
            position_in_season = np.arange(len(X)) % period
            X_features["seasonal_position_sin"] = np.sin(
                2 * np.pi * position_in_season / period
            )
            X_features["seasonal_position_cos"] = np.cos(
                2 * np.pi * position_in_season / period
            )
