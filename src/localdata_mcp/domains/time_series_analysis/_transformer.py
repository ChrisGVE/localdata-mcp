"""
Time Series Analysis - Base transformer ABC.

Contains the TimeSeriesTransformer abstract base class that all concrete
time series transformers inherit from.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import timedelta

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from ...logging_manager import get_logger
from ._base import TimeSeriesAnalysisResult, TimeSeriesValidationError

logger = get_logger(__name__)


class TimeSeriesTransformer(BaseEstimator, TransformerMixin, ABC):
    """
    Abstract base class for sklearn-compatible time series transformers.

    Provides core functionality for temporal data validation, frequency detection,
    and time series specific preprocessing operations. All concrete time series
    transformers should inherit from this class.

    Parameters:
    -----------
    validate_input : bool, default=True
        Whether to validate time series input data
    infer_frequency : bool, default=True
        Whether to automatically infer time series frequency
    handle_missing : str, default='interpolate'
        Strategy for handling missing values: 'interpolate', 'forward_fill', 'drop'
    """

    def __init__(
        self, validate_input=True, infer_frequency=True, handle_missing="interpolate"
    ):
        self.validate_input = validate_input
        self.infer_frequency = infer_frequency
        self.handle_missing = handle_missing

    def _validate_time_series(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """Validate time series data structure and datetime index."""
        if not isinstance(X, pd.DataFrame):
            raise TimeSeriesValidationError("Input data must be a pandas DataFrame")

        # Ensure datetime index
        if not isinstance(X.index, pd.DatetimeIndex):
            X = self._convert_to_datetime_index(X)

        # Check for monotonic index
        if not X.index.is_monotonic_increasing:
            logger.warning("Time series index is not monotonic, sorting by datetime")
            X = X.sort_index()

        # Validate target series if provided
        if y is not None:
            y = self._validate_target_series(y)

        return X, y

    def _convert_to_datetime_index(self, X: pd.DataFrame) -> pd.DataFrame:
        """Convert DataFrame to use a datetime index."""
        datetime_col = next(
            (c for c in ["datetime", "date", "timestamp"] if c in X.columns),
            None,
        )
        if datetime_col:
            X = X.set_index(pd.to_datetime(X[datetime_col]))
            X = X.drop(columns=[datetime_col])
            return X

        try:
            X.index = pd.to_datetime(X.index)
        except (ValueError, TypeError) as e:
            raise TimeSeriesValidationError(f"Cannot convert index to datetime: {e}")
        return X

    def _validate_target_series(self, y: pd.Series) -> pd.Series:
        """Validate and convert target series to datetime index."""
        if not isinstance(y, pd.Series):
            raise TimeSeriesValidationError("Target data must be a pandas Series")
        if not isinstance(y.index, pd.DatetimeIndex):
            try:
                y.index = pd.to_datetime(y.index)
            except (ValueError, TypeError) as e:
                raise TimeSeriesValidationError(
                    f"Cannot convert target index to datetime: {e}"
                )
        return y

    def _infer_frequency(self, X: pd.DataFrame) -> Optional[str]:
        """Infer the frequency of a time series from its datetime index."""
        try:
            inferred_freq = pd.infer_freq(X.index)
            if inferred_freq:
                logger.debug(f"Inferred time series frequency: {inferred_freq}")
                return inferred_freq
        except Exception as e:
            logger.warning(f"Could not infer frequency: {e}")

        return self._fallback_frequency_detection(X)

    def _fallback_frequency_detection(self, X: pd.DataFrame) -> Optional[str]:
        """Detect frequency by examining time differences."""
        if len(X.index) <= 1:
            logger.warning("Could not determine time series frequency")
            return None

        diffs = X.index[1:] - X.index[:-1]
        most_common_diff = diffs.mode()
        if len(most_common_diff) == 0:
            return None

        freq_map = {
            timedelta(days=1): "D",
            timedelta(hours=1): "H",
            timedelta(minutes=1): "T",
            timedelta(seconds=1): "S",
        }
        return freq_map.get(most_common_diff[0])

    def _handle_missing_temporal_data(self, X: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in time series data using temporal-aware methods."""
        if not X.isnull().any().any():
            return X

        X_processed = X.copy()

        if self.handle_missing == "interpolate":
            X_processed = X_processed.interpolate(method="time")
        elif self.handle_missing == "forward_fill":
            X_processed = X_processed.fillna(method="ffill")
        elif self.handle_missing == "drop":
            X_processed = X_processed.dropna()
        else:
            logger.warning(f"Unknown missing value strategy: {self.handle_missing}")

        return X_processed

    def _detect_seasonality(
        self, series: pd.Series, max_period: int = 365
    ) -> Dict[str, Any]:
        """Detect seasonal patterns in time series data."""
        seasonality_info: Dict[str, Any] = {
            "has_seasonality": False,
            "dominant_period": None,
            "seasonal_strength": 0.0,
            "tested_periods": [],
        }

        try:
            freq = self._infer_frequency(pd.DataFrame(index=series.index))
            test_periods = self._get_test_periods(freq)
            test_periods = [
                p for p in test_periods if p < len(series) // 3 and p <= max_period
            ]
            seasonality_info["tested_periods"] = test_periods

            best_period = None
            best_strength = 0.0

            for period in test_periods:
                try:
                    autocorr = series.autocorr(lag=period)
                    if not np.isnan(autocorr) and abs(autocorr) > best_strength:
                        best_strength = abs(autocorr)
                        best_period = period
                except Exception as e:
                    logger.debug(f"Could not test period {period}: {e}")
                    continue

            if best_period is not None and best_strength > 0.3:
                seasonality_info["has_seasonality"] = True
                seasonality_info["dominant_period"] = best_period
                seasonality_info["seasonal_strength"] = best_strength

        except Exception as e:
            logger.warning(f"Error in seasonality detection: {e}")

        return seasonality_info

    @staticmethod
    def _get_test_periods(freq: Optional[str]) -> List[int]:
        """Return candidate seasonal periods based on frequency."""
        if freq is None:
            return [7, 12, 24, 52]
        period_map: Dict[str, List[int]] = {
            "D": [7, 30, 365],
            "H": [24, 168, 720],
            "M": [12],
        }
        return period_map.get(freq, [7, 12, 24, 52])

    def _calculate_data_quality_score(self, X: pd.DataFrame) -> float:
        """Calculate a data quality score for the time series (0.0 to 1.0)."""
        if X.empty:
            return 0.0

        score = 1.0

        # Penalise for missing values
        total_cells = X.shape[0] * max(X.shape[1], 1)
        missing_ratio = X.isnull().sum().sum() / total_cells
        score -= missing_ratio * 0.4

        # Penalise for very short series
        if len(X) < 50:
            score -= 0.2
        elif len(X) < 100:
            score -= 0.1

        # Penalise for constant columns
        if X.shape[1] > 0:
            constant_cols = (X.std() == 0).sum()
            if constant_cols > 0:
                score -= 0.1 * (constant_cols / X.shape[1])

        return max(0.0, min(1.0, score))

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fit the time series transformer."""
        pass

    @abstractmethod
    def transform(
        self, X: pd.DataFrame
    ) -> Union[pd.DataFrame, TimeSeriesAnalysisResult]:
        """Transform the time series data."""
        pass
