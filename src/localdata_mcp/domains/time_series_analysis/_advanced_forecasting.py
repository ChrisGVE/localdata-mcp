"""
Time Series Analysis - Advanced Forecasting Transformer.

Contains the AdvancedForecastingTransformer which provides access to
Exponential Smoothing and Ensemble forecasting with automatic method
selection based on data characteristics.
"""

import time
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.utils.validation import check_is_fitted
from statsmodels.tsa.seasonal import seasonal_decompose

from ...logging_manager import get_logger
from ...pipeline.base import PipelineResult
from ._base import TimeSeriesValidationError
from ._transformer import TimeSeriesTransformer


class AdvancedForecastingTransformer(TimeSeriesTransformer):
    """
    sklearn-compatible transformer for advanced forecasting methods.

    Provides access to Exponential Smoothing and Ensemble forecasting
    with automatic method selection based on data characteristics and user preferences.

    Parameters:
    -----------
    method : str, default='auto'
        Forecasting method: 'exponential_smoothing', 'ensemble', 'auto'
    forecast_steps : int, default=10
        Number of steps to forecast ahead
    confidence_level : float, default=0.95
        Confidence level for prediction intervals
    ensemble_weights : dict, optional
        Weights for ensemble methods (auto-computed if None)
    holt_winters_params : dict, optional
        Additional parameters for Holt-Winters model
    """

    def __init__(
        self,
        method="auto",
        forecast_steps=10,
        confidence_level=0.95,
        ensemble_weights=None,
        holt_winters_params=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.method = method
        self.forecast_steps = forecast_steps
        self.confidence_level = confidence_level
        self.ensemble_weights = ensemble_weights or {}
        self.holt_winters_params = holt_winters_params or {}

        # Model instances
        self.exponential_smoothing_forecaster_ = None
        self.ensemble_forecaster_ = None
        self.evaluator_ = None
        self.selected_method_ = None
        self.training_data_ = None

    def _select_method(self, data: pd.Series) -> str:
        """
        Automatically select the best forecasting method based on data characteristics.

        Parameters:
        -----------
        data : pd.Series
            Time series data for analysis

        Returns:
        --------
        method : str
            Selected forecasting method
        """
        if self.method != "auto":
            return self.method

        # Analyze data characteristics
        n_observations = len(data)
        frequency = self._infer_frequency(pd.DataFrame({"value": data}))

        # Check for strong seasonality
        has_seasonality = self._detect_seasonality(data)

        # Selection logic
        if n_observations < 50:
            return "exponential_smoothing"  # Better for small datasets
        elif has_seasonality and n_observations >= 100:
            return "exponential_smoothing"  # Good for seasonal data
        elif n_observations >= 200:
            return "ensemble"  # Use ensemble for large datasets
        else:
            return "exponential_smoothing"

    def _detect_seasonality(self, data: pd.Series) -> bool:
        """
        Detect if the time series has significant seasonality.

        Parameters:
        -----------
        data : pd.Series
            Time series data

        Returns:
        --------
        has_seasonality : bool
            True if seasonality is detected
        """
        try:
            # Need at least 2 full cycles for seasonality detection
            min_periods = 24  # Default assumption
            if len(data) < 2 * min_periods:
                return False

            decomposition = seasonal_decompose(
                data, model="additive", period=min_periods
            )
            seasonal_strength = np.var(decomposition.seasonal) / np.var(data)

            return bool(
                seasonal_strength > 0.1
            )  # Threshold for significant seasonality
        except Exception:
            return False

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fit the advanced forecasting model."""
        if self.validate_input:
            X, y = self._validate_time_series(X, y)

        logger = get_logger(__name__)

        try:
            # Extract univariate time series
            if isinstance(X, pd.DataFrame):
                if X.shape[1] == 1:
                    data = X.iloc[:, 0]
                else:
                    raise ValueError(
                        "Advanced forecasting requires univariate time series data"
                    )
            elif y is not None:
                data = y
            else:
                raise ValueError("No target variable provided for forecasting")

            self.training_data_ = data

            # Select forecasting method
            self.selected_method_ = self._select_method(data)
            logger.info(f"Selected forecasting method: {self.selected_method_}")

            # Lazy imports to avoid circular dependencies:
            # ExponentialSmoothingForecaster, EnsembleForecaster, and
            # ForecastEvaluator may live in sibling modules or the monolith
            # while extraction is in progress.
            from ._exponential import ExponentialSmoothingForecaster

            # Initialize selected method(s)
            if self.selected_method_ == "exponential_smoothing":
                self.exponential_smoothing_forecaster_ = ExponentialSmoothingForecaster(
                    forecast_steps=self.forecast_steps,
                    confidence_level=self.confidence_level,
                    **self.holt_winters_params,
                )
                self.exponential_smoothing_forecaster_.fit(X, y)

            elif self.selected_method_ == "ensemble":
                # EnsembleForecaster and ForecastEvaluator have not been
                # extracted yet; import from the monolith.
                from ..time_series_analysis import EnsembleForecaster

                self.ensemble_forecaster_ = EnsembleForecaster(
                    forecast_steps=self.forecast_steps,
                    confidence_level=self.confidence_level,
                    weights=self.ensemble_weights,
                )
                self.ensemble_forecaster_.fit(X, y)

            # Initialize evaluator
            from ..time_series_analysis import ForecastEvaluator

            self.evaluator_ = ForecastEvaluator()

        except Exception as e:
            logger.error(f"Error in advanced forecasting fit: {e}")
            raise TimeSeriesValidationError(
                f"Advanced forecasting fit failed: {str(e)}"
            )

        return self

    def transform(self, X: pd.DataFrame) -> PipelineResult:
        """Generate forecasts using the fitted model."""
        check_is_fitted(self, ["selected_method_", "training_data_"])

        start_time = time.time()
        logger = get_logger(__name__)

        try:
            # Generate forecasts based on selected method
            if (
                self.selected_method_ == "exponential_smoothing"
                and self.exponential_smoothing_forecaster_
            ):
                result = self.exponential_smoothing_forecaster_.transform(X)
            elif self.selected_method_ == "ensemble" and self.ensemble_forecaster_:
                result = self.ensemble_forecaster_.transform(X)
            else:
                raise ValueError(
                    f"No fitted model available for method: {self.selected_method_}"
                )

            # Add method information to result
            result.data["selected_method"] = self.selected_method_
            result.data["available_methods"] = ["exponential_smoothing", "ensemble"]

            return result

        except Exception as e:
            logger.error(f"Error in advanced forecasting transform: {e}")
            raise TimeSeriesValidationError(
                f"Advanced forecasting transform failed: {str(e)}"
            )
