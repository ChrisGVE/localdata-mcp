"""
Time Series Analysis - ARIMA Forecast Transformer.

Contains the ARIMAForecastTransformer for AutoRegressive Integrated
Moving Average (ARIMA) time series forecasting.
"""

import time
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.utils.validation import check_is_fitted
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox

from ...logging_manager import get_logger
from ...pipeline.base import PipelineResult, CompositionMetadata
from ._base import TimeSeriesAnalysisResult, TimeSeriesValidationError
from ._transformer import TimeSeriesTransformer


class ARIMAForecastTransformer(TimeSeriesTransformer):
    """
    sklearn-compatible transformer for ARIMA time series forecasting.

    Implements AutoRegressive Integrated Moving Average (ARIMA) models
    for time series prediction with confidence intervals.

    Parameters:
    -----------
    order : tuple of int, default=(1, 1, 1)
        The (p, d, q) order of the model for the number of AR parameters,
        differences, and MA parameters
    seasonal_order : tuple of int, default=(0, 0, 0, 0)
        The (P, D, Q, s) order of the seasonal component
    trend : str, default='c'
        Parameter controlling the deterministic trend polynomial
    enforce_stationarity : bool, default=True
        Whether or not to transform AR parameters to enforce stationarity
    enforce_invertibility : bool, default=True
        Whether or not to transform MA parameters to enforce invertibility
    forecast_steps : int, default=10
        Number of steps to forecast ahead
    alpha : float, default=0.05
        The significance level for confidence intervals (1 - alpha = confidence level)
    """

    def __init__(
        self,
        order=(1, 1, 1),
        seasonal_order=(0, 0, 0, 0),
        trend="c",
        enforce_stationarity=True,
        enforce_invertibility=True,
        forecast_steps=10,
        alpha=0.05,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.order = order
        self.seasonal_order = seasonal_order
        self.trend = trend
        self.enforce_stationarity = enforce_stationarity
        self.enforce_invertibility = enforce_invertibility
        self.forecast_steps = forecast_steps
        self.alpha = alpha
        self.model_ = None
        self.fitted_model_ = None
        self.training_data_ = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fit the ARIMA model to the time series data."""
        if self.validate_input:
            X, y = self._validate_time_series(X, y)

        try:
            # Use the first column if X is a DataFrame, otherwise use y
            if isinstance(X, pd.DataFrame):
                if X.shape[1] == 1:
                    data = X.iloc[:, 0]
                else:
                    raise ValueError("ARIMA model requires univariate time series data")
            elif y is not None:
                data = y
            else:
                raise ValueError("No target variable provided for ARIMA modeling")

            # Store training data for forecasting
            self.training_data_ = data

            # Create and fit ARIMA model
            self.model_ = ARIMA(
                data,
                order=self.order,
                seasonal_order=self.seasonal_order,
                trend=self.trend,
                enforce_stationarity=self.enforce_stationarity,
                enforce_invertibility=self.enforce_invertibility,
            )

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                self.fitted_model_ = self.model_.fit()

        except Exception as e:
            logger = get_logger(__name__)
            logger.error(f"Error fitting ARIMA model: {e}")
            raise

        return self

    def _build_forecast_index(self):
        """Build a forecast index based on the training data's time index."""
        if hasattr(self.training_data_, "index") and isinstance(
            self.training_data_.index, pd.DatetimeIndex
        ):
            last_date = self.training_data_.index[-1]
            freq = self.training_data_.index.freq or pd.infer_freq(
                self.training_data_.index
            )
            if freq is not None:
                return pd.date_range(
                    start=last_date + pd.Timedelta(freq),
                    periods=self.forecast_steps,
                    freq=freq,
                )
        return range(
            len(self.training_data_), len(self.training_data_) + self.forecast_steps
        )

    def _compute_residual_diagnostics(self, residuals):
        """Run Ljung-Box test and return (diagnostics_dict, last_pvalue)."""
        ljung_box = acorr_ljungbox(
            residuals, lags=min(10, len(residuals) // 5), return_df=True
        )
        last_row = ljung_box.iloc[-1]
        return {
            "ljung_box_stat": float(last_row["lb_stat"]),
            "ljung_box_pvalue": float(last_row["lb_pvalue"]),
            "residual_autocorrelation": (
                "No significant autocorrelation"
                if last_row["lb_pvalue"] > 0.05
                else "Significant autocorrelation detected"
            ),
        }, float(last_row["lb_pvalue"])

    def transform(self, X: pd.DataFrame) -> TimeSeriesAnalysisResult:
        """Generate ARIMA forecasts and model diagnostics."""
        start_time = time.time()
        logger = get_logger(__name__)

        try:
            check_is_fitted(self, "fitted_model_")

            if self.validate_input:
                X, _ = self._validate_time_series(X)

            forecast_result = self.fitted_model_.get_forecast(steps=self.forecast_steps)
            forecast_values = forecast_result.predicted_mean
            forecast_ci = forecast_result.conf_int(alpha=self.alpha)
            forecast_index = self._build_forecast_index()

            residuals = self.fitted_model_.resid
            residual_diagnostics, ljung_pvalue = self._compute_residual_diagnostics(
                residuals
            )
            in_sample_pred = self.fitted_model_.fittedvalues

            aic = self.fitted_model_.aic
            recommendations = []
            if ljung_pvalue <= 0.05:
                recommendations.append(
                    "Residuals show significant autocorrelation - consider adjusting model parameters"
                )
            if aic > 1000:
                recommendations.append("High AIC value - model may be overfitting")

            result_data = {
                "model_type": "ARIMA",
                "order": self.order,
                "seasonal_order": self.seasonal_order,
                "forecast_values": forecast_values.tolist(),
                "forecast_index": forecast_index.tolist()
                if hasattr(forecast_index, "tolist")
                else list(forecast_index),
                "forecast_lower_ci": forecast_ci.iloc[:, 0].tolist(),
                "forecast_upper_ci": forecast_ci.iloc[:, 1].tolist(),
                "confidence_level": 1 - self.alpha,
                "in_sample_predictions": in_sample_pred.tolist(),
                "residuals": residuals.tolist(),
                "model_fit": {
                    "aic": float(aic),
                    "bic": float(self.fitted_model_.bic),
                    "log_likelihood": float(self.fitted_model_.llf),
                    "num_params": self.fitted_model_.params.shape[0],
                },
                "residual_diagnostics": residual_diagnostics,
                "model_params": {
                    param: float(value)
                    for param, value in self.fitted_model_.params.items()
                },
                "recommendations": recommendations,
            }

            return PipelineResult(
                success=True,
                data=result_data,
                metadata={},
                execution_time_seconds=time.time() - start_time,
                memory_used_mb=0.0,
                pipeline_stage="forecast",
                composition_metadata=CompositionMetadata(
                    domain="time_series",
                    analysis_type="forecast",
                    result_type="predictions",
                    data_artifacts={
                        "order": list(self.order),
                        "seasonal_order": list(self.seasonal_order),
                        "forecast_steps": self.forecast_steps,
                    },
                ),
            )

        except Exception as e:
            logger.error(f"Error in ARIMA forecasting: {e}")
            raise TimeSeriesValidationError(f"ARIMA forecasting failed: {str(e)}")
