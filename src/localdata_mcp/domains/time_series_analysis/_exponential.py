"""
Time Series Analysis - Exponential Smoothing Forecaster.

Contains the ExponentialSmoothingForecaster implementing simple, double
(Holt's), and triple (Holt-Winters) exponential smoothing methods for
time series forecasting.
"""

import time
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.utils.validation import check_is_fitted
from statsmodels.tsa.seasonal import seasonal_decompose

from ...logging_manager import get_logger
from ...pipeline.base import PipelineResult, CompositionMetadata
from ._base import TimeSeriesValidationError
from ._transformer import TimeSeriesTransformer

logger = get_logger(__name__)


class ExponentialSmoothingForecaster(TimeSeriesTransformer):
    """
    sklearn-compatible transformer for Exponential Smoothing (Holt-Winters) forecasting.

    Implements simple, double (Holt's), and triple (Holt-Winters) exponential smoothing
    methods for time series forecasting. Automatically selects the best model
    configuration based on data characteristics.

    Parameters:
    -----------
    forecast_steps : int, default=10
        Number of steps to forecast ahead
    confidence_level : float, default=0.95
        Confidence level for prediction intervals
    trend : str or None, default='auto'
        Type of trend: None, 'add', 'mul', or 'auto'
    seasonal : str or None, default='auto'
        Type of seasonality: None, 'add', 'mul', or 'auto'
    seasonal_periods : int, default=None
        Number of periods in a complete seasonal cycle (auto-detected if None)
    damped_trend : bool, default=False
        Whether to use a damped trend
    smoothing_level : float, default=None
        Smoothing parameter for level (alpha). Auto-optimized if None
    smoothing_trend : float, default=None
        Smoothing parameter for trend (beta). Auto-optimized if None
    smoothing_seasonal : float, default=None
        Smoothing parameter for seasonal component (gamma). Auto-optimized if None
    use_boxcox : bool or str, default=False
        Whether to use Box-Cox transformation
    """

    def __init__(
        self,
        forecast_steps=10,
        confidence_level=0.95,
        trend="auto",
        seasonal="auto",
        seasonal_periods=None,
        damped_trend=False,
        smoothing_level=None,
        smoothing_trend=None,
        smoothing_seasonal=None,
        use_boxcox=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.forecast_steps = forecast_steps
        self.confidence_level = confidence_level
        self.trend = trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self.damped_trend = damped_trend
        self.smoothing_level = smoothing_level
        self.smoothing_trend = smoothing_trend
        self.smoothing_seasonal = smoothing_seasonal
        self.use_boxcox = use_boxcox

        self.model_ = None
        self.fitted_model_ = None
        self.training_data_ = None
        self.selected_trend_ = None
        self.selected_seasonal_ = None
        self.selected_seasonal_periods_ = None

    def _detect_seasonality_periods(self, data: pd.Series) -> int:
        """
        Detect the number of periods in a seasonal cycle.

        Parameters:
        -----------
        data : pd.Series
            Time series data with datetime index

        Returns:
        --------
        periods : int
            Number of periods in seasonal cycle
        """
        # Try to infer from frequency
        freq = self._infer_frequency(pd.DataFrame({"value": data}))

        if freq:
            freq_map = {
                "H": 24,  # Hourly -> daily seasonality
                "D": 7,  # Daily -> weekly seasonality
                "W": 52,  # Weekly -> yearly seasonality
                "M": 12,  # Monthly -> yearly seasonality
                "Q": 4,  # Quarterly -> yearly seasonality
                "T": 60,  # Minutes -> hourly seasonality
                "S": 60,  # Seconds -> minute seasonality
            }

            for f, periods in freq_map.items():
                if f in str(freq).upper():
                    return periods

        # Fallback: use autocorrelation to detect seasonality
        try:
            max_lag = min(len(data) // 3, 50)
            autocorr = [data.autocorr(lag=i) for i in range(1, max_lag + 1)]

            # Find the lag with maximum autocorrelation
            if autocorr:
                best_lag = np.argmax(np.abs(autocorr)) + 1
                if abs(autocorr[best_lag - 1]) > 0.3:  # Significant autocorrelation
                    return best_lag
        except Exception as e:
            logger.debug(f"Could not detect seasonality via autocorrelation: {e}")

        # Default fallback
        return 12

    def _select_model_components(self, data: pd.Series) -> Tuple[str, str, int]:
        """
        Automatically select trend and seasonality components.

        Parameters:
        -----------
        data : pd.Series
            Time series data

        Returns:
        --------
        trend : str
            Selected trend component
        seasonal : str
            Selected seasonal component
        seasonal_periods : int
            Selected seasonal periods
        """
        if self.trend != "auto" and self.seasonal != "auto":
            return (
                self.trend,
                self.seasonal,
                self.seasonal_periods or self._detect_seasonality_periods(data),
            )

        # Detect trend
        if self.trend == "auto":
            # Simple trend detection using linear regression
            x = np.arange(len(data))
            slope, _, r_value, _, _ = stats.linregress(x, data.values)

            if abs(r_value) > 0.7 and abs(slope) > data.std() * 0.01:
                # Check if multiplicative trend is better
                if data.min() > 0 and (data.max() / data.min()) > 2:
                    selected_trend = "mul"
                else:
                    selected_trend = "add"
            else:
                selected_trend = None
        else:
            selected_trend = self.trend

        # Detect seasonality
        if self.seasonal == "auto":
            seasonal_periods = (
                self.seasonal_periods or self._detect_seasonality_periods(data)
            )

            if len(data) >= 2 * seasonal_periods:
                try:
                    # Try seasonal decomposition to check for seasonality
                    decomposition = seasonal_decompose(
                        data, model="additive", period=seasonal_periods
                    )
                    seasonal_strength = np.var(decomposition.seasonal) / np.var(data)

                    if seasonal_strength > 0.1:
                        # Check if multiplicative seasonality is better
                        if data.min() > 0:
                            # Compare additive vs multiplicative seasonal patterns
                            try:
                                decomp_mul = seasonal_decompose(
                                    data,
                                    model="multiplicative",
                                    period=seasonal_periods,
                                )
                                seasonal_strength_mul = np.var(
                                    decomp_mul.seasonal
                                ) / np.var(data)

                                if seasonal_strength_mul > seasonal_strength * 1.1:
                                    selected_seasonal = "mul"
                                else:
                                    selected_seasonal = "add"
                            except:
                                selected_seasonal = "add"
                        else:
                            selected_seasonal = "add"
                    else:
                        selected_seasonal = None
                        seasonal_periods = None
                except Exception as e:
                    logger.debug(f"Seasonality detection failed: {e}")
                    selected_seasonal = None
                    seasonal_periods = None
            else:
                selected_seasonal = None
                seasonal_periods = None
        else:
            selected_seasonal = self.seasonal
            seasonal_periods = (
                self.seasonal_periods or self._detect_seasonality_periods(data)
            )

        return selected_trend, selected_seasonal, seasonal_periods

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fit the Exponential Smoothing forecasting model."""
        if self.validate_input:
            X, y = self._validate_time_series(X, y)

        logger = get_logger(__name__)

        try:
            # Import here to handle potential ImportError gracefully
            from scipy import stats
            from statsmodels.tsa.holtwinters import ExponentialSmoothing

            # Extract univariate time series
            if isinstance(X, pd.DataFrame):
                if X.shape[1] == 1:
                    data = X.iloc[:, 0]
                else:
                    raise ValueError(
                        "Exponential Smoothing requires univariate time series data"
                    )
            elif y is not None:
                data = y
            else:
                raise ValueError(
                    "No target variable provided for Exponential Smoothing"
                )

            self.training_data_ = data

            # Select model components automatically
            (
                self.selected_trend_,
                self.selected_seasonal_,
                self.selected_seasonal_periods_,
            ) = self._select_model_components(data)

            logger.info(
                f"Selected components - Trend: {self.selected_trend_}, "
                f"Seasonal: {self.selected_seasonal_}, Periods: {self.selected_seasonal_periods_}"
            )

            # Initialize Exponential Smoothing model
            model_params = {
                "trend": self.selected_trend_,
                "seasonal": self.selected_seasonal_,
                "seasonal_periods": self.selected_seasonal_periods_,
                "damped_trend": self.damped_trend,
                "use_boxcox": self.use_boxcox,
            }

            # Remove None values
            model_params = {k: v for k, v in model_params.items() if v is not None}

            self.model_ = ExponentialSmoothing(data, **model_params)

            # Fit model with optional smoothing parameters
            fit_params = {}
            if self.smoothing_level is not None:
                fit_params["smoothing_level"] = self.smoothing_level
            if self.smoothing_trend is not None:
                fit_params["smoothing_trend"] = self.smoothing_trend
            if self.smoothing_seasonal is not None:
                fit_params["smoothing_seasonal"] = self.smoothing_seasonal

            self.fitted_model_ = self.model_.fit(**fit_params)

            logger.info("Exponential Smoothing model fitted successfully")

        except Exception as e:
            logger.error(f"Error in Exponential Smoothing fitting: {e}")
            raise TimeSeriesValidationError(
                f"Exponential Smoothing fitting failed: {str(e)}"
            )

        return self

    def _get_confidence_intervals(
        self, forecast_result: pd.Series, logger
    ) -> pd.DataFrame:
        """Compute prediction intervals, falling back to residual-based estimates."""
        alpha = 1 - self.confidence_level
        try:
            forecast_summary = self.fitted_model_.get_forecast(
                steps=self.forecast_steps
            )
            ci = forecast_summary.conf_int(alpha=alpha)
            ci.columns = ["lower", "upper"]
            return ci
        except Exception as e:
            logger.warning(f"Could not generate prediction intervals: {e}")
            residuals = self.fitted_model_.resid
            std_residual = np.std(residuals)
            z_score = stats.norm.ppf(1 - alpha / 2)
            return pd.DataFrame(
                {
                    "lower": forecast_result - z_score * std_residual,
                    "upper": forecast_result + z_score * std_residual,
                },
                index=forecast_result.index,
            )

    def _build_model_parameters(self) -> dict:
        """Build the model_parameters dict from the fitted model."""
        params = {
            "smoothing_level": float(self.fitted_model_.params["smoothing_level"]),
            "aic": float(self.fitted_model_.aic),
            "bic": float(self.fitted_model_.bic),
            "sse": float(self.fitted_model_.sse),
        }
        if self.selected_trend_ and "smoothing_trend" in self.fitted_model_.params:
            params["smoothing_trend"] = float(
                self.fitted_model_.params["smoothing_trend"]
            )
        if (
            self.selected_seasonal_
            and "smoothing_seasonal" in self.fitted_model_.params
        ):
            params["smoothing_seasonal"] = float(
                self.fitted_model_.params["smoothing_seasonal"]
            )
        return params

    def transform(self, X: pd.DataFrame) -> PipelineResult:
        """Generate forecasts using the fitted Exponential Smoothing model."""
        check_is_fitted(self, ["fitted_model_", "training_data_"])

        start_time = time.time()
        logger = get_logger(__name__)

        try:
            forecast_result = self.fitted_model_.forecast(steps=self.forecast_steps)
            confidence_intervals = self._get_confidence_intervals(
                forecast_result, logger
            )
            interval_width = float(
                (confidence_intervals["upper"] - confidence_intervals["lower"]).mean()
            )
            model_parameters = self._build_model_parameters()

            result_data = {
                "forecast_method": "Exponential Smoothing (Holt-Winters)",
                "forecast_values": forecast_result.to_dict(),
                "confidence_intervals": confidence_intervals.to_dict(),
                "forecast_horizon": self.forecast_steps,
                "confidence_level": self.confidence_level,
                "model_components": {
                    "trend": self.selected_trend_,
                    "seasonal": self.selected_seasonal_,
                    "seasonal_periods": self.selected_seasonal_periods_,
                    "damped_trend": self.damped_trend,
                },
                "model_parameters": model_parameters,
                "diagnostics": {
                    "mean_interval_width": interval_width,
                    "training_data_points": len(self.training_data_),
                    "model_fit_quality": "good"
                    if self.fitted_model_.aic < len(self.training_data_) * 2
                    else "moderate",
                },
                "interpretation": self._generate_interpretation(
                    forecast_result, confidence_intervals
                ),
                "recommendations": self._generate_recommendations(),
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
                        "forecast_steps": self.forecast_steps,
                        "trend": self.selected_trend_,
                        "seasonal": self.selected_seasonal_,
                        "seasonal_periods": self.selected_seasonal_periods_,
                    },
                ),
            )

        except Exception as e:
            logger.error(f"Error in Exponential Smoothing forecasting: {e}")
            raise TimeSeriesValidationError(
                f"Exponential Smoothing forecasting failed: {str(e)}"
            )

    def _generate_interpretation(
        self, forecast_values: pd.Series, confidence_intervals: pd.DataFrame
    ) -> str:
        """
        Generate human-readable interpretation of Exponential Smoothing forecast results.

        Parameters:
        -----------
        forecast_values : pd.Series
            Forecasted values
        confidence_intervals : pd.DataFrame
            Confidence intervals for forecasts

        Returns:
        --------
        interpretation : str
            Human-readable interpretation
        """
        avg_forecast = forecast_values.mean()
        forecast_trend = (
            "increasing"
            if forecast_values.iloc[-1] > forecast_values.iloc[0]
            else "decreasing"
        )
        avg_interval_width = (
            confidence_intervals["upper"] - confidence_intervals["lower"]
        ).mean()

        model_type = "Simple"
        if self.selected_trend_ and self.selected_seasonal_:
            model_type = "Triple (Holt-Winters)"
        elif self.selected_trend_:
            model_type = "Double (Holt's)"

        interpretation = (
            f"{model_type} Exponential Smoothing forecast shows {forecast_trend} trend over {self.forecast_steps} periods. "
            f"Average predicted value: {avg_forecast:.2f}. "
            f"Prediction uncertainty: {'high' if avg_interval_width > abs(avg_forecast) * 0.2 else 'moderate'} "
            f"(interval width: {avg_interval_width:.2f})."
        )

        if self.selected_seasonal_:
            interpretation += f" Seasonal component detected with period {self.selected_seasonal_periods_}."

        return interpretation

    def _generate_recommendations(self) -> List[str]:
        """
        Generate recommendations based on the Exponential Smoothing model.

        Returns:
        --------
        recommendations : List[str]
            List of recommendations for the user
        """
        recommendations = []

        if len(self.training_data_) < 30:
            recommendations.append(
                "Consider collecting more historical data for improved forecast accuracy"
            )

        if self.fitted_model_.aic > len(self.training_data_) * 3:
            recommendations.append(
                "High AIC value suggests model may be overfitting - consider simpler configuration"
            )

        if self.selected_seasonal_ is None and len(self.training_data_) > 24:
            recommendations.append(
                "No seasonality detected - verify if seasonal patterns exist in your data"
            )

        if self.selected_trend_ is None:
            recommendations.append(
                "No trend detected - consider if the data has underlying growth patterns"
            )

        recommendations.append(
            "Exponential smoothing works well for data with clear trends and seasonal patterns"
        )

        return recommendations
