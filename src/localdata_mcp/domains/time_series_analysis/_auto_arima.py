"""
Time Series Analysis - Auto-ARIMA Transformer.

Contains the AutoARIMATransformer for automatic ARIMA parameter selection
using information criteria (AIC/BIC) through grid search.
"""

import time
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.utils.validation import check_is_fitted
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox

from ...logging_manager import get_logger
from ...pipeline.base import PipelineResult, CompositionMetadata
from ._base import TimeSeriesAnalysisResult, TimeSeriesValidationError
from ._transformer import TimeSeriesTransformer


class AutoARIMATransformer(TimeSeriesTransformer):
    """
    sklearn-compatible transformer for automatic ARIMA parameter selection.

    Automatically selects optimal ARIMA parameters using information criteria
    (AIC/BIC) through grid search over specified parameter ranges.

    Parameters:
    -----------
    max_p : int, default=5
        Maximum number of AR terms to test
    max_d : int, default=2
        Maximum number of differences to test
    max_q : int, default=5
        Maximum number of MA terms to test
    max_P : int, default=2
        Maximum number of seasonal AR terms to test
    max_D : int, default=1
        Maximum number of seasonal differences to test
    max_Q : int, default=2
        Maximum number of seasonal MA terms to test
    seasonal_period : int, default=None
        Seasonal period (auto-detected if None)
    information_criterion : str, default='aic'
        Information criterion for model selection ('aic' or 'bic')
    seasonal : bool, default=True
        Whether to consider seasonal models
    stepwise : bool, default=True
        Whether to use stepwise search (faster) or exhaustive search
    forecast_steps : int, default=10
        Number of steps to forecast ahead
    alpha : float, default=0.05
        Significance level for confidence intervals
    """

    def __init__(
        self,
        max_p=5,
        max_d=2,
        max_q=5,
        max_P=2,
        max_D=1,
        max_Q=2,
        seasonal_period=None,
        information_criterion="aic",
        seasonal=True,
        stepwise=True,
        forecast_steps=10,
        alpha=0.05,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.max_p = max_p
        self.max_d = max_d
        self.max_q = max_q
        self.max_P = max_P
        self.max_D = max_D
        self.max_Q = max_Q
        self.seasonal_period = seasonal_period
        self.information_criterion = information_criterion
        self.seasonal = seasonal
        self.stepwise = stepwise
        self.forecast_steps = forecast_steps
        self.alpha = alpha
        self.best_model_ = None
        self.best_order_ = None
        self.best_seasonal_order_ = None
        self.model_results_ = []
        self.training_data_ = None

    def _detect_seasonal_period(self, data):
        """Detect seasonal period from data frequency."""
        if hasattr(data, "index") and isinstance(data.index, pd.DatetimeIndex):
            freq = data.index.freq or pd.infer_freq(data.index)
            if freq:
                # Common seasonal periods based on frequency
                freq_str = str(freq).upper()
                if "H" in freq_str:  # Hourly
                    return 24
                elif "D" in freq_str:  # Daily
                    return 7
                elif "W" in freq_str:  # Weekly
                    return 52
                elif "M" in freq_str:  # Monthly
                    return 12
                elif "Q" in freq_str:  # Quarterly
                    return 4

        # Default seasonal period if cannot detect
        return 12

    def _fit_candidate_model(self, data, order, seasonal_order):
        """Fit a single ARIMA/SARIMAX candidate; return (fitted_model, ic_value) or None."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            if self.seasonal and seasonal_order != (0, 0, 0, 0):
                model = SARIMAX(
                    data,
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=True,
                    enforce_invertibility=True,
                )
            else:
                model = ARIMA(
                    data,
                    order=order,
                    enforce_stationarity=True,
                    enforce_invertibility=True,
                )
            fitted = model.fit()
            ic_value = fitted.aic if self.information_criterion == "aic" else fitted.bic
            return fitted, ic_value

    def _search_best_model(self, data, param_combinations, logger):
        """Search parameter combinations; populate model_results_ and return best_ic."""
        best_ic = np.inf
        logger.info(f"Testing {len(param_combinations)} ARIMA parameter combinations")
        for order, seasonal_order in param_combinations:
            try:
                fitted, ic_value = self._fit_candidate_model(
                    data, order, seasonal_order
                )
                self.model_results_.append(
                    {
                        "order": order,
                        "seasonal_order": seasonal_order,
                        "aic": fitted.aic,
                        "bic": fitted.bic,
                        "log_likelihood": fitted.llf,
                        "selected_ic": ic_value,
                    }
                )
                if ic_value < best_ic:
                    best_ic = ic_value
                    self.best_model_ = fitted
                    self.best_order_ = order
                    self.best_seasonal_order_ = seasonal_order
            except Exception as e:
                logger.debug(f"Failed to fit model {order}, {seasonal_order}: {e}")
        return best_ic

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fit the Auto-ARIMA model by searching optimal parameters."""
        if self.validate_input:
            X, y = self._validate_time_series(X, y)

        logger = get_logger(__name__)

        try:
            if isinstance(X, pd.DataFrame):
                if X.shape[1] == 1:
                    data = X.iloc[:, 0]
                else:
                    raise ValueError("Auto-ARIMA requires univariate time series data")
            elif y is not None:
                data = y
            else:
                raise ValueError("No target variable provided for Auto-ARIMA modeling")

            self.training_data_ = data

            if self.seasonal and self.seasonal_period is None:
                self.seasonal_period = self._detect_seasonal_period(data)

            param_combinations = (
                self._stepwise_search() if self.stepwise else self._exhaustive_search()
            )
            self.model_results_ = []
            best_ic = self._search_best_model(data, param_combinations, logger)

            if self.best_model_ is None:
                raise ValueError("No valid ARIMA model could be fitted to the data")

            logger.info(
                f"Best model found: ARIMA{self.best_order_} x {self.best_seasonal_order_} "
                f"with {self.information_criterion.upper()}={best_ic:.2f}"
            )

        except Exception as e:
            logger.error(f"Error in Auto-ARIMA fitting: {e}")
            raise

        return self

    def _stepwise_search(self):
        """Generate parameter combinations using stepwise approach."""
        combinations = []

        # Start with simple models and expand
        for d in range(self.max_d + 1):
            for p in range(min(3, self.max_p + 1)):
                for q in range(min(3, self.max_q + 1)):
                    if self.seasonal:
                        for D in range(self.max_D + 1):
                            for P in range(min(2, self.max_P + 1)):
                                for Q in range(min(2, self.max_Q + 1)):
                                    seasonal_order = (P, D, Q, self.seasonal_period)
                                    combinations.append(((p, d, q), seasonal_order))
                    else:
                        combinations.append(((p, d, q), (0, 0, 0, 0)))

        return combinations[:50]  # Limit to first 50 combinations for efficiency

    def _exhaustive_search(self):
        """Generate all parameter combinations for exhaustive search."""
        combinations = []

        for p in range(self.max_p + 1):
            for d in range(self.max_d + 1):
                for q in range(self.max_q + 1):
                    if self.seasonal:
                        for P in range(self.max_P + 1):
                            for D in range(self.max_D + 1):
                                for Q in range(self.max_Q + 1):
                                    seasonal_order = (P, D, Q, self.seasonal_period)
                                    combinations.append(((p, d, q), seasonal_order))
                    else:
                        combinations.append(((p, d, q), (0, 0, 0, 0)))

        return combinations

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
        """Run Ljung-Box test and return diagnostics dict."""
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
        }, last_row["lb_pvalue"]

    def _build_auto_arima_recommendations(self, ljung_box_pvalue: float):
        """Build recommendations list based on search results and diagnostics."""
        recommendations = []
        if len(self.model_results_) < 10:
            recommendations.append(
                "Limited parameter search - consider expanding search ranges"
            )
        if ljung_box_pvalue <= 0.05:
            recommendations.append(
                "Residuals show autocorrelation - model may need refinement"
            )
        return recommendations

    def transform(self, X: pd.DataFrame) -> TimeSeriesAnalysisResult:
        """Generate forecasts using the best Auto-ARIMA model."""
        start_time = time.time()
        logger = get_logger(__name__)

        try:
            check_is_fitted(self, "best_model_")

            if self.validate_input:
                X, _ = self._validate_time_series(X)

            forecast_result = self.best_model_.get_forecast(steps=self.forecast_steps)
            forecast_values = forecast_result.predicted_mean
            forecast_ci = forecast_result.conf_int(alpha=self.alpha)
            forecast_index = self._build_forecast_index()

            residuals = self.best_model_.resid
            residual_diagnostics, ljung_pvalue = self._compute_residual_diagnostics(
                residuals
            )
            in_sample_pred = self.best_model_.fittedvalues
            sorted_models = sorted(self.model_results_, key=lambda x: x["selected_ic"])

            result_data = {
                "model_type": "Auto-ARIMA",
                "best_order": self.best_order_,
                "best_seasonal_order": self.best_seasonal_order_,
                "selection_criterion": self.information_criterion.upper(),
                "models_tested": len(self.model_results_),
                "forecast_values": forecast_values.tolist(),
                "forecast_index": forecast_index.tolist()
                if hasattr(forecast_index, "tolist")
                else list(forecast_index),
                "forecast_lower_ci": forecast_ci.iloc[:, 0].tolist(),
                "forecast_upper_ci": forecast_ci.iloc[:, 1].tolist(),
                "confidence_level": 1 - self.alpha,
                "in_sample_predictions": in_sample_pred.tolist(),
                "residuals": residuals.tolist(),
                "best_model_fit": {
                    "aic": float(self.best_model_.aic),
                    "bic": float(self.best_model_.bic),
                    "log_likelihood": float(self.best_model_.llf),
                    "num_params": self.best_model_.params.shape[0],
                },
                "residual_diagnostics": residual_diagnostics,
                "model_comparison": sorted_models[:10],
                "model_params": {
                    param: float(value)
                    for param, value in self.best_model_.params.items()
                },
                "recommendations": self._build_auto_arima_recommendations(ljung_pvalue),
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
                        "max_p": self.max_p,
                        "max_d": self.max_d,
                        "max_q": self.max_q,
                        "seasonal": self.seasonal,
                        "information_criterion": self.information_criterion,
                    },
                ),
            )

        except Exception as e:
            logger.error(f"Error in Auto-ARIMA forecasting: {e}")
            raise TimeSeriesValidationError(f"Auto-ARIMA forecasting failed: {str(e)}")
