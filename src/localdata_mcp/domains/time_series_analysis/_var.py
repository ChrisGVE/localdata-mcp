"""
Time Series Analysis - VAR model forecaster.

Contains the VARModelForecaster class for Vector Autoregression modeling
and multivariate time series forecasting.
"""

import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.stats.diagnostic import acorr_ljungbox

from ...logging_manager import get_logger
from ._base import TimeSeriesAnalysisResult, TimeSeriesValidationError
from ._multivariate_base import MultivariateTimeSeriesTransformer

logger = get_logger(__name__)


class VARModelForecaster(MultivariateTimeSeriesTransformer):
    """
    Vector Autoregression (VAR) model for multivariate time series forecasting.

    VAR models capture linear interdependencies among multiple time series by
    allowing each variable to depend on its own lags and the lags of all other
    variables in the system. This implementation provides automated lag selection,
    forecasting with confidence intervals, and comprehensive model diagnostics.

    Key Features:
    - Automatic optimal lag selection using information criteria
    - Out-of-sample forecasting with confidence intervals
    - Comprehensive model diagnostics and residual analysis
    - Granger causality testing integration
    - Impulse response analysis capabilities
    - Streaming-compatible processing for large datasets

    Parameters:
    -----------
    max_lags : int, default=10
        Maximum number of lags to consider for model selection
    ic : str, default='aic'
        Information criterion for lag selection ('aic', 'bic', 'hqic', 'fpe')
    forecast_horizon : int, default=10
        Number of periods to forecast ahead
    confidence_level : float, default=0.95
        Confidence level for forecast intervals
    trend : str, default='c'
        Trend specification ('n', 'c', 'ct', 'ctt')

    Example:
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from localdata_mcp.domains.time_series_analysis import VARModelForecaster
    >>>
    >>> # Create sample multivariate time series
    >>> dates = pd.date_range('2020-01-01', periods=200, freq='D')
    >>> np.random.seed(42)
    >>> data = pd.DataFrame({
    ...     'series1': np.cumsum(np.random.randn(200)),
    ...     'series2': np.cumsum(np.random.randn(200)),
    ...     'series3': np.cumsum(np.random.randn(200))
    ... }, index=dates)
    >>>
    >>> # Fit VAR model and forecast
    >>> forecaster = VARModelForecaster(forecast_horizon=20)
    >>> result = forecaster.fit_transform(data)
    >>>
    >>> print(f"Optimal lags: {result.model_parameters['optimal_lags']}")
    >>> print(f"Forecast shape: {result.forecast_values.shape}")
    """

    def __init__(
        self,
        max_lags: int = 10,
        ic: str = "aic",
        forecast_horizon: int = 10,
        confidence_level: float = 0.95,
        trend: str = "c",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.max_lags = max_lags
        self.ic = ic.lower()
        self.forecast_horizon = forecast_horizon
        self.confidence_level = confidence_level
        self.trend = trend

        # Validate parameters
        valid_ic = ["aic", "bic", "hqic", "fpe"]
        if self.ic not in valid_ic:
            raise ValueError(f"Information criterion must be one of {valid_ic}")

        valid_trend = ["n", "c", "ct", "ctt"]
        if self.trend not in valid_trend:
            raise ValueError(f"Trend specification must be one of {valid_trend}")

    def _fit_var(self, X: pd.DataFrame):
        """Select optimal lag order and fit VAR model; return (var_fitted, optimal_lags)."""
        var_model = VAR(X)
        lag_order_results = var_model.select_order(maxlags=self.max_lags)
        optimal_lags = getattr(lag_order_results, self.ic)
        logger.info(
            f"Selected optimal lags: {optimal_lags} using {self.ic.upper()} criterion"
        )
        var_fitted = var_model.fit(maxlags=optimal_lags, trend=self.trend)
        return var_fitted, optimal_lags

    def _build_forecast_dataframes(self, X: pd.DataFrame, var_fitted, forecast_result):
        """Build forecast and confidence interval DataFrames."""
        last_date = X.index[-1]
        if isinstance(last_date, pd.Timestamp):
            forecast_index = pd.date_range(
                start=last_date + pd.Timedelta(days=1),
                periods=self.forecast_horizon,
                freq=pd.infer_freq(X.index) or "D",
            )
        else:
            forecast_index = range(len(X), len(X) + self.forecast_horizon)

        forecast_df = pd.DataFrame(
            forecast_result, index=forecast_index, columns=X.columns
        )

        forecast_stderr = var_fitted.forecast_cov(steps=self.forecast_horizon)
        alpha = 1 - self.confidence_level
        z_score = stats.norm.ppf(1 - alpha / 2)

        confidence_intervals: Dict = {}
        for i, col in enumerate(X.columns):
            std_errors = np.sqrt(
                [forecast_stderr[j][i, i] for j in range(self.forecast_horizon)]
            )
            confidence_intervals[f"{col}_lower"] = (
                forecast_result[:, i] - z_score * std_errors
            )
            confidence_intervals[f"{col}_upper"] = (
                forecast_result[:, i] + z_score * std_errors
            )

        confidence_df = pd.DataFrame(confidence_intervals, index=forecast_index)
        return forecast_df, confidence_df

    def _compute_fit_statistics(self, X: pd.DataFrame, var_fitted) -> Dict[str, float]:
        """Compute R-squared fit statistics from VAR residuals."""
        try:
            resid = var_fitted.resid
            y_actual = X.values[var_fitted.k_ar :]
            ss_res = np.sum(resid**2, axis=0)
            ss_tot = np.sum((y_actual - y_actual.mean(axis=0)) ** 2, axis=0)
            rsquared = 1 - ss_res / np.where(ss_tot == 0, 1, ss_tot)
            n, p = var_fitted.nobs, var_fitted.df_model
            rsquared_adj = 1 - (1 - rsquared) * (n - 1) / max(n - p - 1, 1)
        except Exception:
            rsquared = np.zeros(var_fitted.neqs)
            rsquared_adj = np.zeros(var_fitted.neqs)
        return {
            "rsquared_avg": float(np.mean(rsquared)),
            "rsquared_adj_avg": float(np.mean(rsquared_adj)),
        }

    def _analysis_logic(self, X: pd.DataFrame) -> TimeSeriesAnalysisResult:
        """Core VAR modeling and forecasting logic."""
        start_time = time.time()

        try:
            X = self._validate_multivariate_data(X)
            var_fitted, optimal_lags = self._fit_var(X)

            forecast_input = X.values[-var_fitted.k_ar :]
            forecast_result = var_fitted.forecast(
                forecast_input, steps=self.forecast_horizon
            )
            forecast_df, confidence_df = self._build_forecast_dataframes(
                X, var_fitted, forecast_result
            )
            fit_statistics = self._compute_fit_statistics(X, var_fitted)

            model_diagnostics = {
                "aic": var_fitted.aic,
                "bic": var_fitted.bic,
                "hqic": var_fitted.hqic,
                "fpe": var_fitted.fpe,
                "log_likelihood": var_fitted.llf,
                "determinant_cov": np.linalg.det(var_fitted.sigma_u),
                "n_observations": var_fitted.nobs,
                "n_equations": var_fitted.neqs,
                "n_coefficients": var_fitted.df_model,
            }
            model_parameters = {
                "optimal_lags": optimal_lags,
                "trend": self.trend,
                "ic_used": self.ic,
                "lag_order_results": None,
                "coefficients": var_fitted.params,
                "coefficient_stderr": var_fitted.stderr,
                "coefficient_pvalues": var_fitted.pvalues,
            }

            interpretation = self._generate_var_interpretation(
                var_fitted, optimal_lags, model_diagnostics, fit_statistics
            )
            recommendations = self._generate_var_recommendations(
                var_fitted, model_diagnostics, X
            )
            processing_time = time.time() - start_time

            return self._prepare_multivariate_result(
                analysis_type="VAR_forecasting",
                forecast_values=forecast_df,
                forecast_confidence_intervals=confidence_df,
                forecast_horizon=self.forecast_horizon,
                model_parameters=model_parameters,
                model_diagnostics=model_diagnostics,
                fit_statistics=fit_statistics,
                confidence_level=self.confidence_level,
                interpretation=interpretation,
                recommendations=recommendations,
                processing_time=processing_time,
                data_quality_score=self._calculate_data_quality_score(X),
            )

        except Exception as e:
            logger.error(f"VAR modeling failed: {e}")
            return self._prepare_multivariate_result(
                analysis_type="VAR_forecasting",
                interpretation=f"VAR modeling failed: {str(e)}",
                recommendations=["Check data quality and stationarity requirements"],
                processing_time=time.time() - start_time,
            )

    def _generate_var_interpretation(
        self,
        var_fitted,
        optimal_lags: int,
        diagnostics: Dict[str, Any],
        fit_stats: Dict[str, float],
    ) -> str:
        """
        Generate interpretation of VAR model results.

        Parameters:
        -----------
        var_fitted : VARResults
            Fitted VAR model
        optimal_lags : int
            Selected optimal lag order
        diagnostics : dict
            Model diagnostic statistics
        fit_stats : dict
            Model fit statistics

        Returns:
        --------
        interpretation : str
            Human-readable interpretation
        """
        n_series = var_fitted.neqs
        n_obs = var_fitted.nobs
        avg_rsq = fit_stats.get("rsquared_avg", 0)
        aic = diagnostics.get("aic", 0)

        interpretation = (
            f"VAR({optimal_lags}) model fitted to {n_series} time series "
            f"with {n_obs} observations. "
            f"Average R-squared: {avg_rsq:.3f}, AIC: {aic:.2f}. "
        )

        if avg_rsq > 0.7:
            interpretation += "Model shows good explanatory power. "
        elif avg_rsq > 0.5:
            interpretation += "Model shows moderate explanatory power. "
        else:
            interpretation += "Model shows limited explanatory power. "

        interpretation += f"Forecasted {self.forecast_horizon} periods ahead "
        interpretation += (
            f"with {self.confidence_level * 100:.0f}% confidence intervals."
        )

        return interpretation

    def _generate_var_recommendations(
        self, var_fitted, diagnostics: Dict[str, Any], X: pd.DataFrame
    ) -> List[str]:
        """
        Generate recommendations based on VAR model results.

        Parameters:
        -----------
        var_fitted : VARResults
            Fitted VAR model
        diagnostics : dict
            Model diagnostic statistics
        X : pd.DataFrame
            Original data

        Returns:
        --------
        recommendations : List[str]
            List of recommendations
        """
        recommendations = []

        # Check residual properties
        try:
            residuals = var_fitted.resid

            # Test for normality (Jarque-Bera test)
            normality_pvalues = []
            for col in range(residuals.shape[1]):
                _, p_val = stats.jarque_bera(residuals[:, col])
                normality_pvalues.append(p_val)

            if any(p < 0.05 for p in normality_pvalues):
                recommendations.append(
                    "Non-normal residuals detected - consider alternative error distributions"
                )

            # Test for autocorrelation
            autocorr_issues = []
            for col in range(residuals.shape[1]):
                ljung_result = acorr_ljungbox(
                    residuals[:, col], lags=min(10, len(residuals) // 4)
                )
                if any(ljung_result["lb_pvalue"] < 0.05):
                    autocorr_issues.append(col)

            if autocorr_issues:
                recommendations.append(
                    f"Residual autocorrelation detected in equations {autocorr_issues} - consider higher lag order"
                )

        except Exception as e:
            logger.warning(f"Could not perform residual diagnostics: {e}")

        # Check model stability
        try:
            eigenvalues = var_fitted.ma_rep(maxn=1).reshape(-1)
            if any(abs(ev) >= 1.0 for ev in eigenvalues):
                recommendations.append(
                    "VAR system may be unstable - check for unit roots or cointegration"
                )
        except Exception as e:
            logger.warning(f"Could not check VAR stability: {e}")

        # Data quality recommendations
        if len(X) < 100:
            recommendations.append(
                "Limited sample size - collect more data for robust VAR estimation"
            )

        if var_fitted.k_ar > len(X) / 10:
            recommendations.append(
                "High lag order relative to sample size - consider reducing lags"
            )

        if not recommendations:
            recommendations.append(
                "VAR model appears well-specified - continue with current specification"
            )

        return recommendations
