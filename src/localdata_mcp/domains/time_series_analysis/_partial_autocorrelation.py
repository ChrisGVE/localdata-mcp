"""
Partial Autocorrelation (PACF) analysis transformer for time series data.
"""

import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import pacf

from ._base import TimeSeriesAnalysisResult
from ._transformer import TimeSeriesTransformer
from ...logging_manager import get_logger

logger = get_logger(__name__)


class PartialAutocorrelationAnalysisTransformer(TimeSeriesTransformer):
    """
    sklearn-compatible transformer for partial autocorrelation function (PACF) analysis.

    Computes and analyzes the partial autocorrelation function of time series data,
    useful for identifying the order of autoregressive models.

    Parameters:
    -----------
    max_lags : int, optional
        Maximum number of lags to compute. If None, uses min(40, len(series)//4)
    alpha : float, default=0.05
        Significance level for correlation tests
    method : str, default='ywmle'
        Method for PACF computation: 'ywmle', 'ols'
    """

    def __init__(self, max_lags=None, alpha=0.05, method="ywmle", **kwargs):
        super().__init__(**kwargs)
        self.max_lags = max_lags
        self.alpha = alpha
        self.method = method
        self.pacf_values_ = None
        self.confidence_intervals_ = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fit the PACF analysis transformer."""
        if self.validate_input:
            X, y = self._validate_time_series(X, y)
        self.is_fitted_ = True
        return self

    def _compute_pacf_nlags(self, series: pd.Series) -> int:
        """Determine number of lags to compute."""
        if self.max_lags is None:
            return min(40, len(series) // 4)
        return min(self.max_lags, len(series) - 1)

    def _classify_significant_pacf_lags(
        self, pacf_values: np.ndarray, confint: np.ndarray
    ) -> List[Dict]:
        """Build list of lag dicts with significance flags from PACF values and CI."""
        significant_lags = []
        for lag in range(1, len(pacf_values)):
            is_significant = abs(pacf_values[lag]) > abs(
                confint[lag, 1] - pacf_values[lag]
            )
            significant_lags.append(
                {
                    "lag": lag,
                    "partial_correlation": pacf_values[lag],
                    "is_significant": is_significant,
                    "confidence_lower": confint[lag, 0],
                    "confidence_upper": confint[lag, 1],
                }
            )
        return significant_lags

    def _interpret_pacf(self, ar_order_analysis: Dict) -> tuple:
        """Return (interpretation, recommendations, warnings_list) from AR order analysis."""
        recommendations = []
        warnings_list = []
        if ar_order_analysis["suggested_order"] == 0:
            interpretation = "No significant partial autocorrelations - series may be MA or white noise"
            recommendations.append("Consider MA model or simple forecasting methods")
        else:
            interpretation = f"Significant partial autocorrelations up to lag {ar_order_analysis['suggested_order']}"
            recommendations.append(
                f"Consider AR({ar_order_analysis['suggested_order']}) model"
            )
            if ar_order_analysis["seasonal_order"] > 0:
                recommendations.append(
                    f"Seasonal AR component suggested: P={ar_order_analysis['seasonal_order']}"
                )
        recommendations.extend(ar_order_analysis.get("recommendations", []))
        warnings_list.extend(ar_order_analysis.get("warnings", []))
        return interpretation, recommendations, warnings_list

    def transform(self, X: pd.DataFrame) -> TimeSeriesAnalysisResult:
        """
        Compute partial autocorrelation function analysis.

        Returns:
        --------
        result : TimeSeriesAnalysisResult
            PACF analysis results with AR order recommendations
        """
        start_time = time.time()

        if self.validate_input:
            X, _ = self._validate_time_series(X, None)

        try:
            if len(X.columns) == 0:
                raise ValueError("No columns found in time series data")

            series = X.iloc[:, 0].dropna()
            if len(series) < 10:
                raise ValueError(
                    "Insufficient data points for partial autocorrelation analysis"
                )

            nlags = self._compute_pacf_nlags(series)
            pacf_values, confint = pacf(
                series, nlags=nlags, alpha=self.alpha, method=self.method
            )
            self.pacf_values_ = pacf_values
            self.confidence_intervals_ = confint

            significant_lags = self._classify_significant_pacf_lags(
                pacf_values, confint
            )
            ar_order_analysis = self._analyze_ar_order(pacf_values, significant_lags)
            interpretation, recommendations, warnings_list = self._interpret_pacf(
                ar_order_analysis
            )

            return TimeSeriesAnalysisResult(
                analysis_type="partial_autocorrelation_analysis",
                interpretation=interpretation,
                model_diagnostics={
                    "pacf_values": pacf_values.tolist(),
                    "confidence_intervals": confint.tolist(),
                    "significant_lags": significant_lags,
                    "ar_order_analysis": ar_order_analysis,
                    "max_lags_computed": nlags,
                    "series_length": len(series),
                },
                recommendations=recommendations,
                warnings=warnings_list,
                processing_time=time.time() - start_time,
            )

        except Exception as e:
            logger.error(f"Error in partial autocorrelation analysis: {e}")
            return TimeSeriesAnalysisResult(
                analysis_type="partial_autocorrelation_analysis_error",
                interpretation=f"Error during partial autocorrelation analysis: {str(e)}",
                warnings=[str(e)],
                processing_time=time.time() - start_time,
            )

    def _analyze_ar_order(
        self, pacf_values: np.ndarray, significant_lags: List[Dict]
    ) -> Dict[str, Any]:
        """
        Analyze PACF to suggest AR model order.

        Parameters:
        -----------
        pacf_values : np.ndarray
            Partial autocorrelation function values
        significant_lags : list of dict
            Information about significant lags

        Returns:
        --------
        ar_analysis : dict
            AR order analysis and recommendations
        """
        analysis = {
            "suggested_order": 0,
            "seasonal_order": 0,
            "cutoff_pattern": "none",
            "recommendations": [],
            "warnings": [],
        }

        try:
            # Find the last significant lag (AR order suggestion)
            significant_lag_numbers = [
                lag["lag"] for lag in significant_lags if lag["is_significant"]
            ]

            if len(significant_lag_numbers) == 0:
                analysis["suggested_order"] = 0
                analysis["cutoff_pattern"] = "immediate"
                analysis["recommendations"].append("No AR component suggested")
            else:
                # Traditional approach: find where PACF cuts off
                # Look for clear cutoff pattern
                last_significant = max(significant_lag_numbers)

                # Check if there's a clear cutoff (most recent significant lags)
                recent_significant = [
                    lag for lag in significant_lag_numbers if lag <= 10
                ]

                if len(recent_significant) > 0:
                    # Suggest order based on last consecutive significant lag
                    consecutive_order = self._find_consecutive_cutoff(significant_lags)
                    analysis["suggested_order"] = consecutive_order

                    if consecutive_order <= 3:
                        analysis["cutoff_pattern"] = "clear"
                        analysis["recommendations"].append(
                            f"Clear PACF cutoff suggests AR({consecutive_order})"
                        )
                    else:
                        analysis["cutoff_pattern"] = "gradual"
                        analysis["recommendations"].append(
                            "Gradual PACF decay - consider ARMA model"
                        )
                        analysis["suggested_order"] = min(
                            3, consecutive_order
                        )  # Cap at reasonable order
                else:
                    analysis["suggested_order"] = last_significant
                    analysis["cutoff_pattern"] = "unclear"
                    analysis["warnings"].append(
                        "PACF pattern unclear - validate model selection"
                    )

                # Check for seasonal patterns
                seasonal_lags = [lag for lag in significant_lag_numbers if lag >= 4]
                seasonal_periods = self._detect_seasonal_pacf_periods(seasonal_lags)

                if seasonal_periods:
                    analysis["seasonal_order"] = 1
                    analysis["recommendations"].append(
                        f"Seasonal patterns detected at lags: {seasonal_periods}"
                    )

        except Exception as e:
            analysis["warnings"].append(f"Error in AR order analysis: {e}")

        return analysis

    def _find_consecutive_cutoff(self, significant_lags: List[Dict]) -> int:
        """Find the order where PACF cuts off based on consecutive significant lags."""
        # Find the longest sequence of consecutive significant lags starting from lag 1
        order = 0

        for lag_info in sorted(significant_lags, key=lambda x: x["lag"]):
            if lag_info["lag"] == order + 1 and lag_info["is_significant"]:
                order = lag_info["lag"]
            else:
                break

        return order

    def _detect_seasonal_pacf_periods(self, seasonal_lags: List[int]) -> List[int]:
        """Detect seasonal periods in PACF significant lags."""
        periods = []

        for period in [4, 7, 12, 24, 52]:
            seasonal_matches = [lag for lag in seasonal_lags if lag % period == 0]
            if len(seasonal_matches) >= 1:
                periods.append(period)

        return periods
