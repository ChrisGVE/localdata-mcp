"""
Autocorrelation (ACF) analysis transformer for time series data.
"""

import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf

from ._base import TimeSeriesAnalysisResult
from ._transformer import TimeSeriesTransformer
from ...logging_manager import get_logger

logger = get_logger(__name__)


class AutocorrelationAnalysisTransformer(TimeSeriesTransformer):
    """
    sklearn-compatible transformer for autocorrelation function (ACF) analysis.

    Computes and analyzes the autocorrelation function of time series data,
    including significance testing and lag selection recommendations.

    Parameters:
    -----------
    max_lags : int, optional
        Maximum number of lags to compute. If None, uses min(40, len(series)//4)
    alpha : float, default=0.05
        Significance level for correlation tests
    fft : bool, default=True
        Whether to use FFT for computation (faster for long series)
    missing : str, default='none'
        How to handle missing values: 'none', 'drop', 'conservative'
    """

    def __init__(self, max_lags=None, alpha=0.05, fft=True, missing="none", **kwargs):
        super().__init__(**kwargs)
        self.max_lags = max_lags
        self.alpha = alpha
        self.fft = fft
        self.missing = missing
        self.acf_values_ = None
        self.confidence_intervals_ = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fit the ACF analysis transformer."""
        if self.validate_input:
            X, y = self._validate_time_series(X, y)
        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> TimeSeriesAnalysisResult:
        """
        Compute autocorrelation function analysis.

        Returns:
        --------
        result : TimeSeriesAnalysisResult
            ACF analysis results with lag recommendations
        """
        start_time = time.time()

        if self.validate_input:
            X, _ = self._validate_time_series(X, None)

        try:
            # Use first column for analysis
            if len(X.columns) == 0:
                raise ValueError("No columns found in time series data")

            series = X.iloc[:, 0].dropna()
            if len(series) < 10:
                raise ValueError(
                    "Insufficient data points for autocorrelation analysis"
                )

            # Determine number of lags
            if self.max_lags is None:
                nlags = min(40, len(series) // 4)
            else:
                nlags = min(self.max_lags, len(series) - 1)

            # Compute ACF
            acf_values, confint = acf(
                series,
                nlags=nlags,
                alpha=self.alpha,
                fft=self.fft,
                missing=self.missing,
            )

            self.acf_values_ = acf_values
            self.confidence_intervals_ = confint

            # Find significant lags
            significant_lags = []
            for lag in range(1, len(acf_values)):  # Skip lag 0 (always 1.0)
                if abs(acf_values[lag]) > abs(confint[lag, 1] - acf_values[lag]):
                    significant_lags.append(
                        {
                            "lag": lag,
                            "correlation": acf_values[lag],
                            "is_significant": True,
                            "confidence_lower": confint[lag, 0],
                            "confidence_upper": confint[lag, 1],
                        }
                    )
                else:
                    significant_lags.append(
                        {
                            "lag": lag,
                            "correlation": acf_values[lag],
                            "is_significant": False,
                            "confidence_lower": confint[lag, 0],
                            "confidence_upper": confint[lag, 1],
                        }
                    )

            # Identify patterns and generate recommendations
            recommendations = []
            warnings_list = []
            pattern_analysis = self._analyze_acf_patterns(acf_values, significant_lags)

            # Generate interpretation
            num_significant = sum(
                1 for lag_info in significant_lags if lag_info["is_significant"]
            )

            if num_significant == 0:
                interpretation = "No significant autocorrelations detected - series appears to be white noise"
                recommendations.append(
                    "Series may be suitable for simple forecasting methods"
                )
            elif num_significant <= 3:
                interpretation = f"Few significant autocorrelations detected ({num_significant} lags)"
                recommendations.append("Consider MA or low-order ARMA models")
            else:
                interpretation = f"Multiple significant autocorrelations detected ({num_significant} lags)"
                recommendations.append(
                    "Consider higher-order ARMA models or seasonal patterns"
                )

            # Add pattern-specific recommendations
            recommendations.extend(pattern_analysis["recommendations"])
            warnings_list.extend(pattern_analysis.get("warnings", []))

            result = TimeSeriesAnalysisResult(
                analysis_type="autocorrelation_analysis",
                interpretation=interpretation,
                model_diagnostics={
                    "acf_values": acf_values.tolist(),
                    "confidence_intervals": confint.tolist(),
                    "significant_lags": significant_lags,
                    "pattern_analysis": pattern_analysis,
                    "max_lags_computed": nlags,
                    "series_length": len(series),
                },
                recommendations=recommendations,
                warnings=warnings_list,
                processing_time=time.time() - start_time,
            )

            return result

        except Exception as e:
            logger.error(f"Error in autocorrelation analysis: {e}")
            return TimeSeriesAnalysisResult(
                analysis_type="autocorrelation_analysis_error",
                interpretation=f"Error during autocorrelation analysis: {str(e)}",
                warnings=[str(e)],
                processing_time=time.time() - start_time,
            )

    def _analyze_acf_patterns(
        self, acf_values: np.ndarray, significant_lags: List[Dict]
    ) -> Dict[str, Any]:
        """
        Analyze patterns in autocorrelation function.

        Parameters:
        -----------
        acf_values : np.ndarray
            Autocorrelation function values
        significant_lags : list of dict
            Information about significant lags

        Returns:
        --------
        pattern_info : dict
            Analysis of ACF patterns and recommendations
        """
        analysis = {
            "pattern_type": "unknown",
            "decay_rate": "unknown",
            "seasonal_pattern": False,
            "seasonal_period": None,
            "recommendations": [],
            "warnings": [],
        }

        try:
            # Analyze decay pattern
            sig_correlations = [
                lag["correlation"] for lag in significant_lags if lag["is_significant"]
            ]

            if len(sig_correlations) == 0:
                analysis["pattern_type"] = "white_noise"
                analysis["decay_rate"] = "immediate"
                analysis["recommendations"].append(
                    "Series appears to be white noise - no temporal dependence"
                )

            elif len(sig_correlations) <= 2:
                analysis["pattern_type"] = "short_memory"
                analysis["decay_rate"] = "fast"
                analysis["recommendations"].append(
                    "Short-term temporal dependence - consider MA(1) or MA(2) model"
                )

            else:
                # Check for exponential decay (AR pattern)
                if self._has_exponential_decay(acf_values[1:6]):  # Check first 5 lags
                    analysis["pattern_type"] = "autoregressive"
                    analysis["decay_rate"] = "exponential"
                    analysis["recommendations"].append(
                        "Exponential decay pattern - consider AR model"
                    )
                else:
                    analysis["pattern_type"] = "complex"
                    analysis["decay_rate"] = "slow"
                    analysis["recommendations"].append(
                        "Complex autocorrelation pattern - consider ARMA model"
                    )

            # Check for seasonal patterns
            seasonal_info = self._detect_seasonal_acf(acf_values, significant_lags)
            if seasonal_info["has_seasonal"]:
                analysis["seasonal_pattern"] = True
                analysis["seasonal_period"] = seasonal_info["period"]
                analysis["recommendations"].append(
                    f"Seasonal pattern detected (period ≈ {seasonal_info['period']}) - consider seasonal ARIMA"
                )

            # Check for problematic patterns
            if max(abs(acf_values[1:6])) > 0.9:
                analysis["warnings"].append(
                    "Very high autocorrelations suggest possible non-stationarity"
                )
                analysis["recommendations"].append("Consider differencing the series")

        except Exception as e:
            analysis["warnings"].append(f"Error in pattern analysis: {e}")

        return analysis

    def _has_exponential_decay(self, correlations: np.ndarray) -> bool:
        """Check if correlations show exponential decay pattern."""
        if len(correlations) < 3:
            return False

        try:
            # Fit exponential decay: y = a * exp(-b * x)
            # Use log transformation: log(|y|) = log(a) - b * x
            x = np.arange(1, len(correlations) + 1)
            y = np.abs(correlations)

            # Filter out very small values to avoid log issues
            valid_mask = y > 0.01
            if np.sum(valid_mask) < 3:
                return False

            x_valid = x[valid_mask]
            y_valid = y[valid_mask]

            log_y = np.log(y_valid)

            # Simple linear regression on log scale
            slope = np.polyfit(x_valid, log_y, 1)[0]

            # Exponential decay should have negative slope and good fit
            return slope < -0.1  # Threshold for significant decay

        except Exception:
            return False

    def _detect_seasonal_acf(
        self, acf_values: np.ndarray, significant_lags: List[Dict]
    ) -> Dict[str, Any]:
        """Detect seasonal patterns in autocorrelation function."""
        seasonal_info = {"has_seasonal": False, "period": None, "strength": 0.0}

        try:
            # Look for periodic peaks in significant lags
            sig_lags = [lag["lag"] for lag in significant_lags if lag["is_significant"]]

            if len(sig_lags) >= 3:
                # Check for regular spacing (seasonal periods)
                for period in [4, 7, 12, 24, 52]:  # Common seasonal periods
                    if period >= len(acf_values):
                        continue

                    # Check if there are significant correlations at multiples of this period
                    seasonal_lags = [
                        lag for lag in sig_lags if lag % period == 0 and lag > 0
                    ]

                    if len(seasonal_lags) >= 2:
                        # Calculate average correlation at seasonal lags
                        seasonal_correlations = [
                            abs(acf_values[lag])
                            for lag in seasonal_lags
                            if lag < len(acf_values)
                        ]
                        avg_seasonal_corr = np.mean(seasonal_correlations)

                        if avg_seasonal_corr > 0.2:  # Threshold for seasonal strength
                            seasonal_info["has_seasonal"] = True
                            seasonal_info["period"] = period
                            seasonal_info["strength"] = avg_seasonal_corr
                            break

        except Exception as e:
            logger.debug(f"Error in seasonal ACF detection: {e}")

        return seasonal_info
