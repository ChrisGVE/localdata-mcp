"""
Trend analysis transformer.

Analyzes trend characteristics including direction, strength, and changepoints
using various trend detection methods.
"""

import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ._base import TimeSeriesAnalysisResult
from ._transformer import TimeSeriesTransformer
from ...logging_manager import get_logger

logger = get_logger(__name__)


class TrendAnalysisTransformer(TimeSeriesTransformer):
    """
    sklearn-compatible transformer for trend analysis and extraction.

    Analyzes trend characteristics including direction, strength, and changepoints
    using various trend detection methods.

    Parameters:
    -----------
    method : str, default='linear'
        Trend detection method: 'linear', 'polynomial', 'lowess', 'hodrick_prescott'
    degree : int, default=1
        Polynomial degree (for polynomial method)
    alpha : float, default=0.05
        Significance level for trend tests
    changepoint_detection : bool, default=True
        Whether to detect trend changepoints
    min_segment_length : int, default=10
        Minimum segment length for changepoint detection
    """

    def __init__(
        self,
        method="linear",
        degree=1,
        alpha=0.05,
        changepoint_detection=True,
        min_segment_length=10,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.method = method
        self.degree = degree
        self.alpha = alpha
        self.changepoint_detection = changepoint_detection
        self.min_segment_length = min_segment_length
        self.trend_parameters_ = {}
        self.changepoints_ = []

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fit the trend analysis transformer."""
        if self.validate_input:
            X, y = self._validate_time_series(X, y)
        self.is_fitted_ = True
        return self

    def _build_trend_interpretation(
        self, trend_direction: str, trend_strength: float, changepoints: List[int]
    ) -> str:
        """Build human-readable interpretation from trend analysis results."""
        if trend_strength < 0.1:
            interpretation = (
                "No significant trend detected - series appears stationary around mean"
            )
        elif trend_strength < 0.3:
            interpretation = f"Weak {trend_direction} trend detected (strength: {trend_strength:.2f})"
        elif trend_strength < 0.7:
            interpretation = f"Moderate {trend_direction} trend detected (strength: {trend_strength:.2f})"
        else:
            interpretation = f"Strong {trend_direction} trend detected (strength: {trend_strength:.2f})"
        if len(changepoints) > 0:
            interpretation += f" with {len(changepoints)} trend changepoint(s)"
        return interpretation

    def _build_trend_recommendations(
        self,
        trend_strength: float,
        changepoints: List[int],
        series: pd.Series,
        trend_analysis: Dict[str, Any],
    ) -> tuple:
        """Return (recommendations, warnings_list) from trend analysis data."""
        recommendations = []
        warnings_list = []
        if trend_strength > 0.5:
            recommendations.append(
                "Strong trend detected - detrending may be necessary for stationary modeling"
            )
            recommendations.append("Consider trend-aware forecasting methods")
        else:
            recommendations.append(
                "Weak or no trend - focus on seasonal or residual patterns"
            )
        if len(changepoints) > 0:
            recommendations.append(
                "Trend changepoints detected - consider structural break models"
            )
            for i, cp in enumerate(changepoints[:3]):
                cp_date = series.index[cp] if cp < len(series.index) else "unknown"
                recommendations.append(
                    f"Changepoint {i + 1} at position {cp} ({cp_date})"
                )
        recommendations.extend(trend_analysis.get("recommendations", []))
        warnings_list.extend(trend_analysis.get("warnings", []))
        return recommendations, warnings_list

    def transform(self, X: pd.DataFrame) -> TimeSeriesAnalysisResult:
        """
        Perform trend analysis on time series data.

        Returns:
        --------
        result : TimeSeriesAnalysisResult
            Trend analysis results with direction, strength, and changepoints
        """
        start_time = time.time()

        if self.validate_input:
            X, _ = self._validate_time_series(X, None)

        try:
            if len(X.columns) == 0:
                raise ValueError("No columns found in time series data")

            series = X.iloc[:, 0].dropna()
            if len(series) < 10:
                raise ValueError("Insufficient data points for trend analysis")

            trend_result = self._extract_trend(series)
            trend_analysis = self._analyze_trend_properties(
                series, trend_result["trend"]
            )

            changepoints = []
            if self.changepoint_detection:
                changepoints = self._detect_changepoints(series)

            self.trend_parameters_ = trend_result
            self.changepoints_ = changepoints

            trend_direction = trend_analysis["direction"]
            trend_strength = trend_analysis["strength"]
            interpretation = self._build_trend_interpretation(
                trend_direction, trend_strength, changepoints
            )
            recommendations, warnings_list = self._build_trend_recommendations(
                trend_strength, changepoints, series, trend_analysis
            )

            return TimeSeriesAnalysisResult(
                analysis_type="trend_analysis",
                interpretation=interpretation,
                trend_component=trend_result["trend"],
                model_diagnostics={
                    "trend_method": self.method,
                    "trend_direction": trend_direction,
                    "trend_strength": trend_strength,
                    "trend_parameters": trend_result.get("parameters", {}),
                    "changepoints": changepoints,
                    "trend_analysis": trend_analysis,
                    "series_length": len(series),
                },
                recommendations=recommendations,
                warnings=warnings_list,
                processing_time=time.time() - start_time,
            )

        except Exception as e:
            logger.error(f"Error in trend analysis: {e}")
            return TimeSeriesAnalysisResult(
                analysis_type="trend_analysis_error",
                interpretation=f"Error during trend analysis: {str(e)}",
                warnings=[str(e)],
                processing_time=time.time() - start_time,
            )

    def _extract_trend(self, series: pd.Series) -> Dict[str, Any]:
        """Extract trend using the specified method."""
        try:
            x = np.arange(len(series))
            y = series.values

            if self.method == "linear":
                # Linear regression
                coeffs = np.polyfit(x, y, 1)
                trend_values = np.polyval(coeffs, x)

                return {
                    "trend": pd.Series(trend_values, index=series.index),
                    "parameters": {"slope": coeffs[0], "intercept": coeffs[1]},
                    "method": "linear",
                }

            elif self.method == "polynomial":
                # Polynomial regression
                coeffs = np.polyfit(x, y, self.degree)
                trend_values = np.polyval(coeffs, x)

                return {
                    "trend": pd.Series(trend_values, index=series.index),
                    "parameters": {
                        "coefficients": coeffs.tolist(),
                        "degree": self.degree,
                    },
                    "method": "polynomial",
                }

            elif self.method == "lowess":
                # LOWESS smoothing
                from statsmodels.nonparametric.smoothers_lowess import lowess

                smoothed = lowess(y, x, frac=0.3, return_sorted=False)

                return {
                    "trend": pd.Series(smoothed, index=series.index),
                    "parameters": {"method": "lowess", "frac": 0.3},
                    "method": "lowess",
                }

            elif self.method == "hodrick_prescott":
                # Hodrick-Prescott filter
                try:
                    from statsmodels.tsa.filters.hp_filter import hpfilter

                    cycle, trend_values = hpfilter(
                        series, lamb=1600
                    )  # Standard lambda for monthly data

                    return {
                        "trend": trend_values,
                        "cycle": cycle,
                        "parameters": {"lambda": 1600},
                        "method": "hodrick_prescott",
                    }
                except ImportError:
                    logger.warning(
                        "Hodrick-Prescott filter not available, using linear trend"
                    )
                    return self._extract_trend_linear(series)

            else:
                raise ValueError(f"Unknown trend method: {self.method}")

        except Exception as e:
            logger.error(f"Error extracting trend: {e}")
            # Fallback to linear trend
            return self._extract_trend_linear(series)

    def _extract_trend_linear(self, series: pd.Series) -> Dict[str, Any]:
        """Fallback linear trend extraction."""
        x = np.arange(len(series))
        y = series.values
        coeffs = np.polyfit(x, y, 1)
        trend_values = np.polyval(coeffs, x)

        return {
            "trend": pd.Series(trend_values, index=series.index),
            "parameters": {"slope": coeffs[0], "intercept": coeffs[1]},
            "method": "linear_fallback",
        }

    def _analyze_trend_properties(
        self, series: pd.Series, trend: pd.Series
    ) -> Dict[str, Any]:
        """Analyze properties of the extracted trend."""
        analysis = {
            "direction": "unknown",
            "strength": 0.0,
            "significance": 0.0,
            "recommendations": [],
            "warnings": [],
        }

        try:
            # Determine trend direction
            if len(trend) > 1:
                overall_change = trend.iloc[-1] - trend.iloc[0]
                if overall_change > 0:
                    analysis["direction"] = "upward"
                elif overall_change < 0:
                    analysis["direction"] = "downward"
                else:
                    analysis["direction"] = "flat"

            # Calculate trend strength (correlation with time)
            if len(series) > 2:
                time_index = np.arange(len(series))
                correlation = np.corrcoef(series.values, time_index)[0, 1]
                analysis["strength"] = (
                    abs(correlation) if not np.isnan(correlation) else 0.0
                )

            # Statistical significance test (simple t-test on slope)
            if self.method in [
                "linear",
                "polynomial",
            ] and "slope" in self.trend_parameters_.get("parameters", {}):
                try:
                    from scipy import stats

                    x = np.arange(len(series))
                    y = series.values

                    # Simple linear regression t-test
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                    analysis["significance"] = (
                        1.0 - p_value
                    )  # Convert to significance measure

                    if p_value <= self.alpha:
                        analysis["recommendations"].append(
                            f"Trend is statistically significant (p={p_value:.4f})"
                        )
                    else:
                        analysis["warnings"].append(
                            f"Trend is not statistically significant (p={p_value:.4f})"
                        )

                except Exception as e:
                    analysis["warnings"].append(
                        f"Could not test trend significance: {e}"
                    )

        except Exception as e:
            analysis["warnings"].append(f"Error in trend property analysis: {e}")

        return analysis

    def _detect_changepoints(self, series: pd.Series) -> List[int]:
        """Detect trend changepoints using simple methods."""
        changepoints = []

        try:
            # Simple changepoint detection using moving averages
            window_size = max(5, len(series) // 20)

            if len(series) < 2 * window_size:
                return changepoints  # Not enough data

            # Calculate moving averages
            ma_short = series.rolling(window=window_size, center=True).mean()
            ma_long = series.rolling(window=window_size * 2, center=True).mean()

            # Find crossover points
            diff = ma_short - ma_long
            sign_changes = np.diff(np.sign(diff.dropna()))

            # Get changepoint indices
            change_indices = np.where(sign_changes != 0)[0]

            # Filter changepoints by minimum segment length
            filtered_changepoints = []
            last_cp = 0

            for cp in change_indices:
                if cp - last_cp >= self.min_segment_length:
                    filtered_changepoints.append(cp)
                    last_cp = cp

            changepoints = filtered_changepoints

        except Exception as e:
            logger.debug(f"Error in changepoint detection: {e}")

        return changepoints
