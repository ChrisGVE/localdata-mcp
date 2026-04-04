"""
Time series decomposition transformer.

Provides seasonal decomposition to separate time series into trend, seasonal,
and residual components using various decomposition methods.
"""

import time
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose

from ._base import TimeSeriesAnalysisResult
from ._transformer import TimeSeriesTransformer
from ...logging_manager import get_logger

logger = get_logger(__name__)


class TimeSeriesDecompositionTransformer(TimeSeriesTransformer):
    """
    sklearn-compatible transformer for time series decomposition analysis.

    Performs seasonal decomposition to separate time series into trend, seasonal,
    and residual components using various decomposition methods.

    Parameters:
    -----------
    model : str, default='additive'
        Type of decomposition: 'additive' or 'multiplicative'
    period : int, optional
        Period for seasonal decomposition. If None, attempts to detect automatically
    extrapolate_trend : str or int, default='freq'
        How to extrapolate trend at ends: 'freq', integer, or None
    two_sided : bool, default=True
        Whether to use centered moving average for trend estimation
    method : str, default='seasonal_decompose'
        Decomposition method: 'seasonal_decompose', 'stl', 'x13' (if available)
    """

    def __init__(
        self,
        model="additive",
        period=None,
        extrapolate_trend="freq",
        two_sided=True,
        method="seasonal_decompose",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model = model
        self.period = period
        self.extrapolate_trend = extrapolate_trend
        self.two_sided = two_sided
        self.method = method
        self.decomposition_result_ = None
        self.detected_period_ = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fit the decomposition transformer."""
        if self.validate_input:
            X, y = self._validate_time_series(X, y)
        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> TimeSeriesAnalysisResult:
        """
        Perform time series decomposition analysis.

        Returns:
        --------
        result : TimeSeriesAnalysisResult
            Decomposition results with trend, seasonal, and residual components
        """
        start_time = time.time()

        if self.validate_input:
            X, _ = self._validate_time_series(X, None)

        try:
            # Use first column for decomposition
            if len(X.columns) == 0:
                raise ValueError("No columns found in time series data")

            series = X.iloc[:, 0].dropna()
            if len(series) < 20:
                raise ValueError(
                    "Insufficient data points for decomposition (need at least 20)"
                )

            # Detect period if not provided
            detected_period = self.period
            if detected_period is None:
                detected_period = self._detect_seasonal_period(series)

            if detected_period is None or detected_period < 2:
                # Fallback to simple trend extraction without seasonality
                decomp_result = self._simple_trend_decomposition(series)
                interpretation = "No clear seasonal pattern detected - performed trend-only decomposition"
                has_seasonality = False
            else:
                # Perform full seasonal decomposition
                if self.method == "stl":
                    decomp_result = self._stl_decomposition(series, detected_period)
                elif self.method == "x13":
                    decomp_result = self._x13_decomposition(series, detected_period)
                else:
                    decomp_result = self._seasonal_decomposition(
                        series, detected_period
                    )

                interpretation = (
                    f"Decomposition completed with period {detected_period}"
                )
                has_seasonality = True

            self.decomposition_result_ = decomp_result
            self.detected_period_ = detected_period

            # Analyze decomposition components
            component_analysis = self._analyze_components(
                decomp_result, has_seasonality
            )

            # Generate recommendations
            recommendations = []
            warnings_list = []

            if has_seasonality:
                recommendations.append(
                    f"Strong seasonal pattern detected with period {detected_period}"
                )
                recommendations.append(
                    "Consider seasonal ARIMA or seasonal forecasting models"
                )
            else:
                recommendations.append(
                    "No significant seasonal pattern - focus on trend modeling"
                )
                recommendations.append(
                    "Consider non-seasonal ARIMA or trend-based forecasting"
                )

            # Add component-specific recommendations
            recommendations.extend(component_analysis.get("recommendations", []))
            warnings_list.extend(component_analysis.get("warnings", []))

            result = TimeSeriesAnalysisResult(
                analysis_type="time_series_decomposition",
                interpretation=interpretation,
                trend_component=decomp_result.get("trend"),
                seasonal_component=decomp_result.get("seasonal"),
                residual_component=decomp_result.get("resid"),
                seasonality_period=detected_period,
                model_diagnostics={
                    "decomposition_method": self.method,
                    "model_type": self.model,
                    "detected_period": detected_period,
                    "has_seasonality": has_seasonality,
                    "component_analysis": component_analysis,
                    "series_length": len(series),
                },
                recommendations=recommendations,
                warnings=warnings_list,
                processing_time=time.time() - start_time,
            )

            return result

        except Exception as e:
            logger.error(f"Error in time series decomposition: {e}")
            return TimeSeriesAnalysisResult(
                analysis_type="decomposition_error",
                interpretation=f"Error during time series decomposition: {str(e)}",
                warnings=[str(e)],
                processing_time=time.time() - start_time,
            )

    def _detect_seasonal_period(self, series: pd.Series) -> Optional[int]:
        """Detect seasonal period using autocorrelation analysis."""
        try:
            # Use the seasonality detection from base class
            seasonality_info = self._detect_seasonality(series)

            if seasonality_info["has_seasonality"]:
                return seasonality_info["dominant_period"]

            # Fallback: try common periods based on data frequency
            freq = self._infer_frequency(pd.DataFrame(index=series.index))

            if freq == "D":  # Daily data
                test_periods = [7, 30, 365]
            elif freq in ["H", "T"]:  # Hourly or minutely
                test_periods = [24, 168]  # Daily, weekly
            elif freq == "M":  # Monthly
                test_periods = [12]
            else:
                test_periods = [4, 7, 12, 24]

            # Test periods with sufficient data
            for period in test_periods:
                if len(series) >= 3 * period:
                    # Simple test: check autocorrelation at the period
                    try:
                        autocorr_at_period = series.autocorr(lag=period)
                        if (
                            not pd.isna(autocorr_at_period)
                            and abs(autocorr_at_period) > 0.3
                        ):
                            return period
                    except Exception:
                        continue

            return None

        except Exception as e:
            logger.debug(f"Error detecting seasonal period: {e}")
            return None

    def _seasonal_decomposition(
        self, series: pd.Series, period: int
    ) -> Dict[str, pd.Series]:
        """Perform seasonal decomposition using statsmodels."""
        try:
            decomp = seasonal_decompose(
                series,
                model=self.model,
                period=period,
                extrapolate_trend=self.extrapolate_trend,
                two_sided=self.two_sided,
            )

            return {
                "trend": decomp.trend,
                "seasonal": decomp.seasonal,
                "resid": decomp.resid,
                "observed": series,
            }

        except Exception as e:
            logger.error(f"Seasonal decomposition failed: {e}")
            # Fallback to simple trend extraction
            return self._simple_trend_decomposition(series)

    def _stl_decomposition(
        self, series: pd.Series, period: int
    ) -> Dict[str, pd.Series]:
        """Perform STL decomposition for robust trend/seasonal extraction."""
        try:
            from statsmodels.tsa.seasonal import STL

            # STL parameters
            stl = STL(
                series,
                seasonal=7,  # Seasonal smoothing parameter
                trend=None,  # Trend smoothing (auto)
                period=period,
                robust=True,  # Robust to outliers
            )

            result = stl.fit()

            return {
                "trend": result.trend,
                "seasonal": result.seasonal,
                "resid": result.resid,
                "observed": series,
            }

        except Exception as e:
            logger.warning(
                f"STL decomposition failed: {e}, falling back to seasonal_decompose"
            )
            return self._seasonal_decomposition(series, period)

    def _x13_decomposition(
        self, series: pd.Series, period: int
    ) -> Dict[str, pd.Series]:
        """Perform X-13ARIMA-SEATS decomposition if available."""
        try:
            from statsmodels.tsa.x13 import x13_arima_analysis

            # X-13 requires specific frequency
            x13_result = x13_arima_analysis(series)

            return {
                "trend": x13_result.trend,
                "seasonal": x13_result.seasonal,
                "resid": x13_result.irregular,
                "observed": series,
            }

        except Exception as e:
            logger.warning(
                f"X-13 decomposition not available: {e}, falling back to seasonal_decompose"
            )
            return self._seasonal_decomposition(series, period)

    def _simple_trend_decomposition(self, series: pd.Series) -> Dict[str, pd.Series]:
        """Simple trend extraction without seasonal component."""
        try:
            # Use rolling mean as trend
            window_size = max(3, len(series) // 20)  # Adaptive window size
            trend = series.rolling(
                window=window_size, center=True, min_periods=1
            ).mean()
            residual = series - trend

            # No seasonal component
            seasonal = pd.Series(0, index=series.index)

            return {
                "trend": trend,
                "seasonal": seasonal,
                "resid": residual,
                "observed": series,
            }

        except Exception as e:
            logger.error(f"Simple trend decomposition failed: {e}")
            # Ultimate fallback
            return {
                "trend": pd.Series(series.mean(), index=series.index),
                "seasonal": pd.Series(0, index=series.index),
                "resid": series - series.mean(),
                "observed": series,
            }

    def _analyze_components(
        self, decomp_result: Dict[str, pd.Series], has_seasonality: bool
    ) -> Dict[str, Any]:
        """Analyze decomposition components for insights."""
        analysis = {
            "trend_strength": 0.0,
            "seasonal_strength": 0.0,
            "residual_variance": 0.0,
            "decomposition_quality": 0.0,
            "recommendations": [],
            "warnings": [],
        }

        try:
            observed = decomp_result["observed"]
            trend = decomp_result["trend"].dropna()
            seasonal = decomp_result["seasonal"]
            residual = decomp_result["resid"].dropna()

            # Calculate component strengths
            if len(trend) > 0:
                trend_var = trend.var()
                total_var = observed.var()
                if total_var > 0:
                    analysis["trend_strength"] = min(1.0, trend_var / total_var)

            if has_seasonality and len(seasonal) > 0:
                seasonal_var = seasonal.var()
                total_var = observed.var()
                if total_var > 0:
                    analysis["seasonal_strength"] = min(1.0, seasonal_var / total_var)

            # Residual analysis
            if len(residual) > 0:
                analysis["residual_variance"] = residual.var()

                # Check residual properties
                if analysis["residual_variance"] < 0.1 * observed.var():
                    analysis["recommendations"].append(
                        "Low residual variance - good decomposition quality"
                    )
                else:
                    analysis["warnings"].append(
                        "High residual variance - decomposition may be incomplete"
                    )

            # Overall decomposition quality
            explained_variance = (
                analysis["trend_strength"] + analysis["seasonal_strength"]
            )
            analysis["decomposition_quality"] = min(1.0, explained_variance)

            # Generate insights
            if analysis["trend_strength"] > 0.7:
                analysis["recommendations"].append(
                    "Strong trend component - trend modeling is important"
                )
            elif analysis["trend_strength"] < 0.2:
                analysis["recommendations"].append(
                    "Weak trend component - focus on seasonal/residual patterns"
                )

            if has_seasonality:
                if analysis["seasonal_strength"] > 0.5:
                    analysis["recommendations"].append(
                        "Strong seasonal component - seasonal modeling essential"
                    )
                elif analysis["seasonal_strength"] < 0.2:
                    analysis["warnings"].append(
                        "Weak seasonal pattern - verify seasonal period"
                    )

        except Exception as e:
            analysis["warnings"].append(f"Error in component analysis: {e}")

        return analysis
