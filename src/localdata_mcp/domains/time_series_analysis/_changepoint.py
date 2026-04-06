"""
Time Series Analysis - Change Point Detection.

Contains the ChangePointDetector transformer for detecting change points
in time series using ruptures library and statistical methods.
"""

import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import ruptures as rpt

from ...logging_manager import get_logger
from ._base import TimeSeriesAnalysisResult
from ._transformer import TimeSeriesTransformer

logger = get_logger(__name__)


class ChangePointDetector(TimeSeriesTransformer):
    """
    sklearn-compatible transformer for change point detection in time series.

    Implements multiple change point detection algorithms including ruptures library
    integration for advanced detection methods, and statistical approaches for
    structural breaks in time series.

    Parameters:
    -----------
    method : str, default='bcp'
        Change point detection method: 'bcp' (Binary Segmentation), 'pelt',
        'window', 'dynp', 'statistical'
    model : str, default='rbf'
        Statistical model for ruptures methods: 'l1', 'l2', 'rbf', 'normal', 'ar'
    min_size : int, default=10
        Minimum segment size between change points
    penalty : float, optional
        Penalty value for change point detection (automatic if None)
    confidence_level : float, default=0.95
        Confidence level for statistical change point tests
    max_changepoints : int, default=10
        Maximum number of change points to detect
    """

    def __init__(
        self,
        method="bcp",
        model="rbf",
        min_size=10,
        penalty=None,
        confidence_level=0.95,
        max_changepoints=10,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.method = method
        self.model = model
        self.min_size = min_size
        self.penalty = penalty
        self.confidence_level = confidence_level
        self.max_changepoints = max_changepoints
        self.changepoints_ = []
        self.segments_ = []
        self.ruptures_available_ = True  # always True; ruptures is a hard dependency

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fit the change point detector."""
        if self.validate_input:
            X, y = self._validate_time_series(X, y)
        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> TimeSeriesAnalysisResult:
        """
        Detect change points in time series data.

        Returns:
        --------
        result : TimeSeriesAnalysisResult
            Change point detection results with detected change points and segments
        """
        start_time = time.time()

        if self.validate_input:
            X, _ = self._validate_time_series(X, None)

        try:
            # Use first column for change point detection
            if len(X.columns) == 0:
                raise ValueError("No columns found in time series data")

            series = X.iloc[:, 0].dropna()
            if len(series) < 2 * self.min_size:
                raise ValueError(
                    f"Insufficient data points for change point detection (need at least {2 * self.min_size})"
                )

            # Detect change points
            if self.method != "statistical":
                changepoints = self._detect_changepoints_ruptures(series)
            else:
                changepoints = self._detect_changepoints_statistical(series)

            self.changepoints_ = changepoints

            # Create segments based on change points
            segments = self._create_segments(series, changepoints)
            self.segments_ = segments

            # Analyze segment characteristics
            segment_analysis = self._analyze_segments(segments, series)

            # Calculate detection quality metrics
            detection_metrics = self._calculate_detection_metrics(
                series, changepoints, segments
            )

            # Generate interpretation
            interpretation = self._generate_changepoint_interpretation(
                changepoints, segments, series, detection_metrics
            )

            # Generate recommendations
            recommendations = self._generate_changepoint_recommendations(
                changepoints, segments, series, detection_metrics
            )

            processing_time = time.time() - start_time

            # Prepare result
            model_parameters = {
                "changepoints": changepoints,
                "segments": segments,
                "segment_analysis": segment_analysis,
                "detection_method": self.method,
                "model_used": self.model,
                "min_segment_size": self.min_size,
                "penalty_used": self.penalty,
                "ruptures_available": True,
            }

            model_diagnostics = detection_metrics

            return TimeSeriesAnalysisResult(
                analysis_type="change_point_detection",
                model_parameters=model_parameters,
                model_diagnostics=model_diagnostics,
                interpretation=interpretation,
                recommendations=recommendations,
                processing_time=processing_time,
                data_quality_score=self._calculate_data_quality_score(X),
                confidence_level=self.confidence_level,
            )

        except Exception as e:
            logger.error(f"Change point detection failed: {e}")
            return TimeSeriesAnalysisResult(
                analysis_type="change_point_detection",
                interpretation=f"Change point detection failed: {str(e)}",
                recommendations=["Check data quality and method parameters"],
                processing_time=time.time() - start_time,
            )

    def _detect_changepoints_ruptures(self, series: pd.Series) -> List[int]:
        """
        Detect change points using ruptures library methods.
        """
        try:
            # Convert series to numpy array
            signal = series.values

            # Initialize algorithm
            if self.method == "bcp":
                algo = rpt.Binseg(model=self.model, min_size=self.min_size)
            elif self.method == "pelt":
                algo = rpt.Pelt(model=self.model, min_size=self.min_size)
            elif self.method == "window":
                algo = rpt.Window(
                    width=max(20, len(signal) // 10),
                    model=self.model,
                    min_size=self.min_size,
                )
            elif self.method == "dynp":
                algo = rpt.Dynp(model=self.model, min_size=self.min_size)
            else:
                raise ValueError(f"Unknown ruptures method: {self.method}")

            # Fit the algorithm
            algo.fit(signal)

            # Detect change points
            if self.penalty is None:
                # Use automatic penalty selection
                if hasattr(algo, "predict"):
                    if self.method in ["pelt", "bcp"]:
                        # For PELT and Binary Segmentation, use pen parameter
                        penalty_val = len(signal) * np.log(
                            len(signal)
                        )  # BIC-like penalty
                        changepoints = algo.predict(pen=penalty_val)
                    else:
                        # For other methods, specify number of change points
                        n_bkps = min(
                            self.max_changepoints, len(signal) // (2 * self.min_size)
                        )
                        if n_bkps > 0:
                            changepoints = algo.predict(n_bkps=n_bkps)
                        else:
                            changepoints = [len(signal)]
                else:
                    changepoints = [len(signal)]
            else:
                changepoints = algo.predict(pen=self.penalty)

            # Convert to 0-based indexing and exclude the last point (which is always the length)
            changepoints = [cp - 1 for cp in changepoints if cp < len(signal)]

            # Limit to maximum number of change points
            if len(changepoints) > self.max_changepoints:
                changepoints = changepoints[: self.max_changepoints]

            logger.debug(
                f"Detected {len(changepoints)} change points using {self.method}"
            )
            return changepoints

        except Exception as e:
            logger.error(f"Ruptures change point detection failed: {e}")
            return self._detect_changepoints_statistical(series)

    def _detect_changepoints_statistical(self, series: pd.Series) -> List[int]:
        """
        Detect change points using statistical methods (CUSUM-based approach).
        """
        try:
            signal = series.values
            n = len(signal)
            changepoints = []

            # Simple CUSUM-based change point detection
            mean_val = np.mean(signal)
            std_val = np.std(signal)

            if std_val == 0:
                return []

            # Calculate cumulative sum of standardized residuals
            standardized = (signal - mean_val) / std_val
            cusum_pos = np.zeros(n)
            cusum_neg = np.zeros(n)

            # CUSUM parameters
            h = 2.5 * np.sqrt(np.log(n))  # Control limit
            k = 0.5  # Reference value

            for i in range(1, n):
                cusum_pos[i] = max(0, cusum_pos[i - 1] + standardized[i] - k)
                cusum_neg[i] = max(0, cusum_neg[i - 1] - standardized[i] - k)

                # Check for change point
                if cusum_pos[i] > h or cusum_neg[i] > h:
                    # Find the most likely change point in recent history
                    start_idx = max(0, i - 2 * self.min_size)
                    if len(changepoints) == 0 or i - changepoints[-1] >= self.min_size:
                        changepoints.append(i)
                        # Reset CUSUM values
                        cusum_pos[i] = 0
                        cusum_neg[i] = 0

            # Additional method: variance change detection
            if len(changepoints) < self.max_changepoints // 2:
                variance_changes = self._detect_variance_changes(signal)
                for cp in variance_changes:
                    if not any(
                        abs(cp - existing) < self.min_size for existing in changepoints
                    ):
                        changepoints.append(cp)
                        if len(changepoints) >= self.max_changepoints:
                            break

            changepoints = sorted(list(set(changepoints)))
            if len(changepoints) > self.max_changepoints:
                changepoints = changepoints[: self.max_changepoints]

            logger.debug(
                f"Detected {len(changepoints)} change points using statistical method"
            )
            return changepoints

        except Exception as e:
            logger.error(f"Statistical change point detection failed: {e}")
            return []

    def _detect_variance_changes(self, signal: np.ndarray) -> List[int]:
        """Detect change points based on variance changes."""
        n = len(signal)
        changepoints = []
        window_size = max(self.min_size, n // 20)

        for i in range(window_size, n - window_size, window_size // 2):
            left_var = np.var(signal[max(0, i - window_size) : i])
            right_var = np.var(signal[i : min(n, i + window_size)])

            # F-test for variance equality
            if left_var > 0 and right_var > 0:
                f_stat = max(left_var, right_var) / min(left_var, right_var)
                # Critical value for F-test (approximate)
                f_critical = 2.0  # Simplified threshold

                if f_stat > f_critical:
                    changepoints.append(i)

        return changepoints

    def _create_segments(
        self, series: pd.Series, changepoints: List[int]
    ) -> List[Dict]:
        """Create segment information based on detected change points."""
        segments = []
        indices = series.index
        values = series.values

        start_idx = 0
        for i, cp in enumerate(changepoints + [len(series) - 1]):
            end_idx = cp

            if end_idx > start_idx:
                segment_data = values[start_idx : end_idx + 1]
                segments.append(
                    {
                        "segment_id": i,
                        "start_index": indices[start_idx],
                        "end_index": indices[end_idx],
                        "start_position": start_idx,
                        "end_position": end_idx,
                        "length": end_idx - start_idx + 1,
                        "mean": np.mean(segment_data),
                        "std": np.std(segment_data),
                        "min": np.min(segment_data),
                        "max": np.max(segment_data),
                        "trend": self._calculate_segment_trend(segment_data),
                    }
                )

            start_idx = end_idx + 1

        return segments

    def _calculate_segment_trend(self, segment_data: np.ndarray) -> float:
        """Calculate trend slope for a segment."""
        if len(segment_data) < 2:
            return 0.0

        x = np.arange(len(segment_data))
        try:
            slope, _ = np.polyfit(x, segment_data, 1)
            return float(slope)
        except:
            return 0.0

    def _analyze_segments(self, segments: List[Dict], series: pd.Series) -> Dict:
        """Analyze characteristics of detected segments."""
        if not segments:
            return {}

        # Calculate segment statistics
        lengths = [seg["length"] for seg in segments]
        means = [seg["mean"] for seg in segments]
        stds = [seg["std"] for seg in segments]
        trends = [seg["trend"] for seg in segments]

        analysis = {
            "n_segments": len(segments),
            "avg_segment_length": np.mean(lengths),
            "min_segment_length": np.min(lengths),
            "max_segment_length": np.max(lengths),
            "mean_difference_max": max(means) - min(means) if means else 0,
            "std_difference_max": max(stds) - min(stds) if stds else 0,
            "trend_changes": len(
                [i for i in range(1, len(trends)) if trends[i] * trends[i - 1] < 0]
            ),  # Sign changes
            "most_stable_segment": segments[np.argmin(stds)] if stds else None,
            "most_variable_segment": segments[np.argmax(stds)] if stds else None,
        }

        return analysis

    def _calculate_detection_metrics(
        self, series: pd.Series, changepoints: List[int], segments: List[Dict]
    ) -> Dict:
        """Calculate quality metrics for change point detection."""
        metrics = {
            "n_changepoints": len(changepoints),
            "changepoint_density": (
                len(changepoints) / len(series) if len(series) > 0 else 0
            ),
            "avg_segment_length": (
                len(series) / (len(changepoints) + 1) if changepoints else len(series)
            ),
            "detection_confidence": self.confidence_level,
        }

        if segments:
            # Calculate within-segment vs between-segment variance
            within_var = np.mean([seg["std"] ** 2 for seg in segments])
            between_var = np.var([seg["mean"] for seg in segments])

            metrics["within_segment_variance"] = within_var
            metrics["between_segment_variance"] = between_var
            metrics["variance_ratio"] = (
                between_var / within_var if within_var > 0 else 0
            )

        return metrics

    def _generate_changepoint_interpretation(
        self,
        changepoints: List[int],
        segments: List[Dict],
        series: pd.Series,
        metrics: Dict,
    ) -> str:
        """Generate human-readable interpretation of change point results."""
        if not changepoints:
            return f"No significant change points detected in {len(series)} observations. The time series appears stationary with consistent behavior."

        n_cp = len(changepoints)
        n_segments = len(segments)
        avg_length = metrics.get("avg_segment_length", 0)

        interpretation = f"Detected {n_cp} change point(s) dividing the series into {n_segments} distinct segments. "

        if avg_length < len(series) / 10:
            interpretation += (
                "Frequent regime changes detected - highly dynamic system. "
            )
        elif avg_length > len(series) / 3:
            interpretation += "Infrequent but significant regime changes - relatively stable periods. "
        else:
            interpretation += (
                "Moderate frequency of regime changes - balanced dynamics. "
            )

        # Describe the most significant changes
        if segments and len(segments) >= 2:
            means = [seg["mean"] for seg in segments]
            max_mean_diff = max(means) - min(means)
            series_std = series.std()

            if max_mean_diff > 2 * series_std:
                interpretation += "Large magnitude changes detected across segments. "
            elif max_mean_diff > series_std:
                interpretation += (
                    "Moderate magnitude changes detected across segments. "
                )
            else:
                interpretation += (
                    "Subtle but statistically significant changes detected. "
                )

        # Method information
        method_info = (
            f"ruptures-{self.method}" if self.method != "statistical" else "statistical"
        )
        interpretation += f"Detection performed using {method_info} method."

        return interpretation

    def _generate_changepoint_recommendations(
        self,
        changepoints: List[int],
        segments: List[Dict],
        series: pd.Series,
        metrics: Dict,
    ) -> List[str]:
        """Generate recommendations based on change point detection results."""
        recommendations = []

        if not changepoints:
            recommendations.extend(
                [
                    "No change points detected - series appears stationary",
                    "Consider standard time series models (ARIMA, exponential smoothing)",
                    "Validate with stationarity tests to confirm stability",
                ]
            )
            return recommendations

        n_cp = len(changepoints)
        density = metrics.get("changepoint_density", 0)

        # Density-based recommendations
        if density > 0.1:  # More than 10% change points
            recommendations.extend(
                [
                    "High frequency of change points detected",
                    "Consider regime-switching models or piecewise analysis",
                    "Investigate potential causes of frequent structural breaks",
                ]
            )
        elif density < 0.01:  # Less than 1% change points
            recommendations.extend(
                [
                    "Few but significant structural breaks detected",
                    "Consider intervention analysis or breakpoint regression",
                    "Examine periods around change points for external factors",
                ]
            )
        else:
            recommendations.extend(
                [
                    "Moderate frequency of structural changes",
                    "Consider threshold models or segmented regression",
                ]
            )

        # Segment-based recommendations
        if segments:
            variance_ratio = metrics.get("variance_ratio", 0)
            if variance_ratio > 4:
                recommendations.append(
                    "Strong differences between segments - consider separate models per segment"
                )
            elif variance_ratio > 1:
                recommendations.append(
                    "Moderate differences between segments - investigate common patterns"
                )

        return recommendations
