"""
Time Series Analysis - Seasonal Anomaly Detection.

Contains the SeasonalAnomalyDetector transformer for detecting anomalies
in time series by accounting for seasonal patterns and using adaptive
thresholds that adjust to seasonal variations and trend changes.
"""

import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ...logging_manager import get_logger
from ._base import TimeSeriesAnalysisResult
from ._transformer import TimeSeriesTransformer

logger = get_logger(__name__)


class SeasonalAnomalyDetector(TimeSeriesTransformer):
    """
    sklearn-compatible transformer for seasonal anomaly detection with adaptive thresholds.

    Detects anomalies in time series by accounting for seasonal patterns and using
    adaptive thresholds that adjust to seasonal variations and trend changes.

    Parameters:
    -----------
    seasonal_period : int, optional
        Known seasonal period. If None, attempts to detect automatically
    method : str, default='adaptive_threshold'
        Detection method: 'adaptive_threshold', 'seasonal_iqr', 'seasonal_zscore'
    threshold_factor : float, default=2.5
        Multiplier for adaptive threshold calculation
    adaptation_rate : float, default=0.1
        Rate of threshold adaptation (0 = no adaptation, 1 = full adaptation)
    min_history : int, default=50
        Minimum historical observations for threshold calculation
    """

    def __init__(
        self,
        seasonal_period=None,
        method="adaptive_threshold",
        threshold_factor=2.5,
        adaptation_rate=0.1,
        min_history=50,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.seasonal_period = seasonal_period
        self.method = method
        self.threshold_factor = threshold_factor
        self.adaptation_rate = adaptation_rate
        self.min_history = min_history
        self.seasonal_components_ = None
        self.adaptive_thresholds_ = None
        self.seasonal_anomalies_ = []

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fit the seasonal anomaly detector."""
        if self.validate_input:
            X, y = self._validate_time_series(X, y)
        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> TimeSeriesAnalysisResult:
        """
        Detect seasonal anomalies in time series data.

        Returns:
        --------
        result : TimeSeriesAnalysisResult
            Seasonal anomaly detection results with adaptive thresholds
        """
        start_time = time.time()

        if self.validate_input:
            X, _ = self._validate_time_series(X, None)

        try:
            # Use first column for seasonal anomaly detection
            if len(X.columns) == 0:
                raise ValueError("No columns found in time series data")

            series = X.iloc[:, 0].dropna()
            if len(series) < self.min_history:
                raise ValueError(
                    f"Insufficient data points for seasonal anomaly detection (need at least {self.min_history})"
                )

            # Detect seasonal period if not provided
            period = self.seasonal_period
            if period is None:
                period = self._detect_seasonal_period(series)

            if period is None or period < 2:
                # Fallback to non-seasonal anomaly detection
                anomalies, thresholds = self._detect_non_seasonal_anomalies(series)
                seasonal_info = None
            else:
                # Perform seasonal decomposition
                seasonal_info = self._perform_seasonal_decomposition(series, period)

                # Detect anomalies based on method
                if self.method == "adaptive_threshold":
                    anomalies, thresholds = self._detect_adaptive_threshold_anomalies(
                        series, seasonal_info, period
                    )
                elif self.method == "seasonal_iqr":
                    anomalies, thresholds = self._detect_seasonal_iqr_anomalies(
                        series, seasonal_info, period
                    )
                elif self.method == "seasonal_zscore":
                    anomalies, thresholds = self._detect_seasonal_zscore_anomalies(
                        series, seasonal_info, period
                    )
                else:
                    raise ValueError(
                        f"Unknown seasonal anomaly detection method: {self.method}"
                    )

            self.seasonal_components_ = seasonal_info
            self.adaptive_thresholds_ = thresholds
            self.seasonal_anomalies_ = anomalies

            # Calculate seasonal anomaly statistics
            anomaly_stats = self._calculate_seasonal_anomaly_statistics(
                series, anomalies, period
            )

            # Analyze seasonal patterns
            seasonal_analysis = self._analyze_seasonal_patterns(
                series, anomalies, seasonal_info, period
            )

            # Generate interpretation
            interpretation = self._generate_seasonal_interpretation(
                anomalies, series, seasonal_info, period, anomaly_stats
            )

            # Generate recommendations
            recommendations = self._generate_seasonal_recommendations(
                anomalies, series, seasonal_info, period, seasonal_analysis
            )

            processing_time = time.time() - start_time

            # Prepare result
            model_parameters = {
                "seasonal_anomalies": anomalies,
                "adaptive_thresholds": thresholds.tolist()
                if thresholds is not None
                else [],
                "seasonal_period": period,
                "detection_method": self.method,
                "threshold_factor": self.threshold_factor,
                "adaptation_rate": self.adaptation_rate,
                "seasonal_components": seasonal_info,
                "n_seasonal_anomalies": len(anomalies),
            }

            model_diagnostics = {
                "anomaly_statistics": anomaly_stats,
                "seasonal_analysis": seasonal_analysis,
                "period_detected": period,
            }

            return TimeSeriesAnalysisResult(
                analysis_type="seasonal_anomaly_detection",
                model_parameters=model_parameters,
                model_diagnostics=model_diagnostics,
                interpretation=interpretation,
                recommendations=recommendations,
                processing_time=processing_time,
                data_quality_score=self._calculate_data_quality_score(X),
                confidence_level=0.95,
            )

        except Exception as e:
            logger.error(f"Seasonal anomaly detection failed: {e}")
            return TimeSeriesAnalysisResult(
                analysis_type="seasonal_anomaly_detection",
                interpretation=f"Seasonal anomaly detection failed: {str(e)}",
                recommendations=["Check data quality and seasonal parameters"],
                processing_time=time.time() - start_time,
            )

    def _detect_seasonal_period(self, series: pd.Series) -> Optional[int]:
        """Detect seasonal period using autocorrelation analysis."""
        try:
            # Simple seasonal period detection using autocorrelation
            max_lag = min(len(series) // 4, 50)
            if max_lag < 2:
                return None

            autocorr = [series.autocorr(lag=lag) for lag in range(1, max_lag + 1)]
            autocorr = [ac for ac in autocorr if not pd.isna(ac)]

            if not autocorr:
                return None

            # Find peaks in autocorrelation
            peaks = []
            for i in range(1, len(autocorr) - 1):
                if (
                    autocorr[i] > 0.3
                    and autocorr[i] > autocorr[i - 1]
                    and autocorr[i] > autocorr[i + 1]
                ):
                    peaks.append((i + 1, autocorr[i]))  # +1 because lag starts at 1

            if peaks:
                # Return the lag with highest autocorrelation
                best_period = max(peaks, key=lambda x: x[1])[0]
                return best_period

            return None

        except Exception as e:
            logger.debug(f"Seasonal period detection failed: {e}")
            return None

    def _perform_seasonal_decomposition(self, series: pd.Series, period: int) -> Dict:
        """Perform seasonal decomposition and return components."""
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose

            decomposition = seasonal_decompose(
                series, model="additive", period=period, extrapolate_trend="freq"
            )

            return {
                "trend": decomposition.trend.values,
                "seasonal": decomposition.seasonal.values,
                "residual": decomposition.resid.values,
                "period": period,
            }

        except Exception as e:
            logger.debug(f"Seasonal decomposition failed: {e}")
            return {"trend": None, "seasonal": None, "residual": None, "period": period}

    def _detect_non_seasonal_anomalies(
        self, series: pd.Series
    ) -> Tuple[List[int], np.ndarray]:
        """Fallback to non-seasonal anomaly detection."""
        # Use simple statistical method
        values = series.values
        median = np.median(values)
        mad = np.median(np.abs(values - median))

        if mad == 0:
            return [], np.zeros(len(values))

        scores = 0.6745 * (values - median) / mad
        anomaly_mask = np.abs(scores) > self.threshold_factor
        anomalies = list(np.where(anomaly_mask)[0])

        # Create constant thresholds
        thresholds = np.full(len(values), self.threshold_factor)

        return anomalies, thresholds

    def _detect_adaptive_threshold_anomalies(
        self, series: pd.Series, seasonal_info: Dict, period: int
    ) -> Tuple[List[int], np.ndarray]:
        """Detect anomalies using adaptive thresholds that account for seasonality."""
        values = series.values
        n = len(values)
        residuals = seasonal_info.get("residual", values)

        # Initialize adaptive thresholds
        thresholds = np.zeros(n)
        anomalies = []

        # Calculate initial threshold from first few periods
        init_window = min(self.min_history, period * 3, n // 2)
        if init_window > 0:
            init_residuals = residuals[:init_window]
            init_residuals = init_residuals[~np.isnan(init_residuals)]
            if len(init_residuals) > 0:
                init_std = np.std(init_residuals)
                current_threshold = init_std * self.threshold_factor
            else:
                current_threshold = np.std(values) * self.threshold_factor
        else:
            current_threshold = np.std(values) * self.threshold_factor

        # Adaptive threshold calculation
        for i in range(n):
            thresholds[i] = current_threshold

            # Check for anomaly
            if not np.isnan(residuals[i]) and abs(residuals[i]) > current_threshold:
                anomalies.append(i)

            # Update threshold adaptively
            if i >= period and not np.isnan(residuals[i]):
                # Use seasonal lookback window
                lookback_start = max(0, i - period * 2)
                recent_residuals = residuals[lookback_start : i + 1]
                recent_residuals = recent_residuals[~np.isnan(recent_residuals)]

                if len(recent_residuals) > period // 2:
                    recent_std = np.std(recent_residuals)
                    # Adapt threshold
                    current_threshold = (
                        (1 - self.adaptation_rate) * current_threshold
                        + self.adaptation_rate * recent_std * self.threshold_factor
                    )

        return anomalies, thresholds

    def _detect_seasonal_iqr_anomalies(
        self, series: pd.Series, seasonal_info: Dict, period: int
    ) -> Tuple[List[int], np.ndarray]:
        """Detect anomalies using seasonal IQR method."""
        values = series.values
        residuals = seasonal_info.get("residual", values)
        n = len(values)

        # Calculate seasonal IQR for each position in the seasonal cycle
        seasonal_stats = {}
        for pos in range(period):
            seasonal_values = []
            for i in range(pos, n, period):
                if not np.isnan(residuals[i]):
                    seasonal_values.append(residuals[i])

            if len(seasonal_values) >= 3:
                q1 = np.percentile(seasonal_values, 25)
                q3 = np.percentile(seasonal_values, 75)
                iqr = q3 - q1
                seasonal_stats[pos] = {
                    "q1": q1,
                    "q3": q3,
                    "iqr": iqr,
                    "lower": q1 - self.threshold_factor * iqr,
                    "upper": q3 + self.threshold_factor * iqr,
                }
            else:
                seasonal_stats[pos] = None

        # Detect anomalies
        anomalies = []
        thresholds = np.zeros(n)

        for i in range(n):
            seasonal_pos = i % period
            stats = seasonal_stats.get(seasonal_pos)

            if stats is not None and not np.isnan(residuals[i]):
                thresholds[i] = stats["iqr"] * self.threshold_factor
                if residuals[i] < stats["lower"] or residuals[i] > stats["upper"]:
                    anomalies.append(i)
            else:
                # Fallback to global statistics
                global_residuals = residuals[~np.isnan(residuals)]
                if len(global_residuals) > 0:
                    global_std = np.std(global_residuals)
                    thresholds[i] = global_std * self.threshold_factor
                    if not np.isnan(residuals[i]) and abs(residuals[i]) > thresholds[i]:
                        anomalies.append(i)

        return anomalies, thresholds

    def _detect_seasonal_zscore_anomalies(
        self, series: pd.Series, seasonal_info: Dict, period: int
    ) -> Tuple[List[int], np.ndarray]:
        """Detect anomalies using seasonal Z-score method."""
        values = series.values
        residuals = seasonal_info.get("residual", values)
        n = len(values)

        # Calculate seasonal mean and std for each position in the seasonal cycle
        seasonal_stats = {}
        for pos in range(period):
            seasonal_values = []
            for i in range(pos, n, period):
                if not np.isnan(residuals[i]):
                    seasonal_values.append(residuals[i])

            if len(seasonal_values) >= 3:
                seasonal_stats[pos] = {
                    "mean": np.mean(seasonal_values),
                    "std": np.std(seasonal_values),
                }
            else:
                seasonal_stats[pos] = None

        # Detect anomalies
        anomalies = []
        thresholds = np.zeros(n)

        for i in range(n):
            seasonal_pos = i % period
            stats = seasonal_stats.get(seasonal_pos)

            if stats is not None and stats["std"] > 0 and not np.isnan(residuals[i]):
                z_score = abs((residuals[i] - stats["mean"]) / stats["std"])
                thresholds[i] = stats["std"] * self.threshold_factor
                if z_score > self.threshold_factor:
                    anomalies.append(i)
            else:
                # Fallback to global statistics
                global_residuals = residuals[~np.isnan(residuals)]
                if len(global_residuals) > 0:
                    global_mean = np.mean(global_residuals)
                    global_std = np.std(global_residuals)
                    thresholds[i] = global_std * self.threshold_factor
                    if global_std > 0 and not np.isnan(residuals[i]):
                        z_score = abs((residuals[i] - global_mean) / global_std)
                        if z_score > self.threshold_factor:
                            anomalies.append(i)

        return anomalies, thresholds

    def _calculate_seasonal_anomaly_statistics(
        self, series: pd.Series, anomalies: List[int], period: Optional[int]
    ) -> Dict:
        """Calculate statistics about seasonal anomalies."""
        stats = {
            "n_anomalies": len(anomalies),
            "anomaly_rate": len(anomalies) / len(series) if len(series) > 0 else 0,
            "seasonal_period": period,
        }

        if period and anomalies:
            # Analyze anomaly distribution across seasons
            seasonal_counts = {pos: 0 for pos in range(period)}
            for anomaly_idx in anomalies:
                seasonal_pos = anomaly_idx % period
                seasonal_counts[seasonal_pos] += 1

            stats["seasonal_distribution"] = seasonal_counts
            stats["most_anomalous_season"] = max(
                seasonal_counts, key=seasonal_counts.get
            )
            stats["least_anomalous_season"] = min(
                seasonal_counts, key=seasonal_counts.get
            )

        return stats

    def _analyze_seasonal_patterns(
        self,
        series: pd.Series,
        anomalies: List[int],
        seasonal_info: Optional[Dict],
        period: Optional[int],
    ) -> Dict:
        """Analyze patterns in seasonal anomalies."""
        analysis = {}

        if period and anomalies:
            # Check if anomalies cluster in specific seasons
            seasonal_counts = {pos: 0 for pos in range(period)}
            for anomaly_idx in anomalies:
                seasonal_pos = anomaly_idx % period
                seasonal_counts[seasonal_pos] += 1

            max_count = max(seasonal_counts.values())
            min_count = min(seasonal_counts.values())

            analysis["seasonal_clustering"] = max_count > min_count * 2
            analysis["seasonal_variance"] = np.var(list(seasonal_counts.values()))

        # Analyze temporal patterns
        if len(anomalies) > 1:
            gaps = [anomalies[i + 1] - anomalies[i] for i in range(len(anomalies) - 1)]
            analysis["median_gap"] = np.median(gaps)
            analysis["temporal_regularity"] = (
                np.std(gaps) / np.mean(gaps) if np.mean(gaps) > 0 else 0
            )

        return analysis

    def _generate_seasonal_interpretation(
        self,
        anomalies: List[int],
        series: pd.Series,
        seasonal_info: Optional[Dict],
        period: Optional[int],
        stats: Dict,
    ) -> str:
        """Generate interpretation for seasonal anomaly results."""
        if not anomalies:
            if period:
                return f"No seasonal anomalies detected in {len(series)} observations with period {period}. The time series follows expected seasonal patterns."
            else:
                return f"No anomalies detected in {len(series)} observations. No clear seasonal pattern identified."

        n_anomalies = len(anomalies)
        anomaly_rate = stats["anomaly_rate"]

        if period:
            interpretation = (
                f"Detected {n_anomalies} seasonal anomal{'y' if n_anomalies == 1 else 'ies'} "
                f"({anomaly_rate:.1%} of observations) with seasonal period {period}. "
            )

            # Add seasonal distribution information
            if "seasonal_distribution" in stats:
                most_anomalous = stats["most_anomalous_season"]
                interpretation += f"Anomalies are most frequent in seasonal position {most_anomalous}. "
        else:
            interpretation = (
                f"Detected {n_anomalies} anomal{'y' if n_anomalies == 1 else 'ies'} "
                f"({anomaly_rate:.1%} of observations) using non-seasonal method. "
            )

        # Describe frequency
        if anomaly_rate > 0.1:
            interpretation += (
                "High frequency of anomalies suggests systematic seasonal issues. "
            )
        elif anomaly_rate > 0.05:
            interpretation += (
                "Moderate frequency indicates occasional seasonal disruptions. "
            )
        else:
            interpretation += (
                "Low frequency suggests rare exceptional seasonal events. "
            )

        return interpretation

    def _generate_seasonal_recommendations(
        self,
        anomalies: List[int],
        series: pd.Series,
        seasonal_info: Optional[Dict],
        period: Optional[int],
        analysis: Dict,
    ) -> List[str]:
        """Generate recommendations for seasonal anomaly detection results."""
        recommendations = []

        if not anomalies:
            recommendations.extend(
                [
                    "No seasonal anomalies detected - patterns appear normal",
                    "Continue monitoring with current seasonal parameters",
                    "Consider adjusting sensitivity if domain knowledge suggests missed anomalies",
                ]
            )
            return recommendations

        if period:
            recommendations.append(
                f"Seasonal patterns detected with period {period} - anomaly detection adjusted accordingly"
            )

            # Seasonal clustering recommendations
            if analysis.get("seasonal_clustering", False):
                recommendations.extend(
                    [
                        "Anomalies cluster in specific seasonal positions",
                        "Investigate systematic factors affecting particular seasons",
                        "Consider season-specific monitoring thresholds",
                    ]
                )
        else:
            recommendations.extend(
                [
                    "No clear seasonal pattern detected - using non-seasonal detection",
                    "Consider manual specification of seasonal period if domain knowledge available",
                ]
            )

        # Adaptation recommendations
        if self.adaptation_rate > 0:
            recommendations.append(
                "Adaptive thresholds are learning from recent patterns"
            )
        else:
            recommendations.append(
                "Consider enabling threshold adaptation for dynamic environments"
            )

        # Method-specific recommendations
        if self.method == "adaptive_threshold":
            recommendations.append(
                "Adaptive threshold method provides good balance of sensitivity and stability"
            )
        elif self.method == "seasonal_iqr":
            recommendations.append(
                "IQR method provides robust detection against outliers"
            )
        elif self.method == "seasonal_zscore":
            recommendations.append(
                "Z-score method assumes normal distribution of seasonal residuals"
            )

        return recommendations
