"""
Time Series Analysis - Anomaly Detection.

Contains the AnomalyDetector transformer for statistical anomaly detection
in time series using multiple methods including statistical control limits,
isolation forest, z-score, and IQR approaches.
"""

import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ...logging_manager import get_logger
from ._base import TimeSeriesAnalysisResult
from ._transformer import TimeSeriesTransformer

logger = get_logger(__name__)


class AnomalyDetector(TimeSeriesTransformer):
    """
    sklearn-compatible transformer for statistical anomaly detection in time series.

    Implements multiple anomaly detection methods including statistical control limits,
    isolation forest, and time series specific outlier detection algorithms.

    Parameters:
    -----------
    method : str, default='statistical'
        Anomaly detection method: 'statistical', 'isolation_forest', 'zscore', 'iqr'
    threshold : float, default=3.0
        Threshold for anomaly detection (method-dependent)
    window_size : int, optional
        Rolling window size for local anomaly detection
    seasonal_adjustment : bool, default=True
        Whether to adjust for seasonal patterns before anomaly detection
    contamination : float, default=0.05
        Expected proportion of anomalies (for isolation forest)
    """

    def __init__(
        self,
        method="statistical",
        threshold=3.0,
        window_size=None,
        seasonal_adjustment=True,
        contamination=0.05,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.method = method
        self.threshold = threshold
        self.window_size = window_size
        self.seasonal_adjustment = seasonal_adjustment
        self.contamination = contamination
        self.anomalies_ = []
        self.anomaly_scores_ = None
        self.baseline_stats_ = {}

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fit the anomaly detector."""
        if self.validate_input:
            X, y = self._validate_time_series(X, y)
        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> TimeSeriesAnalysisResult:
        """
        Detect anomalies in time series data.

        Returns:
        --------
        result : TimeSeriesAnalysisResult
            Anomaly detection results with detected anomalies and scores
        """
        start_time = time.time()

        if self.validate_input:
            X, _ = self._validate_time_series(X, None)

        try:
            # Use first column for anomaly detection
            if len(X.columns) == 0:
                raise ValueError("No columns found in time series data")

            series = X.iloc[:, 0].dropna()
            if len(series) < 10:
                raise ValueError(
                    "Insufficient data points for anomaly detection (need at least 10)"
                )

            # Seasonal adjustment if requested
            adjusted_series = series
            seasonal_component = None
            if self.seasonal_adjustment:
                adjusted_series, seasonal_component = self._apply_seasonal_adjustment(
                    series
                )

            # Detect anomalies based on method
            if self.method == "statistical":
                anomalies, scores = self._detect_statistical_anomalies(adjusted_series)
            elif self.method == "isolation_forest":
                anomalies, scores = self._detect_isolation_forest_anomalies(
                    adjusted_series
                )
            elif self.method == "zscore":
                anomalies, scores = self._detect_zscore_anomalies(adjusted_series)
            elif self.method == "iqr":
                anomalies, scores = self._detect_iqr_anomalies(adjusted_series)
            else:
                raise ValueError(f"Unknown anomaly detection method: {self.method}")

            self.anomalies_ = anomalies
            self.anomaly_scores_ = scores

            # Calculate anomaly statistics
            anomaly_stats = self._calculate_anomaly_statistics(
                series, anomalies, scores
            )

            # Analyze anomaly patterns
            pattern_analysis = self._analyze_anomaly_patterns(series, anomalies)

            # Generate interpretation
            interpretation = self._generate_anomaly_interpretation(
                anomalies, series, anomaly_stats, pattern_analysis
            )

            # Generate recommendations
            recommendations = self._generate_anomaly_recommendations(
                anomalies, series, anomaly_stats, pattern_analysis
            )

            processing_time = time.time() - start_time

            # Prepare result
            model_parameters = {
                "anomalies": anomalies,
                "anomaly_scores": scores.tolist() if scores is not None else [],
                "n_anomalies": len(anomalies),
                "anomaly_rate": len(anomalies) / len(series) if len(series) > 0 else 0,
                "detection_method": self.method,
                "threshold_used": self.threshold,
                "window_size": self.window_size,
                "seasonal_adjustment_applied": self.seasonal_adjustment
                and seasonal_component is not None,
                "seasonal_component": seasonal_component.tolist()
                if seasonal_component is not None
                else None,
            }

            model_diagnostics = {
                "anomaly_statistics": anomaly_stats,
                "pattern_analysis": pattern_analysis,
                "baseline_statistics": self.baseline_stats_,
            }

            return TimeSeriesAnalysisResult(
                analysis_type="anomaly_detection",
                model_parameters=model_parameters,
                model_diagnostics=model_diagnostics,
                interpretation=interpretation,
                recommendations=recommendations,
                processing_time=processing_time,
                data_quality_score=self._calculate_data_quality_score(X),
                confidence_level=0.95,
            )

        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
            return TimeSeriesAnalysisResult(
                analysis_type="anomaly_detection",
                interpretation=f"Anomaly detection failed: {str(e)}",
                recommendations=["Check data quality and method parameters"],
                processing_time=time.time() - start_time,
            )

    def _apply_seasonal_adjustment(
        self, series: pd.Series
    ) -> Tuple[pd.Series, Optional[np.ndarray]]:
        """Apply seasonal adjustment to the series."""
        try:
            # Simple seasonal decomposition
            if len(series) < 24:  # Not enough data for seasonal decomposition
                return series, None

            # Try to detect seasonality period
            from statsmodels.tsa.seasonal import seasonal_decompose

            # Estimate period (simplified)
            period = min(12, len(series) // 4) if len(series) >= 24 else None
            if period and period >= 2:
                decomposition = seasonal_decompose(
                    series, model="additive", period=period, extrapolate_trend="freq"
                )
                adjusted = series - decomposition.seasonal
                return adjusted, decomposition.seasonal.values
            else:
                return series, None

        except Exception as e:
            logger.debug(f"Seasonal adjustment failed: {e}")
            return series, None

    def _detect_statistical_anomalies(
        self, series: pd.Series
    ) -> Tuple[List[int], np.ndarray]:
        """Detect anomalies using statistical methods (modified Z-score)."""
        values = series.values

        # Calculate modified Z-score (more robust than standard Z-score)
        median = np.median(values)
        mad = np.median(np.abs(values - median))  # Median Absolute Deviation

        if mad == 0:
            # Fallback to standard deviation if MAD is zero
            std_dev = np.std(values)
            if std_dev == 0:
                return [], np.zeros(len(values))
            scores = np.abs(values - median) / std_dev
        else:
            # Modified Z-score
            scores = 0.6745 * (values - median) / mad

        self.baseline_stats_ = {
            "median": median,
            "mad": mad,
            "mean": np.mean(values),
            "std": np.std(values),
        }

        # Find anomalies
        anomaly_mask = np.abs(scores) > self.threshold
        anomalies = list(np.where(anomaly_mask)[0])

        return anomalies, scores

    def _detect_isolation_forest_anomalies(
        self, series: pd.Series
    ) -> Tuple[List[int], np.ndarray]:
        """Detect anomalies using Isolation Forest."""
        try:
            from sklearn.ensemble import IsolationForest

            values = series.values.reshape(-1, 1)

            # Create and fit isolation forest
            clf = IsolationForest(contamination=self.contamination, random_state=42)
            anomaly_labels = clf.fit_predict(values)
            anomaly_scores = clf.decision_function(values)

            # Find anomalies (labeled as -1)
            anomalies = list(np.where(anomaly_labels == -1)[0])

            self.baseline_stats_ = {
                "contamination_rate": self.contamination,
                "n_estimators": clf.n_estimators,
                "mean_score": np.mean(anomaly_scores),
                "std_score": np.std(anomaly_scores),
            }

            return anomalies, anomaly_scores

        except ImportError:
            logger.warning("sklearn not available, falling back to statistical method")
            return self._detect_statistical_anomalies(series)
        except Exception as e:
            logger.error(f"Isolation forest failed: {e}")
            return self._detect_statistical_anomalies(series)

    def _detect_zscore_anomalies(
        self, series: pd.Series
    ) -> Tuple[List[int], np.ndarray]:
        """Detect anomalies using standard Z-score method."""
        values = series.values
        mean_val = np.mean(values)
        std_val = np.std(values)

        if std_val == 0:
            return [], np.zeros(len(values))

        scores = np.abs(values - mean_val) / std_val
        anomaly_mask = scores > self.threshold
        anomalies = list(np.where(anomaly_mask)[0])

        self.baseline_stats_ = {
            "mean": mean_val,
            "std": std_val,
            "threshold": self.threshold,
        }

        return anomalies, scores

    def _detect_iqr_anomalies(self, series: pd.Series) -> Tuple[List[int], np.ndarray]:
        """Detect anomalies using Interquartile Range (IQR) method."""
        values = series.values
        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        iqr = q3 - q1

        if iqr == 0:
            return [], np.zeros(len(values))

        # Calculate anomaly scores based on distance from IQR
        lower_bound = q1 - self.threshold * iqr
        upper_bound = q3 + self.threshold * iqr

        scores = np.maximum(
            (lower_bound - values) / iqr,  # Below lower bound
            (values - upper_bound) / iqr,  # Above upper bound
        )
        scores = np.maximum(scores, 0)  # Only positive scores

        anomaly_mask = scores > 0
        anomalies = list(np.where(anomaly_mask)[0])

        self.baseline_stats_ = {
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
        }

        return anomalies, scores

    def _calculate_anomaly_statistics(
        self, series: pd.Series, anomalies: List[int], scores: np.ndarray
    ) -> Dict:
        """Calculate statistics about detected anomalies."""
        if not anomalies:
            return {
                "n_anomalies": 0,
                "anomaly_rate": 0.0,
                "mean_anomaly_score": 0.0,
                "max_anomaly_score": 0.0,
            }

        anomaly_values = [series.iloc[i] for i in anomalies]
        anomaly_scores_subset = (
            [scores[i] for i in anomalies] if scores is not None else []
        )

        stats = {
            "n_anomalies": len(anomalies),
            "anomaly_rate": len(anomalies) / len(series),
            "anomaly_indices": anomalies,
            "anomaly_values": anomaly_values,
            "mean_anomaly_value": np.mean(anomaly_values),
            "std_anomaly_value": np.std(anomaly_values),
            "min_anomaly_value": np.min(anomaly_values),
            "max_anomaly_value": np.max(anomaly_values),
        }

        if anomaly_scores_subset:
            stats.update(
                {
                    "mean_anomaly_score": np.mean(anomaly_scores_subset),
                    "max_anomaly_score": np.max(anomaly_scores_subset),
                    "min_anomaly_score": np.min(anomaly_scores_subset),
                }
            )

        return stats

    def _analyze_anomaly_patterns(
        self, series: pd.Series, anomalies: List[int]
    ) -> Dict:
        """Analyze patterns in detected anomalies."""
        if not anomalies:
            return {"temporal_clustering": False, "anomaly_clusters": []}

        analysis = {}

        # Temporal clustering analysis
        if len(anomalies) > 1:
            gaps = [anomalies[i + 1] - anomalies[i] for i in range(len(anomalies) - 1)]
            median_gap = np.median(gaps)
            analysis["temporal_clustering"] = any(gap < median_gap / 3 for gap in gaps)

            # Find clusters of anomalies
            clusters = []
            current_cluster = [anomalies[0]]

            for i in range(1, len(anomalies)):
                if anomalies[i] - anomalies[i - 1] <= median_gap / 2:
                    current_cluster.append(anomalies[i])
                else:
                    if len(current_cluster) > 1:
                        clusters.append(current_cluster)
                    current_cluster = [anomalies[i]]

            if len(current_cluster) > 1:
                clusters.append(current_cluster)

            analysis["anomaly_clusters"] = clusters
        else:
            analysis["temporal_clustering"] = False
            analysis["anomaly_clusters"] = []

        # Anomaly direction analysis
        series_median = series.median()
        high_anomalies = [i for i in anomalies if series.iloc[i] > series_median]
        low_anomalies = [i for i in anomalies if series.iloc[i] <= series_median]

        analysis["high_anomalies"] = len(high_anomalies)
        analysis["low_anomalies"] = len(low_anomalies)
        analysis["anomaly_bias"] = (
            "high"
            if len(high_anomalies) > len(low_anomalies)
            else "low"
            if len(low_anomalies) > len(high_anomalies)
            else "balanced"
        )

        return analysis

    def _generate_anomaly_interpretation(
        self, anomalies: List[int], series: pd.Series, stats: Dict, patterns: Dict
    ) -> str:
        """Generate human-readable interpretation of anomaly detection results."""
        if not anomalies:
            return f"No significant anomalies detected in {len(series)} observations using {self.method} method. The time series appears to follow expected patterns."

        n_anomalies = len(anomalies)
        anomaly_rate = stats["anomaly_rate"]

        interpretation = f"Detected {n_anomalies} anomal{'y' if n_anomalies == 1 else 'ies'} ({anomaly_rate:.1%} of observations) using {self.method} method. "

        # Describe anomaly frequency
        if anomaly_rate > 0.1:
            interpretation += "High frequency of anomalies suggests systematic issues or highly volatile behavior. "
        elif anomaly_rate > 0.05:
            interpretation += "Moderate frequency of anomalies indicates occasional irregular behavior. "
        else:
            interpretation += (
                "Low frequency of anomalies suggests rare exceptional events. "
            )

        # Describe clustering patterns
        if patterns.get("temporal_clustering", False):
            n_clusters = len(patterns.get("anomaly_clusters", []))
            interpretation += f"Anomalies show temporal clustering in {n_clusters} distinct period(s). "
        else:
            interpretation += "Anomalies are scattered throughout the time series. "

        # Describe bias
        bias = patterns.get("anomaly_bias", "balanced")
        if bias == "high":
            interpretation += (
                "Anomalies are predominantly above-normal values (positive outliers). "
            )
        elif bias == "low":
            interpretation += (
                "Anomalies are predominantly below-normal values (negative outliers). "
            )
        else:
            interpretation += (
                "Anomalies include both above-normal and below-normal values. "
            )

        return interpretation

    def _generate_anomaly_recommendations(
        self, anomalies: List[int], series: pd.Series, stats: Dict, patterns: Dict
    ) -> List[str]:
        """Generate recommendations based on anomaly detection results."""
        recommendations = []

        if not anomalies:
            recommendations.extend(
                [
                    "No anomalies detected - series appears to follow normal patterns",
                    "Consider monitoring for future anomalies with ongoing analysis",
                    "Validate detection sensitivity with domain knowledge",
                ]
            )
            return recommendations

        anomaly_rate = stats["anomaly_rate"]

        # Rate-based recommendations
        if anomaly_rate > 0.15:
            recommendations.extend(
                [
                    "Very high anomaly rate detected - investigate systematic causes",
                    "Consider adjusting detection parameters or method",
                    "Review data collection and processing procedures",
                ]
            )
        elif anomaly_rate > 0.05:
            recommendations.extend(
                [
                    "Moderate anomaly rate - investigate individual cases",
                    "Consider seasonal or trend adjustments if not already applied",
                ]
            )
        else:
            recommendations.extend(
                [
                    "Low anomaly rate suggests rare exceptional events",
                    "Focus on understanding the context of detected anomalies",
                ]
            )

        # Clustering-based recommendations
        if patterns.get("temporal_clustering", False):
            recommendations.extend(
                [
                    "Temporal clustering of anomalies suggests systematic causes",
                    "Investigate external factors during anomaly periods",
                    "Consider change point detection to identify regime changes",
                ]
            )

        # Bias-based recommendations
        bias = patterns.get("anomaly_bias", "balanced")
        if bias != "balanced":
            direction = "upward" if bias == "high" else "downward"
            recommendations.append(
                f"Anomaly bias toward {direction} outliers suggests systematic {direction} pressure"
            )

        # Method-specific recommendations
        if self.method == "statistical":
            recommendations.append(
                "Consider isolation forest method for complex anomaly patterns"
            )
        elif self.method == "isolation_forest":
            recommendations.append(
                "Statistical methods may provide complementary insights"
            )

        return recommendations
