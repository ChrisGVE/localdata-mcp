"""
Time series domain shim implementation.

Provides TimeSeriesShim for enabling time series integration across domains,
handling temporal indexing, seasonality, trends, and forecast intervals.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ....logging_manager import get_logger
from ..interfaces import ConversionError, ConversionRequest
from ..shim_registry import AdapterConfig
from ._base import BaseDomainShim
from ._types import DomainMapping, DomainShimType, SemanticContext

logger = get_logger(__name__)


class TimeSeriesShim(BaseDomainShim):
    """
    Enable time series integration across domains.

    Converts time series data for statistical analysis and regression models,
    handling temporal indexing, seasonality, trends, and forecast intervals.
    """

    def __init__(
        self,
        adapter_id: str = "time_series_shim",
        config: Optional[AdapterConfig] = None,
        **kwargs,
    ):
        """Initialize TimeSeriesShim."""
        super().__init__(
            adapter_id=adapter_id,
            domain_type=DomainShimType.TIME_SERIES,
            config=config,
            **kwargs,
        )

    def _initialize_domain_knowledge(self) -> None:
        """Initialize time series domain knowledge."""
        self._domain_schemas = {
            "time_series_data": {
                "required_fields": ["values", "timestamps"],
                "optional_fields": ["frequency", "seasonal_period", "trend"],
            },
            "forecast_result": {
                "required_fields": ["forecasted_values", "forecast_horizon"],
                "optional_fields": [
                    "confidence_intervals",
                    "prediction_intervals",
                    "forecast_method",
                ],
            },
            "decomposition": {
                "components": ["trend", "seasonal", "residual", "irregular"],
                "methods": ["additive", "multiplicative"],
            },
        }

    def _load_domain_mappings(self) -> None:
        """Load time series domain mappings."""
        # Time Series to Statistical mapping
        self.supported_mappings.append(
            DomainMapping(
                source_domain="time_series",
                target_domain="statistical",
                parameter_mappings={
                    "trend_component": "trend_statistics",
                    "seasonal_component": "seasonal_statistics",
                    "forecast_errors": "residual_analysis",
                    "autocorrelation_function": "correlation_analysis",
                },
                result_transformations={
                    "time_series_decomposition": "component_statistics",
                    "forecast_accuracy": "prediction_validation",
                },
                semantic_hints={
                    "trend_analysis": "regression_analysis",
                    "seasonality_detection": "periodicity_test",
                    "stationarity_testing": "unit_root_test",
                },
                quality_preservation=0.93,
            )
        )

        # Time Series to Regression mapping
        self.supported_mappings.append(
            DomainMapping(
                source_domain="time_series",
                target_domain="regression",
                parameter_mappings={
                    "lagged_features": "regression_features",
                    "trend_features": "linear_predictors",
                    "seasonal_features": "categorical_predictors",
                    "forecast_horizon": "prediction_horizon",
                },
                result_transformations={
                    "time_series_features": "feature_matrix",
                    "temporal_patterns": "predictor_variables",
                },
                semantic_hints={
                    "autoregression": "lag_regression",
                    "trend_modeling": "linear_regression",
                    "seasonal_adjustment": "dummy_variables",
                },
                quality_preservation=0.90,
            )
        )

        # Time Series to Pattern Recognition mapping
        self.supported_mappings.append(
            DomainMapping(
                source_domain="time_series",
                target_domain="pattern_recognition",
                parameter_mappings={
                    "temporal_features": "pattern_features",
                    "seasonal_patterns": "recurring_patterns",
                    "anomaly_scores": "outlier_scores",
                    "change_points": "pattern_breaks",
                },
                result_transformations={
                    "time_series_patterns": "sequence_patterns",
                    "temporal_clusters": "pattern_clusters",
                },
                semantic_hints={
                    "pattern_mining": "sequence_analysis",
                    "anomaly_detection": "outlier_identification",
                    "clustering": "temporal_clustering",
                },
                quality_preservation=0.87,
            )
        )

    def _perform_domain_conversion(
        self,
        request: ConversionRequest,
        mapping: DomainMapping,
        semantic_context: SemanticContext,
    ) -> Any:
        """Perform time series domain conversion."""
        source_data = request.source_data

        if mapping.target_domain == "statistical":
            return self._convert_time_series_to_statistical(
                source_data, mapping, semantic_context
            )
        elif mapping.target_domain == "regression":
            return self._convert_time_series_to_regression(
                source_data, mapping, semantic_context
            )
        elif mapping.target_domain == "pattern_recognition":
            return self._convert_time_series_to_pattern_recognition(
                source_data, mapping, semantic_context
            )
        else:
            raise ConversionError(
                ConversionError.Type.CONVERSION_FAILED,
                f"Unsupported target domain: {mapping.target_domain}",
            )

    def _convert_time_series_to_statistical(
        self, data: Any, mapping: DomainMapping, semantic_context: SemanticContext
    ) -> Dict[str, Any]:
        """Convert time series data to statistical format."""
        if isinstance(data, pd.DataFrame) and hasattr(data.index, "freq"):
            # Handle time series DataFrame
            values = data.iloc[:, 0].values

            # Calculate basic statistics
            result = {
                "time_series_statistics": {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "count": len(values),
                    "trend_slope": self._calculate_trend_slope(values),
                    "stationarity_score": self._assess_stationarity(values),
                }
            }

            # Add autocorrelation analysis
            acf_values = self._calculate_autocorrelation(values)
            if acf_values is not None:
                result["autocorrelation_analysis"] = {
                    "acf_values": acf_values[: min(20, len(acf_values))].tolist(),
                    "significant_lags": self._find_significant_lags(acf_values),
                }

            return result

        elif isinstance(data, dict) and "values" in data:
            # Handle structured time series data
            values = np.array(data["values"])

            result = {
                "time_series_statistics": {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "variance": np.var(values),
                    "skewness": self._calculate_skewness(values),
                    "kurtosis": self._calculate_kurtosis(values),
                }
            }

            # Add decomposition statistics if available
            if "trend" in data or "seasonal" in data:
                result["decomposition_statistics"] = {
                    "trend_strength": self._calculate_component_strength(
                        data.get("trend")
                    ),
                    "seasonal_strength": self._calculate_component_strength(
                        data.get("seasonal")
                    ),
                    "residual_variance": np.var(data.get("residual", [0])),
                }

            return result

        # Fallback
        return {
            "time_series_input": data,
            "conversion_type": "time_series_to_statistical",
            "statistical_context": semantic_context.transformation_hints,
        }

    def _convert_time_series_to_regression(
        self, data: Any, mapping: DomainMapping, semantic_context: SemanticContext
    ) -> Dict[str, Any]:
        """Convert time series data to regression format."""
        if isinstance(data, pd.DataFrame):
            # Create lagged features for regression
            values = data.iloc[:, 0].values

            # Generate lagged features
            n_lags = min(
                10, len(values) // 4
            )  # Use up to 10 lags or 1/4 of data length
            lagged_features = self._create_lagged_features(values, n_lags)

            # Create trend features (same row count as lagged features)
            n_effective = len(values) - n_lags
            trend_features = self._create_trend_features(n_effective)

            # Combine features
            feature_matrix = np.column_stack([lagged_features, trend_features])

            # Create target variable (next period values)
            target_values = values[n_lags:]  # Skip first n_lags values

            return {
                "feature_matrix": feature_matrix,
                "target_variable": target_values,
                "feature_names": (
                    [f"lag_{i + 1}" for i in range(n_lags)] + ["trend", "trend_squared"]
                ),
                "temporal_info": {
                    "n_lags": n_lags,
                    "sample_size": len(target_values),
                    "feature_type": "time_series_derived",
                },
            }

        elif isinstance(data, dict) and "forecast_result" in data:
            # Handle forecast results
            forecasts = data["forecast_result"]

            result = {
                "prediction_data": {
                    "predicted_values": forecasts.get("forecasted_values"),
                    "prediction_intervals": forecasts.get("confidence_intervals"),
                    "prediction_method": forecasts.get(
                        "forecast_method", "time_series"
                    ),
                }
            }

            if "actual_values" in data:
                result["validation_data"] = {
                    "actual_values": data["actual_values"],
                    "prediction_errors": self._calculate_prediction_errors(
                        data["actual_values"], forecasts.get("forecasted_values")
                    ),
                }

            return result

        # Fallback
        return {
            "time_series_input": data,
            "conversion_type": "time_series_to_regression",
            "regression_context": semantic_context.transformation_hints,
        }

    def _convert_time_series_to_pattern_recognition(
        self, data: Any, mapping: DomainMapping, semantic_context: SemanticContext
    ) -> Dict[str, Any]:
        """Convert time series data to pattern recognition format."""
        if isinstance(data, pd.DataFrame):
            values = data.iloc[:, 0].values

            # Extract temporal features for pattern recognition
            temporal_features = self._extract_temporal_features(values)

            # Detect patterns and anomalies
            pattern_info = self._detect_patterns(values)

            return {
                "temporal_features": temporal_features,
                "pattern_detection": pattern_info,
                "sequence_characteristics": {
                    "length": len(values),
                    "variability": np.std(values),
                    "trend_direction": (
                        "increasing"
                        if temporal_features["trend_slope"] > 0
                        else "decreasing"
                    ),
                    "seasonality_present": pattern_info.get("seasonal_detected", False),
                },
            }

        elif isinstance(data, dict) and "seasonal_patterns" in data:
            # Handle seasonal pattern data
            seasonal_data = data["seasonal_patterns"]

            return {
                "recurring_patterns": {
                    "seasonal_components": seasonal_data,
                    "pattern_strength": self._calculate_pattern_strength(seasonal_data),
                    "pattern_frequency": self._estimate_pattern_frequency(
                        seasonal_data
                    ),
                },
                "pattern_type": "seasonal_time_series",
            }

        # Fallback
        return {
            "time_series_input": data,
            "conversion_type": "time_series_to_pattern_recognition",
            "pattern_context": semantic_context.transformation_hints,
        }

    # Helper methods for time series analysis

    def _calculate_trend_slope(self, values: np.ndarray) -> float:
        """Calculate the trend slope of time series."""
        if len(values) < 2:
            return 0.0

        std_values = np.std(values)
        if std_values == 0:
            return 0.0

        x = np.arange(len(values))
        corr = np.corrcoef(x, values)[0, 1]
        if np.isnan(corr):
            return 0.0
        slope = corr * (std_values / np.std(x))
        return float(slope)

    def _assess_stationarity(self, values: np.ndarray) -> float:
        """Assess stationarity of time series (0-1 score)."""
        if len(values) < 10:
            return 0.5

        # Simple rolling statistics approach
        window_size = min(len(values) // 4, 20)
        rolling_mean = pd.Series(values).rolling(window_size).mean().dropna()
        rolling_std = pd.Series(values).rolling(window_size).std().dropna()

        # Check stability of rolling statistics
        mean_avg = np.mean(rolling_mean)
        mean_stability = 1.0 - np.std(rolling_mean) / (abs(mean_avg) + 1e-8)

        std_avg = np.mean(rolling_std)
        std_stability = 1.0 - np.std(rolling_std) / (std_avg + 1e-8)

        # Clamp to [0, 1]
        score = np.mean([mean_stability, std_stability])
        return float(max(0.0, min(1.0, score)))

    def _calculate_autocorrelation(self, values: np.ndarray) -> Optional[np.ndarray]:
        """Calculate autocorrelation function."""
        try:
            n = len(values)
            max_lags = min(n // 4, 40)  # Up to 40 lags or 1/4 of data

            # Center the data
            centered_values = values - np.mean(values)

            # Calculate autocorrelation
            acf = np.correlate(centered_values, centered_values, mode="full")
            acf = acf[n - 1 :]  # Take positive lags
            acf = acf / acf[0]  # Normalize

            return acf[:max_lags]
        except Exception:
            return None

    def _find_significant_lags(
        self, acf_values: np.ndarray, significance_level: float = 0.05
    ) -> List[int]:
        """Find statistically significant autocorrelation lags."""
        n = len(acf_values)
        threshold = 1.96 / np.sqrt(n)  # 95% confidence bounds

        significant_lags = []
        for i, acf_val in enumerate(acf_values[1:], 1):  # Skip lag 0
            if abs(acf_val) > threshold:
                significant_lags.append(i)

        return significant_lags

    def _calculate_skewness(self, values: np.ndarray) -> float:
        """Calculate skewness of time series."""
        try:
            from scipy.stats import skew

            return skew(values)
        except Exception:
            # Manual calculation
            mean_val = np.mean(values)
            std_val = np.std(values)
            if std_val == 0:
                return 0.0
            return np.mean(((values - mean_val) / std_val) ** 3)

    def _calculate_kurtosis(self, values: np.ndarray) -> float:
        """Calculate kurtosis of time series."""
        try:
            from scipy.stats import kurtosis

            return kurtosis(values, fisher=True)  # Excess kurtosis
        except Exception:
            # Manual calculation
            mean_val = np.mean(values)
            std_val = np.std(values)
            if std_val == 0:
                return 0.0
            return np.mean(((values - mean_val) / std_val) ** 4) - 3

    def _calculate_component_strength(self, component: Optional[Any]) -> float:
        """Calculate strength of a time series component."""
        if component is None:
            return 0.0

        component_array = np.array(component)
        if len(component_array) == 0:
            return 0.0

        return np.std(component_array) / (
            np.std(component_array) + 1e-8
        )  # Avoid division by zero

    def _create_lagged_features(self, values: np.ndarray, n_lags: int) -> np.ndarray:
        """Create lagged features for regression."""
        n = len(values)
        lagged_features = np.zeros((n - n_lags, n_lags))

        for i in range(n_lags):
            lagged_features[:, i] = values[n_lags - i - 1 : n - i - 1]

        return lagged_features

    def _create_trend_features(self, n_points: int) -> np.ndarray:
        """Create trend features for regression."""
        trend = np.arange(n_points)
        trend_squared = trend**2

        # Normalize
        trend = (trend - np.mean(trend)) / np.std(trend)
        trend_squared = (trend_squared - np.mean(trend_squared)) / np.std(trend_squared)

        return np.column_stack([trend, trend_squared])

    def _calculate_prediction_errors(
        self, actual: Any, predicted: Any
    ) -> Dict[str, float]:
        """Calculate prediction error metrics."""
        try:
            actual_array = np.array(actual)
            predicted_array = np.array(predicted)

            if len(actual_array) != len(predicted_array):
                min_len = min(len(actual_array), len(predicted_array))
                actual_array = actual_array[:min_len]
                predicted_array = predicted_array[:min_len]

            errors = actual_array - predicted_array

            return {
                "mae": np.mean(np.abs(errors)),
                "mse": np.mean(errors**2),
                "rmse": np.sqrt(np.mean(errors**2)),
                "mape": np.mean(np.abs(errors / (actual_array + 1e-8))) * 100,
            }
        except Exception as e:
            return {"error": str(e)}

    def _extract_temporal_features(self, values: np.ndarray) -> Dict[str, float]:
        """Extract temporal features for pattern recognition."""
        features = {
            "mean": np.mean(values),
            "std": np.std(values),
            "min": np.min(values),
            "max": np.max(values),
            "range": np.max(values) - np.min(values),
            "trend_slope": self._calculate_trend_slope(values),
            "skewness": self._calculate_skewness(values),
            "kurtosis": self._calculate_kurtosis(values),
        }

        # Add change point features
        change_points = self._detect_change_points(values)
        features["n_change_points"] = len(change_points)
        features["change_point_strength"] = (
            np.mean([abs(cp) for cp in change_points]) if change_points else 0.0
        )

        return features

    def _detect_patterns(self, values: np.ndarray) -> Dict[str, Any]:
        """Detect patterns in time series."""
        pattern_info = {}

        # Simple seasonal pattern detection
        if len(values) >= 12:  # Need at least one potential cycle
            seasonal_detected = self._detect_seasonality(values)
            pattern_info["seasonal_detected"] = seasonal_detected

            if seasonal_detected:
                pattern_info["estimated_period"] = self._estimate_seasonal_period(
                    values
                )

        # Trend detection
        trend_slope = self._calculate_trend_slope(values)
        pattern_info["trend_detected"] = abs(trend_slope) > 0.1
        pattern_info["trend_direction"] = (
            "increasing" if trend_slope > 0 else "decreasing"
        )

        return pattern_info

    def _detect_seasonality(self, values: np.ndarray) -> bool:
        """Simple seasonality detection."""
        try:
            # Test common seasonal periods
            for period in [4, 7, 12, 24]:  # Quarterly, weekly, monthly, daily patterns
                if len(values) < 2 * period:
                    continue

                # Calculate autocorrelation at seasonal lag
                acf = self._calculate_autocorrelation(values)
                if acf is not None and len(acf) > period:
                    if abs(acf[period]) > 0.3:  # Threshold for seasonal correlation
                        return True

            return False
        except Exception:
            return False

    def _estimate_seasonal_period(self, values: np.ndarray) -> int:
        """Estimate seasonal period."""
        best_period = 12  # Default
        max_correlation = 0.0

        for period in range(2, min(len(values) // 2, 50)):
            try:
                acf = self._calculate_autocorrelation(values)
                if acf is not None and len(acf) > period:
                    correlation = abs(acf[period])
                    if correlation > max_correlation:
                        max_correlation = correlation
                        best_period = period
            except Exception:
                continue

        return best_period

    def _detect_change_points(self, values: np.ndarray) -> List[float]:
        """Simple change point detection."""
        if len(values) < 4:
            return []

        change_points = []
        window_size = min(len(values) // 10, 20)

        for i in range(window_size, len(values) - window_size):
            left_mean = np.mean(values[i - window_size : i])
            right_mean = np.mean(values[i : i + window_size])

            change_magnitude = abs(right_mean - left_mean)
            if change_magnitude > 2 * np.std(values):
                change_points.append(change_magnitude)

        return change_points

    def _calculate_pattern_strength(self, pattern_data: Any) -> float:
        """Calculate strength of detected patterns."""
        if isinstance(pattern_data, (list, np.ndarray)):
            pattern_array = np.array(pattern_data)
            return np.std(pattern_array) / (np.mean(np.abs(pattern_array)) + 1e-8)
        return 0.5

    def _estimate_pattern_frequency(self, pattern_data: Any) -> int:
        """Estimate frequency of patterns."""
        if isinstance(pattern_data, (list, np.ndarray)):
            return len(pattern_data)
        return 1

    def _normalize_results(
        self, data: Any, mapping: DomainMapping, semantic_context: SemanticContext
    ) -> Any:
        """Normalize time series conversion results for target domain."""
        if not isinstance(data, dict):
            return data

        # Add domain-specific metadata
        data["domain_conversion"] = {
            "source": "time_series",
            "target": mapping.target_domain,
            "semantic_goal": semantic_context.analytical_goal,
            "quality_preservation": mapping.quality_preservation,
        }

        # Ensure consistent structure based on target domain
        if mapping.target_domain == "statistical":
            # Add time series specific statistical context
            data["temporal_statistics_info"] = {
                "analysis_type": "time_series_statistical",
                "temporal_structure_preserved": True,
                "autocorrelation_available": "autocorrelation_analysis" in data,
            }

        elif mapping.target_domain == "regression":
            # Add regression-ready metadata
            if "feature_matrix" in data:
                data["regression_info"] = {
                    "feature_engineering_method": "time_series_transformation",
                    "temporal_dependencies": True,
                    "lag_structure_preserved": True,
                }

        return data
