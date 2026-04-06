"""
Statistical domain shim implementation.

Provides StatisticalShim for bridging statistical analysis with other domains,
handling correlation matrices, hypothesis test results, and descriptive statistics.
"""

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from ....logging_manager import get_logger
from ..interfaces import ConversionError, ConversionRequest
from ..shim_registry import AdapterConfig
from ._base import BaseDomainShim
from ._types import DomainMapping, DomainShimType, SemanticContext

logger = get_logger(__name__)


class StatisticalShim(BaseDomainShim):
    """
    Bridge statistical analysis with other domains.

    Handles conversions between statistical analysis results and other domain formats,
    including correlation matrices, hypothesis test results, and descriptive statistics.
    """

    def __init__(
        self,
        adapter_id: str = "statistical_shim",
        config: Optional[AdapterConfig] = None,
        **kwargs,
    ):
        """Initialize StatisticalShim."""
        super().__init__(
            adapter_id=adapter_id,
            domain_type=DomainShimType.STATISTICAL,
            config=config,
            **kwargs,
        )

    def _initialize_domain_knowledge(self) -> None:
        """Initialize statistical domain knowledge."""
        self._domain_schemas = {
            "correlation_matrix": {
                "type": "symmetric_matrix",
                "value_range": [-1, 1],
                "diagonal_ones": True,
            },
            "hypothesis_test": {
                "required_fields": ["statistic", "p_value"],
                "optional_fields": [
                    "degrees_of_freedom",
                    "effect_size",
                    "confidence_interval",
                ],
            },
            "descriptive_stats": {
                "required_fields": ["mean", "std", "count"],
                "optional_fields": ["median", "min", "max", "skewness", "kurtosis"],
            },
        }

        self._parameter_maps = {
            "statistical_to_regression": {
                "correlation_coefficient": "feature_correlation",
                "p_value": "significance",
                "confidence_interval": "prediction_interval",
            },
            "statistical_to_time_series": {
                "trend_coefficient": "trend",
                "seasonal_component": "seasonality",
                "residuals": "error_terms",
            },
        }

    def _load_domain_mappings(self) -> None:
        """Load statistical domain mappings."""
        # Statistical to Regression mapping
        self.supported_mappings.append(
            DomainMapping(
                source_domain="statistical",
                target_domain="regression",
                parameter_mappings={
                    "correlation_matrix": "feature_correlation_matrix",
                    "test_statistics": "model_diagnostics",
                    "confidence_intervals": "prediction_intervals",
                    "p_values": "feature_significance",
                },
                result_transformations={
                    "statistical_summary": "regression_features",
                    "hypothesis_results": "model_validation",
                },
                semantic_hints={
                    "correlation_analysis": "feature_selection",
                    "normality_test": "assumption_checking",
                    "outlier_detection": "data_preprocessing",
                },
                quality_preservation=0.95,
            )
        )

        # Statistical to Time Series mapping
        self.supported_mappings.append(
            DomainMapping(
                source_domain="statistical",
                target_domain="time_series",
                parameter_mappings={
                    "trend_statistics": "trend_parameters",
                    "seasonal_tests": "seasonality_detection",
                    "stationarity_tests": "differencing_requirements",
                },
                result_transformations={
                    "time_series_stats": "ts_characteristics",
                    "decomposition_results": "component_analysis",
                },
                semantic_hints={
                    "trend_analysis": "trend_modeling",
                    "seasonal_decomposition": "seasonal_adjustment",
                    "autocorrelation": "lag_analysis",
                },
                quality_preservation=0.90,
            )
        )

        # Statistical to Pattern Recognition mapping
        self.supported_mappings.append(
            DomainMapping(
                source_domain="statistical",
                target_domain="pattern_recognition",
                parameter_mappings={
                    "principal_components": "dimensionality_reduction",
                    "cluster_statistics": "clustering_validation",
                    "distribution_parameters": "model_parameters",
                },
                result_transformations={
                    "statistical_features": "pattern_features",
                    "correlation_structure": "similarity_matrix",
                },
                semantic_hints={
                    "pca_analysis": "dimensionality_reduction",
                    "cluster_validation": "clustering_evaluation",
                    "anomaly_detection": "outlier_identification",
                },
                quality_preservation=0.85,
            )
        )

    def _perform_domain_conversion(
        self,
        request: ConversionRequest,
        mapping: DomainMapping,
        semantic_context: SemanticContext,
    ) -> Any:
        """Perform statistical domain conversion."""
        source_data = request.source_data

        if mapping.target_domain == "regression":
            return self._convert_statistical_to_regression(
                source_data, mapping, semantic_context
            )
        elif mapping.target_domain == "time_series":
            return self._convert_statistical_to_time_series(
                source_data, mapping, semantic_context
            )
        elif mapping.target_domain == "pattern_recognition":
            return self._convert_statistical_to_pattern_recognition(
                source_data, mapping, semantic_context
            )
        else:
            raise ConversionError(
                ConversionError.Type.CONVERSION_FAILED,
                f"Unsupported target domain: {mapping.target_domain}",
            )

    def _convert_statistical_to_regression(
        self, data: Any, mapping: DomainMapping, semantic_context: SemanticContext
    ) -> Dict[str, Any]:
        """Convert statistical results to regression-friendly format."""
        if isinstance(data, dict):
            # Handle statistical test results
            if "correlation_matrix" in data:
                # Convert correlation matrix to feature correlation
                corr_matrix = data["correlation_matrix"]
                if isinstance(corr_matrix, pd.DataFrame):
                    return {
                        "feature_correlation_matrix": corr_matrix.values,
                        "feature_names": list(corr_matrix.columns),
                        "correlation_significance": data.get("p_values", None),
                    }

            # Handle hypothesis test results
            if "test_statistic" in data and "p_value" in data:
                return {
                    "model_diagnostics": {
                        "test_statistic": data["test_statistic"],
                        "significance": data["p_value"],
                        "degrees_of_freedom": data.get("degrees_of_freedom"),
                        "effect_size": data.get("effect_size"),
                    }
                }

        elif isinstance(data, pd.DataFrame):
            # Convert statistical summary to regression features
            if data.index.name in ["mean", "std", "count"] or any(
                stat in str(data.index) for stat in ["mean", "std", "median"]
            ):
                return {
                    "descriptive_statistics": data.to_dict(),
                    "feature_names": list(data.columns),
                    "summary_type": "regression_preprocessing",
                }

        # Fallback: return as-is with regression context
        return {
            "statistical_input": data,
            "conversion_type": "statistical_to_regression",
            "metadata": semantic_context.__dict__,
        }

    def _convert_statistical_to_time_series(
        self, data: Any, mapping: DomainMapping, semantic_context: SemanticContext
    ) -> Dict[str, Any]:
        """Convert statistical results to time series format."""
        if isinstance(data, dict):
            # Handle time series statistical tests
            if "stationarity_test" in data:
                return {
                    "stationarity_info": {
                        "test_statistic": data.get("test_statistic"),
                        "p_value": data.get("p_value"),
                        "is_stationary": data.get("p_value", 1.0) < 0.05,
                        "differencing_suggested": data.get("p_value", 1.0) > 0.05,
                    }
                }

            # Handle autocorrelation analysis
            if "autocorrelation" in data or "acf" in data:
                return {
                    "autocorrelation_structure": {
                        "acf_values": data.get("autocorrelation", data.get("acf")),
                        "pacf_values": data.get(
                            "partial_autocorrelation", data.get("pacf")
                        ),
                        "significant_lags": data.get("significant_lags", []),
                    }
                }

        elif isinstance(data, pd.DataFrame):
            # Handle time series decomposition statistics
            if any(
                col.lower() in ["trend", "seasonal", "residual"] for col in data.columns
            ):
                return {
                    "decomposition_statistics": data.to_dict(),
                    "components": list(data.columns),
                    "time_index": (
                        data.index.tolist() if hasattr(data, "index") else None
                    ),
                }

        # Fallback
        return {
            "statistical_input": data,
            "conversion_type": "statistical_to_time_series",
            "temporal_context": semantic_context.transformation_hints,
        }

    def _convert_statistical_to_pattern_recognition(
        self, data: Any, mapping: DomainMapping, semantic_context: SemanticContext
    ) -> Dict[str, Any]:
        """Convert statistical results to pattern recognition format."""
        if isinstance(data, dict):
            # Handle PCA results
            if "principal_components" in data or "explained_variance" in data:
                return {
                    "dimensionality_reduction": {
                        "components": data.get("principal_components"),
                        "explained_variance_ratio": data.get(
                            "explained_variance_ratio"
                        ),
                        "cumulative_variance": data.get("cumulative_variance"),
                        "n_components": data.get("n_components"),
                    }
                }

            # Handle clustering validation statistics
            if "silhouette_score" in data or "inertia" in data:
                return {
                    "clustering_validation": {
                        "silhouette_score": data.get("silhouette_score"),
                        "inertia": data.get("inertia"),
                        "calinski_harabasz_score": data.get("calinski_harabasz_score"),
                        "davies_bouldin_score": data.get("davies_bouldin_score"),
                    }
                }

        elif isinstance(data, pd.DataFrame):
            # Handle correlation matrix for similarity analysis
            if data.shape[0] == data.shape[1] and all(
                data.iloc[i, i] == 1.0 for i in range(min(3, data.shape[0]))
            ):
                # Likely a correlation matrix
                return {
                    "similarity_matrix": data.values,
                    "feature_names": list(data.columns),
                    "matrix_type": "correlation",
                }

        # Fallback
        return {
            "statistical_input": data,
            "conversion_type": "statistical_to_pattern_recognition",
            "pattern_hints": semantic_context.transformation_hints,
        }

    def _normalize_results(
        self, data: Any, mapping: DomainMapping, semantic_context: SemanticContext
    ) -> Any:
        """Normalize statistical conversion results for target domain."""
        if not isinstance(data, dict):
            return data

        # Add domain-specific metadata
        data["domain_conversion"] = {
            "source": "statistical",
            "target": mapping.target_domain,
            "semantic_goal": semantic_context.analytical_goal,
            "quality_preservation": mapping.quality_preservation,
        }

        # Ensure consistent structure
        if mapping.target_domain == "regression":
            # Ensure regression-compatible structure
            if "feature_correlation_matrix" in data:
                # Add metadata for regression feature selection
                data["feature_selection_hints"] = {
                    "highly_correlated_features": [],
                    "independent_features": [],
                    "multicollinearity_warnings": [],
                }

        return data
