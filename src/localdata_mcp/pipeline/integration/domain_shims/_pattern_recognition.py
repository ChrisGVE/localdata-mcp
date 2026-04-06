"""
Pattern recognition domain shim implementation.

Provides PatternRecognitionShim for bridging pattern recognition with other domains,
converting clustering results, dimensionality reduction outputs, and classification results.
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


class PatternRecognitionShim(BaseDomainShim):
    """
    Bridge pattern recognition with other domains.

    Converts clustering results to statistical summaries, transforms dimensionality
    reduction outputs for visualization, and bridges classification results to business metrics.
    """

    def __init__(
        self,
        adapter_id: str = "pattern_recognition_shim",
        config: Optional[AdapterConfig] = None,
        **kwargs,
    ):
        """Initialize PatternRecognitionShim."""
        super().__init__(
            adapter_id=adapter_id,
            domain_type=DomainShimType.PATTERN_RECOGNITION,
            config=config,
            **kwargs,
        )

    def _initialize_domain_knowledge(self) -> None:
        """Initialize pattern recognition domain knowledge."""
        self._domain_schemas = {
            "clustering_result": {
                "required_fields": ["cluster_labels", "centroids"],
                "optional_fields": ["silhouette_scores", "inertia", "n_clusters"],
            },
            "classification_result": {
                "required_fields": ["predictions", "classes"],
                "optional_fields": [
                    "probabilities",
                    "confidence_scores",
                    "feature_importance",
                ],
            },
            "dimensionality_reduction": {
                "required_fields": ["transformed_data", "n_components"],
                "optional_fields": [
                    "explained_variance_ratio",
                    "components",
                    "reconstruction_error",
                ],
            },
        }

    def _load_domain_mappings(self) -> None:
        """Load pattern recognition domain mappings."""
        # Pattern Recognition to Statistical mapping
        self.supported_mappings.append(
            DomainMapping(
                source_domain="pattern_recognition",
                target_domain="statistical",
                parameter_mappings={
                    "cluster_centroids": "group_means",
                    "silhouette_scores": "cluster_validity",
                    "feature_importance": "variable_importance",
                    "classification_accuracy": "test_statistics",
                },
                result_transformations={
                    "clustering_summary": "group_statistics",
                    "classification_metrics": "hypothesis_test_results",
                    "dimensionality_reduction": "principal_component_analysis",
                },
                semantic_hints={
                    "cluster_analysis": "group_comparison",
                    "feature_selection": "variable_selection",
                    "anomaly_detection": "outlier_analysis",
                },
                quality_preservation=0.91,
            )
        )

        # Pattern Recognition to Regression mapping
        self.supported_mappings.append(
            DomainMapping(
                source_domain="pattern_recognition",
                target_domain="regression",
                parameter_mappings={
                    "feature_weights": "regression_coefficients",
                    "transformed_features": "predictor_variables",
                    "anomaly_scores": "leverage_scores",
                    "cluster_assignments": "categorical_predictors",
                },
                result_transformations={
                    "pattern_features": "regression_features",
                    "classification_scores": "continuous_predictors",
                },
                semantic_hints={
                    "feature_extraction": "feature_engineering",
                    "dimensionality_reduction": "predictor_reduction",
                    "clustering": "categorical_encoding",
                },
                quality_preservation=0.88,
            )
        )

        # Pattern Recognition to Time Series mapping
        self.supported_mappings.append(
            DomainMapping(
                source_domain="pattern_recognition",
                target_domain="time_series",
                parameter_mappings={
                    "sequential_patterns": "temporal_patterns",
                    "change_points": "structural_breaks",
                    "pattern_frequency": "seasonal_components",
                    "anomaly_detection": "outlier_identification",
                },
                result_transformations={
                    "pattern_sequences": "time_series_segments",
                    "clustering_temporal": "regime_identification",
                },
                semantic_hints={
                    "sequence_mining": "pattern_analysis",
                    "temporal_clustering": "regime_detection",
                    "change_detection": "structural_break_analysis",
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
        """Perform pattern recognition domain conversion."""
        source_data = request.source_data

        if mapping.target_domain == "statistical":
            return self._convert_pattern_recognition_to_statistical(
                source_data, mapping, semantic_context
            )
        elif mapping.target_domain == "regression":
            return self._convert_pattern_recognition_to_regression(
                source_data, mapping, semantic_context
            )
        elif mapping.target_domain == "time_series":
            return self._convert_pattern_recognition_to_time_series(
                source_data, mapping, semantic_context
            )
        else:
            raise ConversionError(
                ConversionError.Type.CONVERSION_FAILED,
                f"Unsupported target domain: {mapping.target_domain}",
            )

    def _convert_pattern_recognition_to_statistical(
        self, data: Any, mapping: DomainMapping, semantic_context: SemanticContext
    ) -> Dict[str, Any]:
        """Convert pattern recognition results to statistical format."""
        if isinstance(data, dict):
            # Handle clustering results
            if "cluster_labels" in data and "centroids" in data:
                cluster_labels = np.array(data["cluster_labels"])
                centroids = np.array(data["centroids"])

                # Calculate cluster statistics
                unique_labels = np.unique(cluster_labels)
                cluster_stats = {}

                for label in unique_labels:
                    cluster_mask = cluster_labels == label
                    cluster_stats[f"cluster_{label}"] = {
                        "size": np.sum(cluster_mask),
                        "proportion": np.mean(cluster_mask),
                        "centroid": (
                            centroids[label].tolist()
                            if label < len(centroids)
                            else None
                        ),
                    }

                return {
                    "group_statistics": cluster_stats,
                    "clustering_validity": {
                        "n_clusters": len(unique_labels),
                        "silhouette_score": data.get("silhouette_score"),
                        "inertia": data.get("inertia"),
                        "cluster_sizes": [
                            stats["size"] for stats in cluster_stats.values()
                        ],
                    },
                }

            # Handle classification results
            if "predictions" in data and "probabilities" in data:
                predictions = np.array(data["predictions"])
                probabilities = np.array(data["probabilities"])

                # Calculate classification statistics
                unique_classes = np.unique(predictions)
                class_stats = {}

                for cls in unique_classes:
                    class_mask = predictions == cls
                    class_probs = (
                        probabilities[class_mask]
                        if len(probabilities.shape) == 1
                        else probabilities[class_mask, cls]
                    )

                    class_stats[f"class_{cls}"] = {
                        "frequency": np.sum(class_mask),
                        "proportion": np.mean(class_mask),
                        "avg_confidence": (
                            np.mean(class_probs) if len(class_probs) > 0 else 0.0
                        ),
                        "confidence_std": (
                            np.std(class_probs) if len(class_probs) > 0 else 0.0
                        ),
                    }

                return {
                    "classification_statistics": class_stats,
                    "prediction_confidence": {
                        "overall_confidence": (
                            np.mean(np.max(probabilities, axis=1))
                            if len(probabilities.shape) > 1
                            else np.mean(probabilities)
                        ),
                        "uncertainty_measure": (
                            1.0 - np.mean(np.max(probabilities, axis=1))
                            if len(probabilities.shape) > 1
                            else 1.0 - np.mean(probabilities)
                        ),
                    },
                }

            # Handle dimensionality reduction results
            if "transformed_data" in data and "explained_variance_ratio" in data:
                return {
                    "principal_component_analysis": {
                        "explained_variance_ratio": data["explained_variance_ratio"],
                        "cumulative_variance": np.cumsum(
                            data["explained_variance_ratio"]
                        ).tolist(),
                        "n_components": data.get(
                            "n_components", len(data["explained_variance_ratio"])
                        ),
                        "dimensionality_reduction_ratio": data.get("n_components", 0)
                        / data.get("original_dimensions", 1),
                    }
                }

        # Fallback
        return {
            "pattern_recognition_input": data,
            "conversion_type": "pattern_recognition_to_statistical",
            "statistical_context": semantic_context.transformation_hints,
        }

    def _convert_pattern_recognition_to_regression(
        self, data: Any, mapping: DomainMapping, semantic_context: SemanticContext
    ) -> Dict[str, Any]:
        """Convert pattern recognition results to regression format."""
        if isinstance(data, dict):
            # Handle feature importance from classification/clustering
            if "feature_importance" in data or "feature_weights" in data:
                importance = data.get("feature_importance", data.get("feature_weights"))

                return {
                    "feature_selection": {
                        "feature_importance": importance,
                        "feature_names": data.get("feature_names", []),
                        "selection_method": "pattern_recognition_based",
                        "importance_threshold": (
                            np.percentile(importance, 75)
                            if isinstance(importance, (list, np.ndarray))
                            else 0.5
                        ),
                    }
                }

            # Handle transformed features from dimensionality reduction
            if "transformed_data" in data:
                transformed_data = np.array(data["transformed_data"])

                return {
                    "predictor_variables": {
                        "transformed_features": transformed_data,
                        "n_components": (
                            transformed_data.shape[1]
                            if len(transformed_data.shape) > 1
                            else 1
                        ),
                        "transformation_method": data.get(
                            "method", "dimensionality_reduction"
                        ),
                        "explained_variance": data.get("explained_variance_ratio"),
                    }
                }

            # Handle cluster assignments as categorical predictors
            if "cluster_labels" in data:
                cluster_labels = np.array(data["cluster_labels"])

                # Create dummy variables for clusters
                unique_clusters = np.unique(cluster_labels)
                dummy_matrix = np.zeros((len(cluster_labels), len(unique_clusters)))

                for i, cluster in enumerate(unique_clusters):
                    dummy_matrix[:, i] = (cluster_labels == cluster).astype(int)

                return {
                    "categorical_predictors": {
                        "cluster_dummies": dummy_matrix,
                        "cluster_labels": cluster_labels,
                        "n_clusters": len(unique_clusters),
                        "cluster_names": [f"cluster_{c}" for c in unique_clusters],
                    }
                }

        # Fallback
        return {
            "pattern_recognition_input": data,
            "conversion_type": "pattern_recognition_to_regression",
            "regression_context": semantic_context.transformation_hints,
        }

    def _convert_pattern_recognition_to_time_series(
        self, data: Any, mapping: DomainMapping, semantic_context: SemanticContext
    ) -> Dict[str, Any]:
        """Convert pattern recognition results to time series format."""
        if isinstance(data, dict):
            # Handle sequential patterns
            if "sequential_patterns" in data or "temporal_patterns" in data:
                patterns = data.get(
                    "sequential_patterns", data.get("temporal_patterns")
                )

                return {
                    "pattern_analysis": {
                        "detected_patterns": patterns,
                        "pattern_frequency": self._calculate_pattern_frequency(
                            patterns
                        ),
                        "pattern_strength": self._calculate_pattern_strength(patterns),
                        "temporal_structure": "sequential",
                    }
                }

            # Handle change point detection
            if "change_points" in data or "anomaly_scores" in data:
                change_points = data.get("change_points", [])
                anomaly_scores = data.get("anomaly_scores", [])

                return {
                    "structural_break_analysis": {
                        "change_points": change_points,
                        "anomaly_scores": anomaly_scores,
                        "n_change_points": len(change_points),
                        "break_detection_method": "pattern_recognition_based",
                    }
                }

            # Handle temporal clustering
            if "cluster_labels" in data and "temporal_info" in data:
                cluster_labels = data["cluster_labels"]
                temporal_info = data["temporal_info"]

                # Identify regime changes based on cluster transitions
                regime_changes = []
                if len(cluster_labels) > 1:
                    for i in range(1, len(cluster_labels)):
                        if cluster_labels[i] != cluster_labels[i - 1]:
                            regime_changes.append(i)

                return {
                    "regime_identification": {
                        "regime_labels": cluster_labels,
                        "regime_changes": regime_changes,
                        "n_regimes": len(np.unique(cluster_labels)),
                        "temporal_index": temporal_info.get(
                            "timestamps", list(range(len(cluster_labels)))
                        ),
                    }
                }

        elif isinstance(data, pd.DataFrame):
            # Handle DataFrame with temporal patterns
            if "timestamp" in data.columns or hasattr(data.index, "freq"):
                # Extract pattern features for time series
                pattern_features = {}

                for col in data.columns:
                    if col != "timestamp":
                        values = data[col].values
                        pattern_features[col] = {
                            "pattern_strength": np.std(values),
                            "trend": self._calculate_trend_slope(values),
                            "volatility": self._calculate_volatility(values),
                        }

                return {
                    "temporal_pattern_features": pattern_features,
                    "time_index": (
                        data.index.tolist() if hasattr(data, "index") else None
                    ),
                }

        # Fallback
        return {
            "pattern_recognition_input": data,
            "conversion_type": "pattern_recognition_to_time_series",
            "temporal_context": semantic_context.transformation_hints,
        }

    def _calculate_pattern_frequency(self, patterns: Any) -> Dict[str, int]:
        """Calculate frequency of detected patterns."""
        if isinstance(patterns, list):
            # Handle list of pattern dicts (e.g. [{'pattern': ..., 'frequency': N}])
            if patterns and isinstance(patterns[0], dict):
                return {
                    f"pattern_{i}": p.get("frequency", 1)
                    for i, p in enumerate(patterns)
                }
            try:
                pattern_array = np.array(patterns)
                unique, counts = np.unique(pattern_array, return_counts=True)
                return {f"pattern_{i}": int(count) for i, count in enumerate(counts)}
            except (TypeError, ValueError):
                return {f"pattern_{i}": 1 for i in range(len(patterns))}
        return {}

    def _calculate_pattern_strength(self, patterns: Any) -> float:
        """Calculate strength of detected patterns."""
        if isinstance(patterns, list):
            # Handle list of pattern dicts
            if patterns and isinstance(patterns[0], dict):
                freqs = [p.get("frequency", 1) for p in patterns]
                arr = np.array(freqs, dtype=float)
                return float(np.std(arr) / (np.mean(np.abs(arr)) + 1e-8))
            try:
                pattern_array = np.array(patterns, dtype=float)
                return float(
                    np.std(pattern_array) / (np.mean(np.abs(pattern_array)) + 1e-8)
                )
            except (TypeError, ValueError):
                return 0.5
        if isinstance(patterns, np.ndarray):
            return float(np.std(patterns) / (np.mean(np.abs(patterns)) + 1e-8))
        return 0.5

    def _calculate_trend_slope(self, values: np.ndarray) -> float:
        """Calculate trend slope."""
        if len(values) < 2:
            return 0.0

        std_values = np.std(values)
        if std_values == 0:
            return 0.0

        x = np.arange(len(values))
        corr = np.corrcoef(x, values)[0, 1]
        if np.isnan(corr):
            return 0.0
        return float(corr * (std_values / np.std(x)))

    def _calculate_volatility(self, values: np.ndarray) -> float:
        """Calculate volatility measure."""
        if len(values) < 2:
            return 0.0

        returns = np.diff(values) / (values[:-1] + 1e-8)
        return np.std(returns)

    def _normalize_results(
        self, data: Any, mapping: DomainMapping, semantic_context: SemanticContext
    ) -> Any:
        """Normalize pattern recognition conversion results for target domain."""
        if not isinstance(data, dict):
            return data

        # Add domain-specific metadata
        data["domain_conversion"] = {
            "source": "pattern_recognition",
            "target": mapping.target_domain,
            "semantic_goal": semantic_context.analytical_goal,
            "quality_preservation": mapping.quality_preservation,
        }

        # Ensure consistent structure based on target domain
        if mapping.target_domain == "statistical":
            # Add pattern recognition specific statistical context
            data["pattern_statistics_info"] = {
                "analysis_type": "pattern_recognition_statistical",
                "clustering_available": any("cluster" in key for key in data.keys()),
                "classification_available": any(
                    "classification" in key for key in data.keys()
                ),
                "dimensionality_reduction_available": any(
                    "pca" in key.lower() or "component" in key for key in data.keys()
                ),
            }

        elif mapping.target_domain == "regression":
            # Add regression-ready metadata
            if "feature_selection" in data or "predictor_variables" in data:
                data["regression_info"] = {
                    "feature_engineering_method": "pattern_recognition_transformation",
                    "dimensionality_reduced": "transformed_features" in str(data),
                    "categorical_encoding_applied": "cluster_dummies" in str(data),
                }

        return data
