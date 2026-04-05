"""
Pattern Recognition - Pattern Evaluation Transformer.

Pattern evaluation and quality assessment for clustering, dimensionality reduction,
and anomaly detection results.
"""

import warnings
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)

from ...logging_manager import get_logger
from ...pipeline.base import AnalysisPipelineBase
from ._results import PatternEvaluationResult

logger = get_logger(__name__)

# Suppress specific warnings that are not critical for our use case
warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


class PatternEvaluationTransformer(AnalysisPipelineBase):
    """Pattern evaluation and quality assessment transformer."""

    def __init__(self, evaluation_type: str = "clustering"):
        """
        Initialize pattern evaluation transformer.

        Parameters:
        -----------
        evaluation_type : str
            Type of evaluation ('clustering', 'dimensionality_reduction', 'anomaly_detection')
        """
        super().__init__(analytical_intention=f"{evaluation_type} pattern evaluation")
        self.evaluation_type = evaluation_type

    def get_analysis_type(self) -> str:
        """Get the specific analysis type this pipeline performs."""
        return f"{self.evaluation_type}_evaluation"

    def _configure_analysis_pipeline(self):
        """Configure analysis steps based on intention and complexity level."""
        return [self.fit]

    def _execute_analysis_step(self, step, data, context):
        """Execute individual analysis step with error handling and metadata."""
        result = step(data, context) if callable(step) else step
        return result, {}

    def _execute_streaming_analysis(self, data):
        """Execute analysis with streaming support for large datasets."""
        return self._execute_standard_analysis(data)

    def _execute_standard_analysis(self, data):
        """Execute analysis on full dataset in memory."""
        return None, {}

    def fit(self, X, y=None):
        """Fit is not needed for evaluation."""
        self.is_fitted_ = True
        return self

    def transform(self, X):
        """Transform returns the original data."""
        return X

    def evaluate_clustering(
        self, X, labels, y_true=None, algorithm="unknown"
    ) -> PatternEvaluationResult:
        """Evaluate clustering results."""
        metrics = {}
        detailed_results = {}
        recommendations = []

        try:
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

            if n_clusters > 1:
                # Internal validation metrics
                silhouette_avg = silhouette_score(X, labels)
                calinski_harabasz = calinski_harabasz_score(X, labels)
                davies_bouldin = davies_bouldin_score(X, labels)

                metrics.update(
                    {
                        "silhouette_score": silhouette_avg,
                        "calinski_harabasz_score": calinski_harabasz,
                        "davies_bouldin_score": davies_bouldin,
                        "n_clusters": n_clusters,
                    }
                )

                # Quality assessment
                if silhouette_avg > 0.7:
                    quality = "Excellent"
                elif silhouette_avg > 0.5:
                    quality = "Good"
                elif silhouette_avg > 0.25:
                    quality = "Fair"
                else:
                    quality = "Poor"

                # Recommendations
                if silhouette_avg < 0.25:
                    recommendations.append(
                        "Consider trying different number of clusters"
                    )
                if davies_bouldin > 2.0:
                    recommendations.append(
                        "High cluster overlap detected, try different algorithm"
                    )
                if n_clusters < 2:
                    recommendations.append("Insufficient clustering structure found")

                # External validation (if ground truth available)
                if y_true is not None:
                    ari = adjusted_rand_score(y_true, labels)
                    nmi = normalized_mutual_info_score(y_true, labels)
                    metrics.update(
                        {"adjusted_rand_index": ari, "normalized_mutual_info": nmi}
                    )

            else:
                quality = "Poor"
                recommendations.append("No meaningful clusters found")

        except Exception as e:
            logger.error(f"Clustering evaluation failed: {e}")
            quality = "Error"
            recommendations.append(f"Evaluation failed: {e}")

        detailed_results["cluster_distribution"] = dict(
            zip(*np.unique(labels, return_counts=True))
        )

        return PatternEvaluationResult(
            evaluation_type="clustering",
            metrics=metrics,
            detailed_results=detailed_results,
            recommendations=recommendations,
            quality_assessment=quality,
        )

    def evaluate_dimensionality_reduction(
        self, X, X_reduced, algorithm="unknown"
    ) -> PatternEvaluationResult:
        """Evaluate dimensionality reduction results."""
        metrics = {}
        detailed_results = {}
        recommendations = []

        try:
            original_dims = X.shape[1]
            reduced_dims = X_reduced.shape[1]
            reduction_ratio = reduced_dims / original_dims

            metrics.update(
                {
                    "original_dimensions": original_dims,
                    "reduced_dimensions": reduced_dims,
                    "reduction_ratio": reduction_ratio,
                }
            )

            # Quality assessment based on reduction ratio and preservation
            if reduction_ratio < 0.1:
                quality = "Excellent"
            elif reduction_ratio < 0.3:
                quality = "Good"
            elif reduction_ratio < 0.5:
                quality = "Fair"
            else:
                quality = "Poor"

            # Recommendations
            if reduction_ratio > 0.7:
                recommendations.append("Limited dimensionality reduction achieved")
            if reduced_dims < 2:
                recommendations.append(
                    "Consider increasing number of components for visualization"
                )
            if reduced_dims > 10:
                recommendations.append(
                    "High number of dimensions may limit interpretability"
                )

        except Exception as e:
            logger.error(f"Dimensionality reduction evaluation failed: {e}")
            quality = "Error"
            recommendations.append(f"Evaluation failed: {e}")

        return PatternEvaluationResult(
            evaluation_type="dimensionality_reduction",
            metrics=metrics,
            detailed_results=detailed_results,
            recommendations=recommendations,
            quality_assessment=quality,
        )

    def evaluate_anomaly_detection(
        self, X, labels, scores, y_true=None, algorithm="unknown"
    ) -> PatternEvaluationResult:
        """Evaluate anomaly detection results."""
        metrics = {}
        detailed_results = {}
        recommendations = []

        try:
            n_samples = len(labels)
            n_anomalies = np.sum(labels == -1)
            contamination = n_anomalies / n_samples

            metrics.update(
                {
                    "n_samples": n_samples,
                    "n_anomalies": n_anomalies,
                    "contamination_rate": contamination,
                }
            )

            # Quality assessment
            if 0.01 <= contamination <= 0.1:
                quality = "Good"
            elif 0.001 <= contamination <= 0.2:
                quality = "Fair"
            else:
                quality = "Poor"

            # Recommendations
            if contamination < 0.001:
                recommendations.append(
                    "Very few anomalies detected, consider lowering threshold"
                )
            if contamination > 0.2:
                recommendations.append(
                    "High anomaly rate detected, consider raising threshold"
                )
            if n_anomalies == 0:
                recommendations.append(
                    "No anomalies detected, verify algorithm parameters"
                )

            # External validation (if ground truth available)
            if y_true is not None:
                from sklearn.metrics import (
                    precision_score,
                    recall_score,
                    f1_score as f1,
                )

                y_pred_binary = (labels == 1).astype(int)
                y_true_binary = (y_true == 1).astype(int)

                precision = precision_score(y_true_binary, y_pred_binary)
                recall = recall_score(y_true_binary, y_pred_binary)
                f1_score = f1(y_true_binary, y_pred_binary)

                metrics.update(
                    {"precision": precision, "recall": recall, "f1_score": f1_score}
                )

                if f1_score > 0.8:
                    quality = "Excellent"
                elif f1_score > 0.6:
                    quality = "Good"
                elif f1_score > 0.4:
                    quality = "Fair"
                else:
                    quality = "Poor"

        except Exception as e:
            logger.error(f"Anomaly detection evaluation failed: {e}")
            quality = "Error"
            recommendations.append(f"Evaluation failed: {e}")

        detailed_results["score_distribution"] = {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "min": float(np.min(scores)),
            "max": float(np.max(scores)),
        }

        return PatternEvaluationResult(
            evaluation_type="anomaly_detection",
            metrics=metrics,
            detailed_results=detailed_results,
            recommendations=recommendations,
            quality_assessment=quality,
        )
