"""
Pattern Recognition Result Dataclasses.

Comprehensive result containers for clustering, dimensionality reduction,
anomaly detection, and pattern evaluation analyses.
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass

import numpy as np


@dataclass
class ClusteringResult:
    """Comprehensive clustering analysis results."""

    algorithm: str
    n_clusters: int
    labels: np.ndarray
    cluster_centers: Optional[np.ndarray]
    inertia: Optional[float]

    # Quality metrics
    silhouette_avg: Optional[float]
    silhouette_samples: Optional[np.ndarray]
    calinski_harabasz: Optional[float]
    davies_bouldin: Optional[float]

    # External validation metrics (if ground truth available)
    adjusted_rand_score: Optional[float]
    normalized_mutual_info: Optional[float]
    homogeneity_score: Optional[float]
    completeness_score: Optional[float]
    v_measure_score: Optional[float]

    # Algorithm-specific results
    algorithm_params: Dict[str, Any]
    fit_time: float
    convergence_info: Dict[str, Any]
    cluster_stats: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for JSON serialization."""
        result = {
            "algorithm": self.algorithm,
            "n_clusters": self.n_clusters,
            "labels": self.labels.tolist() if self.labels is not None else None,
            "cluster_centers": self.cluster_centers.tolist()
            if self.cluster_centers is not None
            else None,
            "inertia": self.inertia,
            "silhouette_avg": self.silhouette_avg,
            "silhouette_samples": self.silhouette_samples.tolist()
            if self.silhouette_samples is not None
            else None,
            "calinski_harabasz": self.calinski_harabasz,
            "davies_bouldin": self.davies_bouldin,
            "adjusted_rand_score": self.adjusted_rand_score,
            "normalized_mutual_info": self.normalized_mutual_info,
            "homogeneity_score": self.homogeneity_score,
            "completeness_score": self.completeness_score,
            "v_measure_score": self.v_measure_score,
            "algorithm_params": self.algorithm_params,
            "fit_time": self.fit_time,
            "convergence_info": self.convergence_info,
            "cluster_stats": self.cluster_stats,
        }
        return result


@dataclass
class DimensionalityReductionResult:
    """Comprehensive dimensionality reduction results."""

    algorithm: str
    original_dimensions: int
    reduced_dimensions: int
    transformed_data: np.ndarray

    # Algorithm-specific results
    explained_variance_ratio: Optional[np.ndarray]
    cumulative_variance_ratio: Optional[np.ndarray]
    singular_values: Optional[np.ndarray]
    components: Optional[np.ndarray]

    # Quality metrics
    reconstruction_error: Optional[float]
    kl_divergence: Optional[float]  # For t-SNE
    trustworthiness: Optional[float]
    continuity: Optional[float]

    # Algorithm parameters and metadata
    algorithm_params: Dict[str, Any]
    fit_time: float
    transform_time: float
    convergence_info: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for JSON serialization."""
        result = {
            "algorithm": self.algorithm,
            "original_dimensions": self.original_dimensions,
            "reduced_dimensions": self.reduced_dimensions,
            "transformed_data": self.transformed_data.tolist(),
            "explained_variance_ratio": self.explained_variance_ratio.tolist()
            if self.explained_variance_ratio is not None
            else None,
            "cumulative_variance_ratio": self.cumulative_variance_ratio.tolist()
            if self.cumulative_variance_ratio is not None
            else None,
            "singular_values": self.singular_values.tolist()
            if self.singular_values is not None
            else None,
            "components": self.components.tolist()
            if self.components is not None
            else None,
            "reconstruction_error": self.reconstruction_error,
            "kl_divergence": self.kl_divergence,
            "trustworthiness": self.trustworthiness,
            "continuity": self.continuity,
            "algorithm_params": self.algorithm_params,
            "fit_time": self.fit_time,
            "transform_time": self.transform_time,
            "convergence_info": self.convergence_info,
        }
        return result


@dataclass
class AnomalyDetectionResult:
    """Comprehensive anomaly detection results."""

    algorithm: str
    n_samples: int
    n_anomalies: int
    anomaly_labels: np.ndarray  # -1 for anomalies, 1 for normal
    anomaly_scores: Optional[np.ndarray]

    # Threshold information
    threshold: Optional[float]
    contamination: Optional[float]

    # Quality metrics (if ground truth available)
    precision: Optional[float]
    recall: Optional[float]
    f1_score: Optional[float]
    auc_roc: Optional[float]

    # Statistical information
    score_statistics: Dict[str, float]
    anomaly_statistics: Dict[str, Any]

    # Algorithm parameters and metadata
    algorithm_params: Dict[str, Any]
    fit_time: float
    prediction_time: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for JSON serialization."""
        result = {
            "algorithm": self.algorithm,
            "n_samples": self.n_samples,
            "n_anomalies": self.n_anomalies,
            "anomaly_labels": self.anomaly_labels.tolist(),
            "anomaly_scores": self.anomaly_scores.tolist()
            if self.anomaly_scores is not None
            else None,
            "threshold": self.threshold,
            "contamination": self.contamination,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "auc_roc": self.auc_roc,
            "score_statistics": self.score_statistics,
            "anomaly_statistics": self.anomaly_statistics,
            "algorithm_params": self.algorithm_params,
            "fit_time": self.fit_time,
            "prediction_time": self.prediction_time,
        }
        return result


@dataclass
class PatternEvaluationResult:
    """Comprehensive pattern evaluation results."""

    evaluation_type: str
    metrics: Dict[str, float]
    detailed_results: Dict[str, Any]
    recommendations: List[str]
    quality_assessment: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for JSON serialization."""
        return {
            "evaluation_type": self.evaluation_type,
            "metrics": self.metrics,
            "detailed_results": self.detailed_results,
            "recommendations": self.recommendations,
            "quality_assessment": self.quality_assessment,
        }
