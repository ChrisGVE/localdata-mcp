"""
Pattern Recognition - High-level convenience functions for MCP tool integration.

Provides perform_clustering, reduce_dimensions, detect_anomalies, and evaluate_patterns
as simple entry points that compose transformers and evaluators.
"""

from typing import Any, Dict, Optional

import numpy as np

from ...logging_manager import get_logger
from ._clustering import ClusteringTransformer
from ._reduction import DimensionalityReductionTransformer
from ._anomaly import AnomalyDetectionTransformer
from ._evaluation import PatternEvaluationTransformer

logger = get_logger(__name__)


def perform_clustering(
    X: np.ndarray,
    algorithm: str = "kmeans",
    n_clusters: Optional[int] = None,
    y_true: Optional[np.ndarray] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Perform comprehensive clustering analysis on data.

    Parameters:
    -----------
    X : array-like
        Input data
    algorithm : str
        Clustering algorithm ('kmeans', 'hierarchical', 'dbscan', 'gmm', 'spectral')
    n_clusters : int, optional
        Number of clusters (auto-selected if None)
    y_true : array-like, optional
        Ground truth labels for validation
    **kwargs
        Additional parameters for the clustering algorithm

    Returns:
    --------
    dict
        Comprehensive clustering results including labels, metrics, and quality assessment
    """
    logger.info(f"Performing clustering analysis with {algorithm}")

    try:
        # Create and fit clustering transformer
        clusterer = ClusteringTransformer(
            algorithm=algorithm, n_clusters=n_clusters, **kwargs
        )
        clusterer.fit(X)

        # Get comprehensive results
        result = clusterer.get_clustering_result(X, y_true)

        # Evaluate clustering quality
        evaluator = PatternEvaluationTransformer("clustering")
        evaluation = evaluator.evaluate_clustering(X, result.labels, y_true, algorithm)

        # Combine results
        output = result.to_dict()
        output["evaluation"] = evaluation.to_dict()

        logger.info(
            f"Clustering completed successfully with {result.n_clusters} clusters"
        )
        return output

    except Exception as e:
        logger.error(f"Clustering analysis failed: {e}")
        raise


def reduce_dimensions(
    X: np.ndarray,
    algorithm: str = "pca",
    n_components: Optional[int] = None,
    y: Optional[np.ndarray] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Perform comprehensive dimensionality reduction on data.

    Parameters:
    -----------
    X : array-like
        Input data
    algorithm : str
        Reduction algorithm ('pca', 'tsne', 'umap', 'ica', 'lda')
    n_components : int, optional
        Number of components (auto-selected if None)
    y : array-like, optional
        Target labels (required for LDA)
    **kwargs
        Additional parameters for the reduction algorithm

    Returns:
    --------
    dict
        Comprehensive dimensionality reduction results
    """
    logger.info(f"Performing dimensionality reduction with {algorithm}")

    try:
        # Create and fit reduction transformer
        reducer = DimensionalityReductionTransformer(
            algorithm=algorithm, n_components=n_components, **kwargs
        )
        reducer.fit(X, y)

        # Get comprehensive results
        result = reducer.get_reduction_result(X)

        # Evaluate reduction quality
        evaluator = PatternEvaluationTransformer("dimensionality_reduction")
        evaluation = evaluator.evaluate_dimensionality_reduction(
            X, result.transformed_data, algorithm
        )

        # Combine results
        output = result.to_dict()
        output["evaluation"] = evaluation.to_dict()

        logger.info(
            f"Dimensionality reduction completed: {result.original_dimensions} -> {result.reduced_dimensions}"
        )
        return output

    except Exception as e:
        logger.error(f"Dimensionality reduction analysis failed: {e}")
        raise


def detect_anomalies(
    X: np.ndarray,
    algorithm: str = "isolation_forest",
    contamination: Optional[float] = None,
    y_true: Optional[np.ndarray] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Perform comprehensive anomaly detection on data.

    Parameters:
    -----------
    X : array-like
        Input data
    algorithm : str
        Detection algorithm ('isolation_forest', 'one_class_svm', 'lof', 'statistical')
    contamination : float, optional
        Expected proportion of outliers
    y_true : array-like, optional
        Ground truth labels for validation
    **kwargs
        Additional parameters for the detection algorithm

    Returns:
    --------
    dict
        Comprehensive anomaly detection results
    """
    logger.info(f"Performing anomaly detection with {algorithm}")

    try:
        # Create and fit anomaly detector
        detector = AnomalyDetectionTransformer(
            algorithm=algorithm, contamination=contamination, **kwargs
        )
        detector.fit(X)

        # Get comprehensive results
        result = detector.get_anomaly_result(X, y_true)

        # Evaluate detection quality
        evaluator = PatternEvaluationTransformer("anomaly_detection")
        evaluation = evaluator.evaluate_anomaly_detection(
            X, result.anomaly_labels, result.anomaly_scores, y_true, algorithm
        )

        # Combine results
        output = result.to_dict()
        output["evaluation"] = evaluation.to_dict()

        logger.info(
            f"Anomaly detection completed: {result.n_anomalies}/{result.n_samples} anomalies detected"
        )
        return output

    except Exception as e:
        logger.error(f"Anomaly detection analysis failed: {e}")
        raise


def evaluate_patterns(
    X: np.ndarray,
    pattern_type: str,
    results: Dict[str, Any],
    y_true: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Evaluate pattern recognition results and provide recommendations.

    Parameters:
    -----------
    X : array-like
        Original input data
    pattern_type : str
        Type of pattern analysis ('clustering', 'dimensionality_reduction', 'anomaly_detection')
    results : dict
        Results from pattern recognition analysis
    y_true : array-like, optional
        Ground truth labels for validation

    Returns:
    --------
    dict
        Comprehensive evaluation results and recommendations
    """
    logger.info(f"Evaluating {pattern_type} patterns")

    try:
        evaluator = PatternEvaluationTransformer(pattern_type)

        if pattern_type == "clustering":
            labels = np.array(results.get("labels", []))
            evaluation = evaluator.evaluate_clustering(X, labels, y_true)
        elif pattern_type == "dimensionality_reduction":
            transformed_data = np.array(results.get("transformed_data", []))
            evaluation = evaluator.evaluate_dimensionality_reduction(
                X, transformed_data
            )
        elif pattern_type == "anomaly_detection":
            labels = np.array(results.get("anomaly_labels", []))
            scores = np.array(results.get("anomaly_scores", []))
            evaluation = evaluator.evaluate_anomaly_detection(X, labels, scores, y_true)
        else:
            raise ValueError(f"Unknown pattern type: {pattern_type}")

        logger.info(f"Pattern evaluation completed: {evaluation.quality_assessment}")
        return evaluation.to_dict()

    except Exception as e:
        logger.error(f"Pattern evaluation failed: {e}")
        raise
