"""
Pattern Recognition Domain - Comprehensive clustering, dimensionality reduction, and anomaly detection.

This package implements advanced pattern recognition capabilities including clustering algorithms,
dimensionality reduction techniques, and anomaly detection using scikit-learn and specialized libraries.

Key Features:
- Clustering algorithms (K-means, hierarchical, DBSCAN, GMM, spectral clustering)
- Dimensionality reduction (PCA, t-SNE, UMAP, ICA, LDA)
- Anomaly detection (Isolation Forest, One-Class SVM, LOF, statistical methods)
- Comprehensive evaluation metrics and validation
- Full sklearn pipeline compatibility
- Streaming-compatible processing
- Comprehensive result formatting
"""

from ._anomaly import AnomalyDetectionTransformer
from ._clustering import ClusteringTransformer
from ._evaluation import PatternEvaluationTransformer
from ._functions import (
    detect_anomalies,
    evaluate_patterns,
    perform_clustering,
    reduce_dimensions,
)
from ._reduction import DimensionalityReductionTransformer
from ._results import (
    AnomalyDetectionResult,
    ClusteringResult,
    DimensionalityReductionResult,
    PatternEvaluationResult,
)

__all__ = [
    # Result classes
    "ClusteringResult",
    "DimensionalityReductionResult",
    "AnomalyDetectionResult",
    "PatternEvaluationResult",
    # Core transformers
    "ClusteringTransformer",
    "DimensionalityReductionTransformer",
    "AnomalyDetectionTransformer",
    "PatternEvaluationTransformer",
    # High-level functions
    "perform_clustering",
    "reduce_dimensions",
    "detect_anomalies",
    "evaluate_patterns",
]
