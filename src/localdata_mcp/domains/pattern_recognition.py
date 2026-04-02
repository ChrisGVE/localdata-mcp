"""
Pattern Recognition Domain - Comprehensive clustering, dimensionality reduction, and anomaly detection.

This module implements advanced pattern recognition capabilities including clustering algorithms,
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

import warnings
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import time
import json

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    silhouette_score, calinski_harabasz_score, davies_bouldin_score,
    adjusted_rand_score, normalized_mutual_info_score, homogeneity_score,
    completeness_score, v_measure_score
)

# Clustering algorithms
from sklearn.cluster import (
    KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering
)
from sklearn.mixture import GaussianMixture

# Dimensionality reduction
from sklearn.decomposition import PCA, FastICA
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Anomaly detection
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from scipy import stats

# UMAP import (with fallback if not available)
try:
    import umap.umap_ as umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    umap = None

from ..logging_manager import get_logger
from ..pipeline.base import (
    AnalysisPipelineBase, PipelineResult, CompositionMetadata, 
    StreamingConfig, PipelineState
)

logger = get_logger(__name__)

# Suppress specific warnings that are not critical for our use case
warnings.filterwarnings('ignore', category=RuntimeWarning, module='sklearn')
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')


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
            'algorithm': self.algorithm,
            'n_clusters': self.n_clusters,
            'labels': self.labels.tolist() if self.labels is not None else None,
            'cluster_centers': self.cluster_centers.tolist() if self.cluster_centers is not None else None,
            'inertia': self.inertia,
            'silhouette_avg': self.silhouette_avg,
            'silhouette_samples': self.silhouette_samples.tolist() if self.silhouette_samples is not None else None,
            'calinski_harabasz': self.calinski_harabasz,
            'davies_bouldin': self.davies_bouldin,
            'adjusted_rand_score': self.adjusted_rand_score,
            'normalized_mutual_info': self.normalized_mutual_info,
            'homogeneity_score': self.homogeneity_score,
            'completeness_score': self.completeness_score,
            'v_measure_score': self.v_measure_score,
            'algorithm_params': self.algorithm_params,
            'fit_time': self.fit_time,
            'convergence_info': self.convergence_info,
            'cluster_stats': self.cluster_stats
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
            'algorithm': self.algorithm,
            'original_dimensions': self.original_dimensions,
            'reduced_dimensions': self.reduced_dimensions,
            'transformed_data': self.transformed_data.tolist(),
            'explained_variance_ratio': self.explained_variance_ratio.tolist() if self.explained_variance_ratio is not None else None,
            'cumulative_variance_ratio': self.cumulative_variance_ratio.tolist() if self.cumulative_variance_ratio is not None else None,
            'singular_values': self.singular_values.tolist() if self.singular_values is not None else None,
            'components': self.components.tolist() if self.components is not None else None,
            'reconstruction_error': self.reconstruction_error,
            'kl_divergence': self.kl_divergence,
            'trustworthiness': self.trustworthiness,
            'continuity': self.continuity,
            'algorithm_params': self.algorithm_params,
            'fit_time': self.fit_time,
            'transform_time': self.transform_time,
            'convergence_info': self.convergence_info
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
            'algorithm': self.algorithm,
            'n_samples': self.n_samples,
            'n_anomalies': self.n_anomalies,
            'anomaly_labels': self.anomaly_labels.tolist(),
            'anomaly_scores': self.anomaly_scores.tolist() if self.anomaly_scores is not None else None,
            'threshold': self.threshold,
            'contamination': self.contamination,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'auc_roc': self.auc_roc,
            'score_statistics': self.score_statistics,
            'anomaly_statistics': self.anomaly_statistics,
            'algorithm_params': self.algorithm_params,
            'fit_time': self.fit_time,
            'prediction_time': self.prediction_time
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
            'evaluation_type': self.evaluation_type,
            'metrics': self.metrics,
            'detailed_results': self.detailed_results,
            'recommendations': self.recommendations,
            'quality_assessment': self.quality_assessment
        }


class ClusteringTransformer(BaseEstimator, TransformerMixin, AnalysisPipelineBase):
    """Advanced clustering transformer with multiple algorithms and automatic parameter selection."""
    
    def __init__(self, 
                 algorithm: str = 'kmeans',
                 n_clusters: Optional[int] = None,
                 auto_k_selection: bool = True,
                 k_range: Tuple[int, int] = (2, 10),
                 standardize: bool = True,
                 random_state: int = 42,
                 **algorithm_params):
        """
        Initialize clustering transformer.
        
        Parameters:
        -----------
        algorithm : str
            Clustering algorithm ('kmeans', 'hierarchical', 'dbscan', 'gmm', 'spectral')
        n_clusters : int, optional
            Number of clusters (auto-selected if None)
        auto_k_selection : bool
            Whether to automatically select optimal number of clusters
        k_range : tuple
            Range of k values to try for auto-selection
        standardize : bool
            Whether to standardize features before clustering
        random_state : int
            Random state for reproducibility
        **algorithm_params
            Additional parameters for specific algorithms
        """
        super().__init__()
        self.algorithm = algorithm
        self.n_clusters = n_clusters
        self.auto_k_selection = auto_k_selection
        self.k_range = k_range
        self.standardize = standardize
        self.random_state = random_state
        self.algorithm_params = algorithm_params
        
        # Initialize components
        self.scaler_ = None
        self.clusterer_ = None
        self.optimal_k_ = None
        self.k_scores_ = {}
        
    def fit(self, X, y=None):
        """Fit the clustering model to the data."""
        logger.info(f"Fitting clustering transformer with algorithm: {self.algorithm}")
        start_time = time.time()
        
        # Validate input
        X = check_array(X, accept_sparse=False)
        
        # Standardize if requested
        if self.standardize:
            self.scaler_ = StandardScaler()
            X_scaled = self.scaler_.fit_transform(X)
        else:
            X_scaled = X
            
        # Determine optimal number of clusters if needed
        if self.n_clusters is None and self.auto_k_selection:
            if self.algorithm in ['kmeans', 'hierarchical', 'gmm', 'spectral']:
                self.optimal_k_ = self._select_optimal_k(X_scaled)
            elif self.algorithm == 'dbscan':
                # DBSCAN doesn't need k, use default parameters
                self.optimal_k_ = None
            else:
                # Default fallback
                self.optimal_k_ = 3
        else:
            self.optimal_k_ = self.n_clusters
            
        # Initialize and fit clusterer
        self.clusterer_ = self._create_clusterer(self.optimal_k_)
        
        try:
            if hasattr(self.clusterer_, 'fit_predict'):
                self.labels_ = self.clusterer_.fit_predict(X_scaled)
            else:
                self.clusterer_.fit(X_scaled)
                self.labels_ = self.clusterer_.labels_
        except Exception as e:
            logger.error(f"Clustering failed: {e}")
            raise
            
        self.fit_time_ = time.time() - start_time
        logger.info(f"Clustering completed in {self.fit_time_:.3f} seconds")
        
        return self
        
    def transform(self, X):
        """Transform data by returning cluster labels."""
        check_is_fitted(self, 'clusterer_')
        
        # Scale if necessary
        if self.scaler_ is not None:
            X_scaled = self.scaler_.transform(X)
        else:
            X_scaled = X
            
        # Predict cluster labels
        if hasattr(self.clusterer_, 'predict'):
            return self.clusterer_.predict(X_scaled)
        else:
            # For algorithms without predict method, fit_predict on new data
            logger.warning(f"{self.algorithm} doesn't support prediction on new data")
            return np.full(X.shape[0], -1)
            
    def _create_clusterer(self, n_clusters):
        """Create clusterer based on algorithm choice."""
        base_params = {'random_state': self.random_state} if 'random_state' in self._get_valid_params() else {}
        params = {**base_params, **self.algorithm_params}
        
        if self.algorithm == 'kmeans':
            if n_clusters is not None:
                params['n_clusters'] = n_clusters
            return KMeans(**params)
        elif self.algorithm == 'hierarchical':
            if n_clusters is not None:
                params['n_clusters'] = n_clusters
            return AgglomerativeClustering(**params)
        elif self.algorithm == 'dbscan':
            # DBSCAN doesn't use n_clusters
            return DBSCAN(**params)
        elif self.algorithm == 'gmm':
            if n_clusters is not None:
                params['n_components'] = n_clusters
            return GaussianMixture(**params)
        elif self.algorithm == 'spectral':
            if n_clusters is not None:
                params['n_clusters'] = n_clusters
            return SpectralClustering(**params)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
            
    def _get_valid_params(self):
        """Get valid parameters for the chosen algorithm."""
        param_sets = {
            'kmeans': ['n_clusters', 'init', 'n_init', 'max_iter', 'tol', 'random_state'],
            'hierarchical': ['n_clusters', 'linkage', 'metric'],
            'dbscan': ['eps', 'min_samples', 'metric'],
            'gmm': ['n_components', 'covariance_type', 'max_iter', 'random_state'],
            'spectral': ['n_clusters', 'gamma', 'affinity', 'random_state']
        }
        return param_sets.get(self.algorithm, [])
        
    def _select_optimal_k(self, X):
        """Select optimal number of clusters using multiple metrics."""
        logger.info(f"Selecting optimal k in range {self.k_range}")
        
        k_min, k_max = self.k_range
        silhouette_scores = {}
        calinski_harabasz_scores = {}
        davies_bouldin_scores = {}
        
        for k in range(k_min, min(k_max + 1, X.shape[0])):
            try:
                clusterer = self._create_clusterer(k)
                if hasattr(clusterer, 'fit_predict'):
                    labels = clusterer.fit_predict(X)
                else:
                    clusterer.fit(X)
                    labels = clusterer.labels_
                    
                # Skip if only one cluster or no clusters found
                if len(set(labels)) < 2:
                    continue
                    
                # Calculate metrics
                sil_score = silhouette_score(X, labels)
                ch_score = calinski_harabasz_score(X, labels)
                db_score = davies_bouldin_score(X, labels)
                
                silhouette_scores[k] = sil_score
                calinski_harabasz_scores[k] = ch_score
                davies_bouldin_scores[k] = db_score
                
            except Exception as e:
                logger.warning(f"Failed to evaluate k={k}: {e}")
                continue
                
        self.k_scores_ = {
            'silhouette': silhouette_scores,
            'calinski_harabasz': calinski_harabasz_scores,
            'davies_bouldin': davies_bouldin_scores
        }
        
        if not silhouette_scores:
            logger.warning("No valid k values found, using default k=3")
            return 3
            
        # Select k with best silhouette score
        optimal_k = max(silhouette_scores.keys(), key=lambda k: silhouette_scores[k])
        logger.info(f"Selected optimal k={optimal_k} with silhouette score {silhouette_scores[optimal_k]:.3f}")
        
        return optimal_k
        
    def get_clustering_result(self, X, y_true=None) -> ClusteringResult:
        """Generate comprehensive clustering results."""
        check_is_fitted(self, 'clusterer_')
        
        # Scale data if necessary
        if self.scaler_ is not None:
            X_scaled = self.scaler_.transform(X)
        else:
            X_scaled = X
            
        # Get cluster centers if available
        cluster_centers = None
        if hasattr(self.clusterer_, 'cluster_centers_'):
            cluster_centers = self.clusterer_.cluster_centers_
        elif hasattr(self.clusterer_, 'means_'):  # GMM
            cluster_centers = self.clusterer_.means_
            
        # Calculate quality metrics
        n_clusters = len(set(self.labels_)) - (1 if -1 in self.labels_ else 0)
        
        silhouette_avg = None
        silhouette_samples = None
        calinski_harabasz = None
        davies_bouldin = None
        
        if n_clusters > 1:
            try:
                silhouette_avg = silhouette_score(X_scaled, self.labels_)
                silhouette_samples = None  # Can be computed but memory intensive
                calinski_harabasz = calinski_harabasz_score(X_scaled, self.labels_)
                davies_bouldin = davies_bouldin_score(X_scaled, self.labels_)
            except Exception as e:
                logger.warning(f"Failed to calculate some quality metrics: {e}")
                
        # External validation metrics (if ground truth available)
        adjusted_rand = None
        normalized_mutual_info = None
        homogeneity = None
        completeness = None
        v_measure = None
        
        if y_true is not None:
            try:
                adjusted_rand = adjusted_rand_score(y_true, self.labels_)
                normalized_mutual_info = normalized_mutual_info_score(y_true, self.labels_)
                homogeneity = homogeneity_score(y_true, self.labels_)
                completeness = completeness_score(y_true, self.labels_)
                v_measure = v_measure_score(y_true, self.labels_)
            except Exception as e:
                logger.warning(f"Failed to calculate external validation metrics: {e}")
                
        # Algorithm-specific info
        inertia = getattr(self.clusterer_, 'inertia_', None)
        
        # Cluster statistics
        unique_labels, counts = np.unique(self.labels_, return_counts=True)
        cluster_stats = {
            'cluster_sizes': dict(zip(unique_labels.tolist(), counts.tolist())),
            'largest_cluster_size': int(np.max(counts)),
            'smallest_cluster_size': int(np.min(counts)),
            'size_std': float(np.std(counts))
        }
        
        # Convergence information
        convergence_info = {}
        if hasattr(self.clusterer_, 'n_iter_'):
            convergence_info['n_iterations'] = self.clusterer_.n_iter_
        if hasattr(self.clusterer_, 'converged_'):
            convergence_info['converged'] = self.clusterer_.converged_
            
        return ClusteringResult(
            algorithm=self.algorithm,
            n_clusters=n_clusters,
            labels=self.labels_,
            cluster_centers=cluster_centers,
            inertia=inertia,
            silhouette_avg=silhouette_avg,
            silhouette_samples=silhouette_samples,
            calinski_harabasz=calinski_harabasz,
            davies_bouldin=davies_bouldin,
            adjusted_rand_score=adjusted_rand,
            normalized_mutual_info=normalized_mutual_info,
            homogeneity_score=homogeneity,
            completeness_score=completeness,
            v_measure_score=v_measure,
            algorithm_params=self.algorithm_params,
            fit_time=self.fit_time_,
            convergence_info=convergence_info,
            cluster_stats=cluster_stats
        )


class DimensionalityReductionTransformer(BaseEstimator, TransformerMixin, AnalysisPipelineBase):
    """Advanced dimensionality reduction transformer with multiple algorithms."""
    
    def __init__(self, 
                 algorithm: str = 'pca',
                 n_components: Optional[int] = None,
                 preserve_variance: float = 0.95,
                 standardize: bool = True,
                 random_state: int = 42,
                 **algorithm_params):
        """
        Initialize dimensionality reduction transformer.
        
        Parameters:
        -----------
        algorithm : str
            Reduction algorithm ('pca', 'tsne', 'umap', 'ica', 'lda')
        n_components : int, optional
            Number of components (auto-selected if None)
        preserve_variance : float
            Minimum variance to preserve (for PCA auto-selection)
        standardize : bool
            Whether to standardize features before reduction
        random_state : int
            Random state for reproducibility
        **algorithm_params
            Additional parameters for specific algorithms
        """
        super().__init__()
        self.algorithm = algorithm
        self.n_components = n_components
        self.preserve_variance = preserve_variance
        self.standardize = standardize
        self.random_state = random_state
        self.algorithm_params = algorithm_params
        
        # Initialize components
        self.scaler_ = None
        self.reducer_ = None
        self.optimal_components_ = None
        
    def fit(self, X, y=None):
        """Fit the dimensionality reduction model to the data."""
        logger.info(f"Fitting dimensionality reduction transformer with algorithm: {self.algorithm}")
        start_time = time.time()
        
        # Validate input
        X = check_array(X, accept_sparse=False)
        self.original_dimensions_ = X.shape[1]
        
        # Standardize if requested
        if self.standardize:
            self.scaler_ = StandardScaler()
            X_scaled = self.scaler_.fit_transform(X)
        else:
            X_scaled = X
            
        # Determine optimal number of components if needed
        if self.n_components is None:
            if self.algorithm == 'pca':
                self.optimal_components_ = self._select_optimal_components_pca(X_scaled)
            elif self.algorithm == 'lda' and y is not None:
                # For LDA, max components is min(n_features, n_classes-1)
                n_classes = len(np.unique(y))
                self.optimal_components_ = min(X_scaled.shape[1], n_classes - 1)
            else:
                # Default to 2 components for visualization algorithms
                self.optimal_components_ = min(2, X_scaled.shape[1])
        else:
            self.optimal_components_ = self.n_components
            
        # Create and fit reducer
        self.reducer_ = self._create_reducer(self.optimal_components_, y)
        
        try:
            if self.algorithm == 'lda' and y is None:
                raise ValueError("LDA requires target labels (y)")
            self.reducer_.fit(X_scaled, y)
        except Exception as e:
            logger.error(f"Dimensionality reduction fitting failed: {e}")
            raise
            
        self.fit_time_ = time.time() - start_time
        logger.info(f"Dimensionality reduction completed in {self.fit_time_:.3f} seconds")
        
        return self
        
    def transform(self, X):
        """Transform data to reduced dimensional space."""
        check_is_fitted(self, 'reducer_')
        
        start_time = time.time()
        
        # Scale if necessary
        if self.scaler_ is not None:
            X_scaled = self.scaler_.transform(X)
        else:
            X_scaled = X
            
        # Transform
        try:
            X_transformed = self.reducer_.transform(X_scaled)
        except Exception as e:
            logger.error(f"Dimensionality reduction transform failed: {e}")
            raise
            
        self.transform_time_ = time.time() - start_time
        return X_transformed
        
    def _create_reducer(self, n_components, y=None):
        """Create reducer based on algorithm choice."""
        base_params = {'random_state': self.random_state} if 'random_state' in self._get_valid_params() else {}
        params = {**base_params, **self.algorithm_params}
        
        if self.algorithm == 'pca':
            params['n_components'] = n_components
            return PCA(**params)
        elif self.algorithm == 'tsne':
            params['n_components'] = n_components
            return TSNE(**params)
        elif self.algorithm == 'umap':
            if not UMAP_AVAILABLE:
                raise ImportError("UMAP not available. Install with: pip install umap-learn")
            params['n_components'] = n_components
            return umap.UMAP(**params)
        elif self.algorithm == 'ica':
            params['n_components'] = n_components
            return FastICA(**params)
        elif self.algorithm == 'lda':
            params['n_components'] = n_components
            return LinearDiscriminantAnalysis(**params)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
            
    def _get_valid_params(self):
        """Get valid parameters for the chosen algorithm."""
        param_sets = {
            'pca': ['n_components', 'whiten', 'svd_solver', 'random_state'],
            'tsne': ['n_components', 'perplexity', 'learning_rate', 'n_iter', 'random_state'],
            'umap': ['n_components', 'n_neighbors', 'min_dist', 'metric', 'random_state'],
            'ica': ['n_components', 'algorithm', 'fun', 'max_iter', 'random_state'],
            'lda': ['n_components', 'solver', 'shrinkage']
        }
        return param_sets.get(self.algorithm, [])
        
    def _select_optimal_components_pca(self, X):
        """Select optimal number of components for PCA based on variance preservation."""
        logger.info(f"Selecting optimal components to preserve {self.preserve_variance:.1%} variance")
        
        # Fit PCA with all components first
        pca_full = PCA()
        pca_full.fit(X)
        
        # Find components needed to preserve desired variance
        cumsum_var = np.cumsum(pca_full.explained_variance_ratio_)
        optimal_components = np.argmax(cumsum_var >= self.preserve_variance) + 1
        optimal_components = min(optimal_components, X.shape[1], X.shape[0])
        
        preserved_variance = cumsum_var[optimal_components - 1]
        logger.info(f"Selected {optimal_components} components preserving {preserved_variance:.1%} variance")
        
        return optimal_components
        
    def get_reduction_result(self, X) -> DimensionalityReductionResult:
        """Generate comprehensive dimensionality reduction results."""
        check_is_fitted(self, 'reducer_')
        
        # Transform data
        start_time = time.time()
        X_transformed = self.transform(X)
        transform_time = time.time() - start_time
        
        # Extract algorithm-specific information
        explained_variance_ratio = getattr(self.reducer_, 'explained_variance_ratio_', None)
        cumulative_variance_ratio = None
        if explained_variance_ratio is not None:
            cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
            
        singular_values = getattr(self.reducer_, 'singular_values_', None)
        components = getattr(self.reducer_, 'components_', None)
        
        # Quality metrics
        reconstruction_error = None
        if hasattr(self.reducer_, 'inverse_transform'):
            try:
                X_reconstructed = self.reducer_.inverse_transform(X_transformed)
                reconstruction_error = np.mean(np.sum((X - X_reconstructed) ** 2, axis=1))
            except:
                pass
                
        kl_divergence = getattr(self.reducer_, 'kl_divergence_', None)
        
        # Trust and continuity metrics (simplified versions)
        trustworthiness = None
        continuity = None
        
        # Convergence information
        convergence_info = {}
        if hasattr(self.reducer_, 'n_iter_'):
            convergence_info['n_iterations'] = self.reducer_.n_iter_
        if hasattr(self.reducer_, 'kl_divergence_'):
            convergence_info['final_kl_divergence'] = float(self.reducer_.kl_divergence_)
            
        return DimensionalityReductionResult(
            algorithm=self.algorithm,
            original_dimensions=self.original_dimensions_,
            reduced_dimensions=self.optimal_components_,
            transformed_data=X_transformed,
            explained_variance_ratio=explained_variance_ratio,
            cumulative_variance_ratio=cumulative_variance_ratio,
            singular_values=singular_values,
            components=components,
            reconstruction_error=reconstruction_error,
            kl_divergence=kl_divergence,
            trustworthiness=trustworthiness,
            continuity=continuity,
            algorithm_params=self.algorithm_params,
            fit_time=self.fit_time_,
            transform_time=transform_time,
            convergence_info=convergence_info
        )


class AnomalyDetectionTransformer(BaseEstimator, TransformerMixin, AnalysisPipelineBase):
    """Advanced anomaly detection transformer with multiple algorithms."""
    
    def __init__(self, 
                 algorithm: str = 'isolation_forest',
                 contamination: Optional[float] = None,
                 standardize: bool = True,
                 random_state: int = 42,
                 **algorithm_params):
        """
        Initialize anomaly detection transformer.
        
        Parameters:
        -----------
        algorithm : str
            Detection algorithm ('isolation_forest', 'one_class_svm', 'lof', 'statistical')
        contamination : float, optional
            Expected proportion of outliers (auto-estimated if None)
        standardize : bool
            Whether to standardize features before detection
        random_state : int
            Random state for reproducibility
        **algorithm_params
            Additional parameters for specific algorithms
        """
        super().__init__()
        self.algorithm = algorithm
        self.contamination = contamination
        self.standardize = standardize
        self.random_state = random_state
        self.algorithm_params = algorithm_params
        
        # Initialize components
        self.scaler_ = None
        self.detector_ = None
        self.threshold_ = None
        
    def fit(self, X, y=None):
        """Fit the anomaly detection model to the data."""
        logger.info(f"Fitting anomaly detection transformer with algorithm: {self.algorithm}")
        start_time = time.time()
        
        # Validate input
        X = check_array(X, accept_sparse=False)
        
        # Standardize if requested
        if self.standardize and self.algorithm != 'statistical':
            self.scaler_ = StandardScaler()
            X_scaled = self.scaler_.fit_transform(X)
        else:
            X_scaled = X
            
        # Estimate contamination if not provided
        if self.contamination is None:
            if self.algorithm == 'statistical':
                self.contamination_ = 0.05  # 5% for statistical methods
            else:
                self.contamination_ = min(0.1, max(0.01, 100 / X.shape[0]))  # Adaptive
        else:
            self.contamination_ = self.contamination
            
        # Create and fit detector
        self.detector_ = self._create_detector()
        
        try:
            if self.algorithm == 'statistical':
                # Statistical methods don't need fitting
                pass
            else:
                self.detector_.fit(X_scaled)
        except Exception as e:
            logger.error(f"Anomaly detection fitting failed: {e}")
            raise
            
        self.fit_time_ = time.time() - start_time
        logger.info(f"Anomaly detection completed in {self.fit_time_:.3f} seconds")
        
        return self
        
    def transform(self, X):
        """Transform data by returning anomaly labels."""
        return self.predict(X)
        
    def predict(self, X):
        """Predict anomaly labels for data."""
        check_is_fitted(self, 'detector_')
        
        start_time = time.time()
        
        # Scale if necessary
        if self.scaler_ is not None:
            X_scaled = self.scaler_.transform(X)
        else:
            X_scaled = X
            
        # Predict anomalies
        try:
            if self.algorithm == 'statistical':
                labels = self._statistical_detection(X_scaled)
            else:
                labels = self.detector_.predict(X_scaled)
        except Exception as e:
            logger.error(f"Anomaly detection prediction failed: {e}")
            raise
            
        self.prediction_time_ = time.time() - start_time
        return labels
        
    def decision_function(self, X):
        """Compute anomaly scores for data."""
        check_is_fitted(self, 'detector_')
        
        # Scale if necessary
        if self.scaler_ is not None:
            X_scaled = self.scaler_.transform(X)
        else:
            X_scaled = X
            
        # Get anomaly scores
        if self.algorithm == 'statistical':
            return self._statistical_scores(X_scaled)
        elif hasattr(self.detector_, 'decision_function'):
            return self.detector_.decision_function(X_scaled)
        elif hasattr(self.detector_, 'score_samples'):
            return self.detector_.score_samples(X_scaled)
        else:
            return np.zeros(X.shape[0])  # Fallback
            
    def _create_detector(self):
        """Create detector based on algorithm choice."""
        base_params = {'random_state': self.random_state} if 'random_state' in self._get_valid_params() else {}
        params = {**base_params, **self.algorithm_params}
        
        if self.algorithm == 'isolation_forest':
            params['contamination'] = self.contamination_
            return IsolationForest(**params)
        elif self.algorithm == 'one_class_svm':
            # One-Class SVM doesn't use contamination directly
            return OneClassSVM(**params)
        elif self.algorithm == 'lof':
            params['contamination'] = self.contamination_
            return LocalOutlierFactor(**params)
        elif self.algorithm == 'statistical':
            # Statistical methods don't need a detector object
            return None
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
            
    def _get_valid_params(self):
        """Get valid parameters for the chosen algorithm."""
        param_sets = {
            'isolation_forest': ['contamination', 'n_estimators', 'max_samples', 'random_state'],
            'one_class_svm': ['kernel', 'gamma', 'nu'],
            'lof': ['contamination', 'n_neighbors', 'algorithm', 'metric'],
            'statistical': ['method', 'threshold_std']
        }
        return param_sets.get(self.algorithm, [])
        
    def _statistical_detection(self, X):
        """Perform statistical anomaly detection."""
        method = self.algorithm_params.get('method', 'zscore')
        threshold_std = self.algorithm_params.get('threshold_std', 3.0)
        
        if method == 'zscore':
            # Z-score method
            z_scores = np.abs(stats.zscore(X, axis=0))
            max_z_scores = np.max(z_scores, axis=1)
            labels = np.where(max_z_scores > threshold_std, -1, 1)
        elif method == 'iqr':
            # Interquartile range method
            Q1 = np.percentile(X, 25, axis=0)
            Q3 = np.percentile(X, 75, axis=0)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = np.any((X < lower_bound) | (X > upper_bound), axis=1)
            labels = np.where(outliers, -1, 1)
        else:
            raise ValueError(f"Unknown statistical method: {method}")
            
        return labels
        
    def _statistical_scores(self, X):
        """Compute statistical anomaly scores."""
        method = self.algorithm_params.get('method', 'zscore')
        
        if method == 'zscore':
            z_scores = np.abs(stats.zscore(X, axis=0))
            return np.max(z_scores, axis=1)
        elif method == 'iqr':
            Q1 = np.percentile(X, 25, axis=0)
            Q3 = np.percentile(X, 75, axis=0)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            lower_violations = np.maximum(0, lower_bound - X)
            upper_violations = np.maximum(0, X - upper_bound)
            return np.max(lower_violations + upper_violations, axis=1)
        else:
            return np.zeros(X.shape[0])
            
    def get_anomaly_result(self, X, y_true=None) -> AnomalyDetectionResult:
        """Generate comprehensive anomaly detection results."""
        check_is_fitted(self, 'detector_')
        
        # Get predictions and scores
        start_time = time.time()
        labels = self.predict(X)
        scores = self.decision_function(X)
        prediction_time = time.time() - start_time
        
        n_samples = len(labels)
        n_anomalies = np.sum(labels == -1)
        
        # Quality metrics (if ground truth available)
        precision = None
        recall = None
        f1_score = None
        auc_roc = None
        
        if y_true is not None:
            from sklearn.metrics import precision_score, recall_score, f1_score as f1, roc_auc_score
            try:
                # Convert to binary format (1 for normal, 0 for anomaly)
                y_pred_binary = (labels == 1).astype(int)
                y_true_binary = (y_true == 1).astype(int)
                
                precision = precision_score(y_true_binary, y_pred_binary)
                recall = recall_score(y_true_binary, y_pred_binary)
                f1_score = f1(y_true_binary, y_pred_binary)
                
                if len(np.unique(y_true_binary)) > 1:
                    auc_roc = roc_auc_score(y_true_binary, scores)
            except Exception as e:
                logger.warning(f"Failed to calculate quality metrics: {e}")
                
        # Score statistics
        score_statistics = {
            'mean_score': float(np.mean(scores)),
            'std_score': float(np.std(scores)),
            'min_score': float(np.min(scores)),
            'max_score': float(np.max(scores)),
            'median_score': float(np.median(scores))
        }
        
        # Anomaly statistics
        if n_anomalies > 0:
            anomaly_indices = np.where(labels == -1)[0]
            anomaly_scores_subset = scores[anomaly_indices]
            normal_scores_subset = scores[labels == 1]
            
            anomaly_statistics = {
                'anomaly_indices': anomaly_indices.tolist(),
                'mean_anomaly_score': float(np.mean(anomaly_scores_subset)),
                'mean_normal_score': float(np.mean(normal_scores_subset)) if len(normal_scores_subset) > 0 else None,
                'score_separation': float(np.mean(anomaly_scores_subset) - np.mean(normal_scores_subset)) if len(normal_scores_subset) > 0 else None
            }
        else:
            anomaly_statistics = {
                'anomaly_indices': [],
                'mean_anomaly_score': None,
                'mean_normal_score': float(np.mean(scores)),
                'score_separation': None
            }
            
        return AnomalyDetectionResult(
            algorithm=self.algorithm,
            n_samples=n_samples,
            n_anomalies=n_anomalies,
            anomaly_labels=labels,
            anomaly_scores=scores,
            threshold=self.threshold_,
            contamination=self.contamination_,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            auc_roc=auc_roc,
            score_statistics=score_statistics,
            anomaly_statistics=anomaly_statistics,
            algorithm_params=self.algorithm_params,
            fit_time=self.fit_time_,
            prediction_time=prediction_time
        )


class PatternEvaluationTransformer(BaseEstimator, TransformerMixin, AnalysisPipelineBase):
    """Pattern evaluation and quality assessment transformer."""
    
    def __init__(self, evaluation_type: str = 'clustering'):
        """
        Initialize pattern evaluation transformer.
        
        Parameters:
        -----------
        evaluation_type : str
            Type of evaluation ('clustering', 'dimensionality_reduction', 'anomaly_detection')
        """
        super().__init__()
        self.evaluation_type = evaluation_type
        
    def fit(self, X, y=None):
        """Fit is not needed for evaluation."""
        return self
        
    def transform(self, X):
        """Transform returns the original data."""
        return X
        
    def evaluate_clustering(self, X, labels, y_true=None, algorithm='unknown') -> PatternEvaluationResult:
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
                
                metrics.update({
                    'silhouette_score': silhouette_avg,
                    'calinski_harabasz_score': calinski_harabasz,
                    'davies_bouldin_score': davies_bouldin,
                    'n_clusters': n_clusters
                })
                
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
                    recommendations.append("Consider trying different number of clusters")
                if davies_bouldin > 2.0:
                    recommendations.append("High cluster overlap detected, try different algorithm")
                if n_clusters < 2:
                    recommendations.append("Insufficient clustering structure found")
                    
                # External validation (if ground truth available)
                if y_true is not None:
                    ari = adjusted_rand_score(y_true, labels)
                    nmi = normalized_mutual_info_score(y_true, labels)
                    metrics.update({'adjusted_rand_index': ari, 'normalized_mutual_info': nmi})
                    
            else:
                quality = "Poor"
                recommendations.append("No meaningful clusters found")
                
        except Exception as e:
            logger.error(f"Clustering evaluation failed: {e}")
            quality = "Error"
            recommendations.append(f"Evaluation failed: {e}")
            
        detailed_results['cluster_distribution'] = dict(zip(*np.unique(labels, return_counts=True)))
        
        return PatternEvaluationResult(
            evaluation_type='clustering',
            metrics=metrics,
            detailed_results=detailed_results,
            recommendations=recommendations,
            quality_assessment=quality
        )
        
    def evaluate_dimensionality_reduction(self, X, X_reduced, algorithm='unknown') -> PatternEvaluationResult:
        """Evaluate dimensionality reduction results."""
        metrics = {}
        detailed_results = {}
        recommendations = []
        
        try:
            original_dims = X.shape[1]
            reduced_dims = X_reduced.shape[1]
            reduction_ratio = reduced_dims / original_dims
            
            metrics.update({
                'original_dimensions': original_dims,
                'reduced_dimensions': reduced_dims,
                'reduction_ratio': reduction_ratio
            })
            
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
                recommendations.append("Consider increasing number of components for visualization")
            if reduced_dims > 10:
                recommendations.append("High number of dimensions may limit interpretability")
                
        except Exception as e:
            logger.error(f"Dimensionality reduction evaluation failed: {e}")
            quality = "Error"
            recommendations.append(f"Evaluation failed: {e}")
            
        return PatternEvaluationResult(
            evaluation_type='dimensionality_reduction',
            metrics=metrics,
            detailed_results=detailed_results,
            recommendations=recommendations,
            quality_assessment=quality
        )
        
    def evaluate_anomaly_detection(self, X, labels, scores, y_true=None, algorithm='unknown') -> PatternEvaluationResult:
        """Evaluate anomaly detection results."""
        metrics = {}
        detailed_results = {}
        recommendations = []
        
        try:
            n_samples = len(labels)
            n_anomalies = np.sum(labels == -1)
            contamination = n_anomalies / n_samples
            
            metrics.update({
                'n_samples': n_samples,
                'n_anomalies': n_anomalies,
                'contamination_rate': contamination
            })
            
            # Quality assessment
            if 0.01 <= contamination <= 0.1:
                quality = "Good"
            elif 0.001 <= contamination <= 0.2:
                quality = "Fair"
            else:
                quality = "Poor"
                
            # Recommendations
            if contamination < 0.001:
                recommendations.append("Very few anomalies detected, consider lowering threshold")
            if contamination > 0.2:
                recommendations.append("High anomaly rate detected, consider raising threshold")
            if n_anomalies == 0:
                recommendations.append("No anomalies detected, verify algorithm parameters")
                
            # External validation (if ground truth available)
            if y_true is not None:
                from sklearn.metrics import precision_score, recall_score, f1_score as f1
                
                y_pred_binary = (labels == 1).astype(int)
                y_true_binary = (y_true == 1).astype(int)
                
                precision = precision_score(y_true_binary, y_pred_binary)
                recall = recall_score(y_true_binary, y_pred_binary)
                f1_score = f1(y_true_binary, y_pred_binary)
                
                metrics.update({
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1_score
                })
                
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
            
        detailed_results['score_distribution'] = {
            'mean': float(np.mean(scores)),
            'std': float(np.std(scores)),
            'min': float(np.min(scores)),
            'max': float(np.max(scores))
        }
        
        return PatternEvaluationResult(
            evaluation_type='anomaly_detection',
            metrics=metrics,
            detailed_results=detailed_results,
            recommendations=recommendations,
            quality_assessment=quality
        )


# High-level convenience functions for MCP tool integration

def perform_clustering(X: np.ndarray, 
                      algorithm: str = 'kmeans',
                      n_clusters: Optional[int] = None,
                      y_true: Optional[np.ndarray] = None,
                      **kwargs) -> Dict[str, Any]:
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
            algorithm=algorithm,
            n_clusters=n_clusters,
            **kwargs
        )
        clusterer.fit(X)
        
        # Get comprehensive results
        result = clusterer.get_clustering_result(X, y_true)
        
        # Evaluate clustering quality
        evaluator = PatternEvaluationTransformer('clustering')
        evaluation = evaluator.evaluate_clustering(X, result.labels, y_true, algorithm)
        
        # Combine results
        output = result.to_dict()
        output['evaluation'] = evaluation.to_dict()
        
        logger.info(f"Clustering completed successfully with {result.n_clusters} clusters")
        return output
        
    except Exception as e:
        logger.error(f"Clustering analysis failed: {e}")
        raise


def reduce_dimensions(X: np.ndarray,
                     algorithm: str = 'pca',
                     n_components: Optional[int] = None,
                     y: Optional[np.ndarray] = None,
                     **kwargs) -> Dict[str, Any]:
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
            algorithm=algorithm,
            n_components=n_components,
            **kwargs
        )
        reducer.fit(X, y)
        
        # Get comprehensive results
        result = reducer.get_reduction_result(X)
        
        # Evaluate reduction quality
        evaluator = PatternEvaluationTransformer('dimensionality_reduction')
        evaluation = evaluator.evaluate_dimensionality_reduction(X, result.transformed_data, algorithm)
        
        # Combine results
        output = result.to_dict()
        output['evaluation'] = evaluation.to_dict()
        
        logger.info(f"Dimensionality reduction completed: {result.original_dimensions} -> {result.reduced_dimensions}")
        return output
        
    except Exception as e:
        logger.error(f"Dimensionality reduction analysis failed: {e}")
        raise


def detect_anomalies(X: np.ndarray,
                    algorithm: str = 'isolation_forest',
                    contamination: Optional[float] = None,
                    y_true: Optional[np.ndarray] = None,
                    **kwargs) -> Dict[str, Any]:
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
            algorithm=algorithm,
            contamination=contamination,
            **kwargs
        )
        detector.fit(X)
        
        # Get comprehensive results
        result = detector.get_anomaly_result(X, y_true)
        
        # Evaluate detection quality
        evaluator = PatternEvaluationTransformer('anomaly_detection')
        evaluation = evaluator.evaluate_anomaly_detection(
            X, result.anomaly_labels, result.anomaly_scores, y_true, algorithm
        )
        
        # Combine results
        output = result.to_dict()
        output['evaluation'] = evaluation.to_dict()
        
        logger.info(f"Anomaly detection completed: {result.n_anomalies}/{result.n_samples} anomalies detected")
        return output
        
    except Exception as e:
        logger.error(f"Anomaly detection analysis failed: {e}")
        raise


def evaluate_patterns(X: np.ndarray,
                     pattern_type: str,
                     results: Dict[str, Any],
                     y_true: Optional[np.ndarray] = None) -> Dict[str, Any]:
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
        
        if pattern_type == 'clustering':
            labels = np.array(results.get('labels', []))
            evaluation = evaluator.evaluate_clustering(X, labels, y_true)
        elif pattern_type == 'dimensionality_reduction':
            transformed_data = np.array(results.get('transformed_data', []))
            evaluation = evaluator.evaluate_dimensionality_reduction(X, transformed_data)
        elif pattern_type == 'anomaly_detection':
            labels = np.array(results.get('anomaly_labels', []))
            scores = np.array(results.get('anomaly_scores', []))
            evaluation = evaluator.evaluate_anomaly_detection(X, labels, scores, y_true)
        else:
            raise ValueError(f"Unknown pattern type: {pattern_type}")
            
        logger.info(f"Pattern evaluation completed: {evaluation.quality_assessment}")
        return evaluation.to_dict()
        
    except Exception as e:
        logger.error(f"Pattern evaluation failed: {e}")
        raise