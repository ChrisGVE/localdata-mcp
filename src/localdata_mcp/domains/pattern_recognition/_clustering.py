"""
Pattern Recognition - Clustering Transformer.

Advanced clustering with multiple algorithms (K-means, hierarchical, DBSCAN, GMM, spectral)
and automatic parameter selection.
"""

import warnings
from typing import Any, Dict, Optional, Tuple
import time

import numpy as np
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    homogeneity_score,
    completeness_score,
    v_measure_score,
)

# Clustering algorithms
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering
from sklearn.mixture import GaussianMixture

from ...logging_manager import get_logger
from ...pipeline.base import AnalysisPipelineBase
from ._results import ClusteringResult

logger = get_logger(__name__)

# Suppress specific warnings that are not critical for our use case
warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


class ClusteringTransformer(AnalysisPipelineBase):
    """Advanced clustering transformer with multiple algorithms and automatic parameter selection."""

    def __init__(
        self,
        algorithm: str = "kmeans",
        n_clusters: Optional[int] = None,
        auto_k_selection: bool = True,
        k_range: Tuple[int, int] = (2, 10),
        standardize: bool = True,
        random_state: int = 42,
        **algorithm_params,
    ):
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
        super().__init__(analytical_intention=f"{algorithm} clustering analysis")
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

    def get_analysis_type(self) -> str:
        """Get the specific analysis type this pipeline performs."""
        return f"{self.algorithm}_clustering"

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
        return self.labels_ if hasattr(self, "labels_") else None, {}

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
            if self.algorithm in ["kmeans", "hierarchical", "gmm", "spectral"]:
                self.optimal_k_ = self._select_optimal_k(X_scaled)
            elif self.algorithm == "dbscan":
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
            if hasattr(self.clusterer_, "fit_predict"):
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
        check_is_fitted(self, "clusterer_")

        # Scale if necessary
        if self.scaler_ is not None:
            X_scaled = self.scaler_.transform(X)
        else:
            X_scaled = X

        # Predict cluster labels
        if hasattr(self.clusterer_, "predict"):
            return self.clusterer_.predict(X_scaled)
        else:
            # For algorithms without predict method, fit_predict on new data
            logger.warning(f"{self.algorithm} doesn't support prediction on new data")
            return np.full(X.shape[0], -1)

    def _create_clusterer(self, n_clusters):
        """Create clusterer based on algorithm choice."""
        base_params = (
            {"random_state": self.random_state}
            if "random_state" in self._get_valid_params()
            else {}
        )
        params = {**base_params, **self.algorithm_params}

        if self.algorithm == "kmeans":
            if n_clusters is not None:
                params["n_clusters"] = n_clusters
            return KMeans(**params)
        elif self.algorithm == "hierarchical":
            if n_clusters is not None:
                params["n_clusters"] = n_clusters
            return AgglomerativeClustering(**params)
        elif self.algorithm == "dbscan":
            # DBSCAN doesn't use n_clusters
            return DBSCAN(**params)
        elif self.algorithm == "gmm":
            if n_clusters is not None:
                params["n_components"] = n_clusters
            return GaussianMixture(**params)
        elif self.algorithm == "spectral":
            if n_clusters is not None:
                params["n_clusters"] = n_clusters
            return SpectralClustering(**params)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

    def _get_valid_params(self):
        """Get valid parameters for the chosen algorithm."""
        param_sets = {
            "kmeans": [
                "n_clusters",
                "init",
                "n_init",
                "max_iter",
                "tol",
                "random_state",
            ],
            "hierarchical": ["n_clusters", "linkage", "metric"],
            "dbscan": ["eps", "min_samples", "metric"],
            "gmm": ["n_components", "covariance_type", "max_iter", "random_state"],
            "spectral": ["n_clusters", "gamma", "affinity", "random_state"],
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
                if hasattr(clusterer, "fit_predict"):
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
            "silhouette": silhouette_scores,
            "calinski_harabasz": calinski_harabasz_scores,
            "davies_bouldin": davies_bouldin_scores,
        }

        if not silhouette_scores:
            logger.warning("No valid k values found, using default k=3")
            return 3

        # Select k with best silhouette score
        optimal_k = max(silhouette_scores.keys(), key=lambda k: silhouette_scores[k])
        logger.info(
            f"Selected optimal k={optimal_k} with silhouette score {silhouette_scores[optimal_k]:.3f}"
        )

        return optimal_k

    def get_clustering_result(self, X, y_true=None) -> ClusteringResult:
        """Generate comprehensive clustering results."""
        check_is_fitted(self, "clusterer_")

        # Scale data if necessary
        if self.scaler_ is not None:
            X_scaled = self.scaler_.transform(X)
        else:
            X_scaled = X

        # Get cluster centers if available
        cluster_centers = None
        if hasattr(self.clusterer_, "cluster_centers_"):
            cluster_centers = self.clusterer_.cluster_centers_
        elif hasattr(self.clusterer_, "means_"):  # GMM
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
                normalized_mutual_info = normalized_mutual_info_score(
                    y_true, self.labels_
                )
                homogeneity = homogeneity_score(y_true, self.labels_)
                completeness = completeness_score(y_true, self.labels_)
                v_measure = v_measure_score(y_true, self.labels_)
            except Exception as e:
                logger.warning(f"Failed to calculate external validation metrics: {e}")

        # Algorithm-specific info
        inertia = getattr(self.clusterer_, "inertia_", None)

        # Cluster statistics
        unique_labels, counts = np.unique(self.labels_, return_counts=True)
        cluster_stats = {
            "cluster_sizes": dict(zip(unique_labels.tolist(), counts.tolist())),
            "largest_cluster_size": int(np.max(counts)),
            "smallest_cluster_size": int(np.min(counts)),
            "size_std": float(np.std(counts)),
        }

        # Convergence information
        convergence_info = {}
        if hasattr(self.clusterer_, "n_iter_"):
            convergence_info["n_iterations"] = self.clusterer_.n_iter_
        if hasattr(self.clusterer_, "converged_"):
            convergence_info["converged"] = self.clusterer_.converged_

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
            cluster_stats=cluster_stats,
        )
