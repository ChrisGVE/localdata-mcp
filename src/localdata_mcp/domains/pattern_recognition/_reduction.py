"""
Pattern Recognition - Dimensionality Reduction Transformer.

Advanced dimensionality reduction with multiple algorithms (PCA, t-SNE, UMAP, ICA, LDA)
and automatic component selection.
"""

import time
import warnings
from typing import Any, Dict, Optional

import numpy as np

# Dimensionality reduction
from sklearn.decomposition import PCA, FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_array, check_is_fitted

# UMAP import (with fallback if not available)
try:
    import umap.umap_ as umap

    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    umap = None

from ...logging_manager import get_logger
from ...pipeline.base import AnalysisPipelineBase
from ._results import DimensionalityReductionResult

logger = get_logger(__name__)

# Suppress specific warnings that are not critical for our use case
warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


class DimensionalityReductionTransformer(AnalysisPipelineBase):
    """Advanced dimensionality reduction transformer with multiple algorithms."""

    def __init__(
        self,
        algorithm: str = "pca",
        n_components: Optional[int] = None,
        preserve_variance: float = 0.95,
        standardize: bool = True,
        random_state: int = 42,
        **algorithm_params,
    ):
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
        super().__init__(analytical_intention=f"{algorithm} dimensionality reduction")
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

    def get_analysis_type(self) -> str:
        """Get the specific analysis type this pipeline performs."""
        return f"{self.algorithm}_dimensionality_reduction"

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
        return self.reducer_ if hasattr(self, "reducer_") else None, {}

    def fit(self, X, y=None):
        """Fit the dimensionality reduction model to the data."""
        logger.info(
            f"Fitting dimensionality reduction transformer with algorithm: {self.algorithm}"
        )
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
            if self.algorithm == "pca":
                self.optimal_components_ = self._select_optimal_components_pca(X_scaled)
            elif self.algorithm == "lda" and y is not None:
                # For LDA, max components is min(n_features, n_classes-1)
                n_classes = len(np.unique(y))
                self.optimal_components_ = min(X_scaled.shape[1], n_classes - 1)
            else:
                # Default to 2 components for visualization algorithms
                self.optimal_components_ = min(2, X_scaled.shape[1])
        else:
            # Clamp to valid range: at most min(n_samples, n_features)
            max_components = min(X_scaled.shape[0], X_scaled.shape[1])
            self.optimal_components_ = min(self.n_components, max_components)

        # Create and fit reducer
        self.reducer_ = self._create_reducer(self.optimal_components_, y)

        try:
            if self.algorithm == "lda" and y is None:
                raise ValueError("LDA requires target labels (y)")
            # t-SNE lacks a separate transform(); use fit_transform and cache
            if self.algorithm == "tsne":
                self.embedding_ = self.reducer_.fit_transform(X_scaled)
            else:
                self.reducer_.fit(X_scaled, y)
        except Exception as e:
            logger.error(f"Dimensionality reduction fitting failed: {e}")
            raise

        self.fit_time_ = time.time() - start_time
        logger.info(
            f"Dimensionality reduction completed in {self.fit_time_:.3f} seconds"
        )

        return self

    def transform(self, X):
        """Transform data to reduced dimensional space."""
        check_is_fitted(self, "reducer_")

        start_time = time.time()

        # Scale if necessary
        if self.scaler_ is not None:
            X_scaled = self.scaler_.transform(X)
        else:
            X_scaled = X

        # t-SNE lacks a separate transform(); return cached embedding or re-fit
        if self.algorithm == "tsne":
            if (
                hasattr(self, "embedding_")
                and X_scaled.shape[0] == self.embedding_.shape[0]
            ):
                self.transform_time_ = time.time() - start_time
                return self.embedding_
            # New data: re-fit (t-SNE is transductive)
            try:
                X_transformed = self.reducer_.fit_transform(X_scaled)
            except Exception as e:
                logger.error(f"Dimensionality reduction transform failed: {e}")
                raise
            self.transform_time_ = time.time() - start_time
            return X_transformed

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
        base_params = (
            {"random_state": self.random_state}
            if "random_state" in self._get_valid_params()
            else {}
        )
        params = {**base_params, **self.algorithm_params}

        if self.algorithm == "pca":
            params["n_components"] = n_components
            return PCA(**params)
        elif self.algorithm == "tsne":
            params["n_components"] = n_components
            return TSNE(**params)
        elif self.algorithm == "umap":
            if not UMAP_AVAILABLE:
                raise ImportError(
                    "UMAP not available. Install with: pip install umap-learn"
                )
            params["n_components"] = n_components
            return umap.UMAP(**params)
        elif self.algorithm == "ica":
            params["n_components"] = n_components
            return FastICA(**params)
        elif self.algorithm == "lda":
            params["n_components"] = n_components
            return LinearDiscriminantAnalysis(**params)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

    def _get_valid_params(self):
        """Get valid parameters for the chosen algorithm."""
        param_sets = {
            "pca": ["n_components", "whiten", "svd_solver", "random_state"],
            "tsne": [
                "n_components",
                "perplexity",
                "learning_rate",
                "n_iter",
                "random_state",
            ],
            "umap": [
                "n_components",
                "n_neighbors",
                "min_dist",
                "metric",
                "random_state",
            ],
            "ica": ["n_components", "algorithm", "fun", "max_iter", "random_state"],
            "lda": ["n_components", "solver", "shrinkage"],
        }
        return param_sets.get(self.algorithm, [])

    def _select_optimal_components_pca(self, X):
        """Select optimal number of components for PCA based on variance preservation."""
        logger.info(
            f"Selecting optimal components to preserve {self.preserve_variance:.1%} variance"
        )

        # Fit PCA with all components first
        pca_full = PCA()
        pca_full.fit(X)

        # Find components needed to preserve desired variance
        cumsum_var = np.cumsum(pca_full.explained_variance_ratio_)
        optimal_components = np.argmax(cumsum_var >= self.preserve_variance) + 1
        optimal_components = min(optimal_components, X.shape[1], X.shape[0])

        preserved_variance = cumsum_var[optimal_components - 1]
        logger.info(
            f"Selected {optimal_components} components preserving {preserved_variance:.1%} variance"
        )

        return optimal_components

    def get_reduction_result(self, X) -> DimensionalityReductionResult:
        """Generate comprehensive dimensionality reduction results."""
        check_is_fitted(self, "reducer_")

        # Transform data
        start_time = time.time()
        X_transformed = self.transform(X)
        transform_time = time.time() - start_time

        # Extract algorithm-specific information
        explained_variance_ratio = getattr(
            self.reducer_, "explained_variance_ratio_", None
        )
        cumulative_variance_ratio = None
        if explained_variance_ratio is not None:
            cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

        singular_values = getattr(self.reducer_, "singular_values_", None)
        components = getattr(self.reducer_, "components_", None)

        # Quality metrics
        reconstruction_error = None
        if hasattr(self.reducer_, "inverse_transform"):
            try:
                X_reconstructed = self.reducer_.inverse_transform(X_transformed)
                reconstruction_error = np.mean(
                    np.sum((X - X_reconstructed) ** 2, axis=1)
                )
            except:
                pass

        kl_divergence = getattr(self.reducer_, "kl_divergence_", None)

        # Trust and continuity metrics (simplified versions)
        trustworthiness = None
        continuity = None

        # Convergence information
        convergence_info = {}
        if hasattr(self.reducer_, "n_iter_"):
            convergence_info["n_iterations"] = self.reducer_.n_iter_
        if hasattr(self.reducer_, "kl_divergence_"):
            convergence_info["final_kl_divergence"] = float(
                self.reducer_.kl_divergence_
            )

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
            convergence_info=convergence_info,
        )
