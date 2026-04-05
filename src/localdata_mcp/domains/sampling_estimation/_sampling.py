"""
Sampling & Estimation Domain - SamplingTransformer.

sklearn-compatible transformer for comprehensive sampling techniques including
simple random, stratified, cluster, systematic, and weighted sampling.
"""

from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.cluster import KMeans

from ._results import SamplingResult


class SamplingTransformer(BaseEstimator, TransformerMixin):
    """
    sklearn-compatible transformer for comprehensive sampling techniques.

    Implements various sampling methods including simple random sampling,
    stratified sampling, cluster sampling, systematic sampling, and weighted sampling.

    Parameters:
    -----------
    sampling_method : str, default='simple_random'
        Sampling method: 'simple_random', 'stratified', 'cluster', 'systematic', 'weighted'
    sample_size : int or float, default=0.1
        Sample size as absolute number (int) or fraction (float)
    random_state : int, default=None
        Random seed for reproducibility
    stratify_column : str, default=None
        Column name for stratified sampling
    cluster_column : str, default=None
        Column name for cluster sampling
    weights_column : str, default=None
        Column name containing sampling weights
    replacement : bool, default=False
        Whether to sample with replacement

    Attributes:
    -----------
    sampling_result_ : SamplingResult
        Results of the sampling operation
    sample_indices_ : np.ndarray
        Indices of selected samples
    """

    def __init__(
        self,
        sampling_method: str = "simple_random",
        sample_size: Union[int, float] = 0.1,
        random_state: Optional[int] = None,
        stratify_column: Optional[str] = None,
        cluster_column: Optional[str] = None,
        weights_column: Optional[str] = None,
        replacement: bool = False,
    ):
        self.sampling_method = sampling_method
        self.sample_size = sample_size
        self.random_state = random_state
        self.stratify_column = stratify_column
        self.cluster_column = cluster_column
        self.weights_column = weights_column
        self.replacement = replacement
        self._validate_parameters()

    def fit(self, X, y=None):
        """Fit the transformer (no-op for sampling)."""
        self._validate_parameters()
        self.is_fitted_ = True
        return self

    def transform(self, X):
        """Generate sample from the input data."""
        check_is_fitted(self)

        if isinstance(X, pd.DataFrame):
            data = X
        else:
            data = pd.DataFrame(X)

        # Determine actual sample size
        if isinstance(self.sample_size, float):
            actual_sample_size = int(len(data) * self.sample_size)
        else:
            actual_sample_size = min(self.sample_size, len(data))

        # Generate sample based on method
        if self.sampling_method == "simple_random":
            self._simple_random_sample(data, actual_sample_size)
        elif self.sampling_method == "stratified":
            self._stratified_sample(data, actual_sample_size)
        elif self.sampling_method == "cluster":
            self._cluster_sample(data, actual_sample_size)
        elif self.sampling_method == "systematic":
            self._systematic_sample(data, actual_sample_size)
        elif self.sampling_method == "weighted":
            self._weighted_sample(data, actual_sample_size)
        else:
            raise ValueError(f"Unknown sampling method: {self.sampling_method}")

        # Calculate quality metrics
        self._calculate_quality_metrics(data)

        return self.sampling_result_.sample_data

    def _validate_parameters(self):
        """Validate input parameters."""
        valid_methods = [
            "simple_random",
            "stratified",
            "cluster",
            "systematic",
            "weighted",
        ]
        if self.sampling_method not in valid_methods:
            raise ValueError(f"sampling_method must be one of {valid_methods}")

        if isinstance(self.sample_size, float) and not 0 < self.sample_size <= 1:
            raise ValueError("sample_size as float must be between 0 and 1")
        elif isinstance(self.sample_size, int) and self.sample_size <= 0:
            raise ValueError("sample_size as int must be positive")

    def _simple_random_sample(self, data: pd.DataFrame, sample_size: int):
        """Perform simple random sampling."""
        np.random.seed(self.random_state)

        if self.replacement:
            self.sample_indices_ = np.random.choice(
                len(data), size=sample_size, replace=True
            )
        else:
            self.sample_indices_ = np.random.choice(
                len(data), size=sample_size, replace=False
            )

        sample_data = data.iloc[self.sample_indices_].copy()

        self.sampling_result_ = SamplingResult(
            sampling_method="simple_random",
            sample_size=sample_size,
            population_size=len(data),
            sample_data=sample_data,
            sample_indices=self.sample_indices_,
            sampling_params={
                "replacement": self.replacement,
                "random_state": self.random_state,
            },
        )

    def _stratified_sample(self, data: pd.DataFrame, sample_size: int):
        """Perform stratified sampling."""
        if self.stratify_column is None:
            raise ValueError(
                "stratify_column must be specified for stratified sampling"
            )

        if self.stratify_column not in data.columns:
            raise ValueError(f"Column '{self.stratify_column}' not found in data")

        # Calculate strata proportions
        strata_counts = data[self.stratify_column].value_counts()
        strata_proportions = strata_counts / len(data)

        # Allocate sample sizes to strata
        strata_samples = {}
        total_allocated = 0

        for stratum, proportion in strata_proportions.items():
            stratum_sample_size = max(1, int(sample_size * proportion))
            strata_samples[stratum] = stratum_sample_size
            total_allocated += stratum_sample_size

        # Adjust if needed to match exact sample size
        if total_allocated != sample_size:
            # Simple adjustment: modify the largest stratum
            largest_stratum = max(
                strata_samples.keys(), key=lambda x: strata_samples[x]
            )
            strata_samples[largest_stratum] += sample_size - total_allocated

        # Sample from each stratum
        sample_indices = []
        strata_info = {}

        np.random.seed(self.random_state)

        for stratum, stratum_sample_size in strata_samples.items():
            stratum_data = data[data[self.stratify_column] == stratum]

            if len(stratum_data) == 0:
                continue

            # Ensure we don't sample more than available
            actual_stratum_size = min(stratum_sample_size, len(stratum_data))

            if self.replacement:
                stratum_indices = np.random.choice(
                    stratum_data.index, size=actual_stratum_size, replace=True
                )
            else:
                stratum_indices = np.random.choice(
                    stratum_data.index, size=actual_stratum_size, replace=False
                )

            sample_indices.extend(stratum_indices)

            strata_info[str(stratum)] = {
                "population_size": len(stratum_data),
                "sample_size": actual_stratum_size,
                "proportion_in_population": len(stratum_data) / len(data),
                "proportion_in_sample": actual_stratum_size / sample_size,
            }

        self.sample_indices_ = np.array(sample_indices)
        sample_data = data.loc[self.sample_indices_].copy()

        self.sampling_result_ = SamplingResult(
            sampling_method="stratified",
            sample_size=len(sample_indices),
            population_size=len(data),
            sample_data=sample_data,
            sample_indices=self.sample_indices_,
            sampling_params={
                "stratify_column": self.stratify_column,
                "replacement": self.replacement,
                "random_state": self.random_state,
            },
            strata_info=strata_info,
        )

    def _cluster_sample(self, data: pd.DataFrame, sample_size: int):
        """Perform cluster sampling."""
        if self.cluster_column is None:
            # Create clusters using K-means if no cluster column specified
            numeric_data = data.select_dtypes(include=[np.number])
            if len(numeric_data.columns) == 0:
                raise ValueError(
                    "No numeric columns available for automatic clustering"
                )

            # Determine number of clusters (heuristic: sqrt of sample size)
            n_clusters = max(2, int(np.sqrt(sample_size)))

            kmeans = KMeans(
                n_clusters=n_clusters, random_state=self.random_state, n_init=10
            )
            cluster_labels = kmeans.fit_predict(
                numeric_data.fillna(numeric_data.mean())
            )

        else:
            if self.cluster_column not in data.columns:
                raise ValueError(f"Column '{self.cluster_column}' not found in data")
            cluster_labels = data[self.cluster_column].values

        # Get unique clusters
        unique_clusters = np.unique(cluster_labels)

        # Determine number of clusters to sample
        n_clusters_to_sample = max(1, min(len(unique_clusters), sample_size // 2))

        np.random.seed(self.random_state)
        selected_clusters = np.random.choice(
            unique_clusters, size=n_clusters_to_sample, replace=False
        )

        # Sample from selected clusters
        sample_indices = []
        cluster_info = {}

        for cluster in selected_clusters:
            cluster_mask = cluster_labels == cluster
            cluster_indices = np.where(cluster_mask)[0]

            # Determine how many to sample from this cluster
            cluster_sample_size = sample_size // n_clusters_to_sample
            if cluster == selected_clusters[-1]:  # Last cluster gets remainder
                cluster_sample_size = sample_size - len(sample_indices)

            cluster_sample_size = min(cluster_sample_size, len(cluster_indices))

            if cluster_sample_size > 0:
                if self.replacement:
                    sampled_indices = np.random.choice(
                        cluster_indices, size=cluster_sample_size, replace=True
                    )
                else:
                    sampled_indices = np.random.choice(
                        cluster_indices, size=cluster_sample_size, replace=False
                    )

                sample_indices.extend(sampled_indices)

                cluster_info[str(cluster)] = {
                    "population_size": len(cluster_indices),
                    "sample_size": cluster_sample_size,
                }

        self.sample_indices_ = np.array(sample_indices)
        sample_data = data.iloc[self.sample_indices_].copy()

        self.sampling_result_ = SamplingResult(
            sampling_method="cluster",
            sample_size=len(sample_indices),
            population_size=len(data),
            sample_data=sample_data,
            sample_indices=self.sample_indices_,
            sampling_params={
                "cluster_column": self.cluster_column,
                "n_clusters_selected": n_clusters_to_sample,
                "replacement": self.replacement,
                "random_state": self.random_state,
            },
            cluster_info=cluster_info,
        )

    def _systematic_sample(self, data: pd.DataFrame, sample_size: int):
        """Perform systematic sampling."""
        if sample_size >= len(data):
            self.sample_indices_ = np.arange(len(data))
        else:
            # Calculate sampling interval
            interval = len(data) // sample_size

            # Random starting point
            np.random.seed(self.random_state)
            start = np.random.randint(0, interval)

            # Generate systematic sample indices
            self.sample_indices_ = np.arange(start, len(data), interval)[:sample_size]

        sample_data = data.iloc[self.sample_indices_].copy()

        self.sampling_result_ = SamplingResult(
            sampling_method="systematic",
            sample_size=len(self.sample_indices_),
            population_size=len(data),
            sample_data=sample_data,
            sample_indices=self.sample_indices_,
            sampling_params={
                "sampling_interval": interval if sample_size < len(data) else 1,
                "starting_point": start if sample_size < len(data) else 0,
                "random_state": self.random_state,
            },
        )

    def _weighted_sample(self, data: pd.DataFrame, sample_size: int):
        """Perform weighted sampling."""
        if self.weights_column is None:
            raise ValueError("weights_column must be specified for weighted sampling")

        if self.weights_column not in data.columns:
            raise ValueError(f"Column '{self.weights_column}' not found in data")

        weights = data[self.weights_column].values

        # Normalize weights
        weights = weights / np.sum(weights)

        np.random.seed(self.random_state)
        self.sample_indices_ = np.random.choice(
            len(data), size=sample_size, replace=self.replacement, p=weights
        )

        sample_data = data.iloc[self.sample_indices_].copy()

        self.sampling_result_ = SamplingResult(
            sampling_method="weighted",
            sample_size=sample_size,
            population_size=len(data),
            sample_data=sample_data,
            sample_indices=self.sample_indices_,
            sampling_params={
                "weights_column": self.weights_column,
                "replacement": self.replacement,
                "random_state": self.random_state,
            },
        )

    def _calculate_quality_metrics(self, original_data: pd.DataFrame):
        """Calculate sampling quality metrics."""
        if self.sampling_result_.sample_data is None:
            return

        sample_data = self.sampling_result_.sample_data

        # Calculate representativeness for numeric columns
        numeric_cols = original_data.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) > 0:
            # Compare means and standard deviations
            mean_diffs = []
            std_ratios = []

            for col in numeric_cols:
                if col in sample_data.columns:
                    pop_mean = original_data[col].mean()
                    sample_mean = sample_data[col].mean()

                    pop_std = original_data[col].std()
                    sample_std = sample_data[col].std()

                    if not np.isnan(pop_mean) and not np.isnan(sample_mean):
                        mean_diffs.append(
                            abs(pop_mean - sample_mean) / pop_std if pop_std > 0 else 0
                        )

                    if (
                        not np.isnan(pop_std)
                        and not np.isnan(sample_std)
                        and pop_std > 0
                    ):
                        std_ratios.append(sample_std / pop_std)

            if mean_diffs:
                # Representativeness score (lower is better, convert to higher is better)
                avg_mean_diff = np.mean(mean_diffs)
                self.sampling_result_.representativeness_score = max(
                    0, 1 - avg_mean_diff
                )

                self.sampling_result_.coverage_metrics = {
                    "mean_absolute_difference": avg_mean_diff,
                    "std_ratio_mean": np.mean(std_ratios) if std_ratios else 1.0,
                    "std_ratio_std": np.std(std_ratios) if len(std_ratios) > 1 else 0.0,
                }
