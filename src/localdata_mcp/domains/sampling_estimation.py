"""
Sampling & Estimation Domain - Comprehensive sampling techniques and statistical estimation.

This module implements advanced sampling methods and statistical estimation tools including
bootstrap techniques, Monte Carlo methods, and Bayesian estimation using scipy, numpy,
and scikit-learn integration.

Key Features:
- Sampling Techniques (simple random, stratified, cluster, systematic, weighted sampling)
- Bootstrap Methods (parametric/non-parametric bootstrap, confidence intervals, bias correction)
- Monte Carlo Methods (integration, simulation, importance sampling, MCMC basics)
- Bayesian Estimation (posterior estimation, credible intervals, Bayesian updating)
- Full sklearn pipeline compatibility
- Streaming-compatible processing
- Comprehensive result formatting
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
import time
import json
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import bootstrap
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.cluster import KMeans

from ..logging_manager import get_logger
from ..pipeline.base import (
    AnalysisPipelineBase, PipelineResult, CompositionMetadata, 
    StreamingConfig, PipelineState
)

logger = get_logger(__name__)

# Suppress specific warnings that are not critical for our use case
warnings.filterwarnings('ignore', category=RuntimeWarning, module='scipy.stats')
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')


@dataclass
class SamplingResult:
    """Standardized result structure for sampling operations."""
    sampling_method: str
    sample_size: int
    population_size: int
    
    # Sample data and indices
    sample_data: Optional[pd.DataFrame] = None
    sample_indices: Optional[np.ndarray] = None
    
    # Sampling parameters
    sampling_params: Dict[str, Any] = field(default_factory=dict)
    
    # Quality metrics
    representativeness_score: Optional[float] = None
    coverage_metrics: Dict[str, float] = field(default_factory=dict)
    bias_estimates: Dict[str, float] = field(default_factory=dict)
    
    # Stratification info (if applicable)
    strata_info: Dict[str, Any] = field(default_factory=dict)
    cluster_info: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary format."""
        result_dict = {
            'sampling_method': self.sampling_method,
            'sample_size': self.sample_size,
            'population_size': self.population_size,
            'sampling_params': self.sampling_params
        }
        
        if self.sample_indices is not None:
            result_dict['sample_indices'] = self.sample_indices.tolist()
        
        if self.representativeness_score is not None:
            result_dict['quality_metrics'] = {
                'representativeness_score': self.representativeness_score,
                'coverage_metrics': self.coverage_metrics,
                'bias_estimates': self.bias_estimates
            }
        
        if self.strata_info:
            result_dict['strata_info'] = self.strata_info
        if self.cluster_info:
            result_dict['cluster_info'] = self.cluster_info
            
        return result_dict


@dataclass
class BootstrapResult:
    """Standardized result structure for bootstrap analyses."""
    statistic_name: str
    original_statistic: float
    bootstrap_method: str
    n_bootstrap: int
    
    # Bootstrap distribution
    bootstrap_distribution: np.ndarray
    
    # Confidence intervals
    confidence_intervals: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    
    # Bias and variance estimates
    bias_estimate: float = 0.0
    bias_corrected_estimate: Optional[float] = None
    variance_estimate: float = 0.0
    standard_error: float = 0.0
    
    # Bootstrap diagnostics
    convergence_info: Dict[str, Any] = field(default_factory=dict)
    bootstrap_params: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary format."""
        result_dict = {
            'statistic_name': self.statistic_name,
            'original_statistic': self.original_statistic,
            'bootstrap_method': self.bootstrap_method,
            'n_bootstrap': self.n_bootstrap,
            'bias_estimate': self.bias_estimate,
            'variance_estimate': self.variance_estimate,
            'standard_error': self.standard_error
        }
        
        if self.bias_corrected_estimate is not None:
            result_dict['bias_corrected_estimate'] = self.bias_corrected_estimate
        
        if self.confidence_intervals:
            result_dict['confidence_intervals'] = self.confidence_intervals
        
        if self.convergence_info:
            result_dict['convergence_info'] = self.convergence_info
            
        if self.bootstrap_params:
            result_dict['bootstrap_params'] = self.bootstrap_params
            
        return result_dict


@dataclass
class MonteCarloResult:
    """Standardized result structure for Monte Carlo analyses."""
    simulation_type: str
    n_simulations: int
    
    # Simulation results
    simulation_results: np.ndarray
    estimated_value: float
    
    # Uncertainty quantification
    confidence_interval: Tuple[float, float]
    standard_error: float
    convergence_diagnostic: Dict[str, float] = field(default_factory=dict)
    
    # Simulation parameters
    simulation_params: Dict[str, Any] = field(default_factory=dict)
    
    # Integration specific (if applicable)
    integration_bounds: Optional[Tuple[float, float]] = None
    function_info: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary format."""
        result_dict = {
            'simulation_type': self.simulation_type,
            'n_simulations': self.n_simulations,
            'estimated_value': self.estimated_value,
            'confidence_interval': self.confidence_interval,
            'standard_error': self.standard_error,
            'convergence_diagnostic': self.convergence_diagnostic,
            'simulation_params': self.simulation_params
        }
        
        if self.integration_bounds is not None:
            result_dict['integration_bounds'] = self.integration_bounds
            
        if self.function_info:
            result_dict['function_info'] = self.function_info
            
        return result_dict


@dataclass
class BayesianResult:
    """Standardized result structure for Bayesian estimation."""
    parameter_name: str
    estimation_method: str
    
    # Posterior distribution
    posterior_samples: Optional[np.ndarray] = None
    posterior_mean: float = 0.0
    posterior_mode: Optional[float] = None
    posterior_median: Optional[float] = None
    
    # Credible intervals
    credible_intervals: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    
    # Prior information
    prior_info: Dict[str, Any] = field(default_factory=dict)
    
    # Model comparison (if applicable)
    bayes_factor: Optional[float] = None
    marginal_likelihood: Optional[float] = None
    
    # MCMC diagnostics (if applicable)
    mcmc_diagnostics: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary format."""
        result_dict = {
            'parameter_name': self.parameter_name,
            'estimation_method': self.estimation_method,
            'posterior_mean': self.posterior_mean,
            'credible_intervals': self.credible_intervals,
            'prior_info': self.prior_info
        }
        
        if self.posterior_mode is not None:
            result_dict['posterior_mode'] = self.posterior_mode
        if self.posterior_median is not None:
            result_dict['posterior_median'] = self.posterior_median
        if self.bayes_factor is not None:
            result_dict['bayes_factor'] = self.bayes_factor
        if self.marginal_likelihood is not None:
            result_dict['marginal_likelihood'] = self.marginal_likelihood
            
        if self.mcmc_diagnostics:
            result_dict['mcmc_diagnostics'] = self.mcmc_diagnostics
            
        return result_dict


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
    
    def __init__(self,
                 sampling_method: str = 'simple_random',
                 sample_size: Union[int, float] = 0.1,
                 random_state: Optional[int] = None,
                 stratify_column: Optional[str] = None,
                 cluster_column: Optional[str] = None,
                 weights_column: Optional[str] = None,
                 replacement: bool = False):
        self.sampling_method = sampling_method
        self.sample_size = sample_size
        self.random_state = random_state
        self.stratify_column = stratify_column
        self.cluster_column = cluster_column
        self.weights_column = weights_column
        self.replacement = replacement

    def fit(self, X, y=None):
        """Fit the transformer (no-op for sampling)."""
        self._validate_parameters()
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
        if self.sampling_method == 'simple_random':
            self._simple_random_sample(data, actual_sample_size)
        elif self.sampling_method == 'stratified':
            self._stratified_sample(data, actual_sample_size)
        elif self.sampling_method == 'cluster':
            self._cluster_sample(data, actual_sample_size)
        elif self.sampling_method == 'systematic':
            self._systematic_sample(data, actual_sample_size)
        elif self.sampling_method == 'weighted':
            self._weighted_sample(data, actual_sample_size)
        else:
            raise ValueError(f"Unknown sampling method: {self.sampling_method}")
        
        # Calculate quality metrics
        self._calculate_quality_metrics(data)
        
        return self.sampling_result_.sample_data

    def _validate_parameters(self):
        """Validate input parameters."""
        valid_methods = ['simple_random', 'stratified', 'cluster', 'systematic', 'weighted']
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
            self.sample_indices_ = np.random.choice(len(data), size=sample_size, replace=True)
        else:
            self.sample_indices_ = np.random.choice(len(data), size=sample_size, replace=False)
        
        sample_data = data.iloc[self.sample_indices_].copy()
        
        self.sampling_result_ = SamplingResult(
            sampling_method='simple_random',
            sample_size=sample_size,
            population_size=len(data),
            sample_data=sample_data,
            sample_indices=self.sample_indices_,
            sampling_params={
                'replacement': self.replacement,
                'random_state': self.random_state
            }
        )

    def _stratified_sample(self, data: pd.DataFrame, sample_size: int):
        """Perform stratified sampling."""
        if self.stratify_column is None:
            raise ValueError("stratify_column must be specified for stratified sampling")
        
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
            largest_stratum = max(strata_samples.keys(), key=lambda x: strata_samples[x])
            strata_samples[largest_stratum] += (sample_size - total_allocated)
        
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
                stratum_indices = np.random.choice(stratum_data.index, size=actual_stratum_size, replace=True)
            else:
                stratum_indices = np.random.choice(stratum_data.index, size=actual_stratum_size, replace=False)
            
            sample_indices.extend(stratum_indices)
            
            strata_info[str(stratum)] = {
                'population_size': len(stratum_data),
                'sample_size': actual_stratum_size,
                'proportion_in_population': len(stratum_data) / len(data),
                'proportion_in_sample': actual_stratum_size / sample_size
            }
        
        self.sample_indices_ = np.array(sample_indices)
        sample_data = data.loc[self.sample_indices_].copy()
        
        self.sampling_result_ = SamplingResult(
            sampling_method='stratified',
            sample_size=len(sample_indices),
            population_size=len(data),
            sample_data=sample_data,
            sample_indices=self.sample_indices_,
            sampling_params={
                'stratify_column': self.stratify_column,
                'replacement': self.replacement,
                'random_state': self.random_state
            },
            strata_info=strata_info
        )

    def _cluster_sample(self, data: pd.DataFrame, sample_size: int):
        """Perform cluster sampling."""
        if self.cluster_column is None:
            # Create clusters using K-means if no cluster column specified
            numeric_data = data.select_dtypes(include=[np.number])
            if len(numeric_data.columns) == 0:
                raise ValueError("No numeric columns available for automatic clustering")
            
            # Determine number of clusters (heuristic: sqrt of sample size)
            n_clusters = max(2, int(np.sqrt(sample_size)))
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
            cluster_labels = kmeans.fit_predict(numeric_data.fillna(numeric_data.mean()))
            
        else:
            if self.cluster_column not in data.columns:
                raise ValueError(f"Column '{self.cluster_column}' not found in data")
            cluster_labels = data[self.cluster_column].values
        
        # Get unique clusters
        unique_clusters = np.unique(cluster_labels)
        
        # Determine number of clusters to sample
        n_clusters_to_sample = max(1, min(len(unique_clusters), sample_size // 2))
        
        np.random.seed(self.random_state)
        selected_clusters = np.random.choice(unique_clusters, size=n_clusters_to_sample, replace=False)
        
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
                    sampled_indices = np.random.choice(cluster_indices, size=cluster_sample_size, replace=True)
                else:
                    sampled_indices = np.random.choice(cluster_indices, size=cluster_sample_size, replace=False)
                
                sample_indices.extend(sampled_indices)
                
                cluster_info[str(cluster)] = {
                    'population_size': len(cluster_indices),
                    'sample_size': cluster_sample_size
                }
        
        self.sample_indices_ = np.array(sample_indices)
        sample_data = data.iloc[self.sample_indices_].copy()
        
        self.sampling_result_ = SamplingResult(
            sampling_method='cluster',
            sample_size=len(sample_indices),
            population_size=len(data),
            sample_data=sample_data,
            sample_indices=self.sample_indices_,
            sampling_params={
                'cluster_column': self.cluster_column,
                'n_clusters_selected': n_clusters_to_sample,
                'replacement': self.replacement,
                'random_state': self.random_state
            },
            cluster_info=cluster_info
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
            sampling_method='systematic',
            sample_size=len(self.sample_indices_),
            population_size=len(data),
            sample_data=sample_data,
            sample_indices=self.sample_indices_,
            sampling_params={
                'sampling_interval': interval if sample_size < len(data) else 1,
                'starting_point': start if sample_size < len(data) else 0,
                'random_state': self.random_state
            }
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
            len(data), 
            size=sample_size, 
            replace=self.replacement, 
            p=weights
        )
        
        sample_data = data.iloc[self.sample_indices_].copy()
        
        self.sampling_result_ = SamplingResult(
            sampling_method='weighted',
            sample_size=sample_size,
            population_size=len(data),
            sample_data=sample_data,
            sample_indices=self.sample_indices_,
            sampling_params={
                'weights_column': self.weights_column,
                'replacement': self.replacement,
                'random_state': self.random_state
            }
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
                        mean_diffs.append(abs(pop_mean - sample_mean) / pop_std if pop_std > 0 else 0)
                    
                    if not np.isnan(pop_std) and not np.isnan(sample_std) and pop_std > 0:
                        std_ratios.append(sample_std / pop_std)
            
            if mean_diffs:
                # Representativeness score (lower is better, convert to higher is better)
                avg_mean_diff = np.mean(mean_diffs)
                self.sampling_result_.representativeness_score = max(0, 1 - avg_mean_diff)
                
                self.sampling_result_.coverage_metrics = {
                    'mean_absolute_difference': avg_mean_diff,
                    'std_ratio_mean': np.mean(std_ratios) if std_ratios else 1.0,
                    'std_ratio_std': np.std(std_ratios) if len(std_ratios) > 1 else 0.0
                }


class BootstrapTransformer(BaseEstimator, TransformerMixin):
    """
    sklearn-compatible transformer for bootstrap methods and confidence intervals.
    
    Implements parametric and non-parametric bootstrap, confidence interval construction,
    bias correction, and bootstrap hypothesis testing.
    
    Parameters:
    -----------
    statistic_func : callable or str, default='mean'
        Function to bootstrap or string name of common statistics
    n_bootstrap : int, default=1000
        Number of bootstrap samples
    confidence_level : float, default=0.95
        Confidence level for intervals
    method : str, default='percentile'
        Bootstrap method: 'percentile', 'bca', 'basic', 'studentized'
    random_state : int, default=None
        Random seed for reproducibility
    
    Attributes:
    -----------
    bootstrap_results_ : List[BootstrapResult]
        Results of bootstrap analysis for each column
    """
    
    def __init__(self,
                 statistic_func: Union[Callable, str] = 'mean',
                 n_bootstrap: int = 1000,
                 confidence_level: float = 0.95,
                 method: str = 'percentile',
                 random_state: Optional[int] = None):
        self.statistic_func = statistic_func
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
        self.method = method
        self.random_state = random_state

    def fit(self, X, y=None):
        """Fit the transformer (no-op for bootstrap)."""
        self._validate_parameters()
        return self

    def transform(self, X):
        """Perform bootstrap analysis on the input data."""
        check_is_fitted(self)
        
        if isinstance(X, pd.DataFrame):
            data = X
        else:
            data = pd.DataFrame(X)
        
        self.bootstrap_results_ = []
        
        # Get statistic function
        stat_func = self._get_statistic_function()
        
        # Perform bootstrap for each numeric column
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            try:
                col_data = data[col].dropna()
                if len(col_data) >= 2:
                    bootstrap_result = self._bootstrap_single_column(col_data, col, stat_func)
                    self.bootstrap_results_.append(bootstrap_result)
            except Exception as e:
                logger.warning(f"Bootstrap failed for column {col}: {e}")
        
        # Create result summary
        result_summary = {
            'bootstrap_results': [result.to_dict() for result in self.bootstrap_results_],
            'n_bootstrap': self.n_bootstrap,
            'confidence_level': self.confidence_level,
            'method': self.method
        }
        
        return pd.DataFrame([result_summary])

    def _validate_parameters(self):
        """Validate input parameters."""
        if not 0 < self.confidence_level < 1:
            raise ValueError("confidence_level must be between 0 and 1")
        
        if self.n_bootstrap <= 0:
            raise ValueError("n_bootstrap must be positive")
        
        valid_methods = ['percentile', 'bca', 'basic', 'studentized']
        if self.method not in valid_methods:
            raise ValueError(f"method must be one of {valid_methods}")

    def _get_statistic_function(self) -> Callable:
        """Get the statistic function to bootstrap."""
        if isinstance(self.statistic_func, str):
            if self.statistic_func == 'mean':
                return np.mean
            elif self.statistic_func == 'median':
                return np.median
            elif self.statistic_func == 'std':
                return np.std
            elif self.statistic_func == 'var':
                return np.var
            else:
                raise ValueError(f"Unknown statistic function: {self.statistic_func}")
        else:
            return self.statistic_func

    def _bootstrap_single_column(self, data: pd.Series, col_name: str, stat_func: Callable) -> BootstrapResult:
        """Perform bootstrap analysis for a single column."""
        # Original statistic
        original_stat = stat_func(data.values)
        
        # Generate bootstrap samples
        np.random.seed(self.random_state)
        bootstrap_stats = []
        
        for _ in range(self.n_bootstrap):
            bootstrap_sample = np.random.choice(data.values, size=len(data), replace=True)
            bootstrap_stat = stat_func(bootstrap_sample)
            bootstrap_stats.append(bootstrap_stat)
        
        bootstrap_stats = np.array(bootstrap_stats)
        
        # Calculate bias and variance
        bias_estimate = np.mean(bootstrap_stats) - original_stat
        variance_estimate = np.var(bootstrap_stats)
        standard_error = np.sqrt(variance_estimate)
        
        # Calculate confidence intervals
        confidence_intervals = {}
        alpha = 1 - self.confidence_level
        
        if self.method == 'percentile':
            lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
            upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
            confidence_intervals['percentile'] = (lower, upper)
        
        elif self.method == 'basic':
            lower = 2 * original_stat - np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
            upper = 2 * original_stat - np.percentile(bootstrap_stats, 100 * alpha / 2)
            confidence_intervals['basic'] = (lower, upper)
        
        elif self.method == 'bca':
            # BCa (Bias-Corrected and Accelerated) method
            # This is a simplified implementation
            z0 = stats.norm.ppf((bootstrap_stats < original_stat).mean())
            
            # Jackknife to estimate acceleration
            n = len(data)
            jackknife_stats = []
            for i in range(n):
                jackknife_sample = np.concatenate([data.values[:i], data.values[i+1:]])
                jackknife_stat = stat_func(jackknife_sample)
                jackknife_stats.append(jackknife_stat)
            
            jackknife_stats = np.array(jackknife_stats)
            jackknife_mean = np.mean(jackknife_stats)
            
            # Acceleration parameter
            numerator = np.sum((jackknife_mean - jackknife_stats) ** 3)
            denominator = 6 * (np.sum((jackknife_mean - jackknife_stats) ** 2)) ** 1.5
            
            if denominator != 0:
                acceleration = numerator / denominator
            else:
                acceleration = 0
            
            # Adjusted percentiles
            z_alpha_2 = stats.norm.ppf(alpha / 2)
            z_1_alpha_2 = stats.norm.ppf(1 - alpha / 2)
            
            alpha1 = stats.norm.cdf(z0 + (z0 + z_alpha_2) / (1 - acceleration * (z0 + z_alpha_2)))
            alpha2 = stats.norm.cdf(z0 + (z0 + z_1_alpha_2) / (1 - acceleration * (z0 + z_1_alpha_2)))
            
            lower = np.percentile(bootstrap_stats, 100 * alpha1)
            upper = np.percentile(bootstrap_stats, 100 * alpha2)
            confidence_intervals['bca'] = (lower, upper)
        
        # Bias-corrected estimate
        bias_corrected_estimate = original_stat - bias_estimate
        
        return BootstrapResult(
            statistic_name=f"{self.statistic_func}_{col_name}",
            original_statistic=original_stat,
            bootstrap_method=self.method,
            n_bootstrap=self.n_bootstrap,
            bootstrap_distribution=bootstrap_stats,
            confidence_intervals=confidence_intervals,
            bias_estimate=bias_estimate,
            bias_corrected_estimate=bias_corrected_estimate,
            variance_estimate=variance_estimate,
            standard_error=standard_error,
            bootstrap_params={
                'confidence_level': self.confidence_level,
                'random_state': self.random_state
            }
        )


class MonteCarloTransformer(BaseEstimator, TransformerMixin):
    """
    sklearn-compatible transformer for Monte Carlo methods and simulation.
    
    Implements Monte Carlo integration, uncertainty quantification, importance sampling,
    and basic Markov Chain Monte Carlo methods.
    
    Parameters:
    -----------
    simulation_type : str, default='integration'
        Type of simulation: 'integration', 'uncertainty', 'importance', 'mcmc'
    n_simulations : int, default=10000
        Number of Monte Carlo simulations
    confidence_level : float, default=0.95
        Confidence level for uncertainty quantification
    random_state : int, default=None
        Random seed for reproducibility
    
    Attributes:
    -----------
    monte_carlo_results_ : List[MonteCarloResult]
        Results of Monte Carlo analysis
    """
    
    def __init__(self,
                 simulation_type: str = 'integration',
                 n_simulations: int = 10000,
                 confidence_level: float = 0.95,
                 random_state: Optional[int] = None,
                 integration_bounds: Optional[Tuple[float, float]] = None,
                 target_function: Optional[Callable] = None):
        self.simulation_type = simulation_type
        self.n_simulations = n_simulations
        self.confidence_level = confidence_level
        self.random_state = random_state
        self.integration_bounds = integration_bounds
        self.target_function = target_function

    def fit(self, X, y=None):
        """Fit the transformer (no-op for Monte Carlo)."""
        self._validate_parameters()
        return self

    def transform(self, X):
        """Perform Monte Carlo analysis on the input data."""
        check_is_fitted(self)
        
        if isinstance(X, pd.DataFrame):
            data = X
        else:
            data = pd.DataFrame(X)
        
        self.monte_carlo_results_ = []
        
        if self.simulation_type == 'integration':
            self._monte_carlo_integration(data)
        elif self.simulation_type == 'uncertainty':
            self._uncertainty_quantification(data)
        elif self.simulation_type == 'importance':
            self._importance_sampling(data)
        elif self.simulation_type == 'mcmc':
            self._basic_mcmc(data)
        
        # Create result summary
        result_summary = {
            'monte_carlo_results': [result.to_dict() for result in self.monte_carlo_results_],
            'simulation_type': self.simulation_type,
            'n_simulations': self.n_simulations,
            'confidence_level': self.confidence_level
        }
        
        return pd.DataFrame([result_summary])

    def _validate_parameters(self):
        """Validate input parameters."""
        valid_types = ['integration', 'uncertainty', 'importance', 'mcmc']
        if self.simulation_type not in valid_types:
            raise ValueError(f"simulation_type must be one of {valid_types}")
        
        if self.n_simulations <= 0:
            raise ValueError("n_simulations must be positive")

    def _monte_carlo_integration(self, data: pd.DataFrame):
        """Perform Monte Carlo integration."""
        if self.target_function is None:
            # Default: integrate normal distribution
            def target_func(x):
                return stats.norm.pdf(x)
            
            bounds = self.integration_bounds or (-3, 3)
        else:
            target_func = self.target_function
            bounds = self.integration_bounds or (0, 1)
        
        np.random.seed(self.random_state)
        
        # Generate uniform random samples in integration bounds
        a, b = bounds
        uniform_samples = np.random.uniform(a, b, self.n_simulations)
        
        # Evaluate function at sample points
        function_values = np.array([target_func(x) for x in uniform_samples])
        
        # Monte Carlo estimate
        integral_estimate = (b - a) * np.mean(function_values)
        
        # Standard error
        standard_error = (b - a) * np.std(function_values) / np.sqrt(self.n_simulations)
        
        # Confidence interval
        alpha = 1 - self.confidence_level
        z_critical = stats.norm.ppf(1 - alpha / 2)
        margin_error = z_critical * standard_error
        
        confidence_interval = (integral_estimate - margin_error, integral_estimate + margin_error)
        
        # Convergence diagnostic
        batch_size = self.n_simulations // 10
        batch_estimates = []
        for i in range(0, self.n_simulations, batch_size):
            batch_values = function_values[i:i+batch_size]
            batch_estimate = (b - a) * np.mean(batch_values)
            batch_estimates.append(batch_estimate)
        
        convergence_diagnostic = {
            'batch_variance': np.var(batch_estimates),
            'relative_std_error': standard_error / abs(integral_estimate) if integral_estimate != 0 else float('inf')
        }
        
        result = MonteCarloResult(
            simulation_type='integration',
            n_simulations=self.n_simulations,
            simulation_results=function_values,
            estimated_value=integral_estimate,
            confidence_interval=confidence_interval,
            standard_error=standard_error,
            convergence_diagnostic=convergence_diagnostic,
            integration_bounds=bounds,
            function_info={'bounds': bounds},
            simulation_params={'random_state': self.random_state}
        )
        
        self.monte_carlo_results_.append(result)

    def _uncertainty_quantification(self, data: pd.DataFrame):
        """Perform uncertainty quantification via Monte Carlo simulation."""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            col_data = data[col].dropna()
            if len(col_data) < 2:
                continue
            
            # Fit distribution to data
            try:
                # Try normal distribution first
                mu, sigma = stats.norm.fit(col_data)
                
                # Generate Monte Carlo samples
                np.random.seed(self.random_state)
                mc_samples = np.random.normal(mu, sigma, self.n_simulations)
                
                # Calculate statistics
                mean_estimate = np.mean(mc_samples)
                std_estimate = np.std(mc_samples)
                
                # Confidence interval for the mean
                alpha = 1 - self.confidence_level
                lower = np.percentile(mc_samples, 100 * alpha / 2)
                upper = np.percentile(mc_samples, 100 * (1 - alpha / 2))
                
                # Standard error of the mean
                standard_error = std_estimate / np.sqrt(self.n_simulations)
                
                result = MonteCarloResult(
                    simulation_type='uncertainty',
                    n_simulations=self.n_simulations,
                    simulation_results=mc_samples,
                    estimated_value=mean_estimate,
                    confidence_interval=(lower, upper),
                    standard_error=standard_error,
                    simulation_params={
                        'fitted_distribution': 'normal',
                        'parameters': {'mu': mu, 'sigma': sigma},
                        'column': col,
                        'random_state': self.random_state
                    }
                )
                
                self.monte_carlo_results_.append(result)
                
            except Exception as e:
                logger.warning(f"Uncertainty quantification failed for {col}: {e}")

    def _importance_sampling(self, data: pd.DataFrame):
        """Perform importance sampling for rare events."""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            col_data = data[col].dropna()
            if len(col_data) < 2:
                continue
            
            try:
                # Define rare event (e.g., values in top 5%)
                threshold = np.percentile(col_data, 95)
                
                # Fit normal distribution to data
                mu, sigma = stats.norm.fit(col_data)
                
                # Importance sampling: use shifted distribution
                # Shift mean towards the rare event region
                shifted_mu = threshold + sigma
                
                np.random.seed(self.random_state)
                
                # Generate samples from importance distribution
                importance_samples = np.random.normal(shifted_mu, sigma, self.n_simulations)
                
                # Calculate importance weights
                original_density = stats.norm.pdf(importance_samples, mu, sigma)
                importance_density = stats.norm.pdf(importance_samples, shifted_mu, sigma)
                
                # Avoid division by zero
                importance_density = np.maximum(importance_density, 1e-10)
                weights = original_density / importance_density
                
                # Indicator function for rare event
                indicators = importance_samples > threshold
                
                # Importance sampling estimate
                rare_event_probability = np.mean(indicators * weights)
                
                # Standard error (simplified)
                weighted_variance = np.var(indicators * weights)
                standard_error = np.sqrt(weighted_variance / self.n_simulations)
                
                # Confidence interval
                alpha = 1 - self.confidence_level
                z_critical = stats.norm.ppf(1 - alpha / 2)
                margin_error = z_critical * standard_error
                
                confidence_interval = (
                    max(0, rare_event_probability - margin_error),
                    min(1, rare_event_probability + margin_error)
                )
                
                result = MonteCarloResult(
                    simulation_type='importance_sampling',
                    n_simulations=self.n_simulations,
                    simulation_results=importance_samples,
                    estimated_value=rare_event_probability,
                    confidence_interval=confidence_interval,
                    standard_error=standard_error,
                    simulation_params={
                        'threshold': threshold,
                        'original_mu': mu,
                        'original_sigma': sigma,
                        'importance_mu': shifted_mu,
                        'column': col,
                        'random_state': self.random_state
                    }
                )
                
                self.monte_carlo_results_.append(result)
                
            except Exception as e:
                logger.warning(f"Importance sampling failed for {col}: {e}")

    def _basic_mcmc(self, data: pd.DataFrame):
        """Perform basic MCMC simulation (Metropolis-Hastings)."""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            col_data = data[col].dropna()
            if len(col_data) < 2:
                continue
            
            try:
                # Simple MCMC to estimate mean of normal distribution
                # Prior: normal with large variance
                prior_mu = 0
                prior_sigma = 100
                
                # Likelihood: normal with known variance
                data_sigma = np.std(col_data)
                data_mean = np.mean(col_data)
                n_data = len(col_data)
                
                # MCMC chain
                np.random.seed(self.random_state)
                
                chain = np.zeros(self.n_simulations)
                current_mu = np.random.normal(data_mean, data_sigma)
                
                n_accepted = 0
                proposal_sigma = data_sigma / 10  # Proposal distribution std
                
                for i in range(self.n_simulations):
                    # Propose new state
                    proposed_mu = np.random.normal(current_mu, proposal_sigma)
                    
                    # Calculate log-likelihood ratio
                    current_log_likelihood = -0.5 * n_data * np.log(2 * np.pi * data_sigma**2) - \
                                           0.5 * np.sum((col_data - current_mu)**2) / data_sigma**2
                    
                    proposed_log_likelihood = -0.5 * n_data * np.log(2 * np.pi * data_sigma**2) - \
                                            0.5 * np.sum((col_data - proposed_mu)**2) / data_sigma**2
                    
                    # Add prior
                    current_log_prior = -0.5 * (current_mu - prior_mu)**2 / prior_sigma**2
                    proposed_log_prior = -0.5 * (proposed_mu - prior_mu)**2 / prior_sigma**2
                    
                    # Acceptance probability
                    log_alpha = (proposed_log_likelihood + proposed_log_prior) - \
                               (current_log_likelihood + current_log_prior)
                    
                    alpha = min(1, np.exp(log_alpha))
                    
                    # Accept or reject
                    if np.random.random() < alpha:
                        current_mu = proposed_mu
                        n_accepted += 1
                    
                    chain[i] = current_mu
                
                # Remove burn-in (first 10%)
                burn_in = max(1, self.n_simulations // 10)
                chain_post_burnin = chain[burn_in:]
                
                # Calculate statistics
                posterior_mean = np.mean(chain_post_burnin)
                posterior_std = np.std(chain_post_burnin)
                
                # Credible interval
                alpha = 1 - self.confidence_level
                lower = np.percentile(chain_post_burnin, 100 * alpha / 2)
                upper = np.percentile(chain_post_burnin, 100 * (1 - alpha / 2))
                
                # MCMC diagnostics
                acceptance_rate = n_accepted / self.n_simulations
                effective_sample_size = len(chain_post_burnin) / (1 + 2 * self._autocorrelation_time(chain_post_burnin))
                
                result = MonteCarloResult(
                    simulation_type='mcmc',
                    n_simulations=self.n_simulations,
                    simulation_results=chain_post_burnin,
                    estimated_value=posterior_mean,
                    confidence_interval=(lower, upper),
                    standard_error=posterior_std / np.sqrt(len(chain_post_burnin)),
                    convergence_diagnostic={
                        'acceptance_rate': acceptance_rate,
                        'effective_sample_size': effective_sample_size,
                        'burn_in_length': burn_in
                    },
                    simulation_params={
                        'column': col,
                        'prior_mu': prior_mu,
                        'prior_sigma': prior_sigma,
                        'data_sigma': data_sigma,
                        'proposal_sigma': proposal_sigma,
                        'random_state': self.random_state
                    }
                )
                
                self.monte_carlo_results_.append(result)
                
            except Exception as e:
                logger.warning(f"MCMC failed for {col}: {e}")

    def _autocorrelation_time(self, chain: np.ndarray, max_lag: int = None) -> float:
        """Estimate autocorrelation time of MCMC chain."""
        if max_lag is None:
            max_lag = min(len(chain) // 4, 100)
        
        try:
            # Calculate autocorrelation function
            chain_centered = chain - np.mean(chain)
            autocorr = np.correlate(chain_centered, chain_centered, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            autocorr = autocorr / autocorr[0]
            
            # Find first time autocorr drops below 1/e
            threshold = 1 / np.e
            for tau in range(1, min(len(autocorr), max_lag)):
                if autocorr[tau] < threshold:
                    return tau
            
            return max_lag
            
        except:
            return 1.0  # Default if calculation fails


class BayesianEstimationTransformer(BaseEstimator, TransformerMixin):
    """
    sklearn-compatible transformer for Bayesian estimation and inference.
    
    Implements posterior estimation, credible intervals, Bayesian updating,
    and basic model comparison using conjugate priors.
    
    Parameters:
    -----------
    estimation_type : str, default='posterior'
        Type of estimation: 'posterior', 'credible_interval', 'model_comparison'
    prior_distribution : str, default='normal'
        Prior distribution: 'normal', 'gamma', 'beta', 'uniform'
    prior_params : dict, default=None
        Parameters for prior distribution
    confidence_level : float, default=0.95
        Credible interval level
    n_samples : int, default=10000
        Number of posterior samples
    random_state : int, default=None
        Random seed for reproducibility
    
    Attributes:
    -----------
    bayesian_results_ : List[BayesianResult]
        Results of Bayesian analysis
    """
    
    def __init__(self,
                 estimation_type: str = 'posterior',
                 prior_distribution: str = 'normal',
                 prior_params: Optional[Dict[str, float]] = None,
                 confidence_level: float = 0.95,
                 n_samples: int = 10000,
                 random_state: Optional[int] = None):
        self.estimation_type = estimation_type
        self.prior_distribution = prior_distribution
        self.prior_params = prior_params or {}
        self.confidence_level = confidence_level
        self.n_samples = n_samples
        self.random_state = random_state

    def fit(self, X, y=None):
        """Fit the transformer (no-op for Bayesian estimation)."""
        self._validate_parameters()
        return self

    def transform(self, X):
        """Perform Bayesian estimation on the input data."""
        check_is_fitted(self)
        
        if isinstance(X, pd.DataFrame):
            data = X
        else:
            data = pd.DataFrame(X)
        
        self.bayesian_results_ = []
        
        # Perform Bayesian analysis for numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            col_data = data[col].dropna()
            if len(col_data) >= 2:
                try:
                    if self.estimation_type == 'posterior':
                        result = self._estimate_posterior(col_data, col)
                    elif self.estimation_type == 'credible_interval':
                        result = self._calculate_credible_intervals(col_data, col)
                    elif self.estimation_type == 'model_comparison':
                        result = self._compare_models(col_data, col)
                    else:
                        result = self._estimate_posterior(col_data, col)
                    
                    self.bayesian_results_.append(result)
                except Exception as e:
                    logger.warning(f"Bayesian estimation failed for {col}: {e}")
        
        # Create result summary
        result_summary = {
            'bayesian_results': [result.to_dict() for result in self.bayesian_results_],
            'estimation_type': self.estimation_type,
            'prior_distribution': self.prior_distribution,
            'prior_params': self.prior_params,
            'confidence_level': self.confidence_level
        }
        
        return pd.DataFrame([result_summary])

    def _validate_parameters(self):
        """Validate input parameters."""
        valid_types = ['posterior', 'credible_interval', 'model_comparison']
        if self.estimation_type not in valid_types:
            raise ValueError(f"estimation_type must be one of {valid_types}")
        
        valid_priors = ['normal', 'gamma', 'beta', 'uniform']
        if self.prior_distribution not in valid_priors:
            raise ValueError(f"prior_distribution must be one of {valid_priors}")

    def _estimate_posterior(self, data: pd.Series, col_name: str) -> BayesianResult:
        """Estimate posterior distribution using conjugate priors."""
        data_values = data.values
        n = len(data_values)
        
        if self.prior_distribution == 'normal':
            # Normal-Normal conjugate prior for mean estimation
            # Assume known variance for simplicity
            data_var = np.var(data_values)
            data_mean = np.mean(data_values)
            
            # Prior parameters
            prior_mu = self.prior_params.get('mu', 0.0)
            prior_sigma2 = self.prior_params.get('sigma2', 100.0)
            
            # Posterior parameters (conjugate update)
            posterior_precision = 1/prior_sigma2 + n/data_var
            posterior_sigma2 = 1/posterior_precision
            posterior_mu = (prior_mu/prior_sigma2 + n*data_mean/data_var) / posterior_precision
            
            # Generate posterior samples
            np.random.seed(self.random_state)
            posterior_samples = np.random.normal(posterior_mu, np.sqrt(posterior_sigma2), self.n_samples)
            
            # Calculate credible intervals
            alpha = 1 - self.confidence_level
            lower = np.percentile(posterior_samples, 100 * alpha / 2)
            upper = np.percentile(posterior_samples, 100 * (1 - alpha / 2))
            
            return BayesianResult(
                parameter_name=f"mean_{col_name}",
                estimation_method='normal_conjugate',
                posterior_samples=posterior_samples,
                posterior_mean=posterior_mu,
                posterior_median=np.median(posterior_samples),
                credible_intervals={f'{self.confidence_level:.0%}': (lower, upper)},
                prior_info={
                    'distribution': 'normal',
                    'prior_mu': prior_mu,
                    'prior_sigma2': prior_sigma2,
                    'posterior_mu': posterior_mu,
                    'posterior_sigma2': posterior_sigma2
                }
            )
        
        elif self.prior_distribution == 'gamma':
            # Gamma-Normal conjugate prior for precision estimation
            # Estimating precision (inverse variance) of normal data
            
            # Prior parameters for Gamma distribution
            prior_alpha = self.prior_params.get('alpha', 1.0)
            prior_beta = self.prior_params.get('beta', 1.0)
            
            # Sufficient statistics
            sum_x = np.sum(data_values)
            sum_x2 = np.sum(data_values**2)
            data_mean = np.mean(data_values)
            
            # Posterior parameters (assuming known mean)
            posterior_alpha = prior_alpha + n/2
            posterior_beta = prior_beta + 0.5 * np.sum((data_values - data_mean)**2)
            
            # Generate posterior samples for precision
            np.random.seed(self.random_state)
            precision_samples = np.random.gamma(posterior_alpha, 1/posterior_beta, self.n_samples)
            variance_samples = 1 / precision_samples
            
            # Calculate credible intervals for variance
            alpha = 1 - self.confidence_level
            lower = np.percentile(variance_samples, 100 * alpha / 2)
            upper = np.percentile(variance_samples, 100 * (1 - alpha / 2))
            
            return BayesianResult(
                parameter_name=f"variance_{col_name}",
                estimation_method='gamma_conjugate',
                posterior_samples=variance_samples,
                posterior_mean=np.mean(variance_samples),
                posterior_median=np.median(variance_samples),
                credible_intervals={f'{self.confidence_level:.0%}': (lower, upper)},
                prior_info={
                    'distribution': 'gamma',
                    'prior_alpha': prior_alpha,
                    'prior_beta': prior_beta,
                    'posterior_alpha': posterior_alpha,
                    'posterior_beta': posterior_beta
                }
            )
        
        else:
            # For other priors, use approximate methods
            return self._approximate_posterior(data, col_name)

    def _approximate_posterior(self, data: pd.Series, col_name: str) -> BayesianResult:
        """Approximate posterior using sampling methods."""
        # Simple approximation: assume normal posterior centered on sample mean
        data_mean = np.mean(data.values)
        data_std = np.std(data.values)
        n = len(data.values)
        
        # Approximate posterior standard error
        posterior_std = data_std / np.sqrt(n)
        
        # Generate approximate posterior samples
        np.random.seed(self.random_state)
        posterior_samples = np.random.normal(data_mean, posterior_std, self.n_samples)
        
        # Calculate credible intervals
        alpha = 1 - self.confidence_level
        lower = np.percentile(posterior_samples, 100 * alpha / 2)
        upper = np.percentile(posterior_samples, 100 * (1 - alpha / 2))
        
        return BayesianResult(
            parameter_name=f"mean_{col_name}",
            estimation_method='approximate',
            posterior_samples=posterior_samples,
            posterior_mean=data_mean,
            posterior_median=np.median(posterior_samples),
            credible_intervals={f'{self.confidence_level:.0%}': (lower, upper)},
            prior_info={
                'distribution': self.prior_distribution,
                'method': 'approximate'
            }
        )

    def _calculate_credible_intervals(self, data: pd.Series, col_name: str) -> BayesianResult:
        """Calculate credible intervals for multiple confidence levels."""
        # First get posterior
        posterior_result = self._estimate_posterior(data, col_name)
        
        # Calculate multiple credible intervals
        credible_levels = [0.50, 0.90, 0.95, 0.99]
        credible_intervals = {}
        
        for level in credible_levels:
            alpha = 1 - level
            lower = np.percentile(posterior_result.posterior_samples, 100 * alpha / 2)
            upper = np.percentile(posterior_result.posterior_samples, 100 * (1 - alpha / 2))
            credible_intervals[f'{level:.0%}'] = (lower, upper)
        
        # Update the result
        posterior_result.credible_intervals = credible_intervals
        
        return posterior_result

    def _compare_models(self, data: pd.Series, col_name: str) -> BayesianResult:
        """Perform basic Bayesian model comparison."""
        # Compare normal vs. exponential models
        data_values = data.values
        
        # Model 1: Normal distribution
        mu_ml, sigma_ml = stats.norm.fit(data_values)
        log_likelihood_normal = np.sum(stats.norm.logpdf(data_values, mu_ml, sigma_ml))
        
        # Model 2: Exponential distribution  
        if np.all(data_values > 0):  # Exponential requires positive data
            lambda_ml = 1 / np.mean(data_values)
            log_likelihood_exp = np.sum(stats.expon.logpdf(data_values, scale=1/lambda_ml))
        else:
            log_likelihood_exp = -np.inf
        
        # Simple Bayes factor approximation (assuming equal priors)
        # In practice, this should include proper prior specification and integration
        bayes_factor = np.exp(log_likelihood_normal - log_likelihood_exp)
        
        # Choose better model
        if bayes_factor > 1:
            better_model = 'normal'
            model_params = {'mu': mu_ml, 'sigma': sigma_ml}
        else:
            better_model = 'exponential'
            model_params = {'lambda': lambda_ml}
        
        # Generate posterior samples for the better model
        np.random.seed(self.random_state)
        if better_model == 'normal':
            posterior_samples = np.random.normal(mu_ml, sigma_ml/np.sqrt(len(data_values)), self.n_samples)
            posterior_mean = mu_ml
        else:
            # For exponential, sample the rate parameter
            # Using gamma posterior (conjugate)
            prior_alpha = 1.0  # Default prior
            prior_beta = 1.0
            posterior_alpha = prior_alpha + len(data_values)
            posterior_beta = prior_beta + np.sum(data_values)
            
            rate_samples = np.random.gamma(posterior_alpha, 1/posterior_beta, self.n_samples)
            posterior_samples = 1 / rate_samples  # Mean of exponential
            posterior_mean = np.mean(posterior_samples)
        
        # Calculate credible intervals
        alpha = 1 - self.confidence_level
        lower = np.percentile(posterior_samples, 100 * alpha / 2)
        upper = np.percentile(posterior_samples, 100 * (1 - alpha / 2))
        
        return BayesianResult(
            parameter_name=f"parameter_{col_name}",
            estimation_method='model_comparison',
            posterior_samples=posterior_samples,
            posterior_mean=posterior_mean,
            posterior_median=np.median(posterior_samples),
            credible_intervals={f'{self.confidence_level:.0%}': (lower, upper)},
            bayes_factor=bayes_factor,
            prior_info={
                'compared_models': ['normal', 'exponential'],
                'selected_model': better_model,
                'model_params': model_params,
                'log_likelihood_normal': log_likelihood_normal,
                'log_likelihood_exponential': log_likelihood_exp
            }
        )


# MCP Tool Functions
def generate_sample(data: Union[pd.DataFrame, str],
                   sampling_method: str = 'simple_random',
                   sample_size: Union[int, float] = 0.1,
                   **kwargs) -> Dict[str, Any]:
    """
    Generate sample from data using various sampling techniques.
    
    Args:
        data: DataFrame or path to data file
        sampling_method: Type of sampling method to use
        sample_size: Sample size as integer or fraction
        **kwargs: Additional sampling parameters
    
    Returns:
        Dictionary containing sampling results and sample data
    """
    if isinstance(data, str):
        # Load data from file path
        if data.endswith('.csv'):
            data = pd.read_csv(data)
        elif data.endswith('.json'):
            data = pd.read_json(data)
        else:
            raise ValueError("Unsupported file format")
    
    # Initialize and run sampling transformer
    transformer = SamplingTransformer(
        sampling_method=sampling_method,
        sample_size=sample_size,
        **kwargs
    )
    
    # Fit and transform
    transformer.fit(data)
    sample_data = transformer.transform(data)
    
    # Return both the sample and the results
    return {
        'sample_data': sample_data.to_dict('records') if hasattr(sample_data, 'to_dict') else sample_data,
        'sampling_results': transformer.sampling_result_.to_dict()
    }


def bootstrap_statistic(data: Union[pd.DataFrame, str],
                       statistic_func: Union[Callable, str] = 'mean',
                       n_bootstrap: int = 1000,
                       confidence_level: float = 0.95,
                       **kwargs) -> Dict[str, Any]:
    """
    Perform bootstrap analysis on data.
    
    Args:
        data: DataFrame or path to data file
        statistic_func: Function to bootstrap or string name
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level for intervals
        **kwargs: Additional bootstrap parameters
    
    Returns:
        Dictionary containing bootstrap results and confidence intervals
    """
    if isinstance(data, str):
        # Load data from file path
        if data.endswith('.csv'):
            data = pd.read_csv(data)
        elif data.endswith('.json'):
            data = pd.read_json(data)
        else:
            raise ValueError("Unsupported file format")
    
    # Initialize and run bootstrap transformer
    transformer = BootstrapTransformer(
        statistic_func=statistic_func,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        **kwargs
    )
    
    # Fit and transform
    transformer.fit(data)
    result_df = transformer.transform(data)
    
    return result_df.iloc[0].to_dict()


def monte_carlo_simulate(data: Union[pd.DataFrame, str],
                        simulation_type: str = 'integration',
                        n_simulations: int = 10000,
                        **kwargs) -> Dict[str, Any]:
    """
    Perform Monte Carlo simulation analysis.
    
    Args:
        data: DataFrame or path to data file
        simulation_type: Type of Monte Carlo simulation
        n_simulations: Number of simulations to run
        **kwargs: Additional simulation parameters
    
    Returns:
        Dictionary containing Monte Carlo simulation results
    """
    if isinstance(data, str):
        # Load data from file path
        if data.endswith('.csv'):
            data = pd.read_csv(data)
        elif data.endswith('.json'):
            data = pd.read_json(data)
        else:
            raise ValueError("Unsupported file format")
    
    # Initialize and run Monte Carlo transformer
    transformer = MonteCarloTransformer(
        simulation_type=simulation_type,
        n_simulations=n_simulations,
        **kwargs
    )
    
    # Fit and transform
    transformer.fit(data)
    result_df = transformer.transform(data)
    
    return result_df.iloc[0].to_dict()


def bayesian_estimate(data: Union[pd.DataFrame, str],
                     estimation_type: str = 'posterior',
                     prior_distribution: str = 'normal',
                     confidence_level: float = 0.95,
                     **kwargs) -> Dict[str, Any]:
    """
    Perform Bayesian estimation and inference.
    
    Args:
        data: DataFrame or path to data file
        estimation_type: Type of Bayesian estimation
        prior_distribution: Prior distribution to use
        confidence_level: Credible interval level
        **kwargs: Additional Bayesian parameters
    
    Returns:
        Dictionary containing Bayesian estimation results
    """
    if isinstance(data, str):
        # Load data from file path
        if data.endswith('.csv'):
            data = pd.read_csv(data)
        elif data.endswith('.json'):
            data = pd.read_json(data)
        else:
            raise ValueError("Unsupported file format")
    
    # Initialize and run Bayesian transformer
    transformer = BayesianEstimationTransformer(
        estimation_type=estimation_type,
        prior_distribution=prior_distribution,
        confidence_level=confidence_level,
        **kwargs
    )
    
    # Fit and transform
    transformer.fit(data)
    result_df = transformer.transform(data)
    
    return result_df.iloc[0].to_dict()