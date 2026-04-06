"""
Sampling & Estimation Domain - Result dataclasses.

Standardized result structures for sampling, bootstrap, Monte Carlo,
and Bayesian estimation operations.
"""

import json
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import bootstrap
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils.validation import check_array, check_is_fitted

try:
    from ...logging_manager import get_logger
    from ...pipeline.base import (
        AnalysisPipelineBase,
        CompositionMetadata,
        PipelineResult,
        PipelineState,
        StreamingConfig,
    )
except ImportError:
    # For testing and standalone usage
    import logging

    def get_logger(name):
        return logging.getLogger(name)

    # Mock the pipeline classes if not available
    class AnalysisPipelineBase:
        pass

    class PipelineResult:
        pass

    class CompositionMetadata:
        pass

    class StreamingConfig:
        pass

    class PipelineState:
        pass


logger = get_logger(__name__)

# Suppress specific warnings that are not critical for our use case
warnings.filterwarnings("ignore", category=RuntimeWarning, module="scipy.stats")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


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
            "sampling_method": self.sampling_method,
            "sample_size": self.sample_size,
            "population_size": self.population_size,
            "sampling_params": self.sampling_params,
        }

        if self.sample_indices is not None:
            result_dict["sample_indices"] = self.sample_indices.tolist()

        if self.representativeness_score is not None:
            result_dict["quality_metrics"] = {
                "representativeness_score": self.representativeness_score,
                "coverage_metrics": self.coverage_metrics,
                "bias_estimates": self.bias_estimates,
            }

        if self.strata_info:
            result_dict["strata_info"] = self.strata_info
        if self.cluster_info:
            result_dict["cluster_info"] = self.cluster_info

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
            "statistic_name": self.statistic_name,
            "original_statistic": self.original_statistic,
            "bootstrap_method": self.bootstrap_method,
            "n_bootstrap": self.n_bootstrap,
            "bias_estimate": self.bias_estimate,
            "variance_estimate": self.variance_estimate,
            "standard_error": self.standard_error,
        }

        if self.bias_corrected_estimate is not None:
            result_dict["bias_corrected_estimate"] = self.bias_corrected_estimate

        if self.confidence_intervals:
            result_dict["confidence_intervals"] = self.confidence_intervals

        if self.convergence_info:
            result_dict["convergence_info"] = self.convergence_info

        if self.bootstrap_params:
            result_dict["bootstrap_params"] = self.bootstrap_params

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
            "simulation_type": self.simulation_type,
            "n_simulations": self.n_simulations,
            "estimated_value": self.estimated_value,
            "confidence_interval": self.confidence_interval,
            "standard_error": self.standard_error,
            "convergence_diagnostic": self.convergence_diagnostic,
            "simulation_params": self.simulation_params,
        }

        if self.integration_bounds is not None:
            result_dict["integration_bounds"] = self.integration_bounds

        if self.function_info:
            result_dict["function_info"] = self.function_info

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
            "parameter_name": self.parameter_name,
            "estimation_method": self.estimation_method,
            "posterior_mean": self.posterior_mean,
            "credible_intervals": self.credible_intervals,
            "prior_info": self.prior_info,
        }

        if self.posterior_mode is not None:
            result_dict["posterior_mode"] = self.posterior_mode
        if self.posterior_median is not None:
            result_dict["posterior_median"] = self.posterior_median
        if self.bayes_factor is not None:
            result_dict["bayes_factor"] = self.bayes_factor
        if self.marginal_likelihood is not None:
            result_dict["marginal_likelihood"] = self.marginal_likelihood

        if self.mcmc_diagnostics:
            result_dict["mcmc_diagnostics"] = self.mcmc_diagnostics

        return result_dict
