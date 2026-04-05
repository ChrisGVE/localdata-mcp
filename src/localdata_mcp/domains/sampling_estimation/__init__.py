"""
Sampling & Estimation Domain - Comprehensive sampling techniques and statistical estimation.

This package implements advanced sampling methods and statistical estimation tools including
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

from ._results import (
    SamplingResult,
    BootstrapResult,
    MonteCarloResult,
    BayesianResult,
)

from ._sampling import SamplingTransformer
from ._bootstrap import BootstrapTransformer
from ._monte_carlo import MonteCarloTransformer
from ._bayesian import BayesianEstimationTransformer

from ._functions import (
    generate_sample,
    bootstrap_statistic,
    monte_carlo_simulate,
    bayesian_estimate,
)

__all__ = [
    # Result classes
    "SamplingResult",
    "BootstrapResult",
    "MonteCarloResult",
    "BayesianResult",
    # Transformers
    "SamplingTransformer",
    "BootstrapTransformer",
    "MonteCarloTransformer",
    "BayesianEstimationTransformer",
    # High-level functions
    "generate_sample",
    "bootstrap_statistic",
    "monte_carlo_simulate",
    "bayesian_estimate",
]
