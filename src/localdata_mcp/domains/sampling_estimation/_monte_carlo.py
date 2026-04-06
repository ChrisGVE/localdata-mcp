"""
Sampling & Estimation Domain - MonteCarloTransformer.

sklearn-compatible transformer for Monte Carlo methods and simulation including
integration, uncertainty quantification, importance sampling, and basic MCMC.
"""

from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from ._results import MonteCarloResult, get_logger

logger = get_logger(__name__)


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

    def __init__(
        self,
        simulation_type: str = "integration",
        n_simulations: int = 10000,
        confidence_level: float = 0.95,
        random_state: Optional[int] = None,
        integration_bounds: Optional[Tuple[float, float]] = None,
        target_function: Optional[Callable] = None,
    ):
        self.simulation_type = simulation_type
        self.n_simulations = n_simulations
        self.confidence_level = confidence_level
        self.random_state = random_state
        self.integration_bounds = integration_bounds
        self.target_function = target_function

    def fit(self, X, y=None):
        """Fit the transformer (no-op for Monte Carlo)."""
        self._validate_parameters()
        self.is_fitted_ = True
        return self

    def transform(self, X):
        """Perform Monte Carlo analysis on the input data."""
        check_is_fitted(self)

        if isinstance(X, pd.DataFrame):
            data = X
        else:
            data = pd.DataFrame(X)

        self.monte_carlo_results_ = []

        if self.simulation_type == "integration":
            self._monte_carlo_integration(data)
        elif self.simulation_type == "uncertainty":
            self._uncertainty_quantification(data)
        elif self.simulation_type == "importance":
            self._importance_sampling(data)
        elif self.simulation_type == "mcmc":
            self._basic_mcmc(data)

        # Create result summary
        result_summary = {
            "monte_carlo_results": [
                result.to_dict() for result in self.monte_carlo_results_
            ],
            "simulation_type": self.simulation_type,
            "n_simulations": self.n_simulations,
            "confidence_level": self.confidence_level,
        }

        return pd.DataFrame([result_summary])

    def _validate_parameters(self):
        """Validate input parameters."""
        valid_types = ["integration", "uncertainty", "importance", "mcmc"]
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

        confidence_interval = (
            integral_estimate - margin_error,
            integral_estimate + margin_error,
        )

        # Convergence diagnostic
        batch_size = self.n_simulations // 10
        batch_estimates = []
        for i in range(0, self.n_simulations, batch_size):
            batch_values = function_values[i : i + batch_size]
            batch_estimate = (b - a) * np.mean(batch_values)
            batch_estimates.append(batch_estimate)

        convergence_diagnostic = {
            "batch_variance": np.var(batch_estimates),
            "relative_std_error": (
                standard_error / abs(integral_estimate)
                if integral_estimate != 0
                else float("inf")
            ),
        }

        result = MonteCarloResult(
            simulation_type="integration",
            n_simulations=self.n_simulations,
            simulation_results=function_values,
            estimated_value=integral_estimate,
            confidence_interval=confidence_interval,
            standard_error=standard_error,
            convergence_diagnostic=convergence_diagnostic,
            integration_bounds=bounds,
            function_info={"bounds": bounds},
            simulation_params={"random_state": self.random_state},
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
                    simulation_type="uncertainty",
                    n_simulations=self.n_simulations,
                    simulation_results=mc_samples,
                    estimated_value=mean_estimate,
                    confidence_interval=(lower, upper),
                    standard_error=standard_error,
                    simulation_params={
                        "fitted_distribution": "normal",
                        "parameters": {"mu": mu, "sigma": sigma},
                        "column": col,
                        "random_state": self.random_state,
                    },
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
                importance_samples = np.random.normal(
                    shifted_mu, sigma, self.n_simulations
                )

                # Calculate importance weights
                original_density = stats.norm.pdf(importance_samples, mu, sigma)
                importance_density = stats.norm.pdf(
                    importance_samples, shifted_mu, sigma
                )

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
                    min(1, rare_event_probability + margin_error),
                )

                result = MonteCarloResult(
                    simulation_type="importance_sampling",
                    n_simulations=self.n_simulations,
                    simulation_results=importance_samples,
                    estimated_value=rare_event_probability,
                    confidence_interval=confidence_interval,
                    standard_error=standard_error,
                    simulation_params={
                        "threshold": threshold,
                        "original_mu": mu,
                        "original_sigma": sigma,
                        "importance_mu": shifted_mu,
                        "column": col,
                        "random_state": self.random_state,
                    },
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
                    current_log_likelihood = (
                        -0.5 * n_data * np.log(2 * np.pi * data_sigma**2)
                        - 0.5 * np.sum((col_data - current_mu) ** 2) / data_sigma**2
                    )

                    proposed_log_likelihood = (
                        -0.5 * n_data * np.log(2 * np.pi * data_sigma**2)
                        - 0.5 * np.sum((col_data - proposed_mu) ** 2) / data_sigma**2
                    )

                    # Add prior
                    current_log_prior = (
                        -0.5 * (current_mu - prior_mu) ** 2 / prior_sigma**2
                    )
                    proposed_log_prior = (
                        -0.5 * (proposed_mu - prior_mu) ** 2 / prior_sigma**2
                    )

                    # Acceptance probability
                    log_alpha = (proposed_log_likelihood + proposed_log_prior) - (
                        current_log_likelihood + current_log_prior
                    )

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
                effective_sample_size = len(chain_post_burnin) / (
                    1 + 2 * self._autocorrelation_time(chain_post_burnin)
                )

                result = MonteCarloResult(
                    simulation_type="mcmc",
                    n_simulations=self.n_simulations,
                    simulation_results=chain_post_burnin,
                    estimated_value=posterior_mean,
                    confidence_interval=(lower, upper),
                    standard_error=posterior_std / np.sqrt(len(chain_post_burnin)),
                    convergence_diagnostic={
                        "acceptance_rate": acceptance_rate,
                        "effective_sample_size": effective_sample_size,
                        "burn_in_length": burn_in,
                    },
                    simulation_params={
                        "column": col,
                        "prior_mu": prior_mu,
                        "prior_sigma": prior_sigma,
                        "data_sigma": data_sigma,
                        "proposal_sigma": proposal_sigma,
                        "random_state": self.random_state,
                    },
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
            autocorr = np.correlate(chain_centered, chain_centered, mode="full")
            autocorr = autocorr[len(autocorr) // 2 :]
            autocorr = autocorr / autocorr[0]

            # Find first time autocorr drops below 1/e
            threshold = 1 / np.e
            for tau in range(1, min(len(autocorr), max_lag)):
                if autocorr[tau] < threshold:
                    return tau

            return max_lag

        except:
            return 1.0  # Default if calculation fails
