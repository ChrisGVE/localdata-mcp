"""
Sampling & Estimation Domain - BayesianEstimationTransformer.

sklearn-compatible transformer for Bayesian estimation and inference using
conjugate priors, credible intervals, and basic model comparison.
"""

from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from ._results import BayesianResult, get_logger

logger = get_logger(__name__)


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

    def __init__(
        self,
        estimation_type: str = "posterior",
        prior_distribution: str = "normal",
        prior_params: Optional[Dict[str, float]] = None,
        confidence_level: float = 0.95,
        n_samples: int = 10000,
        random_state: Optional[int] = None,
    ):
        self.estimation_type = estimation_type
        self.prior_distribution = prior_distribution
        self.prior_params = prior_params or {}
        self.confidence_level = confidence_level
        self.n_samples = n_samples
        self.random_state = random_state

    def fit(self, X, y=None):
        """Fit the transformer (no-op for Bayesian estimation)."""
        self._validate_parameters()
        self.is_fitted_ = True
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
                    if self.estimation_type == "posterior":
                        result = self._estimate_posterior(col_data, col)
                    elif self.estimation_type == "credible_interval":
                        result = self._calculate_credible_intervals(col_data, col)
                    elif self.estimation_type == "model_comparison":
                        result = self._compare_models(col_data, col)
                    else:
                        result = self._estimate_posterior(col_data, col)

                    self.bayesian_results_.append(result)
                except Exception as e:
                    logger.warning(f"Bayesian estimation failed for {col}: {e}")

        # Create result summary
        result_summary = {
            "bayesian_results": [result.to_dict() for result in self.bayesian_results_],
            "estimation_type": self.estimation_type,
            "prior_distribution": self.prior_distribution,
            "prior_params": self.prior_params,
            "confidence_level": self.confidence_level,
        }

        return pd.DataFrame([result_summary])

    def _validate_parameters(self):
        """Validate input parameters."""
        valid_types = ["posterior", "credible_interval", "model_comparison"]
        if self.estimation_type not in valid_types:
            raise ValueError(f"estimation_type must be one of {valid_types}")

        valid_priors = ["normal", "gamma", "beta", "uniform"]
        if self.prior_distribution not in valid_priors:
            raise ValueError(f"prior_distribution must be one of {valid_priors}")

    def _estimate_posterior(self, data: pd.Series, col_name: str) -> BayesianResult:
        """Estimate posterior distribution using conjugate priors."""
        data_values = data.values
        n = len(data_values)

        if self.prior_distribution == "normal":
            # Normal-Normal conjugate prior for mean estimation
            # Assume known variance for simplicity
            data_var = np.var(data_values)
            data_mean = np.mean(data_values)

            # Prior parameters
            prior_mu = self.prior_params.get("mu", 0.0)
            prior_sigma2 = self.prior_params.get("sigma2", 100.0)

            # Posterior parameters (conjugate update)
            posterior_precision = 1 / prior_sigma2 + n / data_var
            posterior_sigma2 = 1 / posterior_precision
            posterior_mu = (
                prior_mu / prior_sigma2 + n * data_mean / data_var
            ) / posterior_precision

            # Generate posterior samples
            np.random.seed(self.random_state)
            posterior_samples = np.random.normal(
                posterior_mu, np.sqrt(posterior_sigma2), self.n_samples
            )

            # Calculate credible intervals
            alpha = 1 - self.confidence_level
            lower = np.percentile(posterior_samples, 100 * alpha / 2)
            upper = np.percentile(posterior_samples, 100 * (1 - alpha / 2))

            return BayesianResult(
                parameter_name=f"mean_{col_name}",
                estimation_method="normal_conjugate",
                posterior_samples=posterior_samples,
                posterior_mean=posterior_mu,
                posterior_median=np.median(posterior_samples),
                credible_intervals={f"{self.confidence_level:.0%}": (lower, upper)},
                prior_info={
                    "distribution": "normal",
                    "prior_mu": prior_mu,
                    "prior_sigma2": prior_sigma2,
                    "posterior_mu": posterior_mu,
                    "posterior_sigma2": posterior_sigma2,
                },
            )

        elif self.prior_distribution == "gamma":
            # Gamma-Normal conjugate prior for precision estimation
            # Estimating precision (inverse variance) of normal data

            # Prior parameters for Gamma distribution
            prior_alpha = self.prior_params.get("alpha", 1.0)
            prior_beta = self.prior_params.get("beta", 1.0)

            # Sufficient statistics
            sum_x = np.sum(data_values)
            sum_x2 = np.sum(data_values**2)
            data_mean = np.mean(data_values)

            # Posterior parameters (assuming known mean)
            posterior_alpha = prior_alpha + n / 2
            posterior_beta = prior_beta + 0.5 * np.sum((data_values - data_mean) ** 2)

            # Generate posterior samples for precision
            np.random.seed(self.random_state)
            precision_samples = np.random.gamma(
                posterior_alpha, 1 / posterior_beta, self.n_samples
            )
            variance_samples = 1 / precision_samples

            # Calculate credible intervals for variance
            alpha = 1 - self.confidence_level
            lower = np.percentile(variance_samples, 100 * alpha / 2)
            upper = np.percentile(variance_samples, 100 * (1 - alpha / 2))

            return BayesianResult(
                parameter_name=f"variance_{col_name}",
                estimation_method="gamma_conjugate",
                posterior_samples=variance_samples,
                posterior_mean=np.mean(variance_samples),
                posterior_median=np.median(variance_samples),
                credible_intervals={f"{self.confidence_level:.0%}": (lower, upper)},
                prior_info={
                    "distribution": "gamma",
                    "prior_alpha": prior_alpha,
                    "prior_beta": prior_beta,
                    "posterior_alpha": posterior_alpha,
                    "posterior_beta": posterior_beta,
                },
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
            estimation_method="approximate",
            posterior_samples=posterior_samples,
            posterior_mean=data_mean,
            posterior_median=np.median(posterior_samples),
            credible_intervals={f"{self.confidence_level:.0%}": (lower, upper)},
            prior_info={
                "distribution": self.prior_distribution,
                "method": "approximate",
            },
        )

    def _calculate_credible_intervals(
        self, data: pd.Series, col_name: str
    ) -> BayesianResult:
        """Calculate credible intervals for multiple confidence levels."""
        # First get posterior
        posterior_result = self._estimate_posterior(data, col_name)

        # Calculate multiple credible intervals
        credible_levels = [0.50, 0.90, 0.95, 0.99]
        credible_intervals = {}

        for level in credible_levels:
            alpha = 1 - level
            lower = np.percentile(posterior_result.posterior_samples, 100 * alpha / 2)
            upper = np.percentile(
                posterior_result.posterior_samples, 100 * (1 - alpha / 2)
            )
            credible_intervals[f"{level:.0%}"] = (lower, upper)

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
            log_likelihood_exp = np.sum(
                stats.expon.logpdf(data_values, scale=1 / lambda_ml)
            )
        else:
            log_likelihood_exp = -np.inf

        # Simple Bayes factor approximation (assuming equal priors)
        # In practice, this should include proper prior specification and integration
        bayes_factor = np.exp(log_likelihood_normal - log_likelihood_exp)

        # Choose better model
        if bayes_factor > 1:
            better_model = "normal"
            model_params = {"mu": mu_ml, "sigma": sigma_ml}
        else:
            better_model = "exponential"
            model_params = {"lambda": lambda_ml}

        # Generate posterior samples for the better model
        np.random.seed(self.random_state)
        if better_model == "normal":
            posterior_samples = np.random.normal(
                mu_ml, sigma_ml / np.sqrt(len(data_values)), self.n_samples
            )
            posterior_mean = mu_ml
        else:
            # For exponential, sample the rate parameter
            # Using gamma posterior (conjugate)
            prior_alpha = 1.0  # Default prior
            prior_beta = 1.0
            posterior_alpha = prior_alpha + len(data_values)
            posterior_beta = prior_beta + np.sum(data_values)

            rate_samples = np.random.gamma(
                posterior_alpha, 1 / posterior_beta, self.n_samples
            )
            posterior_samples = 1 / rate_samples  # Mean of exponential
            posterior_mean = np.mean(posterior_samples)

        # Calculate credible intervals
        alpha = 1 - self.confidence_level
        lower = np.percentile(posterior_samples, 100 * alpha / 2)
        upper = np.percentile(posterior_samples, 100 * (1 - alpha / 2))

        return BayesianResult(
            parameter_name=f"parameter_{col_name}",
            estimation_method="model_comparison",
            posterior_samples=posterior_samples,
            posterior_mean=posterior_mean,
            posterior_median=np.median(posterior_samples),
            credible_intervals={f"{self.confidence_level:.0%}": (lower, upper)},
            bayes_factor=bayes_factor,
            prior_info={
                "compared_models": ["normal", "exponential"],
                "selected_model": better_model,
                "model_params": model_params,
                "log_likelihood_normal": log_likelihood_normal,
                "log_likelihood_exponential": log_likelihood_exp,
            },
        )
