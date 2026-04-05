"""
Sampling & Estimation Domain - High-level MCP tool functions.

Convenience functions that wrap the transformer classes for direct use
as MCP tool endpoints.
"""

from typing import Any, Callable, Dict, Union

import pandas as pd

from ._sampling import SamplingTransformer
from ._bootstrap import BootstrapTransformer
from ._monte_carlo import MonteCarloTransformer
from ._bayesian import BayesianEstimationTransformer


def generate_sample(
    data: Union[pd.DataFrame, str],
    sampling_method: str = "simple_random",
    sample_size: Union[int, float] = 0.1,
    **kwargs,
) -> Dict[str, Any]:
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
        if data.endswith(".csv"):
            data = pd.read_csv(data)
        elif data.endswith(".json"):
            data = pd.read_json(data)
        else:
            raise ValueError("Unsupported file format")

    # Initialize and run sampling transformer
    transformer = SamplingTransformer(
        sampling_method=sampling_method, sample_size=sample_size, **kwargs
    )

    # Fit and transform
    transformer.fit(data)
    sample_data = transformer.transform(data)

    # Return both the sample and the results
    return {
        "sample_data": sample_data.to_dict("records")
        if hasattr(sample_data, "to_dict")
        else sample_data,
        "sampling_results": transformer.sampling_result_.to_dict(),
    }


def bootstrap_statistic(
    data: Union[pd.DataFrame, str],
    statistic_func: Union[Callable, str] = "mean",
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    **kwargs,
) -> Dict[str, Any]:
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
        if data.endswith(".csv"):
            data = pd.read_csv(data)
        elif data.endswith(".json"):
            data = pd.read_json(data)
        else:
            raise ValueError("Unsupported file format")

    # Initialize and run bootstrap transformer
    transformer = BootstrapTransformer(
        statistic_func=statistic_func,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        **kwargs,
    )

    # Fit and transform
    transformer.fit(data)
    result_df = transformer.transform(data)

    return result_df.iloc[0].to_dict()


def monte_carlo_simulate(
    data: Union[pd.DataFrame, str],
    simulation_type: str = "integration",
    n_simulations: int = 10000,
    **kwargs,
) -> Dict[str, Any]:
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
        if data.endswith(".csv"):
            data = pd.read_csv(data)
        elif data.endswith(".json"):
            data = pd.read_json(data)
        else:
            raise ValueError("Unsupported file format")

    # Initialize and run Monte Carlo transformer
    transformer = MonteCarloTransformer(
        simulation_type=simulation_type, n_simulations=n_simulations, **kwargs
    )

    # Fit and transform
    transformer.fit(data)
    result_df = transformer.transform(data)

    return result_df.iloc[0].to_dict()


def bayesian_estimate(
    data: Union[pd.DataFrame, str],
    estimation_type: str = "posterior",
    prior_distribution: str = "normal",
    confidence_level: float = 0.95,
    **kwargs,
) -> Dict[str, Any]:
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
        if data.endswith(".csv"):
            data = pd.read_csv(data)
        elif data.endswith(".json"):
            data = pd.read_json(data)
        else:
            raise ValueError("Unsupported file format")

    # Initialize and run Bayesian transformer
    transformer = BayesianEstimationTransformer(
        estimation_type=estimation_type,
        prior_distribution=prior_distribution,
        confidence_level=confidence_level,
        **kwargs,
    )

    # Fit and transform
    transformer.fit(data)
    result_df = transformer.transform(data)

    return result_df.iloc[0].to_dict()
