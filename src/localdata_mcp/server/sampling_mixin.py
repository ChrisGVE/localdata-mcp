"""MCP tool methods for the sampling and estimation domain.

These are the thin wrappers registered as MCP tools: resolve the named
connection, delegate to the adapter in :mod:`localdata_mcp.sampling_tools`, and
serialize the result. They live in a mixin rather than in ``database_manager``
so that module does not keep growing as domains are registered.
"""

from typing import List, Optional

from ..json_utils import safe_dumps
from ..sampling_tools import (
    tool_bayesian_estimate,
    tool_bootstrap_statistic,
    tool_generate_sample,
    tool_monte_carlo_simulate,
)


class SamplingToolsMixin:
    """Sampling and estimation MCP tools, mixed into ``DatabaseManager``."""

    def generate_sample(
        self,
        connection_name: str,
        query: str,
        sampling_method: str = "simple_random",
        sample_size: float = 0.1,
        columns: Optional[List[str]] = None,
        stratify_column: str = "",
    ) -> str:
        """Draw a representative sample from query results.

        Args:
            connection_name: Name of the connected database.
            query: SQL query returning the population to sample from.
            sampling_method: 'simple_random', 'stratified', 'systematic', or 'cluster'.
            sample_size: Row count when >= 1, or a fraction of the population when < 1.
            columns: Restrict the sample to these columns (default: all).
            stratify_column: Column defining strata for stratified sampling.
        """
        engine = self._get_connection(connection_name)
        result = tool_generate_sample(
            engine,
            query,
            sampling_method=sampling_method,
            sample_size=sample_size,
            columns=columns,
            stratify_column=stratify_column or None,
        )
        return safe_dumps(result)

    def bootstrap_statistic(
        self,
        connection_name: str,
        query: str,
        column: str = "",
        statistic: str = "mean",
        n_bootstrap: int = 1000,
        confidence_level: float = 0.95,
    ) -> str:
        """Estimate a statistic's sampling distribution and confidence interval.

        Args:
            connection_name: Name of the connected database.
            query: SQL query returning data to resample.
            column: Column to compute the statistic on (default: all numeric columns).
            statistic: 'mean', 'median', 'std', or 'var'.
            n_bootstrap: Number of bootstrap resamples (default 1000).
            confidence_level: Confidence level for the interval (default 0.95).
        """
        engine = self._get_connection(connection_name)
        result = tool_bootstrap_statistic(
            engine,
            query,
            column=column or None,
            statistic=statistic,
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level,
        )
        return safe_dumps(result)

    def monte_carlo_simulate(
        self,
        connection_name: str,
        query: str,
        simulation_type: str = "integration",
        n_simulations: int = 10000,
        columns: Optional[List[str]] = None,
    ) -> str:
        """Run a Monte Carlo simulation parameterised by query results.

        Args:
            connection_name: Name of the connected database.
            query: SQL query returning the data the simulation draws on.
            simulation_type: 'integration', 'optimization', or 'risk_analysis'.
            n_simulations: Number of simulation draws (default 10000).
            columns: Restrict the simulation to these columns (default: all).
        """
        engine = self._get_connection(connection_name)
        result = tool_monte_carlo_simulate(
            engine,
            query,
            simulation_type=simulation_type,
            n_simulations=n_simulations,
            columns=columns,
        )
        return safe_dumps(result)

    def bayesian_estimate(
        self,
        connection_name: str,
        query: str,
        column: str = "",
        estimation_type: str = "posterior",
        prior_distribution: str = "normal",
        confidence_level: float = 0.95,
    ) -> str:
        """Estimate a posterior distribution and credible interval from query results.

        Args:
            connection_name: Name of the connected database.
            query: SQL query returning the observed data.
            column: Column to estimate on (default: all numeric columns).
            estimation_type: 'posterior', 'credible_interval', or 'model_comparison'.
            prior_distribution: 'normal', 'beta', 'gamma', or 'uniform'.
            confidence_level: Credible interval level (default 0.95).
        """
        engine = self._get_connection(connection_name)
        result = tool_bayesian_estimate(
            engine,
            query,
            column=column or None,
            estimation_type=estimation_type,
            prior_distribution=prior_distribution,
            confidence_level=confidence_level,
        )
        return safe_dumps(result)
