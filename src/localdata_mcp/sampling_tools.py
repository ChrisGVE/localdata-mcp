"""MCP tool functions for sampling and estimation analysis.

Pure functions that accept a SQLAlchemy engine and parameters, execute queries
to obtain DataFrames, and delegate to :mod:`localdata_mcp.domains.sampling_estimation`.
Results are returned as JSON-serializable dicts.

Follows the same pattern as :mod:`datascience_tools`.
"""

from typing import Any, Dict, List, Optional

from sqlalchemy.engine import Engine

from .datascience_tools import _query_to_dataframe, _require_columns, _subset
from .logging_manager import get_logger

logger = get_logger(__name__)


def tool_generate_sample(
    engine: Engine,
    query: str,
    sampling_method: str = "simple_random",
    sample_size: float = 0.1,
    columns: Optional[List[str]] = None,
    stratify_column: Optional[str] = None,
    max_rows: Optional[int] = None,
) -> Dict[str, Any]:
    """Draw a sample from query results using a named sampling design."""
    from .domains.sampling_estimation import generate_sample

    df = _query_to_dataframe(engine, query, max_rows)
    if columns:
        _require_columns(df, *columns)
        df = df[columns]

    kwargs: Dict[str, Any] = {}
    if stratify_column:
        _require_columns(df, stratify_column)
        kwargs["stratify_column"] = stratify_column

    # An integer means "this many rows"; a fraction means "this share of them".
    size: Any = int(sample_size) if float(sample_size).is_integer() else sample_size
    return generate_sample(
        data=df, sampling_method=sampling_method, sample_size=size, **kwargs
    )


def tool_bootstrap_statistic(
    engine: Engine,
    query: str,
    column: Optional[str] = None,
    statistic: str = "mean",
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    max_rows: Optional[int] = None,
) -> Dict[str, Any]:
    """Bootstrap a statistic and its confidence interval from query results."""
    from .domains.sampling_estimation import bootstrap_statistic

    df = _query_to_dataframe(engine, query, max_rows)
    return bootstrap_statistic(
        data=_subset(df, column),
        statistic_func=statistic,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
    )


def tool_monte_carlo_simulate(
    engine: Engine,
    query: str,
    simulation_type: str = "integration",
    n_simulations: int = 10000,
    columns: Optional[List[str]] = None,
    max_rows: Optional[int] = None,
) -> Dict[str, Any]:
    """Run a Monte Carlo simulation seeded from query results."""
    from .domains.sampling_estimation import monte_carlo_simulate

    df = _query_to_dataframe(engine, query, max_rows)
    if columns:
        _require_columns(df, *columns)
        df = df[columns]
    return monte_carlo_simulate(
        data=df, simulation_type=simulation_type, n_simulations=n_simulations
    )


def tool_bayesian_estimate(
    engine: Engine,
    query: str,
    column: Optional[str] = None,
    estimation_type: str = "posterior",
    prior_distribution: str = "normal",
    confidence_level: float = 0.95,
    max_rows: Optional[int] = None,
) -> Dict[str, Any]:
    """Estimate a posterior (or credible interval) from query results."""
    from .domains.sampling_estimation import bayesian_estimate

    df = _query_to_dataframe(engine, query, max_rows)
    return bayesian_estimate(
        data=_subset(df, column),
        estimation_type=estimation_type,
        prior_distribution=prior_distribution,
        confidence_level=confidence_level,
    )
