"""MCP tool functions for data science analysis.

Pure functions that accept a SQLAlchemy engine and parameters, execute queries
to obtain DataFrames, and delegate to domain modules for analysis. Results are
returned as JSON-serializable dicts.

Follows the same pattern as :mod:`graph_tools` and :mod:`tree_tools`.
"""

from typing import Any, Dict, List, Optional

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from .json_utils import safe_dumps
from .logging_manager import get_logger

logger = get_logger(__name__)

# Maximum rows to load into memory for data science operations
MAX_ANALYSIS_ROWS = 500_000


def _query_to_dataframe(
    engine: Engine,
    query: str,
    max_rows: Optional[int] = None,
) -> pd.DataFrame:
    """Execute a SQL query and return results as a DataFrame.

    Parameters
    ----------
    engine : Engine
        SQLAlchemy engine from an active connection.
    query : str
        SQL query to execute.
    max_rows : int, optional
        Maximum rows to return. Defaults to MAX_ANALYSIS_ROWS.

    Returns
    -------
    pd.DataFrame
    """
    limit = max_rows or MAX_ANALYSIS_ROWS
    with engine.connect() as conn:
        df = pd.read_sql(text(query), conn)
    if len(df) > limit:
        logger.warning(
            "Query returned %d rows, truncating to %d for analysis",
            len(df),
            limit,
        )
        df = df.head(limit)
    return df


# ---------------------------------------------------------------------------
# Statistical Analysis tools
# ---------------------------------------------------------------------------


def tool_hypothesis_test(
    engine: Engine,
    query: str,
    test_type: str = "auto",
    column: Optional[str] = None,
    group_column: Optional[str] = None,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    max_rows: Optional[int] = None,
) -> Dict[str, Any]:
    """Run hypothesis test on query results."""
    from .domains.statistical_analysis import run_hypothesis_test

    df = _query_to_dataframe(engine, query, max_rows)
    kwargs: Dict[str, Any] = {}
    if column:
        kwargs["column"] = column
    if group_column:
        kwargs["group_column"] = group_column
    return run_hypothesis_test(
        data=df, test_type=test_type, alpha=alpha, alternative=alternative, **kwargs
    )


def tool_anova_analysis(
    engine: Engine,
    query: str,
    dependent_var: str,
    group_var: str,
    alpha: float = 0.05,
    max_rows: Optional[int] = None,
) -> Dict[str, Any]:
    """Perform ANOVA analysis on query results."""
    from .domains.statistical_analysis import perform_anova

    df = _query_to_dataframe(engine, query, max_rows)
    return perform_anova(
        data=df, dependent_var=dependent_var, group_var=group_var, alpha=alpha
    )


def tool_effect_sizes(
    engine: Engine,
    query: str,
    column: str,
    group_column: str,
    max_rows: Optional[int] = None,
) -> Dict[str, Any]:
    """Calculate effect sizes for group comparisons."""
    from .domains.statistical_analysis import calculate_effect_sizes

    df = _query_to_dataframe(engine, query, max_rows)
    return calculate_effect_sizes(data=df, column=column, group_column=group_column)


# ---------------------------------------------------------------------------
# Regression & Modeling tools
# ---------------------------------------------------------------------------


def tool_fit_regression(
    engine: Engine,
    query: str,
    target_column: str,
    feature_columns: Optional[List[str]] = None,
    model_type: str = "linear",
    regularization: Optional[str] = None,
    max_rows: Optional[int] = None,
) -> Dict[str, Any]:
    """Fit a regression model on query results."""
    from .domains.regression_modeling import fit_regression_model

    df = _query_to_dataframe(engine, query, max_rows)
    kwargs: Dict[str, Any] = {}
    if feature_columns:
        kwargs["feature_columns"] = feature_columns
    if regularization:
        kwargs["regularization"] = regularization
    return fit_regression_model(
        data=df, target_column=target_column, model_type=model_type, **kwargs
    )


def tool_evaluate_model(
    engine: Engine,
    query: str,
    target_column: str,
    prediction_column: str,
    model_type: str = "regression",
    max_rows: Optional[int] = None,
) -> Dict[str, Any]:
    """Evaluate model performance on query results."""
    from .domains.regression_modeling import evaluate_model_performance

    df = _query_to_dataframe(engine, query, max_rows)
    return evaluate_model_performance(
        data=df,
        target_column=target_column,
        prediction_column=prediction_column,
        model_type=model_type,
    )


# ---------------------------------------------------------------------------
# Pattern Recognition tools
# ---------------------------------------------------------------------------


def tool_clustering(
    engine: Engine,
    query: str,
    columns: Optional[List[str]] = None,
    method: str = "kmeans",
    n_clusters: Optional[int] = None,
    max_rows: Optional[int] = None,
) -> Dict[str, Any]:
    """Perform clustering analysis on query results."""
    from .domains.pattern_recognition import perform_clustering

    df = _query_to_dataframe(engine, query, max_rows)
    kwargs: Dict[str, Any] = {}
    if columns:
        kwargs["columns"] = columns
    if n_clusters is not None:
        kwargs["n_clusters"] = n_clusters
    return perform_clustering(data=df, method=method, **kwargs)


def tool_anomaly_detection(
    engine: Engine,
    query: str,
    columns: Optional[List[str]] = None,
    method: str = "isolation_forest",
    contamination: float = 0.1,
    max_rows: Optional[int] = None,
) -> Dict[str, Any]:
    """Detect anomalies in query results."""
    from .domains.pattern_recognition import detect_anomalies

    df = _query_to_dataframe(engine, query, max_rows)
    kwargs: Dict[str, Any] = {}
    if columns:
        kwargs["columns"] = columns
    return detect_anomalies(
        data=df, method=method, contamination=contamination, **kwargs
    )


def tool_dimensionality_reduction(
    engine: Engine,
    query: str,
    columns: Optional[List[str]] = None,
    method: str = "pca",
    n_components: int = 2,
    max_rows: Optional[int] = None,
) -> Dict[str, Any]:
    """Reduce dimensionality of query results."""
    from .domains.pattern_recognition import reduce_dimensions

    df = _query_to_dataframe(engine, query, max_rows)
    kwargs: Dict[str, Any] = {}
    if columns:
        kwargs["columns"] = columns
    return reduce_dimensions(
        data=df, method=method, n_components=n_components, **kwargs
    )


# ---------------------------------------------------------------------------
# Time Series tools
# ---------------------------------------------------------------------------


def tool_time_series_analysis(
    engine: Engine,
    query: str,
    date_column: str,
    value_column: str,
    frequency: Optional[str] = None,
    max_rows: Optional[int] = None,
) -> Dict[str, Any]:
    """Analyze time series data from query results."""
    from .domains.time_series import analyze_time_series_basic

    df = _query_to_dataframe(engine, query, max_rows)
    kwargs: Dict[str, Any] = {}
    if frequency:
        kwargs["frequency"] = frequency
    return analyze_time_series_basic(
        data=df, date_column=date_column, value_column=value_column, **kwargs
    )


def tool_time_series_forecast(
    engine: Engine,
    query: str,
    date_column: str,
    value_column: str,
    horizon: int = 10,
    method: str = "arima",
    max_rows: Optional[int] = None,
) -> Dict[str, Any]:
    """Generate time series forecasts from query results."""
    from .domains.time_series import forecast_arima, forecast_exponential_smoothing

    df = _query_to_dataframe(engine, query, max_rows)
    if method == "arima":
        return forecast_arima(
            data=df,
            date_column=date_column,
            value_column=value_column,
            forecast_horizon=horizon,
        )
    elif method in ("ets", "exponential_smoothing"):
        return forecast_exponential_smoothing(
            data=df,
            date_column=date_column,
            value_column=value_column,
            forecast_horizon=horizon,
        )
    else:
        raise ValueError(f"Unknown forecast method: {method}. Use 'arima' or 'ets'.")


# ---------------------------------------------------------------------------
# Business Intelligence tools
# ---------------------------------------------------------------------------


def tool_rfm_analysis(
    engine: Engine,
    query: str,
    customer_column: str,
    date_column: str,
    value_column: str,
    max_rows: Optional[int] = None,
) -> Dict[str, Any]:
    """Perform RFM customer segmentation on query results."""
    from .domains.business_intelligence import analyze_rfm

    df = _query_to_dataframe(engine, query, max_rows)
    return analyze_rfm(
        data=df,
        customer_column=customer_column,
        date_column=date_column,
        value_column=value_column,
    )


def tool_ab_test(
    engine: Engine,
    query: str,
    variant_column: str,
    metric_column: str,
    alpha: float = 0.05,
    max_rows: Optional[int] = None,
) -> Dict[str, Any]:
    """Analyze A/B test results from query data."""
    from .domains.business_intelligence import perform_ab_test

    df = _query_to_dataframe(engine, query, max_rows)
    return perform_ab_test(
        data=df,
        variant_column=variant_column,
        metric_column=metric_column,
        alpha=alpha,
    )
