"""MCP tool functions for data science analysis.

Pure functions that accept a SQLAlchemy engine and parameters, execute queries
to obtain DataFrames, and delegate to domain modules for analysis. Results are
returned as JSON-serializable dicts.

Follows the same pattern as :mod:`graph_tools` and :mod:`tree_tools`.
"""

from typing import Any, Dict, List, Optional

import numpy as np
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


def _as_dict(result: Any) -> Dict[str, Any]:
    """Normalise a domain result object into the JSON-serializable dict MCP returns.

    Some domains return plain dicts, others a dataclass result object. Tools must
    hand back a mapping either way.
    """
    if isinstance(result, dict):
        return result
    if hasattr(result, "to_dict"):
        return result.to_dict()
    return dict(vars(result))


def _require_columns(df: pd.DataFrame, *columns: Optional[str]) -> None:
    """Raise a clear error naming any requested column the query did not return."""
    missing = [c for c in columns if c and c not in df.columns]
    if missing:
        raise ValueError(
            f"Column(s) {missing} not found in query results. "
            f"Available columns: {list(df.columns)}"
        )


def _subset(df: pd.DataFrame, *columns: Optional[str]) -> pd.DataFrame:
    """Narrow a frame to the requested columns.

    The statistical transformers take no column arguments — they infer the value
    and grouping columns from the frame's dtypes. Selecting the caller's columns
    here is therefore how a column choice is honoured; forwarding the names as
    keywords reaches the transformer constructors and raises (issue #23).
    """
    selected = [c for c in columns if c]
    if not selected:
        return df
    _require_columns(df, *selected)
    return df[selected]


def _numeric_matrix(
    df: pd.DataFrame, columns: Optional[List[str]] = None
) -> np.ndarray:
    """Build the 2-D float array the pattern-recognition functions expect as ``X``.

    Those functions take a positional ``X: np.ndarray``, not a DataFrame. When no
    columns are named, every numeric column is used and non-numeric ones are
    dropped rather than crashing the underlying estimator on a string value.
    """
    if columns:
        _require_columns(df, *columns)
        frame = df[columns]
    else:
        frame = df.select_dtypes(include=[np.number])

    if frame.empty or frame.shape[1] == 0:
        raise ValueError(
            "No numeric columns available for analysis. "
            f"Query returned columns: {list(df.columns)}"
        )

    frame = frame.apply(pd.to_numeric, errors="coerce").dropna()
    if frame.empty:
        raise ValueError(
            "No complete numeric rows remain after dropping missing values."
        )
    return frame.to_numpy(dtype=float)


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
    return run_hypothesis_test(
        data=_subset(df, column, group_column),
        test_type=test_type,
        alpha=alpha,
        alternative=alternative,
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
        data=_subset(df, dependent_var, group_var),
        anova_type="one_way",
        alpha=alpha,
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
    return calculate_effect_sizes(data=_subset(df, column, group_column))


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
    _require_columns(df, target_column)

    if feature_columns:
        _require_columns(df, *feature_columns)
    else:
        # Without an explicit choice, use every other numeric column. Passing the
        # whole frame lets a text column reach the estimator, which fails with an
        # opaque "could not convert string to float" (issue #23).
        feature_columns = [
            c
            for c in df.select_dtypes(include=[np.number]).columns
            if c != target_column
        ]
        if not feature_columns:
            raise ValueError(
                f"No numeric feature columns found besides '{target_column}'. "
                f"Query returned columns: {list(df.columns)}"
            )

    kwargs: Dict[str, Any] = {}
    if regularization:
        kwargs["regularization"] = regularization
    return _as_dict(
        fit_regression_model(
            data=df[[*feature_columns, target_column]],
            target_column=target_column,
            model_type=model_type,
            feature_columns=feature_columns,
            **kwargs,
        )
    )


def tool_evaluate_model(
    engine: Engine,
    query: str,
    target_column: str,
    prediction_column: str,
    model_type: str = "regression",
    max_rows: Optional[int] = None,
) -> Dict[str, Any]:
    """Score predictions that are already stored alongside their actual values.

    The domain's ``evaluate_model_performance`` scores a *fitted estimator*
    against held-out arrays. This tool has no estimator to hand it — its inputs
    are two columns of a query result — so the metrics are computed here from
    the actual/predicted pair, which is what the tool has always claimed to do.
    """
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        mean_absolute_error,
        mean_squared_error,
        precision_score,
        r2_score,
        recall_score,
    )

    df = _query_to_dataframe(engine, query, max_rows)
    _require_columns(df, target_column, prediction_column)

    paired = df[[target_column, prediction_column]].dropna()
    if paired.empty:
        raise ValueError(
            f"No rows with both '{target_column}' and '{prediction_column}' present."
        )
    y_true = paired[target_column]
    y_pred = paired[prediction_column]

    if model_type == "classification":
        return {
            "model_type": "classification",
            "n_samples": int(len(paired)),
            "target_column": target_column,
            "prediction_column": prediction_column,
            "metrics": {
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "precision": float(
                    precision_score(y_true, y_pred, average="weighted", zero_division=0)
                ),
                "recall": float(
                    recall_score(y_true, y_pred, average="weighted", zero_division=0)
                ),
                "f1": float(
                    f1_score(y_true, y_pred, average="weighted", zero_division=0)
                ),
            },
        }

    if model_type != "regression":
        raise ValueError(
            f"Unknown model_type: {model_type}. Use 'regression' or 'classification'."
        )

    y_true = pd.to_numeric(y_true, errors="coerce")
    y_pred = pd.to_numeric(y_pred, errors="coerce")
    numeric = pd.concat([y_true, y_pred], axis=1).dropna()
    if numeric.empty:
        raise ValueError(
            f"Columns '{target_column}' and '{prediction_column}' are not numeric; "
            "regression evaluation needs numeric values."
        )
    y_true, y_pred = numeric.iloc[:, 0], numeric.iloc[:, 1]

    mse = float(mean_squared_error(y_true, y_pred))
    residuals = y_true - y_pred
    return {
        "model_type": "regression",
        "n_samples": int(len(numeric)),
        "target_column": target_column,
        "prediction_column": prediction_column,
        "metrics": {
            "r2": float(r2_score(y_true, y_pred)),
            "mse": mse,
            "rmse": float(np.sqrt(mse)),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "mean_residual": float(residuals.mean()),
            "residual_std": float(residuals.std()),
        },
    }


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
    return perform_clustering(
        _numeric_matrix(df, columns),
        algorithm=method,
        n_clusters=n_clusters,
    )


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
    return detect_anomalies(
        _numeric_matrix(df, columns),
        algorithm=method,
        contamination=contamination,
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
    return reduce_dimensions(
        _numeric_matrix(df, columns),
        algorithm=method,
        n_components=n_components,
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
    _require_columns(df, date_column, value_column)
    kwargs: Dict[str, Any] = {}
    if frequency:
        # The analyzer names this parameter `freq`, not `frequency`.
        kwargs["freq"] = frequency
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
    _require_columns(df, date_column, value_column)
    if method == "arima":
        return forecast_arima(
            data=df,
            date_column=date_column,
            value_column=value_column,
            forecast_steps=horizon,
        )
    elif method in ("ets", "exponential_smoothing"):
        return forecast_exponential_smoothing(
            data=df,
            date_column=date_column,
            value_column=value_column,
            forecast_steps=horizon,
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
    _require_columns(df, customer_column, date_column, value_column)
    result = analyze_rfm(
        data=df,
        customer_column=customer_column,
        date_column=date_column,
        # The domain function calls the monetary column `amount_column`.
        amount_column=value_column,
    )
    return _as_dict(result)


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
    _require_columns(df, variant_column, metric_column)
    result = perform_ab_test(
        data=df,
        # The domain function names these `group_column` and `outcome_column`.
        group_column=variant_column,
        outcome_column=metric_column,
        alpha=alpha,
    )
    return _as_dict(result)
