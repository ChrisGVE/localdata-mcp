"""localdata_mcp/process/domains/regression_modeling/tools.py — E10.b ToolSpecs.

The regression family's two tools, carried by name from `main`
(DR GP2): `analyze_regression` (fit + fit-quality + the sentinel's
rank/condition keys; FR-306's algorithm_params passthrough) and
`evaluate_model_performance` (actual-vs-predicted scoring). Thin over
regression.py / evaluation.py with the X-2 addressing contract.
Neighbors: estimators.py builds; spec_modules.py rosters this module.
"""

from __future__ import annotations

from typing import Any

from localdata_mcp.nexus.contract.spec import Param, TypeShape, tool_spec

from ..support import addressed_frame, source_params
from .evaluation import evaluate_predictions
from .regression import fit_regression


@tool_spec(
    name="analyze_regression",
    summary=(
        "Fit a regression model on an addressed tabular source: "
        "model_type linear (default), ridge, lasso, elastic_net, "
        "logistic, or polynomial (regularization l1/l2/elastic_net maps "
        "onto the penalised estimators). Reports coefficients, fit "
        "metrics, and design-matrix health (rank, condition number); "
        "algorithm_params passes tuning straight to the estimator."
    ),
    params=(
        *source_params(),
        Param("target_column", str, "The numeric outcome column to fit."),
        Param(
            "feature_columns",
            list,
            "Feature columns (default: every other numeric column).",
            required=False,
        ),
        Param(
            "model_type",
            str,
            "linear (default), ridge, lasso, elastic_net, logistic, polynomial.",
            required=False,
        ),
        Param(
            "regularization",
            str,
            "Penalty spelling l1, l2, or elastic_net — maps onto the estimator.",
            required=False,
        ),
        Param(
            "degree",
            int,
            "Polynomial expansion degree (implementation default 2).",
            required=False,
        ),
        Param(
            "algorithm_params",
            dict,
            "Estimator constructor parameters, passed through verbatim (FR-306).",
            required=False,
        ),
    ),
    input_shape=TypeShape.TABULAR,
    output_shape=TypeShape.FITTED_MODEL,
    domain="process",
)
def analyze_regression(
    target_column: str,
    endpoint: str | None = None,
    path: str | None = None,
    table: str | None = None,
    query: str | None = None,
    **knobs: Any,
) -> Any:
    frame, source = addressed_frame(endpoint, path, table, query)
    result = fit_regression(frame, target_column, **knobs)
    result["source"] = source
    return result


@tool_spec(
    name="evaluate_model_performance",
    summary=(
        "Score stored predictions against actuals on an addressed "
        "tabular source: regression metrics (r2, mse, rmse, mae, "
        "residual summary) for a numeric pair, weighted classification "
        "metrics (accuracy, precision, recall, f1) on request."
    ),
    params=(
        *source_params(),
        Param("target_column", str, "The actual-values column."),
        Param("prediction_column", str, "The predicted-values column."),
        Param(
            "model_type",
            str,
            "regression (default) or classification.",
            required=False,
        ),
    ),
    input_shape=TypeShape.TABULAR,
    output_shape=TypeShape.SCALAR,
    domain="process",
)
def evaluate_model_performance(
    target_column: str,
    prediction_column: str,
    endpoint: str | None = None,
    path: str | None = None,
    table: str | None = None,
    query: str | None = None,
    **knobs: Any,
) -> Any:
    frame, source = addressed_frame(endpoint, path, table, query)
    result = evaluate_predictions(frame, target_column, prediction_column, **knobs)
    result["source"] = source
    return result
