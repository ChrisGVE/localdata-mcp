<!-- MACHINE-WRITTEN by localdata_mcp.nexus.contract.generators.docs — DO NOT EDIT; regenerate via `python -m localdata_mcp.nexus.contract.generate` -->

# Tools — regression_modeling

| Tool | Summary | Input shape | Output shape | Streaming | Params |
|---|---|---|---|---|---|
| `analyze_regression` | Fit a regression model on an addressed tabular source: model_type linear (default), ridge, lasso, elastic_net, logistic, or polynomial (regularization l1/l2/elastic_net maps onto the penalised estimators). Reports coefficients, fit metrics, and design-matrix health (rank, condition number); algorithm_params passes tuning straight to the estimator. | TABULAR | FITTED_MODEL | no | `endpoint?`, `path?`, `table?`, `query?`, `target_column`, `feature_columns?`, `model_type?`, `regularization?`, `degree?`, `algorithm_params?` |
| `evaluate_model_performance` | Score stored predictions against actuals on an addressed tabular source: regression metrics (r2, mse, rmse, mae, residual summary) for a numeric pair, weighted classification metrics (accuracy, precision, recall, f1) on request. | TABULAR | SCALAR | no | `endpoint?`, `path?`, `table?`, `query?`, `target_column`, `prediction_column`, `model_type?` |
