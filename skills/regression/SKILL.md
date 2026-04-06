---
name: regression
description: Build and evaluate regression models to predict a target variable. Use when modeling relationships between features and an outcome.
allowed-tools: mcp__localdata__describe_database mcp__localdata__execute_query mcp__localdata__analyze_regression mcp__localdata__evaluate_model_performance
argument-hint: "<database-name> <target-column>"
---

# Regression

Build, evaluate, and compare regression models to predict a target variable from features.

## Steps

1. **Parse arguments.** Extract the database name and target column from `$ARGUMENTS`. The first argument is the database name, the second is the target variable to predict.

2. **Explore available features.** Call `describe_database` with the database name. Identify numeric columns as potential predictors. Note the target column's distribution (continuous vs discrete) to confirm regression is appropriate.

3. **Extract and inspect data.** Call `execute_query` to select the target column and candidate feature columns. Check for nulls, outliers, and sufficient row count (at least 30 observations per feature as a rule of thumb).

4. **Build a linear baseline.** Call `analyze_regression` with the database name, target column, feature columns, and model type "linear". Review the results: R-squared, adjusted R-squared, coefficients, p-values, and residual diagnostics.

5. **Evaluate baseline performance.** Call `evaluate_model_performance` to get detailed metrics: RMSE, MAE, R-squared, and residual analysis. Assess whether the linear model captures the relationship adequately.

6. **Interpret baseline results.** Check for:
   - R-squared below 0.3: poor fit, consider non-linear models
   - Features with p-values above 0.05: candidates for removal
   - Multicollinearity: correlated predictors inflating variance
   - Residual patterns: non-random residuals suggest model misspecification

7. **Try alternative models if needed.** If the linear baseline performs poorly (R-squared below 0.5 or patterned residuals), call `analyze_regression` with alternative model types:
   - "ridge" for multicollinearity issues
   - "lasso" for feature selection (many predictors)
   - "polynomial" for curved relationships

8. **Compare models.** For each model fitted, compare R-squared, RMSE, and MAE. Select the model with the best balance of performance and interpretability. Simpler models are preferred when performance is comparable.

9. **Present the final model.** Report:
   - Selected model type and rationale
   - Key performance metrics (R-squared, RMSE, MAE)
   - Most important features and their coefficients
   - Practical interpretation of the model equation
   - Limitations and assumptions

10. **Recommend next steps.** Suggest validating on held-out data, running `/localdata-mcp:analyze-correlations` to find additional predictors, or using `/localdata-mcp:cluster-analysis` to identify subgroups where the model performs differently.
