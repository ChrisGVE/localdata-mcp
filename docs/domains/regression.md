# Regression and Modeling Domain

## Overview

The regression and modeling domain fits regression models, evaluates their performance, and diagnoses their residuals. Use it when you need to quantify the relationship between a continuous outcome and one or more predictor variables, or when you need to predict numeric values from a set of features.

**When to use this domain:**

- Estimating the effect of one or more variables on a continuous outcome
- Predicting numeric values from structured features
- Selecting the most informative features from a large feature set
- Checking whether model assumptions (linearity, homoscedasticity, independence) hold
- Comparing in-sample versus out-of-sample performance to detect overfitting

**Source:** `src/localdata_mcp/domains/regression_modeling/`

---

## Available Analyses

| Method | Class | Description |
|---|---|---|
| Ordinary least squares | `LinearRegressionTransformer` | Standard linear regression with full statistical diagnostics |
| Ridge regression | `RegularizedRegressionTransformer` | L2 regularisation; shrinks coefficients without eliminating them |
| Lasso regression | `RegularizedRegressionTransformer` | L1 regularisation; performs automatic feature selection |
| Elastic net | `RegularizedRegressionTransformer` | L1+L2 combination; balances Ridge and Lasso properties |
| Logistic regression | `LogisticRegressionTransformer` | Binary or multi-class classification |
| Polynomial regression | `PolynomialRegressionTransformer` | Non-linear relationships via polynomial feature expansion |
| Model-based feature selection | `FeatureSelectionTransformer` | Select features via Lasso coefficient shrinkage |
| Recursive feature elimination | `FeatureSelectionTransformer` | Iteratively remove least important features (RFE / RFECV) |
| Univariate feature selection | `FeatureSelectionTransformer` | F-statistic based selection (SelectKBest) |
| Residual normality tests | `ResidualAnalysisTransformer` | Shapiro-Wilk, Anderson-Darling, Jarque-Bera |
| Homoscedasticity tests | `ResidualAnalysisTransformer` | Breusch-Pagan and White tests |
| Autocorrelation test | `ResidualAnalysisTransformer` | Durbin-Watson statistic |
| Influence measures | `ResidualAnalysisTransformer` | Leverage, Cook's distance, studentised residuals |
| Cross-validation | `RegressionModelingPipeline` | K-fold R² and RMSE |

---

## MCP Tool Reference

The domain is reached through two MCP tools. Like every other analytical tool,
each takes the name of a live connection and a SQL query — there is no
data-frame parameter and no separate load step, and column parameters name
columns in the query's result set. The classes listed under *Available Analyses*
above are the internal implementation those tools call; they are not reachable
from an MCP client.

Full parameter tables for both live in the
[tools reference](../tools-reference.md#data-science-12-tools). This page covers
what each tool is for and when to reach for it.

### `analyze_regression`

Answers "how does this outcome depend on these predictors?" `target_column` is
the column being explained; `feature_columns` is a genuine list of column names
(`["sqft", "bedrooms"]`, not a comma-separated string) and defaults to every
other numeric column in the result set. `model_type` selects `linear` (default),
`ridge`, `lasso`, `elastic_net`, `logistic` or `polynomial`.

`regularization` names the penalty instead of the estimator: `l1` fits lasso,
`l2` fits ridge, and `elastic_net` fits elastic net. It is an alternative to
naming the model — reach for it when you are thinking in terms of the penalty
you want rather than the estimator that carries it. Combined with
`model_type="polynomial"` it penalises the expanded basis. Anything else raises,
as does a `regularization` that contradicts an explicitly chosen `model_type`.

Returns coefficients with their standard errors and p-values, R² and the other
fit statistics, and — for every model except logistic — a `residual_analysis`
block with the normality, homoscedasticity and influence diagnostics described
below. Use `model_type="logistic"` when the target is binary; the tool does not
infer that from the data.

Fitting and scoring are separate concerns here: the model is not persisted, so
there is nothing to call `predict` on afterwards.

### `evaluate_model_performance`

Answers "how good were these predictions?" It scores predictions that already
sit in the database next to their actual values — `target_column` holds the
truth, `prediction_column` the estimate — so write your model's output back
before calling it. `model_type` is `regression` (default) or `classification`
and decides the metric set; there is no `metric_type` parameter.

Because it compares two stored columns, it works on predictions from any source,
including models fitted outside LocalData.

---

## Method Details

### Linear Regression (OLS)

Fits ordinary least squares with `sklearn.linear_model.LinearRegression` and computes full statistical diagnostics via `statsmodels`.

Outputs include:
- Coefficients with standard errors, t-statistics, and p-values for each feature
- R² and adjusted R²
- F-statistic for overall model significance
- AIC and BIC for model comparison

**Key parameters of `LinearRegressionTransformer`:**

| Parameter | Default | Description |
|---|---|---|
| `fit_intercept` | `True` | Include intercept term |
| `include_diagnostics` | `True` | Run full statsmodels diagnostics |
| `alpha` | `0.05` | Significance level for tests |

---

### Regularised Regression

All three variants use cross-validation to select the optimal regularisation strength when `alpha="auto"` (default).

**Ridge**: Penalises the sum of squared coefficients (L2). All features remain in the model; coefficients shrink toward zero. Use when you want to reduce variance without eliminating predictors.

**Lasso**: Penalises the sum of absolute coefficients (L1). Drives some coefficients exactly to zero, performing automatic feature selection. Use when you suspect many irrelevant features.

**Elastic Net**: Combines L1 and L2 penalties. The `l1_ratio` parameter controls the mix (0 = Ridge, 1 = Lasso). Use when features are correlated and Lasso tends to arbitrarily drop one from a correlated group.

**Key parameters of `RegularizedRegressionTransformer`:**

| Parameter | Default | Description |
|---|---|---|
| `method` | `"ridge"` | `"ridge"`, `"lasso"`, `"elastic_net"` |
| `alpha` | `"auto"` | Regularisation strength; `"auto"` uses CV |
| `l1_ratio` | `0.5` | ElasticNet L1/L2 mix (only for elastic_net) |
| `cv` | `5` | Cross-validation folds for hyperparameter search |
| `max_iter` | `1000` | Solver iteration limit |

---

### Logistic Regression

`LogisticRegressionTransformer` fits a regularised logistic regression for binary or multiclass classification. Reports coefficients, odds ratios, and classification metrics (accuracy, precision, recall, F1, AUC-ROC).

---

### Polynomial Regression

`PolynomialRegressionTransformer` expands features to polynomial terms up to a specified degree, then fits OLS. Use for capturing non-linear relationships in low-dimensional data. Beware overfitting at high degrees.

---

### Feature Selection

Three methods are available via `FeatureSelectionTransformer`:

**Model-based** (`method="model_based"`): Uses `sklearn.feature_selection.SelectFromModel` with a LassoCV estimator. Features with near-zero Lasso coefficients are dropped.

**Recursive Feature Elimination** (`method="rfe"`): Iteratively fits the model and removes the least important feature. The number of features to keep is set by `k`.

**RFECV** (`method="rfecv"`): Like RFE but selects k automatically via cross-validation. Reports the optimal number of features and cross-validation scores.

**Univariate** (`method="univariate"`): Ranks features by F-statistic from `SelectKBest(f_regression)`. Fast but ignores feature interactions.

**Key parameters of `FeatureSelectionTransformer`:**

| Parameter | Default | Description |
|---|---|---|
| `method` | `"model_based"` | Selection method |
| `k` | `"all"` | Number of features to select (RFE, univariate) |
| `cv` | `5` | Cross-validation folds (RFECV) |
| `scoring` | `"r2"` | Evaluation metric for RFECV |

**Returns** include: `selected_features`, `feature_importance`, R² before and after selection, and feature reduction ratio.

---

### Residual Analysis

`ResidualAnalysisTransformer` performs full residual diagnostics automatically when `residual_analysis=True` in the pipeline.

**Normality tests:**
- **Shapiro-Wilk** (n < 5,000): most powerful for small samples
- **Anderson-Darling**: compared against critical values at 5% significance
- **Jarque-Bera**: tests whether skewness and kurtosis match the normal distribution

**Homoscedasticity tests:**
- **Breusch-Pagan**: regresses squared residuals on features; significant result indicates heteroscedasticity
- **White**: tests for non-linear forms of heteroscedasticity

**Autocorrelation:**
- **Durbin-Watson** statistic: values near 2 indicate no autocorrelation; < 1.5 suggests positive, > 2.5 suggests negative autocorrelation

**Influence measures:**
- **Leverage**: diagonal of the hat matrix; high leverage points have unusual feature values
- **Cook's distance**: measures overall influence on all fitted values; values > 4/n are flagged
- **Studentised residuals**: standardised by leave-one-out standard error; |value| > 2.5 are flagged as outliers

---

### Model Evaluation Metrics

| Metric | Range | Better when |
|---|---|---|
| R² | 0 – 1 | Higher |
| Adjusted R² | < R² | Higher (penalises extra features) |
| RMSE | ≥ 0 | Lower (same units as target) |
| MAE | ≥ 0 | Lower (robust to outliers) |
| AIC | any | Lower (model comparison) |
| BIC | any | Lower (stronger penalty for complexity) |

The MCP tool `evaluate_model_performance` scores one actual/predicted pair at a
time and reports R², MSE, RMSE, MAE and residuals for a regression target, or
accuracy, precision, recall and F1 for a classification one. It has no view of
the training set, so it cannot flag overfitting on its own: run it once over
training rows and once over held-out rows and compare the two R² values
yourself. A gap above 0.1 is the usual warning line.

---

## Composition

| Next step | Purpose |
|---|---|
| `statistical_analysis` | Validate model assumptions; test residual normality and correlation between residuals and features |
| `pattern_recognition` | Identify clusters in residuals that may indicate omitted subgroup structure |
| `time_series` | Use fitted regression as part of a decomposition or as a feature in forecasting |
| `business_intelligence` | Translate model coefficients into business impact estimates |

Each step is a separate call. The `regression_analysis` block is JSON returned to
the caller, not a handle another tool can consume — to test residuals with
`analyze_hypothesis_test`, write them back to the source as a column first.

---

## Examples

Every example below is an MCP tool call, the way an agent would issue it.

### What drives house prices?

```python
analyze_regression(
    "housing", "SELECT price, sqft, bedrooms, age FROM listings",
    target_column="price", model_type="linear",
)
```

Read the coefficients with their p-values, then read the `residual_analysis`
block before believing them: a Breusch-Pagan rejection means the standard errors
— and therefore those p-values — are understated.

### Too many correlated predictors

```python
analyze_regression(
    "housing", "SELECT * FROM listings",
    target_column="price", model_type="lasso",
)
```

Lasso drives the coefficients of uninformative features to exactly zero, so the
surviving non-zero set is itself the answer to "which of these matter?". Use
`ridge` instead when you want every predictor kept but shrunk, and
`elastic_net` when predictors come in correlated groups.

### Is a customer going to churn?

```python
analyze_regression(
    "crm", "SELECT churned, tenure_months, support_tickets, plan_tier FROM accounts",
    target_column="churned", feature_columns=["tenure_months", "support_tickets"],
    model_type="logistic",
)
```

A binary target needs `model_type="logistic"` — nothing infers it from the
column. Logistic fits skip the residual diagnostics, which assume a continuous
outcome.

### How good were last quarter's forecasts?

```python
evaluate_model_performance(
    "forecasts", "SELECT actual_revenue, predicted_revenue FROM q3_results",
    target_column="actual_revenue", prediction_column="predicted_revenue",
)
```

Both columns must already exist in the source. To check for overfitting, run
this twice — once over the rows the model was fitted on, once over held-out rows
— and compare the two R² values.

### Score a classifier instead

```python
evaluate_model_performance(
    "crm", "SELECT churned, churn_prediction FROM accounts",
    target_column="churned", prediction_column="churn_prediction",
    model_type="classification",
)
```

`model_type="classification"` swaps R² and RMSE for accuracy, precision, recall
and F1. The prediction column must hold class labels, not probabilities.
