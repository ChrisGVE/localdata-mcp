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

The domain exposes two primary MCP tools via `src/localdata_mcp/datascience_tools.py`.

### `tool_fit_regression`

Fit a regression model on data retrieved from a SQL query, with optional residual analysis and cross-validation.

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `engine` | `Engine` | required | SQLAlchemy engine from an active connection |
| `query` | `str` | required | SQL query returning features and target column |
| `target_column` | `str` | required | Name of the numeric outcome column |
| `feature_columns` | `list[str]` | `None` | Feature columns; all non-target columns used if None |
| `model_type` | `str` | `"linear"` | `"linear"`, `"ridge"`, `"lasso"`, `"elastic_net"`, `"logistic"`, `"polynomial"` |
| `regularization` | `str` | `None` | Override regularisation method (alternative to `model_type`) |
| `max_rows` | `int` | `None` | Row cap (default 500,000) |

Underlying `RegressionModelingPipeline` also accepts:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `cross_validation` | `bool` | `True` | Perform K-fold cross-validation |
| `residual_analysis` | `bool` | `True` | Run residual diagnostics after fitting |
| `feature_selection` | `bool` | `False` | Run automatic feature selection before fitting |
| `preprocessing` | `str` | `"auto"` | Preprocessing level: `"minimal"`, `"auto"`, `"comprehensive"` |

**Returns:** `dict` with keys:

- `model_type` — model type fitted
- `regression_analysis` — coefficients, standard errors, p-values, R², adjusted R², RMSE, MAE, AIC, BIC
- `residual_analysis` — normality tests, homoscedasticity tests, autocorrelation, outlier indices, Cook's distances
- `feature_selection` — selected features and importance scores (when enabled)
- `pipeline_config` — configuration settings used

---

### `tool_evaluate_model`

Evaluate a fitted model's performance on held-out data.

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `engine` | `Engine` | required | SQLAlchemy engine |
| `query` | `str` | required | SQL query returning test data |
| `target_column` | `str` | required | Ground-truth outcome column |
| `prediction_column` | `str` | required | Column containing model predictions |
| `model_type` | `str` | `"regression"` | Model type for interpretation context |
| `max_rows` | `int` | `None` | Row cap |

For direct use of `evaluate_model_performance` from the domain:

```python
evaluation = evaluate_model_performance(
    model=fitted_model,
    X_test=X_test,
    y_test=y_test,
    X_train=X_train,   # optional, enables overfitting check
    y_train=y_train,
)
```

**Returns:** `dict` with keys:

- `test_metrics` — R², MSE, RMSE, MAE, explained variance
- `train_metrics` — same metrics for training data (when provided)
- `overfitting_check` — R² gap between train and test; `likely_overfitting=True` when gap > 0.1
- `test_predictions` — model predictions as list
- `test_residuals` — prediction errors as list

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

The overfitting check in `evaluate_model_performance` flags when the train R² exceeds test R² by more than 0.1. The MSE ratio (test / train) above 1.5 is a secondary signal.

---

## Composition

| Next step | Purpose |
|---|---|
| `statistical_analysis` | Validate model assumptions; test residual normality and correlation between residuals and features |
| `pattern_recognition` | Identify clusters in residuals that may indicate omitted subgroup structure |
| `time_series` | Use fitted regression as part of a decomposition or as a feature in forecasting |
| `business_intelligence` | Translate model coefficients into business impact estimates |

The `regression_analysis` result dict from the pipeline can be passed directly to `statistical_analysis` tools by supplying the residuals array and feature matrix.

---

## Examples

### Fit a linear model with diagnostics

```python
result = tool_fit_regression(
    engine=engine,
    query="SELECT price, sqft, bedrooms, age, neighborhood FROM housing",
    target_column="price",
    model_type="linear",
)

reg = result["regression_analysis"]
print(f"R² = {reg['r2']:.3f}, Adjusted R² = {reg['adj_r2']:.3f}")
print(f"RMSE = {reg['rmse']:.1f}")

# Feature coefficients
for feat, coef in reg["coefficients"].items():
    print(f"  {feat}: {coef:.3f} (p={reg['p_values'][feat]:.4f})")

# Residual diagnostics
res = result["residual_analysis"]
print("Residuals normal?", res["normality_test"]["shapiro_wilk"]["is_normal"])
print("Homoscedastic?", res["homoscedasticity_test"]["breusch_pagan"]["is_homoscedastic"])
```

### Regularised regression with automatic alpha selection

```python
from localdata_mcp.domains.regression_modeling import RegularizedRegressionTransformer
import pandas as pd

df = pd.read_sql("SELECT * FROM features", engine)
X = df.drop(columns=["target"]).values
y = df["target"].values
feature_names = df.drop(columns=["target"]).columns.tolist()

transformer = RegularizedRegressionTransformer(method="lasso", alpha="auto", cv=5)
transformer.fit(X, y, feature_names=feature_names)
result = transformer.get_result()

print(f"Best alpha: {result['best_alpha']:.5f}")
print("Non-zero features:", result["non_zero_features"])
```

### Feature selection before model fitting

```python
result = tool_fit_regression(
    engine=engine,
    query="SELECT * FROM wide_feature_table",
    target_column="outcome",
    model_type="linear",
    feature_selection=True,  # enable RFECV-based selection
)

sel = result["feature_selection"]
print(f"Selected {sel['n_selected']} of {sel['n_original']} features")
print("Selected:", sel["selected_features"])
print(f"R² retained: {sel['comparison']['r2_selected']:.3f}")
```

### Evaluate overfitting on a hold-out set

```python
# Fit on training data
train_result = tool_fit_regression(
    engine=engine,
    query="SELECT * FROM train_data",
    target_column="sales",
    model_type="ridge",
)

# Evaluate on test data
eval_result = tool_evaluate_model(
    engine=engine,
    query="SELECT * FROM test_data",
    target_column="sales",
    prediction_column="predicted_sales",  # pre-computed or use model directly
)

check = eval_result["overfitting_check"]
print(f"R² gap: {check['r2_gap']:.3f}")
print(f"Likely overfitting: {check['likely_overfitting']}")
```

### Full pipeline: feature selection → lasso → residual diagnostics

```python
from localdata_mcp.domains.regression_modeling import RegressionModelingPipeline

pipeline = RegressionModelingPipeline(
    model_type="lasso",
    cross_validation=True,
    residual_analysis=True,
    feature_selection=True,
)
pipeline.fit(X_train, y_train, feature_names=feature_names)
results = pipeline.get_results()

# Report
print("AIC:", results["regression_analysis"]["aic"])
outliers = results["residual_analysis"]["outliers"]
print(f"Potential outliers at indices: {outliers}")
cooks = results["residual_analysis"]["cooks_distance"]
high_influence = [i for i, c in enumerate(cooks) if c is not None and c > 4/len(y_train)]
print(f"High-influence observations: {high_influence}")
```
