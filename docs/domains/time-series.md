# Time Series Domain

## Overview

The time series domain provides decomposition, stationarity testing, forecasting, change point detection, and multivariate analysis for sequential data indexed by time. Use it when your data has a temporal structure and you need to understand trends, seasonality, or predict future values.

**When to use this domain:**

- Separating trend, seasonal, and residual components from a series
- Testing whether a series is stationary before ARIMA modelling
- Forecasting future values with confidence intervals
- Detecting structural breaks or sudden shifts in a series
- Identifying lead-lag relationships between two or more series (Granger causality)
- Modelling multiple interdependent time series jointly (VAR)

**Source:** `src/localdata_mcp/domains/time_series_analysis/`

---

## Available Analyses

| Method | Class | Description |
|---|---|---|
| Additive/multiplicative decomposition | `TimeSeriesDecompositionTransformer` | Separate trend, seasonal, and residual components |
| ADF stationarity test | `StationarityTestTransformer` | Augmented Dickey-Fuller unit root test |
| KPSS stationarity test | `StationarityTestTransformer` | KPSS test for trend stationarity |
| ACF analysis | `AutocorrelationAnalysisTransformer` | Autocorrelation function with significance bands |
| PACF analysis | `PartialAutocorrelationAnalysisTransformer` | Partial autocorrelation for AR order selection |
| Lag selection | `LagSelectionTransformer` | Optimal lag via AIC/BIC/HQ criteria |
| ARIMA forecasting | `ARIMAForecastTransformer` | AutoRegressive Integrated Moving Average |
| SARIMA forecasting | `SARIMAForecastTransformer` | Seasonal ARIMA with explicit seasonal orders |
| Auto-ARIMA | `AutoARIMATransformer` | Automatic order selection via information criteria |
| Exponential smoothing (ETS) | `ExponentialSmoothingForecaster` | Error-Trend-Seasonality state space models |
| Ensemble forecasting | `EnsembleForecaster` | Weighted combination of ETS and ARIMA |
| Change point detection | `ChangePointDetector` | Structural breaks via ruptures library |
| Anomaly detection | `AnomalyDetector` | Point and contextual anomaly detection |
| Seasonal anomaly detection | `SeasonalAnomalyDetector` | Anomalies relative to seasonal expectations |
| Granger causality | `GrangerCausalityAnalyzer` | Predictive causality between series pairs |
| Cointegration | `CointegrationAnalyzer` | Long-run equilibrium relationships |
| VAR modelling | `VARModelForecaster` | Vector AutoRegression for multivariate series |
| Impulse response | `ImpulseResponseAnalyzer` | System response to shocks in VAR models |

---

## MCP Tool Reference

The domain exposes two MCP tools via `src/localdata_mcp/datascience_tools.py`.

### `tool_time_series_analysis`

Analyse a time series retrieved from a SQL query: decomposition, stationarity, autocorrelation, and feature extraction.

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `engine` | `Engine` | required | SQLAlchemy engine from an active connection |
| `query` | `str` | required | SQL query returning at minimum a date and value column |
| `date_column` | `str` | required | Name of the datetime column |
| `value_column` | `str` | required | Name of the numeric series column |
| `frequency` | `str` | `None` | Pandas offset alias (e.g. `"D"`, `"M"`, `"H"`); inferred if None |
| `max_rows` | `int` | `None` | Row cap (default 500,000) |

**Returns:** `dict` including:

- `decomposition` — trend, seasonal, and residual arrays
- `stationarity` — ADF and KPSS test results and recommendations
- `autocorrelation` — ACF and PACF values with significance lags
- `features` — statistical features (mean, variance, trend slope, seasonality strength)
- `quality` — gap detection and continuity diagnostics

---

### `tool_time_series_forecast`

Generate point forecasts with confidence intervals.

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `engine` | `Engine` | required | SQLAlchemy engine |
| `query` | `str` | required | SQL query |
| `date_column` | `str` | required | Datetime column name |
| `value_column` | `str` | required | Numeric series column name |
| `horizon` | `int` | `10` | Number of periods to forecast ahead |
| `method` | `str` | `"arima"` | Forecasting method: `"arima"` or `"ets"` / `"exponential_smoothing"` |
| `max_rows` | `int` | `None` | Row cap |

**Returns:** `dict` including:

- `forecast` — array of point forecasts
- `confidence_intervals` — lower and upper bounds at the configured level
- `model_summary` — fitted model parameters and information criteria (AIC, BIC)
- `diagnostics` — Ljung-Box residual test results
- `performance` — in-sample metrics (RMSE, MAE, MAPE)

---

## Method Details

### Decomposition

**Additive model** (`model="additive"`): Assumes seasonal and trend components add to the observed series: `Y = Trend + Seasonal + Residual`. Use when seasonal fluctuations are roughly constant in magnitude over time.

**Multiplicative model** (`model="multiplicative"`): Assumes components multiply: `Y = Trend × Seasonal × Residual`. Use when seasonal fluctuations grow proportionally with the trend level (common in sales data).

The period is detected automatically from the DatetimeIndex frequency. Override with `period=12` for monthly data, `period=7` for daily data with weekly seasonality.

**Key parameters of `TimeSeriesDecompositionTransformer`:**

| Parameter | Default | Description |
|---|---|---|
| `model` | `"additive"` | `"additive"` or `"multiplicative"` |
| `period` | auto | Seasonal period (e.g. 12 for monthly) |
| `method` | `"seasonal_decompose"` | `"seasonal_decompose"` or `"stl"` |
| `two_sided` | `True` | Use centred moving average for trend |

---

### Stationarity Testing

A stationary series has constant mean and variance over time. Most forecasting models (ARIMA, VAR) require stationarity.

**ADF test** (Augmented Dickey-Fuller): Null hypothesis is that a unit root exists (non-stationary). Rejecting H₀ (p < alpha) indicates stationarity.

**KPSS test**: Null hypothesis is that the series is stationary. Rejecting H₀ indicates non-stationarity. Used together with ADF to distinguish between difference-stationary and trend-stationary processes.

**Interpretation guide:**

| ADF result | KPSS result | Conclusion |
|---|---|---|
| Reject H₀ | Do not reject H₀ | Stationary |
| Do not reject H₀ | Reject H₀ | Non-stationary, needs differencing |
| Reject H₀ | Reject H₀ | Trend-stationary |
| Do not reject H₀ | Do not reject H₀ | Inconclusive |

When `auto_differencing=True`, the transformer suggests the differencing order needed to achieve stationarity.

---

### ACF and PACF

**ACF** (Autocorrelation Function): Measures correlation of a series with its own lagged values. Used to identify the MA order (q) in ARIMA: ACF cuts off at lag q.

**PACF** (Partial Autocorrelation Function): Measures direct correlation at each lag after removing effects of shorter lags. Used to identify the AR order (p): PACF cuts off at lag p.

**Significance bands** are drawn at ±1.96/√n (95% confidence).

---

### ARIMA Forecasting

ARIMA(p, d, q) combines:
- p autoregressive terms (past values)
- d differencing operations (to achieve stationarity)
- q moving average terms (past forecast errors)

**Key parameters of `ARIMAForecastTransformer`:**

| Parameter | Default | Description |
|---|---|---|
| `order` | `(1, 1, 1)` | (p, d, q) orders |
| `seasonal_order` | `(0, 0, 0, 0)` | (P, D, Q, s) seasonal orders |
| `forecast_steps` | `10` | Number of periods to forecast |
| `alpha` | `0.05` | Confidence interval level (1 - alpha) |
| `trend` | `"c"` | Trend parameter: `"n"`, `"c"`, `"t"`, `"ct"` |

**Residual diagnostics** via Ljung-Box test check whether residuals are white noise. A non-significant Ljung-Box result (p > 0.05) indicates a good fit.

---

### SARIMA

SARIMA(p, d, q)(P, D, Q, s) extends ARIMA with explicit seasonal autoregressive and moving average terms at lag s. Use for series with strong, regular seasonality (e.g. monthly retail data with s=12).

---

### Auto-ARIMA

`AutoARIMATransformer` searches the order space and selects the best ARIMA(p, d, q) specification by AIC or BIC. It tests stationarity automatically and determines d. Suitable when you do not want to inspect ACF/PACF plots manually.

---

### Exponential Smoothing (ETS)

State space models with Error, Trend, and Seasonality components. Each component is either None, additive (A), or multiplicative (M).

Common specifications:
- **Simple exponential smoothing** (N, N, N): level only, no trend or seasonality
- **Holt's linear** (A, A, N): level and additive trend
- **Holt-Winters additive** (A, A, A): level, trend, additive seasonality
- **Holt-Winters multiplicative** (M, A, M): level, trend, multiplicative seasonality

Model selection is automatic when `error="auto"`, `trend="auto"`, `seasonal="auto"`.

---

### Ensemble Forecasting

`EnsembleForecaster` fits multiple models (exponential smoothing and ARIMA by default) on a training split, evaluates each on a validation split, and combines forecasts by weighted average. Weights are inversely proportional to validation RMSE.

**Key parameters:**

| Parameter | Default | Description |
|---|---|---|
| `methods` | `["exponential_smoothing", "arima"]` | Models to combine |
| `combination_method` | `"weighted_average"` | `"weighted_average"`, `"median"`, `"best_performer"` |
| `validation_split` | `0.2` | Fraction of data held out for weight optimisation |
| `forecast_steps` | `10` | Forecast horizon |

---

### Change Point Detection

`ChangePointDetector` wraps the `ruptures` library for segmenting a series at structural breaks.

**Key parameters:**

| Parameter | Default | Description |
|---|---|---|
| `method` | `"bcp"` | `"bcp"` (binary segmentation), `"pelt"`, `"window"`, `"dynp"`, `"statistical"` |
| `model` | `"rbf"` | Cost model: `"l1"`, `"l2"`, `"rbf"`, `"normal"`, `"ar"` |
| `min_size` | `10` | Minimum segment length |
| `max_changepoints` | `10` | Upper bound on detected breakpoints |

---

### Granger Causality

`GrangerCausalityAnalyzer` tests whether lagged values of series X improve the prediction of series Y beyond what Y's own lags provide. A significant result (p < alpha) means X Granger-causes Y.

**Key parameters:**

| Parameter | Default | Description |
|---|---|---|
| `max_lags` | `4` | Maximum lag order to test |
| `significance_level` | `0.05` | Rejection threshold |
| `test_all_pairs` | `True` | Test all variable combinations |

Note: Granger causality is a predictive, not causal, concept. Significant results warrant further investigation but do not establish true causality.

---

### Cointegration and VAR

**Cointegration** (`CointegrationAnalyzer`): Tests whether two or more non-stationary series share a long-run equilibrium relationship using the Johansen procedure. Cointegrated series should be modelled with a Vector Error Correction Model (VECM) rather than differenced independently.

**VAR** (`VARModelForecaster`): Fits a Vector AutoRegression to model mutual dependencies among multiple stationary series. Each variable is regressed on its own lags and the lags of all other variables.

---

## Composition

| Next step | Purpose |
|---|---|
| `regression_modeling` | Use decomposed trend or seasonal features as regression inputs |
| `statistical_analysis` (hypothesis test) | Test whether detected change points correspond to significant mean shifts |
| `pattern_recognition` (anomaly detection) | Cross-validate time series anomalies with multivariate anomaly detection |
| `business_intelligence` | Feed forecasts into revenue projections or capacity planning |

Decomposition results (trend array, residuals) and forecast outputs (point estimates, confidence intervals) are structured dicts suitable for downstream composition.

---

## Examples

### Decompose a monthly sales series

```python
import pandas as pd
from localdata_mcp.domains.time_series_analysis import TimeSeriesDecompositionTransformer

df = pd.read_sql("SELECT sale_date, revenue FROM monthly_sales ORDER BY sale_date", engine)
df = df.set_index(pd.to_datetime(df["sale_date"])).drop(columns=["sale_date"])

transformer = TimeSeriesDecompositionTransformer(model="multiplicative", period=12)
transformer.fit(df)
result = transformer.transform(df)

# result.metadata contains trend, seasonal, residual arrays
print(result.metadata["trend"][:5])
print(result.metadata["seasonal"][:5])
```

### Check stationarity and difference if needed

```python
from localdata_mcp.domains.time_series_analysis import StationarityTestTransformer

transformer = StationarityTestTransformer(tests=["adf", "kpss"], auto_differencing=True)
transformer.fit(df)
result = transformer.transform(df)

print(result.metadata["overall_stationary"])
for rec in result.recommendations:
    print(rec)
```

### Forecast with ARIMA via MCP tool

```python
forecast = tool_time_series_forecast(
    engine=engine,
    query="SELECT order_date, units_sold FROM orders ORDER BY order_date",
    date_column="order_date",
    value_column="units_sold",
    horizon=12,
    method="arima",
)

print(forecast["forecast"])            # 12-period point forecasts
print(forecast["confidence_intervals"])  # lower/upper bounds
print(forecast["model_summary"]["aic"])
```

### Ensemble forecast with automatic model weighting

```python
from localdata_mcp.domains.time_series_analysis import EnsembleForecaster

forecaster = EnsembleForecaster(
    forecast_steps=12,
    methods=["exponential_smoothing", "arima"],
    combination_method="weighted_average",
    validation_split=0.2,
)
forecaster.fit(df)
result = forecaster.transform(df)

print(result.metadata["ensemble_forecast"])
print(result.metadata["model_weights"])   # {"exponential_smoothing": 0.6, "arima": 0.4}
```

### Granger causality between two economic indicators

```python
from localdata_mcp.domains.time_series_analysis import GrangerCausalityAnalyzer

# Requires a stationary multivariate DataFrame
analyzer = GrangerCausalityAnalyzer(max_lags=4, significance_level=0.05)
analyzer.fit(macro_df)
result = analyzer.transform(macro_df)

for pair, tests in result.metadata["causality_results"].items():
    print(f"{pair}: significant={tests['is_significant']}, best_lag={tests['best_lag']}")
```

### Full pipeline: stationarity → ARIMA order selection → forecast

```python
from localdata_mcp.domains.time_series_analysis import (
    StationarityTestTransformer,
    AutoARIMATransformer,
)

# 1. Test stationarity
stat_result = StationarityTestTransformer().fit(df).transform(df)
d = 1 if not stat_result.metadata["overall_stationary"] else 0

# 2. Auto-select ARIMA orders
auto = AutoARIMATransformer(max_p=5, max_q=5, d=d, forecast_steps=10)
auto.fit(df)
forecast_result = auto.transform(df)

print(f"Selected order: {auto.best_order_}")
print(forecast_result.metadata["forecast"])
```
