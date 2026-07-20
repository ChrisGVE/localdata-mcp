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

The domain is reached through two MCP tools. Like every other analytical tool,
each takes the name of a live connection and a SQL query — there is no
data-frame parameter and no separate load step, and column parameters name
columns in the query's result set. The classes listed under *Available Analyses*
above are the internal implementation those tools call; they are not reachable
from an MCP client.

Both tools work on one series at a time: a `date_column` and a `value_column`
from the same query. Full parameter tables live in the
[tools reference](../tools-reference.md#data-science-12-tools); this page covers
what each tool is for and when to reach for it.

### `analyze_time_series`

Answers "what is this series made of, and is it stable?" Returns the trend,
seasonal and residual components of a decomposition together with a stationarity
test. `frequency` takes a pandas offset alias — `D`, `W`, `M`, `Q`, `Y` — not a
word like `daily`; leave it empty and the frequency is inferred from the
timestamps.

Run this before forecasting. A series the stationarity test rejects has a trend
or a changing variance that a model must difference away first, and the seasonal
component tells you whether a seasonal period exists at all.

### `forecast_time_series`

Answers "what comes next?" `horizon` is the number of periods ahead (default 10)
and `method` is `arima` (default) or `ets` / `exponential_smoothing`. Those are
the only accepted values: `prophet` and `sarima` raise `ValueError`, and the
SARIMA, auto-ARIMA and ensemble forecasters listed under *Available Analyses*
are exposed by no MCP tool in this release.

Returns point forecasts with confidence intervals. Compare `arima` against `ets`
on a held-out tail of the series rather than trusting either by default — which
one wins depends on whether the seasonality is additive and how much of the
signal is trend.

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

Each step is a separate call. Decomposition components and forecast bounds come
back as JSON to the caller, not as a handle another tool can consume — write
them back to the source as columns to carry them into the next analysis.

---

## Examples

Every example below is an MCP tool call, the way an agent would issue it.

### Does this series have a seasonal pattern?

```python
analyze_time_series(
    "sales", "SELECT sale_date, revenue FROM monthly_totals ORDER BY sale_date",
    date_column="sale_date", value_column="revenue", frequency="M",
)
```

The seasonal component answers it. A stationarity test that fails at the same
time means the level is drifting, which is a separate problem from seasonality
and needs differencing rather than a seasonal term.

### Forecast the next twelve months

```python
forecast_time_series(
    "sales", "SELECT sale_date, revenue FROM monthly_totals ORDER BY sale_date",
    date_column="sale_date", value_column="revenue", horizon=12, method="arima",
)
```

`ORDER BY` the date column: the tools read the series in the order the query
returns it. Confidence intervals widen with the horizon — a twelve-month band
that spans zero is telling you the history does not support a twelve-month call.

### Compare ARIMA against exponential smoothing

```python
forecast_time_series(
    "sales", "SELECT sale_date, revenue FROM monthly_totals WHERE sale_date < '2026-01-01' ORDER BY sale_date",
    date_column="sale_date", value_column="revenue", horizon=6, method="ets",
)
```

Fit both methods on history that stops short of the last six months, then score
each against what actually happened with `evaluate_model_performance`. That
requires writing the two forecasts back next to the actuals first.

### Daily data from a CSV

```python
analyze_time_series(
    "readings", "SELECT ts, value FROM data_table ORDER BY ts",
    date_column="ts", value_column="value", frequency="D",
)
```

A CSV connection loads into a single table named `data_table` whatever the
connection is called.
