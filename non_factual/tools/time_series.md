<!-- MACHINE-WRITTEN by localdata_mcp.nexus.contract.generators.docs — DO NOT EDIT; regenerate via `python -m localdata_mcp.nexus.contract.generate` -->

# Tools — time_series

| Tool | Summary | Input shape | Output shape | Streaming | Params |
|---|---|---|---|---|---|
| `analyze_time_series` | Analyze a time series on an addressed tabular source: trend direction and slope, ADF stationarity, autocorrelation with significant lags, and calendar seasonality strength. | TABULAR | SCALAR | no | `endpoint?`, `path?`, `table?`, `query?`, `date_column`, `value_column`, `frequency?` |
| `forecast_time_series` | Forecast a time series on an addressed tabular source: method arima (default, order= [p,d,q]), sarima (seasonal_order= [P,D,Q,s], defaulted from the calendar frequency), auto_arima (AIC grid search), or ets. Returns the forecast with confidence intervals and the solver's convergence verdict. | TABULAR | VECTOR | no | `endpoint?`, `path?`, `table?`, `query?`, `date_column`, `value_column`, `horizon?`, `method?`, `order?`, `seasonal_order?` |
