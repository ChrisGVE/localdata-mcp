---
name: forecast
description: Analyze a time series and generate forecasts with confidence intervals. Use when predicting future values from historical data.
allowed-tools: mcp__localdata__execute_query mcp__localdata__analyze_time_series mcp__localdata__forecast_time_series
argument-hint: "<database-name> <column-name>"
---

# Forecast

Decompose a time series, test for stationarity, and generate forecasts with confidence intervals.

## Steps

1. **Parse arguments.** Extract the database name and target column name from `$ARGUMENTS`. The first argument is the database name, the second is the column to forecast.

2. **Query the time series.** Call `execute_query` to select the datetime column and the target column, ordered by time ascending. If the table is not obvious, query the schema first. Ensure no gaps in the time index. Limit to the most recent 10,000 observations if the series is very long.

3. **Analyze the series.** Call `analyze_time_series` with the database name and column. Review the decomposition results:
   - Trend: is the series trending up, down, or flat?
   - Seasonality: what periodic patterns exist and at what frequency?
   - Residuals: are they random (good) or structured (model may miss patterns)?
   - Stationarity test: note the ADF test result and p-value.

4. **Interpret the analysis.** Summarize the time series characteristics:
   - Overall direction and rate of change
   - Seasonal period (daily, weekly, monthly, yearly)
   - Volatility and any structural breaks
   - Whether differencing is needed (non-stationary series)

5. **Choose the model and forecast.** Call `forecast_time_series` with the database name, the date and value columns, a `horizon`, and a `method`. There is no automatic selection: `method` accepts `"arima"` (the default) or `"ets"`, and nothing else — `"sarima"` and `"prophet"` are rejected with `ValueError: Unknown forecast method`. Decide from step 4: pick `"arima"` when the series is stationary or becomes so after differencing, `"ets"` when a smooth trend and seasonality dominate and the residuals are not autocorrelated. If neither is clearly better, run both and compare.

   Confidence intervals come back with the forecast at a single level set by the model's `alpha`; the tool does not accept a level argument, so do not promise the user a choice of 80% and 95%.

6. **Present the forecast.** Display results in a structured format:
   - Forecast values for each future period
   - The `forecast_lower_ci` and `forecast_upper_ci` bounds, labelled with the `confidence_level` the response reports
   - Which method you chose and what in step 4 led you to it
   - Key assumptions and limitations

7. **Assess forecast reliability.** Comment on:
   - Width of confidence intervals (narrow = more certain)
   - How far the forecast extends relative to historical data length
   - Whether recent trends or seasonality shifts may affect accuracy

8. **Recommend next steps.** Suggest monitoring actual values against forecasts, re-fitting when new data arrives, or running `/localdata-mcp:analyze-correlations` to find external predictors that could improve the model.
