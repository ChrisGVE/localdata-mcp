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

5. **Generate forecasts.** Call `forecast_time_series` with the database name, column, and desired forecast horizon. The tool will select an appropriate model (ARIMA, ETS, or SARIMA) based on the series characteristics. Request confidence intervals at the 80% and 95% levels.

6. **Present the forecast.** Display results in a structured format:
   - Forecast values for each future period
   - 80% and 95% confidence interval bounds
   - Model selected and why
   - Key assumptions and limitations

7. **Assess forecast reliability.** Comment on:
   - Width of confidence intervals (narrow = more certain)
   - How far the forecast extends relative to historical data length
   - Whether recent trends or seasonality shifts may affect accuracy

8. **Recommend next steps.** Suggest monitoring actual values against forecasts, re-fitting when new data arrives, or running `/localdata-mcp:analyze-correlations` to find external predictors that could improve the model.
