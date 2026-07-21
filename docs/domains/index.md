# Data Science Domains

LocalData MCP includes 8 analytical domains, each providing specialized tools for
LLM-driven data analysis. Domains compose, but you do the composing: a result
carries no handle the next tool consumes, so you read a value out of one result
and narrow the next query with it.

## Available domains

Each row lists what the domain's **MCP tools** do. Where a domain's Python
package carries more than its tools expose, the domain's own page says so.

| Domain | Key capabilities |
|--------|-----------------|
| [Statistical Analysis](statistical-analysis.md) | Hypothesis tests, ANOVA, effect sizes, non-parametric methods |
| [Time Series](time-series.md) | Trend and seasonality summaries, stationarity testing, autocorrelation, ARIMA and ETS forecasting |
| [Regression](regression.md) | Linear, polynomial, logistic, ridge, lasso, elastic net |
| [Pattern Recognition](pattern-recognition.md) | Clustering, anomaly detection, dimensionality reduction |
| [Business Intelligence](business-intelligence.md) | A/B testing, RFM segmentation |
| [Geospatial](geospatial.md) | Spatial autocorrelation, hotspots, distance, joins, overlay, routing |
| [Optimization](optimization.md) | Linear programming, constrained optimization, assignment |
| [Sampling & Estimation](sampling-estimation.md) | Bootstrap, Bayesian, Monte Carlo, stratified sampling |

```{toctree}
:maxdepth: 2
:hidden:

statistical-analysis
time-series
regression
pattern-recognition
business-intelligence
geospatial
optimization
sampling-estimation
```
