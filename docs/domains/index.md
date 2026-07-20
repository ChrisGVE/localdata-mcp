# Data Science Domains

LocalData MCP includes 8 analytical domains, each providing specialized tools for
LLM-driven data analysis. Domains compose, but you do the composing: a result
carries no handle the next tool consumes, so you read a value out of one result
and narrow the next query with it.

## Available domains

| Domain | Key capabilities |
|--------|-----------------|
| [Statistical Analysis](statistical-analysis.md) | Hypothesis tests, ANOVA, effect sizes, non-parametric methods |
| [Time Series](time-series.md) | Decomposition, ARIMA, exponential smoothing, forecasting, change points |
| [Regression](regression.md) | Linear, polynomial, logistic, ridge, lasso, feature selection |
| [Pattern Recognition](pattern-recognition.md) | Clustering, anomaly detection, dimensionality reduction |
| [Business Intelligence](business-intelligence.md) | A/B testing, cohort analysis, CLV, attribution, RFM |
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
