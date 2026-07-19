---
name: data-pipeline
description: Run an end-to-end data analysis pipeline — connect, profile, analyze, and report. Use when performing a complete analysis workflow.
allowed-tools: mcp__localdata__connect_database mcp__localdata__describe_database mcp__localdata__execute_query mcp__localdata__get_data_quality_report mcp__localdata__analyze_hypothesis_test mcp__localdata__analyze_regression mcp__localdata__analyze_clusters mcp__localdata__reduce_dimensions mcp__localdata__analyze_time_series mcp__localdata__forecast_time_series mcp__localdata__analyze_ab_test mcp__localdata__analyze_effect_sizes mcp__localdata__export_schema mcp__localdata__export_structured
argument-hint: "<source-path> <analysis-type: regression|clustering|forecast|correlation|ab-test>"
---

# Data Pipeline

Run a complete analysis workflow from connection through profiling, analysis, and structured reporting.

## Steps

1. **Parse arguments.** Extract the source path and analysis type from `$ARGUMENTS`. The first argument is the file path or connection string. The second argument is the analysis type: one of regression, clustering, forecast, correlation, or ab-test.

2. **Connect and profile.** Call `connect_database` with the source path. Then call `describe_database` to get the schema and `get_data_quality_report` to assess data quality. Summarize table structure, row counts, and quality scores.

3. **Assess data readiness.** Review the quality report. If critical issues exist (more than 30% nulls in key columns, severe duplicates), note them as caveats. Identify the columns relevant to the requested analysis type.

4. **Route to the appropriate analysis.** Based on the analysis type argument:

   **regression** -- Call `analyze_regression` with the target column and feature columns. Then call `execute_query` if needed to inspect residuals. Report model coefficients, R-squared, and feature importance.

   **clustering** -- Call `analyze_clusters` with numeric feature columns and algorithm "kmeans". Try k=2 through k=5 and compare silhouette scores. Call `reduce_dimensions` with PCA for a 2D summary. Report cluster profiles and quality.

   **forecast** -- Call `analyze_time_series` to decompose the series and test stationarity. Then call `forecast_time_series` with the target column and desired horizon. Report trend, seasonality, and forecast values with confidence intervals.

   **correlation** -- Call `analyze_hypothesis_test` with test type "correlation" for each numeric column pair. Call `analyze_effect_sizes` for the strongest relationships. Report a ranked correlation matrix with significance levels.

   **ab-test** -- Call `analyze_ab_test` with the metric and group columns. Call `analyze_effect_sizes` to quantify practical significance. Report group comparison, p-value, effect size, and ship/iterate/no-ship recommendation.

5. **Export results.** Call `export_schema` to capture the data structure. If the analysis produced structured output suitable for downstream use, call `export_structured` to save it in a portable format.

6. **Present the report.** Deliver a structured summary covering:
   - Data source: file type, tables, row and column counts
   - Quality assessment: completeness, consistency, issues found
   - Analysis results: key findings from the selected analysis type
   - Confidence: statistical significance levels and effect sizes where applicable
   - Limitations: data quality caveats, sample size concerns, assumption violations

7. **Recommend follow-up.** Suggest complementary analyses:
   - After regression: `/localdata-mcp:analyze-correlations` to validate predictor selection
   - After clustering: `/localdata-mcp:regression` using cluster labels as features
   - After forecast: `/localdata-mcp:analyze-correlations` to find external predictors
   - After correlation: `/localdata-mcp:regression` with the strongest predictors
   - After ab-test: `/localdata-mcp:cluster-analysis` to find segment-level effects
