---
name: anomaly-detection
description: Detect outliers and anomalies in data using isolation forest or local outlier factor. Use when identifying unusual observations or suspicious patterns.
allowed-tools: mcp__localdata__describe_database mcp__localdata__execute_query mcp__localdata__detect_anomalies mcp__localdata__reduce_dimensions mcp__localdata__get_data_quality_report
argument-hint: "<database-name>"
---

# Anomaly Detection

Identify unusual observations in the data using appropriate outlier detection algorithms.

## Steps

1. **Explore features.** Call `describe_database` with the database name from `$ARGUMENTS`. Identify numeric columns suitable for anomaly detection. Call `get_data_quality_report` to check for missing values and understand baseline distributions.

2. **Extract and inspect data.** Call `execute_query` to select the feature columns. Check the number of rows and features. Note the expected contamination rate if the user has a prior estimate; otherwise default to 5%.

3. **Select the algorithm.** Choose based on data characteristics:
   - **Isolation Forest**: good general-purpose detector, works well up to ~50 features, fast on large datasets
   - **Local Outlier Factor**: better when anomalies are defined by local density differences (normal behavior varies across data regions)

4. **Run anomaly detection.** Call `detect_anomalies` with the database name, feature columns, and selected algorithm. Review the results: number of anomalies flagged, anomaly scores distribution, and the score threshold used.

5. **Inspect the anomalies.** Call `execute_query` to retrieve the flagged anomalous rows. Examine what makes them unusual: which feature values are extreme? Are there common patterns among the anomalies?

6. **Visualize in reduced dimensions.** Call `reduce_dimensions` with PCA (2 components) on the same features. Map anomaly labels onto the 2D representation to see whether anomalies cluster together or are scattered.

7. **Assess sensitivity.** If the contamination rate strongly affects results, note this. Report how the number of flagged anomalies changes with different thresholds.

8. **Present results.** Provide:
   - Algorithm used and rationale
   - Number of anomalies detected (count and percentage)
   - Characterization of anomalies: common traits, most extreme cases
   - Anomaly score distribution for context
   - Recommendations: investigate flagged records, adjust detection parameters, or monitor for recurring anomaly patterns
