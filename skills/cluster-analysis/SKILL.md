---
name: cluster-analysis
description: Discover natural groupings in data using clustering algorithms with evaluation and visualization. Use when segmenting data or finding patterns.
allowed-tools: mcp__localdata__describe_database mcp__localdata__execute_query mcp__localdata__analyze_clusters mcp__localdata__reduce_dimensions
argument-hint: "<database-name>"
---

# Cluster Analysis

Find natural groupings in data using multiple clustering approaches, evaluate quality, and interpret results.

## Steps

1. **Explore features.** Call `describe_database` with the database name from `$ARGUMENTS`. Identify numeric columns suitable for clustering. Note any categorical columns that could provide context for interpreting clusters later.

2. **Extract and review data.** Call `execute_query` to select the numeric feature columns. Check for nulls and extreme outliers in the sample. Note the number of observations and features.

3. **Run K-Means clustering.** Call `analyze_clusters` with the database name, feature columns, and algorithm set to "kmeans". Start with k=3 unless domain knowledge suggests otherwise. Review the silhouette score and cluster sizes.

4. **Try alternative k values.** If the silhouette score is below 0.5, re-run `analyze_clusters` with k=2, k=4, and k=5. Compare silhouette scores to find the optimal number of clusters.

5. **Run DBSCAN for comparison.** Call `analyze_clusters` with algorithm set to "dbscan". This density-based approach does not require specifying k and can find irregularly shaped clusters. Compare the number of clusters found and the noise point percentage.

6. **Evaluate and compare.** Assess both approaches:
   - Silhouette scores (higher is better, above 0.5 is good)
   - Cluster balance (are clusters roughly even or heavily skewed?)
   - Number of noise points in DBSCAN
   - Which method produces more interpretable groupings

7. **Reduce dimensions for visualization.** Call `reduce_dimensions` with the database name and feature columns, using PCA with 2 components. This provides a 2D representation of the clusters for interpretation.

8. **Interpret clusters.** For the best clustering result, describe each cluster by its feature averages. Give each cluster a descriptive label based on its defining characteristics. Note which features most differentiate the clusters.

9. **Present results.** Provide:
   - Recommended number of clusters and algorithm
   - Cluster profiles with feature summaries
   - Silhouette score and quality assessment
   - Observations about cluster separation and overlap

10. **Recommend next steps.** Suggest using cluster labels as a feature in `/localdata-mcp:regression`, running `/localdata-mcp:analyze-correlations` within each cluster, or applying the clustering to new data for segmentation.
