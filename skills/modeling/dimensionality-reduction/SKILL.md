---
name: dimensionality-reduction
description: Reduce high-dimensional data to fewer components for visualization or feature engineering using PCA, t-SNE, or UMAP. Use when data has many features and needs simplification.
allowed-tools: mcp__localdata__describe_database mcp__localdata__execute_query mcp__localdata__reduce_dimensions mcp__localdata__get_data_quality_report
argument-hint: "<database-name>"
---

# Dimensionality Reduction

Reduce data to fewer dimensions for visualization, pattern discovery, or feature engineering.

## Steps

1. **Explore features.** Call `describe_database` with the database name from `$ARGUMENTS`. Identify all numeric columns. Call `get_data_quality_report` to check for nulls and assess feature distributions.

2. **Extract data.** Call `execute_query` to select the numeric feature columns. Note the number of rows and features. High-dimensional data (10+ features) benefits most from reduction.

3. **Run PCA first.** Call `reduce_dimensions` with algorithm "pca" and 2-3 components. Review:
   - Explained variance ratio per component (how much information each captures)
   - Cumulative explained variance (target > 70% in 2-3 components for good visualization)
   - Component loadings (which original features contribute most to each component)

4. **Interpret PCA components.** Describe each component by its top-loading features. Name the components in domain terms when possible (e.g., "size factor" if height, weight, and volume all load heavily on PC1).

5. **Try t-SNE or UMAP for visualization.** If PCA explains less than 50% of variance in 2D (data has complex nonlinear structure), call `reduce_dimensions` with "tsne" or "umap". These methods preserve local structure better but:
   - Distances between distant points are not meaningful
   - Results depend on hyperparameters (perplexity for t-SNE, n_neighbors for UMAP)
   - Not suitable for downstream modeling, only for visualization

6. **Compare methods.** Assess which reduction best reveals structure: clusters, gradients, or outliers in the 2D view. PCA is interpretable; t-SNE/UMAP reveal groupings.

7. **Present results.** Provide:
   - Method selected and rationale
   - Explained variance (PCA) or stress metric
   - Component interpretation with feature loadings
   - Visual description of the 2D structure (clusters, gradients, outliers)
   - Recommendations: use PCA components as features in `/localdata-mcp:regression`, or use groupings visible in t-SNE as input to `/localdata-mcp:cluster-analysis`
