# Pattern Recognition Domain

## Overview

The pattern recognition domain provides clustering, dimensionality reduction, and anomaly detection for unlabelled or partially labelled datasets. Use it when you need to discover natural groupings in data, visualise high-dimensional structure in two or three dimensions, or identify observations that deviate significantly from normal behaviour.

**When to use this domain:**

- Segmenting customers, products, or events into natural groups
- Reducing many correlated features to a compact representation before modelling
- Visualising high-dimensional data for exploration
- Flagging unusual observations for manual review or downstream investigation
- Validating whether group labels correspond to real data structure

**Source:** `src/localdata_mcp/domains/pattern_recognition/`

---

## Available Analyses

| Method | Class | Description |
|---|---|---|
| K-means clustering | `ClusteringTransformer` | Partition-based clustering with automatic k selection |
| Hierarchical clustering | `ClusteringTransformer` | Agglomerative clustering with configurable linkage |
| DBSCAN | `ClusteringTransformer` | Density-based clustering; handles arbitrary shapes and noise |
| Gaussian mixture models | `ClusteringTransformer` | Soft probabilistic cluster assignments |
| Spectral clustering | `ClusteringTransformer` | Graph-based clustering for non-convex structures |
| PCA | `DimensionalityReductionTransformer` | Linear projection maximising variance |
| t-SNE | `DimensionalityReductionTransformer` | Non-linear neighbourhood-preserving embedding |
| UMAP | `DimensionalityReductionTransformer` | Fast non-linear embedding; preserves global structure better than t-SNE |
| ICA | `DimensionalityReductionTransformer` | Independent component decomposition |
| LDA | `DimensionalityReductionTransformer` | Supervised linear projection maximising class separability |
| Isolation Forest | `AnomalyDetectionTransformer` | Anomaly detection via random feature splitting |
| One-Class SVM | `AnomalyDetectionTransformer` | Boundary-based anomaly detection |
| Local Outlier Factor (LOF) | `AnomalyDetectionTransformer` | Density-based local anomaly scoring |
| Statistical anomaly detection | `AnomalyDetectionTransformer` | Z-score and IQR based outlier flagging |
| Silhouette score | `PatternEvaluationTransformer` | Average inter-cluster separation vs intra-cluster cohesion |
| Davies-Bouldin index | `PatternEvaluationTransformer` | Average cluster similarity measure (lower is better) |
| Calinski-Harabasz score | `PatternEvaluationTransformer` | Variance ratio criterion (higher is better) |
| Adjusted Rand Index | `PatternEvaluationTransformer` | Cluster agreement with ground truth labels |
| Normalised Mutual Information | `PatternEvaluationTransformer` | Information-theoretic cluster agreement |

---

## MCP Tool Reference

The domain exposes three MCP tools via `src/localdata_mcp/datascience_tools.py`.

### `tool_clustering`

Perform clustering on data retrieved from a SQL query.

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `engine` | `Engine` | required | SQLAlchemy engine from an active connection |
| `query` | `str` | required | SQL query returning numeric feature columns |
| `columns` | `list[str]` | `None` | Columns to use as features; all numeric columns used if None |
| `method` | `str` | `"kmeans"` | Algorithm: `"kmeans"`, `"hierarchical"`, `"dbscan"`, `"gmm"`, `"spectral"` |
| `n_clusters` | `int` | `None` | Number of clusters; auto-selected if None |
| `max_rows` | `int` | `None` | Row cap (default 500,000) |

Underlying `ClusteringTransformer` also accepts:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `auto_k_selection` | `bool` | `True` | Search k_range for optimal k when n_clusters is None |
| `k_range` | `tuple` | `(2, 10)` | Range to search for k |
| `standardize` | `bool` | `True` | Standardise features before clustering |
| `random_state` | `int` | `42` | Reproducibility seed |

**Returns:** `dict` with keys:

- `labels` — cluster assignment per observation
- `n_clusters` — number of clusters found
- `cluster_centers` — centroid coordinates (K-means and GMM)
- `cluster_sizes` — count per cluster
- `inertia` — within-cluster sum of squares (K-means only)
- `evaluation` — silhouette score, Davies-Bouldin index, Calinski-Harabasz score
- `k_scores` — scores across k values when auto-selection ran

---

### `tool_anomaly_detection`

Detect anomalous observations in data retrieved from a SQL query.

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `engine` | `Engine` | required | SQLAlchemy engine |
| `query` | `str` | required | SQL query |
| `columns` | `list[str]` | `None` | Feature columns; all numeric columns used if None |
| `method` | `str` | `"isolation_forest"` | Algorithm: `"isolation_forest"`, `"one_class_svm"`, `"lof"`, `"statistical"` |
| `contamination` | `float` | `0.1` | Expected proportion of anomalies (0.0 – 0.5) |
| `max_rows` | `int` | `None` | Row cap |

Underlying `AnomalyDetectionTransformer` also accepts:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `standardize` | `bool` | `True` | Standardise features before detection |
| `random_state` | `int` | `42` | Reproducibility seed |

**Returns:** `dict` with keys:

- `anomaly_labels` — 1 for normal, -1 for anomaly per observation
- `anomaly_scores` — continuous anomaly score (lower = more anomalous for Isolation Forest)
- `n_anomalies` — count of detected anomalies
- `n_samples` — total observation count
- `anomaly_rate` — fraction of observations flagged
- `anomaly_indices` — indices of flagged observations
- `evaluation` — precision, recall, F1 when ground truth provided; otherwise threshold and score distribution

---

### `tool_dimensionality_reduction`

Reduce feature dimensions in data retrieved from a SQL query.

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `engine` | `Engine` | required | SQLAlchemy engine |
| `query` | `str` | required | SQL query |
| `columns` | `list[str]` | `None` | Feature columns; all numeric columns used if None |
| `method` | `str` | `"pca"` | Algorithm: `"pca"`, `"tsne"`, `"umap"`, `"ica"`, `"lda"` |
| `n_components` | `int` | `2` | Number of output dimensions |
| `max_rows` | `int` | `None` | Row cap |

Underlying `DimensionalityReductionTransformer` also accepts:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `preserve_variance` | `float` | `0.95` | Minimum variance to preserve for PCA auto-selection |
| `standardize` | `bool` | `True` | Standardise features before reduction |
| `random_state` | `int` | `42` | Reproducibility seed |

**Returns:** `dict` with keys:

- `transformed_data` — reduced-dimension representation
- `original_dimensions` — input feature count
- `reduced_dimensions` — output component count
- `explained_variance_ratio` — per-component variance explained (PCA only)
- `cumulative_variance_explained` — cumulative variance (PCA only)
- `evaluation` — reconstruction error and variance preservation metrics
- `loadings` — feature loadings per component (PCA, ICA)

---

## Method Details

### K-means Clustering

Partitions data into k clusters by minimising within-cluster sum of squared distances to centroids. Assumes roughly spherical, equal-sized clusters.

**When to use:** Large datasets, known approximate number of clusters, roughly convex cluster shapes.

**Auto k-selection**: When `n_clusters=None` and `auto_k_selection=True`, the transformer evaluates k across `k_range` using the silhouette score and selects the k with the highest value.

**Key limitations:** Sensitive to outliers; assumes equal cluster variances; does not handle non-convex shapes.

---

### Hierarchical Clustering

Builds a dendrogram by iteratively merging the two closest clusters (agglomerative). Does not require specifying k in advance; the dendrogram can be cut at any level.

**When to use:** Exploratory analysis where the number of clusters is unknown; when a hierarchical structure in the data is expected.

**Linkage methods** (set via `algorithm_params`): `"ward"` (minimises within-cluster variance), `"complete"`, `"average"`, `"single"`.

---

### DBSCAN

Groups points that are closely packed together and marks low-density points as noise (label -1). Does not require specifying k.

**When to use:** Arbitrarily shaped clusters; datasets with noise; unknown number of clusters.

**Key parameters** (passed via `algorithm_params`):

| Parameter | Default | Description |
|---|---|---|
| `eps` | `0.5` | Maximum distance between two samples in the same neighbourhood |
| `min_samples` | `5` | Minimum samples in a neighbourhood to form a core point |

**Note:** DBSCAN does not produce centroid coordinates. Cluster labels start from 0; -1 denotes noise.

---

### Gaussian Mixture Models (GMM)

Fits a mixture of Gaussian distributions and assigns each observation a soft probability of belonging to each component.

**When to use:** When clusters overlap or have different covariance structures; when you want probabilistic membership.

**Key parameters** (via `algorithm_params`):

| Parameter | Default | Description |
|---|---|---|
| `covariance_type` | `"full"` | `"full"`, `"tied"`, `"diag"`, `"spherical"` |
| `max_iter` | `100` | EM algorithm iterations |

---

### PCA

Linear projection that finds orthogonal directions of maximum variance. Components are ordered by explained variance.

**When to use:** Pre-processing before other algorithms; visualisation; removing correlated features.

**Auto component selection**: When `n_components=None`, PCA selects the minimum number of components that preserve `preserve_variance` (default 95%) of total variance.

**Interpretation**: `explained_variance_ratio` tells you how much information each component captures. `loadings` show which original features contribute to each component.

---

### t-SNE

Non-linear dimensionality reduction that places similar observations close together in a 2D or 3D embedding. Optimised for visualisation.

**When to use:** Visualising cluster structure in high-dimensional data.

**Key limitations:** Does not preserve global distances reliably; stochastic (results vary across runs unless `random_state` is fixed); not suitable for dimensionality reduction before machine learning (use PCA for that).

**Key parameters** (via `algorithm_params`):

| Parameter | Default | Description |
|---|---|---|
| `perplexity` | `30` | Balances local vs global structure (typical range 5–50) |
| `n_iter` | `1000` | Optimisation iterations |
| `learning_rate` | `"auto"` | Step size for gradient descent |

---

### UMAP

Non-linear manifold learning that is faster than t-SNE and preserves both local and global structure better.

**When to use:** Large datasets; when t-SNE is too slow; when global structure matters for interpretation.

**Key parameters** (via `algorithm_params`):

| Parameter | Default | Description |
|---|---|---|
| `n_neighbors` | `15` | Local neighbourhood size |
| `min_dist` | `0.1` | Minimum distance between embedded points |
| `metric` | `"euclidean"` | Distance metric |

**Dependency note:** UMAP requires the `umap-learn` package. The transformer falls back gracefully if it is not installed.

---

### Isolation Forest

Detects anomalies by randomly partitioning the feature space. Anomalies are isolated in fewer splits than normal points and therefore have shorter average path lengths.

**When to use:** General-purpose anomaly detection; high-dimensional data; no assumptions about the anomaly distribution.

**Key parameters** (via `algorithm_params`):

| Parameter | Default | Description |
|---|---|---|
| `n_estimators` | `100` | Number of isolation trees |
| `max_samples` | `"auto"` | Samples per tree (256 by default) |

---

### Local Outlier Factor (LOF)

Measures the local density deviation of each point relative to its k nearest neighbours. Points in low-density regions compared to neighbours are flagged.

**When to use:** Detecting anomalies that are only outliers relative to their local neighbourhood; useful when data has clusters of varying density.

**Key parameters** (via `algorithm_params`):

| Parameter | Default | Description |
|---|---|---|
| `n_neighbors` | `20` | Neighbourhood size |
| `metric` | `"minkowski"` | Distance metric |

---

### Clustering Quality Metrics

**Silhouette score** (−1 to 1): Measures how similar each observation is to its own cluster compared to other clusters. Higher is better. Values > 0.5 indicate reasonable separation; > 0.7 indicates strong structure.

**Davies-Bouldin index** (≥ 0): Average ratio of within-cluster scatter to between-cluster separation. Lower is better. Zero indicates perfect separation.

**Calinski-Harabasz score** (≥ 0): Ratio of between-cluster to within-cluster dispersion. Higher is better. No absolute threshold; use for comparing k values.

**Adjusted Rand Index** (−1 to 1): Agreement between predicted labels and ground truth. 1 = perfect agreement; 0 = random; negative = worse than random.

**Normalised Mutual Information** (0 to 1): Information-theoretic agreement with ground truth. 1 = perfect; 0 = no mutual information.

---

## Composition

| Next step | Purpose |
|---|---|
| `statistical_analysis` | Test whether clusters differ significantly on key variables |
| `regression_modeling` | Use cluster labels as features or stratify model fitting per cluster |
| `time_series` | Detect time-series anomalies; compare with spatial anomalies |
| `business_intelligence` | Translate customer clusters into segments for targeting |
| `reduce_dimensions` | Reduce dimensions first, then cluster in the lower-dimensional space |

Typical composition patterns:

1. **Cluster then test**: Run `tool_clustering`, then pass cluster labels to `tool_hypothesis_test` with cluster as the group variable to determine which features drive cluster separation.
2. **Reduce then cluster**: Run `tool_dimensionality_reduction` (PCA, n_components=10) to reduce noise, then run `tool_clustering` on the reduced representation.
3. **Detect then investigate**: Run `tool_anomaly_detection`, extract anomaly indices, then query those rows back from the database for detailed review.

---

## Examples

### Customer segmentation with automatic k selection

```python
result = tool_clustering(
    engine=engine,
    query="SELECT avg_order_value, order_frequency, days_since_last_order FROM customers",
    method="kmeans",
    n_clusters=None,  # auto-select
)

print(f"Optimal k: {result['n_clusters']}")
print(f"Silhouette score: {result['evaluation']['silhouette_score']:.3f}")
print(f"Davies-Bouldin: {result['evaluation']['davies_bouldin_score']:.3f}")

for k, score in result["k_scores"].items():
    print(f"  k={k}: silhouette={score:.3f}")
```

### Fraud detection with Isolation Forest

```python
result = tool_anomaly_detection(
    engine=engine,
    query="SELECT amount, merchant_category, hour_of_day, distance_from_home FROM transactions",
    method="isolation_forest",
    contamination=0.02,   # expect ~2% fraud rate
)

print(f"Anomalies detected: {result['n_anomalies']} / {result['n_samples']}")
print(f"Anomaly rate: {result['anomaly_rate']:.2%}")

# Get row indices to investigate
anomaly_indices = result["anomaly_indices"]
```

### PCA for visualisation and noise reduction

```python
result = tool_dimensionality_reduction(
    engine=engine,
    query="SELECT * FROM high_dimensional_features",
    method="pca",
    n_components=2,
)

print(f"Explained variance: {result['cumulative_variance_explained'][-1]:.1%}")
print("Component loadings:", result["loadings"])

# 2D embedding ready for plotting
import numpy as np
coords = np.array(result["transformed_data"])
# coords[:, 0] = PC1, coords[:, 1] = PC2
```

### Reduce dimensions then cluster

```python
# Step 1: reduce with PCA to remove noise
reduction = tool_dimensionality_reduction(
    engine=engine,
    query="SELECT * FROM sensor_readings",
    method="pca",
    n_components=10,
)

# Step 2: cluster in reduced space using direct transformer
import numpy as np
from localdata_mcp.domains.pattern_recognition import ClusteringTransformer, PatternEvaluationTransformer

X_reduced = np.array(reduction["transformed_data"])

clusterer = ClusteringTransformer(algorithm="dbscan", auto_k_selection=False)
clusterer.fit(X_reduced)
cluster_result = clusterer.get_clustering_result(X_reduced)

evaluator = PatternEvaluationTransformer("clustering")
eval_result = evaluator.evaluate_clustering(X_reduced, cluster_result.labels)

print(f"Clusters found: {cluster_result.n_clusters}")
print(f"Noise points: {(cluster_result.labels == -1).sum()}")
print(f"Silhouette: {eval_result.metrics['silhouette_score']:.3f}")
```

### Compare clustering against known labels

```python
from localdata_mcp.domains.pattern_recognition import perform_clustering
import numpy as np

# Assume X is feature array and y_true are known labels
result = perform_clustering(X, algorithm="gmm", n_clusters=4, y_true=y_true)

eval_data = result["evaluation"]
print(f"Adjusted Rand Index: {eval_data['adjusted_rand_score']:.3f}")
print(f"Normalised Mutual Info: {eval_data['normalized_mutual_info']:.3f}")
print(eval_data["quality_assessment"])  # e.g. "Good clustering structure"
```

### t-SNE visualisation of cluster structure

```python
result = tool_dimensionality_reduction(
    engine=engine,
    query="SELECT * FROM embedding_features",
    method="tsne",
    n_components=2,
    max_rows=5000,   # t-SNE is expensive; limit rows
)

coords = result["transformed_data"]  # shape (n, 2)
# Use with any plotting library
```
