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
| UMAP | `DimensionalityReductionTransformer` | Fast non-linear embedding; requires `umap-learn`, which this package does not install |
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

The domain is reached through three MCP tools. Like every other analytical tool,
each takes the name of a live connection and a SQL query — there is no
data-frame parameter and no separate load step, and column parameters name
columns in the query's result set. The classes listed under *Available Analyses*
above are the internal implementation those tools call; they are not reachable
from an MCP client.

All three share a `columns` parameter: a genuine list of column names
(`["price", "sqft"]`, not a comma-separated string). Omit it and every numeric
column in the result set is used. Full parameter tables live in the
[tools reference](../tools-reference.md#data-science-12-tools); this page covers
what each tool is for and when to reach for it.

### `analyze_clusters`

Answers "what natural groups are in this data?" `method` selects `kmeans`
(default), `dbscan`, `hierarchical`, `gmm` or `spectral`. Leave `n_clusters`
unset and k is chosen by searching 2 through 10 for the best silhouette score,
which is what you want when the number of segments is the question rather than
an input. Returns the label per row, the centroids and a silhouette score.

DBSCAN is the one to reach for when clusters are not blobs or when some rows
should belong to nothing: it labels those `-1` instead of forcing them into the
nearest group.

### `detect_anomalies`

Answers "which rows do not belong?" `method` selects `isolation_forest`
(default), `lof`, `one_class_svm` or `statistical`. These are multivariate
detectors over a set of columns; there is no single-column mode.

`statistical` **is** the z-score and IQR method. Through `algorithm_params` it
takes `method` (`"zscore"`, the default, or `"iqr"`) and `threshold_std`
(default 3.0), and it flags a row when any column breaches the threshold. Reach
for it when you want a rule you can explain to a stakeholder rather than a
learned boundary.

For the other three, `contamination` is the expected anomaly rate and defaults
to 0.1 — a tenth of the rows will be flagged whatever the data looks like, so
set it from what you actually expect. `statistical` ignores it and uses its
threshold instead. Returns a
label per row (`-1` for anomaly), a continuous score, and the flagged indices.

The tool ranks rows by unusualness; it does not know which unusual rows are
problems. Query the flagged indices back and look at them.

### `reduce_dimensions`

Answers "can these many correlated columns be summarised in a few?" `method`
selects `pca` (default), `tsne` or `ica`, and `n_components` sets the output
width (default 2). PCA additionally reports `explained_variance_ratio` and
`cumulative_variance_ratio`, which say how much information survived the
projection. The LDA transformer listed under *Available Analyses* has no MCP
tool in this release; ICA does, through this `method` value.

t-SNE and UMAP produce embeddings for looking at, not for feeding into a model:
they distort distances by design. `umap` is accepted as a `method` but raises
`ImportError` — `umap-learn` is neither a dependency of this package nor one of
its optional extras, so it works only if you have installed it into the
environment yourself.

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

**Dependency note:** `umap-learn` is neither a dependency of this package nor one of its optional extras, so `pip install localdata-mcp[...]` will not bring it in under any combination. Without it the call raises `ImportError` and there is no graceful fallback. Treat UMAP as unavailable unless you have installed `umap-learn` into the environment yourself.

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

1. **Cluster then test**: run `analyze_clusters`, write the labels back to the
   source (or join them in SQL), then call `analyze_anova` with the cluster
   column as `group_var` to see which features separate the groups.
2. **Reduce then cluster**: run `reduce_dimensions` with `method="pca"` and
   roughly ten components to drop noise, then cluster the reduced columns.
3. **Detect then investigate**: run `detect_anomalies`, take the flagged
   indices, and query those rows back with `execute_query` for review.

Each step is a separate call. Results carry no handle that the next tool can
consume, so passing data between them is the caller's work.

---

## Examples

Every example below is an MCP tool call, the way an agent would issue it.

### How many customer segments are there?

```python
analyze_clusters(
    "crm",
    "SELECT avg_order_value, order_frequency, days_since_last_order FROM customers",
    method="kmeans",
)
```

Omitting `n_clusters` lets the silhouette search pick k. A best silhouette below
about 0.25 means the data has no clear segment structure, whatever k comes back
— the answer is "there aren't any", not "here are three".

### Which transactions look like fraud?

```python
detect_anomalies(
    "payments",
    "SELECT amount, hour_of_day, distance_from_home FROM transactions",
    ["amount", "hour_of_day", "distance_from_home"],
    method="isolation_forest",
    contamination=0.02,
)
```

`contamination=0.02` says roughly two per cent of rows are expected to be
anomalous. The detector flags that share regardless, so the number is a budget
for review capacity as much as an estimate.

### Collapse forty sensor channels to something plottable

```python
reduce_dimensions(
    "telemetry", "SELECT * FROM data_table", method="pca", n_components=2,
)
```

`telemetry` here is a Parquet connection, so its single table is `data_table`.
Check the explained variance before trusting the picture: two components holding
40% of the variance make a plot that hides more than it shows.

### Reduce, then cluster

```python
reduce_dimensions(
    "sensors", "SELECT * FROM readings", method="pca", n_components=10,
)
```

Write the ten components back to the source, then cluster them:

```python
analyze_clusters(
    "sensors",
    "SELECT pc1, pc2, pc3, pc4, pc5, pc6, pc7, pc8, pc9, pc10 FROM readings_pca",
    method="dbscan",
)
```

The intermediate table is yours to create — nothing in the first result feeds
the second automatically.
