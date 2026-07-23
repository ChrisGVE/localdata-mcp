<!-- MACHINE-WRITTEN by localdata_mcp.nexus.contract.generators.docs — DO NOT EDIT; regenerate via `python -m localdata_mcp.nexus.contract.generate` -->

# Tools — pattern_recognition

| Tool | Summary | Input shape | Output shape | Streaming | Params |
|---|---|---|---|---|---|
| `analyze_clusters` | Cluster an addressed tabular source: method kmeans (default), hierarchical, dbscan, gmm, or spectral. Without n_clusters= a silhouette sweep picks k. Reports labels, cluster sizes, and silhouette score; seed= pins stochastic initialization. | TABULAR | SCALAR | no | `endpoint?`, `path?`, `table?`, `query?`, `columns?`, `method?`, `n_clusters?`, `seed?`, `algorithm_params?` |
| `detect_anomalies` | Find anomalous rows in an addressed tabular source: method isolation_forest (default), lof, or zscore (three-sigma rule). Reports anomaly indices, share, and a score summary. | TABULAR | SCALAR | no | `endpoint?`, `path?`, `table?`, `query?`, `columns?`, `method?`, `contamination?`, `seed?`, `algorithm_params?` |
| `reduce_dimensions` | Embed an addressed tabular source into fewer dimensions: method pca (default, always reports explained_variance_ratio) or tsne (reports trustworthiness against the original data). | TABULAR | MATRIX | no | `endpoint?`, `path?`, `table?`, `query?`, `columns?`, `method?`, `n_components?`, `seed?`, `algorithm_params?` |
| `transform_data` | Regex find/replace over one column of an addressed tabular source (pattern crosses the hardened safety screen). Returns the rewritten relation plus a change summary — composable into downstream stages. | TABULAR | TABULAR | no | `endpoint?`, `path?`, `table?`, `query?`, `column`, `find`, `replace`, `case_sensitive?` |
