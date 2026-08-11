<!-- MACHINE-WRITTEN by localdata_mcp.nexus.contract.generators.docs — DO NOT EDIT; regenerate via `python -m localdata_mcp.nexus.contract.generate` -->

# Tools — sampling_estimation

| Tool | Summary | Input shape | Output shape | Streaming | Params |
|---|---|---|---|---|---|
| `generate_sample` | Draw a sample from an addressed tabular source: sampling_method simple_random (default), stratified, systematic, cluster, or weighted. sample_size: integer = row count, fraction = share (default 0.1). Returns the drawn relation plus the design summary. | TABULAR | TABULAR | no | `endpoint?`, `path?`, `table?`, `query?`, `sampling_method?`, `sample_size?`, `stratify_column?`, `cluster_column?`, `weight_column?`, `seed?` |
| `bootstrap_statistic` | Percentile-bootstrap a statistic (mean, median, std, var) of one column on an addressed tabular source: estimate, confidence interval, and standard error. resamples defaults to the operator-configured count. | TABULAR | SCALAR | no | `endpoint?`, `path?`, `table?`, `query?`, `column`, `statistic?`, `resamples?`, `confidence_level?`, `seed?` |
| `monte_carlo_simulate` | Monte Carlo over one column of an addressed tabular source: simulation_type uncertainty (default — resampled distribution of the mean) or integration (probability mass inside bounds=[lower, upper] under the fitted normal). iterations defaults to the operator-configured count. | TABULAR | SCALAR | no | `endpoint?`, `path?`, `table?`, `query?`, `column`, `simulation_type?`, `iterations?`, `bounds?`, `seed?` |
| `bayesian_estimate` | Conjugate-normal Bayesian posterior of one column's mean on an addressed tabular source (noninformative prior): posterior mean, scale, and the Student-t credible interval. | TABULAR | SCALAR | no | `endpoint?`, `path?`, `table?`, `query?`, `column`, `prior_distribution?`, `credible_level?` |
