<!-- MACHINE-WRITTEN by localdata_mcp.nexus.contract.generators.docs — DO NOT EDIT; regenerate via `python -m localdata_mcp.nexus.contract.generate` -->

# Tools — preprocessing

| Tool | Summary | Input shape | Output shape | Streaming | Params |
|---|---|---|---|---|---|
| `prepare_missing_values` | Handle missing values on an addressed tabular source: missing_strategy drop (default — fabricates nothing), mean, median, mode, forward_fill, or constant (needs fill_value). Returns the cleaned relation; composes as a pipeline stage. | TABULAR | TABULAR | no | `endpoint?`, `path?`, `table?`, `query?`, `columns?`, `missing_strategy?`, `fill_value?` |
| `convert_types` | Coerce named columns of an addressed tabular source to a target type (numeric, integer, string, datetime, boolean) via the conversions map. Reports failed casts; composes as a pipeline stage. | TABULAR | TABULAR | no | `endpoint?`, `path?`, `table?`, `query?`, `conversions` |
