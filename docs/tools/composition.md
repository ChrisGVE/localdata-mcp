<!-- MACHINE-WRITTEN by localdata_mcp.nexus.contract.generators.docs — DO NOT EDIT; regenerate via `python -m localdata_mcp.nexus.contract.generate` -->

# Tools — composition

| Tool | Summary | Input shape | Output shape | Streaming | Params |
|---|---|---|---|---|---|
| `compose_pipeline` | Compose registered tools into one validated pipeline: dag_spec is an ordered list of {stage, tool, params, depends_on} entries (linear chains with fan-out; a chain-initial stage addresses its own source, a dependent stage consumes its upstream stage's output). The whole chain is validated against the tool contracts and the type-shape adjacency table BEFORE anything runs — an incompatible chain is rejected naming the offending stage, with no partial run. Returns one envelope per terminal stage under a single provenance chain. | DYNAMIC | DYNAMIC | no | `dag_spec` |
| `clean_then_profile` | Convenience pipeline: prepare_missing_values -> profile_data on one addressed source. missing_strategy defaults to drop (fabricates nothing); callable with a source alone. Equivalent to the explicit two-stage compose_pipeline. | DYNAMIC | DYNAMIC | no | `endpoint?`, `path?`, `table?`, `query?`, `missing_strategy?` |
| `clean_then_regress` | Convenience pipeline: prepare_missing_values -> analyze_regression on one addressed source, predicting target. missing_strategy defaults to drop. Equivalent to the explicit two-stage compose_pipeline. | DYNAMIC | DYNAMIC | no | `endpoint?`, `path?`, `table?`, `query?`, `target`, `missing_strategy?` |
