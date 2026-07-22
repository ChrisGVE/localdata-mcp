<!-- MACHINE-WRITTEN by localdata_mcp.nexus.contract.generators.docs — DO NOT EDIT; regenerate via `python -m localdata_mcp.nexus.contract.generate` -->

# Tools — core

| Tool | Summary | Input shape | Output shape | Streaming | Params |
|---|---|---|---|---|---|
| `ping` | Report server liveness with a constant probe response. | NONE | SCALAR | no | — |
| `probe_table` | Produce a small numbered table of squares for pipeline probing. | NONE | TABULAR | no | `rows` |
| `probe_vector` | Produce an ordered series of triangular numbers for probing. | NONE | VECTOR | no | `length` |
| `probe_matrix` | Produce an identity matrix of the requested size for probing. | NONE | MATRIX | no | `size` |
| `probe_model` | Fit a line to a tiny generated sample and report coefficients. | NONE | FITTED_MODEL | no | `points` |
| `probe_graph` | Produce a path graph with the requested node count for probing. | NONE | GRAPH | no | `nodes` |
| `probe_geo` | Produce evenly spaced points along the equator for probing. | NONE | GEO | no | `points` |
| `probe_chart` | Build a line-chart specification over computed square values. | NONE | CHART_SPEC | no | `points` |
| `probe_sink` | Measure a text payload and report its size as a terminal result. | TABULAR | NONE | no | `text` |
