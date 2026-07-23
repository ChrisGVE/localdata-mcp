<!-- MACHINE-WRITTEN by localdata_mcp.nexus.contract.generators.docs — DO NOT EDIT; regenerate via `python -m localdata_mcp.nexus.contract.generate` -->

# Tools — network_graph

| Tool | Summary | Input shape | Output shape | Streaming | Params |
|---|---|---|---|---|---|
| `analyze_network` | Analyze a network stored as a tabular edge list (addressed source with source_column/target_column, optional weight_column, directed on request): density, connectivity, components, degree summary, clustering, and top centrality nodes. | TABULAR | SCALAR | no | `endpoint?`, `path?`, `table?`, `query?`, `source_column`, `target_column`, `weight_column?`, `directed?`, `include_centrality?` |
