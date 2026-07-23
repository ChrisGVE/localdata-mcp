<!-- MACHINE-WRITTEN by localdata_mcp.nexus.contract.generators.docs — DO NOT EDIT; regenerate via `python -m localdata_mcp.nexus.contract.generate` -->

# Tools — visualize

| Tool | Summary | Input shape | Output shape | Streaming | Params |
|---|---|---|---|---|---|
| `render_chart` | Render an addressed tabular source as a chart image. kind is one of histogram, heatmap, scatter_fit, line_timeseries, geo_map, network_layout (one per analysis domain). encoding maps the kind's visual channels to columns (e.g. {'x': col, 'y': col}; heatmap defaults to all numeric columns). format is svg (default, sanitized to an inert document) or png. The artifact is returned inline in the envelope, extractable through the Output surface. | TABULAR | NONE | no | `endpoint?`, `path?`, `table?`, `query?`, `kind`, `encoding?`, `format?`, `title?` |
