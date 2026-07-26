<!-- MACHINE-WRITTEN by localdata_mcp.nexus.contract.generators.docs — DO NOT EDIT; regenerate via `python -m localdata_mcp.nexus.contract.generate` -->

# Tools — output

| Tool | Summary | Input shape | Output shape | Streaming | Params |
|---|---|---|---|---|---|
| `export_result` | Write a result to a file in any supported format. format is one of csv, parquet, arrow, json, excel, markdown (tabular data), schema (a table mapping), graph or tree (a structure mapping), or svg/png (a rendered chart artifact). path is the destination file (inside allowed_paths). The source is exactly one of: source (inline data — a records list, a mapping, or a rendered chart envelope), stream_id (a buffered result drained to the file), or — as a composition terminal — the upstream stage's output, injected automatically when both are omitted. An existing target is refused unless overwrite=true. Round-trip fidelity is type-preserving for parquet/arrow, documented-lossy for markdown. | TABULAR | NONE | no | `format`, `path`, `source?`, `stream_id?`, `overwrite?` |
