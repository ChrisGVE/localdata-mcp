<!-- MACHINE-WRITTEN by localdata_mcp.nexus.contract.generators.docs — DO NOT EDIT; regenerate via `python -m localdata_mcp.nexus.contract.generate` -->

# Tools — explore

| Tool | Summary | Input shape | Output shape | Streaming | Params |
|---|---|---|---|---|---|
| `describe_database` | Describe a declared endpoint's schema: SQL kinds return the table catalog (columns, keys, row counts), kv/tree stores their key-space shape, graph stores their node/edge shape, rdf stores their triple shape. | NONE | SCALAR | no | `endpoint` |
| `describe_table` | Describe one table of a declared SQL-kind endpoint: columns with types and nullability, primary key, row count. | NONE | SCALAR | no | `endpoint`, `table` |
| `find_table` | Find tables on a declared SQL-kind endpoint whose names match a glob pattern (e.g. 'sales_*'). | NONE | SCALAR | no | `endpoint`, `name_pattern` |
