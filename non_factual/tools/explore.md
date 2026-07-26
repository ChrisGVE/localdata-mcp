<!-- MACHINE-WRITTEN by localdata_mcp.nexus.contract.generators.docs — DO NOT EDIT; regenerate via `python -m localdata_mcp.nexus.contract.generate` -->

# Tools — explore

| Tool | Summary | Input shape | Output shape | Streaming | Params |
|---|---|---|---|---|---|
| `describe_database` | Describe a declared endpoint's schema: SQL kinds return the table catalog (columns, keys, row counts), kv/tree stores their key-space shape, graph stores their node/edge shape, rdf stores their triple shape. | NONE | SCALAR | no | `endpoint` |
| `describe_table` | Describe one table of a declared SQL-kind endpoint: columns with types and nullability, primary key, row count. | NONE | SCALAR | no | `endpoint`, `table` |
| `find_table` | Find tables on a declared SQL-kind endpoint whose names match a glob pattern (e.g. 'sales_*'). | NONE | SCALAR | no | `endpoint`, `name_pattern` |
| `profile_data` | Profile a tabular source's data quality: per-column null counts, inferred types, numeric ranges, and cardinality. Address with exactly one of endpoint= or path=; endpoint sources take exactly one of table= or query=. | TABULAR | SCALAR | no | `endpoint?`, `path?`, `table?`, `query?` |
| `search_data` | Regex-search a tabular source's cell values. Address with exactly one of endpoint= or path=; target= is the table or SQL statement to search (omit for a document/table file); query= is the search pattern. | TABULAR | SCALAR | no | `endpoint?`, `path?`, `target?`, `query`, `columns?`, `case_sensitive?` |
| `map_categories` | Map one column's categorical values: distinct values with frequencies and a suggested encoding (label vs one-hot) — a report only, nothing is transformed or persisted. Address with exactly one of endpoint= or path=. | TABULAR | SCALAR | no | `endpoint?`, `path?`, `table?`, `query?`, `column` |
