<!-- MACHINE-WRITTEN by localdata_mcp.nexus.contract.generators.docs — DO NOT EDIT; regenerate via `python -m localdata_mcp.nexus.contract.generate` -->

# Tools — ingest

| Tool | Summary | Input shape | Output shape | Streaming | Params |
|---|---|---|---|---|---|
| `list_endpoints` | Enumerate every operator-declared endpoint (SQL, key-value, and graph/tree alike) with its backend kind, posture, and health. | NONE | TABULAR | no | — |
| `query` | Run a read-only SQL statement against a declared endpoint and return the rows (guarded: allow-list validated, any posture). | NONE | TABULAR | no | `endpoint`, `sql` |
| `write_query` | Run a mutating SQL statement (INSERT/UPDATE/DELETE or a write-side local-file construct) against a declared read-write endpoint (guarded: posture enforced, allow-list validated). | NONE | TABULAR | no | `endpoint`, `sql` |
