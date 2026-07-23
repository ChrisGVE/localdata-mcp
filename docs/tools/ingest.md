<!-- MACHINE-WRITTEN by localdata_mcp.nexus.contract.generators.docs — DO NOT EDIT; regenerate via `python -m localdata_mcp.nexus.contract.generate` -->

# Tools — ingest

| Tool | Summary | Input shape | Output shape | Streaming | Params |
|---|---|---|---|---|---|
| `list_endpoints` | Enumerate every operator-declared endpoint (SQL, key-value, and graph/tree alike) with its backend kind, posture, and health. | NONE | TABULAR | no | — |
| `fetch_chunk` | Retrieve the next servable chunk of a streamed result (cursor semantics: a served chunk leaves the buffer). Once the source is exhausted the answer reports the final total and the stream closes. | NONE | TABULAR | yes | `stream_id` |
| `close_stream` | Release a streamed result ahead of the idle TTL, returning its buffer memory (and any pinned connection) immediately. Idempotent. | NONE | SCALAR | no | `stream_id` |
| `query` | Run a read-only SQL statement against a declared endpoint and return the rows (guarded: allow-list validated, any posture). | NONE | TABULAR | yes | `endpoint`, `sql` |
| `write_query` | Run a mutating SQL statement (INSERT/UPDATE/DELETE or a write-side local-file construct) against a declared read-write endpoint (guarded: posture enforced, allow-list validated). | NONE | TABULAR | no | `endpoint`, `sql` |
| `read_file` | Read a local data file (14 core formats: CSV, TSV, JSON, YAML, TOML, INI, XML, Excel, ODS, Numbers, Parquet, Feather, Arrow, HDF5) inside the operator's allowed paths. | NONE | TABULAR | yes | `path`, `format?` |
| `query_file` | Run a read-only SQL statement over a local SQLite or DuckDB file (ad-hoc, contained, read-only unless the operator grants otherwise; results are read whole under the memory budget). | NONE | TABULAR | yes | `path`, `sql` |
| `get_value` | Get one property value from a node of a declared kv, tree, or graph store endpoint (path addresses the node; node_id for graph stores). | NONE | SCALAR | no | `endpoint`, `path`, `key` |
| `set_value` | Set (upsert) one property on a node of a declared read-write store endpoint, auto-creating the node; string values infer their type unless value_type names one. | NONE | SCALAR | no | `endpoint`, `path`, `key`, `value`, `value_type?` |
| `delete_key` | Delete one property from a node of a declared read-write store endpoint. | NONE | SCALAR | no | `endpoint`, `path`, `key` |
| `list_keys` | List a node's properties (key, value, value_type) from a declared kv, tree, or graph store endpoint, key-ordered. | NONE | TABULAR | no | `endpoint`, `path`, `offset?`, `limit?` |
| `get_node` | Get node details from a declared tree or graph store endpoint (counts and addressing; properties via list_keys); omit path for the store-level summary. | NONE | SCALAR | no | `endpoint`, `path?` |
| `set_node` | Create a node on a declared read-write store endpoint: tree kinds create the path (and missing ancestors), graph kinds upsert the node with an optional label. | NONE | SCALAR | no | `endpoint`, `path`, `label?` |
| `delete_node` | Delete a node from a declared read-write store endpoint: tree kinds delete the whole subtree (properties cascade), graph kinds cascade the node's edges and properties. | NONE | SCALAR | no | `endpoint`, `path` |
| `get_children` | List direct children of a tree-store node (root nodes when path is omitted), name-ordered with counts. | NONE | TABULAR | no | `endpoint`, `path?`, `offset?`, `limit?` |
| `move_node` | Move a tree-store node and its whole subtree under a new parent (or to root level when new_parent is omitted). | NONE | SCALAR | no | `endpoint`, `path`, `new_parent?` |
| `get_neighbors` | List a graph node's neighbors with edge label/weight and direction ('in', 'out', or 'both'). | NONE | TABULAR | no | `endpoint`, `node_id`, `direction?`, `offset?`, `limit?` |
| `get_edges` | List a graph store's edges (source, target, label, weight), optionally filtered to those touching one node. | NONE | TABULAR | no | `endpoint`, `node_id?`, `offset?`, `limit?` |
| `add_edge` | Add (or re-weight) a directed edge on a declared read-write graph endpoint, auto-creating missing nodes; returns the harvested integrity warnings (self-loop, duplicates, contradictory reverse edge). | NONE | SCALAR | no | `endpoint`, `source`, `target`, `label?`, `weight?` |
| `remove_edge` | Remove a directed edge (and its properties) from a declared read-write graph endpoint; warns when a node becomes an orphan. | NONE | SCALAR | no | `endpoint`, `source`, `target`, `label?` |
| `find_path` | Find path(s) between two graph nodes: the shortest path, or all simple paths (bounded enumeration). | NONE | SCALAR | no | `endpoint`, `source`, `target`, `algorithm?` |
| `get_graph_stats` | Summary statistics for a declared graph endpoint: node, edge, and property counts plus density. | NONE | SCALAR | no | `endpoint` |
