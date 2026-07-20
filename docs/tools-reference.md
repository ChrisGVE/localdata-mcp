# LocalData MCP Tools Reference

Complete reference documentation for all 53 MCP tools. Organized by category for quick navigation.

## Table of Contents

1. Core Database (8 tools)
2. Streaming & Memory (9 tools)
3. Tree/Structured Data (10 tools)
4. Graph Operations (7 tools)
5. Search & Transform (2 tools)
6. Schema & Audit (3 tools)
7. System (2 tools)
8. Data Science (12 tools)

The tree tools in section 3 double as the node-level graph API: `get_node`,
`set_node`, `delete_node`, `list_keys`, `get_value`, `set_value`, and
`delete_key` detect a graph connection and read their `path` argument as a node
ID. `get_children` and `move_node` work on trees only.

---

## Core Database (8 tools)

### connect_database

Open a connection to a database.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `name` | string | Yes | Unique connection identifier (e.g., "analytics_db", "user_data") |
| `db_type` | string | Yes | Database type: sqlite, postgresql, mysql, duckdb, csv, json, yaml, toml, excel, ods, numbers, xml, ini, tsv, parquet, feather, arrow, hdf5, dot, gml, graphml, mermaid, turtle, ntriples, sparql |
| `conn_string` | string | Yes | Connection string or file path |
| `sheet_name` | string | No | Sheet name for Excel/ODS/Numbers or dataset name for HDF5 |
| `auth` | string | No | JSON authentication config (e.g., `{"method": "wallet", "wallet_path": "/path"}`) |

**Returns:** Connection summary with metadata (JSON)

**Example:**
```python
# SQL database
connect_database("mydb", "postgresql", "postgresql://user:pass@localhost/dbname")

# CSV file
connect_database("data", "csv", "/path/to/file.csv")

# Graph file
connect_database("network", "graphml", "/path/to/network.graphml")
```

**Composition hints:** Use with `execute_query`, `describe_database`, or data manipulation tools.

---

### disconnect_database

Close a connection to a database. All connections close automatically on script termination.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `name` | string | Yes | Connection name to close |

**Returns:** Success/error message with cleanup details (JSON)

**Example:**
```python
disconnect_database("mydb")
```

**Composition hints:** Call after completing work with a database to free resources.

---

### execute_query

Execute a SQL query and return results as JSON with memory-aware streaming.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `name` | string | Yes | Database connection name |
| `query` | string | Yes | SQL query to execute |
| `chunk_size` | integer | No | Results per chunk for pagination |
| `enable_analysis` | boolean | No | Perform pre-query analysis (default: true) |
| `include_blobs` | boolean | No | Base64-encode BLOBs in results (default: false) |
| `preflight` | boolean | No | Run EXPLAIN only (default: false) |

**Returns:** Query results as JSON or streaming metadata (JSON)

**Example:**
```python
execute_query("mydb", "SELECT * FROM users WHERE active = true", chunk_size=100)
```

**Composition hints:** Use with `next_chunk` for large result sets, `get_query_metadata` for analysis.

---

### analyze_query_preview

Analyze a query without executing it to preview resource requirements.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `name` | string | Yes | Database connection name |
| `query` | string | Yes | SQL query to analyze |

**Returns:** Analysis including estimated rows, memory, execution time, and risks (JSON)

**Example:**
```python
analyze_query_preview("mydb", "SELECT * FROM large_table JOIN other_table ON ...")
```

**Composition hints:** Use before `execute_query` on complex queries to assess feasibility.

---

### list_databases

List all available database connections with their SQL flavor information.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `include_staging` | boolean | No | Include active staging databases (default: false) |

**Returns:** Array of connection objects with names and types (JSON)

**Example:**
```python
list_databases()
```

**Composition hints:** Use to discover available connections for workflow orchestration.

---

### describe_database

Get detailed information about a database including its schema in JSON format.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `name` | string | Yes | Database connection name |

**Returns:** Database schema with tables, columns, types, and relationships (JSON)

**Example:**
```python
describe_database("mydb")
```

**Composition hints:** Use with `find_table` to locate specific tables, or `describe_table` for details.

---

### find_table

Find which database contains a specific table by name.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `table_name` | string | Yes | Name of table to locate |

**Returns:** Connection name containing the table, or error if not found (JSON)

**Example:**
```python
find_table("users")
```

**Composition hints:** Use in multi-database workflows to locate data without knowing connection.

---

### describe_table

Get detailed schema information for a specific table.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `name` | string | Yes | Database connection name |
| `table_name` | string | Yes | Name of table to describe |

**Returns:** Table schema with columns, types, constraints, and indexes (JSON)

**Example:**
```python
describe_table("mydb", "users")
```

**Composition hints:** Use before writing queries to understand table structure and column types.

---

## Streaming & Memory (9 tools)

### next_chunk

Retrieve the next chunk of rows from a buffered query result.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `query_id` | string | Yes | ID of buffered query result |
| `start_row` | integer | Yes | Starting row number (1-based) |
| `chunk_size` | string | Yes | Number of rows or "all" for remaining |

**Returns:** Chunk of rows with metadata (JSON)

**Example:**
```python
next_chunk("mydb_12345_abc1", 1, "100")
```

**Composition hints:** Chain multiple calls to paginate through large result sets.

---

### manage_memory_bounds

Monitor and manage memory usage across all streaming operations.

**Parameters:** None

**Returns:** Memory status, usage statistics, and cleanup actions taken (JSON)

**Example:**
```python
manage_memory_bounds()
```

**Composition hints:** Call when memory warnings appear or before large operations.

---

### get_streaming_status

Get detailed status of all active streaming operations and memory usage.

**Parameters:** None

**Returns:** Active buffers, memory usage, performance metrics (JSON)

**Example:**
```python
get_streaming_status()
```

**Composition hints:** Monitor streaming workloads and identify bottlenecks.

---

### clear_streaming_buffer

Clear a specific streaming result buffer to free memory immediately.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `query_id` | string | Yes | ID of buffer to clear |

**Returns:** Confirmation message (JSON)

**Example:**
```python
clear_streaming_buffer("mydb_12345_abc1")
```

**Composition hints:** Use after finishing with a large result set.

---

### get_query_metadata

Get comprehensive metadata for a query result including quality metrics and processing recommendations.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `query_id` | string | Yes | ID of query result |

**Returns:** LLM-friendly summary, quality metrics, complexity analysis (JSON)

**Example:**
```python
get_query_metadata("mydb_12345_abc1")
```

**Composition hints:** Use with LLM-based analysis workflows for decision making.

---

### request_data_chunk

Retrieve a specific chunk of data using the LLM communication protocol.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `query_id` | string | Yes | ID of query result |
| `chunk_id` | integer | Yes | Chunk ID to retrieve (0-based) |

**Returns:** Chunk data with metadata (JSON)

**Example:**
```python
request_data_chunk("mydb_12345_abc1", 0)
```

**Composition hints:** Use for targeted chunk retrieval in progressive loading workflows.

---

### request_multiple_chunks

Retrieve multiple chunks efficiently in a single call.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `query_id` | string | Yes | ID of query result |
| `chunk_ids` | string | Yes | Comma-separated chunk IDs (e.g., "0,1,2") |

**Returns:** Multiple chunks with metadata (JSON)

**Example:**
```python
request_multiple_chunks("mydb_12345_abc1", "0,1,2,3,4")
```

**Composition hints:** Use to load specific chunks in parallel.

---

### cancel_query_operation

Cancel an ongoing query operation and free resources.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `query_id` | string | Yes | ID of query to cancel |
| `reason` | string | No | Reason for cancellation (default: "User requested") |

**Returns:** Cancellation status message (JSON)

**Example:**
```python
cancel_query_operation("mydb_12345_abc1", "User interrupt")
```

**Composition hints:** Use on long-running queries to stop execution.

---

### get_data_quality_report

Get comprehensive data quality assessment for a query result.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `query_id` | string | Yes | ID of query to assess |

**Returns:** Detailed quality report including nulls, duplicates, outliers (JSON)

**Example:**
```python
get_data_quality_report("mydb_12345_abc1")
```

**Composition hints:** Use before analytics to understand data cleanliness.

---

## Tree/Structured Data (10 tools)

### get_node

Get node details or summary for tree, graph, or RDF connections.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `name` | string | Yes | Connection name |
| `path` | string | No | Node path (tree), node_id (graph), or subject URI (RDF) |

**Returns:** Node details or root summary (JSON)

**Example:**
```python
get_node("config", "server/host")
get_node("network", "node-123")
```

**Composition hints:** Use with `get_children` to traverse hierarchies.

---

### get_children

Get children of a node with pagination.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `name` | string | Yes | Connection name |
| `path` | string | No | Parent node path (null for root) |
| `offset` | integer | No | Pagination offset (default: 0) |
| `limit` | integer | No | Rows per page (default: 50) |

**Returns:** Array of child nodes with metadata (JSON)

**Example:**
```python
get_children("config", "database", offset=0, limit=20)
```

**Composition hints:** Chain calls with different offsets to paginate through large hierarchies.

---

### set_node

Create a node in tree or graph connection.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `name` | string | Yes | Connection name |
| `path` | string | Yes | Path for new node |
| `label` | string | No | Display label (graph only) |

**Returns:** Confirmation with node details (JSON)

**Example:**
```python
set_node("config", "app/features/new_feature", "New Feature")
```

**Composition hints:** Use with `set_value` to add properties to nodes.

---

### move_node

Move a node and its subtree under a new parent or to root.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `name` | string | Yes | Connection name |
| `path` | string | Yes | Node path to move |
| `new_parent` | string | No | New parent path (null for root) |

**Returns:** Confirmation with new location (JSON)

**Example:**
```python
move_node("config", "old/location/node", "new/location")
```

**Composition hints:** Use for reorganizing hierarchical structures.

---

### delete_node

Delete a node and all its descendants.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `name` | string | Yes | Connection name |
| `path` | string | Yes | Node path to delete |

**Returns:** Confirmation with deletion details (JSON)

**Example:**
```python
delete_node("config", "deprecated/feature")
```

**Composition hints:** Use carefully as deletion cascades to all children.

---

### list_keys

List key-value pairs at a node with pagination.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `name` | string | Yes | Connection name |
| `path` | string | Yes | Node path |
| `offset` | integer | No | Pagination offset (default: 0) |
| `limit` | integer | No | Results per page (default: 50) |

**Returns:** Array of key-value pairs (JSON)

**Example:**
```python
list_keys("config", "database")
```

**Composition hints:** Use with `get_value` to inspect node properties.

---

### get_value

Get a specific property value from a node.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `name` | string | Yes | Connection name |
| `path` | string | Yes | Node path |
| `key` | string | Yes | Property key to retrieve |

**Returns:** Property value and metadata (JSON)

**Example:**
```python
get_value("config", "database", "host")
```

**Composition hints:** Use to read individual node properties.

---

### set_value

Set a property on a node (auto-creates node if needed).

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `name` | string | Yes | Connection name |
| `path` | string | Yes | Node path |
| `key` | string | Yes | Property key |
| `value` | string | Yes | Property value |
| `value_type` | string | No | Data type hint |

**Returns:** Confirmation with updated node (JSON)

**Example:**
```python
set_value("config", "database", "host", "localhost", "string")
```

**Composition hints:** Use to modify node properties.

---

### delete_key

Delete a property from a node.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `name` | string | Yes | Connection name |
| `path` | string | Yes | Node path |
| `key` | string | Yes | Property key to delete |

**Returns:** Confirmation with remaining properties (JSON)

**Example:**
```python
delete_key("config", "database", "deprecated_setting")
```

**Composition hints:** Use to remove obsolete node properties.

---

### export_structured

Export tree data as TOML, JSON, YAML, or Markdown — or RDF data as Turtle or
N-Triples. The accepted formats depend on the connection type; passing an RDF
format to a tree connection (or the reverse) returns an error.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `name` | string | Yes | Connection name |
| `format` | string | Yes | Tree connections: `json`, `yaml`, `toml`, `markdown` (alias `md`). RDF connections: `turtle` (alias `ttl`), `ntriples` (alias `nt`) |
| `path` | string | No | Root path to export (null for all). Ignored for RDF connections, which always serialize the whole graph |

**Returns:** `{"format": ..., "content": ...}`. Output above 100 KB is halved and
returned with `truncated: true` and a `notice`

**Example:**
```python
export_structured("config", "yaml", "server")
```

**Composition hints:** Use for data portability and backup.

---

## Graph Operations (7 tools)

### get_neighbors

Get neighbors of a graph node with edge information.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `name` | string | Yes | Connection name |
| `node_id` | string | Yes | Node ID |
| `direction` | string | No | Direction: "in", "out", or "both" (default: both) |
| `offset` | integer | No | Pagination offset (default: 0) |
| `limit` | integer | No | Results per page (default: 50) |

**Returns:** Array of neighbors with edge information (JSON)

**Example:**
```python
get_neighbors("network", "user-123", direction="out")
```

**Composition hints:** Use for network analysis and traversal.

---

### get_edges

List edges in a graph, optionally filtered by node.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `name` | string | Yes | Connection name |
| `node_id` | string | No | Filter by node (null for all edges) |
| `offset` | integer | No | Pagination offset (default: 0) |
| `limit` | integer | No | Results per page (default: 50) |

**Returns:** Array of edges with source, target, and properties (JSON)

**Example:**
```python
get_edges("network", node_id="user-123")
```

**Composition hints:** Use for graph structure analysis and export.

---

### add_edge

Add an edge to a graph (auto-creates nodes if needed).

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `name` | string | Yes | Connection name |
| `source` | string | Yes | Source node ID |
| `target` | string | Yes | Target node ID |
| `label` | string | No | Edge label or relationship type |
| `weight` | number | No | Edge weight (for weighted graphs) |

**Returns:** Confirmation with edge details (JSON)

**Example:**
```python
add_edge("network", "user-1", "user-2", "follows", weight=1.0)
```

**Composition hints:** Use to build or modify graphs programmatically.

---

### remove_edge

Remove an edge from a graph.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `name` | string | Yes | Connection name |
| `source` | string | Yes | Source node ID |
| `target` | string | Yes | Target node ID |
| `label` | string | No | Edge label (for filtering) |

**Returns:** Confirmation of removal (JSON)

**Example:**
```python
remove_edge("network", "user-1", "user-2", "follows")
```

**Composition hints:** Use to modify graph topology.

---

### find_path

Find path(s) between two graph nodes.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `name` | string | Yes | Connection name |
| `source` | string | Yes | Start node ID |
| `target` | string | Yes | End node ID |
| `algorithm` | string | No | Algorithm: "shortest" (default) or "all" |

**Returns:** Path(s) with nodes and edges (JSON)

**Example:**
```python
find_path("network", "user-1", "user-5", algorithm="shortest")
```

**Composition hints:** Use for network analysis and influence tracing.

---

### get_graph_stats

Get advanced graph statistics including centrality measures.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `name` | string | Yes | Connection name |

**Returns:** Graph metrics: node count, edge count, density, diameter, clustering (JSON)

**Example:**
```python
get_graph_stats("network")
```

**Composition hints:** Use to characterize network properties.

---

### export_graph

Export a graph in a machine-readable or agent-readable format.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `name` | string | Yes | Connection name |
| `format` | string | Yes | `dot`, `gml`, `graphml`, `mermaid` (alias `mmd`), `markdown` (alias `md`), `hierarchy`, `adjacency`, or `detailed` |
| `node_id` | string | No | Export the ego subgraph around this node (null for the whole graph). Ignored by the Markdown styles, which always render the full graph |

`hierarchy`, `adjacency`, and `detailed` are Markdown styles: an indented tree
for DAGs, a compact `A -> B [label]` list, and full per-node property sections
respectively. Output above 100 KB is truncated and flagged with `truncated: true`
plus a `notice`. For the non-Markdown formats, pass `node_id` to export a smaller
subgraph instead.

**Returns:** `{"format": ..., "content": ...}`, or `{"error": ...}` for an unknown format or missing node

**Example:**
```python
export_graph("network", "graphml")
export_graph("network", "adjacency")
export_graph("network", "mermaid", node_id="root")
```

**Composition hints:** Use for graph visualization and portability.

---

## Search & Transform (2 tools)

### search_data

Search query results for regex pattern matches.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `name` | string | Yes | Database connection name |
| `query` | string | Yes | SQL query to search within |
| `pattern` | string | Yes | Regex pattern to find |
| `columns` | string | No | Comma-separated column names (null for all) |
| `case_sensitive` | boolean | No | Case-sensitive search (default: true) |
| `max_matches` | integer | No | Maximum matches to return (default: 100) |

**Returns:** Matching rows with metadata (JSON)

**Example:**
```python
search_data("mydb", "SELECT * FROM emails", ".*@company\\.com", columns="email", case_sensitive=False)
```

**Composition hints:** Use for content discovery and data validation.

---

### transform_data

Apply regex find/replace to a column in query results.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `name` | string | Yes | Database connection name |
| `query` | string | Yes | SQL query to transform |
| `column` | string | Yes | Column name to apply transformation |
| `find` | string | Yes | Regex pattern to find |
| `replace` | string | Yes | Replacement string (supports capture groups) |
| `max_rows` | integer | No | Maximum rows to process (default: 1000) |

**Returns:** Transformed data preview (JSON)

**Example:**
```python
transform_data("mydb", "SELECT * FROM logs", "message", "ERROR: (.*)", "CRITICAL: $1")
```

**Composition hints:** Use for data cleaning and standardization.

---

## Schema & Audit (3 tools)

### export_schema

Export database schema in various formats.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `name` | string | Yes | Database connection name |
| `format` | string | No | Format: json_schema, python, typescript, sql_ddl (default: json_schema) |
| `tables` | string | No | Comma-separated table names (null for all) |

**Returns:** Schema in requested format (string/JSON)

**Example:**
```python
export_schema("mydb", format="typescript", tables="users,posts")
```

**Composition hints:** Use for code generation and documentation.

---

### get_query_log

Get recent query execution history.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `database` | string | No | Filter by database (null for all) |
| `status` | string | No | Filter by status: success, error, timeout |
| `since_minutes` | integer | No | Look back this many minutes (default: 60) |
| `limit` | integer | No | Maximum entries to return (default: 50) |

**Returns:** Query history entries with execution details (JSON)

**Example:**
```python
get_query_log(database="mydb", status="error", since_minutes=30)
```

**Composition hints:** Use for debugging and performance analysis.

---

### get_error_log

Get recent error and timeout history.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `database` | string | No | Filter by database (null for all) |
| `since_minutes` | integer | No | Look back this many minutes (default: 60) |
| `limit` | integer | No | Maximum entries to return (default: 50) |

**Returns:** Error entries with context and suggestions (JSON)

**Example:**
```python
get_error_log(database="mydb", since_minutes=60)
```

**Composition hints:** Use for troubleshooting connection and query issues.

---

## System (2 tools)

### check_compatibility

Check backward compatibility status and get migration recommendations.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `generate_migration_script` | boolean | No | Generate migration script for legacy config (default: false) |

**Returns:** Compatibility report and migration guidance (JSON)

**Example:**
```python
check_compatibility(generate_migration_script=True)
```

**Composition hints:** Use during upgrades or configuration migrations.

---

### get_metrics

Return the server's Prometheus metrics as text.

This tool is registered only when metrics collection is enabled
(`logging.enable_metrics`), which is the default. Disable it and the tool
disappears from the tool list, leaving 52 tools.

**Parameters:** none

**Returns:** Prometheus exposition-format metrics (string). On failure, the
string `Error exporting metrics: <reason>` rather than an exception.

**Example:**
```python
get_metrics()
```

**Composition hints:** Use to check memory pressure and query throughput before
launching a large streaming job.

---

## Data Science (12 tools)

Every tool in this section has the same first two parameters: the name of a live
connection and a SQL query. The query selects the data; there is no separate
load step and no data-frame parameter. Column parameters name columns in the
query's result set.

Where a parameter takes a list of columns, it is a genuine list
(`["price", "sqft"]`), not a comma-separated string. Where a parameter selects a
method, only the values listed below are accepted; anything else raises
`ValueError`.

```{warning}
Verified by execution against a live connection: `analyze_anova`,
`analyze_effect_sizes`, `evaluate_model_performance`, `analyze_clusters`,
`detect_anomalies`, `reduce_dimensions`, `analyze_time_series`,
`forecast_time_series`, `analyze_rfm` and `analyze_ab_test` currently raise
`TypeError` inside the server before returning a result, as does
`analyze_hypothesis_test` when `column` or `group_column` is supplied. The
signatures documented below are correct; the adapter layer that forwards them to
the analysis code is not. Tracked as
[issue #23](https://github.com/ChrisGVE/localdata-mcp/issues/23).
`analyze_regression`, and `analyze_hypothesis_test` without column arguments,
work as documented.
```

### analyze_hypothesis_test

Run a statistical hypothesis test on query results.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `connection_name` | string | Yes | Database connection name |
| `query` | string | Yes | SQL query returning data |
| `test_type` | string | No | `auto`, `ttest_1samp`, `ttest_ind`, `ttest_rel`, `chi2`, `normality`, `correlation` (default: `auto`) |
| `column` | string | No | Column to test (default: empty, meaning all numeric columns) |
| `group_column` | string | No | Grouping column for two-sample tests |
| `alpha` | number | No | Significance level (default: 0.05) |
| `alternative` | string | No | `two-sided`, `less`, or `greater` (default: `two-sided`) |

**Returns:** A `test_results` list, one entry per test performed, each with
`test_name`, `statistic`, `p_value`, and an `interpretation` string (JSON).

**Example:**
```python
analyze_hypothesis_test("mydb", "SELECT score FROM experiments", test_type="ttest_1samp")
```

**Composition hints:** With `test_type="auto"` and no column named, this reports
normality and correlation checks across the numeric columns — a reasonable first
call before choosing a specific test.

---

### analyze_anova

Compare group means with a one-way ANOVA.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `connection_name` | string | Yes | Database connection name |
| `query` | string | Yes | SQL query with the dependent and grouping columns |
| `dependent_var` | string | Yes | Column holding the numeric values being compared |
| `group_var` | string | Yes | Column defining the groups |
| `alpha` | number | No | Significance level (default: 0.05) |

**Returns:** F-statistic, p-value, per-group means, and effect size (JSON)

**Example:**
```python
analyze_anova("mydb", "SELECT revenue, region FROM sales", "revenue", "region")
```

**Composition hints:** Follow a significant result with `analyze_effect_sizes`
to judge whether the difference matters in practice.

---

### analyze_effect_sizes

Calculate effect sizes for a comparison between groups.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `connection_name` | string | Yes | Database connection name |
| `query` | string | Yes | SQL query returning the value and grouping columns |
| `column` | string | Yes | Numeric column to compare |
| `group_column` | string | Yes | Column defining the groups |

There is no `effect_type` parameter; the measure follows from the data. The tool
takes four arguments, not five.

**Returns:** Effect size value and interpretation (JSON)

**Example:**
```python
analyze_effect_sizes("mydb", "SELECT score, treatment FROM experiments", "score", "treatment")
```

**Composition hints:** Pair with a hypothesis test — significance answers
"is there a difference", effect size answers "is it worth acting on".

---

### analyze_regression

Fit a regression model to query results.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `connection_name` | string | Yes | Database connection name |
| `query` | string | Yes | SQL query with the feature and target columns |
| `target_column` | string | Yes | Column to predict |
| `feature_columns` | list of strings | No | Columns to use as features. Omit to use every numeric column except the target |
| `model_type` | string | No | `linear`, `ridge`, `lasso`, `elastic_net`, `logistic`, `polynomial` (default: `linear`) |
| `regularization` | string | No | `l1`, `l2`, `elastic_net`, or empty for none (default: empty) |

**Returns:** `model_type`, the resolved `pipeline_config`, a
`regression_analysis` block with coefficients and fit statistics, and a
`residual_analysis` block for non-logistic models (JSON)

**Example:**
```python
analyze_regression("mydb", "SELECT * FROM properties", "price", ["sqft", "bedrooms", "year_built"], model_type="ridge")
```

**Composition hints:** Use `model_type="logistic"` for a binary target. Pass the
model's predictions back through `evaluate_model_performance` to score them.

---

### evaluate_model_performance

Score predictions that are already stored alongside their actual values.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `connection_name` | string | Yes | Database connection name |
| `query` | string | Yes | SQL query returning both the actual and the predicted column |
| `target_column` | string | Yes | Column with the actual values |
| `prediction_column` | string | Yes | Column with the predicted values |
| `model_type` | string | No | `regression` or `classification` (default: `regression`) |

The metric set follows from `model_type`; there is no `metric_type` parameter.

**Returns:** Performance metrics and diagnostics for the chosen model type (JSON)

**Example:**
```python
evaluate_model_performance("mydb", "SELECT actual_price, predicted_price FROM test_results", "actual_price", "predicted_price")
```

**Composition hints:** Use after scoring a model whose predictions you have
written back to the database.

---

### analyze_clusters

Group rows by similarity across numeric columns.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `connection_name` | string | Yes | Database connection name |
| `query` | string | Yes | SQL query returning the columns to cluster on |
| `columns` | list of strings | No | Columns to cluster on. Omit to use every numeric column |
| `method` | string | No | `kmeans`, `dbscan`, `hierarchical`, `gaussian_mixture` (default: `kmeans`) |
| `n_clusters` | integer | No | Number of clusters. Omit to let the method choose |

**Returns:** Cluster assignments, centroids, and a silhouette score (JSON)

**Example:**
```python
analyze_clusters("mydb", "SELECT * FROM customers", ["spending", "frequency", "recency"], method="kmeans", n_clusters=5)
```

**Composition hints:** Reduce dimensionality first when clustering on more than a
handful of columns.

---

### detect_anomalies

Flag rows that do not fit the rest of the data.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `connection_name` | string | Yes | Database connection name |
| `query` | string | Yes | SQL query returning the columns to examine |
| `columns` | list of strings | No | Columns to examine. Omit to use every numeric column |
| `method` | string | No | `isolation_forest`, `local_outlier_factor`, `one_class_svm` (default: `isolation_forest`) |
| `contamination` | number | No | Expected proportion of anomalies, 0.0 to 0.5 (default: 0.1) |

These are multivariate detectors over a set of columns. There is no single
`column` parameter, no `threshold`, and no z-score or IQR method.

**Returns:** Anomaly flags, scores, and the flagged rows (JSON)

**Example:**
```python
detect_anomalies("mydb", "SELECT * FROM transactions", ["amount", "item_count"], method="isolation_forest", contamination=0.05)
```

**Composition hints:** Use for data quality screening and fraud detection.
Lower `contamination` when false positives are expensive.

---

### reduce_dimensions

Project numeric columns onto fewer dimensions.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `connection_name` | string | Yes | Database connection name |
| `query` | string | Yes | SQL query returning the columns to reduce |
| `columns` | list of strings | No | Columns to include. Omit to use every numeric column |
| `method` | string | No | `pca`, `tsne`, `umap` (default: `pca`) |
| `n_components` | integer | No | Target number of dimensions (default: 2) |

**Returns:** The reduced components and, for PCA, explained variance (JSON)

**Example:**
```python
reduce_dimensions("mydb", "SELECT * FROM gene_data", ["gene_a", "gene_b", "gene_c"], method="pca", n_components=3)
```

**Composition hints:** Run before `analyze_clusters` on wide data. `pca` is
reversible and cheap; `tsne` and `umap` are for visualization, not for feeding
another model.

---

### analyze_time_series

Decompose a series and test it for stationarity.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `connection_name` | string | Yes | Database connection name |
| `query` | string | Yes | SQL query returning the date and value columns |
| `date_column` | string | Yes | Column with datetime values |
| `value_column` | string | Yes | Column with numeric values |
| `frequency` | string | No | `D`, `W`, `M`, `Q`, `Y`, or empty to infer (default: empty) |

The frequency codes are the pandas offset aliases, not words like `daily`.

**Returns:** Trend, seasonal and residual components, plus a stationarity test
(JSON)

**Example:**
```python
analyze_time_series("mydb", "SELECT date, price FROM stock_prices", "date", "price", frequency="D")
```

**Composition hints:** Run before `forecast_time_series` — a non-stationary
series usually needs differencing or a different model.

---

### forecast_time_series

Project a series forward.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `connection_name` | string | Yes | Database connection name |
| `query` | string | Yes | SQL query returning the historical date and value columns |
| `date_column` | string | Yes | Column with datetime values |
| `value_column` | string | Yes | Column with numeric values |
| `horizon` | integer | No | Number of periods to forecast (default: 10) |
| `method` | string | No | `arima`, or `ets` / `exponential_smoothing` (default: `arima`) |

Those are the only accepted values. There is no automatic model selection, and
`prophet` and `sarima` both raise
`ValueError: Unknown forecast method: <name>. Use 'arima' or 'ets'.` The SARIMA
and ensemble forecasters in `localdata_mcp.domains.time_series_analysis` are
reachable from Python but are exposed by no MCP tool.

**Returns:** Forecast values with confidence intervals (JSON)

**Example:**
```python
forecast_time_series("mydb", "SELECT date, sales FROM sales_history", "date", "sales", horizon=30, method="arima")
```

**Composition hints:** Use for demand and capacity planning. Compare `arima`
against `ets` on a held-out tail rather than trusting either by default.

---

### analyze_rfm

Segment customers by recency, frequency, and monetary value.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `connection_name` | string | Yes | Database connection name |
| `query` | string | Yes | SQL query returning transaction rows |
| `customer_column` | string | Yes | Column identifying the customer |
| `date_column` | string | Yes | Column with the transaction date |
| `value_column` | string | Yes | Column with the transaction value |

**Returns:** RFM scores, customer segments, and value tiers (JSON)

**Example:**
```python
analyze_rfm("mydb", "SELECT * FROM transactions", "customer_id", "order_date", "order_value")
```

**Composition hints:** Feed the resulting segments back into a query to size or
target each one.

---

### analyze_ab_test

Compare an experiment's variants.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `connection_name` | string | Yes | Database connection name |
| `query` | string | Yes | SQL query returning the variant and metric columns |
| `variant_column` | string | Yes | Column identifying control and treatment |
| `metric_column` | string | Yes | Column with the metric being compared |
| `alpha` | number | No | Significance level (default: 0.05) |

The threshold is `alpha`, the significance level — not a `confidence_level`. For
a 95% interval, pass `alpha=0.05`.

**Returns:** Test statistics, p-value, sample sizes, and power analysis (JSON)

**Example:**
```python
analyze_ab_test("mydb", "SELECT variant, converted FROM experiment_results", "variant", "converted", alpha=0.05)
```

**Composition hints:** Pair with `analyze_effect_sizes` on the same columns to
report lift alongside significance.

---

## Quick Reference Summary

| Category | Count | Purpose |
|----------|-------|---------|
| Core Database | 8 | Connect, query, inspect databases |
| Streaming & Memory | 9 | Handle large results, memory management |
| Tree/Structured | 10 | Hierarchical JSON, YAML, TOML data |
| Graph | 7 | Network analysis and manipulation |
| Search & Transform | 2 | Pattern matching and text replacement |
| Schema & Audit | 3 | Introspection and query history |
| System | 2 | Compatibility check and Prometheus metrics |
| Data Science | 12 | Statistical analysis and forecasting |
| **Total** | **53** | Complete LLM-native data platform |

Fifty-two tools are registered unconditionally; `get_metrics` is the
fifty-third, registered only when `logging.enable_metrics` is true, which is the
default.

## Parameter Type Reference

| Type | Format | Example |
|------|--------|---------|
| `string` | UTF-8 text | "mydb", "users", "/path/to/file" |
| `integer` | Whole numbers | 100, -1, 0 |
| `number` | Float/decimal | 0.05, 3.14, 2.5 |
| `boolean` | true/false | true, false |
| `optional` | [ ] indicates optional | Parameter with `[No]` in Required column |

## Return Format Convention

All tools return JSON-formatted responses with:

```json
{
  "status": "success|error|warning",
  "data": { },
  "metadata": { }
}
```

Successful queries return `"status": "success"`. Errors include context in `metadata` for debugging.

## Composition Patterns

### Sequential Loading
```
execute_query() -> get_query_metadata() -> request_multiple_chunks()
```

### Exploration Workflow
```
list_databases() -> describe_database() -> find_table() -> describe_table()
```

### Data Transformation
```
execute_query() -> search_data() -> transform_data() -> export_schema()
```

### Analysis Pipeline
```
execute_query() -> analyze_clusters() -> reduce_dimensions() -> get_graph_stats()
```

---

**For integration examples and advanced workflows, see the main documentation.**
