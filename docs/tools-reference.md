# LocalData MCP Tools Reference

Complete reference documentation for all 52 MCP tools. Organized by category for quick navigation.

## Table of Contents

1. Core Database (8 tools)
2. Streaming & Memory (9 tools)
3. Tree/Structured Data (10 tools)
4. Graph Operations (7 tools)
5. Search & Transform (2 tools)
6. Schema & Audit (3 tools)
7. System (1 tool)
8. Data Science (12 tools)

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

Export tree or RDF data as TOML, JSON, YAML, Turtle, or N-Triples.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `name` | string | Yes | Connection name |
| `format` | string | Yes | Export format: json, yaml, toml, turtle, ntriples |
| `path` | string | No | Root path to export (null for all) |

**Returns:** Exported data in requested format (string)

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

Export graph as DOT, GML, or GraphML format.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `name` | string | Yes | Connection name |
| `format` | string | Yes | Format: dot, gml, graphml |
| `node_id` | string | No | Export subgraph from node (null for all) |

**Returns:** Graph in requested format (string)

**Example:**
```python
export_graph("network", "graphml")
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

## System (1 tool)

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

## Data Science (12 tools)

### analyze_hypothesis_test

Run statistical hypothesis test on query results.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `connection_name` | string | Yes | Database connection name |
| `query` | string | Yes | SQL query returning data |
| `test_type` | string | Yes | Test type: t_test, chi_square, mann_whitney, wilcoxon, kruskal, fisher |
| `column` | string | Yes | Column to test |
| `group_column` | string | No | Column defining groups for comparison |
| `alpha` | number | No | Significance level (default: 0.05) |
| `alternative` | string | No | Two-sided, less, or greater (default: two-sided) |

**Returns:** Test statistics, p-value, and interpretation (JSON)

**Example:**
```python
analyze_hypothesis_test("mydb", "SELECT * FROM experiments", "t_test", "score", "treatment", alpha=0.05)
```

**Composition hints:** Use for A/B testing and experiment validation.

---

### analyze_anova

Analyze variance across multiple groups using ANOVA.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `connection_name` | string | Yes | Database connection name |
| `query` | string | Yes | SQL query with group and value columns |
| `value_column` | string | Yes | Column with numeric values |
| `group_column` | string | Yes | Column defining groups |

**Returns:** F-statistic, p-value, group means, effect size (JSON)

**Example:**
```python
analyze_anova("mydb", "SELECT * FROM sales", "revenue", "region")
```

**Composition hints:** Use for comparing means across multiple groups.

---

### analyze_effect_sizes

Calculate effect sizes for statistical tests.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `connection_name` | string | Yes | Database connection name |
| `query` | string | Yes | SQL query returning data |
| `effect_type` | string | Yes | Type: cohens_d, cramers_v, eta_squared |
| `column1` | string | Yes | First column/group |
| `column2` | string | No | Second column for comparison |

**Returns:** Effect size value and interpretation (JSON)

**Example:**
```python
analyze_effect_sizes("mydb", "SELECT * FROM experiments", "cohens_d", "control", "treatment")
```

**Composition hints:** Pair with hypothesis tests for practical significance.

---

### analyze_regression

Fit regression models to data.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `connection_name` | string | Yes | Database connection name |
| `query` | string | Yes | SQL query with feature and target columns |
| `target_column` | string | Yes | Column to predict |
| `feature_columns` | string | Yes | Comma-separated feature columns |
| `model_type` | string | No | linear, ridge, lasso, polynomial (default: linear) |

**Returns:** Model coefficients, R², predictions (JSON)

**Example:**
```python
analyze_regression("mydb", "SELECT * FROM properties", "price", "sqft,bedrooms,year_built", model_type="linear")
```

**Composition hints:** Use with `evaluate_model_performance` for validation.

---

### evaluate_model_performance

Evaluate trained model performance on test data.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `connection_name` | string | Yes | Database connection name |
| `query` | string | Yes | SQL query with actual and predicted columns |
| `actual_column` | string | Yes | Column with true values |
| `predicted_column` | string | Yes | Column with predictions |
| `metric_type` | string | No | r2, rmse, mae, accuracy (default: r2) |

**Returns:** Performance metrics and diagnostics (JSON)

**Example:**
```python
evaluate_model_performance("mydb", "SELECT * FROM test_results", "actual_price", "predicted_price", metric_type="rmse")
```

**Composition hints:** Use after regression or classification.

---

### analyze_clusters

Perform clustering analysis on query data.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `connection_name` | string | Yes | Database connection name |
| `query` | string | Yes | SQL query with feature columns |
| `feature_columns` | string | Yes | Comma-separated numeric columns |
| `n_clusters` | integer | No | Number of clusters (default: auto) |
| `algorithm` | string | No | kmeans, hierarchical, dbscan (default: kmeans) |

**Returns:** Cluster assignments, centroids, silhouette score (JSON)

**Example:**
```python
analyze_clusters("mydb", "SELECT * FROM customers", "spending,frequency,recency", n_clusters=5, algorithm="kmeans")
```

**Composition hints:** Use for customer segmentation and pattern discovery.

---

### detect_anomalies

Detect anomalies in query data using statistical methods.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `connection_name` | string | Yes | Database connection name |
| `query` | string | Yes | SQL query with numeric columns |
| `column` | string | Yes | Column to analyze for anomalies |
| `threshold` | number | No | Standard deviation threshold (default: 3.0) |
| `method` | string | No | zscore, iqr, isolation_forest (default: zscore) |

**Returns:** Anomaly flags, scores, flagged rows (JSON)

**Example:**
```python
detect_anomalies("mydb", "SELECT * FROM transactions", "amount", threshold=2.5, method="iqr")
```

**Composition hints:** Use for data quality and fraud detection.

---

### reduce_dimensions

Perform dimensionality reduction on high-dimensional data.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `connection_name` | string | Yes | Database connection name |
| `query` | string | Yes | SQL query with feature columns |
| `feature_columns` | string | Yes | Comma-separated numeric columns |
| `n_components` | integer | No | Target number of dimensions (default: 2) |
| `method` | string | No | pca, tsne, umap (default: pca) |

**Returns:** Reduced components, explained variance (JSON)

**Example:**
```python
reduce_dimensions("mydb", "SELECT * FROM gene_data", "gene_*", n_components=3, method="pca")
```

**Composition hints:** Use before clustering on high-dimensional data.

---

### analyze_time_series

Analyze time series data for trends and patterns.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `connection_name` | string | Yes | Database connection name |
| `query` | string | Yes | SQL query with timestamp and value columns |
| `time_column` | string | Yes | Timestamp column name |
| `value_column` | string | Yes | Numeric value column |
| `period` | string | No | Decomposition period: daily, weekly, monthly |

**Returns:** Trend, seasonal, residual components; stationarity test (JSON)

**Example:**
```python
analyze_time_series("mydb", "SELECT * FROM stock_prices", "date", "price", period="daily")
```

**Composition hints:** Use before `forecast_time_series`.

---

### forecast_time_series

Generate time series forecasts.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `connection_name` | string | Yes | Database connection name |
| `query` | string | Yes | SQL query with historical data |
| `time_column` | string | Yes | Timestamp column name |
| `value_column` | string | Yes | Numeric value column |
| `periods_ahead` | integer | No | Number of periods to forecast (default: 10) |
| `method` | string | No | arima, exponential_smoothing, prophet (default: arima) |

**Returns:** Forecast values, confidence intervals (JSON)

**Example:**
```python
forecast_time_series("mydb", "SELECT * FROM sales_history", "date", "sales", periods_ahead=30, method="arima")
```

**Composition hints:** Use for sales, demand, and resource planning.

---

### analyze_rfm

Perform RFM (Recency, Frequency, Monetary) customer analysis.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `connection_name` | string | Yes | Database connection name |
| `query` | string | Yes | SQL query with customer transactions |
| `customer_id` | string | Yes | Column with customer identifiers |
| `date_column` | string | Yes | Transaction date column |
| `amount_column` | string | Yes | Transaction amount column |

**Returns:** RFM scores, customer segments, value tiers (JSON)

**Example:**
```python
analyze_rfm("mydb", "SELECT * FROM transactions", "customer_id", "order_date", "order_value")
```

**Composition hints:** Use for customer segmentation and targeting.

---

### analyze_ab_test

Analyze results from A/B tests with statistical rigor.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `connection_name` | string | Yes | Database connection name |
| `query` | string | Yes | SQL query with control/test and outcome |
| `group_column` | string | Yes | Column denoting control/treatment groups |
| `outcome_column` | string | Yes | Column with binary or continuous outcome |
| `confidence_level` | number | No | Confidence level (default: 0.95) |

**Returns:** Test statistics, p-value, sample size, power analysis (JSON)

**Example:**
```python
analyze_ab_test("mydb", "SELECT * FROM experiment_results", "group", "converted", confidence_level=0.95)
```

**Composition hints:** Use for experimentation and decision-making.

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
| System | 1 | Compatibility and migration |
| Data Science | 12 | Statistical analysis and forecasting |
| **Total** | **52** | Complete LLM-native data platform |

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
