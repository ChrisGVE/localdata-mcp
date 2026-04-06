<p align="center">
  <img src="assets/logo.png" alt="LocalData MCP Server" width="250">
</p>

# LocalData MCP Server

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub Release](https://img.shields.io/github/v/release/ChrisGVE/localdata-mcp)](https://github.com/ChrisGVE/localdata-mcp/releases)
[![CI](https://img.shields.io/github/actions/workflow/status/ChrisGVE/localdata-mcp/ci.yml?branch=main&label=CI)](https://github.com/ChrisGVE/localdata-mcp/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/localdata-mcp.svg)](https://pypi.org/project/localdata-mcp/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Documentation](https://img.shields.io/badge/docs-readthedocs-blue.svg)](https://localdata-mcp.readthedocs.io)
[![FastMCP](https://img.shields.io/badge/FastMCP-Compatible-green.svg)](https://github.com/jlowin/fastmcp)
[![Verified on MseeP](https://mseep.ai/badge.svg)](https://mseep.ai/app/cd737717-02f3-4388-bab7-5ec7cbe40713)
![PyPI downloads](https://img.shields.io/pypi/dm/localdata-mcp)
![GitHub stars](https://img.shields.io/github/stars/ChrisGVE/localdata-mcp?style=social)

LocalData MCP gives LLM agents access to local and remote data — databases, files, graphs, and structured documents — along with a full data science toolkit for analysis and modeling. It exposes 52 MCP tools across 13 database types and 20+ file formats, with memory-bounded streaming so agents can work safely on large datasets without exceeding available RAM.

[![MseeP.ai Security Assessment Badge](https://mseep.net/pr/chrisgve-localdata-mcp-badge.png)](https://mseep.ai/app/chrisgve-localdata-mcp)

## Quick Start

```bash
# Install permanently
uv tool install localdata-mcp

# Or run directly without installing
uvx localdata-mcp
```

> **First-run note:** Data science dependencies (scipy, scikit-learn, statsmodels, geopandas) total around 200 MB and are downloaded on first use. Subsequent starts reuse the cache. If your MCP client times out on the first launch, reconnect — the next start will be immediate.

Add to your MCP client configuration:

```json
{
  "mcpServers": {
    "localdata": {
      "command": "localdata-mcp"
    }
  }
}
```

For `uvx` (no permanent install):

```json
{
  "mcpServers": {
    "localdata": {
      "command": "uvx",
      "args": ["localdata-mcp"]
    }
  }
}
```

Then connect to any supported source and start querying:

```python
connect_database("sales", "postgresql", "postgresql://user:pass@localhost/db")
execute_query("sales", "SELECT product, SUM(amount) FROM orders GROUP BY product")

connect_database("data", "csv", "./records.csv")
analyze_hypothesis_test("data", "SELECT amount, region FROM data", column="amount", group_column="region")
```

## Feature Overview

### Core Database (8 tools)

Connect, query, and inspect databases and files. All queries execute within configurable memory limits (default 2 GB) with automatic chunked streaming for large result sets.

| Tool | Description |
| --- | --- |
| `connect_database` | Open a connection to any supported database or file |
| `disconnect_database` | Close a connection |
| `list_databases` | List active connections |
| `execute_query` | Run SQL with streaming, chunking, and preflight mode |
| `describe_database` | Show schema and table list |
| `describe_table` | Column types, indexes, row count |
| `find_table` | Locate a table across all active connections |
| `analyze_query_preview` | Estimate query cost before execution |

### Streaming and Memory (9 tools)

| Tool | Description |
| --- | --- |
| `next_chunk` | Retrieve the next chunk of a streamed result |
| `request_data_chunk` | Fetch a specific chunk by row range |
| `request_multiple_chunks` | Batch-fetch multiple chunks in one call |
| `manage_memory_bounds` | View and configure memory limits |
| `get_streaming_status` | Check active streams and buffer usage |
| `clear_streaming_buffer` | Free memory from a specific buffer |
| `get_query_metadata` | Rich metadata for a completed query |
| `cancel_query_operation` | Cancel a running or buffered query |
| `get_data_quality_report` | Column statistics, null rates, and quality metrics |

### Tree / Structured Data (10 tools)

Navigate and edit TOML, JSON, and YAML files as navigable trees. Supports full CRUD with auto-creation of ancestor nodes and round-trip export to any supported format.

| Tool | Description |
| --- | --- |
| `get_node` / `get_children` | Navigate the tree |
| `set_node` / `delete_node` | Create or remove nodes |
| `get_value` / `set_value` / `delete_key` | Read and write properties |
| `list_keys` | List key-value pairs at a node |
| `move_node` | Relocate a node within the tree |
| `export_structured` | Export as TOML, JSON, or YAML |

### Graph (14 tools)

Work with DOT, GML, GraphML, and Mermaid files as directed multigraphs. Supports full CRUD on nodes and edges, shortest-path and all-paths queries, structural statistics, and multi-format export.

| Tool | Description |
| --- | --- |
| `get_node_graph` / `get_neighbors` / `get_edges` | Navigate the graph |
| `set_node_graph` / `delete_node_graph` | Create or remove nodes |
| `add_edge` / `remove_edge` | Manage edges |
| `get_value_graph` / `set_value_graph` / `delete_key_graph` / `list_keys_graph` | Node properties |
| `find_path` | Shortest or all paths between two nodes |
| `get_graph_stats` | Node/edge counts, density, DAG validation |
| `export_graph` | Export as DOT, GML, GraphML, or Mermaid |

### Search and Transform (2 tools)

| Tool | Description |
| --- | --- |
| `search_data` | Regex search across query results |
| `transform_data` | Apply column transformations to result sets |

### Schema and Audit (3 tools)

| Tool | Description |
| --- | --- |
| `export_schema` | Export full schema as JSON |
| `get_query_log` | Recent query execution history |
| `get_error_log` | Recent error log |

### System (2 tools)

| Tool | Description |
| --- | --- |
| `check_compatibility` | Verify API backward compatibility |
| `get_metrics` | Server performance and resource metrics |

### Data Science (12 tools)

Run statistical analysis, modeling, and pattern detection directly on query results from any connected source.

| Tool | Domain |
| --- | --- |
| `analyze_hypothesis_test` | Statistical Analysis |
| `analyze_anova` | Statistical Analysis |
| `analyze_effect_sizes` | Statistical Analysis |
| `analyze_regression` | Regression and Modeling |
| `evaluate_model_performance` | Regression and Modeling |
| `analyze_clusters` | Pattern Recognition |
| `detect_anomalies` | Pattern Recognition |
| `reduce_dimensions` | Pattern Recognition |
| `analyze_time_series` | Time Series |
| `forecast_time_series` | Time Series |
| `analyze_rfm` | Business Intelligence |
| `analyze_ab_test` | Business Intelligence |

## Supported Data Sources

### Databases

| Type | Engines |
| --- | --- |
| SQL | SQLite, PostgreSQL, MySQL, DuckDB |
| SQL (enterprise) | Oracle, MS SQL Server (`pip install localdata-mcp[enterprise]`) |
| Document | MongoDB, CouchDB (`pip install localdata-mcp[modern-databases]`) |
| Key-value | Redis (`pip install localdata-mcp[modern-databases]`) |
| Search | Elasticsearch (`pip install localdata-mcp[modern-databases]`) |
| Time series | InfluxDB (`pip install localdata-mcp[modern-databases]`) |
| Graph | Neo4j (`pip install localdata-mcp[modern-databases]`) |
| RDF / SPARQL | Turtle (.ttl), N-Triples (.nt), remote SPARQL endpoints |

### File Formats

| Category | Formats |
| --- | --- |
| Tabular | CSV, TSV |
| Structured | JSON, JSONL, YAML, TOML, XML, INI |
| Spreadsheet | Excel (.xlsx, .xls), LibreOffice Calc (.ods), Apple Numbers (.numbers) |
| Analytical | Parquet, Feather, Arrow, HDF5 |
| Graph | DOT (Graphviz), GML, GraphML, Mermaid |
| RDF | Turtle (.ttl), N-Triples (.nt) |

Multi-sheet spreadsheets are fully supported: each sheet becomes a separately queryable table. Connect to a specific sheet with `?sheet=SheetName` in the path.

## Data Science Domains

**Statistical Analysis** — t-tests, chi-squared, Mann-Whitney, Kruskal-Wallis, and related hypothesis tests; one-way ANOVA with post-hoc tests; Cohen's d, eta-squared, and other effect size measures.

**Regression and Modeling** — linear, polynomial, logistic, ridge, lasso, and elastic net regression; model evaluation with R², RMSE, MAE, and classification metrics; automated feature selection.

**Pattern Recognition** — K-means, DBSCAN, and hierarchical clustering; anomaly detection via isolation forest, LOF, and one-class SVM; dimensionality reduction with PCA, t-SNE, and UMAP.

**Time Series** — decomposition, stationarity testing, autocorrelation analysis; ARIMA, SARIMA, and ETS forecasting; change point detection; multivariate analysis with VAR, Granger causality, and cointegration tests.

**Business Intelligence** — A/B test statistical analysis; RFM customer segmentation; cohort analysis, CLV modeling, and funnel analysis.

**Geospatial** — distance and coordinate calculations, spatial joins, interpolation, and network analysis.

**Optimization** — linear programming, constrained optimization, assignment problems, and network optimization.

**Sampling and Estimation** — bootstrap confidence intervals, Bayesian estimation, Monte Carlo simulation, and stratified sampling.

## Architecture

- **Intention-driven interface** — tools accept semantic parameters ("find strong correlations") rather than requiring statistical procedure names or threshold values
- **Progressive disclosure** — simple calls return high-level insights with sensible defaults; advanced parameters are available when needed
- **Streaming-first execution** — all operations are designed for chunked processing; tools automatically switch strategies based on data size, keeping memory usage within configured bounds
- **Composition metadata** — every tool result includes metadata that downstream tools can use directly, enabling chained analysis without manual wiring

## Configuration

LocalData MCP uses environment variables for optional settings. The defaults work for most cases.

| Variable | Default | Description |
| --- | --- | --- |
| `LOCALDATA_MEMORY_LIMIT_MB` | `2048` | Maximum memory per query result (MB) |
| `LOCALDATA_MAX_CONNECTIONS` | `10` | Maximum concurrent database connections |
| `LOCALDATA_CHUNK_SIZE` | `500` | Default rows per streaming chunk |
| `LOCALDATA_BUFFER_TTL` | `600` | Streaming buffer expiry in seconds |
| `LOCALDATA_WORKING_DIR` | process cwd | Root directory for file access (file paths are restricted to this tree) |

Set in your MCP server configuration under `"env"`, or in a `.env` file in the working directory.

## Documentation

- [Database Connection Guide](DATABASE_CONNECTIONS.md) — connection strings, driver setup, and security practices
- [Docker Usage Guide](DOCKER_USAGE.md) — container deployment and configuration
- [Advanced Examples](ADVANCED_EXAMPLES.md) — production-ready usage patterns
- [Troubleshooting Guide](TROUBLESHOOTING.md) — common issues and solutions
- [FAQ](FAQ.md) — frequently asked questions
- [API Reference](https://localdata-mcp.readthedocs.io) — full tool and parameter reference

## Development

```bash
git clone https://github.com/ChrisGVE/localdata-mcp.git
cd localdata-mcp
uv sync --all-extras
uv run pytest
```

The test suite includes 1,600+ unit tests, 234+ integration tests, and 62 enterprise-scale tests across 7 database types with 100K rows each.

## Contributing

Contributions are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) before submitting a pull request.

## License

MIT — see [LICENSE](LICENSE) for details.
