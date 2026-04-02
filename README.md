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

**A comprehensive MCP server for databases, spreadsheets, structured data files, and directed graphs. Connects LLM agents to 13 database types, 20+ file formats, and RDF/SPARQL endpoints with streaming query execution, memory-bounded operations, and enterprise-scale support.**

[![MseeP.ai Security Assessment Badge](https://mseep.net/pr/chrisgve-localdata-mcp-badge.png)](https://mseep.ai/app/chrisgve-localdata-mcp)

## Table of Contents

- [Features](#features)
- [Quick Start](#quick-start)
- [Available Tools](#available-tools)
- [Working with Graphs](#working-with-graphs)
- [Supported Data Sources](#supported-data-sources)
- [Security Features](#security-features)
- [Performance & Scalability](#performance--scalability)
- [Examples](#examples)
- [Testing & Quality](#testing--quality)
- [Troubleshooting](#troubleshooting)
- [Roadmap](#roadmap)
- [Contributing](#contributing)

## Features

### **Multi-Database Support**

- **SQL Databases**: PostgreSQL, MySQL, SQLite, DuckDB, Oracle, MS SQL Server
- **NoSQL Databases**: MongoDB, Redis, Elasticsearch, InfluxDB, Neo4j, CouchDB
- **RDF/SPARQL**: Turtle, N-Triples files and remote SPARQL endpoints
- **Spreadsheets**: Excel (.xlsx/.xls), LibreOffice Calc (.ods), Apple Numbers (.numbers)
- **Structured Files**: CSV, TSV, JSON, YAML, TOML, XML, INI
- **Graph Files**: DOT (Graphviz), GML, GraphML, Mermaid flowcharts
- **Analytical Formats**: Parquet, Feather, Arrow, HDF5

### **Streaming Query Execution**

- **Memory-Bounded**: All queries execute within configurable memory limits (default 2GB)
- **Automatic Streaming**: Large result sets stream in chunks with pagination
- **Pre-Query Analysis**: Resource estimation before execution to prevent crashes
- **Preflight Mode**: EXPLAIN-based cost analysis without executing the query

### **Security**

- **Path Restriction**: File access confined to the current working directory
- **SQL Validation**: Only SELECT and CTE queries allowed; injection prevention
- **Connection Limits**: Maximum 10 concurrent database connections
- **Query Audit**: Full query and error logging for compliance

### **40 MCP Tools**

- 8 core database tools, 9 tree tools, 14 graph tools
- 9 streaming and memory management tools
- Schema export, data quality reports, regex search, and data transformation

## Quick Start

### Installation

```bash
# Using pip
pip install localdata-mcp

# Using uv (recommended)
uv tool install localdata-mcp

# Development installation
git clone https://github.com/ChrisGVE/localdata-mcp.git
cd localdata-mcp
pip install -e .
```

### Configuration

Add to your MCP client configuration:

```json
{
  "mcpServers": {
    "localdata": {
      "command": "localdata-mcp",
      "env": {}
    }
  }
}
```

**Docker Usage**: See [Docker Usage Guide](DOCKER_USAGE.md) for container deployment and configuration.

### Usage Examples

#### Connect to Databases

```python
# SQL databases
connect_database("analytics", "postgresql", "postgresql://user:pass@localhost/db")
connect_database("local", "sqlite", "./data.sqlite")
connect_database("erp", "oracle", "oracle+oracledb://user:pass@host:1521/?service_name=ORCL")
connect_database("warehouse", "mssql", "mssql+pymssql://sa:pass@host:1433/master")

# NoSQL databases
connect_database("docs", "mongodb", "mongodb://localhost:27017/mydb")
connect_database("cache", "redis", "redis://localhost:6379/0")
connect_database("search", "elasticsearch", "http://localhost:9200")

# Files — CSV, Excel, Parquet, JSON, YAML, TOML, XML, INI, and more
connect_database("csvdata", "csv", "./data.csv")
connect_database("sales", "xlsx", "./sales_data.xlsx")
connect_database("analytics", "parquet", "./data.parquet")
connect_database("config", "toml", "./config.toml")
```

#### Work with Structured Files (TOML, JSON, YAML)

```python
# Connect — returns tree summary (root nodes, depths, property counts)
connect_database("cfg", "toml", "./config.toml")

# Navigate the tree
get_node("cfg")                          # Root summary
get_node("cfg", "server")               # Node properties + children info
get_children("cfg", "server")           # List child nodes

# Read and write properties
get_value("cfg", "server", "port")       # → 8080
set_value("cfg", "server", "port", "9090")
set_value("cfg", "monitoring.alerts", "enabled", "true")  # auto-creates node

# Export back to any format
export_structured("cfg", "toml")         # Modified TOML output
export_structured("cfg", "json")         # Convert to JSON
```

#### Query Data

```python
# Execute queries — large results automatically stream in chunks
execute_query("analytics", "SELECT * FROM users LIMIT 50")

# Pre-flight analysis without executing
execute_query("analytics", "SELECT * FROM large_table", preflight=True)

# Retrieve subsequent chunks using query_id from the response metadata
next_chunk(query_id="analytics_1712345678_a1b2", start_row=101, chunk_size="500")

# Check streaming status and memory usage
get_streaming_status()
manage_memory_bounds()
```

## Available Tools

### Core Database Tools

| Tool                 | Description                                    |
| -------------------- | ---------------------------------------------- |
| `connect_database`   | Connect to a database, file, or SPARQL endpoint |
| `disconnect_database`| Close a connection                             |
| `list_databases`     | List active connections                        |
| `execute_query`      | Run SQL with streaming, chunking, and preflight |
| `next_chunk`         | Retrieve the next chunk of a streamed result   |
| `describe_database`  | Show database schema and table list            |
| `describe_table`     | Show column types, indexes, row count          |
| `find_table`         | Locate a table across all connections          |
| `export_schema`      | Export full schema as JSON                     |

### Streaming and Memory Tools

| Tool                       | Description                                     |
| -------------------------- | ----------------------------------------------- |
| `analyze_query_preview`    | Pre-query cost and resource estimation          |
| `manage_memory_bounds`     | View and configure memory limits                |
| `get_streaming_status`     | Check active streams and buffer usage           |
| `clear_streaming_buffer`   | Free memory by clearing a specific buffer       |
| `get_query_metadata`       | Rich metadata for a completed query             |
| `request_data_chunk`       | Request a specific chunk by row range           |
| `request_multiple_chunks`  | Batch-request multiple chunks at once           |
| `cancel_query_operation`   | Cancel a running or buffered query              |
| `get_data_quality_report`  | Column stats, nulls, and data quality metrics   |
| `check_compatibility`      | Check API backward compatibility                |

### Query Audit Tools

| Tool                 | Description                                    |
| -------------------- | ---------------------------------------------- |
| `get_query_log`      | View recent query execution history            |
| `get_error_log`      | View recent error log                          |
| `search_data`        | Regex search across query results              |
| `transform_data`     | Apply transformations to query results         |

### Tree Tools (TOML, JSON, YAML)

These tools are available when connected to a structured data file.

| Tool                 | Description                              | Use Case         |
| -------------------- | ---------------------------------------- | ---------------- |
| `get_node`           | View node properties and children info   | Navigation       |
| `get_children`       | List child nodes with pagination         | Navigation       |
| `set_node`           | Create a node (auto-creates ancestors)   | Structure edit   |
| `delete_node`        | Remove node and all descendants          | Structure edit   |
| `list_keys`          | List key-value pairs at a node           | Inspection       |
| `get_value`          | Read a single property value             | Inspection       |
| `set_value`          | Create or update a property              | Data edit        |
| `delete_key`         | Remove a property from a node            | Data edit        |
| `export_structured`  | Export tree as TOML, JSON, or YAML       | Export           |

### Graph Tools (DOT, GML, GraphML, Mermaid)

These tools are available when connected to a graph file.

| Tool                 | Description                              | Use Case         |
| -------------------- | ---------------------------------------- | ---------------- |
| `set_node`           | Create or update a graph node            | Structure edit   |
| `delete_node`        | Remove node, cascade edges and properties| Structure edit   |
| `get_node`           | Inspect a node's label and properties    | Navigation       |
| `add_edge`           | Create a directed edge between nodes     | Structure edit   |
| `remove_edge`        | Remove an edge by source/target/label    | Structure edit   |
| `get_edges`          | List edges for a node                    | Navigation       |
| `get_neighbors`      | List adjacent nodes (in/out/both)        | Navigation       |
| `get_value`          | Read a typed property from a node        | Inspection       |
| `set_value`          | Set a typed property on a node           | Data edit        |
| `delete_key`         | Remove a property from a node            | Data edit        |
| `list_keys`          | List properties on a node with pagination| Inspection       |
| `find_path`          | Find shortest or all paths between nodes | Analysis         |
| `get_graph_stats`    | Node/edge counts, density, DAG test      | Analysis         |
| `export_graph`       | Export as DOT, GML, GraphML, or Mermaid  | Export           |

## Working with Graphs

```python
# Connect to a graph file
connect_database("taxonomy", "graphml", "./knowledge_graph.graphml")
connect_database("flow", "dot", "./pipeline.dot")
connect_database("diagram", "mermaid", "./architecture.mmd")

# Navigate the graph
get_node("taxonomy", "machine_learning")
get_neighbors("taxonomy", "machine_learning", direction="out")
get_edges("taxonomy", node_id="machine_learning")

# Edit nodes and edges
set_node("taxonomy", "deep_learning", label="Deep Learning")
add_edge("taxonomy", "machine_learning", "deep_learning", label="broader")
set_value("taxonomy", "deep_learning", "field", "Computer Science")

# Algorithmic analysis
find_path("taxonomy", "statistics", "neural_networks")
get_graph_stats("taxonomy")

# Export to any supported format (includes metadata)
export_graph("taxonomy", "graphml")
export_graph("taxonomy", "mermaid")
```

Mutation and import operations return validation warnings when issues are detected:

```python
# Adding a self-loop returns a warning
add_edge("taxonomy", "A", "A", label="related")
# → {"source": "A", "target": "A", ..., "warnings": [{"code": "self_loop", ...}]}

# Importing a file with issues reports them all
connect_database("g", "dot", "./messy_graph.dot")
# → {"node_count": 42, ..., "warnings": [{"code": "orphan_nodes", ...}, ...]}
```

## Supported Data Sources

**Detailed Connection Guide**: See [Database Connections Guide](DATABASE_CONNECTIONS.md) for setup instructions, connection strings, and security practices.

### SQL Databases

- **PostgreSQL**: Full support with connection pooling
- **MySQL**: Complete MySQL/MariaDB compatibility
- **SQLite**: Local file and in-memory databases
- **DuckDB**: High-performance analytical SQL database
- **Oracle**: Enterprise support via oracledb thin mode (`pip install localdata-mcp[enterprise]`)
- **MS SQL Server**: Enterprise support via pymssql (`pip install localdata-mcp[enterprise]`)

### NoSQL Databases

- **MongoDB**: Document store with collection queries and aggregation
- **Redis**: High-performance key-value store
- **Elasticsearch**: Full-text search and analytics engine
- **InfluxDB**: Time-series database for metrics and IoT data
- **Neo4j**: Graph database for relationship queries
- **CouchDB**: Document-oriented database with HTTP API

### RDF and SPARQL

- **Turtle (.ttl)**: RDF file format with SPARQL query support
- **N-Triples (.nt)**: Line-based RDF format
- **SPARQL Endpoints**: Connect to remote RDF stores (Wikidata, DBpedia, etc.)

### Structured Files

#### Spreadsheet Formats

- **Excel (.xlsx, .xls)**: Full multi-sheet support with automatic table creation
- **LibreOffice Calc (.ods)**: Complete ODS support with sheet handling
- **Apple Numbers (.numbers)**: Native support for Numbers documents
- **Multi-sheet handling**: Each sheet becomes a separate queryable table

#### Text-Based Formats

- **CSV**: Large file automatic SQLite conversion
- **TSV**: Tab-separated values with same features as CSV
- **JSON**: Tree storage with node navigation and CRUD (v1.4.0)
- **YAML**: Tree storage with multi-document support (v1.4.0)
- **TOML**: Tree storage with array-of-tables support (v1.4.0)
- **XML**: Structured XML document parsing
- **INI**: Configuration file format support

#### Graph Formats (v1.5.0)

- **DOT** (Graphviz): Directed and undirected graph descriptions
- **GML**: Graph Modelling Language with nested attributes
- **GraphML**: XML-based format with typed node/edge properties
- **Mermaid**: Flowchart syntax with subgraphs, edge labels, and node shapes

#### Analytical Formats

- **Parquet**: High-performance columnar data format
- **Feather**: Fast binary format for data interchange
- **Arrow**: In-memory columnar format support
- **HDF5**: Hierarchical data format for scientific computing

## Security Features

### Path Security

```python
# ✅ Allowed - current directory and subdirectories
"./data/users.csv"
"data/config.json"
"subdir/file.yaml"

# ❌ Blocked - parent directory access
"../etc/passwd"
"../../sensitive.db"
"/etc/hosts"
```

### SQL Injection Prevention

```python
# ✅ Safe - parameterized queries
describe_table("mydb", "users")  # Validates table name

# ❌ Blocked - malicious input
describe_table("mydb", "users; DROP TABLE users; --")
```

### Resource Limits

- **Connection Limit**: Maximum 10 concurrent connections
- **File Size Threshold**: 100MB triggers temporary storage
- **Query Buffering**: Automatic for 100+ row results
- **Auto-Cleanup**: Buffers expire after 10 minutes

## Performance & Scalability

### Large File Handling

- Files over 100MB automatically use temporary SQLite storage
- Memory-efficient streaming for large datasets
- Automatic cleanup of temporary files

### Query Optimization

- Results with 100+ rows automatically use buffering system
- Chunk-based retrieval for large datasets
- File modification detection for cache invalidation

### Concurrency

- Thread-safe connection management
- Concurrent query execution support
- Resource pooling and limits

## Testing & Quality

- **1,600+ unit tests** covering all tools, security, and edge cases
- **234+ integration tests** for SQL, NoSQL, file format, and graph connections
- **62 enterprise-scale tests** across 7 database types with 100K rows each (PostgreSQL, MySQL, MSSQL, Oracle, SQLite, MongoDB, Elasticsearch)
- Security testing: path traversal, SQL injection, resource exhaustion
- Streaming and memory-bounds validation under load

## Examples

**Production Examples**: See [Advanced Examples](ADVANCED_EXAMPLES.md) for production-ready usage patterns and complex scenarios.

### Basic Database Operations

```python
# Connect to SQLite
connect_database("sales", "sqlite", "./sales.db")

# Explore structure
describe_database("sales")
describe_table("sales", "orders")

# Query data
execute_query("sales", "SELECT product, SUM(amount) FROM orders GROUP BY product")
```

### Large Dataset Processing

```python
# Connect to large CSV (100MB+ files use temporary SQLite storage)
connect_database("bigdata", "csv", "./million_records.csv")

# Check estimated cost before running
execute_query("bigdata", "SELECT * FROM data WHERE category = 'A'", preflight=True)

# Execute — large results stream automatically
result = execute_query("bigdata", "SELECT * FROM data WHERE category = 'A'")
# result.metadata contains query_id, total_rows, streaming status

# Retrieve additional chunks
next_chunk(query_id="bigdata_1712345678_a1b2", start_row=201, chunk_size="500")
```

### Multi-Database Analysis

```python
# Connect multiple sources
connect_database("postgres", "postgresql", "postgresql://localhost/prod")
connect_database("config", "yaml", "./config.yaml")
connect_database("logs", "json", "./logs.json")

# Query SQL databases
execute_query("postgres", "SELECT * FROM users")

# Navigate structured files as trees
get_node("config", "database.settings")
get_value("config", "database.settings", "host")
```

### Multi-Sheet Spreadsheet Handling

LocalData MCP Server provides comprehensive support for multi-sheet spreadsheets (Excel and LibreOffice Calc):

#### Automatic Multi-Sheet Processing

```python
# Connect to Excel file - all sheets become separate tables
connect_database("workbook", "xlsx", "./financial_data.xlsx")

# Query specific sheet (table names are sanitized sheet names)
execute_query("workbook", "SELECT * FROM Q1_Sales")
execute_query("workbook", "SELECT * FROM Q2_Budget")
execute_query("workbook", "SELECT * FROM Annual_Summary")
```

#### Single Sheet Selection

```python
# Connect to specific sheet only using ?sheet=SheetName syntax
connect_database("q1only", "xlsx", "./financial_data.xlsx?sheet=Q1 Sales")

# The data is available as the default table
execute_query("q1only", "SELECT * FROM data")
```

#### Sheet Name Sanitization

Sheet names are automatically sanitized for SQL compatibility:

| Original Sheet Name | SQL Table Name   |
| ------------------- | ---------------- |
| "Q1 Sales"          | Q1_Sales         |
| "2024-Budget"       | \_2024_Budget    |
| "Summary & Notes"   | Summary\_\_Notes |

#### Discovering Available Sheets

```python
# Connect to multi-sheet workbook
connect_database("workbook", "xlsx", "./data.xlsx")

# List all available tables (sheets)
describe_database("workbook")

# Query a specific sheet
execute_query("workbook", "SELECT * FROM Sheet1 LIMIT 10")
```


## Troubleshooting

For comprehensive troubleshooting guidance, see [Troubleshooting Guide](TROUBLESHOOTING.md). For common questions, check the [FAQ](FAQ.md).

## Release History

### v1.7.0 (Current)

- **Enterprise Database Support**: Oracle (oracledb thin mode) and MS SQL Server (pymssql)
- **Enterprise-Scale Testing**: 62 tests across 7 databases with 100K rows each
- **RDF/SPARQL Support**: Turtle, N-Triples files and remote SPARQL endpoints
- **Pydantic v2 Migration**: Configuration system updated to Pydantic v2

### v1.6.0

- **Streaming Query Executor**: Memory-bounded query execution with automatic chunking
- **Pre-Query Analysis**: Cost estimation via `analyze_query_preview` and `preflight` mode
- **Query Audit Logging**: Full query and error logs with `get_query_log` / `get_error_log`
- **Data Quality Reports**: Column statistics and null analysis via `get_data_quality_report`
- **Schema Export**: Full schema export as JSON via `export_schema`
- **Regex Search and Transform**: `search_data` and `transform_data` tools
- **Backward Compatibility Manager**: API version checking via `check_compatibility`

### v1.5.0

- **Graph Storage**: DOT, GML, GraphML, and Mermaid files as directed multigraphs
- **14 Graph Tools**: Node/edge CRUD, path finding, statistics, multi-format export
- **Graph Validation**: 13 automated checks for structural, semantic, and DAG issues

### v1.4.0

- **Tree Storage**: TOML, JSON, YAML as navigable trees with full CRUD
- **9 Tree Tools**: Navigation, property management, format conversion
- **FastMCP v3**: Updated tool registration

### v1.3.0

- **Structured Logging**: JSON logging with Prometheus metrics
- **Timeout System**: Configurable query timeouts with monitoring

### v1.1.0

- **Spreadsheet Formats**: Excel, LibreOffice Calc, Apple Numbers
- **Analytical Formats**: Parquet, Feather, Arrow, HDF5

## Contributing

Contributions welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

```bash
git clone https://github.com/ChrisGVE/localdata-mcp.git
cd localdata-mcp
uv sync --all-extras
uv run pytest
```

## License

MIT License - see the [LICENSE](LICENSE) file for details.

## Links

- **GitHub**: [localdata-mcp](https://github.com/ChrisGVE/localdata-mcp)
- **PyPI**: [localdata-mcp](https://pypi.org/project/localdata-mcp/)
- **MCP Protocol**: [Model Context Protocol](https://modelcontextprotocol.io/)
- **FastMCP**: [FastMCP Framework](https://github.com/jlowin/fastmcp)



## Tags

`mcp` `model-context-protocol` `database` `postgresql` `mysql` `sqlite` `oracle` `mssql` `mongodb` `redis` `elasticsearch` `sparql` `rdf` `spreadsheet` `excel` `csv` `json` `yaml` `toml` `parquet` `graph` `graphml` `dot` `mermaid` `streaming` `data-science` `python`
