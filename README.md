<p align="center"> 
  <img src="assets/logo.png" alt="LocalData MCP Server" width="250">
</p>

# LocalData MCP Server

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://img.shields.io/pypi/v/localdata-mcp.svg)](https://pypi.org/project/localdata-mcp/)
[![FastMCP](https://img.shields.io/badge/FastMCP-Compatible-green.svg)](https://github.com/jlowin/fastmcp)
[![Verified on MseeP](https://mseep.ai/badge.svg)](https://mseep.ai/app/cd737717-02f3-4388-bab7-5ec7cbe40713)
![GitHub stars](https://img.shields.io/github/stars/ChrisGVE/localdata-mcp?style=social)
![GitHub forks](https://img.shields.io/github/forks/ChrisGVE/localdata-mcp?style=social)
![PyPI downloads](https://img.shields.io/pypi/dm/localdata-mcp)

**A comprehensive MCP server for databases, spreadsheets, and structured data files with security features, performance optimization, and extensive format support.**

[![MseeP.ai Security Assessment Badge](https://mseep.net/pr/chrisgve-localdata-mcp-badge.png)](https://mseep.ai/app/chrisgve-localdata-mcp)

## What's New in v1.4.0

### Tree Storage for Structured Data

TOML, JSON, and YAML files are now stored as trees instead of flat tables. Each node in the hierarchy is individually addressable, with key-value properties attached to it. This means an LLM can navigate, query, edit, and export configuration files and other hierarchical data without loading megabytes of flattened content.

### 9 New Tree Tools

| Tool | Purpose |
|------|---------|
| `get_node` | View a node's properties and whether it has children |
| `get_children` | List child nodes with pagination |
| `set_node` | Create a node (auto-creates missing ancestors) |
| `delete_node` | Remove a node and all descendants |
| `list_keys` | List key-value pairs at a node |
| `get_value` | Read a single property |
| `set_value` | Create or update a property (type-inferred) |
| `delete_key` | Remove a property |
| `export_structured` | Reconstruct and export as TOML, JSON, or YAML |

### Connection Summary on Connect

When connecting to any data source, the response now includes a schema summary (table names, column types, row counts, sample rows for flat data; tree structure overview for hierarchical data). This gives the LLM enough context to write targeted queries from the start.

### FastMCP v3 Compatibility

Updated to work with fastmcp v3.x API for tool registration.

## Table of Contents

- [Features](#features)
- [Quick Start](#quick-start)
- [Available Tools](#available-tools)
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

- **SQL Databases**: PostgreSQL, MySQL, SQLite, DuckDB
- **Modern Databases**: MongoDB, Redis, Elasticsearch, InfluxDB, Neo4j, CouchDB
- **Spreadsheets**: Excel (.xlsx/.xls), LibreOffice Calc (.ods), Apple Numbers (.numbers)
- **Structured Files**: CSV, TSV, JSON, YAML, TOML, XML, INI
- **Analytical Formats**: Parquet, Feather, Arrow, HDF5

### **Advanced Security**

- **Path Security**: Restricts file access to current working directory only
- **SQL Injection Prevention**: Parameterized queries and safe table identifiers
- **Connection Limits**: Maximum 10 concurrent database connections
- **Input Validation**: Comprehensive validation and sanitization

### **Large Dataset Handling**

- **Query Buffering**: Automatic buffering for results with 100+ rows
- **Large File Support**: 100MB+ files automatically use temporary SQLite storage
- **Chunk Retrieval**: Paginated access to large result sets
- **Auto-Cleanup**: 10-minute expiry with file modification detection

### **Developer Experience**

- **Clean Tool Surface**: Core database tools plus 9 tree tools for structured data
- **Error Handling**: Detailed, actionable error messages
- **Thread Safety**: Concurrent operation support
- **Backward Compatible**: All existing APIs preserved

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
# PostgreSQL
connect_database("analytics", "postgresql", "postgresql://user:pass@localhost/db")

# SQLite
connect_database("local", "sqlite", "./data.sqlite")

# CSV Files
connect_database("csvdata", "csv", "./data.csv")

# JSON Files
connect_database("config", "json", "./config.json")

# Excel Spreadsheets (all sheets)
connect_database("sales", "xlsx", "./sales_data.xlsx")

# Excel with specific sheet
connect_database("q1data", "xlsx", "./quarterly.xlsx?sheet=Q1_Sales")

# LibreOffice Calc
connect_database("budget", "ods", "./budget_2024.ods")

# Tab-separated values
connect_database("exports", "tsv", "./export_data.tsv")

# XML structured data
connect_database("config_xml", "xml", "./config.xml")

# INI configuration files
connect_database("settings", "ini", "./app.ini")

# Analytical formats
connect_database("analytics", "parquet", "./data.parquet")
connect_database("features", "feather", "./features.feather")
connect_database("vectors", "arrow", "./vectors.arrow")
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
# Execute queries with automatic result formatting
execute_query("analytics", "SELECT * FROM users LIMIT 50")

# Large result sets use buffering automatically
execute_query_json("analytics", "SELECT * FROM large_table")
```

#### Handle Large Results

```python
# Get chunked results for large datasets
get_query_chunk("analytics_1640995200_a1b2", 101, "100")

# Check buffer status
get_buffered_query_info("analytics_1640995200_a1b2")

# Manual cleanup
clear_query_buffer("analytics_1640995200_a1b2")
```

## Available Tools

### Core Database Tools

| Tool                 | Description                              | Use Case         |
| -------------------- | ---------------------------------------- | ---------------- |
| `connect_database`   | Connect to databases/files               | Initial setup    |
| `disconnect_database`| Close connections                        | Cleanup          |
| `list_databases`     | Show active connections                  | Status check     |
| `execute_query`      | Run SQL with automatic chunking          | All query needs  |
| `next_chunk`         | Get next chunk of large result sets     | Large data       |
| `describe_database`  | Show database schema                     | Exploration      |
| `describe_table`     | Show table structure                     | Analysis         |
| `find_table`         | Locate tables by name                    | Navigation       |

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

## Supported Data Sources

**Detailed Connection Guide**: See [Database Connections Guide](DATABASE_CONNECTIONS.md) for setup instructions, connection strings, and security practices.

### SQL Databases

- **PostgreSQL**: Full support with connection pooling
- **MySQL**: Complete MySQL/MariaDB compatibility
- **SQLite**: Local file and in-memory databases
- **DuckDB**: High-performance analytical SQL database

### Modern Databases

- **MongoDB**: Document store with collection queries and aggregation
- **Redis**: High-performance key-value store
- **Elasticsearch**: Full-text search and analytics engine
- **InfluxDB**: Time-series database for metrics and IoT data
- **Neo4j**: Graph database for relationship queries
- **CouchDB**: Document-oriented database with HTTP API

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

**Comprehensive Test Coverage**

- 68% test coverage with 500+ test cases
- Import error handling and graceful degradation
- Security vulnerability testing
- Performance benchmarking with large datasets
- Modern database connection testing

**Security Validated**

- Path traversal prevention
- SQL injection protection
- Resource exhaustion testing
- Malicious input handling

**Performance Tested**

- Large file processing
- Concurrent connection handling
- Memory usage optimization
- Query response times

## API Compatibility

All existing MCP tool signatures remain **100% backward compatible**. New functionality is additive only:

- All original tools work unchanged
- Enhanced responses with additional metadata
- New buffering tools for large datasets
- Improved error messages and validation

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
# Connect to large CSV
connect_database("bigdata", "csv", "./million_records.csv")

# Query returns buffer info for large results
result = execute_query_json("bigdata", "SELECT * FROM data WHERE category = 'A'")

# Access results in chunks
chunk = get_query_chunk("bigdata_1640995200_a1b2", 1, "1000")
```

### Multi-Database Analysis

```python
# Connect multiple sources
connect_database("postgres", "postgresql", "postgresql://localhost/prod")
connect_database("config", "yaml", "./config.yaml")
connect_database("logs", "json", "./logs.json")

# Query across sources (in application logic)
user_data = execute_query("postgres", "SELECT * FROM users")
config = read_text_file("./config.yaml", "yaml")
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

# Get sample data from specific sheet
get_table_sample("workbook", "Sheet1")
```


## Troubleshooting

For comprehensive troubleshooting guidance, see [Troubleshooting Guide](TROUBLESHOOTING.md). For common questions, check the [FAQ](FAQ.md).

## Roadmap

### Completed (v1.4.0)

- **Tree Storage**: TOML, JSON, YAML stored as navigable trees with full CRUD
- **9 Tree Tools**: get_node, get_children, set_node, delete_node, list_keys, get_value, set_value, delete_key, export_structured
- **Format Conversion**: Export between TOML, JSON, and YAML
- **Connection Summaries**: Schema/tree overview returned on connect
- **FastMCP v3**: Updated tool registration for latest fastmcp

### Completed (v1.3.x)

- **Memory-Safe Streaming**: Chunked query execution with configurable limits
- **Pre-Query Analysis**: Resource estimation before execution
- **Structured Logging**: JSON logging with Prometheus metrics

### Completed (v1.1.0)

- **Spreadsheet Formats**: Excel (.xlsx/.xls), LibreOffice Calc (.ods) with full multi-sheet support
- **Enhanced File Formats**: XML, INI, TSV support
- **Analytical Formats**: Parquet, Feather, Arrow support

## Contributing

Contributions welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

```bash
git clone https://github.com/ChrisGVE/localdata-mcp.git
cd localdata-mcp
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e ".[dev]"
pytest
```

## License

MIT License - see the [LICENSE](LICENSE) file for details.

## Links

- **GitHub**: [localdata-mcp](https://github.com/ChrisGVE/localdata-mcp)
- **PyPI**: [localdata-mcp](https://pypi.org/project/localdata-mcp/)
- **MCP Protocol**: [Model Context Protocol](https://modelcontextprotocol.io/)
- **FastMCP**: [FastMCP Framework](https://github.com/jlowin/fastmcp)



## Tags

`mcp` `model-context-protocol` `database` `postgresql` `mysql` `sqlite` `mongodb` `spreadsheet` `excel` `xlsx` `ods` `csv` `tsv` `json` `yaml` `toml` `xml` `ini` `parquet` `feather` `arrow` `ai` `machine-learning` `data-integration` `python` `security` `performance`

---

**Made with care for the MCP Community**
