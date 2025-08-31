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

## What's New in v1.3.1 🚀

### Memory-Safe Query Architecture
- **Intelligent Pre-Query Analysis**: Automatic query complexity assessment using `COUNT(*)` and sample row analysis
- **Memory-Bounded Streaming**: Predictable memory usage with configurable limits and streaming pipeline
- **Smart Token Management**: AI-optimized response sizes with automatic chunking for large datasets

### Advanced Configuration System
- **Dual Configuration Mode**: Simple environment variables for basic setups, powerful YAML for complex scenarios
- **Hot Configuration Reload**: Dynamic configuration updates without service restart
- **Multi-Database Support**: Granular per-database settings with timeout and memory controls

### Enhanced Security & Performance
- **SQL Query Validation**: Complete protection against non-SELECT operations
- **Configurable Timeouts**: Per-database and global query timeout enforcement
- **Structured Logging**: JSON logging with detailed query metrics and security events
- **Connection Optimization**: Intelligent connection pooling and resource management

### Developer Experience Improvements
- **Rich Response Metadata**: Query execution stats, memory usage, and optimization hints
- **Progressive Data Loading**: Chunk-based access for massive datasets
- **Enhanced Error Messages**: Actionable guidance for configuration and query issues
- **Backward Compatibility**: 100% API compatibility with automated migration tools

Ready to upgrade? See the [Migration Guide](MIGRATION_GUIDE.md) for step-by-step instructions.

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

- **Comprehensive Tools**: 12 database operation tools
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

| Tool                      | Description                | Use Case      |
| ------------------------- | -------------------------- | ------------- |
| `connect_database`        | Connect to databases/files | Initial setup |
| `disconnect_database`     | Close connections          | Cleanup       |
| `list_databases`          | Show active connections    | Status check  |
| `execute_query`           | Run SQL (markdown output)  | Small results |
| `execute_query_json`      | Run SQL (JSON output)      | Large results |
| `describe_database`       | Show schema/structure      | Exploration   |
| `describe_table`          | Show table details         | Analysis      |
| `get_table_sample`        | Preview table data         | Quick look    |
| `get_table_sample_json`   | Preview (JSON format)      | Development   |
| `find_table`              | Locate tables by name      | Navigation    |
| `read_text_file`          | Read structured files      | File access   |
| `get_query_chunk`         | Paginated result access    | Large data    |
| `get_buffered_query_info` | Buffer status info         | Monitoring    |
| `clear_query_buffer`      | Manual buffer cleanup      | Management    |

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
- **JSON**: Nested structure flattening
- **YAML**: Configuration file support
- **TOML**: Settings and config files
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

### Completed (v1.1.0)

- **Spreadsheet Formats**: Excel (.xlsx/.xls), LibreOffice Calc (.ods) with full multi-sheet support
- **Enhanced File Formats**: XML, INI, TSV support
- **Analytical Formats**: Parquet, Feather, Arrow support

### Planned Features

- **Caching Layer**: Configurable query result caching
- **Connection Pooling**: Advanced connection management
- **Streaming APIs**: Real-time data processing
- **Monitoring Tools**: Connection and performance metrics
- **Export Capabilities**: Query results to various formats

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
