[![MseeP.ai Security Assessment Badge](https://mseep.net/pr/chrisgve-localdata-mcp-badge.png)](https://mseep.ai/app/chrisgve-localdata-mcp)

# LocalData MCP Server

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://img.shields.io/pypi/v/localdata-mcp.svg)](https://pypi.org/project/localdata-mcp/)
[![FastMCP](https://img.shields.io/badge/FastMCP-Compatible-green.svg)](https://github.com/jlowin/fastmcp)
[![Verified on MseeP](https://mseep.ai/badge.svg)](https://mseep.ai/app/cd737717-02f3-4388-bab7-5ec7cbe40713)

**A powerful, secure MCP server for local databases, spreadsheets, and structured data files with advanced security features and large dataset handling.**

## ✨ Features

### 🗄️ **Multi-Database Support**

- **SQL Databases**: PostgreSQL, MySQL, SQLite
- **Document Databases**: MongoDB
- **Spreadsheets**: Excel (.xlsx/.xls), LibreOffice Calc (.ods) with multi-sheet support
- **Structured Files**: CSV, TSV, JSON, YAML, TOML, XML, INI
- **Analytical Formats**: Parquet, Feather, Arrow

### 🔒 **Advanced Security**

- **Path Security**: Restricts file access to current working directory only
- **SQL Injection Prevention**: Parameterized queries and safe table identifiers
- **Connection Limits**: Maximum 10 concurrent database connections
- **Input Validation**: Comprehensive validation and sanitization

### 📊 **Large Dataset Handling**

- **Query Buffering**: Automatic buffering for results with 100+ rows
- **Large File Support**: 100MB+ files automatically use temporary SQLite storage
- **Chunk Retrieval**: Paginated access to large result sets
- **Auto-Cleanup**: 10-minute expiry with file modification detection

### 🛠️ **Developer Experience**

- **Comprehensive Tools**: 12 database operation tools
- **Error Handling**: Detailed, actionable error messages
- **Thread Safety**: Concurrent operation support
- **Backward Compatible**: All existing APIs preserved

## 🚀 Quick Start

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

## 🔧 Available Tools

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

## 📋 Supported Data Sources

### SQL Databases

- **PostgreSQL**: Full support with connection pooling
- **MySQL**: Complete MySQL/MariaDB compatibility
- **SQLite**: Local file and in-memory databases

### Document Databases

- **MongoDB**: Collection queries and aggregation

### Structured Files

#### Spreadsheet Formats
- **Excel (.xlsx, .xls)**: Full multi-sheet support with automatic table creation
- **LibreOffice Calc (.ods)**: Complete ODS support with sheet handling
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

## 🛡️ Security Features

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

## 📊 Performance & Scalability

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

## 🧪 Testing & Quality

**✅ 100% Test Coverage**

- 100+ comprehensive test cases
- Security vulnerability testing
- Performance benchmarking
- Edge case validation

**🔒 Security Validated**

- Path traversal prevention
- SQL injection protection
- Resource exhaustion testing
- Malicious input handling

**⚡ Performance Tested**

- Large file processing
- Concurrent connection handling
- Memory usage optimization
- Query response times

## 🔄 API Compatibility

All existing MCP tool signatures remain **100% backward compatible**. New functionality is additive only:

- ✅ All original tools work unchanged
- ✅ Enhanced responses with additional metadata
- ✅ New buffering tools for large datasets
- ✅ Improved error messages and validation

## 📖 Examples

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

| Original Sheet Name | SQL Table Name |
| ------------------- | -------------- |
| "Q1 Sales"          | Q1_Sales       |
| "2024-Budget"       | _2024_Budget   |
| "Summary & Notes"   | Summary__Notes |

#### Discovering Available Sheets

```python
# Connect to multi-sheet workbook
connect_database("workbook", "xlsx", "./data.xlsx")

# List all available tables (sheets)
describe_database("workbook")

# Get sample data from specific sheet
get_table_sample("workbook", "Sheet1")
```

## 🚧 Roadmap

### Completed (v1.1.0)
- [x] **Spreadsheet Formats**: Excel (.xlsx/.xls), LibreOffice Calc (.ods) with full multi-sheet support
- [x] **Enhanced File Formats**: XML, INI, TSV support
- [x] **Analytical Formats**: Parquet, Feather, Arrow support

### Planned Features
- [ ] **Caching Layer**: Configurable query result caching
- [ ] **Connection Pooling**: Advanced connection management
- [ ] **Streaming APIs**: Real-time data processing
- [ ] **Monitoring Tools**: Connection and performance metrics
- [ ] **Export Capabilities**: Query results to various formats

## 🛠️ Troubleshooting

### Spreadsheet Format Issues

#### Large Excel Files
```python
# For files over 100MB, temporary SQLite storage is used automatically
connect_database("largefile", "xlsx", "./large_workbook.xlsx")

# Monitor processing with describe_database
describe_database("largefile")  # Shows processing status
```

#### Sheet Name Conflicts
```python
# If sheet names conflict after sanitization, use specific sheet selection
connect_database("specific", "xlsx", "./workbook.xlsx?sheet=Sheet1")

# Check sanitized names
describe_database("workbook")  # Lists all table names
```

#### Format Detection
```python
# Ensure correct file extension for proper format detection
connect_database("data", "xlsx", "./file.xlsx")  # ✅ Correct
connect_database("data", "xlsx", "./file.xls")   # ⚠️ May cause issues

# Use explicit format specification
connect_database("data", "xls", "./old_format.xls")  # ✅ Better
```

#### Multi-Sheet Selection Issues
```python
# Sheet names with special characters need URL encoding
connect_database("data", "xlsx", "./file.xlsx?sheet=Q1%20Sales")  # For "Q1 Sales"

# Or use the sanitized table name after connecting all sheets
connect_database("workbook", "xlsx", "./file.xlsx")
execute_query("workbook", "SELECT * FROM Q1_Sales")  # Use sanitized name
```

#### Performance Optimization
```python
# For better performance with large spreadsheets:
# 1. Use specific sheet selection when possible
connect_database("q1", "xlsx", "./large.xlsx?sheet=Q1_Data")

# 2. Use LIMIT clauses for large datasets
execute_query("data", "SELECT * FROM large_sheet LIMIT 1000")

# 3. Consider converting to Parquet for repeated analysis
# (Manual conversion outside of LocalData MCP recommended for very large files)
```

### General File Issues

#### Path Security Errors
```python
# ✅ Allowed paths (current directory and subdirectories)
connect_database("data", "csv", "./data/file.csv")
connect_database("data", "csv", "subfolder/file.csv")

# ❌ Blocked paths (parent directories)
connect_database("data", "csv", "../data/file.csv")  # Security error
```

#### Connection Limits
```python
# Maximum 10 concurrent connections
# Use disconnect_database() to free up connections when done
disconnect_database("old_connection")
```

## 🤝 Contributing

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

## 📄 License

MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Links

- **GitHub**: [localdata-mcp](https://github.com/ChrisGVE/localdata-mcp)
- **PyPI**: [localdata-mcp](https://pypi.org/project/localdata-mcp/)
- **MCP Protocol**: [Model Context Protocol](https://modelcontextprotocol.io/)
- **FastMCP**: [FastMCP Framework](https://github.com/jlowin/fastmcp)

## 📊 Stats

![GitHub stars](https://img.shields.io/github/stars/ChrisGVE/localdata-mcp?style=social)
![GitHub forks](https://img.shields.io/github/forks/ChrisGVE/localdata-mcp?style=social)
![PyPI downloads](https://img.shields.io/pypi/dm/localdata-mcp)

## 📚 Additional Resources

- **[FAQ](FAQ.md)**: Common questions and troubleshooting
- **[Troubleshooting Guide](TROUBLESHOOTING.md)**: Comprehensive problem resolution
- **[Advanced Examples](ADVANCED_EXAMPLES.md)**: Production-ready usage patterns
- **[Blog Post](BLOG_POST.md)**: Technical deep dive and use cases

## 🤔 Need Help?

- **Issues**: [GitHub Issues](https://github.com/ChrisGVE/localdata-mcp/issues)
- **Discussions**: [GitHub Discussions](https://github.com/ChrisGVE/localdata-mcp/discussions)
- **Email**: Available in GitHub profile
- **Community**: Join MCP community forums

## 🏷️ Tags

`mcp` `model-context-protocol` `database` `postgresql` `mysql` `sqlite` `mongodb` `spreadsheet` `excel` `xlsx` `ods` `csv` `tsv` `json` `yaml` `toml` `xml` `ini` `parquet` `feather` `arrow` `ai` `machine-learning` `data-integration` `python` `security` `performance`

---

**Made with ❤️ for the MCP Community**
