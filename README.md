# LocalData MCP

A secure, feature-rich Model Context Protocol (MCP) server for connecting to local databases and structured data files. Designed for data analysis, debugging, and development workflows with Claude Code and other MCP clients.

## Overview

LocalData MCP enables seamless integration between AI assistants and your local data sources through a unified interface. It provides secure database connectivity, intelligent query buffering for large result sets, and comprehensive support for multiple database types and structured file formats.

### Key Features

- **Multi-Database Support**: SQLite, PostgreSQL, MySQL, CSV, JSON, YAML, and TOML files
- **Intelligent Query Buffering**: Automatic handling of large result sets with chunked retrieval
- **Enhanced Security**: Path restrictions, connection limits, and SQL injection prevention
- **Large File Handling**: Automatic temporary storage for files >100MB
- **Thread-Safe Operations**: Concurrent connection management with resource limiting
- **Comprehensive API**: Full database introspection and data access capabilities

## Installation

### Requirements

- Python 3.8+
- pip or uv package manager

### Install via pip

```bash
pip install localdata-mcp
```

### Install via uv

```bash
uv add localdata-mcp
```

### Development Installation

```bash
git clone https://github.com/your-repo/localdata-mcp
cd localdata-mcp
pip install -e .
```

## Configuration for Claude Code

To use LocalData MCP with Claude Code, add the server configuration to your MCP settings:

### Option 1: Using the executable

Add to your MCP client configuration (e.g., `~/.config/claude-desktop/config.json`):

```json
{
  "mcpServers": {
    "localdata": {
      "command": "localdata-mcp"
    }
  }
}
```

### Option 2: Using Python module

```json
{
  "mcpServers": {
    "localdata": {
      "command": "python",
      "args": ["-m", "localdata_mcp.localdata_mcp"]
    }
  }
}
```

### Option 3: Direct script execution

```json
{
  "mcpServers": {
    "localdata": {
      "command": "python",
      "args": ["/path/to/localdata-mcp/src/localdata_mcp/localdata_mcp.py"]
    }
  }
}
```

### Restart Claude Code

After adding the configuration, restart Claude Code to load the new MCP server.

## Usage Examples

### Connecting to Databases

```python
# Connect to SQLite database
connect_database(name="my_db", db_type="sqlite", conn_string="data.db")

# Connect to PostgreSQL
connect_database(
    name="postgres_db", 
    db_type="postgresql", 
    conn_string="postgresql://user:pass@localhost/dbname"
)

# Connect to CSV file
connect_database(name="csv_data", db_type="csv", conn_string="data.csv")

# Connect to JSON file  
connect_database(name="json_data", db_type="json", conn_string="data.json")
```

### Querying Data

```python
# Execute query with markdown output (≤100 rows)
execute_query(name="my_db", query="SELECT * FROM users LIMIT 10")

# Execute query with JSON output (supports large results)
execute_query_json(name="my_db", query="SELECT * FROM large_table")

# Get table sample
get_table_sample(name="my_db", table_name="users", limit=5)
```

### Working with Large Result Sets

```python
# Query returns >100 rows - automatically buffered
result = execute_query_json(name="my_db", query="SELECT * FROM large_table")

# Extract query_id from result metadata
query_id = "my_db_1642681234_a1b2"

# Get next chunk of results
get_query_chunk(query_id=query_id, start_row=11, chunk_size="100")

# Get all remaining results
get_query_chunk(query_id=query_id, start_row=11, chunk_size="all")

# Check buffer information
get_buffered_query_info(query_id=query_id)

# Clean up buffer when done
clear_query_buffer(query_id=query_id)
```

### Database Introspection

```python
# List all connected databases
list_databases()

# Get detailed database schema
describe_database(name="my_db")

# Find which database contains a table
find_table(table_name="users")

# Get detailed table information
describe_table(name="my_db", table_name="users")

# View query history
get_query_history(name="my_db")
```

## Security

### Important Security Notice

⚠️ **LocalData MCP is designed for controlled local data sources only. It is NOT secure for production environments or untrusted data sources.**

### Security Features

- **Path Restriction**: File access limited to current working directory and subdirectories
- **Connection Limiting**: Maximum 10 concurrent database connections
- **SQL Injection Prevention**: Parameterized queries and safe identifier validation
- **Input Validation**: Comprehensive validation of table names and file paths
- **Resource Management**: Automatic cleanup of temporary files and connections

### Security Limitations

- **No Authentication**: The MCP server has no built-in authentication
- **Local Access Only**: Designed for trusted local development environments
- **File System Access**: Can access any file in allowed directories
- **SQL Execution**: Executes arbitrary SQL queries on connected databases

### Recommended Use Cases

✅ **Appropriate Uses:**
- Local data analysis and exploration
- Development and debugging workflows
- Personal data processing tasks
- Prototype development with trusted data

❌ **Inappropriate Uses:**
- Production applications
- Multi-user environments
- Untrusted or sensitive data sources
- Public-facing services

## Supported Databases

| Database Type | Description | Connection String Format |
|---------------|-------------|-------------------------|
| SQLite | Lightweight file-based database | `path/to/database.db` |
| PostgreSQL | Enterprise database system | `postgresql://user:pass@host:port/dbname` |
| MySQL | Popular relational database | `mysql://user:pass@host:port/dbname` |
| CSV | Comma-separated values | `path/to/file.csv` |
| JSON | JavaScript Object Notation | `path/to/file.json` |
| YAML | Human-readable data format | `path/to/file.yaml` |
| TOML | Configuration file format | `path/to/file.toml` |

### Large File Handling

Files larger than 100MB are automatically stored in temporary SQLite databases to optimize memory usage and query performance.

## API Reference

### Connection Management

#### `connect_database(name: str, db_type: str, conn_string: str)`
Establish a connection to a database or structured file.

**Parameters:**
- `name`: Unique identifier for the connection
- `db_type`: Database type (`sqlite`, `postgresql`, `mysql`, `csv`, `json`, `yaml`, `toml`)
- `conn_string`: Connection string or file path

#### `disconnect_database(name: str)`
Close a database connection and free resources.

#### `list_databases()`
Return list of all active database connections.

### Query Execution

#### `execute_query(name: str, query: str)`
Execute SQL query and return results as markdown table. Limited to ≤100 rows.

#### `execute_query_json(name: str, query: str)`
Execute SQL query and return JSON results. Automatically buffers large result sets.

#### `get_query_history(name: str)`
Retrieve recent query history for a database connection.

### Database Introspection

#### `describe_database(name: str)`
Get comprehensive database schema information including tables, columns, indexes, and relationships.

#### `describe_table(name: str, table_name: str)`
Get detailed information about a specific table including column definitions and constraints.

#### `find_table(table_name: str)`
Find which connected databases contain a specific table.

#### `get_table_sample(name: str, table_name: str, limit: int = 10)`
Retrieve sample data from a table in markdown format.

#### `get_table_sample_json(name: str, table_name: str, limit: int = 10)`
Retrieve sample data from a table in JSON format.

### Query Buffering

#### `get_query_chunk(query_id: str, start_row: int, chunk_size: str)`
Retrieve a specific chunk from a buffered query result.

**Parameters:**
- `query_id`: Buffer identifier from large query result
- `start_row`: Starting row (1-based indexing)
- `chunk_size`: Number of rows or `"all"` for remaining rows

#### `get_buffered_query_info(query_id: str)`
Get metadata about a buffered query including row count and expiration time.

#### `clear_query_buffer(query_id: str)`
Manually remove a query buffer to free memory.

### File Operations

#### `read_text_file(file_path: str, format: str)`
Read and parse structured text files (JSON, YAML, TOML) with security validation.

## Limitations

### Current Limitations

- **Authentication**: No built-in user authentication or authorization
- **Concurrent Writes**: No transaction management for concurrent write operations
- **Database Creation**: Cannot create new databases, only connect to existing ones
- **Schema Modification**: Limited DDL support (varies by database type)
- **Binary Data**: Limited support for binary/blob data types
- **Streaming**: No support for real-time data streaming

### Performance Considerations

- **Memory Usage**: Large result sets are buffered in memory (consider using chunked retrieval)
- **File Size**: Files >100MB use temporary storage (may impact performance)
- **Connection Limits**: Maximum 10 concurrent connections to prevent resource exhaustion
- **Buffer Expiration**: Query buffers automatically expire after 10 minutes

### Platform Support

- **Operating Systems**: Cross-platform (Windows, macOS, Linux)
- **Python Versions**: Requires Python 3.8+
- **Database Drivers**: Depends on system-installed database client libraries

## Contributing

We welcome contributions to improve LocalData MCP! Here's how you can help:

### Reporting Issues

- **Bug Reports**: Use GitHub Issues to report bugs with detailed reproduction steps
- **Feature Requests**: Propose new features or enhancements
- **Security Issues**: Report security vulnerabilities privately via email

### Development

1. **Fork the Repository**: Create your own fork of the project
2. **Create Feature Branch**: `git checkout -b feature/your-feature-name`
3. **Write Tests**: Add tests for new functionality
4. **Follow Code Style**: Maintain consistency with existing code
5. **Submit Pull Request**: Create a PR with clear description of changes

### Testing

```bash
# Run comprehensive tests
python test_comprehensive.py

# Run security validation
python test_security_validation.py

# Run functional tests
python test_functional.py
```

## License and Attribution

### Copyright

Copyright (c) 2024 Christian C. Berclaz. All rights reserved.

### Development

This project was developed using advanced AI assistance:
- **Primary Development**: Gemini 2.5 Flash
- **Code Review and Enhancement**: Claude Code (Sonnet-4)
- **Architecture and Design**: Christian C. Berclaz

### License

[Add your chosen license here - MIT, Apache 2.0, etc.]

## Changelog

### v0.1.0 - Initial Release

- ✅ Multi-database support (SQLite, PostgreSQL, MySQL, CSV, JSON, YAML, TOML)
- ✅ Query buffering system for large result sets
- ✅ Enhanced security with path restrictions and SQL injection prevention
- ✅ Thread-safe connection management
- ✅ Comprehensive API for database operations
- ✅ Large file handling with temporary storage
- ✅ Automatic resource cleanup and connection limiting

## Support

- **Documentation**: This README and inline code documentation
- **Issues**: GitHub Issues for bug reports and feature requests  
- **Discussions**: GitHub Discussions for questions and community support

---

**Note**: LocalData MCP is designed for development and analysis workflows in trusted environments. Always follow security best practices and avoid using with sensitive or production data.