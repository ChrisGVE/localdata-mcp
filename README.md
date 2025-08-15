# LocalData MCP

A dynamic MCP (Model Context Protocol) server for local databases and text files, providing secure database connectivity and query capabilities for AI assistants.

## Features

- **Multi-Database Support**: SQLite, PostgreSQL, MySQL, and text files (CSV, JSON, YAML, TOML)
- **Enhanced Security**: Path sanitization, connection limiting, SQL injection prevention
- **Large Query Handling**: Automatic buffering system for queries returning >100 rows
- **Thread-Safe Operations**: Safe concurrent database access with connection management
- **Temporary File Management**: Automatic cleanup of temporary SQLite databases for large files

## Installation

```bash
pip install localdata-mcp
```

## Usage

Configure the MCP server in your Claude Desktop or other MCP-compatible client:

```json
{
  "mcpServers": {
    "localdata": {
      "command": "localdata-mcp",
      "args": []
    }
  }
}
```

## Available Tools

### Connection Management
- `connect_database` - Open a connection to a database
- `disconnect_database` - Close a database connection
- `list_databases` - List all available connections

### Query Operations
- `execute_query` - Execute SQL query (markdown table output, 100-row limit)
- `execute_query_json` - Execute SQL query (JSON output with buffering for large results)
- `get_query_history` - View recent query history

### Schema Inspection
- `describe_database` - Get database schema information
- `describe_table` - Get detailed table schema
- `find_table` - Find which database contains a specific table

### Data Sampling
- `get_table_sample` - Get sample data from a table (markdown format)
- `get_table_sample_json` - Get sample data from a table (JSON format)

### Query Buffering (for large results)
- `get_query_chunk` - Retrieve chunks from buffered query results
- `get_buffered_query_info` - Get information about buffered queries
- `clear_query_buffer` - Clear specific query buffers

### File Operations
- `read_text_file` - Read structured text files (JSON, YAML, TOML)

## Security Features

- **Path Security**: Restricts file access to current working directory and subdirectories
- **Connection Limiting**: Maximum 10 concurrent database connections
- **SQL Injection Prevention**: Parameterized queries and safe identifier validation
- **Input Validation**: Comprehensive error handling for malicious inputs

## Large File Handling

- Files >100MB are automatically stored in temporary SQLite databases
- Query results >100 rows use buffering system with 10-minute auto-cleanup
- Thread-safe resource management with proper cleanup

## Supported Database Types

- **SQLite**: Local SQLite database files
- **PostgreSQL**: PostgreSQL server connections
- **MySQL**: MySQL server connections
- **CSV**: Comma-separated value files
- **JSON**: JSON data files
- **YAML**: YAML configuration files
- **TOML**: TOML configuration files

## Development

### Testing

Run the comprehensive test suite:

```bash
python test_comprehensive.py
```

### Security Testing

```bash
python test_security_validation.py
```

## License

This project is licensed under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## Author

Christian C. Berclaz