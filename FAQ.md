# LocalData MCP - Frequently Asked Questions

## General Questions

### Q: What is LocalData MCP?
**A:** LocalData MCP is a production-ready MCP (Model Context Protocol) server that provides secure, comprehensive database integration for AI applications. It supports multiple database types (PostgreSQL, MySQL, SQLite, MongoDB) and structured files (CSV, JSON, YAML, TOML) through a unified API with enterprise-grade security features.

### Q: How is LocalData MCP different from other MCP database servers?
**A:** LocalData MCP offers several unique advantages:
- **Universal support**: One server handles SQL databases, document databases, and structured files
- **Enterprise security**: Built-in path restrictions, SQL injection prevention, and connection limits
- **Large dataset intelligence**: Automatic handling of 100MB+ files and large query results
- **Production ready**: 100+ test cases covering security, performance, and edge cases
- **Backward compatible**: All existing MCP APIs remain unchanged

### Q: Is LocalData MCP ready for production use?
**A:** Yes. LocalData MCP has been designed specifically for production deployment with:
- Comprehensive security controls
- Extensive test coverage (100+ test cases)
- Thread-safe architecture
- Memory-efficient processing
- Proper error handling and resource cleanup
- Battle-tested with large datasets and concurrent usage

## Installation & Setup

### Q: How do I install LocalData MCP?
**A:** Installation is straightforward using pip:
```bash
# Using pip (standard)
pip install localdata-mcp

# Using uv (recommended for faster installation)
uv tool install localdata-mcp

# Development installation
git clone https://github.com/ChrisGVE/localdata-mcp.git
cd localdata-mcp
pip install -e .
```

### Q: What are the system requirements?
**A:** LocalData MCP requires:
- **Python**: 3.8 or higher
- **Operating System**: Windows, macOS, or Linux
- **Memory**: Minimum 512MB RAM (more recommended for large datasets)
- **Disk Space**: Temporary space for large file processing (automatically managed)

### Q: How do I configure LocalData MCP with my MCP client?
**A:** Add the following to your MCP client configuration:
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

### Q: Do I need to install database drivers separately?
**A:** No. LocalData MCP includes all necessary database drivers:
- PostgreSQL: `psycopg2-binary`
- MySQL: `mysql-connector-python`
- SQLite: Built into Python
- MongoDB: Support included
- File formats: Built-in pandas, pyyaml, and toml support

## Database Connectivity

### Q: What databases are supported?
**A:** LocalData MCP supports:
- **SQL Databases**: PostgreSQL, MySQL, SQLite
- **Document Databases**: MongoDB
- **Structured Files**: CSV, JSON, YAML, TOML

### Q: How do I connect to a PostgreSQL database?
**A:** Use the `connect_database` tool:
```python
connect_database(
    "mydb", 
    "postgresql", 
    "postgresql://username:password@hostname:port/database_name"
)
```

### Q: Can I connect to multiple databases simultaneously?
**A:** Yes. LocalData MCP supports up to 10 concurrent database connections for performance and security reasons. You can connect to different database types simultaneously:
```python
connect_database("postgres_db", "postgresql", "postgresql://localhost/app_db")
connect_database("mysql_db", "mysql", "mysql://localhost/analytics") 
connect_database("config", "yaml", "./config.yaml")
connect_database("data", "csv", "./dataset.csv")
```

### Q: How do I handle connection strings with special characters?
**A:** URL-encode special characters in connection strings:
```python
# Password with special characters
password = "myp@ss!word"
# URL-encode as: myp%40ss%21word
connect_database("db", "postgresql", "postgresql://user:myp%40ss%21word@localhost/db")
```

### Q: Can I connect to databases on remote servers?
**A:** Yes, LocalData MCP supports remote database connections. Ensure:
- Network connectivity to the remote server
- Proper authentication credentials
- Firewall rules allow the connection
- Use SSL/TLS for security when possible

## Security

### Q: How does LocalData MCP prevent path traversal attacks?
**A:** LocalData MCP implements robust path security:
- All file paths are resolved to absolute paths
- Paths are validated to ensure they're within the current working directory
- Attempts to access parent directories (`../`) or system files are blocked
- Example: `"../../../etc/passwd"` → **BLOCKED**
- Allowed: `"./data/file.csv"` or `"subdir/data.json"`

### Q: How is SQL injection prevented?
**A:** LocalData MCP uses multiple layers of SQL injection protection:
- **Parameterized queries**: All user input is properly parameterized
- **Table name validation**: Table names are validated against known schema
- **Quoted identifiers**: SQLAlchemy's `quoted_name()` for safe identifiers
- **Input sanitization**: Comprehensive validation of all inputs

### Q: What are the connection limits and why?
**A:** LocalData MCP limits concurrent connections to 10 to:
- Prevent resource exhaustion
- Maintain stable performance
- Protect against potential abuse
- Ensure fair resource allocation
This limit is thread-safe and automatically managed.

### Q: Can I access files outside my project directory?
**A:** No. For security reasons, LocalData MCP restricts file access to the current working directory and its subdirectories only. This prevents accidental or malicious access to system files, user data, or files outside your project scope.

## Performance & Large Data

### Q: How does LocalData MCP handle large CSV files?
**A:** LocalData MCP automatically detects file sizes and optimizes processing:
- **Files < 100MB**: Processed directly in memory using pandas
- **Files ≥ 100MB**: Automatically converted to temporary SQLite database for efficient querying
- **Memory usage**: Remains constant regardless of file size
- **Cleanup**: Temporary files are automatically removed when no longer needed

### Q: What happens when query results are very large?
**A:** LocalData MCP implements intelligent query buffering:
- **Results < 100 rows**: Returned immediately in full
- **Results ≥ 100 rows**: Automatically buffered with chunk-based access
- **Pagination**: Use `get_query_chunk()` to retrieve specific portions
- **Auto-cleanup**: Buffers expire after 10 minutes with file modification detection

### Q: How do I access large query results?
**A:** Use the buffering system:
```python
# Large query triggers automatic buffering
result = execute_query_json("bigdata", "SELECT * FROM large_table")
# Returns: {"metadata": {...}, "first_10_rows": [...], "buffering_info": {...}}

# Access specific chunks
chunk = get_query_chunk("bigdata_1640995200_a1b2", 101, "1000")  # Rows 101-1100

# Check buffer status
info = get_buffered_query_info("bigdata_1640995200_a1b2")

# Manual cleanup (optional)
clear_query_buffer("bigdata_1640995200_a1b2")
```

### Q: Is LocalData MCP thread-safe for concurrent use?
**A:** Yes. LocalData MCP is designed with thread safety in mind:
- Connection management uses semaphores and locks
- Query buffering is protected with thread locks
- Resource cleanup is handled atomically
- Concurrent access is tested and validated

## Error Handling & Troubleshooting

### Q: What should I do if I get a "Database not connected" error?
**A:** This error means you haven't established a connection to the specified database. Use `connect_database` first:
```python
# First connect
connect_database("mydb", "sqlite", "./data.db")

# Then query
execute_query("mydb", "SELECT * FROM users")
```

### Q: How do I debug connection issues?
**A:** Check these common issues:
1. **Database running**: Ensure the database server is running
2. **Credentials**: Verify username, password, host, port, and database name
3. **Network**: Test network connectivity to remote databases
4. **Permissions**: Ensure the user has necessary database permissions
5. **Connection string format**: Verify the connection string syntax

### Q: What does "File not found" mean for CSV/JSON files?
**A:** This error indicates:
- The file path doesn't exist
- The file is outside the allowed directory (security restriction)
- Incorrect file permissions
- File path contains special characters not properly handled

### Q: How do I handle "Query buffer expired" errors?
**A:** Query buffers automatically expire after 10 minutes. To handle this:
```python
# Check if buffer still exists
try:
    chunk = get_query_chunk("query_id", 1, "100")
except ValueError as e:
    # Buffer expired, re-run the query
    result = execute_query_json("db", "SELECT * FROM table")
```

## Advanced Usage

### Q: Can I use LocalData MCP with environment variables for sensitive data?
**A:** Yes, you can use environment variables in your connection strings:
```python
import os

# Connection with environment variables
db_password = os.getenv('DB_PASSWORD')
connect_database(
    "prod", 
    "postgresql", 
    f"postgresql://user:{db_password}@localhost/db"
)
```

### Q: How do I migrate from development to production?
**A:** LocalData MCP makes environment transitions seamless:
```python
# Development
connect_database("app_db", "sqlite", "./dev.db")

# Production (same queries work)
connect_database("app_db", "postgresql", "postgresql://prod-server/db")

# Queries remain identical
data = execute_query("app_db", "SELECT * FROM users WHERE active = true")
```

### Q: Can I use LocalData MCP for data analysis workflows?
**A:** Absolutely. LocalData MCP is perfect for data analysis:
```python
# Connect multiple data sources
connect_database("sales", "postgresql", "postgresql://localhost/sales_db")
connect_database("config", "yaml", "./analysis_config.yaml")
connect_database("external", "csv", "./market_data.csv")

# Analyze across sources
sales_data = execute_query("sales", "SELECT product, SUM(revenue) FROM orders GROUP BY product")
config = read_text_file("./analysis_config.yaml", "yaml")
market_data = execute_query("external", "SELECT * FROM data WHERE sector = 'tech'")
```

### Q: How do I contribute to LocalData MCP development?
**A:** Contributions are welcome! Here's how to get started:
1. **Fork** the repository: https://github.com/ChrisGVE/localdata-mcp
2. **Clone** your fork and create a feature branch
3. **Install** development dependencies: `pip install -e ".[dev]"`
4. **Make** your changes with tests
5. **Run** the test suite: `pytest`
6. **Submit** a pull request with a clear description

Areas where contributions are especially appreciated:
- Additional database driver support
- Performance optimizations
- Security enhancements
- Documentation improvements
- Bug fixes and testing

## Licensing & Support

### Q: What license is LocalData MCP released under?
**A:** LocalData MCP is released under the MIT License, which allows for both commercial and non-commercial use, modification, and distribution.

### Q: How do I get support?
**A:** Support is available through:
- **GitHub Issues**: https://github.com/ChrisGVE/localdata-mcp/issues
- **Documentation**: Comprehensive README and examples
- **Community**: MCP community forums and discussions
- **Email**: For specific questions (see GitHub profile)

### Q: Is commercial use permitted?
**A:** Yes. The MIT License explicitly permits commercial use without restrictions.

### Q: How often is LocalData MCP updated?
**A:** Updates are released as needed for:
- Bug fixes (patch releases)
- New features (minor releases)
- Security updates (immediate releases)
- Major architectural changes (major releases)

Follow the GitHub repository for notifications of new releases.

---

## Getting Help

**Can't find your question here?** 

- **Check the README**: https://github.com/ChrisGVE/localdata-mcp/blob/main/README.md
- **Browse Issues**: https://github.com/ChrisGVE/localdata-mcp/issues
- **Create an Issue**: For bugs or feature requests
- **Start a Discussion**: For general questions and community support

**For immediate help**, the comprehensive examples in the README and test files provide practical guidance for most common use cases.