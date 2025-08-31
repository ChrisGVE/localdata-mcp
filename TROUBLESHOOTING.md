# LocalData MCP - Troubleshooting Guide

## Quick Diagnostic Checklist

Before diving into specific issues, run through this quick checklist:

- [ ] LocalData MCP is properly installed: `pip show localdata-mcp`
- [ ] MCP client configuration is correct
- [ ] Database/file paths are accessible
- [ ] Network connectivity is available (for remote databases)
- [ ] You're using supported Python version (3.8+)

## Installation Issues

### Issue: `pip install localdata-mcp` fails

**Common Causes:**
- Outdated pip version
- Python version incompatibility
- Network connectivity issues
- Insufficient permissions

**Solutions:**
```bash
# Update pip first
python -m pip install --upgrade pip

# Install with user permissions if needed
pip install --user localdata-mcp

# Use specific Python version
python3.9 -m pip install localdata-mcp

# Install with verbose output for debugging
pip install -v localdata-mcp
```

### Issue: ModuleNotFoundError after installation

**Symptoms:**
```
ModuleNotFoundError: No module named 'localdata_mcp'
```

**Causes & Solutions:**
1. **Multiple Python environments:**
   ```bash
   # Check which Python pip used
   which pip
   which python
   
   # Ensure they match or use explicit paths
   /usr/local/bin/python3.9 -m pip install localdata-mcp
   ```

2. **Virtual environment issues:**
   ```bash
   # Activate correct virtual environment
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   
   # Then install
   pip install localdata-mcp
   ```

3. **Path issues:**
   ```bash
   # Check installation location
   pip show -f localdata-mcp
   
   # Check Python path
   python -c "import sys; print(sys.path)"
   ```

### Issue: Dependency conflicts

**Symptoms:**
```
ERROR: pip's dependency resolver does not currently consider all installed packages
```

**Solutions:**
```bash
# Create clean virtual environment
python -m venv fresh_env
source fresh_env/bin/activate
pip install localdata-mcp

# Or resolve conflicts
pip install --upgrade --force-reinstall localdata-mcp

# Use pip-tools for dependency management
pip install pip-tools
pip-compile requirements.in
pip-sync requirements.txt
```

## MCP Configuration Issues

### Issue: MCP client doesn't recognize LocalData server

**Check MCP Configuration:**
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

**Common Issues:**
1. **Incorrect command name:**
   ```bash
   # Verify the command exists
   which localdata-mcp
   
   # If not found, use full path
   which python
   # Then use: "/full/path/to/python -m localdata_mcp.localdata_mcp"
   ```

2. **JSON syntax errors:**
   - Use a JSON validator to check syntax
   - Ensure proper escaping of paths on Windows
   - Check for trailing commas

3. **Environment variable issues:**
   ```json
   {
     "mcpServers": {
       "localdata": {
         "command": "localdata-mcp",
         "env": {
           "PYTHONPATH": "/path/to/your/python/packages"
         }
       }
     }
   }
   ```

### Issue: Server starts but tools not available

**Diagnostic Steps:**
1. **Check server startup:**
   ```bash
   # Run directly to see startup messages
   localdata-mcp
   ```

2. **Verify FastMCP integration:**
   ```python
   from localdata_mcp import mcp
   print(mcp.list_tools())
   ```

3. **Check for import errors:**
   ```python
   try:
       from localdata_mcp.localdata_mcp import DatabaseManager
       print("Import successful")
   except ImportError as e:
       print(f"Import error: {e}")
   ```

## Database Connection Issues

### PostgreSQL Connection Problems

**Issue: `psycopg2.OperationalError: could not connect to server`**

**Diagnostic Steps:**
```bash
# Test PostgreSQL connectivity
pg_isready -h hostname -p port -U username

# Check PostgreSQL service
systemctl status postgresql  # Linux
brew services list | grep postgresql  # Mac
```

**Common Solutions:**
1. **PostgreSQL not running:**
   ```bash
   # Start PostgreSQL
   systemctl start postgresql  # Linux
   brew services start postgresql  # Mac
   ```

2. **Wrong connection parameters:**
   ```python
   # Verify each parameter
   connect_database("test", "postgresql", "postgresql://user:pass@host:5432/dbname")
   #                                                    ^     ^     ^     ^      ^
   #                                                  user  pass  host  port   db
   ```

3. **Authentication issues:**
   - Check PostgreSQL `pg_hba.conf` for authentication rules
   - Verify user permissions: `GRANT ALL PRIVILEGES ON DATABASE dbname TO username;`

4. **Network/firewall issues:**
   ```bash
   # Test port connectivity
   telnet hostname 5432
   nc -zv hostname 5432
   ```

### MySQL Connection Problems

**Issue: `mysql.connector.errors.DatabaseError: 2003 (HY000): Can't connect to MySQL server`**

**Diagnostic Steps:**
```bash
# Test MySQL connectivity
mysql -h hostname -P port -u username -p

# Check MySQL service
systemctl status mysql  # Linux
brew services list | grep mysql  # Mac
```

**Common Solutions:**
1. **MySQL not running:**
   ```bash
   # Start MySQL
   systemctl start mysql  # Linux
   brew services start mysql  # Mac
   ```

2. **Wrong connection string format:**
   ```python
   # Correct format
   connect_database("mysql_db", "mysql", "mysql://user:pass@host:3306/dbname")
   
   # Alternative format
   connect_database("mysql_db", "mysql", "mysql+pymysql://user:pass@host:3306/dbname")
   ```

3. **SSL/TLS issues:**
   ```python
   # Disable SSL if needed (not recommended for production)
   connect_database("mysql_db", "mysql", "mysql://user:pass@host:3306/dbname?ssl_disabled=true")
   ```

### SQLite Connection Problems

**Issue: `sqlite3.OperationalError: unable to open database file`**

**Diagnostic Steps:**
```bash
# Check file exists and permissions
ls -la path/to/database.db

# Test SQLite file integrity
sqlite3 path/to/database.db ".schema"
```

**Common Solutions:**
1. **File doesn't exist:**
   ```bash
   # Create empty SQLite database
   sqlite3 new_database.db "CREATE TABLE test (id INTEGER);"
   ```

2. **Permission issues:**
   ```bash
   # Fix file permissions
   chmod 644 database.db
   chmod 755 directory/
   ```

3. **Path security restrictions:**
   ```python
   # LocalData MCP restricts to current working directory
   # Use relative paths from project root
   connect_database("db", "sqlite", "./data/database.db")  # ✅ Allowed
   connect_database("db", "sqlite", "../outside.db")       # ❌ Blocked
   ```

### MongoDB Connection Problems

**Issue: `pymongo.errors.ServerSelectionTimeoutError`**

**Diagnostic Steps:**
```bash
# Test MongoDB connectivity
mongo --host hostname --port 27017

# Check MongoDB service
systemctl status mongod  # Linux
brew services list | grep mongodb  # Mac
```

**Common Solutions:**
1. **MongoDB not running:**
   ```bash
   # Start MongoDB
   systemctl start mongod  # Linux
   brew services start mongodb-community  # Mac
   ```

2. **Connection string format:**
   ```python
   # Standard connection
   connect_database("mongo", "mongodb", "mongodb://localhost:27017/database_name")
   
   # With authentication
   connect_database("mongo", "mongodb", "mongodb://user:pass@localhost:27017/database_name")
   
   # Replica set
   connect_database("mongo", "mongodb", "mongodb://host1:27017,host2:27017/database_name?replicaSet=rs0")
   ```

## File Access Issues

### Issue: "Path outside allowed directory" error

**Cause:** LocalData MCP restricts file access to the current working directory for security.

**Understanding the restriction:**
```python
# Current working directory: /home/user/project

# ✅ Allowed paths
"./data/file.csv"           # /home/user/project/data/file.csv
"data/file.csv"             # /home/user/project/data/file.csv
"subdir/nested/file.json"   # /home/user/project/subdir/nested/file.json

# ❌ Blocked paths  
"../outside.csv"            # /home/user/outside.csv
"/etc/passwd"               # /etc/passwd
"~/Documents/file.yaml"     # /home/user/Documents/file.yaml
```

**Solutions:**
1. **Move files to project directory:**
   ```bash
   # Copy files to project directory
   cp /path/to/external/file.csv ./data/
   ```

2. **Change working directory:**
   ```bash
   # Change to directory containing your files
   cd /path/to/your/data
   # Then run your MCP client from there
   ```

3. **Use symbolic links (if appropriate):**
   ```bash
   # Create symbolic link in project directory
   ln -s /path/to/external/file.csv ./data/external_file.csv
   ```

### Issue: "File not found" for CSV/JSON files

**Diagnostic Steps:**
```bash
# Check if file exists
ls -la path/to/file.csv

# Check current working directory
pwd

# Check file permissions
ls -la path/to/file.csv
```

**Common Solutions:**
1. **Incorrect path:**
   ```python
   # Use absolute paths or relative to current working directory
   import os
   print(f"Current directory: {os.getcwd()}")
   print(f"File exists: {os.path.exists('path/to/file.csv')}")
   ```

2. **File permissions:**
   ```bash
   # Fix file permissions
   chmod 644 file.csv
   ```

3. **File encoding issues:**
   ```python
   # For CSV files with encoding issues
   import pandas as pd
   try:
       df = pd.read_csv('file.csv', encoding='utf-8')
   except UnicodeDecodeError:
       df = pd.read_csv('file.csv', encoding='latin-1')
   ```

## Query Execution Issues

### Issue: "SQL injection attempt detected"

**Cause:** LocalData MCP's security features detected potentially malicious input.

**Example of blocked input:**
```python
# ❌ This will be blocked
describe_table("db", "users; DROP TABLE users; --")
```

**Solutions:**
1. **Use legitimate table names:**
   ```python
   # ✅ Correct usage
   describe_table("db", "users")
   describe_table("db", "order_items")  # Underscores are fine
   ```

2. **Check for special characters:**
   ```python
   # Avoid SQL keywords and special characters in table names
   valid_tables = execute_query("db", "SHOW TABLES")  # or equivalent
   ```

### Issue: "Query buffer expired"

**Cause:** Query results were buffered but the buffer expired after 10 minutes.

**Solutions:**
1. **Re-run the query:**
   ```python
   # Original query that created buffer
   result = execute_query_json("db", "SELECT * FROM large_table")
   
   # If buffer expired, re-run
   result = execute_query_json("db", "SELECT * FROM large_table")
   ```

2. **Access chunks more quickly:**
   ```python
   # Get buffer info immediately after query
   result = execute_query_json("db", "SELECT * FROM large_table")
   buffer_id = result["buffering_info"]["query_id"]
   
   # Access chunks promptly
   chunk1 = get_query_chunk(buffer_id, 11, "1000")
   chunk2 = get_query_chunk(buffer_id, 1011, "1000")
   ```

### Issue: Large queries hang or timeout

**Diagnostic Steps:**
```python
# Check query complexity
explain_result = execute_query("db", "EXPLAIN ANALYZE SELECT * FROM large_table")

# Check table size
table_info = execute_query("db", "SELECT COUNT(*) FROM large_table")
```

**Solutions:**
1. **Add LIMIT clauses:**
   ```python
   # Instead of: SELECT * FROM large_table
   execute_query("db", "SELECT * FROM large_table LIMIT 1000")
   ```

2. **Use pagination:**
   ```python
   # Process in chunks
   offset = 0
   limit = 1000
   while True:
       chunk = execute_query("db", f"SELECT * FROM large_table LIMIT {limit} OFFSET {offset}")
       if not chunk:
           break
       # Process chunk
       offset += limit
   ```

3. **Optimize queries:**
   ```python
   # Add indexes for better performance
   execute_query("db", "CREATE INDEX idx_column ON table_name (column_name)")
   
   # Use specific columns instead of *
   execute_query("db", "SELECT id, name, created_at FROM table_name WHERE condition")
   ```

## Performance Issues

### Issue: High memory usage with large files

**Diagnostic:**
```python
import psutil
import os

process = psutil.Process(os.getpid())
memory_usage = process.memory_info().rss / 1024 / 1024  # MB
print(f"Memory usage: {memory_usage:.2f} MB")
```

**Solutions:**
1. **Verify large file threshold:**
   ```python
   # Files >100MB automatically use SQLite storage
   file_size = os.path.getsize("large_file.csv") / 1024 / 1024  # MB
   print(f"File size: {file_size:.2f} MB")
   ```

2. **Check for memory leaks:**
   ```python
   # Monitor memory before and after operations
   import gc
   gc.collect()  # Force garbage collection
   ```

3. **Process files in chunks:**
   ```python
   # For very large files, consider preprocessing
   import pandas as pd
   
   chunk_size = 10000
   for chunk in pd.read_csv('large_file.csv', chunksize=chunk_size):
       # Process each chunk
       pass
   ```

### Issue: Slow query performance

**Diagnostic Steps:**
```python
import time

# Measure query execution time
start_time = time.time()
result = execute_query("db", "SELECT * FROM table WHERE condition")
end_time = time.time()
print(f"Query took: {end_time - start_time:.2f} seconds")
```

**Solutions:**
1. **Database optimization:**
   ```sql
   -- Add indexes
   CREATE INDEX idx_frequently_queried_column ON table_name (column_name);
   
   -- Update table statistics
   ANALYZE table_name;
   
   -- Check query execution plan
   EXPLAIN ANALYZE SELECT * FROM table_name WHERE condition;
   ```

2. **Query optimization:**
   ```python
   # Use specific columns
   execute_query("db", "SELECT id, name FROM users WHERE active = true")
   
   # Add appropriate WHERE clauses
   execute_query("db", "SELECT * FROM logs WHERE date >= '2024-01-01' LIMIT 1000")
   
   # Use pagination for large results
   execute_query("db", "SELECT * FROM table ORDER BY id LIMIT 100 OFFSET 0")
   ```

## Connection Limit Issues

### Issue: "Maximum connections reached" error

**Cause:** LocalData MCP limits concurrent connections to 10 for stability and security.

**Diagnostic:**
```python
# Check current connections
list_databases()  # Shows active connections
```

**Solutions:**
1. **Close unused connections:**
   ```python
   # Disconnect databases you're no longer using
   disconnect_database("unused_db")
   disconnect_database("temporary_connection")
   ```

2. **Reuse connections:**
   ```python
   # Instead of creating multiple connections to same database
   connect_database("db1", "postgresql", "postgresql://localhost/db")
   connect_database("db2", "postgresql", "postgresql://localhost/db")  # Unnecessary
   
   # Use one connection for multiple operations
   connect_database("main_db", "postgresql", "postgresql://localhost/db")
   result1 = execute_query("main_db", "SELECT * FROM table1")
   result2 = execute_query("main_db", "SELECT * FROM table2")
   ```

3. **Sequential processing:**
   ```python
   # Process databases sequentially instead of concurrently
   for db_config in database_configs:
       connect_database("temp", db_config["type"], db_config["connection"])
       process_database("temp")
       disconnect_database("temp")
   ```

## Logging and Debugging

### Enable Debug Logging

```python
import logging

# Enable debug logging for LocalData MCP
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('localdata_mcp')
logger.setLevel(logging.DEBUG)

# Now run your operations - you'll see detailed logs
connect_database("debug_db", "sqlite", "./debug.db")
```

### Common Log Messages and Meanings

**INFO Messages:**
- `Connection established for database 'name'` - Successful connection
- `Query executed successfully` - Query completed without errors
- `Buffer created for query_id` - Large result set was buffered

**WARNING Messages:**
- `Connection limit approaching` - Near the 10 connection limit
- `Large file detected, using SQLite storage` - File >100MB processing
- `Query buffer cleanup triggered` - Automatic buffer cleanup occurred

**ERROR Messages:**
- `Database connection failed` - Connection string or server issues
- `Path security violation` - File access outside allowed directory
- `SQL injection attempt blocked` - Malicious input detected
- `Query execution failed` - Database or SQL syntax error

### Collecting Diagnostic Information

For bug reports or support requests, collect this information:

```python
import sys
import platform
import localdata_mcp

print(f"Python version: {sys.version}")
print(f"Platform: {platform.platform()}")
print(f"LocalData MCP version: {localdata_mcp.__version__}")
print(f"Current working directory: {os.getcwd()}")

# Database-specific versions
import sqlalchemy
import pandas
print(f"SQLAlchemy version: {sqlalchemy.__version__}")
print(f"Pandas version: {pandas.__version__}")
```

## Getting Additional Help

If you can't resolve your issue using this guide:

1. **Check the FAQ**: [FAQ.md](FAQ.md) for common questions
2. **Search Issues**: Check [GitHub Issues](https://github.com/ChrisGVE/localdata-mcp/issues) for similar problems
3. **Create an Issue**: Include:
   - Python version and platform
   - LocalData MCP version
   - Complete error messages
   - Minimal code to reproduce the issue
   - Relevant log output (with sensitive data removed)

4. **Community Support**: Join MCP community discussions for broader help

## Emergency Recovery

### Complete Reset

If LocalData MCP is completely broken:

```bash
# 1. Uninstall completely
pip uninstall localdata-mcp

# 2. Clear Python cache
python -Bc "import pathlib; [p.unlink() for p in pathlib.Path('.').rglob('*.py[co]')]"
python -Bc "import pathlib, shutil; [shutil.rmtree(p) for p in pathlib.Path('.').rglob('__pycache__')]"

# 3. Clean install
pip install --no-cache-dir localdata-mcp

# 4. Verify installation
python -c "from localdata_mcp import DatabaseManager; print('Installation successful')"
```

### Data Recovery

If you've lost database connections or query buffers:

```python
# Reconnect to databases
connect_database("restored_db", "sqlite", "./backup.db")

# Query history is preserved per session
get_query_history("restored_db")  # Shows recent queries

# Re-run important queries
important_data = execute_query("restored_db", "SELECT * FROM critical_table")
```

## Spreadsheet Format Issues

### Large Excel Files

```python
# For files over 100MB, temporary SQLite storage is used automatically
connect_database("largefile", "xlsx", "./large_workbook.xlsx")

# Monitor processing with describe_database
describe_database("largefile")  # Shows processing status
```

### Sheet Name Conflicts

```python
# If sheet names conflict after sanitization, use specific sheet selection
connect_database("specific", "xlsx", "./workbook.xlsx?sheet=Sheet1")

# Check sanitized names
describe_database("workbook")  # Lists all table names
```

### Format Detection

```python
# Ensure correct file extension for proper format detection
connect_database("data", "xlsx", "./file.xlsx")  # ✅ Correct
connect_database("data", "xlsx", "./file.xls")   # ⚠️ May cause issues

# Use explicit format specification
connect_database("data", "xls", "./old_format.xls")  # ✅ Better
```

### Multi-Sheet Selection Issues

```python
# Sheet names with special characters need URL encoding
connect_database("data", "xlsx", "./file.xlsx?sheet=Q1%20Sales")  # For "Q1 Sales"

# Or use the sanitized table name after connecting all sheets
connect_database("workbook", "xlsx", "./file.xlsx")
execute_query("workbook", "SELECT * FROM Q1_Sales")  # Use sanitized name
```

### Performance Optimization

```python
# For better performance with large spreadsheets:
# 1. Use specific sheet selection when possible
connect_database("q1", "xlsx", "./large.xlsx?sheet=Q1_Data")

# 2. Use LIMIT clauses for large datasets
execute_query("data", "SELECT * FROM large_sheet LIMIT 1000")

# 3. Consider converting to Parquet for repeated analysis
# (Manual conversion outside of LocalData MCP recommended for very large files)
```

## General File Issues

### Path Security Errors

```python
# ✅ Allowed paths (current directory and subdirectories)
connect_database("data", "csv", "./data/file.csv")
connect_database("data", "csv", "subfolder/file.csv")

# ❌ Blocked paths (parent directories)
connect_database("data", "csv", "../data/file.csv")  # Security error
```

### Connection Limits

```python
# Maximum 10 concurrent connections
# Use disconnect_database() to free up connections when done
disconnect_database("old_connection")
```

## Need Help?

- **Issues**: [GitHub Issues](https://github.com/ChrisGVE/localdata-mcp/issues)
- **Discussions**: [GitHub Discussions](https://github.com/ChrisGVE/localdata-mcp/discussions)
- **FAQ**: [FAQ.md](FAQ.md) for common questions

This guide covers the most common issues. For problems not addressed here, please create an issue on GitHub with detailed information about your specific situation.