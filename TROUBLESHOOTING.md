# Troubleshooting

## Quick diagnostic checklist

- [ ] LocalData MCP is installed: `localdata-mcp --version`
- [ ] Python 3.10+ is available: `python --version`
- [ ] MCP client configuration points to the correct command
- [ ] Database services are running (for remote connections)
- [ ] File paths are within allowed directories

## Installation

### pip install fails

```bash
# Update pip first
python -m pip install --upgrade pip

# Use uv for faster, more reliable installs
uv tool install localdata-mcp

# For development
git clone https://github.com/ChrisGVE/localdata-mcp.git
cd localdata-mcp
uv sync --dev
```

### ModuleNotFoundError after installation

Check that `pip` and `python` point to the same environment:

```bash
which pip
which python

# If using a virtual environment, make sure it is activated
source venv/bin/activate
```

### First-run timeout

Data science dependencies (scipy, scikit-learn, statsmodels, geopandas) total around 200 MB and are downloaded on first use. If your MCP client times out waiting for the server to start, reconnect from the client interface. Subsequent starts reuse the cached packages and are immediate.

## MCP client configuration

### Server not recognized

Verify the command exists and is on your PATH:

```bash
which localdata-mcp

# If not found, use the full path
localdata-mcp --version
```

Minimal MCP client configuration:

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

### Server starts but tools are not listed

Run the server directly to check for startup errors:

```bash
localdata-mcp 2>&1 | head -20
```

Verify the package imports correctly:

```bash
python -c "from localdata_mcp.localdata_mcp import main; print('OK')"
```

## Database connections

### PostgreSQL

**"could not connect to server"**

```bash
# Check if PostgreSQL is running
pg_isready -h localhost -p 5432

# Verify the connection string format
# postgresql://user:password@host:5432/dbname
```

Common causes: service not running, wrong host/port, firewall rules, authentication method mismatch in `pg_hba.conf`.

For SSL connections, append `?sslmode=require` to the connection string.

### MySQL / MariaDB

**"Can't connect to MySQL server"**

```bash
# Check if MySQL is running
mysqladmin -h localhost -P 3306 status

# Connection string format
# mysql://user:password@host:3306/dbname
```

### SQLite

**"unable to open database file"**

- Verify the file exists and has read permissions
- Ensure the path is within the allowed directory (security restriction)

```python
# Relative paths resolve from the server's working directory
connect_database("db", "sqlite", "./data/mydb.db")      # allowed
connect_database("db", "sqlite", "../outside/mydb.db")   # blocked
```

### DuckDB

Same path restrictions as SQLite. Use `:memory:` for temporary databases:

```python
connect_database("tmp", "duckdb", ":memory:")
```

### MongoDB, Redis, Elasticsearch, and other NoSQL

These require the `modern-databases` extra:

```bash
uv tool install "localdata-mcp[modern-databases]"
```

### Oracle and MS SQL Server

These require the `enterprise` extra:

```bash
uv tool install "localdata-mcp[enterprise]"
```

See the [configuration guide](docs/configuration.md) for authentication methods (Kerberos, Oracle Wallet, Azure AD, certificates).

### Connection limits

LocalData MCP limits concurrent connections to 10 by default. Check active connections with `list_databases` and close unused ones with `disconnect_database`. Adjust the limit via `LOCALDATA_CONNECTIONS_MAX_CONCURRENT` or in your YAML config.

## File access

### "Path outside allowed directory"

LocalData MCP restricts file access to the current working directory by default. This is a security feature.

```python
# Allowed (within working directory)
connect_database("data", "csv", "./reports/sales.csv")
connect_database("data", "csv", "subfolder/data.json")

# Blocked (outside working directory)
connect_database("data", "csv", "../other-project/data.csv")
connect_database("data", "csv", "/etc/passwd")
```

**Solutions:**

1. Move files into the project directory
2. Change the server's working directory to the parent of your data
3. Add allowed paths in configuration:
   ```yaml
   security:
     allowed_paths: ["/data/shared", "/home/user/datasets"]
   ```

### Spreadsheet issues

- Large Excel/ODS/Numbers files are automatically staged into a temporary SQLite database for efficient querying
- Use `describe_database` after connecting to see the sanitized table (sheet) names
- For specific sheets: set `sheet_name` in the YAML config or pass it at connect time

## Query execution

### Streaming and chunked results

Queries that return more than `default_chunk_size` rows (100 by default) are streamed in chunks. The first call returns metadata and the initial chunk. Use `next_chunk` to retrieve subsequent chunks, or `request_data_chunk` for random access.

If a buffer expires (default: 10 minutes), re-run the query. Adjust timeouts via:

```yaml
query:
  buffer_timeout_seconds: 1800   # 30 minutes
  default_chunk_size: 500        # rows per chunk
```

### Queries are slow

- Use `preflight=true` on `execute_query` to get a cost estimate before running expensive queries
- Add `LIMIT` clauses for exploratory queries
- Check that tables have appropriate indexes
- For large file-based sources, the initial staging step may take time; subsequent queries against the staged data are fast

### Memory issues

The server automatically manages memory with configurable budgets. If you see memory warnings:

```yaml
memory:
  max_budget_mb: 1024
  budget_percent: 15

performance:
  memory_limit_mb: 4096
```

The streaming executor processes results in chunks without loading entire result sets into memory.

## Configuration

### Validate your config

```bash
localdata-mcp --validate-config
```

### View resolved config (credentials redacted)

```bash
localdata-mcp --show-config
```

### Create a default config file

```bash
localdata-mcp --init-config
```

### Migrate legacy config location

If you have a config at `~/.localdata.yaml`, move it to the platform-appropriate location:

```bash
localdata-mcp --migrate-config
```

See the [configuration documentation](docs/configuration.md) for the full reference.

## Logging

Enable debug logging to diagnose issues:

```bash
# Via environment variable
LOCALDATA_LOG_LEVEL=debug localdata-mcp

# Via YAML config
logging:
  level: debug
  console_output: true
  file_path: ./localdata-debug.log
```

## Docker

See [DOCKER_USAGE.md](DOCKER_USAGE.md) for container-specific troubleshooting (permissions, networking, resource limits).

## Collecting diagnostic information

When filing a bug report, include:

```bash
localdata-mcp --version
python --version
uname -a                    # OS info
localdata-mcp --show-config # Resolved config (credentials are redacted)
```

Plus the complete error message and minimal steps to reproduce the issue.

## Getting help

1. Search [GitHub Issues](https://github.com/ChrisGVE/localdata-mcp/issues) for similar problems
2. Read the [documentation](https://localdata-mcp.readthedocs.io)
3. Open a new issue with diagnostic information
4. For security vulnerabilities, email `christian@berclaz.org` directly
