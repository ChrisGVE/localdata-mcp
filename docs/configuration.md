# Configuration

LocalData MCP supports configuration through YAML files, environment variables, or a combination of both. Environment variables always override YAML values, and YAML values override built-in defaults.

## Config file locations

The server searches for configuration files in the following order and uses the **first one found**:

| Priority | Path | Platform | Description |
|----------|------|----------|-------------|
| 0 (highest) | `$LOCALDATA_CONFIG` | All | Explicit env var override |
| 1 | `./.localdata.yaml` | All | Project-local config |
| 2 | `$XDG_CONFIG_HOME/localdata/config.yaml` | Linux, macOS | XDG user config |
| 3 | `~/Library/Application Support/localdata/config.yaml` | macOS | macOS user config |
| 2 | `%APPDATA%/localdata/config.yaml` | Windows | Windows user config |
| 4 | `/etc/localdata/config.yaml` | Linux, macOS | System-wide config |
| 5 (lowest) | `~/.localdata.yaml` | All | Legacy (deprecated) |

On Linux, `$XDG_CONFIG_HOME` defaults to `~/.config` when unset.

Using the legacy path (`~/.localdata.yaml`) emits a deprecation warning at startup. See [Migration guide](#migration-guide) for how to move to the recommended location.

## Environment variables

Environment variables override any values set in YAML files. All variables use the `LOCALDATA_` prefix.

### General

| Variable | Type | Description |
|----------|------|-------------|
| `LOCALDATA_CONFIG` | `str` | Path to a specific YAML config file |
| `LOCALDATA_LOG_LEVEL` | `str` | Log level: `debug`, `info`, `warning`, `error`, `critical` |
| `LOCALDATA_LOG_FILE` | `str` | Path to log output file |
| `LOCALDATA_MEMORY_LIMIT_MB` | `int` | Overall memory limit in MB (performance section) |

### Database connections

Define databases with the pattern `LOCALDATA_DB_<NAME>_<PROPERTY>`:

| Variable pattern | Type | Description |
|------------------|------|-------------|
| `LOCALDATA_DB_<NAME>_TYPE` | `str` | Database type (e.g., `sqlite`, `postgresql`) |
| `LOCALDATA_DB_<NAME>_CONNECTION_STRING` | `str` | Connection string or file path |
| `LOCALDATA_DB_<NAME>_SHEET_NAME` | `str` | Sheet name for multi-sheet formats |
| `LOCALDATA_DB_<NAME>_ENABLED` | `bool` | Enable or disable (`true`/`false`) |
| `LOCALDATA_DB_<NAME>_MAX_CONNECTIONS` | `int` | Connection pool size |
| `LOCALDATA_DB_<NAME>_CONNECTION_TIMEOUT` | `int` | Connection timeout in seconds |
| `LOCALDATA_DB_<NAME>_QUERY_TIMEOUT` | `int` | Query timeout in seconds |

`<NAME>` is case-insensitive and converted to lowercase internally. For example, `LOCALDATA_DB_MYDB_TYPE=sqlite` creates a database named `mydb`.

### Staging

| Variable | Maps to | Type | Default |
|----------|---------|------|---------|
| `LOCALDATA_STAGING_MAX_CONCURRENT` | `staging.max_concurrent` | `int` | `10` |
| `LOCALDATA_STAGING_MAX_SIZE_MB` | `staging.max_size_mb` | `int` | `2048` |
| `LOCALDATA_STAGING_MAX_TOTAL_MB` | `staging.max_total_mb` | `int` | `10240` |
| `LOCALDATA_STAGING_TIMEOUT_MINUTES` | `staging.timeout_minutes` | `int` | `30` |
| `LOCALDATA_STAGING_EVICTION_POLICY` | `staging.eviction_policy` | `str` | `lru` |

### Memory

| Variable | Maps to | Type | Default |
|----------|---------|------|---------|
| `LOCALDATA_MEMORY_MAX_BUDGET_MB` | `memory.max_budget_mb` | `int` | `512` |
| `LOCALDATA_MEMORY_BUDGET_PERCENT` | `memory.budget_percent` | `int` | `10` |
| `LOCALDATA_MEMORY_LOW_THRESHOLD_GB` | `memory.low_memory_threshold_gb` | `float` | `1.0` |

### Query execution

| Variable | Maps to | Type | Default |
|----------|---------|------|---------|
| `LOCALDATA_QUERY_CHUNK_SIZE` | `query.default_chunk_size` | `int` | `100` |
| `LOCALDATA_QUERY_BUFFER_TIMEOUT` | `query.buffer_timeout_seconds` | `int` | `600` |
| `LOCALDATA_QUERY_BLOB_HANDLING` | `query.blob_handling` | `str` | `exclude` |
| `LOCALDATA_QUERY_BLOB_MAX_SIZE_MB` | `query.blob_max_size_mb` | `int` | `5` |
| `LOCALDATA_QUERY_PREFLIGHT_DEFAULT` | `query.preflight_default` | `bool` | `false` |

### Connections

| Variable | Maps to | Type | Default |
|----------|---------|------|---------|
| `LOCALDATA_CONNECTIONS_MAX_CONCURRENT` | `connections.max_concurrent` | `int` | `10` |
| `LOCALDATA_CONNECTIONS_TIMEOUT` | `connections.timeout_seconds` | `int` | `30` |

### Security

| Variable | Maps to | Type | Default |
|----------|---------|------|---------|
| `LOCALDATA_SECURITY_MAX_QUERY_LENGTH` | `security.max_query_length` | `int` | `10000` |

## Configuration reference

### `databases`

Each key under `databases` defines a named data source. Required fields: `type` and `connection_string`.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `type` | `str` | *(required)* | One of: `sqlite`, `postgresql`, `mysql`, `duckdb`, `mssql`, `oracle`, `redis`, `elasticsearch`, `mongodb`, `influxdb`, `neo4j`, `couchdb`, `csv`, `json`, `yaml`, `toml`, `excel`, `ods`, `numbers`, `xml`, `ini`, `tsv`, `parquet`, `feather`, `arrow`, `hdf5` |
| `connection_string` | `str` | *(required)* | Connection URI or file path |
| `sheet_name` | `str` | `null` | Sheet or dataset name (Excel, ODS, Numbers, HDF5) |
| `enabled` | `bool` | `true` | Whether this source is active |
| `max_connections` | `int` | `10` | Maximum pool size |
| `connection_timeout` | `int` | `30` | Connection timeout in seconds |
| `query_timeout` | `int` | `300` | Query timeout in seconds |
| `tags` | `list[str]` | `[]` | Arbitrary tags for organizing sources |
| `metadata` | `dict` | `{}` | Additional key-value metadata |

### `staging`

Controls temporary staging databases used during file processing.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `max_concurrent` | `int` | `10` | Maximum concurrent staging databases (1-100) |
| `max_size_mb` | `int` | `2048` | Maximum size per staging database in MB |
| `max_total_mb` | `int` | `10240` | Total disk budget for all staging databases |
| `timeout_minutes` | `int` | `30` | Idle timeout before automatic cleanup |
| `eviction_policy` | `str` | `lru` | Eviction strategy: `lru` or `oldest` |

### `memory`

Controls memory budgeting for query operations.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `max_budget_mb` | `int` | `512` | Hard memory budget cap in MB |
| `budget_percent` | `int` | `10` | Percentage of system RAM to use (1-100) |
| `low_memory_threshold_gb` | `float` | `1.0` | Free memory threshold that triggers conservation mode |

### `query`

Controls query execution behavior.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `default_chunk_size` | `int` | `100` | Number of rows per streamed chunk |
| `buffer_timeout_seconds` | `int` | `600` | How long buffered results are kept (seconds) |
| `blob_handling` | `str` | `exclude` | BLOB column handling: `exclude`, `include`, or `placeholder` |
| `blob_max_size_mb` | `int` | `5` | Maximum BLOB size to inline when handling is `include` |
| `preflight_default` | `bool` | `false` | Run query analysis before execution by default |

### `connections`

Controls global connection limits.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `max_concurrent` | `int` | `10` | Maximum concurrent database connections |
| `timeout_seconds` | `int` | `30` | Default connection timeout in seconds |

### `security`

Controls query and path security.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `allowed_paths` | `list[str]` | `["."]` | Directories the server can access |
| `max_query_length` | `int` | `10000` | Maximum allowed SQL query length in characters |
| `blocked_keywords` | `list[str]` | `[]` | SQL keywords to reject (e.g., `["DROP", "TRUNCATE"]`) |

### `logging`

Controls log output.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `level` | `str` | `info` | Log level: `debug`, `info`, `warning`, `error`, `critical` |
| `format` | `str` | `%(asctime)s - %(name)s - %(levelname)s - %(message)s` | Python log format string |
| `file_path` | `str` | `null` | Path to write log files |
| `max_file_size` | `int` | `10485760` | Maximum log file size in bytes (10 MB) |
| `backup_count` | `int` | `5` | Number of rotated log files to keep |
| `console_output` | `bool` | `true` | Print logs to stdout |

### `performance`

Controls resource limits and tuning.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `memory_limit_mb` | `int` | `2048` | Overall memory limit in MB |
| `query_buffer_timeout` | `int` | `600` | Buffer timeout in seconds |
| `max_concurrent_connections` | `int` | `10` | Maximum concurrent connections |
| `chunk_size` | `int` | `100` | Default chunk size for streaming |
| `enable_query_analysis` | `bool` | `true` | Enable automatic query analysis |
| `auto_cleanup_buffers` | `bool` | `true` | Automatically clean up expired buffers |
| `memory_warning_threshold` | `float` | `0.85` | Memory usage ratio that triggers warnings (0-1) |

## YAML variable substitution

YAML config files support environment variable substitution with the `${VAR}` syntax. You can provide a default value with `${VAR:default}`:

```yaml
databases:
  production:
    type: postgresql
    connection_string: postgresql://${DB_USER:admin}:${DB_PASSWORD}@${DB_HOST:localhost}:5432/${DB_NAME:mydb}
```

If the variable is unset and no default is given, the literal `${VAR}` string is preserved.

## Example configurations

### Minimal

A single SQLite database with all other settings left at defaults:

```yaml
databases:
  main:
    type: sqlite
    connection_string: ./data/app.db
```

### Development

SQLite for local work and a PostgreSQL instance for shared data, with debug logging enabled:

```yaml
databases:
  local:
    type: sqlite
    connection_string: ./dev.db

  shared:
    type: postgresql
    connection_string: postgresql://dev:dev@localhost:5432/devdb

logging:
  level: debug
  console_output: true

performance:
  memory_limit_mb: 1024
  chunk_size: 50
```

### MS SQL Server

Connect to a SQL Server instance. Requires the `enterprise` extra (`pip install localdata-mcp[enterprise]`). Authentication parameters can be passed via the `metadata.auth` field:

```yaml
databases:
  erp:
    type: mssql
    connection_string: mssql+pymssql://${MSSQL_USER}:${MSSQL_PASS}@sqlserver.internal/erp_db
    max_connections: 15
    query_timeout: 120

  erp_kerberos:
    type: mssql
    connection_string: mssql+pyodbc://sqlserver.internal/erp_db
    metadata:
      auth:
        method: kerberos
```

Supported `auth.method` values: `password` (default), `trusted`, `azure_ad`, `kerberos`, `certificate`.

### Production

Multiple databases with tuned resource limits:

```yaml
databases:
  analytics:
    type: postgresql
    connection_string: postgresql://${PG_USER}:${PG_PASS}@db.internal:5432/analytics
    max_connections: 20
    query_timeout: 600

  warehouse:
    type: duckdb
    connection_string: /data/warehouse.duckdb

  reports:
    type: excel
    connection_string: /data/monthly_reports.xlsx
    sheet_name: Summary

staging:
  max_concurrent: 20
  max_size_mb: 4096
  max_total_mb: 20480
  timeout_minutes: 60
  eviction_policy: lru

memory:
  max_budget_mb: 1024
  budget_percent: 15
  low_memory_threshold_gb: 2.0

query:
  default_chunk_size: 500
  buffer_timeout_seconds: 1200
  blob_handling: placeholder
  preflight_default: true

connections:
  max_concurrent: 25
  timeout_seconds: 60

security:
  allowed_paths: ["/data", "/reports"]
  max_query_length: 50000
  blocked_keywords: ["DROP", "TRUNCATE", "ALTER"]

logging:
  level: warning
  file_path: /var/log/localdata-mcp/server.log

performance:
  memory_limit_mb: 4096
  memory_warning_threshold: 0.90
```

## Migration guide

If you have a config file at the legacy location (`~/.localdata.yaml`), migrate it to the platform-appropriate path.

### Automatic migration

Run:

```bash
localdata-mcp --migrate-config
```

This will:

1. Copy `~/.localdata.yaml` to the recommended path (e.g., `~/.config/localdata/config.yaml` on Linux)
2. Create a backup at `~/.localdata.yaml.bak`
3. Create any necessary parent directories

If a file already exists at the destination, the command exits with an error. Use `--force` to overwrite:

```bash
localdata-mcp --migrate-config --force
```

### Manual migration

1. Find the recommended path for your OS:
   - **Linux**: `~/.config/localdata/config.yaml` (or `$XDG_CONFIG_HOME/localdata/config.yaml`)
   - **macOS**: `~/.config/localdata/config.yaml` or `~/Library/Application Support/localdata/config.yaml`
   - **Windows**: `%APPDATA%\localdata\config.yaml`

2. Create the directory and copy:

   ```bash
   mkdir -p ~/.config/localdata
   cp ~/.localdata.yaml ~/.config/localdata/config.yaml
   ```

3. Verify the server starts without deprecation warnings.

4. Optionally remove the old file once you have confirmed the new location works.

## CLI flags

| Flag | Description |
|------|-------------|
| `--config PATH`, `-c PATH` | Use a specific config file (takes highest precedence) |
| `--version`, `-V` | Print the server version and exit |
| `--migrate-config` | Migrate `~/.localdata.yaml` to the recommended platform path |
| `--force` | Force overwrite if the destination file already exists during migration |
