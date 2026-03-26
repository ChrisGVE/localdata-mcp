# SQL Databases

LocalData MCP connects to local and remote SQL databases using the same `connect_database` interface as flat files. Once connected, you query the database with `execute_query` using standard SQL. The connection stays active until you disconnect.

## Supported databases

### SQLite

SQLite databases are local files. No server or authentication is required.

```python
connect_database("app", "sqlite", "./app.db")
describe_database("app")
execute_query("app", "SELECT * FROM users WHERE active = 1 LIMIT 50")
```

To use an in-memory database — useful for temporary computation or testing — pass `:memory:` as the path:

```python
connect_database("tmp", "sqlite", ":memory:")
execute_query("tmp", "CREATE TABLE staging AS SELECT 1 AS id, 'test' AS name")
```

In-memory databases are discarded when the connection is closed.

### PostgreSQL

PostgreSQL connections use a standard connection URI. Include the username, password, host, port, and database name.

```python
connect_database("prod", "postgresql", "postgresql://user:password@host:5432/dbname")
describe_database("prod")
execute_query("prod", "SELECT schemaname, tablename FROM pg_tables WHERE schemaname = 'public'")
```

LocalData MCP maintains a connection pool for PostgreSQL. Multiple queries reuse existing connections rather than opening a new one each time. The pool is sized automatically and released when you call `disconnect_database`.

If your PostgreSQL server uses SSL, append the appropriate parameter to the URI:

```
postgresql://user:password@host:5432/dbname?sslmode=require
```

### MySQL and MariaDB

MySQL and MariaDB use the same connection URI format with the `mysql` scheme. MariaDB is fully compatible and requires no special handling.

```python
connect_database("app", "mysql", "mysql://user:password@host:3306/dbname")
describe_database("app")
execute_query("app", "SELECT table_name, table_rows FROM information_schema.tables WHERE table_schema = 'dbname'")
```

For SSL connections:

```
mysql://user:password@host:3306/dbname?ssl=true
```

### DuckDB

DuckDB is an embedded analytical database well-suited to columnar workloads and large aggregations. It runs in-process, so no server is required.

```python
connect_database("analytics", "duckdb", "./analytics.duckdb")
execute_query("analytics", "SELECT year, SUM(revenue) FROM sales GROUP BY year ORDER BY year")
```

DuckDB supports a wide SQL dialect including window functions, list aggregates, and JSON operations. It can also read Parquet and CSV files directly via SQL if needed.

## Available tools

| Tool | Description |
|---|---|
| `connect_database(name, type, path)` | Open a database connection and register it under `name` |
| `disconnect_database(name)` | Close the connection and release resources |
| `list_databases()` | Show all active connections and their types |
| `describe_database(name)` | List tables and views with row counts |
| `describe_table(name, table)` | Show column names, types, and constraints |
| `find_table(name, pattern)` | Search table names by substring or pattern |
| `execute_query(name, sql)` | Run a SQL SELECT statement |
| `next_chunk(name, buffer_id)` | Retrieve the next chunk from a buffered result |

Results larger than 100 rows are automatically buffered. Use `next_chunk` to retrieve subsequent pages. See the [flat files documentation](flat-files.md) for the full chunking pattern.

## Security considerations

### Credentials in connection strings

Connection URIs for PostgreSQL and MySQL include the username and password in plain text. Keep the following in mind:

- Do not store connection strings in version-controlled files. Use environment variables or a secrets manager and pass them at runtime.
- LocalData MCP logs connection attempts but does not log the full URI. Check that any external logging systems in your environment do the same.
- If a connection string is exposed, rotate the database credentials immediately.

### Network access

Remote database connections go directly from the machine running LocalData MCP to the database host. No traffic passes through any LocalData MCP infrastructure.

- Use encrypted connections (`sslmode=require` for PostgreSQL, `ssl=true` for MySQL) when the network path is not fully trusted.
- Prefer a database user with read-only privileges. LocalData MCP only issues SELECT queries, but the database user's permissions are your last line of defense if a query is constructed from untrusted input.
- If the database is on a private network, ensure the host running LocalData MCP has appropriate network access (VPN, bastion host, or VPC peering as required).

### File path restrictions

SQLite paths are subject to the same directory restriction as flat files: access is limited to the current working directory and its subdirectories. Paths containing `../` are rejected. Remote database URIs are not subject to this restriction.
