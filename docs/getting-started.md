# Getting Started

## Installation

### Using uv (recommended)

```bash
# Install permanently
uv tool install localdata-mcp

# Update to latest version
uv tool upgrade localdata-mcp

# Or run directly without installing
uvx localdata-mcp
```

> **First install note:** LocalData MCP includes data science libraries
> (scipy, scikit-learn, statsmodels, geopandas, ruptures) that total around
> 200 MB. The first install or first `uvx` run may take a minute or two while
> these are downloaded and cached. Subsequent runs reuse the cache and start
> immediately.
>
> If your LLM client times out waiting for the MCP server to start on the
> first run, reconnect the MCP server from your client's interface or restart
> the LLM application. The dependencies will already be cached and the next
> start will be fast.

### From source

```bash
git clone https://github.com/ChrisGVE/localdata-mcp.git
cd localdata-mcp
uv sync --extra dev
```

The dev tools are declared as a project extra, not a uv dependency group, so `uv sync --dev` will not install them.

### What's included

The base install covers all supported functionality: SQL databases (SQLite, PostgreSQL, MySQL, DuckDB), all spreadsheet formats, flat files (CSV, TSV, Parquet, Feather, Arrow, HDF5), structured data (JSON, YAML, TOML, XML, INI), directed graphs (DOT, GML, GraphML, Mermaid), and the full data science suite (statistical analysis, regression, pattern recognition, time series, geospatial, optimization).

### Additional database drivers

Support for Redis, MongoDB, Elasticsearch, InfluxDB, Neo4j, and CouchDB requires the `modern-databases` extra. These are not installed by default because they require database-specific native drivers.

```bash
# uv tool
uv tool install "localdata-mcp[modern-databases]"

# uvx (note the --from syntax)
uvx --from "localdata-mcp[modern-databases]" localdata-mcp
```

## MCP client configuration

Add LocalData MCP to your MCP client configuration file. The exact location depends on your client:

- **Claude Desktop**: `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows)
- **Claude Code**: `.mcp.json` in your project root

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

If installed with `uvx`:

```json
{
  "mcpServers": {
    "localdata": {
      "command": "uvx",
      "args": ["localdata-mcp"],
      "env": {}
    }
  }
}
```

## First steps

Once configured, your LLM agent has access to all LocalData MCP tools. Here are some typical first interactions:

### Connect to a local SQLite database

```python
connect_database("mydb", "sqlite", "./data.sqlite")
describe_database("mydb")
execute_query("mydb", "SELECT * FROM users LIMIT 10")
```

### Open a CSV file

```python
connect_database("sales", "csv", "./sales_data.csv")
execute_query("sales", "SELECT product, SUM(amount) FROM data_table GROUP BY product")
```

A single-table file is loaded into a table named `data_table` regardless of the
file name or the connection name. `describe_database("sales")` lists what a
connection actually exposes.

### Navigate a JSON config file

```python
connect_database("cfg", "json", "./config.json")
get_node("cfg")
get_children("cfg", "server")
get_value("cfg", "server", "port")
```

### Load and analyze a graph

```python
connect_database("g", "graphml", "./knowledge_graph.graphml")
get_graph_stats("g")
find_path("g", "node_a", "node_b")
export_graph("g", "mermaid")
```

## Security

File access is confined to the paths in `security.allowed_paths`, which defaults to `["."]` — the process working directory and its subdirectories. Parent-directory traversal (`../`) out of that tree is rejected. Set `LOCALDATA_SECURITY_RESTRICT_PATHS=false` to lift the restriction.

Every SQL statement passes a whitelist validator before it reaches a driver: only `SELECT` and `WITH` are accepted, and `INSERT`, `UPDATE`, `DELETE`, `DROP`, `CREATE`, `ALTER`, `ATTACH`, `PRAGMA`, `EXEC` and 16 other operations are rejected with a `SQLSecurityError`. Setting `security.readonly: true` additionally blocks writes disguised as reads — `SELECT ... INTO`, `CREATE TABLE ... AS SELECT`, `COPY ... TO`, `MERGE INTO`.

Note what this is not: your agent writes the SQL, so the server cannot parameterize it for you. The validator constrains the *operation*, not the values inside it. Treat any query built from untrusted input as your responsibility.

Concurrent connections are capped at 10 (`LOCALDATA_CONNECTIONS_MAX_CONCURRENT`).

## Next steps

- [Flat files and databases](data-sources/flat-files.md) — CSV, Excel, Parquet, and SQL databases
- [Databases](data-sources/databases.md) — SQLite, PostgreSQL, MySQL, DuckDB with remote auth
- [Structured data](data-sources/structured-data.md) — JSON, YAML, TOML tree storage
- [Directed graphs](data-sources/directed-graphs.md) — DOT, GML, GraphML, Mermaid
