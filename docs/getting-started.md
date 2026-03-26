# Getting Started

## Installation

### Using uv (recommended)

```bash
# Install as a global tool
uv tool install localdata-mcp

# Or run directly without installing
uvx localdata-mcp
```

### Using pip

```bash
pip install localdata-mcp
```

### From source

```bash
git clone https://github.com/ChrisGVE/localdata-mcp.git
cd localdata-mcp
pip install -e .
```

### Optional dependencies

For remote database support (Redis, MongoDB, Elasticsearch, InfluxDB, Neo4j, CouchDB):

```bash
pip install localdata-mcp[modern-databases]
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
execute_query("sales", "SELECT product, SUM(amount) FROM data GROUP BY product")
```

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

LocalData MCP restricts file access to the current working directory and its subdirectories. Parent directory traversal (`../`) is blocked. SQL queries are parameterized to prevent injection. A maximum of 10 concurrent connections are allowed.

## Next steps

- [Flat files and databases](data-sources/flat-files.md) — CSV, Excel, Parquet, and SQL databases
- [Databases](data-sources/databases.md) — SQLite, PostgreSQL, MySQL, DuckDB with remote auth
- [Structured data](data-sources/structured-data.md) — JSON, YAML, TOML tree storage
- [Directed graphs](data-sources/directed-graphs.md) — DOT, GML, GraphML, Mermaid
