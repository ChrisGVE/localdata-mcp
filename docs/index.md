# LocalData MCP

A comprehensive MCP server for databases, spreadsheets, structured files, and directed graphs.

LocalData MCP gives LLM agents direct access to local data sources through the [Model Context Protocol](https://modelcontextprotocol.io/). Connect to SQL databases, open spreadsheets, navigate JSON/YAML/TOML trees, or load and analyze directed graphs — all through a consistent tool interface.

## Key capabilities

- **SQL databases**: PostgreSQL, MySQL, SQLite, DuckDB — local or remote with authentication
- **Spreadsheets**: Excel, LibreOffice Calc, Apple Numbers with multi-sheet support
- **Flat files**: CSV, TSV, Parquet, Feather, Arrow, HDF5 with large file streaming
- **Structured data**: JSON, YAML, TOML stored as navigable trees with full CRUD
- **Directed graphs**: DOT, GML, GraphML, Mermaid with path finding, statistics, and validation
- **Security**: Path restrictions, SQL injection prevention, connection limits

```{toctree}
:maxdepth: 2
:caption: Contents

getting-started
configuration
data-sources/index
error-classification
changelog
```
