# LocalData MCP

A data science plugin for LLM agents. Connects to 13 database types, 20+ file formats, graph and RDF sources, and provides 8 analytical domains through 71 MCP tools.

LocalData MCP gives LLM agents direct access to local and remote data sources through the [Model Context Protocol](https://modelcontextprotocol.io/). Beyond data connectivity, it provides a full data science toolkit: statistical analysis, time series forecasting, regression modeling, clustering, business intelligence, geospatial analysis, optimization, and sampling methods. Every analytical tool takes a connection name and a SQL query, so an agent moves from raw source to result without a separate load step, and results come back as JSON, carrying an interpretation string wherever a statistic needs one. Chaining tools is the caller's work — a result carries no handle the next tool consumes.

## Key capabilities

- **SQL databases**: PostgreSQL, MySQL, SQLite, DuckDB, Oracle, MS SQL Server
- **NoSQL**: MongoDB, Redis, Elasticsearch, InfluxDB, Neo4j, CouchDB
- **File formats**: CSV, TSV, JSON, YAML, TOML, XML, Excel, Parquet, Feather, HDF5
- **Structured data**: JSON, YAML, TOML trees with full CRUD operations
- **Graphs**: DOT, GML, GraphML, Mermaid with path finding and statistics
- **RDF/SPARQL**: Turtle, N-Triples files and remote SPARQL endpoints
- **Data science**: 8 analytical domains with 30 specialized tools
- **Streaming**: Memory-bounded query execution with adaptive chunk sizing
- **Security**: path restrictions, connection limits, and SELECT-only SQL validation on `execute_query` and `analyze_query_preview`. The analytical tools bypass that gate (issue #25)
- **Claude Code plugin**: 18 skills and 11 agents that drive the tools

```{toctree}
:maxdepth: 2
:caption: Contents

getting-started
tools-reference
configuration
data-sources/index
domains/index
advanced-examples
architecture/index
plugin
error-classification
troubleshooting
docker
changelog
```
