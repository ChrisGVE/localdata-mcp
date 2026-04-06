# LocalData MCP

A data science plugin for LLM agents. Connects to 13 database types, 20+ file formats, graph and RDF sources, and provides 8 analytical domains through 52 MCP tools.

LocalData MCP gives LLM agents direct access to local and remote data sources through the [Model Context Protocol](https://modelcontextprotocol.io/). Beyond data connectivity, it provides a full data science toolkit: statistical analysis, time series forecasting, regression modeling, clustering, business intelligence, geospatial analysis, optimization, and sampling methods — all designed for LLM-driven workflows with composition metadata and progressive disclosure.

## Key capabilities

- **SQL databases**: PostgreSQL, MySQL, SQLite, DuckDB, Oracle, MS SQL Server
- **NoSQL**: MongoDB, Redis, Elasticsearch, InfluxDB, Neo4j, CouchDB
- **File formats**: CSV, TSV, JSON, YAML, TOML, XML, Excel, Parquet, Feather, HDF5
- **Structured data**: JSON, YAML, TOML trees with full CRUD operations
- **Graphs**: DOT, GML, GraphML, Mermaid with path finding and statistics
- **RDF/SPARQL**: Turtle, N-Triples files and remote SPARQL endpoints
- **Data science**: 8 analytical domains with 12 specialized tools
- **Streaming**: Memory-bounded query execution with adaptive chunk sizing
- **Security**: Path restrictions, SQL injection prevention, connection limits

```{toctree}
:maxdepth: 2
:caption: Contents

getting-started
tools-reference
configuration
data-sources/index
domains/index
error-classification
changelog
```
