# LocalData MCP Data Sources Reference

Complete reference for connecting to all supported data sources with connection examples and best practices.

## Table of Contents

1. [SQL Databases](#sql-databases-6)
2. [NoSQL Databases](#nosql-databases-6)
3. [File Formats](#file-formats-8)
4. [Spreadsheets](#spreadsheets-3)
5. [Analytical Formats](#analytical-formats-4)
6. [Graph Formats](#graph-formats-4)
7. [RDF & Semantic Web](#rdf--semantic-web-3)
8. [SPARQL Endpoints](#sparql-endpoints-1)
9. [Authentication & Security](#authentication--security)
10. [Connection Best Practices](#connection-best-practices)

---

## SQL Databases (6)

### SQLite

File-based SQL database, ideal for local development and embedded use.

**Connection String Format:**
```
sqlite:///path/to/database.db
```

**MCP Tool Call:**
```python
connect_database("local_db", "sqlite", "sqlite:///data/mydata.db")
```

**Examples:**

```python
# In-memory database (for testing)
connect_database("test", "sqlite", "sqlite:///:memory:")

# Local file
connect_database("myapp", "sqlite", "sqlite:////var/data/app.db")

# Relative path
connect_database("project", "sqlite", "sqlite:///./data.db")
```

**Special Notes:**
- Supports full SQL: SELECT, INSERT, UPDATE, DELETE
- File locking may occur with concurrent access
- Best for < 1GB databases
- No authentication required

---

### PostgreSQL

Enterprise-grade relational database with advanced features.

**Connection String Format:**
```
postgresql://user:password@host:port/database
postgresql+psycopg2://user:password@host:port/database
```

**MCP Tool Call:**
```python
connect_database("prod_db", "postgresql", "postgresql://user:password@localhost:5432/production")
```

**Examples:**

```python
# Standard connection
connect_database("analytics", "postgresql", "postgresql://analyst:secret@db.company.com:5432/analytics")

# With SSL (recommended for remote)
connect_database("secure", "postgresql", "postgresql://user:pass@remote.db:5432/db?sslmode=require")

# Connection pooling
connect_database("pool", "postgresql", "postgresql://user:pass@localhost/db?poolclass=StaticPool&pool_size=10")
```

**Authentication:**
```python
# Standard username/password (in connection string)
connect_database("mydb", "postgresql", "postgresql://user:password@host/db")

# Environment variables
# Set: PGUSER, PGPASSWORD, PGHOST, PGPORT, PGDATABASE
```

**Special Notes:**
- Full SQL support with advanced features (CTEs, window functions, JSON operators)
- Excellent performance for analytical workloads
- Supports array types, JSON/JSONB, full-text search
- SSL/TLS encryption available

---

### MySQL

Widely-used open-source relational database.

**Connection String Format:**
```
mysql+pymysql://user:password@host:port/database
mysql+mysqldb://user:password@host:port/database
```

**MCP Tool Call:**
```python
connect_database("app_db", "mysql", "mysql+pymysql://user:password@localhost:3306/appdb")
```

**Examples:**

```python
# Basic connection
connect_database("sales", "mysql", "mysql+pymysql://user:pass@localhost:3306/sales_db")

# Remote with SSL
connect_database("remote", "mysql", "mysql+pymysql://user:pass@remote.host:3306/db?ssl_ca=/path/to/ca.pem")

# MariaDB (MySQL-compatible)
connect_database("mariadb_server", "mysql", "mysql+pymysql://user:pass@mariadb.server:3306/mydb")
```

**Special Notes:**
- Compatible with MySQL and MariaDB
- Good for web applications
- Text search and JSON support available
- Connection timeout defaults to 30 seconds

---

### DuckDB

Fast analytical SQL engine, great for interactive analysis.

**Connection String Format:**
```
duckdb:///path/to/database.db
duckdb:///:memory:
```

**MCP Tool Call:**
```python
connect_database("analytics", "duckdb", "duckdb:///data/analytics.db")
```

**Examples:**

```python
# In-memory for fast analysis
connect_database("temp", "duckdb", "duckdb:///:memory:")

# Persistent file
connect_database("warehouse", "duckdb", "duckdb:////data/warehouse.db")

# Read Parquet directly
connect_database("parquet_src", "duckdb", "duckdb:///data/")
```

**Special Notes:**
- Optimized for OLAP (analytical) queries
- Excellent performance on compressed data
- Direct Parquet, CSV, JSON file support
- Can query files without loading into database
- No authentication required

---

### Oracle Database

Enterprise database system with advanced features.

**Connection String Format:**
```
oracle+cx_oracle://user:password@host:port/service_name
oracle+oracledb://user:password@host:port/service_name
```

**MCP Tool Call:**
```python
connect_database("enterprise", "oracle", "oracle+cx_oracle://user:password@oracle.server:1521/ORCL")
```

**Examples:**

```python
# Basic connection
connect_database("prod", "oracle", "oracle+cx_oracle://admin:secret@orahost:1521/PROD")

# TNS alias (from tnsnames.ora)
connect_database("tns_db", "oracle", "oracle+cx_oracle://user:password@")

# Wallet authentication (PKI)
connect_database("secure", "oracle", "oracle+cx_oracle://user@oracle.server:1521/ORCL", 
    auth='{"method": "wallet", "wallet_path": "/opt/oracle/wallet"}')
```

**Authentication:**

```python
# Username/password (in connection string)
connect_database("db", "oracle", "oracle+cx_oracle://user:pass@host:1521/SERVICE")

# Kerberos (requires LDAP configured)
connect_database("kerberos_db", "oracle", "oracle+cx_oracle://host:1521/SERVICE",
    auth='{"method": "kerberos"}')

# Wallet (PKI certificates)
connect_database("wallet_db", "oracle", "oracle+cx_oracle://user@host:1521/SERVICE",
    auth='{"method": "wallet", "wallet_path": "/opt/oracle/wallet", "wallet_password": "secret"}')
```

**Special Notes:**
- Enterprise features: partitioning, compression, advanced security
- Requires Oracle client libraries installed
- Large result sets benefit from memory-aware streaming
- Wallet authentication ideal for production

---

### MS SQL Server

Microsoft's enterprise database platform.

**Connection String Format:**
```
mssql+pyodbc://user:password@host:port/database?driver=ODBC+Driver+17+for+SQL+Server
mssql+pymssql://user:password@host:port/database
```

**MCP Tool Call:**
```python
connect_database("mssql_db", "mssql", "mssql+pyodbc://user:password@sqlserver:1433/mydb?driver=ODBC+Driver+17+for+SQL+Server")
```

**Examples:**

```python
# With ODBC
connect_database("reporting", "mssql", "mssql+pyodbc://reporter:pass@mssql.company.com:1433/Reports?driver=ODBC+Driver+17+for+SQL+Server")

# Azure SQL Database
connect_database("azure", "mssql", "mssql+pyodbc://user@server:pass@server.database.windows.net:1433/db?driver=ODBC+Driver+17+for+SQL+Server")

# Named instance
connect_database("instance", "mssql", "mssql+pyodbc://user:pass@host\\INSTANCE:1433/db?driver=ODBC+Driver+17+for+SQL+Server")
```

**Special Notes:**
- Windows and Azure SQL variants supported
- Requires ODBC driver installation
- T-SQL extensions available
- Active Directory authentication supported on Windows

---

## NoSQL Databases (6)

### MongoDB

Document-oriented NoSQL database.

**Connection String Format:**
```
mongodb://user:password@host:port/database
mongodb+srv://user:password@cluster.mongodb.net/database
```

**MCP Tool Call:**
```python
connect_database("mongo_db", "mongodb", "mongodb://user:password@localhost:27017/mydb")
```

**Examples:**

```python
# Local instance
connect_database("local_mongo", "mongodb", "mongodb://localhost:27017/test")

# MongoDB Atlas (cloud)
connect_database("atlas", "mongodb", "mongodb+srv://user:password@cluster0.abc123.mongodb.net/dbname")

# With authentication
connect_database("auth_db", "mongodb", "mongodb://user:password@mongo.example.com:27017/mydb?authSource=admin")
```

**Special Notes:**
- Flexible schema with BSON documents
- Collections accessible as tables through LocalData
- Complex nested documents automatically normalized
- Supports aggregation pipeline queries

---

### Redis

In-memory data store for caching and sessions.

**Connection String Format:**
```
redis://[:password@]host:port/db
rediss://[:password@]host:port/db
```

**MCP Tool Call:**
```python
connect_database("cache", "redis", "redis://localhost:6379/0")
```

**Examples:**

```python
# Default instance
connect_database("cache", "redis", "redis://localhost:6379")

# With authentication
connect_database("secure_cache", "redis", "redis://:mypassword@redis.company.com:6379/0")

# SSL/TLS
connect_database("ssl_redis", "redis", "rediss://user:password@secure.redis.io:6380/0")
```

**Special Notes:**
- Key-value store, not relational
- Excellent for caching and real-time analytics
- TTL support for expiring keys
- Pub/sub messaging available

---

### Elasticsearch

Search and analytics engine for full-text search and logging.

**Connection String Format:**
```
elasticsearch://host:port
elasticsearch+https://user:password@host:port
```

**MCP Tool Call:**
```python
connect_database("search_index", "elasticsearch", "elasticsearch://localhost:9200")
```

**Examples:**

```python
# Local cluster
connect_database("local_es", "elasticsearch", "elasticsearch://localhost:9200")

# With authentication
connect_database("cloud_es", "elasticsearch", "elasticsearch+https://user:password@elastic.company.com:9200")

# Multiple nodes (load balanced)
connect_database("cluster", "elasticsearch", "elasticsearch://node1:9200,node2:9200,node3:9200")
```

**Special Notes:**
- Documents organized in indices (like tables)
- Full-text search with ranking and scoring
- Aggregations for analytics (facets, histograms, etc.)
- Real-time indexing available

---

### InfluxDB

Time-series database optimized for metrics and events.

**Connection String Format:**
```
influxdb://host:port/database
influxdb://user:password@host:port/database
```

**MCP Tool Call:**
```python
connect_database("metrics", "influxdb", "influxdb://localhost:8086/mymetrics")
```

**Examples:**

```python
# InfluxDB 1.x
connect_database("tsdb", "influxdb", "influxdb://user:password@influx.server:8086/metrics")

# Local development
connect_database("local_ts", "influxdb", "influxdb://localhost:8086/test")

# InfluxDB Cloud
connect_database("cloud", "influxdb", "influxdb://token@us-west-2-1.aws.cloud2.influxdata.com:443/org")
```

**Special Notes:**
- Schema-less time series storage
- Tags (indexed) and fields (not indexed)
- Excellent for monitoring and sensor data
- Retention policies for data aging

---

### Neo4j

Graph database for connected data.

**Connection String Format:**
```
neo4j://user:password@host:port
neo4j+s://user:password@host:port
neo4j+ssc://user:password@host:port
```

**MCP Tool Call:**
```python
connect_database("graph_db", "neo4j", "neo4j://user:password@localhost:7687")
```

**Examples:**

```python
# Local instance
connect_database("local_graph", "neo4j", "neo4j://user:password@localhost:7687")

# Aura (cloud)
connect_database("aura", "neo4j", "neo4j+s://user:password@abcd1234.databases.neo4j.io:7687")

# Cluster
connect_database("cluster", "neo4j", "neo4j://user:password@neo4j-lb.company.com:7687")
```

**Special Notes:**
- Property graph with nodes, relationships, properties
- Cypher query language for graph traversal
- Excellent for social networks, recommendations, hierarchies
- ACID transactions supported

---

### CouchDB

Document store with HTTP API and replication.

**Connection String Format:**
```
couchdb://user:password@host:port
```

**MCP Tool Call:**
```python
connect_database("doc_store", "couchdb", "couchdb://user:password@localhost:5984")
```

**Examples:**

```python
# Local Couch
connect_database("local", "couchdb", "couchdb://admin:password@localhost:5984")

# Cloudant (IBM's hosted Couch)
connect_database("cloudant", "couchdb", "couchdb://user:password@account.cloudant.com")
```

**Special Notes:**
- JSON documents with flexible schema
- Built-in replication and sync
- View-based indexing with MapReduce
- REST API for HTTP access

---

## File Formats (8)

### CSV (Comma-Separated Values)

Tabular text format, widely supported and human-readable.

**Connection String:** Path to CSV file

**MCP Tool Call:**
```python
connect_database("data", "csv", "/path/to/file.csv")
```

**Examples:**

```python
# Local file
connect_database("sales", "csv", "/data/sales_2024.csv")

# With options (in URL)
connect_database("quoted", "csv", "/data/file.csv?delimiter=;")

# Remote HTTP
connect_database("remote", "csv", "https://example.com/data.csv")
```

**Special Notes:**
- Auto-detects delimiter (comma, semicolon, tab)
- Header row inferred from first line
- No type inference; all columns start as text
- Large files automatically streamed

---

### TSV (Tab-Separated Values)

Tab-delimited variant of CSV, common in bioinformatics.

**Connection String:** Path to TSV file

**MCP Tool Call:**
```python
connect_database("genes", "tsv", "/path/to/data.tsv")
```

**Examples:**

```python
# Bioinformatics data
connect_database("variants", "tsv", "/genomics/variants.tsv")

# Standard TSV
connect_database("tab_data", "tsv", "/data/file.txt")
```

**Special Notes:**
- Tab delimiter enforced
- Same behavior as CSV otherwise
- Common in Unix text processing

---

### JSON (JavaScript Object Notation)

Hierarchical text format, ideal for nested data.

**Connection String:** Path to JSON file

**MCP Tool Call:**
```python
connect_database("config", "json", "/path/to/file.json")
```

**Examples:**

```python
# Object with nested structure
connect_database("nested", "json", "/data/config.json")

# Array of objects
connect_database("records", "json", "/data/items.json")

# Single large object (normalized to rows)
connect_database("hierarchy", "json", "/data/taxonomy.json")
```

**Special Notes:**
- Supports hierarchical navigation with tree tools
- Deep nesting automatically normalized to rows
- Good for configuration files and API responses
- Unicode and special characters preserved

---

### JSONL (JSON Lines)

One JSON object per line, ideal for streaming and large files.

**Connection String:** Path to JSONL file

**MCP Tool Call:**
```python
connect_database("logs", "jsonl", "/path/to/file.jsonl")
```

**Examples:**

```python
# Log file with JSON events
connect_database("events", "jsonl", "/logs/application.jsonl")

# Newline-delimited objects
connect_database("stream", "jsonl", "/data/stream.ndjson")
```

**Special Notes:**
- One complete JSON object per line
- Excellent for append-only logs
- Efficient streaming and chunking
- Each line parsed independently

---

### YAML (YAML Ain't Markup Language)

Human-friendly structured data format.

**Connection String:** Path to YAML file

**MCP Tool Call:**
```python
connect_database("manifest", "yaml", "/path/to/file.yaml")
```

**Examples:**

```python
# Kubernetes manifest
connect_database("k8s", "yaml", "/config/deployment.yaml")

# Configuration file
connect_database("app_config", "yaml", "/etc/config.yml")

# Multi-document YAML
connect_database("multi", "yaml", "/data/documents.yaml")
```

**Special Notes:**
- Supports hierarchical tree navigation
- Human-readable with indentation-based structure
- Type hints with tags (!!int, !!str, etc.)
- Comments preserved as metadata

---

### TOML (Tom's Obvious, Minimal Language)

Config file format designed for clarity.

**Connection String:** Path to TOML file

**MCP Tool Call:**
```python
connect_database("cfg", "toml", "/path/to/config.toml")
```

**Examples:**

```python
# Application config
connect_database("settings", "toml", "/etc/app/settings.toml")

# Cargo.toml (Rust projects)
connect_database("rust_meta", "toml", "/project/Cargo.toml")
```

**Special Notes:**
- Strict schema with clear semantics
- Tables and array-of-tables support
- Good for application configuration
- Comments supported

---

### XML (eXtensible Markup Language)

Hierarchical markup language with schema support.

**Connection String:** Path to XML file

**MCP Tool Call:**
```python
connect_database("doc", "xml", "/path/to/file.xml")
```

**Examples:**

```python
# SVG document
connect_database("diagram", "xml", "/images/drawing.svg")

# RSS feed
connect_database("feed", "xml", "/feeds/news.xml")

# SOAP response
connect_database("soap", "xml", "/responses/service.xml")
```

**Special Notes:**
- Hierarchical tree navigation available
- Attributes and text content both accessible
- Schema (XSD) validation available
- Namespaces preserved

---

### INI (Initialization File Format)

Simple key-value format for configuration.

**Connection String:** Path to INI file

**MCP Tool Call:**
```python
connect_database("ini_cfg", "ini", "/path/to/file.ini")
```

**Examples:**

```python
# Windows configuration
connect_database("windows_cfg", "ini", "C:\\Program Files\\app\\config.ini")

# Unix config
connect_database("unix_cfg", "ini", "/etc/application/settings.ini")
```

**Special Notes:**
- Sections group related key-value pairs
- Comments with semicolon or hash
- Simple tree structure
- Flat sections (no nesting)

---

## Spreadsheets (3)

### Excel (XLSX, XLS)

Spreadsheet format from Microsoft Office.

**Connection String:** Path to Excel file

**MCP Tool Call:**
```python
connect_database("sheet", "excel", "/path/to/file.xlsx")
```

**MCP Tool Call with Sheet Selection:**
```python
connect_database("sheet", "excel", "/path/to/file.xlsx", sheet_name="Sales")
```

**Examples:**

```python
# All sheets
connect_database("workbook", "excel", "/data/report.xlsx")

# Specific sheet
connect_database("sales_data", "excel", "/data/financial.xlsx", sheet_name="Monthly Sales")

# Multiple connections to different sheets
connect_database("q1_sales", "excel", "/data/sales.xlsx", sheet_name="Q1")
connect_database("q2_sales", "excel", "/data/sales.xlsx", sheet_name="Q2")
```

**Special Notes:**
- Auto-detects XLSX (modern) vs XLS (legacy)
- Merged cells automatically expanded
- Formulas converted to values
- Empty rows and columns handled gracefully

---

### ODS (OpenDocument Spreadsheet)

Open standard spreadsheet from LibreOffice/OpenOffice.

**Connection String:** Path to ODS file

**MCP Tool Call:**
```python
connect_database("ods_sheet", "ods", "/path/to/file.ods", sheet_name="Data")
```

**Examples:**

```python
# Default sheet
connect_database("calc", "ods", "/data/spreadsheet.ods")

# Named sheet
connect_database("results", "ods", "/data/results.ods", sheet_name="Analysis")
```

**Special Notes:**
- Open standard format, fully supported
- Same behavior as Excel
- Good for cross-platform compatibility

---

### Numbers (NUMBERS)

Spreadsheet format from Apple iWork.

**Connection String:** Path to Numbers file

**MCP Tool Call:**
```python
connect_database("numbers_sheet", "numbers", "/path/to/file.numbers", sheet_name="Sheet 1")
```

**Examples:**

```python
# Default sheet
connect_database("apple", "numbers", "/Users/username/Documents/data.numbers")

# Named sheet
connect_database("monthly", "numbers", "/data/reports.numbers", sheet_name="January")
```

**Special Notes:**
- macOS/iOS native format
- Supports multiple sheets and tables
- Styled cells converted to plain values
- Large files automatically streamed

---

## Analytical Formats (4)

### Parquet

Columnar storage format optimized for analytical queries.

**Connection String:** Path to Parquet file or directory

**MCP Tool Call:**
```python
connect_database("analytics", "parquet", "/path/to/file.parquet")
```

**Examples:**

```python
# Single file
connect_database("sales_parquet", "parquet", "/data/sales.parquet")

# Directory with multiple Parquet files
connect_database("dataset", "parquet", "/warehouse/data/")

# Partitioned directory
connect_database("partitioned", "parquet", "/data/year=2024/month=01/")
```

**Special Notes:**
- Highly compressed columnar format
- Excellent for large analytical datasets
- Schema preserved (data types, null info)
- Partition pruning for efficient querying
- Industry standard for data warehouses

---

### Feather

Fast serialization format for data frames.

**Connection String:** Path to Feather file

**MCP Tool Call:**
```python
connect_database("feather_data", "feather", "/path/to/file.feather")
```

**Examples:**

```python
# Arrow/Feather V2 format
connect_database("fast", "feather", "/data/cache.feather")
```

**Special Notes:**
- Fast read/write with minimal parsing
- Language-agnostic (Python, R, etc.)
- Good for intermediate storage in pipelines
- Preserves data types exactly

---

### Arrow

Apache Arrow columnar in-memory format.

**Connection String:** Path to Arrow file

**MCP Tool Call:**
```python
connect_database("arrow_data", "arrow", "/path/to/file.arrow")
```

**Examples:**

```python
# Arrow IPC format
connect_database("ipc", "arrow", "/data/buffer.arrow")
```

**Special Notes:**
- Industry standard columnar format
- Zero-copy data access
- GPU acceleration possible
- Excellent for high-performance workflows

---

### HDF5 (Hierarchical Data Format)

Scientific data format for complex hierarchical data.

**Connection String:** Path to HDF5 file

**MCP Tool Call:**
```python
connect_database("scientific", "hdf5", "/path/to/file.h5")
```

**MCP Tool Call with Dataset Selection:**
```python
connect_database("dataset", "hdf5", "/path/to/file.h5", sheet_name="dataset_name")
```

**Examples:**

```python
# Scientific data
connect_database("genomics", "hdf5", "/data/sequences.h5", sheet_name="DNA")

# Multi-dataset HDF5
connect_database("imaging", "hdf5", "/data/microscopy.h5", sheet_name="channel_1")
```

**Special Notes:**
- Hierarchical groups and datasets
- Supports arrays of arbitrary dimensions
- Metadata and attributes preserved
- Compression built-in
- Common in scientific computing

---

## Graph Formats (4)

### DOT

Graph description language from Graphviz.

**Connection String:** Path to DOT file

**MCP Tool Call:**
```python
connect_database("graph", "dot", "/path/to/file.dot")
```

**Examples:**

```python
# Directed graph
connect_database("digraph", "dot", "/graphs/network.dot")

# Undirected graph
connect_database("network", "dot", "/graphs/undirected.dot")
```

**Special Notes:**
- Human-readable text format
- Supports hierarchical clusters
- Node and edge attributes
- Use `export_graph` to convert to other formats

---

### GML (Graph Modelling Language)

Graph format with attributes and metadata.

**Connection String:** Path to GML file

**MCP Tool Call:**
```python
connect_database("gml_graph", "gml", "/path/to/file.gml")
```

**Examples:**

```python
# Complex network
connect_database("social_network", "gml", "/networks/communities.gml")
```

**Special Notes:**
- Structured attribute support
- Weighted and directed graphs
- Good for social networks
- Preserves node/edge metadata

---

### GraphML

XML-based graph format with rich metadata.

**Connection String:** Path to GraphML file

**MCP Tool Call:**
```python
connect_database("graphml_data", "graphml", "/path/to/file.graphml")
```

**Examples:**

```python
# Network with attributes
connect_database("org_chart", "graphml", "/data/organization.graphml")

# Yfiles format
connect_database("yfiles_export", "graphml", "/graphs/diagram.graphml")
```

**Special Notes:**
- XML-based, extensible
- Widely supported by graph tools
- Rich attribute system
- Can include visualization hints

---

### Mermaid

ASCII-based diagram format.

**Connection String:** Path to Mermaid file

**MCP Tool Call:**
```python
connect_database("diagram", "mermaid", "/path/to/file.mmd")
```

**Examples:**

```python
# Flowchart
connect_database("flow", "mermaid", "/diagrams/workflow.mmd")

# Sequence diagram
connect_database("sequence", "mermaid", "/diagrams/interaction.mmd")
```

**Special Notes:**
- Human-readable ASCII diagrams
- Supports flowcharts, sequences, gantt, state diagrams
- Markdown-compatible
- Good for documentation

---

## RDF & Semantic Web (3)

### Turtle (TTL)

Terse RDF Triple Language, human-readable format.

**Connection String:** Path to Turtle file

**MCP Tool Call:**
```python
connect_database("rdf_data", "turtle", "/path/to/file.ttl")
```

**Examples:**

```python
# Ontology or knowledge graph
connect_database("ontology", "turtle", "/rdf/dbpedia.ttl")

# Linked data
connect_database("linked_data", "turtle", "/data/schema.ttl")
```

**Special Notes:**
- Subject-Predicate-Object triples
- Namespace prefixes for IRIs
- Literal values with language tags and types
- Equivalent to RDF/XML but more readable

---

### N-Triples

Line-based RDF format, one triple per line.

**Connection String:** Path to N-Triples file

**MCP Tool Call:**
```python
connect_database("ntriples_data", "ntriples", "/path/to/file.nt")
```

**Examples:**

```python
# Streaming RDF
connect_database("stream", "ntriples", "/data/large.nt")

# SPARQL query results
connect_database("results", "ntriples", "/sparql/results.nt")
```

**Special Notes:**
- Simpler subset of Turtle
- One triple (S, P, O) per line
- Easier to parse and stream
- Good for large RDF datasets

---

### SPARQL

Query language and endpoint for RDF data.

**Connection String:** SPARQL endpoint URL

**MCP Tool Call:**
```python
connect_database("sparql_endpoint", "sparql", "http://dbpedia.org/sparql")
```

**Examples:**

```python
# DBpedia
connect_database("dbpedia", "sparql", "http://dbpedia.org/sparql")

# Wikidata
connect_database("wikidata", "sparql", "https://query.wikidata.org/sparql")

# Local Fuseki server
connect_database("local_sparql", "sparql", "http://localhost:3030/ds/sparql")
```

**Special Notes:**
- Remote query interface to RDF data
- SPARQL query language (similar to SQL)
- Supports CONSTRUCT, SELECT, ASK, DESCRIBE queries
- Service federation available
- Public endpoints usually rate-limited

---

## Authentication & Security

### Basic Patterns

**Username/Password (in connection string):**
```python
# PostgreSQL
connect_database("db", "postgresql", "postgresql://user:password@host/db")

# MySQL
connect_database("db", "mysql", "mysql+pymysql://user:password@host/db")

# MongoDB
connect_database("db", "mongodb", "mongodb://user:password@host/db")
```

**Environment Variables:**
```python
# Set environment variables first
# export PGUSER=user PGPASSWORD=secret PGHOST=localhost

# Then connection string uses environment
connect_database("db", "postgresql", "postgresql://localhost/db")
```

**OAuth 2.0 / OAuth Credentials:**
```python
# Not directly supported in connection string
# Use service account JSON key for cloud services
```

### Advanced Authentication

**Oracle Wallet (PKI):**
```python
connect_database(
    "prod",
    "oracle",
    "oracle+cx_oracle://user@host:1521/SERVICE",
    auth='{"method": "wallet", "wallet_path": "/opt/oracle/wallet", "wallet_password": "secret"}'
)
```

**Kerberos (Enterprise AD):**
```python
connect_database(
    "enterprise",
    "oracle",
    "oracle+cx_oracle://host:1521/SERVICE",
    auth='{"method": "kerberos"}'
)
```

**AWS IAM (RDS):**
```python
# AWS RDS with IAM auth token
connect_database(
    "rds",
    "postgresql",
    "postgresql://user@rds-instance.abc123.us-east-1.rds.amazonaws.com/db?sslmode=require"
)
```

**Azure AD (SQL Server):**
```python
# Uses Windows authentication or service principal
connect_database(
    "azure",
    "mssql",
    "mssql+pyodbc://SERVER.database.windows.net/DATABASE?driver=ODBC+Driver+17+for+SQL+Server&Authentication=ActiveDirectoryIntegrated"
)
```

---

## Connection Best Practices

### 1. Connection Naming

Use descriptive, consistent names:

```python
# Good: describes purpose and environment
connect_database("prod_analytics", "postgresql", ...)
connect_database("staging_warehouse", "duckdb", ...)
connect_database("cache_layer", "redis", ...)

# Avoid: ambiguous or generic
connect_database("db1", "postgresql", ...)
connect_database("temp", "csv", ...)
```

### 2. Credential Management

Never hardcode credentials:

```python
# Bad: credentials in code
connect_database("mydb", "postgresql", "postgresql://user:password@host/db")

# Good: use environment variables
import os
user = os.getenv("DB_USER")
password = os.getenv("DB_PASSWORD")
host = os.getenv("DB_HOST")
connect_database("mydb", "postgresql", f"postgresql://{user}:{password}@{host}/db")

# Good: use config file (not in version control)
import json
with open("/etc/config.json") as f:
    config = json.load(f)
connect_database("mydb", "postgresql", config["database_url"])
```

### 3. Connection Pooling

For production workloads:

```python
# PostgreSQL with connection pooling
connect_database(
    "prod_pool",
    "postgresql",
    "postgresql://user:pass@host/db?poolclass=QueuePool&pool_size=20&max_overflow=40"
)
```

### 4. SSL/TLS for Remote Connections

Always use encryption for network traffic:

```python
# PostgreSQL with SSL
connect_database(
    "secure_pg",
    "postgresql",
    "postgresql://user:pass@remote.host/db?sslmode=require"
)

# MySQL with SSL certificates
connect_database(
    "secure_mysql",
    "mysql",
    "mysql+pymysql://user:pass@host/db?ssl_ca=/etc/ssl/certs/ca.pem"
)
```

### 5. Connection Lifecycle

Properly manage connections:

```python
# Connect
connect_database("mydb", "postgresql", "postgresql://localhost/db")

# Use queries
execute_query("mydb", "SELECT * FROM users")

# Disconnect when done
disconnect_database("mydb")
```

### 6. Large File Streaming

For large files, use memory-aware settings:

```python
# CSV with chunking
connect_database("large_csv", "csv", "/data/huge_file.csv")
results = execute_query("large_csv", "SELECT * FROM data")

# Then use next_chunk for pagination
next_chunk(query_id, 1, "1000")  # 1000 rows at a time
```

### 7. Remote Data Sources

For HTTP-accessible files:

```python
# Remote CSV
connect_database("remote_data", "csv", "https://data.example.com/export.csv")

# Remote Parquet dataset
connect_database("cloud_data", "parquet", "https://warehouse.example.com/data/")
```

### 8. Testing Connections

Verify connectivity before processing:

```python
# Test connection
databases = list_databases()

# Check if connection exists
if any(db["name"] == "mydb" for db in databases):
    # Use connection
    describe_database("mydb")
```

### 9. Error Handling

Anticipate and handle connection failures:

```python
try:
    connect_database("flaky_db", "postgresql", "postgresql://unreliable.host/db")
    execute_query("flaky_db", "SELECT 1")
except Exception as e:
    # Fallback or retry logic
    print(f"Connection failed: {e}")
```

### 10. Performance Optimization

Match database features to workload:

```python
# OLAP analysis: use DuckDB or analytical database
connect_database("warehouse", "duckdb", "duckdb:///data/warehouse.db")

# Real-time: use PostgreSQL or MongoDB
connect_database("app_db", "postgresql", "postgresql://localhost/app")

# Caching: use Redis or similar
connect_database("cache", "redis", "redis://localhost:6379/0")

# Search: use Elasticsearch
connect_database("search", "elasticsearch", "elasticsearch://localhost:9200")
```

---

## Quick Reference Table

| Data Source | Type | Best For | Auth Type |
|-------------|------|----------|-----------|
| SQLite | SQL | Local dev | None |
| PostgreSQL | SQL | Analytics | User/pass |
| MySQL | SQL | Web apps | User/pass |
| DuckDB | SQL | OLAP | None |
| Oracle | SQL | Enterprise | Wallet/Kerberos |
| MS SQL | SQL | Azure/Windows | AD/User |
| MongoDB | NoSQL | Documents | User/pass |
| Redis | NoSQL | Caching | Password |
| Elasticsearch | Search | Full-text | User/pass |
| InfluxDB | Time-series | Metrics | Token |
| Neo4j | Graph | Networks | User/pass |
| CouchDB | Document | Sync | User/pass |
| CSV/TSV | File | Bulk | None |
| JSON/JSONL | File | Hierarchical | None |
| YAML/TOML | File | Config | None |
| XML | File | Markup | None |
| Excel/ODS | Sheet | Spreadsheets | None |
| Parquet | Analytics | Warehouse | None |
| Feather | Analytics | Pipelines | None |
| HDF5 | Scientific | Research | None |
| DOT/GML/GraphML | Graph | Diagrams | None |
| Turtle/N-Triples | RDF | Semantic web | None |
| SPARQL | RDF | Linked data | Endpoint |

---

**For tool usage examples and workflow composition, see the main tools reference guide.**
