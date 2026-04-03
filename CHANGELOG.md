# Changelog

All notable changes to LocalData MCP Server will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

<!-- sphinx-start -->

## [2.0.0a1] - 2026-04-03

### Added
- Data science domain modules with sklearn-compatible transformers:
  - Statistical analysis: hypothesis testing, ANOVA, effect sizes
  - Regression modeling: linear, regularized, polynomial with diagnostics
  - Pattern recognition: clustering, anomaly detection, dimensionality reduction
  - Time series: decomposition, ARIMA/ETS forecasting, change detection
  - Business intelligence: RFM segmentation, cohort analysis, A/B testing
  - Optimization: linear programming, constrained optimization, network analysis
  - Sampling & estimation: bootstrap, Bayesian estimation, Monte Carlo simulation
  - Geospatial analysis: spatial joins, routing, clustering (optional geopandas)
- Pipeline framework: sklearn-compatible data science pipeline with streaming support
- 12 new MCP tools for data science operations
- Integration shim layer for domain interoperability

### Changed
- Added scipy, scikit-learn, statsmodels, numpy as core dependencies
- Geospatial imports guarded with try/except for optional geopandas dependency

### Notes
- Alpha release; API may change before 2.0.0 stable
- All 1621 v1 unit tests continue to pass with zero regressions

## [1.7.1] - 2026-04-02

### Added
- Enterprise-scale integration testing with NYC Taxi dataset (100K rows per database)
- 62 enterprise tests across 7 databases: PostgreSQL, MySQL, MSSQL, Oracle, SQLite, MongoDB, Elasticsearch
- Oracle integration tests via oracledb thin mode (no Oracle Client required)
- Automated integration test runner (`scripts/run_integration_tests.py`) with Docker lifecycle management
- Public dataset download script (`scripts/download_test_datasets.py`) for NYC Taxi, Ames Housing, World Bank
- Test fixture file generator (`scripts/generate_test_files.py`) for all supported formats
- InfluxDB, Neo4j, CouchDB integration test suites and Docker services

### Fixed
- Pydantic v1 to v2 migration for configuration validators
- Oracle SQLAlchemy connection string: use `?service_name=` format for thin mode
- MSSQL bulk loader: switch to executemany (2100 parameter limit prevents multi-row INSERT)
- Oracle bulk loader: switch to executemany (multi-row INSERT VALUES not supported)
- Enterprise parquet reader: stream via `iter_batches` to avoid loading 9.5M rows into memory
- Enterprise fixture: read dataset once and pass to all database loaders
- Various test assertion fixes for MCP response shapes, timeout manager mocks, security event fields

### Changed
- README updated: added Oracle, MSSQL, RDF/SPARQL to supported databases; documented 18 missing tools; removed 6 dead tool references; updated test stats and examples
- Version references updated from v1.3.1 to v1.7.0 across API_REFERENCE.md, ARCHITECTURE.md, CONFIGURATION.md, ADVANCED_EXAMPLES.md
- Test count: 1,621 unit tests + 234 integration tests + 62 enterprise tests

## [1.7.0] - 2026-03-28

### Added

#### Regex Search & Transform Tools
- `search_data` MCP tool: search query results for regex patterns with column filtering, case sensitivity, and match position tracking
- `transform_data` MCP tool: apply regex find/replace to query result columns with capture group support and change sampling
- ReDoS (Regular Expression Denial of Service) prevention via `sre_parse` AST analysis detecting nested quantifiers and dangerous patterns
- Pattern complexity scoring with configurable limits (200 char max, 10 groups max)
- Timeout-protected execution (5 second default) via ThreadPoolExecutor

#### Enhanced Graph & Tree Export
- Graph hierarchy export (`style="hierarchy"`): renders DAGs as indented trees with multi-parent annotations (`also child of: X`)
- Graph detailed export (`style="detailed"`): full node property sections with incoming/outgoing edge tables
- Graph adjacency list export (`style="adjacency"`): compact `A -> B [label]` format for token efficiency
- Tree export with path breadcrumbs (`include_path=True`): shows `root > parent > child` navigation context
- Cycle detection with automatic fallback to adjacency list for non-DAG graphs

#### Schema Export
- `export_schema` MCP tool with four output formats:
  - JSON Schema with type mapping, required fields, maxLength, x-primary-key, x-foreign-keys, x-indexes
  - Python dataclass definitions with Optional fields for nullable columns
  - TypeScript interfaces with optional property markers
  - SQL DDL with CREATE TABLE, PRIMARY KEY, FOREIGN KEY, CREATE INDEX statements
- `SchemaIntrospector` class using SQLAlchemy Inspector for cross-database schema extraction

#### Query Audit Log
- `get_query_log` MCP tool: retrieve recent query execution history with database, status, and time filters
- `get_error_log` MCP tool: filtered view of errors and timeouts with structured classification
- In-memory ring buffer (configurable max 1000 entries) with thread-safe deque
- Automatic audit recording integrated into `execute_query` with timing and error capture
- Query hash generation for deduplication statistics

#### Readonly Mode
- `security.readonly: true` configuration option that hard-blocks write-disguised-as-read SQL
- Detects SELECT INTO, CREATE TABLE AS SELECT, COPY TO, INSERT ALL, MERGE INTO, OUTPUT INTO
- Database-specific pattern coverage for Oracle, MS SQL, PostgreSQL, MySQL

## [1.6.0] - 2026-03-28

### Added

#### Enterprise Database Support
- Oracle Database support via `oracledb` driver with password, Oracle Wallet, Kerberos, and certificate authentication
- MS SQL Server support via `pymssql`/`pyodbc` with SQL Auth, Windows Integrated Auth, Azure AD, Kerberos, and certificate authentication
- Enterprise authentication framework (`auth` parameter on `connect_database`) with support matrix per database type
- Oracle and MSSQL error mappers with ORA-XXXXX and Msg/Severity code classification

#### Configuration System Upgrade
- OS-aware configuration paths: XDG on Linux, `~/Library/Application Support` on macOS, `%APPDATA%` on Windows
- Legacy `~/.localdata.yaml` path deprecated with migration support
- New config sections: `staging`, `memory`, `query`, `connections`, `security`, `disk_budget`
- Environment variable overrides for all new config sections (`LOCALDATA_STAGING_*`, `LOCALDATA_MEMORY_*`, etc.)
- CLI flags: `--config`, `--version`, `--migrate-config`, `--force`, `--validate-config`, `--show-config`, `--init-config`
- Default config file creation with `create_default_config()`

#### Structured Error Classification
- `StructuredErrorResponse` with `error_type`, `is_retryable`, `message`, `suggestion` for LLM agents
- Database-specific error mappers for SQLite, PostgreSQL, MySQL, DuckDB, Oracle, MS SQL
- Error mapper registry with pluggable architecture for custom databases
- Structured errors integrated into `execute_query`, `connect_database`, and streaming executor
- Helper functions: `classify_error()`, `is_error_retryable()`, `get_error_suggestion()`

#### Query Execution Intelligence
- Size estimation engine combining column type metadata with EXPLAIN row estimates
- EXPLAIN parsers for SQLite, PostgreSQL, MySQL, Oracle (DBMS_XPLAN), MS SQL (SHOWPLAN_XML)
- Pre-flight query estimation via `preflight=True` parameter on `execute_query`
- Memory-aware execution with `MemoryBudget` auto-calculated from available RAM (10%, max 512MB)
- Aggressive mode when RAM drops below 1GB (5%, max 128MB)
- BLOB column detection with placeholder handling (`[BLOB: size, mime_type]`) and opt-in base64 encoding

#### Staging and Disk Management
- Staging database manager with LRU eviction for large query results
- Disk space monitoring during streaming with configurable thresholds
- Graceful abort with partial results when disk limits are reached
- `include_staging` parameter for `list_databases` to show staging databases
- Cascade cleanup of staging databases on `disconnect_database`

#### Markdown Export
- Markdown table export for query results with alignment and truncation
- Tree/structured data export as heading hierarchy
- Graph export with summary stats, node/edge tables, and Mermaid diagram embedding
- Registered as `markdown`/`md` format in `export_structured` and `export_graph`

#### Documentation and Testing
- Comprehensive configuration reference (`docs/configuration.md`)
- Error classification reference (`docs/error-classification.md`)
- Docker-based integration testing infrastructure (`docker-compose.test.yml`)
- 663 tests across 26 test files

## [1.5.2] - 2026-03-26

### Added
- Validation warnings on graph import and edit operations

## [1.5.1] - 2026-03-26

### Added
- Mermaid flowchart parser support for graph import
- Mermaid export support with round-trip fidelity
- Main README section documenting graph file, Mermaid, and SPARQL support

### Fixed
- Include node and edge metadata in graph exports
- Six graph bugs found during code review
- Mermaid parsing and export edge cases
- File and function size limit compliance for graph modules

## [1.5.0] - 2026-03-24

### Added
- Graph file format support: DOT, GML, GraphML
- RDF format support: Turtle and N-Triples parsing
- GraphStorageManager and RDFStorageManager for structured graph persistence
- SPARQL query endpoint for RDF data
- Graph-specific MCP tools for node, edge, path, and statistics operations
- RDF export and describe_database support for graph connections
- End-to-end integration tests for graph/RDF/SPARQL pipeline
- move_node tool to relocate subtrees in tree storage

### Fixed
- Return dicts from tree tools to prevent double JSON serialization
- Deserialize string values when explicit value_type is provided

## [1.4.2] - 2026-03-23

### Fixed
- Use StaticPool for in-memory SQLite to fix cross-thread access errors
- Clean up tree managers on disconnect to prevent resource leaks

## [1.4.0] - 2026-03-23

### Added
- Tree storage model, manager, and hierarchical data persistence
- Tree parsers for TOML, JSON, and YAML formats
- MCP tool functions for tree navigation and mutation (get_node, set_node, get_children, delete_node)
- Tree export to TOML, JSON, and YAML formats
- Integration of tree storage with connect_database and tool registration
- Data summary returned on connect_database
- Comprehensive tests for tree tools, parsers, and export

### Fixed
- Migrate to fastmcp v3 API for tool registration
- LogContext now accepts arbitrary keyword arguments
- Row-normalise deeply nested TOML/YAML instead of column explosion

## [1.3.1] - 2025-08-31

### Added
- SQL query parser and safety validator
- QueryAnalyzer class with pre-query analysis system
- ConfigManager class with pydantic-based configuration
- StreamingQueryExecutor for memory-bounded streaming pipeline
- QueryTimeoutManager for advanced timeout management
- Streaming file processors integrated into file engine creation
- TokenManager class with DataFrame-based token estimation
- EnhancedConnectionManager with connection pooling and health monitoring
- SecurityManager class with OWASP-aligned security patterns
- Enhanced response metadata and LLM communication protocol with progressive data loading
- Error handling system with custom exceptions, retry mechanisms, and circuit breaker pattern
- Enhanced database manager with integrated error handling and recovery strategies
- Structured logging system with structlog, Prometheus metrics, and JSON logging
- CompatibilityManager for backward compatibility with v1.3.0
- Performance benchmarking suite with CI integration
- Comprehensive documentation: architecture, configuration, API reference, migration guide

### Changed
- Clean tool surface for production release (breaking: removed deprecated tool aliases)
- Major cleanup and reorganization of project structure
- README restructured with table of contents

### Fixed
- Environment variable names in migration script generation
- Boolean column categorization and test data alignment
- TokenManager integration robustness
- DataFrame boolean ambiguity in return statements
- Function abuse attack pattern regex
- Leaked credential detection in documentation examples

## [1.3.0] - 2025-08-30

### Added
- Modern database support: Redis, Elasticsearch, MongoDB, InfluxDB, Neo4j, CouchDB
- DuckDB support for analytical SQL queries
- HDF5 format support for scientific data
- Apple Numbers (.numbers) document support
- Docker containerization with complete docker-compose setup
- Comprehensive test coverage improvements (68% coverage)

### Changed
- Updated documentation to reflect actual test coverage (68% vs previous claims)
- Enhanced README with modern database features and Docker support
- Improved error handling and graceful degradation for optional dependencies

### Fixed
- ODS import detection with proper library imports
- JSON/TOML serialization for complex nested data structures
- INI file parsing with special character handling
- InfluxDB service configuration for proper startup

## [1.2.0] - 2025-08-29

### Added
- Multi-sheet spreadsheet support (Excel .xlsx/.xls, LibreOffice .ods)
- Enhanced file format support: XML, INI, TSV
- Analytical formats: Parquet, Feather, Arrow
- Large file handling with automatic SQLite conversion (100MB+ files)
- Query buffering system for large result sets (100+ rows)
- Advanced security features and input validation

### Changed
- Improved performance for large datasets
- Enhanced error messages and validation
- Better memory management and resource cleanup

### Fixed
- Path traversal security issues
- SQL injection prevention
- Connection limit enforcement

## [1.1.0] - 2025-08-23

### Added
- Enhanced spreadsheet format support
- Improved file type detection
- Better error handling for malformed files

### Changed
- Performance optimizations for file processing
- Updated documentation and examples

## [1.0.3] - 2025-08-23

### Fixed
- Minor bug fixes and stability improvements
- Documentation updates

## [1.0.2] - 2025-08-23

### Fixed
- Package dependencies and compatibility issues
- Installation process improvements

## [1.0.1] - 2025-08-23

### Fixed
- Initial release bug fixes
- Documentation corrections

## [1.0.0] - 2025-08-23

### Added
- Initial release of LocalData MCP Server
- Basic database connectivity (PostgreSQL, MySQL, SQLite)
- File format support (CSV, JSON, YAML)
- Core MCP protocol implementation
- Security features and input validation