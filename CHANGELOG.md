# Changelog

All notable changes to LocalData MCP Server will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

<!-- sphinx-start -->

## [2.1.0] - unreleased

### Upgrading from 2.0.0

Read this section even if you use only the MCP server. Two components were
renamed, three groups of tools that never worked now do, and the geospatial
statistics in 2.0.0 returned wrong numbers.

**Renames and moves.** Both renames break callers by name, and nothing warns
you — an old name simply resolves to nothing.

1. **Rename `graph-explore` to `graph-data-explore`** wherever you invoke it:
   `/graph-explore` becomes `/graph-data-explore`.
2. **Rename `graph-analyst` to `graph-data-analyst`** in any request or
   configuration that names the agent.
3. **Update any pinned path** of the form `skills/<name>/SKILL.md`. All eighteen
   skills now sit one level deeper, under a group directory:
   `skills/<group>/<name>/SKILL.md`. The eight that existed in 2.0.0 moved to
   `skills/exploration/explore-data/`, `skills/statistical/ab-test/`,
   `skills/statistical/analyze-correlations/`,
   `skills/modeling/cluster-analysis/`, `skills/modeling/forecast/`,
   `skills/modeling/regression/`, `skills/graph-data/graph-data-explore/`, and
   `skills/workflow/data-pipeline/`.
4. **Reload the plugin** so the moved directories are rescanned. Confirm with
   `claude --plugin-dir <path> plugin details localdata-mcp`, which should
   report 18 skills and 11 agents.

**Re-run any geospatial statistic computed on 2.0.0.** Moran's I, Geary's C and
Getis-Ord Gi* all returned plausible, wrong numbers — inflated z-scores, a
Geary's C sixty times too small, and a `p_value` of `None` on every Geary call
ever made. The corrected formulas are described under Fixed below. Any
conclusion drawn from those statistics needs recomputing, not adjusting.

**Recheck any YAML configuration you wrote for 2.0.0.** Settings the server
parsed and then ignored now take effect: `query.chunk_size`,
`query.buffer_timeout`, `connections.max_concurrent` and nineteen of the
twenty-five logging fields, `enable_metrics` among them. Defaults are unchanged
when nothing is configured, so an installation that configured nothing behaves
as before — but a file that set one of those keys will now be obeyed for the
first time.

**Nothing was removed from the tool surface.** It grew from 53 tools to 71; the
53 that existed in 2.0.0 keep their names, parameters and return shapes. A
client that names tools explicitly needs no edit for the new ones to appear.

### Added

- Eighteen MCP tools, taking the surface from 53 to 71. Seventy are registered
  by `DatabaseManager`; the seventy-first, `get_metrics`, is registered at import
  when metrics collection is enabled, which is the default.
  - **Sampling and estimation (4):** `generate_sample`, `bootstrap_statistic`,
    `monte_carlo_simulate`, `bayesian_estimate`.
  - **Optimization (4):** `solve_linear_program`, `optimize_constrained`,
    `analyze_network`, `solve_assignment_problem`. These four read a whole table
    rather than a query, since a solver needs the full constraint set.
  - **Geospatial (10):** `check_geospatial_capabilities`,
    `analyze_spatial_autocorrelation`, `find_spatial_hotspots`,
    `calculate_spatial_distances`, `optimize_route`, `analyze_accessibility`,
    `generate_service_isochrones`, `perform_spatial_join`,
    `perform_spatial_overlay`, `aggregate_points_in_polygons`. geopandas,
    shapely and pyproj are required; call `check_geospatial_capabilities` to see
    which optional backends (scikit-gstat, rasterio) are present.
- `LOCALDATA_LOGGING_ENABLE_METRICS`, `LOCALDATA_LOGGING_METRICS_PORT` and
  `LOCALDATA_LOGGING_METRICS_ENDPOINT`. `enable_metrics` had no environment
  variable and no working YAML path, so the `get_metrics` tool could not be
  turned off.
- Ten skills: data-quality, find-reference-data, anomaly-detection,
  dimensionality-reduction, geospatial, optimization, hypothesis-test,
  sampling-estimation, process-control, research-pipeline. The plugin now ships 18.
- Five agents: data-researcher, data-scientist, geospatial-analyst,
  operations-analyst, research-analyst. The plugin now ships 11.
- `server.json` describing the package for the official MCP registry, plus the
  `mcp-name` marker in README.
- Docker Hub publish workflow and MCP metadata label on the image.
- Kaggle demo: an RFM replication notebook over the UCI Online Retail dataset,
  with a Google Colab variant.
- `uv.lock`, committed so development environments are reproducible.
- `docs/plugin.md`, documenting every shipped skill and agent.

### Changed

- Configuration that was parsed and discarded now reaches the code that uses it.
  `query.chunk_size`, `query.buffer_timeout` and `connections.max_concurrent`
  had no call sites, so the live paths used hardcoded literals — a
  `Semaphore(10)`, 600s and 3600s buffer expiries, a 1000-row chunk fallback.
  `get_logging_config()` built `LoggingConfig` from a hand-written list of six
  keys and dropped the other nineteen. Where a setting has two homes, the newer
  `query`/`connections` section wins over the older `performance` keys, which
  stay honoured. Every existing default is unchanged when nothing is configured.
- Licence migrated from MIT to Apache 2.0. `LICENSE` carries the Apache 2.0 text,
  `NOTICE` carries the attribution notice Apache 2.0 requires, and `pyproject.toml`
  declares `license = "Apache-2.0"` as a PEP 639 expression.
- **Breaking:** skills are grouped into five domain directories (exploration,
  statistical, modeling, graph-data, workflow). All eight 2.0.0 skills moved, and
  `graph-explore` is now `graph-data-explore`. See Upgrading above.
- **Breaking:** `graph-analyst` is now `graph-data-analyst`. See Upgrading above.
- `statistical-analyst` covers sampling design and estimation alongside hypothesis testing.
- PyPI metadata: added AI and data-science classifiers and cross-platform keywords.
- Version raised to 2.1.0 in `pyproject.toml`, `.claude-plugin/plugin.json`, and `server.json`.

### Removed

- The skill name `graph-explore` and the agent name `graph-analyst`. Both are
  renames, not deletions of function — see Upgrading above for the replacements.
- The flat `skills/<name>/SKILL.md` layout, replaced by
  `skills/<group>/<name>/SKILL.md`.
- Four root documents, each describing a system that no longer matched the code:
  `API_REFERENCE.md`, `ADVANCED_EXAMPLES.md`, `ARCHITECTURE.md` and
  `CONFIGURATION.md`. They are replaced by `docs/tools-reference.md`,
  `docs/advanced-examples.md`, `docs/architecture/index.md` and
  `docs/configuration.md`.
- A duplicated tool section in the configuration documentation, which documented
  two parameters that do not exist: `style` on `export_graph` and `include_path`
  on `export_structured`.

### Fixed

- **`analyze_regression`'s `regularization` parameter did nothing at all.** It
  was forwarded into the modelling pipeline, which selects its estimator from
  `model_type` alone and never reads a `regularization` key, so `l1`, `l2`,
  `elastic_net` and even a misspelled value all returned the identical
  unpenalised fit. Anyone who reached for it to control overfitting on a wide
  feature set received ordinary least squares and no indication of it. The
  penalty now resolves to the estimator it names -- `l1` fits lasso, `l2` fits
  ridge, `elastic_net` fits elastic net -- and refines the expanded basis when
  combined with `model_type="polynomial"`. An unrecognised value, or one that
  contradicts an explicitly chosen `model_type`, now raises with the accepted
  values named.
- **`analyze_rfm` aborted on an ordinary order log.** When every customer places
  the same number of orders, the frequency quartiles collapse onto a single
  value and `pandas.cut` rejects the duplicate bin edges with `ValueError: Bin
  edges must be unique` -- an entirely normal shape for a subscription or
  replenishment business, failing the whole analysis. Duplicate edges are now
  collapsed and the surviving buckets spread back across the same 1-4 scale, and
  a dimension with no spread at all scores every customer the neutral 2 rather
  than branding them all with an extreme. Data with three distinct quartile
  boundaries scores exactly as it did before.
- **`analyze_network` never returned shortest paths.** NetworkX's
  `floyd_warshall_predecessor_and_distance` returns a `(predecessors,
  distances)` tuple, which was passed to `dict()`; that raised, the exception
  was swallowed, and the result carried `{"algorithm": "failed"}` instead of any
  paths. The Floyd-Warshall branch handles every graph of 100 nodes or fewer, so
  the feature was dead for graphs of the size the tool is used on. Distances and
  predecessors are now returned for every pair.
- **`analyze_hypothesis_test` never compared the groups it was given.** With the
  default `test_type="auto"` and a `group_column`, it returned only Shapiro-Wilk
  and Kolmogorov-Smirnov normality tests -- no group comparison at any point.
  This is the call shown as the flagship analytical example in the README, so an
  agent following the documentation received a well-formed result that answered a
  different question and reported no difference where one existed. The automatic
  path now selects from the data's assumptions: Welch's or Student's t-test for
  two normal groups depending on Levene, Mann-Whitney U with a rank-biserial
  effect size when either group is non-normal, one-way ANOVA with eta squared for
  three or more normal homoscedastic groups, Kruskal-Wallis H with epsilon
  squared otherwise. The comparison leads `test_results` and the normality
  profile follows it. Behaviour without a `group_column` is unchanged.
- **`analyze_anova` returned no post-hoc comparisons, ever.** `post_hoc_results`
  was `{}` on every call, including a three-group fixture with F = 27.9 and
  p = 1.2e-10, because the extraction indexed a raw observation array as though it
  were a labelled table and the resulting error was swallowed. Pairwise
  comparisons are now returned with their mean differences, p-values and
  confidence intervals, and the exception handling around them is narrowed so a
  future failure of this kind surfaces instead of returning an empty result.
- **`export_structured(name, "markdown")` returned an empty document.** The call
  produced `{"format": "markdown", "content": "## "}` for every tree, flat or
  nested, while `json`, `yaml` and `toml` returned the full structure. Markdown
  export now renders the tree.
- **Spatial statistics returned wrong numbers.** Anyone who ran them on 2.0.0
  should re-run rather than reinterpret. Four defects, all in the variance terms:
  the S2 term shared by Moran's I and Geary's C summed squared row sums instead
  of squared row-plus-column sums (60 instead of 256.9 for a row-standardised
  8-nearest-neighbour matrix over 60 points); Moran's variance ran roughly ten
  times small, inflating a z-score of 18.9 to 59.9, and now uses the closed-form
  normality variance of Cliff & Ord (1981); Geary's C divided by S0 twice,
  deflating C by a factor of sixty, and its variance came out negative, so its
  `p_value` was `None` on every call ever made; Getis-Ord Gi* used a variance
  with no textbook counterpart and added a self-weight to an already
  row-standardised matrix, and now uses the standard form of Getis & Ord (1992).
  On three well-separated blobs it now labels every high-value point a hot spot
  and every low-value point a cold spot, where before it found neither.
- **All twelve data science tools raised before returning.** The adapter between
  the MCP tools and the domain modules called almost every domain function with
  the wrong signature — column selectors forwarded as keywords the transformers
  do not accept, a DataFrame passed where a positional ndarray was required, and
  renamed parameters (`amount_column`, not `value_column`; `algorithm`, not
  `method`). The MCP signatures themselves were correct throughout.
- **The time series tools could not run at all.** Every analyzer subclassed an
  abstract pipeline base whose five hooks none of them implement, so
  instantiation failed with "Can't instantiate abstract class"; and
  `TimeSeriesResult.to_dict()` listed the base fields by hand, silently dropping
  every field a subclass added.
- **`request_data_chunk`, `request_multiple_chunks` and `get_query_metadata`
  were non-functional from the day they were registered.** The chunk loader was a
  placeholder returning `None`, so every chunk of every query came back as "Chunk
  N not available". Chunk availability was sized from the buffered head rather
  than the whole result, so a 50,000-row result advertised ten chunks covering
  2,170 rows and refused every id past that. `get_query_metadata` serialized its
  response with the stdlib encoder while the quality report carries numpy
  scalars, so it returned "Object of type bool is not JSON serializable" and
  nothing else, always.
- Clustering results keyed per-cluster statistics by numpy integer labels, which
  `json.dumps` rejects as mapping keys regardless of the `default` handler, so the
  tool failed at serialization after the analysis had already succeeded.
- Four geospatial entry points read result attributes their transformers do not
  set, returning `{"error": "Analysis failed"}` while the analysis underneath had
  succeeded.
- The advertised tool count. Six files stated 53; the server registers 71.
- Performance benchmarks workflow was invalid YAML and had never executed. Embedded
  Python scripts written at column 0 terminated the enclosing `run:` block scalar,
  so GitHub could not parse the file and every run completed with zero jobs.
- Performance benchmarks workflow installed dependencies from `requirements.txt` and
  `requirements-dev.txt`, which no longer exist; it now installs from `pyproject.toml`.
  Deprecated actions were updated and the Python matrix aligned with CI.
- Benchmark harness called token estimation with the wrong parameter name.
- Retry-logic test counted sleeps raised on other threads and failed intermittently.
- Plugin manifest and contributing guide still referenced the MIT licence after the
  migration to Apache 2.0. The `License :: OSI Approved :: MIT License` classifier was
  removed, since a PEP 639 `license` expression supersedes it.
- README documented seven graph tools that do not exist (`get_node_graph`,
  `set_node_graph`, `delete_node_graph`, `get_value_graph`, `set_value_graph`,
  `delete_key_graph`, `list_keys_graph`). Graph node operations are served by the tree
  tools, which read `path` as a node ID for graph connections.
- README named four environment variables the server never reads
  (`LOCALDATA_MAX_CONNECTIONS`, `LOCALDATA_CHUNK_SIZE`, `LOCALDATA_BUFFER_TTL`,
  `LOCALDATA_WORKING_DIR`) and gave a chunk-size default of 500 where the code uses 100.
- README linked `DATABASE_CONNECTIONS.md` and `FAQ.md`, both deleted before 2.0.0.
- `CONTRIBUTING.md`, `TROUBLESHOOTING.md`, and `docs/getting-started.md` told
  contributors to run `uv sync --dev`. The dev tools are a project extra, not a uv
  dependency group, so that command uninstalls pytest, black, and mypy instead of
  installing them. Corrected to `uv sync --extra dev`.
- Four domain guides documented internal Python adapters whose first parameter is
  a SQLAlchemy engine as if they were MCP tools. `docs/domains/` now names the
  tools an MCP client can actually call.

## [2.0.0] - 2026-04-06

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
- 12 new MCP tools for data science operations (52 total)
- Integration shim layer for domain interoperability
- Graph format support: DOT, GML, GraphML, Mermaid
- RDF/SPARQL support: Turtle, N-Triples files and remote SPARQL endpoints
- Memory-aware query execution with 3-path decision flow (in-memory, staging, refinement)
- Graceful abort handler for streaming with partial result return
- Claude Code plugin structure: 8 skills, 6 agents, plugin manifest
- Comprehensive documentation: tools reference, data sources, 8 domain guides

### Changed
- Added scipy, scikit-learn, statsmodels, numpy as core dependencies
- Added networkx, pydot, rdflib, SPARQLWrapper as core dependencies
- Geospatial imports guarded with try/except for optional geopandas dependency
- README rewritten for v2.0 (663 lines to 261 lines)
- Replaced fixed 100-row query threshold with dynamic memory-based threshold
- Refactored oversized modules into sub-packages for maintainability

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