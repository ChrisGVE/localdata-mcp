# Changelog

All notable changes to LocalData MCP Server will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

<!-- sphinx-start -->

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