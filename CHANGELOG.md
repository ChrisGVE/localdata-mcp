# Changelog

All notable changes to LocalData MCP Server will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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