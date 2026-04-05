"""Enumeration types for LocalData MCP configuration."""

from enum import Enum


class LogLevel(str, Enum):
    """Supported logging levels."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class DatabaseType(str, Enum):
    """Supported database types."""

    SQLITE = "sqlite"
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    DUCKDB = "duckdb"
    REDIS = "redis"
    ELASTICSEARCH = "elasticsearch"
    MONGODB = "mongodb"
    INFLUXDB = "influxdb"
    NEO4J = "neo4j"
    COUCHDB = "couchdb"
    ORACLE = "oracle"
    MSSQL = "mssql"
    # File formats
    CSV = "csv"
    JSON = "json"
    YAML = "yaml"
    TOML = "toml"
    EXCEL = "excel"
    ODS = "ods"
    NUMBERS = "numbers"
    XML = "xml"
    INI = "ini"
    TSV = "tsv"
    PARQUET = "parquet"
    FEATHER = "feather"
    ARROW = "arrow"
    HDF5 = "hdf5"


class OutputFormat(str, Enum):
    """Supported log output formats."""

    JSON = "json"
    TEXT = "text"
    CONSOLE = "console"


class OutputDestination(str, Enum):
    """Supported log output destinations."""

    STDOUT = "stdout"
    FILE = "file"
    SYSLOG = "syslog"
    JSON_FILE = "json_file"
