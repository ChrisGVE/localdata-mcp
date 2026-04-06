"""Environment variable loading for LocalData MCP configuration."""

import os
from typing import Any, Dict


def load_env_config() -> Dict[str, Any]:
    """Load configuration from environment variables with backward compatibility."""
    env_config: Dict[str, Any] = {
        "databases": {},
        "logging": {},
        "performance": {},
    }

    _load_legacy_db_vars(env_config)
    _load_modern_db_vars(env_config)
    _load_logging_vars(env_config)
    _load_performance_vars(env_config)
    _load_section_overrides(env_config)

    return env_config


def _load_legacy_db_vars(env_config: Dict[str, Any]) -> None:
    """Load legacy single-database environment variables."""
    legacy_vars = {
        "POSTGRES_HOST": ("postgresql", "host"),
        "POSTGRES_PORT": ("postgresql", "port"),
        "POSTGRES_USER": ("postgresql", "user"),
        "POSTGRES_PASSWORD": ("postgresql", "password"),
        "POSTGRES_DB": ("postgresql", "database"),
        "MYSQL_HOST": ("mysql", "host"),
        "MYSQL_PORT": ("mysql", "port"),
        "MYSQL_USER": ("mysql", "user"),
        "MYSQL_PASSWORD": ("mysql", "password"),
        "MYSQL_DB": ("mysql", "database"),
        "SQLITE_PATH": ("sqlite", "path"),
        "DUCKDB_PATH": ("duckdb", "path"),
    }

    # Process legacy variables and build connection strings
    db_parts: Dict[str, Dict[str, str]] = {}
    for env_var, (db_type, part) in legacy_vars.items():
        value = os.getenv(env_var)
        if value:
            if db_type not in db_parts:
                db_parts[db_type] = {}
            db_parts[db_type][part] = value

    # Build connection strings from parts
    for db_type, parts in db_parts.items():
        if db_type == "postgresql":
            if all(k in parts for k in ["host", "user", "password", "database"]):
                port = parts.get("port", "5432")
                conn_str = (
                    f"postgresql://{parts['user']}:{parts['password']}"
                    f"@{parts['host']}:{port}/{parts['database']}"
                )
                env_config["databases"][f"default_{db_type}"] = {
                    "type": db_type,
                    "connection_string": conn_str,
                }
        elif db_type == "mysql":
            if all(k in parts for k in ["host", "user", "password", "database"]):
                port = parts.get("port", "3306")
                conn_str = (
                    f"mysql+mysqlconnector://{parts['user']}:{parts['password']}"
                    f"@{parts['host']}:{port}/{parts['database']}"
                )
                env_config["databases"][f"default_{db_type}"] = {
                    "type": db_type,
                    "connection_string": conn_str,
                }
        elif db_type in ["sqlite", "duckdb"]:
            if "path" in parts:
                env_config["databases"][f"default_{db_type}"] = {
                    "type": db_type,
                    "connection_string": parts["path"],
                }


_KNOWN_SUFFIXES = [
    "CONNECTION_STRING",
    "SHEET_NAME",
    "AUTO_CLEANUP_BUFFERS",
    "CONSOLE_OUTPUT",
    "MAX_CONNECTIONS",
    "CONNECTION_TIMEOUT",
    "QUERY_TIMEOUT",
    "MAX_FILE_SIZE",
    "BACKUP_COUNT",
    "CHUNK_SIZE",
    "MEMORY_LIMIT_MB",
    "QUERY_BUFFER_TIMEOUT",
    "MAX_CONCURRENT_CONNECTIONS",
    "MEMORY_WARNING_THRESHOLD",
    "ENABLED",
    "TYPE",
]

_DB_PREFIX = "LOCALDATA_DB_"


def _load_modern_db_vars(env_config: Dict[str, Any]) -> None:
    """Load LOCALDATA_DB_* environment variables."""
    for key, value in os.environ.items():
        if not key.startswith(_DB_PREFIX):
            continue
        remainder = key[len(_DB_PREFIX) :]
        db_name = None
        property_name = None
        for suffix in _KNOWN_SUFFIXES:
            if remainder.endswith("_" + suffix):
                db_name = remainder[: -(len(suffix) + 1)].lower()
                property_name = suffix.lower()
                break
        if db_name is None or property_name is None:
            continue

        if db_name not in env_config["databases"]:
            env_config["databases"][db_name] = {}

        _set_db_property(env_config["databases"][db_name], key, property_name, value)


def _set_db_property(
    db_dict: Dict[str, Any], key: str, property_name: str, value: str
) -> None:
    """Set a typed property on a database config dict."""
    if property_name == "type":
        db_dict["type"] = value
    elif property_name in ["connection_string", "sheet_name"]:
        db_dict[property_name] = value
    elif property_name in ["enabled", "auto_cleanup_buffers", "console_output"]:
        db_dict[property_name] = value.lower() in ("true", "1", "yes", "on")
    elif property_name in [
        "max_connections",
        "connection_timeout",
        "query_timeout",
        "max_file_size",
        "backup_count",
        "chunk_size",
        "memory_limit_mb",
        "query_buffer_timeout",
        "max_concurrent_connections",
    ]:
        try:
            db_dict[property_name] = int(value)
        except ValueError:
            print(f"Warning: Invalid integer value for {key}: {value}")
    elif property_name == "memory_warning_threshold":
        try:
            db_dict[property_name] = float(value)
        except ValueError:
            print(f"Warning: Invalid float value for {key}: {value}")


def _load_logging_vars(env_config: Dict[str, Any]) -> None:
    """Load logging-related environment variables."""
    log_level = os.getenv("LOCALDATA_LOG_LEVEL")
    if log_level:
        env_config["logging"]["level"] = log_level.lower()

    log_file = os.getenv("LOCALDATA_LOG_FILE")
    if log_file:
        env_config["logging"]["file_path"] = log_file


def _load_performance_vars(env_config: Dict[str, Any]) -> None:
    """Load performance-related environment variables."""
    memory_limit = os.getenv("LOCALDATA_MEMORY_LIMIT_MB")
    if memory_limit:
        try:
            env_config["performance"]["memory_limit_mb"] = int(memory_limit)
        except ValueError:
            print(f"Warning: Invalid memory limit: {memory_limit}")


_SECTION_VARS = {
    "LOCALDATA_STAGING_MAX_CONCURRENT": ("staging", "max_concurrent", int),
    "LOCALDATA_STAGING_MAX_SIZE_MB": ("staging", "max_size_mb", int),
    "LOCALDATA_STAGING_MAX_TOTAL_MB": ("staging", "max_total_mb", int),
    "LOCALDATA_STAGING_TIMEOUT_MINUTES": ("staging", "timeout_minutes", int),
    "LOCALDATA_STAGING_EVICTION_POLICY": ("staging", "eviction_policy", str),
    "LOCALDATA_MEMORY_MAX_BUDGET_MB": ("memory", "max_budget_mb", int),
    "LOCALDATA_MEMORY_BUDGET_PERCENT": ("memory", "budget_percent", int),
    "LOCALDATA_MEMORY_LOW_THRESHOLD_GB": (
        "memory",
        "low_memory_threshold_gb",
        float,
    ),
    "LOCALDATA_MEMORY_AGGRESSIVE_PERCENT": (
        "memory",
        "aggressive_budget_percent",
        int,
    ),
    "LOCALDATA_MEMORY_AGGRESSIVE_MAX_MB": ("memory", "aggressive_max_mb", int),
    "LOCALDATA_QUERY_CHUNK_SIZE": ("query", "default_chunk_size", int),
    "LOCALDATA_QUERY_BUFFER_TIMEOUT": ("query", "buffer_timeout_seconds", int),
    "LOCALDATA_QUERY_BLOB_HANDLING": ("query", "blob_handling", str),
    "LOCALDATA_QUERY_BLOB_MAX_SIZE_MB": ("query", "blob_max_size_mb", int),
    "LOCALDATA_QUERY_PREFLIGHT_DEFAULT": ("query", "preflight_default", bool),
    "LOCALDATA_CONNECTIONS_MAX_CONCURRENT": ("connections", "max_concurrent", int),
    "LOCALDATA_CONNECTIONS_TIMEOUT": ("connections", "timeout_seconds", int),
    "LOCALDATA_SECURITY_MAX_QUERY_LENGTH": ("security", "max_query_length", int),
    "LOCALDATA_SECURITY_RESTRICT_PATHS": ("security", "restrict_paths", bool),
    "LOCALDATA_DISK_BUDGET_MAX_STAGING_MB": (
        "disk_budget",
        "max_staging_size_mb",
        int,
    ),
    "LOCALDATA_DISK_BUDGET_MAX_TOTAL_MB": (
        "disk_budget",
        "max_total_staging_mb",
        int,
    ),
    "LOCALDATA_DISK_BUDGET_WARNING_THRESHOLD": (
        "disk_budget",
        "disk_warning_threshold",
        float,
    ),
    "LOCALDATA_DISK_BUDGET_HEADROOM_MB": ("disk_budget", "headroom_mb", int),
    "LOCALDATA_DISK_BUDGET_CHECK_INTERVAL": (
        "disk_budget",
        "check_interval_rows",
        int,
    ),
}


def _load_section_overrides(env_config: Dict[str, Any]) -> None:
    """Load section-specific environment variable overrides."""
    for env_name, (section, key, type_fn) in _SECTION_VARS.items():
        value = os.getenv(env_name)
        if value is not None:
            if section not in env_config:
                env_config[section] = {}
            try:
                if type_fn is bool:
                    env_config[section][key] = value.lower() in (
                        "true",
                        "1",
                        "yes",
                        "on",
                    )
                else:
                    env_config[section][key] = type_fn(value)
            except (ValueError, TypeError):
                print(f"Warning: Invalid value for {env_name}: {value}")
