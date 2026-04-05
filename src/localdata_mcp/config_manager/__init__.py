"""Configuration management sub-package for LocalData MCP.

Provides dual configuration support for both environment variables (simple setups)
and YAML files (complex multi-database configurations) with validation,
environment variable substitution, and hot-reload capabilities.
"""

from .types import (
    DatabaseType,
    LogLevel,
    OutputDestination,
    OutputFormat,
)
from .models import (
    DatabaseConfig,
    LocalDataConfig,
    LoggingConfig,
    PerformanceConfig,
)
from .manager import (
    ConfigManager,
    get_config_manager,
    initialize_config,
)

__all__ = [
    "ConfigManager",
    "DatabaseConfig",
    "DatabaseType",
    "LocalDataConfig",
    "LogLevel",
    "LoggingConfig",
    "OutputDestination",
    "OutputFormat",
    "PerformanceConfig",
    "get_config_manager",
    "initialize_config",
]
