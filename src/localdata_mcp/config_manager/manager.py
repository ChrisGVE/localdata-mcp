"""ConfigManager class and singleton access for LocalData MCP."""

import os
import threading
import time
from typing import Any, Dict, Optional

from dotenv import load_dotenv

from .env_loader import load_env_config
from .loaders import deep_merge, load_yaml_config, validate_config
from .models import (
    DatabaseConfig,
    LoggingConfig,
    PerformanceConfig,
)
from .types import DatabaseType, LogLevel


class ConfigManager:
    """Centralized configuration manager supporting environment variables and YAML files."""

    # Legacy paths kept as property for backward compatibility
    DEFAULT_CONFIG_FILES = [
        "./.localdata.yaml",
        "~/.localdata.yaml",
        "/etc/localdata.yaml",
    ]

    def __init__(self, config_file: Optional[str] = None, auto_reload: bool = False):
        """Initialize configuration manager.

        Args:
            config_file: Specific config file to load. If None, searches default locations.
            auto_reload: Enable automatic reloading when config files change.
        """
        self._config_file = config_file
        self._auto_reload = auto_reload
        self._config_data: Dict[str, Any] = {}
        self._last_reload = 0.0
        self._file_mtimes: Dict[str, float] = {}
        self._lock = threading.Lock()

        # Load environment variables
        load_dotenv()

        # Initial configuration load
        self.reload_config()

        # Start auto-reload thread if enabled
        if self._auto_reload:
            self._start_reload_thread()

    def reload_config(self) -> None:
        """Reload configuration from all sources with proper precedence."""
        with self._lock:
            self._config_data = {}

            # 1. Load defaults
            self._apply_defaults()

            # 2. Load from YAML files (discovery order)
            yaml_data = load_yaml_config(
                self._config_file, self._file_mtimes, self._merge_config
            )
            if yaml_data:
                self._merge_config(yaml_data)

            # 3. Override with environment variables
            env_data = load_env_config()
            if env_data:
                self._merge_config(env_data)

            # 4. Validate final configuration
            self._validate_config()

            self._last_reload = time.time()

    def get_database_config(self, name: str) -> Optional[DatabaseConfig]:
        """Get database configuration by name."""
        db_configs = self.get_database_configs()
        return db_configs.get(name)

    def get_database_configs(self) -> Dict[str, DatabaseConfig]:
        """Get all database configurations."""
        db_configs = {}

        databases = self._config_data.get("databases", {})
        for name, config in databases.items():
            try:
                db_config = DatabaseConfig(
                    name=name,
                    type=DatabaseType(config["type"]),
                    connection_string=config["connection_string"],
                    sheet_name=config.get("sheet_name"),
                    enabled=config.get("enabled", True),
                    max_connections=config.get("max_connections", 10),
                    connection_timeout=config.get("connection_timeout", 30),
                    query_timeout=config.get("query_timeout", 300),
                    tags=config.get("tags", []),
                    metadata=config.get("metadata", {}),
                )
                db_configs[name] = db_config
            except Exception as e:
                print(f"Warning: Invalid database config for '{name}': {e}")
                continue

        return db_configs

    def get_logging_config(self) -> LoggingConfig:
        """Get logging configuration."""
        logging_data = self._config_data.get("logging", {})
        return LoggingConfig(
            level=LogLevel(logging_data.get("level", LogLevel.INFO.value)),
            format=logging_data.get(
                "format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            ),
            file_path=logging_data.get("file_path"),
            max_file_size=logging_data.get("max_file_size", 10 * 1024 * 1024),
            backup_count=logging_data.get("backup_count", 5),
            console_output=logging_data.get("console_output", True),
        )

    def get_performance_config(self) -> PerformanceConfig:
        """Get performance configuration."""
        perf_data = self._config_data.get("performance", {})
        return PerformanceConfig(
            memory_limit_mb=perf_data.get("memory_limit_mb", 2048),
            query_buffer_timeout=perf_data.get("query_buffer_timeout", 600),
            max_concurrent_connections=perf_data.get("max_concurrent_connections", 10),
            chunk_size=perf_data.get("chunk_size", 100),
            enable_query_analysis=perf_data.get("enable_query_analysis", True),
            auto_cleanup_buffers=perf_data.get("auto_cleanup_buffers", True),
            memory_warning_threshold=perf_data.get("memory_warning_threshold", 0.85),
        )

    def get_staging_config(self):
        """Get staging database configuration."""
        from ..config_schemas import StagingConfig

        data = self._config_data.get("staging", {})
        return StagingConfig(**data) if data else StagingConfig()

    def get_memory_config(self):
        """Get memory budget configuration."""
        from ..config_schemas import MemoryConfig

        data = self._config_data.get("memory", {})
        return MemoryConfig(**data) if data else MemoryConfig()

    def get_query_config(self):
        """Get query execution configuration."""
        from ..config_schemas import QueryConfig

        data = self._config_data.get("query", {})
        return QueryConfig(**data) if data else QueryConfig()

    def get_connections_config(self):
        """Get connections configuration."""
        from ..config_schemas import ConnectionsConfig

        data = self._config_data.get("connections", {})
        return ConnectionsConfig(**data) if data else ConnectionsConfig()

    def get_security_config(self):
        """Get security configuration."""
        from ..config_schemas import SecurityConfig

        data = self._config_data.get("security", {})
        return SecurityConfig(**data) if data else SecurityConfig()

    def get_disk_budget_config(self):
        """Get disk budget configuration."""
        from ..config_schemas import DiskBudgetConfig

        data = self._config_data.get("disk_budget", {})
        return DiskBudgetConfig(**data) if data else DiskBudgetConfig()

    def has_config_changed(self) -> bool:
        """Check if any configuration files have been modified since last reload."""
        for file_path, last_mtime in self._file_mtimes.items():
            try:
                current_mtime = os.path.getmtime(file_path)
                if current_mtime > last_mtime:
                    return True
            except OSError:
                continue
        return False

    def _apply_defaults(self) -> None:
        """Apply default configuration values."""
        self._config_data = {
            "databases": {},
            "logging": {"level": LogLevel.INFO.value, "console_output": True},
            "performance": {
                "memory_limit_mb": 2048,
                "query_buffer_timeout": 600,
                "max_concurrent_connections": 10,
                "chunk_size": 100,
                "enable_query_analysis": True,
                "auto_cleanup_buffers": True,
                "memory_warning_threshold": 0.85,
            },
        }

    def _merge_config(self, new_config: Dict[str, Any]) -> None:
        """Deep merge new configuration into existing configuration."""
        deep_merge(self._config_data, new_config)

    def _validate_config(self) -> None:
        """Validate the final merged configuration."""
        validate_config(self._config_data)

    def _start_reload_thread(self) -> None:
        """Start background thread for automatic config reloading."""

        def reload_worker():
            while self._auto_reload:
                time.sleep(5)
                if self.has_config_changed():
                    try:
                        self.reload_config()
                        print("Configuration reloaded due to file changes")
                    except Exception as e:
                        print(f"Failed to reload configuration: {e}")

        thread = threading.Thread(target=reload_worker, daemon=True)
        thread.start()


# Global configuration manager instance
_config_manager: Optional[ConfigManager] = None


def get_config_manager() -> ConfigManager:
    """Get or create global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def initialize_config(
    config_file: Optional[str] = None, auto_reload: bool = False
) -> ConfigManager:
    """Initialize global configuration manager with specific settings."""
    global _config_manager
    _config_manager = ConfigManager(config_file=config_file, auto_reload=auto_reload)
    return _config_manager
