"""Configuration management system for LocalData MCP.

Provides dual configuration support for both environment variables (simple setups)
and YAML files (complex multi-database configurations) with validation,
environment variable substitution, and hot-reload capabilities.
"""

import os
import re
import time
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from enum import Enum
from dataclasses import dataclass, field

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, validator, root_validator
from pydantic import ValidationError as PydanticValidationError


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


@dataclass
class DatabaseConfig:
    """Database connection configuration."""
    name: str
    type: DatabaseType
    connection_string: str
    sheet_name: Optional[str] = None
    enabled: bool = True
    max_connections: int = 10
    connection_timeout: int = 30
    query_timeout: int = 300
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.type in [DatabaseType.EXCEL, DatabaseType.ODS, DatabaseType.NUMBERS, DatabaseType.HDF5]:
            # For file formats that support multiple sheets/datasets
            pass
        if self.max_connections <= 0:
            raise ValueError(f"max_connections must be positive, got {self.max_connections}")
        if self.connection_timeout <= 0:
            raise ValueError(f"connection_timeout must be positive, got {self.connection_timeout}")
        if self.query_timeout <= 0:
            raise ValueError(f"query_timeout must be positive, got {self.query_timeout}")


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


@dataclass
class LoggingConfig:
    """Enhanced logging configuration with structured logging support."""
    level: LogLevel = LogLevel.INFO
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    output_format: OutputFormat = OutputFormat.JSON
    destinations: List[OutputDestination] = field(default_factory=lambda: [OutputDestination.STDOUT])
    file_path: Optional[str] = None
    json_file_path: Optional[str] = None
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    console_output: bool = True
    
    # Structured logging features
    enable_correlation_ids: bool = True
    enable_context_propagation: bool = True
    enable_query_audit: bool = True
    enable_performance_logging: bool = True
    enable_security_logging: bool = True
    
    # Performance logging thresholds
    slow_query_threshold: float = 1.0  # seconds
    very_slow_query_threshold: float = 5.0  # seconds
    
    # Metrics configuration
    enable_metrics: bool = True
    metrics_port: int = 8000
    metrics_endpoint: str = "/metrics"
    
    # Security logging configuration
    log_blocked_queries: bool = True
    log_timeout_events: bool = True
    log_resource_limits: bool = True
    log_failed_connections: bool = True
    
    # Debug configuration
    enable_debug_traces: bool = False
    debug_sql_queries: bool = False
    debug_connection_pool: bool = False

    def __post_init__(self):
        """Validate logging configuration."""
        if self.max_file_size <= 0:
            raise ValueError(f"max_file_size must be positive, got {self.max_file_size}")
        if self.backup_count < 0:
            raise ValueError(f"backup_count must be non-negative, got {self.backup_count}")
        if self.slow_query_threshold <= 0:
            raise ValueError(f"slow_query_threshold must be positive, got {self.slow_query_threshold}")
        if self.very_slow_query_threshold <= self.slow_query_threshold:
            raise ValueError(f"very_slow_query_threshold must be greater than slow_query_threshold")
        if self.metrics_port <= 0 or self.metrics_port > 65535:
            raise ValueError(f"metrics_port must be between 1-65535, got {self.metrics_port}")
        
        # Set default file paths if destinations include file types but paths not specified
        if OutputDestination.FILE in self.destinations and not self.file_path:
            self.file_path = "./logs/localdata-mcp.log"
        if OutputDestination.JSON_FILE in self.destinations and not self.json_file_path:
            self.json_file_path = "./logs/localdata-mcp.json"


@dataclass
class PerformanceConfig:
    """Performance and resource configuration."""
    memory_limit_mb: int = 2048
    query_buffer_timeout: int = 600  # 10 minutes
    max_concurrent_connections: int = 10
    chunk_size: int = 100
    enable_query_analysis: bool = True
    auto_cleanup_buffers: bool = True
    memory_warning_threshold: float = 0.85  # 85%

    def __post_init__(self):
        """Validate performance configuration."""
        if self.memory_limit_mb <= 0:
            raise ValueError(f"memory_limit_mb must be positive, got {self.memory_limit_mb}")
        if self.query_buffer_timeout < 0:
            raise ValueError(f"query_buffer_timeout must be non-negative, got {self.query_buffer_timeout}")
        if self.max_concurrent_connections <= 0:
            raise ValueError(f"max_concurrent_connections must be positive, got {self.max_concurrent_connections}")
        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {self.chunk_size}")
        if not 0 <= self.memory_warning_threshold <= 1:
            raise ValueError(f"memory_warning_threshold must be between 0 and 1, got {self.memory_warning_threshold}")


class LocalDataConfig(BaseModel):
    """Root configuration model with validation."""
    
    databases: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    logging: Dict[str, Any] = Field(default_factory=dict)
    performance: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        extra = "allow"  # Allow additional fields for extensibility

    @root_validator(pre=True)
    def validate_structure(cls, values):
        """Validate overall configuration structure."""
        if not isinstance(values, dict):
            raise ValueError("Configuration must be a dictionary")
        return values

    @validator('databases')
    def validate_databases(cls, v):
        """Validate database configurations."""
        if not isinstance(v, dict):
            raise ValueError("databases must be a dictionary")
        
        for name, config in v.items():
            if not isinstance(config, dict):
                raise ValueError(f"Database '{name}' configuration must be a dictionary")
            
            # Ensure required fields
            if 'type' not in config:
                raise ValueError(f"Database '{name}' missing required field 'type'")
            if 'connection_string' not in config:
                raise ValueError(f"Database '{name}' missing required field 'connection_string'")
            
            # Validate database type
            try:
                DatabaseType(config['type'])
            except ValueError:
                valid_types = [t.value for t in DatabaseType]
                raise ValueError(f"Database '{name}' has invalid type '{config['type']}'. Valid types: {valid_types}")
        
        return v


class ConfigManager:
    """Centralized configuration manager supporting environment variables and YAML files."""

    DEFAULT_CONFIG_FILES = [
        "./localdata.yaml",
        "~/.localdata.yaml", 
        "/etc/localdata.yaml"
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
            yaml_data = self._load_yaml_config()
            if yaml_data:
                self._merge_config(yaml_data)
            
            # 3. Override with environment variables
            env_data = self._load_env_config()
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
        
        databases = self._config_data.get('databases', {})
        for name, config in databases.items():
            try:
                db_config = DatabaseConfig(
                    name=name,
                    type=DatabaseType(config['type']),
                    connection_string=config['connection_string'],
                    sheet_name=config.get('sheet_name'),
                    enabled=config.get('enabled', True),
                    max_connections=config.get('max_connections', 10),
                    connection_timeout=config.get('connection_timeout', 30),
                    query_timeout=config.get('query_timeout', 300),
                    tags=config.get('tags', []),
                    metadata=config.get('metadata', {})
                )
                db_configs[name] = db_config
            except Exception as e:
                # Log error but don't fail completely
                print(f"Warning: Invalid database config for '{name}': {e}")
                continue
                
        return db_configs

    def get_logging_config(self) -> LoggingConfig:
        """Get logging configuration."""
        logging_data = self._config_data.get('logging', {})
        return LoggingConfig(
            level=LogLevel(logging_data.get('level', LogLevel.INFO.value)),
            format=logging_data.get('format', "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
            file_path=logging_data.get('file_path'),
            max_file_size=logging_data.get('max_file_size', 10 * 1024 * 1024),
            backup_count=logging_data.get('backup_count', 5),
            console_output=logging_data.get('console_output', True)
        )

    def get_performance_config(self) -> PerformanceConfig:
        """Get performance configuration."""
        perf_data = self._config_data.get('performance', {})
        return PerformanceConfig(
            memory_limit_mb=perf_data.get('memory_limit_mb', 2048),
            query_buffer_timeout=perf_data.get('query_buffer_timeout', 600),
            max_concurrent_connections=perf_data.get('max_concurrent_connections', 10),
            chunk_size=perf_data.get('chunk_size', 100),
            enable_query_analysis=perf_data.get('enable_query_analysis', True),
            auto_cleanup_buffers=perf_data.get('auto_cleanup_buffers', True),
            memory_warning_threshold=perf_data.get('memory_warning_threshold', 0.85)
        )

    def has_config_changed(self) -> bool:
        """Check if any configuration files have been modified since last reload."""
        for file_path, last_mtime in self._file_mtimes.items():
            try:
                current_mtime = os.path.getmtime(file_path)
                if current_mtime > last_mtime:
                    return True
            except OSError:
                # File might have been deleted
                continue
        return False

    def _apply_defaults(self) -> None:
        """Apply default configuration values."""
        self._config_data = {
            'databases': {},
            'logging': {
                'level': LogLevel.INFO.value,
                'console_output': True
            },
            'performance': {
                'memory_limit_mb': 2048,
                'query_buffer_timeout': 600,
                'max_concurrent_connections': 10,
                'chunk_size': 100,
                'enable_query_analysis': True,
                'auto_cleanup_buffers': True,
                'memory_warning_threshold': 0.85
            }
        }

    def _load_yaml_config(self) -> Optional[Dict[str, Any]]:
        """Load configuration from YAML files with discovery."""
        config_files = [self._config_file] if self._config_file else self.DEFAULT_CONFIG_FILES
        
        for file_path in config_files:
            if not file_path:
                continue
                
            # Expand user home directory
            expanded_path = Path(file_path).expanduser()
            
            if expanded_path.exists():
                try:
                    with open(expanded_path, 'r') as f:
                        content = f.read()
                    
                    # Substitute environment variables
                    content = self._substitute_env_vars(content)
                    
                    # Parse YAML
                    yaml_data = yaml.safe_load(content)
                    
                    # Track file modification time
                    self._file_mtimes[str(expanded_path)] = os.path.getmtime(expanded_path)
                    
                    return yaml_data
                    
                except Exception as e:
                    print(f"Warning: Could not load config file {file_path}: {e}")
                    continue
        
        return None

    def _load_env_config(self) -> Dict[str, Any]:
        """Load configuration from environment variables with backward compatibility."""
        env_config: Dict[str, Any] = {
            'databases': {},
            'logging': {},
            'performance': {}
        }
        
        # Legacy single database environment variables (backward compatibility)
        legacy_vars = {
            'POSTGRES_HOST': ('postgresql', 'host'),
            'POSTGRES_PORT': ('postgresql', 'port'), 
            'POSTGRES_USER': ('postgresql', 'user'),
            'POSTGRES_PASSWORD': ('postgresql', 'password'),
            'POSTGRES_DB': ('postgresql', 'database'),
            'MYSQL_HOST': ('mysql', 'host'),
            'MYSQL_PORT': ('mysql', 'port'),
            'MYSQL_USER': ('mysql', 'user'),
            'MYSQL_PASSWORD': ('mysql', 'password'),
            'MYSQL_DB': ('mysql', 'database'),
            'SQLITE_PATH': ('sqlite', 'path'),
            'DUCKDB_PATH': ('duckdb', 'path'),
        }
        
        # Process legacy variables and build connection strings
        db_parts = {}
        for env_var, (db_type, part) in legacy_vars.items():
            value = os.getenv(env_var)
            if value:
                if db_type not in db_parts:
                    db_parts[db_type] = {}
                db_parts[db_type][part] = value
        
        # Build connection strings from parts
        for db_type, parts in db_parts.items():
            if db_type == 'postgresql':
                if all(k in parts for k in ['host', 'user', 'password', 'database']):
                    port = parts.get('port', '5432')
                    conn_str = f"postgresql://{parts['user']}:{parts['password']}@{parts['host']}:{port}/{parts['database']}"
                    env_config['databases'][f'default_{db_type}'] = {
                        'type': db_type,
                        'connection_string': conn_str
                    }
            elif db_type == 'mysql':
                if all(k in parts for k in ['host', 'user', 'password', 'database']):
                    port = parts.get('port', '3306')
                    conn_str = f"mysql+mysqlconnector://{parts['user']}:{parts['password']}@{parts['host']}:{port}/{parts['database']}"
                    env_config['databases'][f'default_{db_type}'] = {
                        'type': db_type,
                        'connection_string': conn_str
                    }
            elif db_type in ['sqlite', 'duckdb']:
                if 'path' in parts:
                    env_config['databases'][f'default_{db_type}'] = {
                        'type': db_type,
                        'connection_string': parts['path']
                    }
        
        # Modern granular environment variables (prefixed approach)
        # LOCALDATA_DB_<name>_<property>=value
        db_pattern = re.compile(r'^LOCALDATA_DB_([A-Z0-9_]+)_(.+)$')
        
        for key, value in os.environ.items():
            match = db_pattern.match(key)
            if match:
                db_name = match.group(1).lower()
                property_name = match.group(2).lower()
                
                if db_name not in env_config['databases']:
                    env_config['databases'][db_name] = {}
                
                # Convert property names
                if property_name == 'type':
                    env_config['databases'][db_name]['type'] = value
                elif property_name in ['connection_string', 'sheet_name']:
                    env_config['databases'][db_name][property_name] = value
                elif property_name in ['enabled', 'auto_cleanup_buffers', 'console_output']:
                    env_config['databases'][db_name][property_name] = value.lower() in ('true', '1', 'yes', 'on')
                elif property_name in ['max_connections', 'connection_timeout', 'query_timeout', 'max_file_size', 'backup_count', 'chunk_size', 'memory_limit_mb', 'query_buffer_timeout', 'max_concurrent_connections']:
                    try:
                        env_config['databases'][db_name][property_name] = int(value)
                    except ValueError:
                        print(f"Warning: Invalid integer value for {key}: {value}")
                elif property_name == 'memory_warning_threshold':
                    try:
                        env_config['databases'][db_name][property_name] = float(value)
                    except ValueError:
                        print(f"Warning: Invalid float value for {key}: {value}")
        
        # Logging configuration from environment
        log_level = os.getenv('LOCALDATA_LOG_LEVEL')
        if log_level:
            env_config['logging']['level'] = log_level.lower()
        
        log_file = os.getenv('LOCALDATA_LOG_FILE')
        if log_file:
            env_config['logging']['file_path'] = log_file
        
        # Performance configuration from environment  
        memory_limit = os.getenv('LOCALDATA_MEMORY_LIMIT_MB')
        if memory_limit:
            try:
                env_config['performance']['memory_limit_mb'] = int(memory_limit)
            except ValueError:
                print(f"Warning: Invalid memory limit: {memory_limit}")
        
        return env_config

    def _substitute_env_vars(self, content: str) -> str:
        """Substitute environment variables in YAML content using ${VAR} syntax."""
        pattern = re.compile(r'\$\{([^}]+)\}')
        
        def replace_var(match):
            var_name = match.group(1)
            # Support default values: ${VAR:default_value}
            if ':' in var_name:
                var_name, default = var_name.split(':', 1)
                return os.getenv(var_name, default)
            else:
                return os.getenv(var_name, match.group(0))  # Return original if not found
        
        return pattern.sub(replace_var, content)

    def _merge_config(self, new_config: Dict[str, Any]) -> None:
        """Deep merge new configuration into existing configuration."""
        def deep_merge(target, source):
            for key, value in source.items():
                if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                    deep_merge(target[key], value)
                else:
                    target[key] = value
        
        deep_merge(self._config_data, new_config)

    def _validate_config(self) -> None:
        """Validate the final merged configuration."""
        try:
            LocalDataConfig(**self._config_data)
        except PydanticValidationError as e:
            print(f"Configuration validation errors: {e}")
            # Don't raise exception - allow partial configs to work

    def _start_reload_thread(self) -> None:
        """Start background thread for automatic config reloading."""
        def reload_worker():
            while self._auto_reload:
                time.sleep(5)  # Check every 5 seconds
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


def initialize_config(config_file: Optional[str] = None, auto_reload: bool = False) -> ConfigManager:
    """Initialize global configuration manager with specific settings."""
    global _config_manager
    _config_manager = ConfigManager(config_file=config_file, auto_reload=auto_reload)
    return _config_manager