"""Configuration dataclasses and Pydantic models for LocalData MCP."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .types import (
    DatabaseType,
    LogLevel,
    OutputDestination,
    OutputFormat,
)


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
        if self.type in [
            DatabaseType.EXCEL,
            DatabaseType.ODS,
            DatabaseType.NUMBERS,
            DatabaseType.HDF5,
        ]:
            # For file formats that support multiple sheets/datasets
            pass
        if self.max_connections <= 0:
            raise ValueError(
                f"max_connections must be positive, got {self.max_connections}"
            )
        if self.connection_timeout <= 0:
            raise ValueError(
                f"connection_timeout must be positive, got {self.connection_timeout}"
            )
        if self.query_timeout <= 0:
            raise ValueError(
                f"query_timeout must be positive, got {self.query_timeout}"
            )


@dataclass
class LoggingConfig:
    """Enhanced logging configuration with structured logging support."""

    level: LogLevel = LogLevel.INFO
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    output_format: OutputFormat = OutputFormat.JSON
    destinations: List[OutputDestination] = field(
        default_factory=lambda: [OutputDestination.STDOUT]
    )
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
            raise ValueError(
                f"max_file_size must be positive, got {self.max_file_size}"
            )
        if self.backup_count < 0:
            raise ValueError(
                f"backup_count must be non-negative, got {self.backup_count}"
            )
        if self.slow_query_threshold <= 0:
            raise ValueError(
                f"slow_query_threshold must be positive, got {self.slow_query_threshold}"
            )
        if self.very_slow_query_threshold <= self.slow_query_threshold:
            raise ValueError(
                f"very_slow_query_threshold must be greater than slow_query_threshold"
            )
        if self.metrics_port <= 0 or self.metrics_port > 65535:
            raise ValueError(
                f"metrics_port must be between 1-65535, got {self.metrics_port}"
            )

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
            raise ValueError(
                f"memory_limit_mb must be positive, got {self.memory_limit_mb}"
            )
        if self.query_buffer_timeout < 0:
            raise ValueError(
                f"query_buffer_timeout must be non-negative, got {self.query_buffer_timeout}"
            )
        if self.max_concurrent_connections <= 0:
            raise ValueError(
                f"max_concurrent_connections must be positive, got {self.max_concurrent_connections}"
            )
        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {self.chunk_size}")
        if not 0 <= self.memory_warning_threshold <= 1:
            raise ValueError(
                f"memory_warning_threshold must be between 0 and 1, got {self.memory_warning_threshold}"
            )


class LocalDataConfig(BaseModel):
    """Root configuration model with validation."""

    databases: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    logging: Dict[str, Any] = Field(default_factory=dict)
    performance: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="allow")

    @model_validator(mode="before")
    @classmethod
    def validate_structure(cls, values):
        """Validate overall configuration structure."""
        if not isinstance(values, dict):
            raise ValueError("Configuration must be a dictionary")
        return values

    @field_validator("databases")
    @classmethod
    def validate_databases(cls, v):
        """Validate database configurations."""
        if not isinstance(v, dict):
            raise ValueError("databases must be a dictionary")

        for name, config in v.items():
            if not isinstance(config, dict):
                raise ValueError(
                    f"Database '{name}' configuration must be a dictionary"
                )

            # Ensure required fields
            if "type" not in config:
                raise ValueError(f"Database '{name}' missing required field 'type'")
            if "connection_string" not in config:
                raise ValueError(
                    f"Database '{name}' missing required field 'connection_string'"
                )

            # Validate database type
            try:
                DatabaseType(config["type"])
            except ValueError:
                valid_types = [t.value for t in DatabaseType]
                raise ValueError(
                    f"Database '{name}' has invalid type '{config['type']}'. Valid types: {valid_types}"
                )

        return v
