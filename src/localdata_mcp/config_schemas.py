"""Configuration schema dataclasses for LocalData MCP v1.6.0.

New config sections for staging, memory, query execution, connections,
and security.  Separated from config_manager.py to keep file sizes
within limits.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List


class BlobHandling(str, Enum):
    """How to handle BLOB columns in query results."""

    EXCLUDE = "exclude"
    INCLUDE = "include"
    PLACEHOLDER = "placeholder"


class EvictionPolicy(str, Enum):
    """Eviction strategy for staging databases."""

    LRU = "lru"
    OLDEST = "oldest"


@dataclass
class StagingConfig:
    """Staging database settings."""

    max_concurrent: int = 10
    max_size_mb: int = 2048
    max_total_mb: int = 10240
    timeout_minutes: int = 30
    eviction_policy: EvictionPolicy = EvictionPolicy.LRU

    def __post_init__(self):
        if isinstance(self.eviction_policy, str):
            self.eviction_policy = EvictionPolicy(self.eviction_policy)
        if not 1 <= self.max_concurrent <= 100:
            raise ValueError(f"max_concurrent must be 1-100, got {self.max_concurrent}")
        if self.max_size_mb <= 0:
            raise ValueError(f"max_size_mb must be positive, got {self.max_size_mb}")
        if self.max_total_mb < self.max_size_mb:
            raise ValueError(
                f"max_total_mb ({self.max_total_mb}) must be >= "
                f"max_size_mb ({self.max_size_mb})"
            )
        if self.timeout_minutes <= 0:
            raise ValueError(
                f"timeout_minutes must be positive, got {self.timeout_minutes}"
            )


@dataclass
class MemoryConfig:
    """Memory budget settings."""

    max_budget_mb: int = 512
    budget_percent: int = 10
    low_memory_threshold_gb: float = 1.0
    aggressive_budget_percent: int = 5
    aggressive_max_mb: int = 128

    def __post_init__(self):
        if self.max_budget_mb <= 0:
            raise ValueError(
                f"max_budget_mb must be positive, got {self.max_budget_mb}"
            )
        if not 1 <= self.budget_percent <= 100:
            raise ValueError(f"budget_percent must be 1-100, got {self.budget_percent}")
        if self.low_memory_threshold_gb <= 0:
            raise ValueError(
                f"low_memory_threshold_gb must be positive, "
                f"got {self.low_memory_threshold_gb}"
            )
        if not 1 <= self.aggressive_budget_percent <= 100:
            raise ValueError(
                f"aggressive_budget_percent must be 1-100, "
                f"got {self.aggressive_budget_percent}"
            )
        if self.aggressive_max_mb <= 0:
            raise ValueError(
                f"aggressive_max_mb must be positive, got {self.aggressive_max_mb}"
            )


@dataclass
class QueryConfig:
    """Query execution settings."""

    default_chunk_size: int = 100
    buffer_timeout_seconds: int = 600
    blob_handling: BlobHandling = BlobHandling.EXCLUDE
    blob_max_size_mb: int = 5
    preflight_default: bool = False

    def __post_init__(self):
        if isinstance(self.blob_handling, str):
            self.blob_handling = BlobHandling(self.blob_handling)
        if self.default_chunk_size <= 0:
            raise ValueError(
                f"default_chunk_size must be positive, got {self.default_chunk_size}"
            )
        if self.buffer_timeout_seconds < 0:
            raise ValueError(
                f"buffer_timeout_seconds must be non-negative, "
                f"got {self.buffer_timeout_seconds}"
            )
        if self.blob_max_size_mb <= 0:
            raise ValueError(
                f"blob_max_size_mb must be positive, got {self.blob_max_size_mb}"
            )


@dataclass
class ConnectionsConfig:
    """Connection limit settings."""

    max_concurrent: int = 10
    timeout_seconds: int = 30

    def __post_init__(self):
        if self.max_concurrent <= 0:
            raise ValueError(
                f"max_concurrent must be positive, got {self.max_concurrent}"
            )
        if self.timeout_seconds <= 0:
            raise ValueError(
                f"timeout_seconds must be positive, got {self.timeout_seconds}"
            )


@dataclass
class SecurityConfig:
    """Security settings."""

    allowed_paths: List[str] = None  # type: ignore[assignment]
    restrict_paths: bool = True
    max_query_length: int = 10000
    blocked_keywords: List[str] = None  # type: ignore[assignment]
    readonly: bool = False

    def __post_init__(self):
        if self.allowed_paths is None:
            self.allowed_paths = ["."]
        if self.blocked_keywords is None:
            self.blocked_keywords = []
        if self.max_query_length <= 0:
            raise ValueError(
                f"max_query_length must be positive, got {self.max_query_length}"
            )


@dataclass
class DiskBudgetConfig:
    """Disk budget settings for staging databases."""

    max_staging_size_mb: int = 2048
    max_total_staging_mb: int = 10240
    disk_warning_threshold: float = 0.90
    headroom_mb: int = 500
    check_interval_rows: int = 1000

    def __post_init__(self):
        if self.max_staging_size_mb <= 0:
            raise ValueError(
                f"max_staging_size_mb must be positive, got {self.max_staging_size_mb}"
            )
        if self.max_total_staging_mb < self.max_staging_size_mb:
            raise ValueError("max_total_staging_mb must be >= max_staging_size_mb")
        if not 0 < self.disk_warning_threshold <= 1:
            raise ValueError(
                f"disk_warning_threshold must be 0-1, got {self.disk_warning_threshold}"
            )
        if self.headroom_mb <= 0:
            raise ValueError(f"headroom_mb must be positive, got {self.headroom_mb}")
        if self.check_interval_rows <= 0:
            raise ValueError(
                f"check_interval_rows must be positive, got {self.check_interval_rows}"
            )
