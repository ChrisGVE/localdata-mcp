"""
Type definitions for the Performance Optimization System.

Enums and dataclasses used across the performance optimization sub-package.
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

from ..interfaces import DataFormat


class OptimizationStrategy(Enum):
    """Available optimization strategies for conversion operations."""

    CACHE_FIRST = "cache_first"  # Prioritize cache hits for repeated operations
    LAZY_LOADING = "lazy_loading"  # Defer conversion until data is actually needed
    STREAMING = "streaming"  # Use chunked processing for large datasets
    MEMORY_POOL = "memory_pool"  # Reuse allocated memory for similar operations
    PARALLEL = "parallel"  # Use multiple threads/processes for independent conversions
    ADAPTIVE = (
        "adaptive"  # Dynamically select best strategy based on data characteristics
    )


class CacheEvictionPolicy(Enum):
    """Cache eviction policies for memory management."""

    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    FIFO = "fifo"  # First In, First Out
    MEMORY_PRESSURE = "memory_pressure"  # Based on memory usage


@dataclass
class CacheStatistics:
    """Statistics for conversion cache performance."""

    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    evictions: int = 0
    memory_usage_mb: float = 0.0
    hit_rate: float = 0.0
    average_lookup_time_ms: float = 0.0
    cache_size: int = 0
    max_cache_size: int = 0
    last_cleanup_time: Optional[float] = None


@dataclass
class CachedConversion:
    """A cached conversion result with metadata."""

    cache_key: str
    converted_data: Any
    original_format: DataFormat
    target_format: DataFormat
    metadata: Dict[str, Any]
    creation_time: float = field(default_factory=time.time)
    last_access_time: float = field(default_factory=time.time)
    access_count: int = 1
    size_mb: float = 0.0
    ttl_seconds: Optional[int] = None

    def is_expired(self) -> bool:
        """Check if cached conversion has expired."""
        if self.ttl_seconds is None:
            return False
        return time.time() - self.creation_time > self.ttl_seconds

    def touch(self) -> None:
        """Update last access time and increment access count."""
        self.last_access_time = time.time()
        self.access_count += 1
