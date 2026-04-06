"""
Performance optimization for the pipeline integration layer.

Provides caching, lazy loading, and performance monitoring for
data format conversions.
"""

from ._cache import ConversionCache
from ._lazy import LazyConversionState, LazyConverter, LazyLoadingManager
from ._types import (
    CachedConversion,
    CacheEvictionPolicy,
    CacheStatistics,
    OptimizationStrategy,
)

__all__ = [
    "CacheEvictionPolicy",
    "CacheStatistics",
    "CachedConversion",
    "ConversionCache",
    "LazyConversionState",
    "LazyConverter",
    "LazyLoadingManager",
    "OptimizationStrategy",
]
