"""
Performance optimization for the pipeline integration layer.

Provides caching, lazy loading, and performance monitoring for
data format conversions.
"""

from ._types import (
    CacheEvictionPolicy,
    CacheStatistics,
    CachedConversion,
    OptimizationStrategy,
)
from ._cache import ConversionCache
from ._lazy import LazyConversionState, LazyConverter, LazyLoadingManager

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
