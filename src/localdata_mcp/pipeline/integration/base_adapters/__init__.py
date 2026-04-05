"""
Base adapter implementations for the Integration Shims Framework.

This package provides foundational converter classes with sklearn-compatible
transformer interfaces, streaming capabilities, and comprehensive error handling.

Key Features:
- BaseShimAdapter with sklearn-compatible fit/transform pattern
- StreamingShimAdapter for memory-efficient large dataset processing
- CachingShimAdapter for performance optimization
- Utility adapters for common conversion patterns
- Comprehensive validation and error handling
"""

from ._core import ConversionContext, BaseShimAdapter
from ._streaming import StreamingShimAdapter
from ._caching import CachingShimAdapter
from ._utility import PassThroughAdapter, ValidationAdapter

__all__ = [
    "ConversionContext",
    "BaseShimAdapter",
    "StreamingShimAdapter",
    "CachingShimAdapter",
    "PassThroughAdapter",
    "ValidationAdapter",
]
