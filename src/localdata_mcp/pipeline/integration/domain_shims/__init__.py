"""
Pre-built Domain Shims for LocalData MCP v2.0 Integration Framework.

This package provides pre-built domain shims for common data science workflows,
enabling seamless integration between statistical, regression, time series,
and pattern recognition domains.

Key Features:
- StatisticalShim: Bridge statistical analysis with other domains
- RegressionShim: Connect regression modeling with other domains
- TimeSeriesShim: Enable time series integration across domains
- PatternRecognitionShim: Bridge pattern recognition with other domains
- Domain-specific parameter mapping and result normalization
- Intelligent semantic understanding of data transformations

Design Principles:
- Intention-Driven Interface: Shims understand analytical goals, not just data formats
- Context-Aware Composition: Preserve semantic meaning across domain boundaries
- Progressive Disclosure: Simple defaults with advanced customization options
- Streaming-First: Memory-efficient processing for large datasets
- Modular Domain Integration: Easy extension to new domains
"""

from ._types import (
    DomainShimType,
    DomainMapping,
    SemanticContext,
)

from ._base import (
    BaseDomainShim,
)

from ._statistical import (
    StatisticalShim,
)

from ._regression import (
    RegressionShim,
)

from ._time_series import (
    TimeSeriesShim,
)

from ._pattern_recognition import (
    PatternRecognitionShim,
)

from ._factories import (
    create_statistical_shim,
    create_regression_shim,
    create_time_series_shim,
    create_pattern_recognition_shim,
    create_all_domain_shims,
    get_compatible_domain_shims,
    validate_domain_shim_configuration,
)

__all__ = [
    # Domain configuration and mapping
    "DomainShimType",
    "DomainMapping",
    "SemanticContext",
    # Domain-specific shim classes
    "BaseDomainShim",
    "StatisticalShim",
    "RegressionShim",
    "TimeSeriesShim",
    "PatternRecognitionShim",
    # Factory functions
    "create_statistical_shim",
    "create_regression_shim",
    "create_time_series_shim",
    "create_pattern_recognition_shim",
    "create_all_domain_shims",
    # Utility functions
    "get_compatible_domain_shims",
    "validate_domain_shim_configuration",
]
