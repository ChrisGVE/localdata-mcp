"""
Pipeline Compatibility Matrix for LocalData MCP v2.0 Integration Shims Framework.

This module provides domain compatibility mapping, format specification management,
and automated conversion pathway discovery to enable seamless cross-domain pipeline
composition for LLM agents.

Key Features:
- PipelineCompatibilityMatrix for comprehensive domain compatibility assessment
- Domain profiles for Statistical, Regression, Time Series, and Pattern Recognition domains
- Automatic compatibility scoring and pathway discovery
- Pipeline validation with detailed error reporting and recommendations
- Integration with existing TypeDetectionEngine and converter framework
- Extensible architecture for future domain additions

Design Principles:
- Intention-Driven Interface: Score compatibility based on analytical goals
- Context-Aware Composition: Consider upstream/downstream context in validation
- Progressive Disclosure: Simple scoring with detailed breakdowns available
- Streaming-First: Memory-efficient compatibility checking
- Modular Integration: Easy addition of new domains and formats
"""

from ._matrix import PipelineCompatibilityMatrix
from ._types import CompatibilityLevel, DomainProfile
from ._utilities import (
    assess_pipeline_compatibility,
    create_compatibility_matrix,
    create_minimal_compatibility_matrix,
    find_optimal_format_for_domains,
    suggest_pipeline_improvements,
)

__all__ = [
    "CompatibilityLevel",
    "DomainProfile",
    "PipelineCompatibilityMatrix",
    "create_compatibility_matrix",
    "create_minimal_compatibility_matrix",
    "assess_pipeline_compatibility",
    "find_optimal_format_for_domains",
    "suggest_pipeline_improvements",
]
