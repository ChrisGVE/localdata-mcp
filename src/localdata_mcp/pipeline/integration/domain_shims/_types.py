"""
Shared types and data structures for domain shims.

Provides DomainShimType enum, DomainMapping and SemanticContext dataclasses
used across all domain shim implementations.
"""

from typing import Any, Dict
from dataclasses import dataclass, field
from enum import Enum


class DomainShimType(Enum):
    """Types of domain shims available."""

    STATISTICAL = "statistical"
    REGRESSION = "regression"
    TIME_SERIES = "time_series"
    PATTERN_RECOGNITION = "pattern_recognition"


@dataclass
class DomainMapping:
    """Mapping configuration for cross-domain transformations."""

    source_domain: str
    target_domain: str
    parameter_mappings: Dict[str, str] = field(default_factory=dict)
    result_transformations: Dict[str, str] = field(default_factory=dict)
    semantic_hints: Dict[str, Any] = field(default_factory=dict)
    quality_preservation: float = 1.0  # 0-1 score for information preservation


@dataclass
class SemanticContext:
    """Semantic context for domain-aware transformations."""

    analytical_goal: str  # Primary analysis intention
    domain_context: str  # Source domain context
    target_use_case: str  # Target domain use case
    data_characteristics: Dict[str, Any] = field(default_factory=dict)
    transformation_hints: Dict[str, Any] = field(default_factory=dict)
    quality_requirements: Dict[str, float] = field(default_factory=dict)
