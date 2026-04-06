"""
Type definitions for the Pipeline Compatibility Matrix.

Contains enumerations and dataclass definitions used across the
compatibility matrix sub-package.
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Tuple

from ..interfaces import DataFormat, DataFormatSpec, DomainRequirements


class CompatibilityLevel(Enum):
    """Compatibility levels between formats and domains."""

    PERFECT = "perfect"  # Direct compatibility, no conversion needed (0.95-1.0)
    HIGH = "high"  # Compatible with minimal conversion (0.8-0.94)
    MODERATE = "moderate"  # Compatible with standard conversion (0.6-0.79)
    LOW = "low"  # Compatible with complex conversion (0.3-0.59)
    INCOMPATIBLE = "incompatible"  # Not compatible (0.0-0.29)

    @property
    def score_threshold(self) -> float:
        """Get minimum score threshold for this level."""
        return {
            CompatibilityLevel.PERFECT: 0.95,
            CompatibilityLevel.HIGH: 0.8,
            CompatibilityLevel.MODERATE: 0.6,
            CompatibilityLevel.LOW: 0.3,
            CompatibilityLevel.INCOMPATIBLE: 0.0,
        }[self]


@dataclass
class DomainProfile:
    """Extended domain profile with tool-specific requirements."""

    domain_name: str
    base_requirements: DomainRequirements
    tool_specifications: Dict[str, DataFormatSpec] = field(default_factory=dict)
    compatibility_preferences: Dict[DataFormat, float] = field(default_factory=dict)
    conversion_costs: Dict[Tuple[DataFormat, DataFormat], float] = field(
        default_factory=dict
    )
    last_updated: float = field(default_factory=time.time)
