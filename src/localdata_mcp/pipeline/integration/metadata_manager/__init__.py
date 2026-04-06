"""
Metadata Management System for Integration Shims Framework.

This package provides comprehensive metadata preservation, transformation, and
validation capabilities to ensure context preservation throughout data conversions.

Key Features:
- MetadataManager for centralized metadata operations
- PreservationRule system for configurable metadata handling
- MetadataSchema validation and enforcement
- Cross-format metadata mapping and transformation
- Metadata lineage tracking and audit trails
- Integration with existing pipeline metadata systems
"""

from ._manager import MetadataManager
from ._transformers import (
    MetadataTransformer,
    NumpyMetadataTransformer,
    PandasMetadataTransformer,
)
from ._types import (
    MetadataLineage,
    MetadataSchema,
    MetadataType,
    PreservationRule,
    PreservationStrategy,
)
from ._utils import create_metadata_schema, create_preservation_rule

__all__ = [
    # Core manager
    "MetadataManager",
    # Types and enums
    "PreservationStrategy",
    "MetadataType",
    "PreservationRule",
    "MetadataSchema",
    "MetadataLineage",
    # Transformers
    "MetadataTransformer",
    "PandasMetadataTransformer",
    "NumpyMetadataTransformer",
    # Utility functions
    "create_preservation_rule",
    "create_metadata_schema",
]
