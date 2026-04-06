"""
Utility functions for easy metadata operations.

Provides factory functions for creating preservation rules and metadata schemas.
"""

from typing import List, Optional

from ..interfaces import DataFormat
from ._types import (
    MetadataSchema,
    MetadataType,
    PreservationRule,
    PreservationStrategy,
)


def create_preservation_rule(
    metadata_key: str,
    strategy: PreservationStrategy = PreservationStrategy.ADAPTIVE,
    metadata_type: MetadataType = MetadataType.SEMANTIC,
    **kwargs,
) -> PreservationRule:
    """Create a metadata preservation rule."""
    return PreservationRule(
        metadata_key=metadata_key,
        metadata_type=metadata_type,
        preservation_strategy=strategy,
        **kwargs,
    )


def create_metadata_schema(
    schema_name: str,
    data_format: DataFormat,
    required_fields: List[str] = None,
    optional_fields: List[str] = None,
    **kwargs,
) -> MetadataSchema:
    """Create a metadata schema."""
    return MetadataSchema(
        schema_name=schema_name,
        data_format=data_format,
        required_fields=set(required_fields or []),
        optional_fields=set(optional_fields or []),
        **kwargs,
    )
