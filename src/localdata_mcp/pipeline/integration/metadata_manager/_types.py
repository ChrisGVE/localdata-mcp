"""
Metadata type definitions, enums, and dataclasses.

Defines the core data structures used throughout the metadata management system:
PreservationStrategy, MetadataType enums, and PreservationRule, MetadataSchema,
MetadataLineage dataclasses.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

from ..interfaces import DataFormat


class PreservationStrategy(Enum):
    """Strategies for metadata preservation during conversion."""

    STRICT = "strict"  # Preserve all metadata exactly
    ADAPTIVE = "adaptive"  # Adapt metadata to target format
    MINIMAL = "minimal"  # Preserve only essential metadata
    CUSTOM = "custom"  # Use custom preservation rules


class MetadataType(Enum):
    """Types of metadata that can be preserved."""

    STRUCTURAL = "structural"  # Shape, dimensions, schema info
    SEMANTIC = "semantic"  # Column meanings, units, descriptions
    OPERATIONAL = "operational"  # Creation time, source, processing history
    QUALITY = "quality"  # Completeness, consistency, accuracy metrics
    LINEAGE = "lineage"  # Data source and transformation history
    CUSTOM = "custom"  # Domain-specific metadata


@dataclass
class PreservationRule:
    """Rule defining how to preserve specific metadata during conversion."""

    metadata_key: str
    metadata_type: MetadataType
    preservation_strategy: PreservationStrategy
    source_formats: Set[DataFormat] = field(default_factory=set)
    target_formats: Set[DataFormat] = field(default_factory=set)

    # Transformation functions
    transformer_func: Optional[Callable[[Any], Any]] = None
    validator_func: Optional[Callable[[Any], bool]] = None

    # Rule metadata
    priority: int = 0
    description: str = ""
    created_by: str = "system"
    created_at: datetime = field(default_factory=datetime.now)
    is_active: bool = True


@dataclass
class MetadataSchema:
    """Schema definition for metadata structure and validation."""

    schema_name: str
    data_format: DataFormat
    required_fields: Set[str] = field(default_factory=set)
    optional_fields: Set[str] = field(default_factory=set)
    field_types: Dict[str, type] = field(default_factory=dict)
    field_constraints: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Schema versioning
    version: str = "1.0"
    compatibility_versions: List[str] = field(default_factory=list)

    # Validation options
    strict_validation: bool = False
    allow_extra_fields: bool = True


@dataclass
class MetadataLineage:
    """Tracks the lineage and history of metadata transformations."""

    original_source: str
    transformation_history: List[Dict[str, Any]] = field(default_factory=list)
    current_format: Optional[DataFormat] = None
    created_at: datetime = field(default_factory=datetime.now)
    last_modified: datetime = field(default_factory=datetime.now)

    def add_transformation(
        self,
        operation: str,
        source_format: DataFormat,
        target_format: DataFormat,
        adapter_id: str,
        metadata_changes: Dict[str, Any],
    ):
        """Add transformation record to lineage."""
        transformation = {
            "timestamp": datetime.now(),
            "operation": operation,
            "source_format": source_format.value,
            "target_format": target_format.value,
            "adapter_id": adapter_id,
            "metadata_changes": metadata_changes,
        }
        self.transformation_history.append(transformation)
        self.current_format = target_format
        self.last_modified = datetime.now()
