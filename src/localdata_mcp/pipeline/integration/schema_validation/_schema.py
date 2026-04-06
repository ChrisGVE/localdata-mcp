"""
DataSchema class definition.
"""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Set

from ..interfaces import DataFormat
from ._types import SchemaConstraint


@dataclass
class DataSchema:
    """Comprehensive data schema definition."""

    schema_id: str
    schema_name: str
    data_format: DataFormat
    schema_version: str = "1.0"

    # Core schema definition
    fields: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    constraints: List[SchemaConstraint] = field(default_factory=list)
    relationships: Dict[str, Any] = field(default_factory=dict)

    # Schema metadata
    created_at: datetime = field(default_factory=datetime.now)
    created_by: str = "system"
    description: str = ""
    tags: Set[str] = field(default_factory=set)

    # Schema properties
    is_strict: bool = False
    allow_extra_fields: bool = True
    nullable_by_default: bool = True

    # Compatibility information
    compatible_versions: List[str] = field(default_factory=list)
    deprecated_fields: Set[str] = field(default_factory=set)

    def get_schema_hash(self) -> str:
        """Generate a hash representing the schema structure."""
        schema_repr = json.dumps(
            {
                "fields": self.fields,
                "constraints": [
                    (c.constraint_id, c.constraint_type.value, c.field_name)
                    for c in self.constraints
                ],
                "relationships": self.relationships,
            },
            sort_keys=True,
        )
        return hashlib.sha256(schema_repr.encode()).hexdigest()[:16]
