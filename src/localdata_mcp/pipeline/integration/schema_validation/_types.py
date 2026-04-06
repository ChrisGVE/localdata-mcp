"""
Schema validation types: enums and dataclasses.
"""

from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

import pandas as pd


class SchemaValidationLevel(Enum):
    """Levels of schema validation strictness."""

    BASIC = "basic"  # Basic type and structure validation
    STANDARD = "standard"  # Standard validation with constraints
    STRICT = "strict"  # Strict validation with all rules
    CUSTOM = "custom"  # Custom validation with user-defined rules


class SchemaConformanceLevel(Enum):
    """Levels of schema conformance assessment."""

    PERFECT = "perfect"  # 100% conformance
    COMPATIBLE = "compatible"  # Compatible with minor issues
    DEGRADED = "degraded"  # Significant issues but usable
    INCOMPATIBLE = "incompatible"  # Cannot be used safely


class ValidationRuleType(Enum):
    """Types of validation rules."""

    TYPE_CHECK = "type_check"
    RANGE_CHECK = "range_check"
    NULL_CHECK = "null_check"
    UNIQUENESS_CHECK = "uniqueness_check"
    PATTERN_CHECK = "pattern_check"
    RELATIONSHIP_CHECK = "relationship_check"
    DOMAIN_SPECIFIC = "domain_specific"
    CUSTOM = "custom"


@dataclass
class SchemaConstraint:
    """Represents a single schema constraint."""

    constraint_id: str
    constraint_type: ValidationRuleType
    field_name: str
    constraint_value: Any
    description: str = ""
    is_required: bool = True
    error_message: str = ""

    # Constraint metadata
    created_at: datetime = field(default_factory=datetime.now)
    priority: int = 0  # Higher values = higher priority


@dataclass
class ValidationError:
    """Detailed validation error information."""

    error_id: str
    error_type: ValidationRuleType
    field_name: str
    expected_value: Any
    actual_value: Any
    error_message: str
    severity: str = "error"  # error, warning, info

    # Error context
    row_index: Optional[int] = None
    column_index: Optional[int] = None
    constraint_id: Optional[str] = None

    # Error metadata
    detected_at: datetime = field(default_factory=datetime.now)


@dataclass
class SchemaInferenceResult:
    """Result of schema inference operation."""

    inferred_schema: "DataSchema"  # noqa: F821
    confidence_score: float
    inference_details: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    # Inference metrics
    sample_size: int = 0
    inference_time: float = 0.0
    alternative_schemas: List[Tuple["DataSchema", float]]  # noqa: F821 = field(default_factory=list)

    # Quality assessment
    data_quality_score: float = 1.0
    completeness_score: float = 1.0
    consistency_score: float = 1.0


@dataclass
class SchemaValidationResult:
    """Comprehensive result of schema validation."""

    is_valid: bool
    conformance_level: SchemaConformanceLevel
    validation_score: float  # 0.0 to 1.0

    # Error details
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[ValidationError] = field(default_factory=list)

    # Validation statistics
    total_checks: int = 0
    passed_checks: int = 0
    failed_checks: int = 0

    # Performance metrics
    validation_time: float = 0.0
    memory_usage: float = 0.0

    # Additional context
    schema_version: str = "1.0"
    validation_level: SchemaValidationLevel = SchemaValidationLevel.STANDARD
    validation_metadata: Dict[str, Any] = field(default_factory=dict)
