"""
Schema Validation System for LocalData MCP Integration Shims Framework.

This module provides comprehensive schema inference, validation, and evolution
capabilities to ensure data integrity throughout the integration framework.

Key Features:
- SchemaInference: Automatic schema detection with confidence scoring
- SchemaValidator: Data integrity checking with detailed error reporting
- SchemaEvolution: Schema change management and compatibility analysis
- ValidationRuleEngine: Extensible custom validation logic framework
- Schema format support: pandas, numpy, time series, and domain-specific schemas

Core Components:
1. SchemaInference Engine - Detects schemas from data samples automatically
2. SchemaValidator Classes - Validates data against inferred/provided schemas
3. SchemaEvolution System - Manages schema changes and compatibility
4. ValidationRule Framework - Extensible validation logic system

Integration with existing framework:
- Extends MetadataManager schema functionality
- Integrates with TypeDetectionEngine for type inference
- Uses existing ValidationResult patterns
- Supports streaming and memory-efficient validation
"""

from ._evolution import SchemaEvolutionManager
from ._inference import SchemaInferenceEngine
from ._rules import (
    NullValidationRule,
    RangeValidationRule,
    TypeValidationRule,
    ValidationRule,
)
from ._schema import DataSchema
from ._types import (
    SchemaConformanceLevel,
    SchemaConstraint,
    SchemaInferenceResult,
    SchemaValidationLevel,
    SchemaValidationResult,
    ValidationError,
    ValidationRuleType,
)
from ._validator import SchemaValidator

__all__ = [
    "SchemaValidationLevel",
    "SchemaConformanceLevel",
    "ValidationRuleType",
    "SchemaConstraint",
    "ValidationError",
    "SchemaInferenceResult",
    "SchemaValidationResult",
    "DataSchema",
    "ValidationRule",
    "TypeValidationRule",
    "RangeValidationRule",
    "NullValidationRule",
    "SchemaInferenceEngine",
    "SchemaValidator",
    "SchemaEvolutionManager",
]
