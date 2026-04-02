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

import time
import logging
import hashlib
from typing import Any, Dict, List, Optional, Set, Union, Callable, Iterator, Tuple
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
from datetime import datetime, date
import json
import copy

import pandas as pd
import numpy as np
from scipy import sparse

from .interfaces import DataFormat, ValidationResult
from .metadata_manager import MetadataManager, MetadataSchema, PreservationStrategy
from .type_detection import TypeDetectionEngine, FormatDetectionResult, SchemaInfo
from ...logging_manager import get_logger

logger = get_logger(__name__)


class SchemaValidationLevel(Enum):
    """Levels of schema validation strictness."""
    BASIC = "basic"           # Basic type and structure validation
    STANDARD = "standard"     # Standard validation with constraints
    STRICT = "strict"         # Strict validation with all rules
    CUSTOM = "custom"         # Custom validation with user-defined rules


class SchemaConformanceLevel(Enum):
    """Levels of schema conformance assessment."""
    PERFECT = "perfect"       # 100% conformance
    COMPATIBLE = "compatible" # Compatible with minor issues
    DEGRADED = "degraded"     # Significant issues but usable
    INCOMPATIBLE = "incompatible" # Cannot be used safely


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
    
    inferred_schema: 'DataSchema'
    confidence_score: float
    inference_details: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    
    # Inference metrics
    sample_size: int = 0
    inference_time: float = 0.0
    alternative_schemas: List[Tuple['DataSchema', float]] = field(default_factory=list)
    
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
        schema_repr = json.dumps({
            'fields': self.fields,
            'constraints': [(c.constraint_id, c.constraint_type.value, c.field_name) for c in self.constraints],
            'relationships': self.relationships
        }, sort_keys=True)
        return hashlib.sha256(schema_repr.encode()).hexdigest()[:16]


class ValidationRule(ABC):
    """Abstract base class for validation rules."""
    
    def __init__(self, rule_id: str, rule_type: ValidationRuleType, 
                 description: str = ""):
        self.rule_id = rule_id
        self.rule_type = rule_type
        self.description = description
        self.is_enabled = True
        self.priority = 0
    
    @abstractmethod
    def validate(self, field_name: str, value: Any, context: Dict[str, Any]) -> List[ValidationError]:
        """Execute the validation rule."""
        pass
    
    @abstractmethod
    def is_applicable(self, field_name: str, field_schema: Dict[str, Any]) -> bool:
        """Check if this rule applies to the given field."""
        pass


class TypeValidationRule(ValidationRule):
    """Validates field data types."""
    
    def __init__(self, expected_type: Union[type, str], allow_coercion: bool = True):
        super().__init__("type_validation", ValidationRuleType.TYPE_CHECK, 
                        f"Type validation for {expected_type}")
        self.expected_type = expected_type
        self.allow_coercion = allow_coercion
    
    def validate(self, field_name: str, value: Any, context: Dict[str, Any]) -> List[ValidationError]:
        """Validate field type."""
        errors = []
        
        if pd.isna(value) and context.get('nullable', True):
            return errors
        
        # Type checking logic
        if isinstance(self.expected_type, str):
            expected_type_name = self.expected_type
            is_valid_type = self._check_string_type(value, expected_type_name)
        else:
            expected_type_name = self.expected_type.__name__
            is_valid_type = isinstance(value, self.expected_type)
        
        if not is_valid_type:
            if self.allow_coercion:
                try:
                    # Attempt type coercion
                    if self.expected_type in [int, float, str, bool]:
                        coerced_value = self.expected_type(value)
                        return errors  # Successful coercion
                except (ValueError, TypeError):
                    pass
            
            # Type validation failed
            error = ValidationError(
                error_id=f"type_error_{field_name}_{id(value)}",
                error_type=ValidationRuleType.TYPE_CHECK,
                field_name=field_name,
                expected_value=expected_type_name,
                actual_value=type(value).__name__,
                error_message=f"Expected {expected_type_name}, got {type(value).__name__}"
            )
            errors.append(error)
        
        return errors
    
    def is_applicable(self, field_name: str, field_schema: Dict[str, Any]) -> bool:
        """Check if type validation applies."""
        return 'type' in field_schema
    
    def _check_string_type(self, value: Any, type_name: str) -> bool:
        """Check type based on string type name."""
        type_mapping = {
            'int': int,
            'float': float,
            'str': str,
            'bool': bool,
            'datetime': (datetime, date, pd.Timestamp),
            'numeric': (int, float, np.number),
        }
        
        expected_types = type_mapping.get(type_name, str)
        if isinstance(expected_types, tuple):
            return isinstance(value, expected_types)
        else:
            return isinstance(value, expected_types)


class RangeValidationRule(ValidationRule):
    """Validates numeric range constraints."""
    
    def __init__(self, min_value: Optional[float] = None, max_value: Optional[float] = None,
                 inclusive: bool = True):
        super().__init__("range_validation", ValidationRuleType.RANGE_CHECK,
                        f"Range validation [{min_value}, {max_value}]")
        self.min_value = min_value
        self.max_value = max_value
        self.inclusive = inclusive
    
    def validate(self, field_name: str, value: Any, context: Dict[str, Any]) -> List[ValidationError]:
        """Validate numeric range."""
        errors = []
        
        if pd.isna(value):
            return errors
        
        try:
            numeric_value = float(value)
        except (ValueError, TypeError):
            return errors  # Not a numeric value, skip range validation
        
        # Check minimum value
        if self.min_value is not None:
            if self.inclusive and numeric_value < self.min_value:
                errors.append(ValidationError(
                    error_id=f"range_min_error_{field_name}_{id(value)}",
                    error_type=ValidationRuleType.RANGE_CHECK,
                    field_name=field_name,
                    expected_value=f">= {self.min_value}",
                    actual_value=numeric_value,
                    error_message=f"Value {numeric_value} is below minimum {self.min_value}"
                ))
            elif not self.inclusive and numeric_value <= self.min_value:
                errors.append(ValidationError(
                    error_id=f"range_min_error_{field_name}_{id(value)}",
                    error_type=ValidationRuleType.RANGE_CHECK,
                    field_name=field_name,
                    expected_value=f"> {self.min_value}",
                    actual_value=numeric_value,
                    error_message=f"Value {numeric_value} is not above minimum {self.min_value}"
                ))
        
        # Check maximum value
        if self.max_value is not None:
            if self.inclusive and numeric_value > self.max_value:
                errors.append(ValidationError(
                    error_id=f"range_max_error_{field_name}_{id(value)}",
                    error_type=ValidationRuleType.RANGE_CHECK,
                    field_name=field_name,
                    expected_value=f"<= {self.max_value}",
                    actual_value=numeric_value,
                    error_message=f"Value {numeric_value} is above maximum {self.max_value}"
                ))
            elif not self.inclusive and numeric_value >= self.max_value:
                errors.append(ValidationError(
                    error_id=f"range_max_error_{field_name}_{id(value)}",
                    error_type=ValidationRuleType.RANGE_CHECK,
                    field_name=field_name,
                    expected_value=f"< {self.max_value}",
                    actual_value=numeric_value,
                    error_message=f"Value {numeric_value} is not below maximum {self.max_value}"
                ))
        
        return errors
    
    def is_applicable(self, field_name: str, field_schema: Dict[str, Any]) -> bool:
        """Check if range validation applies."""
        field_type = field_schema.get('type', '')
        return field_type in ['int', 'float', 'numeric'] or 'min' in field_schema or 'max' in field_schema


class NullValidationRule(ValidationRule):
    """Validates null/missing value constraints."""
    
    def __init__(self, allow_nulls: bool = True):
        super().__init__("null_validation", ValidationRuleType.NULL_CHECK,
                        f"Null validation (allow_nulls={allow_nulls})")
        self.allow_nulls = allow_nulls
    
    def validate(self, field_name: str, value: Any, context: Dict[str, Any]) -> List[ValidationError]:
        """Validate null constraints."""
        errors = []
        
        is_null = pd.isna(value) or value is None
        
        if is_null and not self.allow_nulls:
            error = ValidationError(
                error_id=f"null_error_{field_name}_{id(value)}",
                error_type=ValidationRuleType.NULL_CHECK,
                field_name=field_name,
                expected_value="non-null value",
                actual_value="null/missing",
                error_message=f"Field '{field_name}' cannot be null"
            )
            errors.append(error)
        
        return errors
    
    def is_applicable(self, field_name: str, field_schema: Dict[str, Any]) -> bool:
        """Check if null validation applies."""
        return 'nullable' in field_schema or 'required' in field_schema


class SchemaInferenceEngine:
    """Engine for automatically inferring data schemas from samples."""
    
    def __init__(self, type_detection_engine: Optional[TypeDetectionEngine] = None):
        self.type_detection_engine = type_detection_engine or TypeDetectionEngine()
        self.inference_strategies = {
            DataFormat.PANDAS_DATAFRAME: self._infer_dataframe_schema,
            DataFormat.NUMPY_ARRAY: self._infer_numpy_schema,
            DataFormat.TIME_SERIES: self._infer_timeseries_schema,
            DataFormat.SCIPY_SPARSE: self._infer_sparse_schema,
        }
    
    def infer_schema(self, data: Any, data_format: Optional[DataFormat] = None,
                    sample_size: Optional[int] = None, 
                    confidence_threshold: float = 0.7) -> SchemaInferenceResult:
        """
        Infer schema from data sample with confidence scoring.
        
        Args:
            data: Data sample to analyze
            data_format: Optional explicit format hint
            sample_size: Maximum sample size for analysis
            confidence_threshold: Minimum confidence for schema acceptance
        
        Returns:
            SchemaInferenceResult with inferred schema and metrics
        """
        start_time = time.time()
        
        # Detect data format if not provided
        if data_format is None:
            format_result = self.type_detection_engine.detect_format(data)
            data_format = format_result.detected_format
        
        # Sample data if necessary
        sampled_data, actual_sample_size = self._sample_data(data, sample_size)
        
        # Infer schema using appropriate strategy
        inference_func = self.inference_strategies.get(data_format, self._infer_generic_schema)
        schema, confidence, details = inference_func(sampled_data)
        
        # Calculate quality metrics
        quality_metrics = self._calculate_quality_metrics(sampled_data, schema)
        
        inference_time = time.time() - start_time
        
        result = SchemaInferenceResult(
            inferred_schema=schema,
            confidence_score=confidence,
            inference_details=details,
            sample_size=actual_sample_size,
            inference_time=inference_time,
            data_quality_score=quality_metrics.get('data_quality', 1.0),
            completeness_score=quality_metrics.get('completeness', 1.0),
            consistency_score=quality_metrics.get('consistency', 1.0)
        )
        
        # Add warnings for low confidence
        if confidence < confidence_threshold:
            result.warnings.append(f"Schema inference confidence ({confidence:.2f}) below threshold ({confidence_threshold})")
        
        return result
    
    def _sample_data(self, data: Any, sample_size: Optional[int]) -> Tuple[Any, int]:
        """Sample data for schema inference."""
        if sample_size is None:
            return data, self._get_data_size(data)
        
        if isinstance(data, pd.DataFrame):
            if len(data) <= sample_size:
                return data, len(data)
            else:
                return data.sample(n=sample_size), sample_size
        elif isinstance(data, np.ndarray):
            if data.shape[0] <= sample_size:
                return data, data.shape[0]
            else:
                indices = np.random.choice(data.shape[0], sample_size, replace=False)
                return data[indices], sample_size
        else:
            return data, self._get_data_size(data)
    
    def _get_data_size(self, data: Any) -> int:
        """Get size of data sample."""
        if isinstance(data, (pd.DataFrame, pd.Series)):
            return len(data)
        elif isinstance(data, np.ndarray):
            return data.shape[0]
        elif isinstance(data, (list, tuple)):
            return len(data)
        else:
            return 1
    
    def _infer_dataframe_schema(self, data: pd.DataFrame) -> Tuple[DataSchema, float, Dict[str, Any]]:
        """Infer schema from pandas DataFrame."""
        schema_id = f"df_schema_{int(time.time())}"
        
        fields = {}
        constraints = []
        
        for col in data.columns:
            column_data = data[col]
            
            # Infer column type
            dtype = column_data.dtype
            if pd.api.types.is_numeric_dtype(dtype):
                if pd.api.types.is_integer_dtype(dtype):
                    field_type = 'int'
                else:
                    field_type = 'float'
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                field_type = 'datetime'
            elif pd.api.types.is_bool_dtype(dtype):
                field_type = 'bool'
            else:
                field_type = 'str'
            
            # Analyze column properties
            null_count = column_data.isnull().sum()
            total_count = len(column_data)
            nullable = null_count > 0
            
            field_schema = {
                'type': field_type,
                'nullable': nullable,
                'null_percentage': null_count / total_count if total_count > 0 else 0,
            }
            
            # Add constraints for numeric fields
            if field_type in ['int', 'float']:
                non_null_data = column_data.dropna()
                if len(non_null_data) > 0:
                    field_schema['min'] = float(non_null_data.min())
                    field_schema['max'] = float(non_null_data.max())
                    field_schema['mean'] = float(non_null_data.mean())
                    field_schema['std'] = float(non_null_data.std()) if len(non_null_data) > 1 else 0.0
                    
                    # Add range constraints
                    min_constraint = SchemaConstraint(
                        constraint_id=f"{col}_min",
                        constraint_type=ValidationRuleType.RANGE_CHECK,
                        field_name=col,
                        constraint_value=field_schema['min'],
                        description=f"Minimum value constraint for {col}",
                        is_required=False
                    )
                    constraints.append(min_constraint)
            
            # Add null constraints
            if not nullable:
                null_constraint = SchemaConstraint(
                    constraint_id=f"{col}_not_null",
                    constraint_type=ValidationRuleType.NULL_CHECK,
                    field_name=col,
                    constraint_value=False,
                    description=f"Not null constraint for {col}"
                )
                constraints.append(null_constraint)
            
            fields[col] = field_schema
        
        schema = DataSchema(
            schema_id=schema_id,
            schema_name="Inferred DataFrame Schema",
            data_format=DataFormat.PANDAS_DATAFRAME,
            fields=fields,
            constraints=constraints,
            description=f"Auto-inferred schema for DataFrame with {len(fields)} columns"
        )
        
        # Calculate confidence based on data quality
        confidence = self._calculate_dataframe_confidence(data, fields)
        
        details = {
            'column_count': len(fields),
            'row_count': len(data),
            'numeric_columns': sum(1 for f in fields.values() if f['type'] in ['int', 'float']),
            'categorical_columns': sum(1 for f in fields.values() if f['type'] == 'str'),
            'datetime_columns': sum(1 for f in fields.values() if f['type'] == 'datetime'),
        }
        
        return schema, confidence, details
    
    def _infer_numpy_schema(self, data: np.ndarray) -> Tuple[DataSchema, float, Dict[str, Any]]:
        """Infer schema from numpy array."""
        schema_id = f"np_schema_{int(time.time())}"
        
        # Determine array properties
        shape = data.shape
        dtype = data.dtype
        
        # Map numpy dtype to schema type
        if np.issubdtype(dtype, np.integer):
            element_type = 'int'
        elif np.issubdtype(dtype, np.floating):
            element_type = 'float'
        elif np.issubdtype(dtype, np.bool_):
            element_type = 'bool'
        elif np.issubdtype(dtype, np.datetime64):
            element_type = 'datetime'
        else:
            element_type = 'object'
        
        fields = {
            'array_data': {
                'type': element_type,
                'shape': shape,
                'ndim': data.ndim,
                'size': data.size,
                'nullable': False
            }
        }
        
        # Add statistical constraints for numeric data
        constraints = []
        if element_type in ['int', 'float']:
            flat_data = data.flatten()
            finite_data = flat_data[np.isfinite(flat_data)]
            
            if len(finite_data) > 0:
                min_val = float(finite_data.min())
                max_val = float(finite_data.max())
                
                constraints.append(SchemaConstraint(
                    constraint_id="array_min",
                    constraint_type=ValidationRuleType.RANGE_CHECK,
                    field_name="array_data",
                    constraint_value=min_val,
                    description="Minimum value constraint for array data"
                ))
        
        schema = DataSchema(
            schema_id=schema_id,
            schema_name="Inferred NumPy Array Schema",
            data_format=DataFormat.NUMPY_ARRAY,
            fields=fields,
            constraints=constraints,
            description=f"Auto-inferred schema for {element_type} array with shape {shape}"
        )
        
        confidence = 0.9 if element_type != 'object' else 0.7
        
        details = {
            'shape': shape,
            'dtype': str(dtype),
            'element_type': element_type,
            'total_elements': data.size,
            'memory_usage': data.nbytes
        }
        
        return schema, confidence, details
    
    def _infer_timeseries_schema(self, data: Any) -> Tuple[DataSchema, float, Dict[str, Any]]:
        """Infer schema from time series data."""
        # This is a placeholder implementation
        # In practice, this would analyze time series specific properties
        schema_id = f"ts_schema_{int(time.time())}"
        
        if isinstance(data, pd.DataFrame):
            return self._infer_dataframe_schema(data)
        elif isinstance(data, pd.Series):
            fields = {
                'timestamp': {'type': 'datetime', 'nullable': False},
                'value': {'type': 'float', 'nullable': True}
            }
            
            schema = DataSchema(
                schema_id=schema_id,
                schema_name="Inferred Time Series Schema",
                data_format=DataFormat.TIME_SERIES,
                fields=fields,
                description="Auto-inferred schema for time series data"
            )
            
            return schema, 0.8, {'series_length': len(data)}
        else:
            return self._infer_generic_schema(data)
    
    def _infer_sparse_schema(self, data: sparse.spmatrix) -> Tuple[DataSchema, float, Dict[str, Any]]:
        """Infer schema from scipy sparse matrix."""
        schema_id = f"sparse_schema_{int(time.time())}"
        
        fields = {
            'sparse_data': {
                'type': 'float',
                'shape': data.shape,
                'format': data.format,
                'nnz': data.nnz,
                'density': data.nnz / (data.shape[0] * data.shape[1]) if data.shape[0] * data.shape[1] > 0 else 0,
                'nullable': True
            }
        }
        
        schema = DataSchema(
            schema_id=schema_id,
            schema_name="Inferred Sparse Matrix Schema",
            data_format=DataFormat.SCIPY_SPARSE,
            fields=fields,
            description=f"Auto-inferred schema for sparse matrix ({data.format}, {data.shape})"
        )
        
        confidence = 0.85
        details = {
            'shape': data.shape,
            'format': data.format,
            'nnz': data.nnz,
            'density': fields['sparse_data']['density']
        }
        
        return schema, confidence, details
    
    def _infer_generic_schema(self, data: Any) -> Tuple[DataSchema, float, Dict[str, Any]]:
        """Fallback schema inference for unknown data types."""
        schema_id = f"generic_schema_{int(time.time())}"
        
        fields = {
            'data': {
                'type': 'object',
                'python_type': type(data).__name__,
                'nullable': True
            }
        }
        
        schema = DataSchema(
            schema_id=schema_id,
            schema_name="Generic Inferred Schema",
            data_format=DataFormat.UNKNOWN,
            fields=fields,
            description=f"Generic schema for {type(data).__name__} data"
        )
        
        confidence = 0.3  # Low confidence for generic schema
        details = {'data_type': type(data).__name__}
        
        return schema, confidence, details
    
    def _calculate_dataframe_confidence(self, data: pd.DataFrame, fields: Dict[str, Dict]) -> float:
        """Calculate confidence score for DataFrame schema inference."""
        if len(data) == 0:
            return 0.1
        
        # Base confidence
        confidence = 0.8
        
        # Adjust based on data quality indicators
        total_cells = len(data) * len(data.columns)
        null_cells = data.isnull().sum().sum()
        null_ratio = null_cells / total_cells if total_cells > 0 else 0
        
        # Lower confidence for high null ratios
        if null_ratio > 0.5:
            confidence -= 0.3
        elif null_ratio > 0.2:
            confidence -= 0.1
        
        # Adjust based on type consistency
        type_consistency_score = self._calculate_type_consistency(data)
        confidence += (type_consistency_score - 0.5) * 0.4
        
        return max(0.1, min(1.0, confidence))
    
    def _calculate_type_consistency(self, data: pd.DataFrame) -> float:
        """Calculate type consistency score for DataFrame."""
        if len(data.columns) == 0:
            return 1.0
        
        consistent_columns = 0
        for col in data.columns:
            column_data = data[col].dropna()
            if len(column_data) == 0:
                continue
            
            # Check if all non-null values have consistent types
            first_type = type(column_data.iloc[0])
            if all(isinstance(val, first_type) for val in column_data):
                consistent_columns += 1
        
        return consistent_columns / len(data.columns)
    
    def _calculate_quality_metrics(self, data: Any, schema: DataSchema) -> Dict[str, float]:
        """Calculate data quality metrics."""
        metrics = {
            'data_quality': 1.0,
            'completeness': 1.0,
            'consistency': 1.0
        }
        
        if isinstance(data, pd.DataFrame):
            # Calculate completeness (ratio of non-null values)
            total_cells = len(data) * len(data.columns)
            non_null_cells = total_cells - data.isnull().sum().sum()
            metrics['completeness'] = non_null_cells / total_cells if total_cells > 0 else 1.0
            
            # Calculate consistency (type consistency score)
            metrics['consistency'] = self._calculate_type_consistency(data)
            
            # Overall data quality combines completeness and consistency
            metrics['data_quality'] = (metrics['completeness'] + metrics['consistency']) / 2
        
        return metrics


class SchemaValidator:
    """Comprehensive data validation against schema definitions."""
    
    def __init__(self):
        self.validation_rules = {
            ValidationRuleType.TYPE_CHECK: TypeValidationRule,
            ValidationRuleType.RANGE_CHECK: RangeValidationRule,
            ValidationRuleType.NULL_CHECK: NullValidationRule,
        }
        self.custom_rules: List[ValidationRule] = []
    
    def add_custom_rule(self, rule: ValidationRule):
        """Add a custom validation rule."""
        self.custom_rules.append(rule)
        logger.info(f"Added custom validation rule: {rule.rule_id}")
    
    def validate_data(self, data: Any, schema: DataSchema, 
                     validation_level: SchemaValidationLevel = SchemaValidationLevel.STANDARD,
                     max_errors: Optional[int] = None) -> SchemaValidationResult:
        """
        Validate data against schema with comprehensive error reporting.
        
        Args:
            data: Data to validate
            schema: Schema definition to validate against
            validation_level: Level of validation strictness
            max_errors: Maximum number of errors to collect (None for unlimited)
        
        Returns:
            SchemaValidationResult with detailed validation results
        """
        start_time = time.time()
        
        errors = []
        warnings = []
        total_checks = 0
        passed_checks = 0
        
        try:
            # Validate based on data format
            if schema.data_format == DataFormat.PANDAS_DATAFRAME:
                errors, warnings, total_checks, passed_checks = self._validate_dataframe(
                    data, schema, validation_level, max_errors)
            elif schema.data_format == DataFormat.NUMPY_ARRAY:
                errors, warnings, total_checks, passed_checks = self._validate_numpy(
                    data, schema, validation_level, max_errors)
            else:
                errors, warnings, total_checks, passed_checks = self._validate_generic(
                    data, schema, validation_level, max_errors)
            
            # Determine validation result
            failed_checks = total_checks - passed_checks
            is_valid = len(errors) == 0
            validation_score = passed_checks / total_checks if total_checks > 0 else 1.0
            
            # Determine conformance level
            if validation_score >= 1.0:
                conformance = SchemaConformanceLevel.PERFECT
            elif validation_score >= 0.9:
                conformance = SchemaConformanceLevel.COMPATIBLE
            elif validation_score >= 0.7:
                conformance = SchemaConformanceLevel.DEGRADED
            else:
                conformance = SchemaConformanceLevel.INCOMPATIBLE
            
            validation_time = time.time() - start_time
            
            result = SchemaValidationResult(
                is_valid=is_valid,
                conformance_level=conformance,
                validation_score=validation_score,
                errors=errors,
                warnings=warnings,
                total_checks=total_checks,
                passed_checks=passed_checks,
                failed_checks=failed_checks,
                validation_time=validation_time,
                validation_level=validation_level,
                schema_version=schema.schema_version
            )
            
            logger.info(f"Schema validation completed: {validation_score:.2f} score, "
                       f"{len(errors)} errors, {len(warnings)} warnings")
            
            return result
            
        except Exception as e:
            logger.error(f"Schema validation failed with exception: {e}")
            
            # Return failed validation result
            return SchemaValidationResult(
                is_valid=False,
                conformance_level=SchemaConformanceLevel.INCOMPATIBLE,
                validation_score=0.0,
                errors=[ValidationError(
                    error_id="validation_exception",
                    error_type=ValidationRuleType.CUSTOM,
                    field_name="__validation__",
                    expected_value="successful validation",
                    actual_value="exception",
                    error_message=str(e),
                    severity="error"
                )],
                validation_time=time.time() - start_time,
                validation_level=validation_level
            )
    
    def _validate_dataframe(self, data: pd.DataFrame, schema: DataSchema, 
                          validation_level: SchemaValidationLevel,
                          max_errors: Optional[int]) -> Tuple[List[ValidationError], 
                                                           List[ValidationError], int, int]:
        """Validate pandas DataFrame against schema."""
        errors = []
        warnings = []
        total_checks = 0
        passed_checks = 0
        
        # Check if data is actually a DataFrame
        if not isinstance(data, pd.DataFrame):
            error = ValidationError(
                error_id="format_error",
                error_type=ValidationRuleType.TYPE_CHECK,
                field_name="__data__",
                expected_value="pandas.DataFrame",
                actual_value=type(data).__name__,
                error_message=f"Expected DataFrame, got {type(data).__name__}"
            )
            errors.append(error)
            return errors, warnings, 1, 0
        
        # Validate column presence
        schema_fields = set(schema.fields.keys())
        data_columns = set(data.columns)
        
        # Check for missing required columns
        missing_columns = schema_fields - data_columns
        for col in missing_columns:
            field_schema = schema.fields[col]
            if not field_schema.get('nullable', True):
                error = ValidationError(
                    error_id=f"missing_column_{col}",
                    error_type=ValidationRuleType.TYPE_CHECK,
                    field_name=col,
                    expected_value="column present",
                    actual_value="column missing",
                    error_message=f"Required column '{col}' is missing from data"
                )
                errors.append(error)
        
        # Check for extra columns (if strict mode)
        extra_columns = data_columns - schema_fields
        if extra_columns and schema.is_strict:
            for col in extra_columns:
                warning = ValidationError(
                    error_id=f"extra_column_{col}",
                    error_type=ValidationRuleType.TYPE_CHECK,
                    field_name=col,
                    expected_value="column not present",
                    actual_value="extra column",
                    error_message=f"Unexpected column '{col}' found in data",
                    severity="warning"
                )
                warnings.append(warning)
        
        # Validate each column
        for col in data.columns:
            if col not in schema.fields:
                continue  # Skip extra columns in non-strict mode
            
            field_schema = schema.fields[col]
            column_data = data[col]
            
            # Validate each value in the column
            for idx, value in column_data.items():
                if max_errors and len(errors) >= max_errors:
                    break
                
                total_checks += 1
                field_errors = self._validate_field_value(col, value, field_schema, {'row_index': idx})
                
                if field_errors:
                    errors.extend(field_errors)
                else:
                    passed_checks += 1
        
        # Validate schema constraints
        constraint_errors, constraint_warnings, constraint_total, constraint_passed = self._validate_constraints(
            data, schema, max_errors)
        
        errors.extend(constraint_errors)
        warnings.extend(constraint_warnings)
        total_checks += constraint_total
        passed_checks += constraint_passed
        
        return errors, warnings, total_checks, passed_checks
    
    def _validate_numpy(self, data: np.ndarray, schema: DataSchema,
                       validation_level: SchemaValidationLevel,
                       max_errors: Optional[int]) -> Tuple[List[ValidationError], 
                                                        List[ValidationError], int, int]:
        """Validate numpy array against schema."""
        errors = []
        warnings = []
        total_checks = 0
        passed_checks = 0
        
        # Check if data is actually a numpy array
        if not isinstance(data, np.ndarray):
            error = ValidationError(
                error_id="format_error",
                error_type=ValidationRuleType.TYPE_CHECK,
                field_name="__data__",
                expected_value="numpy.ndarray",
                actual_value=type(data).__name__,
                error_message=f"Expected numpy array, got {type(data).__name__}"
            )
            errors.append(error)
            return errors, warnings, 1, 0
        
        # Validate array properties from schema
        if 'array_data' in schema.fields:
            field_schema = schema.fields['array_data']
            
            # Check shape if specified
            expected_shape = field_schema.get('shape')
            if expected_shape and data.shape != expected_shape:
                error = ValidationError(
                    error_id="shape_mismatch",
                    error_type=ValidationRuleType.TYPE_CHECK,
                    field_name="shape",
                    expected_value=expected_shape,
                    actual_value=data.shape,
                    error_message=f"Shape mismatch: expected {expected_shape}, got {data.shape}"
                )
                errors.append(error)
            
            total_checks += 1
            if not errors:
                passed_checks += 1
        
        return errors, warnings, total_checks, passed_checks
    
    def _validate_generic(self, data: Any, schema: DataSchema,
                         validation_level: SchemaValidationLevel,
                         max_errors: Optional[int]) -> Tuple[List[ValidationError],
                                                          List[ValidationError], int, int]:
        """Validate generic data against schema."""
        errors = []
        warnings = []
        total_checks = 1
        passed_checks = 0
        
        # Basic type validation
        if schema.fields:
            first_field = next(iter(schema.fields.values()))
            expected_type = first_field.get('python_type', 'object')
            actual_type = type(data).__name__
            
            if expected_type != 'object' and actual_type != expected_type:
                error = ValidationError(
                    error_id="type_mismatch",
                    error_type=ValidationRuleType.TYPE_CHECK,
                    field_name="__data__",
                    expected_value=expected_type,
                    actual_value=actual_type,
                    error_message=f"Type mismatch: expected {expected_type}, got {actual_type}"
                )
                errors.append(error)
            else:
                passed_checks += 1
        else:
            passed_checks += 1  # No schema to validate against
        
        return errors, warnings, total_checks, passed_checks
    
    def _validate_field_value(self, field_name: str, value: Any, 
                             field_schema: Dict[str, Any], 
                             context: Dict[str, Any]) -> List[ValidationError]:
        """Validate a single field value against its schema definition."""
        errors = []
        
        # Type validation
        expected_type = field_schema.get('type')
        if expected_type:
            type_rule = TypeValidationRule(expected_type)
            type_errors = type_rule.validate(field_name, value, field_schema)
            errors.extend(type_errors)
        
        # Null validation
        nullable = field_schema.get('nullable', True)
        null_rule = NullValidationRule(allow_nulls=nullable)
        null_errors = null_rule.validate(field_name, value, field_schema)
        errors.extend(null_errors)
        
        # Range validation for numeric types
        if expected_type in ['int', 'float', 'numeric']:
            min_val = field_schema.get('min')
            max_val = field_schema.get('max')
            if min_val is not None or max_val is not None:
                range_rule = RangeValidationRule(min_val, max_val)
                range_errors = range_rule.validate(field_name, value, field_schema)
                errors.extend(range_errors)
        
        # Custom rule validation
        for rule in self.custom_rules:
            if rule.is_enabled and rule.is_applicable(field_name, field_schema):
                custom_errors = rule.validate(field_name, value, context)
                errors.extend(custom_errors)
        
        return errors
    
    def _validate_constraints(self, data: Any, schema: DataSchema,
                             max_errors: Optional[int]) -> Tuple[List[ValidationError],
                                                              List[ValidationError], int, int]:
        """Validate schema constraints."""
        errors = []
        warnings = []
        total_checks = len(schema.constraints)
        passed_checks = 0
        
        for constraint in schema.constraints:
            if max_errors and len(errors) >= max_errors:
                break
            
            try:
                constraint_errors = self._validate_single_constraint(data, constraint)
                if constraint_errors:
                    errors.extend(constraint_errors)
                else:
                    passed_checks += 1
            except Exception as e:
                error = ValidationError(
                    error_id=f"constraint_error_{constraint.constraint_id}",
                    error_type=constraint.constraint_type,
                    field_name=constraint.field_name,
                    expected_value="constraint validation",
                    actual_value="validation exception",
                    error_message=f"Constraint validation failed: {e}"
                )
                errors.append(error)
        
        return errors, warnings, total_checks, passed_checks
    
    def _validate_single_constraint(self, data: Any, constraint: SchemaConstraint) -> List[ValidationError]:
        """Validate a single schema constraint."""
        errors = []
        
        if constraint.constraint_type == ValidationRuleType.RANGE_CHECK:
            # Range constraint validation
            if isinstance(data, pd.DataFrame) and constraint.field_name in data.columns:
                column_data = data[constraint.field_name].dropna()
                min_value = constraint.constraint_value
                
                violations = column_data[column_data < min_value]
                for idx, value in violations.items():
                    error = ValidationError(
                        error_id=f"range_violation_{constraint.constraint_id}_{idx}",
                        error_type=ValidationRuleType.RANGE_CHECK,
                        field_name=constraint.field_name,
                        expected_value=f">= {min_value}",
                        actual_value=value,
                        error_message=constraint.error_message or f"Value {value} violates minimum constraint",
                        row_index=idx
                    )
                    errors.append(error)
        
        elif constraint.constraint_type == ValidationRuleType.NULL_CHECK:
            # Null constraint validation
            if isinstance(data, pd.DataFrame) and constraint.field_name in data.columns:
                column_data = data[constraint.field_name]
                null_values = column_data.isnull()
                
                if not constraint.constraint_value and null_values.any():  # constraint_value=False means no nulls allowed
                    null_indices = null_values[null_values].index
                    for idx in null_indices:
                        error = ValidationError(
                            error_id=f"null_violation_{constraint.constraint_id}_{idx}",
                            error_type=ValidationRuleType.NULL_CHECK,
                            field_name=constraint.field_name,
                            expected_value="non-null",
                            actual_value="null",
                            error_message=constraint.error_message or f"Null value not allowed in {constraint.field_name}",
                            row_index=idx
                        )
                        errors.append(error)
        
        return errors


class SchemaEvolutionManager:
    """Manages schema evolution, versioning, and compatibility analysis."""
    
    def __init__(self):
        self.schema_versions: Dict[str, List[DataSchema]] = {}
        self.compatibility_cache: Dict[str, Dict[str, float]] = {}
    
    def add_schema_version(self, schema: DataSchema):
        """Add a new version of a schema."""
        schema_name = schema.schema_name
        if schema_name not in self.schema_versions:
            self.schema_versions[schema_name] = []
        
        self.schema_versions[schema_name].append(schema)
        self.schema_versions[schema_name].sort(key=lambda s: s.created_at)
        
        logger.info(f"Added schema version {schema.schema_version} for {schema_name}")
    
    def get_latest_schema(self, schema_name: str) -> Optional[DataSchema]:
        """Get the latest version of a schema."""
        if schema_name in self.schema_versions and self.schema_versions[schema_name]:
            return self.schema_versions[schema_name][-1]
        return None
    
    def get_schema_version(self, schema_name: str, version: str) -> Optional[DataSchema]:
        """Get a specific version of a schema."""
        if schema_name in self.schema_versions:
            for schema in self.schema_versions[schema_name]:
                if schema.schema_version == version:
                    return schema
        return None
    
    def detect_schema_drift(self, current_data: Any, reference_schema: DataSchema,
                           drift_threshold: float = 0.1) -> Dict[str, Any]:
        """
        Detect schema drift by comparing current data against reference schema.
        
        Args:
            current_data: Current data sample
            reference_schema: Reference schema to compare against
            drift_threshold: Threshold for detecting significant drift
        
        Returns:
            Dictionary with drift analysis results
        """
        # Infer current schema from data
        inference_engine = SchemaInferenceEngine()
        current_result = inference_engine.infer_schema(current_data)
        current_schema = current_result.inferred_schema
        
        # Calculate compatibility score
        compatibility_score = self.calculate_compatibility_score(current_schema, reference_schema)
        
        # Detect specific changes
        field_changes = self._detect_field_changes(current_schema, reference_schema)
        constraint_changes = self._detect_constraint_changes(current_schema, reference_schema)
        
        # Determine drift level
        drift_detected = compatibility_score < (1.0 - drift_threshold)
        drift_level = self._categorize_drift_level(compatibility_score)
        
        drift_analysis = {
            'drift_detected': drift_detected,
            'drift_level': drift_level,
            'compatibility_score': compatibility_score,
            'current_schema': current_schema,
            'reference_schema': reference_schema,
            'field_changes': field_changes,
            'constraint_changes': constraint_changes,
            'analysis_timestamp': datetime.now(),
            'recommendations': self._generate_drift_recommendations(field_changes, constraint_changes)
        }
        
        return drift_analysis
    
    def calculate_compatibility_score(self, schema1: DataSchema, schema2: DataSchema) -> float:
        """Calculate compatibility score between two schemas (0.0 to 1.0)."""
        cache_key = f"{schema1.get_schema_hash()}_{schema2.get_schema_hash()}"
        if cache_key in self.compatibility_cache:
            return self.compatibility_cache[cache_key]
        
        score = 0.0
        total_weight = 0.0
        
        # Field compatibility (40% weight)
        field_score, field_weight = self._calculate_field_compatibility(schema1, schema2)
        score += field_score * field_weight
        total_weight += field_weight
        
        # Type compatibility (30% weight)
        type_score, type_weight = self._calculate_type_compatibility(schema1, schema2)
        score += type_score * type_weight
        total_weight += type_weight
        
        # Constraint compatibility (20% weight)
        constraint_score, constraint_weight = self._calculate_constraint_compatibility(schema1, schema2)
        score += constraint_score * constraint_weight
        total_weight += constraint_weight
        
        # Structure compatibility (10% weight)
        structure_score, structure_weight = self._calculate_structure_compatibility(schema1, schema2)
        score += structure_score * structure_weight
        total_weight += structure_weight
        
        final_score = score / total_weight if total_weight > 0 else 0.0
        
        # Cache the result
        self.compatibility_cache[cache_key] = final_score
        
        return final_score
    
    def _calculate_field_compatibility(self, schema1: DataSchema, schema2: DataSchema) -> Tuple[float, float]:
        """Calculate field-level compatibility score."""
        fields1 = set(schema1.fields.keys())
        fields2 = set(schema2.fields.keys())
        
        if not fields1 and not fields2:
            return 1.0, 0.4  # Both empty, perfect compatibility
        
        if not fields1 or not fields2:
            return 0.0, 0.4  # One empty, no compatibility
        
        # Calculate overlap
        common_fields = fields1 & fields2
        all_fields = fields1 | fields2
        
        overlap_score = len(common_fields) / len(all_fields) if all_fields else 0.0
        
        # Penalize missing required fields
        required_fields1 = {f for f, schema in schema1.fields.items() 
                          if not schema.get('nullable', True)}
        required_fields2 = {f for f, schema in schema2.fields.items() 
                          if not schema.get('nullable', True)}
        
        missing_required = (required_fields1 - fields2) | (required_fields2 - fields1)
        required_penalty = len(missing_required) * 0.2
        
        field_score = max(0.0, overlap_score - required_penalty)
        
        return field_score, 0.4
    
    def _calculate_type_compatibility(self, schema1: DataSchema, schema2: DataSchema) -> Tuple[float, float]:
        """Calculate type-level compatibility score."""
        common_fields = set(schema1.fields.keys()) & set(schema2.fields.keys())
        
        if not common_fields:
            return 1.0, 0.3  # No common fields to compare
        
        compatible_types = 0
        total_comparisons = len(common_fields)
        
        for field in common_fields:
            type1 = schema1.fields[field].get('type', 'object')
            type2 = schema2.fields[field].get('type', 'object')
            
            if self._are_types_compatible(type1, type2):
                compatible_types += 1
        
        type_score = compatible_types / total_comparisons if total_comparisons > 0 else 1.0
        
        return type_score, 0.3
    
    def _calculate_constraint_compatibility(self, schema1: DataSchema, schema2: DataSchema) -> Tuple[float, float]:
        """Calculate constraint-level compatibility score."""
        constraints1 = {(c.field_name, c.constraint_type): c for c in schema1.constraints}
        constraints2 = {(c.field_name, c.constraint_type): c for c in schema2.constraints}
        
        all_constraint_keys = set(constraints1.keys()) | set(constraints2.keys())
        
        if not all_constraint_keys:
            return 1.0, 0.2  # No constraints to compare
        
        compatible_constraints = 0
        
        for key in all_constraint_keys:
            if key in constraints1 and key in constraints2:
                # Both schemas have this constraint
                c1 = constraints1[key]
                c2 = constraints2[key]
                if self._are_constraints_compatible(c1, c2):
                    compatible_constraints += 1
            elif key not in constraints1 or key not in constraints2:
                # Constraint exists in only one schema - partial compatibility
                compatible_constraints += 0.5
        
        constraint_score = compatible_constraints / len(all_constraint_keys) if all_constraint_keys else 1.0
        
        return constraint_score, 0.2
    
    def _calculate_structure_compatibility(self, schema1: DataSchema, schema2: DataSchema) -> Tuple[float, float]:
        """Calculate structural compatibility score."""
        # Format compatibility
        format_compatible = schema1.data_format == schema2.data_format
        format_score = 1.0 if format_compatible else 0.5
        
        # Strictness compatibility
        strictness_compatible = schema1.is_strict == schema2.is_strict
        strictness_score = 1.0 if strictness_compatible else 0.8
        
        structure_score = (format_score + strictness_score) / 2
        
        return structure_score, 0.1
    
    def _are_types_compatible(self, type1: str, type2: str) -> bool:
        """Check if two types are compatible."""
        if type1 == type2:
            return True
        
        # Define compatible type groups
        numeric_types = {'int', 'float', 'numeric'}
        string_types = {'str', 'string', 'text'}
        
        if type1 in numeric_types and type2 in numeric_types:
            return True
        if type1 in string_types and type2 in string_types:
            return True
        
        # Special compatibility rules
        compatibility_map = {
            ('int', 'float'): True,
            ('float', 'int'): True,
            ('datetime', 'str'): True,  # Datetime can be serialized as string
            ('bool', 'int'): True,     # Boolean can be represented as integer
        }
        
        return compatibility_map.get((type1, type2), False)
    
    def _are_constraints_compatible(self, constraint1: SchemaConstraint, 
                                   constraint2: SchemaConstraint) -> bool:
        """Check if two constraints are compatible."""
        if constraint1.constraint_type != constraint2.constraint_type:
            return False
        
        # For range constraints, check if ranges overlap
        if constraint1.constraint_type == ValidationRuleType.RANGE_CHECK:
            # This is a simplified check - in practice you'd want more sophisticated logic
            return True  # Assume compatible for now
        
        # For null constraints, check if both allow/disallow nulls
        if constraint1.constraint_type == ValidationRuleType.NULL_CHECK:
            return constraint1.constraint_value == constraint2.constraint_value
        
        return True
    
    def _detect_field_changes(self, current_schema: DataSchema, 
                             reference_schema: DataSchema) -> Dict[str, Any]:
        """Detect changes in schema fields."""
        current_fields = set(current_schema.fields.keys())
        reference_fields = set(reference_schema.fields.keys())
        
        added_fields = current_fields - reference_fields
        removed_fields = reference_fields - current_fields
        common_fields = current_fields & reference_fields
        
        modified_fields = []
        for field in common_fields:
            current_field = current_schema.fields[field]
            reference_field = reference_schema.fields[field]
            
            if current_field != reference_field:
                changes = self._detect_field_property_changes(current_field, reference_field)
                if changes:
                    modified_fields.append({
                        'field_name': field,
                        'changes': changes
                    })
        
        return {
            'added_fields': list(added_fields),
            'removed_fields': list(removed_fields),
            'modified_fields': modified_fields,
            'total_changes': len(added_fields) + len(removed_fields) + len(modified_fields)
        }
    
    def _detect_field_property_changes(self, current_field: Dict[str, Any], 
                                      reference_field: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect changes in individual field properties."""
        changes = []
        
        all_props = set(current_field.keys()) | set(reference_field.keys())
        
        for prop in all_props:
            current_val = current_field.get(prop)
            reference_val = reference_field.get(prop)
            
            if current_val != reference_val:
                changes.append({
                    'property': prop,
                    'old_value': reference_val,
                    'new_value': current_val
                })
        
        return changes
    
    def _detect_constraint_changes(self, current_schema: DataSchema, 
                                  reference_schema: DataSchema) -> Dict[str, Any]:
        """Detect changes in schema constraints."""
        current_constraints = {c.constraint_id: c for c in current_schema.constraints}
        reference_constraints = {c.constraint_id: c for c in reference_schema.constraints}
        
        added_constraints = set(current_constraints.keys()) - set(reference_constraints.keys())
        removed_constraints = set(reference_constraints.keys()) - set(current_constraints.keys())
        
        modified_constraints = []
        common_constraints = set(current_constraints.keys()) & set(reference_constraints.keys())
        
        for constraint_id in common_constraints:
            current_constraint = current_constraints[constraint_id]
            reference_constraint = reference_constraints[constraint_id]
            
            if current_constraint.constraint_value != reference_constraint.constraint_value:
                modified_constraints.append({
                    'constraint_id': constraint_id,
                    'old_value': reference_constraint.constraint_value,
                    'new_value': current_constraint.constraint_value
                })
        
        return {
            'added_constraints': list(added_constraints),
            'removed_constraints': list(removed_constraints),
            'modified_constraints': modified_constraints,
            'total_changes': len(added_constraints) + len(removed_constraints) + len(modified_constraints)
        }
    
    def _categorize_drift_level(self, compatibility_score: float) -> str:
        """Categorize the level of schema drift."""
        if compatibility_score >= 0.95:
            return "minimal"
        elif compatibility_score >= 0.8:
            return "moderate"
        elif compatibility_score >= 0.6:
            return "significant"
        else:
            return "major"
    
    def _generate_drift_recommendations(self, field_changes: Dict[str, Any], 
                                       constraint_changes: Dict[str, Any]) -> List[str]:
        """Generate recommendations for handling schema drift."""
        recommendations = []
        
        if field_changes['added_fields']:
            recommendations.append(f"Consider adding default values for new fields: {', '.join(field_changes['added_fields'])}")
        
        if field_changes['removed_fields']:
            recommendations.append(f"Archive or migrate data for removed fields: {', '.join(field_changes['removed_fields'])}")
        
        if field_changes['modified_fields']:
            recommendations.append(f"Review type changes for modified fields - may require data transformation")
        
        if constraint_changes['total_changes'] > 0:
            recommendations.append("Review constraint changes to ensure data quality requirements are maintained")
        
        if not recommendations:
            recommendations.append("Schema compatibility is high - minimal migration effort required")
        
        return recommendations


# Export all public classes and functions
__all__ = [
    'SchemaValidationLevel',
    'SchemaConformanceLevel',
    'ValidationRuleType',
    'SchemaConstraint',
    'ValidationError',
    'SchemaInferenceResult',
    'SchemaValidationResult',
    'DataSchema',
    'ValidationRule',
    'TypeValidationRule',
    'RangeValidationRule',
    'NullValidationRule',
    'SchemaInferenceEngine',
    'SchemaValidator',
    'SchemaEvolutionManager'
]