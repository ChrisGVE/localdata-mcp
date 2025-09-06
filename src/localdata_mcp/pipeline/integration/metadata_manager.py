"""
Metadata Management System for Integration Shims Framework.

This module provides comprehensive metadata preservation, transformation, and 
validation capabilities to ensure context preservation throughout data conversions.

Key Features:
- MetadataManager for centralized metadata operations
- PreservationRule system for configurable metadata handling
- MetadataSchema validation and enforcement
- Cross-format metadata mapping and transformation
- Metadata lineage tracking and audit trails
- Integration with existing pipeline metadata systems
"""

import time
import logging
from typing import Any, Dict, List, Optional, Set, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
from datetime import datetime
import copy
import pandas as pd
import numpy as np

from .interfaces import DataFormat, MetadataPreserver, ValidationResult
from ...logging_manager import get_logger

logger = get_logger(__name__)


class PreservationStrategy(Enum):
    """Strategies for metadata preservation during conversion."""
    STRICT = "strict"           # Preserve all metadata exactly
    ADAPTIVE = "adaptive"       # Adapt metadata to target format
    MINIMAL = "minimal"         # Preserve only essential metadata
    CUSTOM = "custom"           # Use custom preservation rules


class MetadataType(Enum):
    """Types of metadata that can be preserved."""
    STRUCTURAL = "structural"   # Shape, dimensions, schema info
    SEMANTIC = "semantic"       # Column meanings, units, descriptions
    OPERATIONAL = "operational" # Creation time, source, processing history
    QUALITY = "quality"         # Completeness, consistency, accuracy metrics
    LINEAGE = "lineage"         # Data source and transformation history
    CUSTOM = "custom"           # Domain-specific metadata


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
    
    def add_transformation(self, 
                          operation: str,
                          source_format: DataFormat,
                          target_format: DataFormat,
                          adapter_id: str,
                          metadata_changes: Dict[str, Any]):
        """Add transformation record to lineage."""
        transformation = {
            'timestamp': datetime.now(),
            'operation': operation,
            'source_format': source_format.value,
            'target_format': target_format.value,
            'adapter_id': adapter_id,
            'metadata_changes': metadata_changes
        }
        self.transformation_history.append(transformation)
        self.current_format = target_format
        self.last_modified = datetime.now()


class MetadataTransformer(ABC):
    """Abstract base class for format-specific metadata transformers."""
    
    @abstractmethod
    def can_transform(self, 
                     source_format: DataFormat, 
                     target_format: DataFormat) -> bool:
        """Check if this transformer can handle the format conversion."""
        pass
    
    @abstractmethod
    def transform_metadata(self, 
                          metadata: Dict[str, Any],
                          source_format: DataFormat,
                          target_format: DataFormat) -> Dict[str, Any]:
        """Transform metadata between formats."""
        pass
    
    @abstractmethod
    def get_supported_metadata_types(self) -> Set[MetadataType]:
        """Get metadata types supported by this transformer."""
        pass


class PandasMetadataTransformer(MetadataTransformer):
    """Metadata transformer for pandas DataFrame formats."""
    
    def can_transform(self, source_format: DataFormat, target_format: DataFormat) -> bool:
        """Check if transformation is supported."""
        pandas_formats = {DataFormat.PANDAS_DATAFRAME, DataFormat.TIME_SERIES, DataFormat.CATEGORICAL}
        return source_format in pandas_formats or target_format in pandas_formats
    
    def transform_metadata(self, 
                          metadata: Dict[str, Any],
                          source_format: DataFormat,
                          target_format: DataFormat) -> Dict[str, Any]:
        """Transform metadata for pandas-related conversions."""
        transformed = metadata.copy()
        
        # Handle DataFrame-specific metadata
        if source_format == DataFormat.PANDAS_DATAFRAME:
            if target_format == DataFormat.NUMPY_ARRAY:
                # Convert DataFrame metadata to array metadata
                if 'columns' in transformed:
                    transformed['original_columns'] = transformed['columns']
                    del transformed['columns']
                if 'dtypes' in transformed:
                    transformed['original_dtypes'] = transformed['dtypes']
                    del transformed['dtypes']
                if 'shape' in transformed:
                    # Shape remains relevant for arrays
                    pass
        
        elif target_format == DataFormat.PANDAS_DATAFRAME:
            # Restore DataFrame metadata from other formats
            if 'original_columns' in transformed:
                transformed['columns'] = transformed['original_columns']
                del transformed['original_columns']
            if 'original_dtypes' in transformed:
                transformed['dtypes'] = transformed['original_dtypes']
                del transformed['original_dtypes']
        
        # Handle time series specific metadata
        if target_format == DataFormat.TIME_SERIES:
            transformed.update({
                'temporal_metadata': {
                    'is_time_series': True,
                    'conversion_timestamp': datetime.now().isoformat()
                }
            })
        
        return transformed
    
    def get_supported_metadata_types(self) -> Set[MetadataType]:
        """Get supported metadata types."""
        return {
            MetadataType.STRUCTURAL,
            MetadataType.SEMANTIC,
            MetadataType.OPERATIONAL,
            MetadataType.QUALITY
        }


class NumpyMetadataTransformer(MetadataTransformer):
    """Metadata transformer for numpy array formats."""
    
    def can_transform(self, source_format: DataFormat, target_format: DataFormat) -> bool:
        """Check if transformation is supported."""
        numpy_formats = {DataFormat.NUMPY_ARRAY}
        return source_format in numpy_formats or target_format in numpy_formats
    
    def transform_metadata(self, 
                          metadata: Dict[str, Any],
                          source_format: DataFormat,
                          target_format: DataFormat) -> Dict[str, Any]:
        """Transform metadata for numpy-related conversions."""
        transformed = metadata.copy()
        
        if source_format == DataFormat.NUMPY_ARRAY:
            if target_format == DataFormat.PANDAS_DATAFRAME:
                # Preserve array-specific metadata
                if 'shape' in transformed:
                    transformed['original_array_shape'] = transformed['shape']
                if 'dtype' in transformed:
                    transformed['original_array_dtype'] = transformed['dtype']
        
        elif target_format == DataFormat.NUMPY_ARRAY:
            # Add array-specific metadata
            transformed.update({
                'array_metadata': {
                    'conversion_source': source_format.value,
                    'conversion_timestamp': datetime.now().isoformat()
                }
            })
        
        return transformed
    
    def get_supported_metadata_types(self) -> Set[MetadataType]:
        """Get supported metadata types."""
        return {
            MetadataType.STRUCTURAL,
            MetadataType.OPERATIONAL,
            MetadataType.QUALITY
        }


class MetadataManager(MetadataPreserver):
    """
    Comprehensive metadata management system for integration shims.
    
    Handles metadata preservation, transformation, validation, and lineage
    tracking across all supported data formats and conversion operations.
    """
    
    def __init__(self,
                 default_strategy: PreservationStrategy = PreservationStrategy.ADAPTIVE,
                 enable_lineage_tracking: bool = True,
                 enable_validation: bool = True,
                 max_lineage_history: int = 100):
        """
        Initialize MetadataManager.
        
        Args:
            default_strategy: Default preservation strategy
            enable_lineage_tracking: Enable metadata lineage tracking
            enable_validation: Enable metadata validation
            max_lineage_history: Maximum lineage history entries to keep
        """
        self.default_strategy = default_strategy
        self.enable_lineage_tracking = enable_lineage_tracking
        self.enable_validation = enable_validation
        self.max_lineage_history = max_lineage_history
        
        # Internal state
        self._preservation_rules: Dict[str, PreservationRule] = {}
        self._metadata_schemas: Dict[DataFormat, MetadataSchema] = {}
        self._transformers: List[MetadataTransformer] = []
        self._lineage_tracking: Dict[str, MetadataLineage] = {}
        
        # Initialize default transformers
        self._initialize_default_transformers()
        
        # Initialize default schemas
        self._initialize_default_schemas()
        
        # Initialize default preservation rules
        self._initialize_default_rules()
        
        logger.info("MetadataManager initialized",
                   default_strategy=default_strategy.value,
                   enable_lineage=enable_lineage_tracking,
                   enable_validation=enable_validation)
    
    def extract_metadata(self, data: Any, format_type: DataFormat) -> Dict[str, Any]:
        """
        Extract metadata from data based on format type.
        
        Args:
            data: Data to extract metadata from
            format_type: Format of the data
            
        Returns:
            Extracted metadata dictionary
        """
        metadata = {
            'extraction_timestamp': datetime.now().isoformat(),
            'data_format': format_type.value,
            'extracted_by': 'MetadataManager'
        }
        
        # Format-specific extraction
        if format_type == DataFormat.PANDAS_DATAFRAME and isinstance(data, pd.DataFrame):
            metadata.update(self._extract_dataframe_metadata(data))
        elif format_type == DataFormat.NUMPY_ARRAY and isinstance(data, np.ndarray):
            metadata.update(self._extract_numpy_metadata(data))
        elif format_type == DataFormat.TIME_SERIES:
            metadata.update(self._extract_timeseries_metadata(data))
        else:
            # Generic extraction
            metadata.update(self._extract_generic_metadata(data))
        
        # Add lineage tracking
        if self.enable_lineage_tracking:
            lineage_id = self._generate_lineage_id(data)
            if lineage_id not in self._lineage_tracking:
                self._lineage_tracking[lineage_id] = MetadataLineage(
                    original_source=f"{format_type.value}_extraction",
                    current_format=format_type
                )
            metadata['lineage_id'] = lineage_id
        
        logger.debug(f"Extracted metadata for {format_type.value}",
                    metadata_keys=list(metadata.keys()))
        
        return metadata
    
    def apply_metadata(self, data: Any, 
                      metadata: Dict[str, Any], 
                      target_format: DataFormat) -> Any:
        """
        Apply metadata to converted data.
        
        Args:
            data: Converted data
            metadata: Metadata to apply
            target_format: Target format of the data
            
        Returns:
            Data with applied metadata
        """
        # Validate metadata if enabled
        if self.enable_validation and target_format in self._metadata_schemas:
            validation_result = self._validate_metadata(metadata, target_format)
            if not validation_result.is_valid:
                logger.warning("Metadata validation failed",
                             errors=validation_result.errors)
        
        # Apply format-specific metadata
        enhanced_data = self._apply_format_specific_metadata(data, metadata, target_format)
        
        logger.debug(f"Applied metadata to {target_format.value}",
                    applied_keys=list(metadata.keys()))
        
        return enhanced_data
    
    def merge_metadata(self, metadata_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Merge metadata from multiple sources.
        
        Args:
            metadata_list: List of metadata dictionaries to merge
            
        Returns:
            Merged metadata dictionary
        """
        if not metadata_list:
            return {}
        
        merged = {}
        merge_conflicts = []
        
        for metadata in metadata_list:
            for key, value in metadata.items():
                if key in merged:
                    if merged[key] != value:
                        # Handle merge conflict
                        conflict = {
                            'key': key,
                            'existing_value': merged[key],
                            'new_value': value,
                            'resolution': self._resolve_merge_conflict(key, merged[key], value)
                        }
                        merge_conflicts.append(conflict)
                        merged[key] = conflict['resolution']
                    # If values are the same, keep existing
                else:
                    merged[key] = value
        
        # Add merge information
        merged.update({
            'merge_timestamp': datetime.now().isoformat(),
            'source_count': len(metadata_list),
            'merge_conflicts': merge_conflicts,
            'merged_by': 'MetadataManager'
        })
        
        if merge_conflicts:
            logger.warning(f"Resolved {len(merge_conflicts)} merge conflicts during metadata merge")
        
        return merged
    
    def transform_metadata(self, 
                          metadata: Dict[str, Any],
                          source_format: DataFormat,
                          target_format: DataFormat,
                          adapter_id: str) -> Dict[str, Any]:
        """
        Transform metadata during format conversion.
        
        Args:
            metadata: Original metadata
            source_format: Source data format
            target_format: Target data format
            adapter_id: ID of the adapter performing conversion
            
        Returns:
            Transformed metadata
        """
        transformed = metadata.copy()
        
        # Find appropriate transformer
        transformer = self._find_transformer(source_format, target_format)
        if transformer:
            transformed = transformer.transform_metadata(transformed, source_format, target_format)
        
        # Apply preservation rules
        transformed = self._apply_preservation_rules(transformed, source_format, target_format)
        
        # Add transformation metadata
        transformation_info = {
            'transformation_timestamp': datetime.now().isoformat(),
            'source_format': source_format.value,
            'target_format': target_format.value,
            'adapter_id': adapter_id,
            'transformer_used': transformer.__class__.__name__ if transformer else None
        }
        
        if 'transformation_history' not in transformed:
            transformed['transformation_history'] = []
        transformed['transformation_history'].append(transformation_info)
        
        # Update lineage if tracking enabled
        if self.enable_lineage_tracking and 'lineage_id' in metadata:
            lineage_id = metadata['lineage_id']
            if lineage_id in self._lineage_tracking:
                self._lineage_tracking[lineage_id].add_transformation(
                    operation='metadata_transform',
                    source_format=source_format,
                    target_format=target_format,
                    adapter_id=adapter_id,
                    metadata_changes=self._calculate_metadata_changes(metadata, transformed)
                )
        
        logger.debug("Metadata transformed",
                    source_format=source_format.value,
                    target_format=target_format.value,
                    transformer=transformer.__class__.__name__ if transformer else "none")
        
        return transformed
    
    def add_preservation_rule(self, rule: PreservationRule) -> None:
        """Add a metadata preservation rule."""
        self._preservation_rules[rule.metadata_key] = rule
        logger.info(f"Added preservation rule for '{rule.metadata_key}'",
                   strategy=rule.preservation_strategy.value,
                   priority=rule.priority)
    
    def add_metadata_schema(self, schema: MetadataSchema) -> None:
        """Add a metadata schema for validation."""
        self._metadata_schemas[schema.data_format] = schema
        logger.info(f"Added metadata schema for {schema.data_format.value}",
                   required_fields=len(schema.required_fields),
                   optional_fields=len(schema.optional_fields))
    
    def add_transformer(self, transformer: MetadataTransformer) -> None:
        """Add a metadata transformer."""
        self._transformers.append(transformer)
        supported_types = transformer.get_supported_metadata_types()
        logger.info(f"Added metadata transformer {transformer.__class__.__name__}",
                   supported_types=[t.value for t in supported_types])
    
    def validate_metadata(self, 
                         metadata: Dict[str, Any], 
                         format_type: DataFormat) -> ValidationResult:
        """
        Validate metadata against schema.
        
        Args:
            metadata: Metadata to validate
            format_type: Expected format type
            
        Returns:
            ValidationResult
        """
        return self._validate_metadata(metadata, format_type)
    
    def get_lineage_info(self, lineage_id: str) -> Optional[MetadataLineage]:
        """Get lineage information for a specific lineage ID."""
        return self._lineage_tracking.get(lineage_id)
    
    # Private helper methods
    
    def _initialize_default_transformers(self):
        """Initialize default metadata transformers."""
        self.add_transformer(PandasMetadataTransformer())
        self.add_transformer(NumpyMetadataTransformer())
    
    def _initialize_default_schemas(self):
        """Initialize default metadata schemas."""
        # Pandas DataFrame schema
        pandas_schema = MetadataSchema(
            schema_name="pandas_dataframe",
            data_format=DataFormat.PANDAS_DATAFRAME,
            required_fields={'data_format', 'extraction_timestamp'},
            optional_fields={'shape', 'columns', 'dtypes', 'memory_usage', 'null_counts'},
            field_types={
                'shape': tuple,
                'columns': (list, dict),
                'dtypes': dict,
                'memory_usage': (int, float),
                'null_counts': dict
            }
        )
        self.add_metadata_schema(pandas_schema)
        
        # Numpy Array schema
        numpy_schema = MetadataSchema(
            schema_name="numpy_array",
            data_format=DataFormat.NUMPY_ARRAY,
            required_fields={'data_format', 'extraction_timestamp'},
            optional_fields={'shape', 'dtype', 'ndim', 'size', 'memory_bytes'},
            field_types={
                'shape': tuple,
                'dtype': str,
                'ndim': int,
                'size': int,
                'memory_bytes': int
            }
        )
        self.add_metadata_schema(numpy_schema)
    
    def _initialize_default_rules(self):
        """Initialize default preservation rules."""
        # High priority rule for structural metadata
        structural_rule = PreservationRule(
            metadata_key="structural_metadata",
            metadata_type=MetadataType.STRUCTURAL,
            preservation_strategy=PreservationStrategy.ADAPTIVE,
            priority=100,
            description="Preserve structural metadata with adaptation"
        )
        self.add_preservation_rule(structural_rule)
        
        # Medium priority rule for operational metadata
        operational_rule = PreservationRule(
            metadata_key="operational_metadata",
            metadata_type=MetadataType.OPERATIONAL,
            preservation_strategy=PreservationStrategy.STRICT,
            priority=50,
            description="Strictly preserve operational metadata"
        )
        self.add_preservation_rule(operational_rule)
    
    def _extract_dataframe_metadata(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Extract metadata from pandas DataFrame."""
        metadata = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum(),
            'null_counts': df.isnull().sum().to_dict(),
            'index_type': type(df.index).__name__
        }
        
        # Add additional DataFrame-specific metadata
        if hasattr(df.index, 'freq') and df.index.freq:
            metadata['index_frequency'] = str(df.index.freq)
        
        return metadata
    
    def _extract_numpy_metadata(self, arr: np.ndarray) -> Dict[str, Any]:
        """Extract metadata from numpy array."""
        return {
            'shape': arr.shape,
            'dtype': str(arr.dtype),
            'ndim': arr.ndim,
            'size': arr.size,
            'memory_bytes': arr.nbytes,
            'is_contiguous': arr.flags.c_contiguous,
            'is_writeable': arr.flags.writeable
        }
    
    def _extract_timeseries_metadata(self, data: Any) -> Dict[str, Any]:
        """Extract metadata from time series data."""
        metadata = {'is_time_series': True}
        
        if isinstance(data, pd.DataFrame):
            metadata.update(self._extract_dataframe_metadata(data))
            
            # Time series specific metadata
            if isinstance(data.index, pd.DatetimeIndex):
                metadata.update({
                    'temporal_range': (data.index.min(), data.index.max()),
                    'frequency': pd.infer_freq(data.index),
                    'has_regular_frequency': data.index.freq is not None
                })
        
        return metadata
    
    def _extract_generic_metadata(self, data: Any) -> Dict[str, Any]:
        """Extract generic metadata from any data type."""
        metadata = {
            'data_type': type(data).__name__,
            'python_type': str(type(data))
        }
        
        # Add size information if available
        if hasattr(data, '__len__'):
            metadata['length'] = len(data)
        
        if hasattr(data, 'size'):
            metadata['size'] = getattr(data, 'size')
        
        return metadata
    
    def _apply_format_specific_metadata(self, data: Any, 
                                      metadata: Dict[str, Any], 
                                      format_type: DataFormat) -> Any:
        """Apply format-specific metadata to data."""
        # For most formats, metadata is stored separately
        # Some formats might support embedded metadata
        
        if format_type == DataFormat.PANDAS_DATAFRAME and isinstance(data, pd.DataFrame):
            # For DataFrames, we can store some metadata as attributes
            if hasattr(data, 'attrs'):
                # Store compatible metadata in DataFrame.attrs (pandas >= 1.3)
                compatible_metadata = {
                    k: v for k, v in metadata.items()
                    if isinstance(v, (str, int, float, bool, list, dict))
                }
                data.attrs.update(compatible_metadata)
        
        return data
    
    def _find_transformer(self, 
                         source_format: DataFormat, 
                         target_format: DataFormat) -> Optional[MetadataTransformer]:
        """Find appropriate transformer for format conversion."""
        for transformer in self._transformers:
            if transformer.can_transform(source_format, target_format):
                return transformer
        return None
    
    def _apply_preservation_rules(self, 
                                 metadata: Dict[str, Any],
                                 source_format: DataFormat,
                                 target_format: DataFormat) -> Dict[str, Any]:
        """Apply preservation rules to metadata."""
        modified_metadata = metadata.copy()
        
        # Sort rules by priority (higher priority first)
        sorted_rules = sorted(self._preservation_rules.values(), 
                            key=lambda r: r.priority, reverse=True)
        
        for rule in sorted_rules:
            if not rule.is_active:
                continue
            
            # Check if rule applies to this conversion
            if (rule.source_formats and source_format not in rule.source_formats):
                continue
            if (rule.target_formats and target_format not in rule.target_formats):
                continue
            
            # Apply rule
            if rule.metadata_key in modified_metadata:
                if rule.preservation_strategy == PreservationStrategy.STRICT:
                    # Keep as is
                    pass
                elif rule.preservation_strategy == PreservationStrategy.MINIMAL:
                    # Remove if not essential
                    if rule.metadata_type not in {MetadataType.STRUCTURAL, MetadataType.OPERATIONAL}:
                        del modified_metadata[rule.metadata_key]
                elif rule.preservation_strategy == PreservationStrategy.CUSTOM and rule.transformer_func:
                    # Apply custom transformation
                    modified_metadata[rule.metadata_key] = rule.transformer_func(
                        modified_metadata[rule.metadata_key]
                    )
        
        return modified_metadata
    
    def _validate_metadata(self, 
                          metadata: Dict[str, Any], 
                          format_type: DataFormat) -> ValidationResult:
        """Validate metadata against schema."""
        if format_type not in self._metadata_schemas:
            return ValidationResult(
                is_valid=True,
                score=1.0,
                warnings=[f"No schema defined for {format_type.value}"]
            )
        
        schema = self._metadata_schemas[format_type]
        errors = []
        warnings = []
        
        # Check required fields
        for field in schema.required_fields:
            if field not in metadata:
                errors.append(f"Required field '{field}' missing")
        
        # Validate field types
        for field, expected_type in schema.field_types.items():
            if field in metadata:
                value = metadata[field]
                if not isinstance(value, expected_type):
                    if schema.strict_validation:
                        errors.append(f"Field '{field}' has incorrect type: expected {expected_type}, got {type(value)}")
                    else:
                        warnings.append(f"Field '{field}' has unexpected type: expected {expected_type}, got {type(value)}")
        
        # Check for extra fields
        if not schema.allow_extra_fields:
            allowed_fields = schema.required_fields | schema.optional_fields
            extra_fields = set(metadata.keys()) - allowed_fields
            if extra_fields:
                warnings.extend([f"Extra field '{field}' not in schema" for field in extra_fields])
        
        is_valid = len(errors) == 0
        score = 1.0 if is_valid else max(0.0, 1.0 - len(errors) * 0.2)
        
        return ValidationResult(
            is_valid=is_valid,
            score=score,
            errors=errors,
            warnings=warnings,
            details={
                'schema_name': schema.schema_name,
                'schema_version': schema.version,
                'validation_type': 'strict' if schema.strict_validation else 'lenient'
            }
        )
    
    def _resolve_merge_conflict(self, key: str, existing_value: Any, new_value: Any) -> Any:
        """Resolve conflict when merging metadata."""
        # Default resolution strategy - could be made configurable
        
        # For timestamps, prefer the more recent one
        if 'timestamp' in key.lower():
            try:
                existing_time = datetime.fromisoformat(existing_value.replace('Z', '+00:00'))
                new_time = datetime.fromisoformat(new_value.replace('Z', '+00:00'))
                return new_value if new_time > existing_time else existing_value
            except:
                pass
        
        # For lists, merge them
        if isinstance(existing_value, list) and isinstance(new_value, list):
            combined = existing_value + new_value
            return list(dict.fromkeys(combined))  # Remove duplicates while preserving order
        
        # For dicts, merge them
        if isinstance(existing_value, dict) and isinstance(new_value, dict):
            merged = existing_value.copy()
            merged.update(new_value)
            return merged
        
        # For numbers, take the average
        if isinstance(existing_value, (int, float)) and isinstance(new_value, (int, float)):
            return (existing_value + new_value) / 2
        
        # Default: prefer new value
        return new_value
    
    def _calculate_metadata_changes(self, 
                                   original: Dict[str, Any], 
                                   transformed: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate changes between original and transformed metadata."""
        changes = {
            'added_keys': set(transformed.keys()) - set(original.keys()),
            'removed_keys': set(original.keys()) - set(transformed.keys()),
            'modified_keys': []
        }
        
        for key in set(original.keys()) & set(transformed.keys()):
            if original[key] != transformed[key]:
                changes['modified_keys'].append({
                    'key': key,
                    'original_value': original[key],
                    'new_value': transformed[key]
                })
        
        return changes
    
    def _generate_lineage_id(self, data: Any) -> str:
        """Generate unique lineage ID for data."""
        import hashlib
        
        # Create hash based on data identity and timestamp
        components = [
            str(id(data)),
            str(type(data).__name__),
            str(datetime.now().timestamp())
        ]
        
        lineage_string = "_".join(components)
        return hashlib.md5(lineage_string.encode()).hexdigest()[:16]


# Utility functions for easy metadata operations

def create_preservation_rule(metadata_key: str,
                           strategy: PreservationStrategy = PreservationStrategy.ADAPTIVE,
                           metadata_type: MetadataType = MetadataType.SEMANTIC,
                           **kwargs) -> PreservationRule:
    """Create a metadata preservation rule."""
    return PreservationRule(
        metadata_key=metadata_key,
        metadata_type=metadata_type,
        preservation_strategy=strategy,
        **kwargs
    )


def create_metadata_schema(schema_name: str,
                         data_format: DataFormat,
                         required_fields: List[str] = None,
                         optional_fields: List[str] = None,
                         **kwargs) -> MetadataSchema:
    """Create a metadata schema."""
    return MetadataSchema(
        schema_name=schema_name,
        data_format=data_format,
        required_fields=set(required_fields or []),
        optional_fields=set(optional_fields or []),
        **kwargs
    )