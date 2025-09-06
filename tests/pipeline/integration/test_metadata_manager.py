"""
Unit tests for metadata management system in the Integration Shims Framework.

Tests cover:
- MetadataManager with preservation, transformation, and validation
- PreservationRule system for configurable metadata handling
- MetadataSchema validation and enforcement
- Format-specific metadata transformers
- MetadataLineage tracking with transformation history
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
import time

from src.localdata_mcp.pipeline.integration.metadata_manager import (
    MetadataManager,
    PreservationRule,
    MetadataSchema,
    PreservationStrategy,
    MetadataType,
    MetadataLineage,
    PandasMetadataTransformer,
    NumpyMetadataTransformer,
    create_preservation_rule,
    create_metadata_schema
)
from src.localdata_mcp.pipeline.integration.interfaces import (
    DataFormat,
    ValidationResult
)


class TestPreservationRule:
    """Test cases for PreservationRule dataclass."""
    
    def test_preservation_rule_creation(self):
        """Test PreservationRule creation."""
        rule = PreservationRule(
            metadata_key="test_key",
            metadata_type=MetadataType.SEMANTIC,
            preservation_strategy=PreservationStrategy.STRICT,
            priority=100,
            description="Test rule"
        )
        
        assert rule.metadata_key == "test_key"
        assert rule.metadata_type == MetadataType.SEMANTIC
        assert rule.preservation_strategy == PreservationStrategy.STRICT
        assert rule.priority == 100
        assert rule.is_active is True
        assert isinstance(rule.created_at, datetime)
    
    def test_preservation_rule_with_formats(self):
        """Test PreservationRule with format constraints."""
        rule = PreservationRule(
            metadata_key="format_specific",
            metadata_type=MetadataType.STRUCTURAL,
            preservation_strategy=PreservationStrategy.ADAPTIVE,
            source_formats={DataFormat.PANDAS_DATAFRAME},
            target_formats={DataFormat.NUMPY_ARRAY}
        )
        
        assert DataFormat.PANDAS_DATAFRAME in rule.source_formats
        assert DataFormat.NUMPY_ARRAY in rule.target_formats


class TestMetadataSchema:
    """Test cases for MetadataSchema dataclass."""
    
    def test_metadata_schema_creation(self):
        """Test MetadataSchema creation."""
        schema = MetadataSchema(
            schema_name="test_schema",
            data_format=DataFormat.PANDAS_DATAFRAME,
            required_fields={'field1', 'field2'},
            optional_fields={'field3', 'field4'},
            field_types={'field1': str, 'field2': int}
        )
        
        assert schema.schema_name == "test_schema"
        assert schema.data_format == DataFormat.PANDAS_DATAFRAME
        assert 'field1' in schema.required_fields
        assert 'field3' in schema.optional_fields
        assert schema.field_types['field1'] == str
        assert schema.version == "1.0"
    
    def test_metadata_schema_validation_options(self):
        """Test MetadataSchema validation options."""
        strict_schema = MetadataSchema(
            schema_name="strict_schema",
            data_format=DataFormat.NUMPY_ARRAY,
            strict_validation=True,
            allow_extra_fields=False
        )
        
        assert strict_schema.strict_validation is True
        assert strict_schema.allow_extra_fields is False


class TestMetadataLineage:
    """Test cases for MetadataLineage dataclass."""
    
    def test_lineage_creation(self):
        """Test MetadataLineage creation."""
        lineage = MetadataLineage(
            original_source="test_source",
            current_format=DataFormat.PANDAS_DATAFRAME
        )
        
        assert lineage.original_source == "test_source"
        assert lineage.current_format == DataFormat.PANDAS_DATAFRAME
        assert isinstance(lineage.created_at, datetime)
        assert len(lineage.transformation_history) == 0
    
    def test_add_transformation(self):
        """Test adding transformation to lineage."""
        lineage = MetadataLineage(original_source="test")
        
        lineage.add_transformation(
            operation="convert",
            source_format=DataFormat.PANDAS_DATAFRAME,
            target_format=DataFormat.NUMPY_ARRAY,
            adapter_id="test_adapter",
            metadata_changes={'added_keys': ['new_field']}
        )
        
        assert len(lineage.transformation_history) == 1
        transformation = lineage.transformation_history[0]
        assert transformation['operation'] == 'convert'
        assert transformation['adapter_id'] == 'test_adapter'
        assert transformation['source_format'] == DataFormat.PANDAS_DATAFRAME.value
        assert lineage.current_format == DataFormat.NUMPY_ARRAY


class TestPandasMetadataTransformer:
    """Test cases for PandasMetadataTransformer."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.transformer = PandasMetadataTransformer()
    
    def test_can_transform_pandas_formats(self):
        """Test can_transform for pandas-related formats."""
        # DataFrame to array
        assert self.transformer.can_transform(
            DataFormat.PANDAS_DATAFRAME, 
            DataFormat.NUMPY_ARRAY
        ) is True
        
        # Array to DataFrame
        assert self.transformer.can_transform(
            DataFormat.NUMPY_ARRAY,
            DataFormat.PANDAS_DATAFRAME
        ) is True
        
        # Time series
        assert self.transformer.can_transform(
            DataFormat.PANDAS_DATAFRAME,
            DataFormat.TIME_SERIES
        ) is True
        
        # Unsupported
        assert self.transformer.can_transform(
            DataFormat.JSON,
            DataFormat.XML
        ) is False
    
    def test_transform_dataframe_to_array(self):
        """Test transforming DataFrame metadata to array metadata."""
        original_metadata = {
            'data_format': 'pandas_dataframe',
            'columns': ['A', 'B', 'C'],
            'dtypes': {'A': 'int64', 'B': 'float64', 'C': 'object'},
            'shape': (100, 3)
        }
        
        transformed = self.transformer.transform_metadata(
            original_metadata,
            DataFormat.PANDAS_DATAFRAME,
            DataFormat.NUMPY_ARRAY
        )
        
        assert 'original_columns' in transformed
        assert 'original_dtypes' in transformed
        assert 'columns' not in transformed
        assert 'dtypes' not in transformed
        assert transformed['original_columns'] == ['A', 'B', 'C']
    
    def test_transform_array_to_dataframe(self):
        """Test transforming array metadata to DataFrame metadata."""
        original_metadata = {
            'data_format': 'numpy_array',
            'original_columns': ['A', 'B'],
            'original_dtypes': {'A': 'int64', 'B': 'float64'},
            'shape': (100, 2)
        }
        
        transformed = self.transformer.transform_metadata(
            original_metadata,
            DataFormat.NUMPY_ARRAY,
            DataFormat.PANDAS_DATAFRAME
        )
        
        assert 'columns' in transformed
        assert 'dtypes' in transformed
        assert 'original_columns' not in transformed
        assert 'original_dtypes' not in transformed
        assert transformed['columns'] == ['A', 'B']
    
    def test_transform_to_time_series(self):
        """Test transforming to time series metadata."""
        original_metadata = {
            'data_format': 'pandas_dataframe',
            'shape': (100, 2)
        }
        
        transformed = self.transformer.transform_metadata(
            original_metadata,
            DataFormat.PANDAS_DATAFRAME,
            DataFormat.TIME_SERIES
        )
        
        assert 'temporal_metadata' in transformed
        assert transformed['temporal_metadata']['is_time_series'] is True
    
    def test_supported_metadata_types(self):
        """Test supported metadata types."""
        supported_types = self.transformer.get_supported_metadata_types()
        
        assert MetadataType.STRUCTURAL in supported_types
        assert MetadataType.SEMANTIC in supported_types
        assert MetadataType.OPERATIONAL in supported_types
        assert MetadataType.QUALITY in supported_types


class TestNumpyMetadataTransformer:
    """Test cases for NumpyMetadataTransformer."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.transformer = NumpyMetadataTransformer()
    
    def test_can_transform_numpy_formats(self):
        """Test can_transform for numpy-related formats."""
        assert self.transformer.can_transform(
            DataFormat.NUMPY_ARRAY,
            DataFormat.PANDAS_DATAFRAME
        ) is True
        
        assert self.transformer.can_transform(
            DataFormat.PANDAS_DATAFRAME,
            DataFormat.NUMPY_ARRAY
        ) is True
        
        assert self.transformer.can_transform(
            DataFormat.JSON,
            DataFormat.CSV
        ) is False
    
    def test_transform_array_metadata(self):
        """Test transforming array metadata."""
        original_metadata = {
            'data_format': 'numpy_array',
            'shape': (100, 3),
            'dtype': 'float64'
        }
        
        transformed = self.transformer.transform_metadata(
            original_metadata,
            DataFormat.NUMPY_ARRAY,
            DataFormat.PANDAS_DATAFRAME
        )
        
        assert 'original_array_shape' in transformed
        assert 'original_array_dtype' in transformed
        assert transformed['original_array_shape'] == (100, 3)
    
    def test_transform_to_array(self):
        """Test transforming to array format."""
        original_metadata = {
            'data_format': 'pandas_dataframe',
            'shape': (100, 2)
        }
        
        transformed = self.transformer.transform_metadata(
            original_metadata,
            DataFormat.PANDAS_DATAFRAME,
            DataFormat.NUMPY_ARRAY
        )
        
        assert 'array_metadata' in transformed
        assert transformed['array_metadata']['conversion_source'] == 'pandas_dataframe'
    
    def test_supported_metadata_types(self):
        """Test supported metadata types."""
        supported_types = self.transformer.get_supported_metadata_types()
        
        assert MetadataType.STRUCTURAL in supported_types
        assert MetadataType.OPERATIONAL in supported_types
        assert MetadataType.QUALITY in supported_types


class TestMetadataManager:
    """Test cases for MetadataManager."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.manager = MetadataManager(
            default_strategy=PreservationStrategy.ADAPTIVE,
            enable_lineage_tracking=True,
            enable_validation=True
        )
        
        self.sample_df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [1.1, 2.2, 3.3, 4.4, 5.5],
            'C': ['a', 'b', 'c', 'd', 'e']
        })
        
        self.sample_array = np.array([[1, 2], [3, 4], [5, 6]])
    
    def test_manager_initialization(self):
        """Test MetadataManager initialization."""
        assert self.manager.default_strategy == PreservationStrategy.ADAPTIVE
        assert self.manager.enable_lineage_tracking is True
        assert self.manager.enable_validation is True
        assert len(self.manager._transformers) >= 2  # At least pandas and numpy transformers
        assert len(self.manager._metadata_schemas) >= 2  # At least pandas and numpy schemas
    
    def test_extract_dataframe_metadata(self):
        """Test extracting metadata from DataFrame."""
        metadata = self.manager.extract_metadata(self.sample_df, DataFormat.PANDAS_DATAFRAME)
        
        assert 'extraction_timestamp' in metadata
        assert 'data_format' in metadata
        assert 'shape' in metadata
        assert 'columns' in metadata
        assert 'dtypes' in metadata
        assert metadata['data_format'] == DataFormat.PANDAS_DATAFRAME.value
        assert metadata['shape'] == (5, 3)
    
    def test_extract_numpy_metadata(self):
        """Test extracting metadata from numpy array."""
        metadata = self.manager.extract_metadata(self.sample_array, DataFormat.NUMPY_ARRAY)
        
        assert 'extraction_timestamp' in metadata
        assert 'data_format' in metadata
        assert 'shape' in metadata
        assert 'dtype' in metadata
        assert 'ndim' in metadata
        assert metadata['data_format'] == DataFormat.NUMPY_ARRAY.value
        assert metadata['shape'] == (3, 2)
    
    def test_extract_with_lineage_tracking(self):
        """Test metadata extraction with lineage tracking."""
        metadata = self.manager.extract_metadata(self.sample_df, DataFormat.PANDAS_DATAFRAME)
        
        assert 'lineage_id' in metadata
        lineage_id = metadata['lineage_id']
        assert lineage_id in self.manager._lineage_tracking
        
        lineage = self.manager._lineage_tracking[lineage_id]
        assert lineage.current_format == DataFormat.PANDAS_DATAFRAME
    
    def test_apply_metadata_to_data(self):
        """Test applying metadata to converted data."""
        metadata = {
            'data_format': 'numpy_array',
            'shape': (3, 2),
            'dtype': 'int64'
        }
        
        enhanced_data = self.manager.apply_metadata(
            self.sample_array,
            metadata,
            DataFormat.NUMPY_ARRAY
        )
        
        # For numpy arrays, metadata is typically stored separately
        # The data itself should remain unchanged
        np.testing.assert_array_equal(enhanced_data, self.sample_array)
    
    def test_merge_metadata_without_conflicts(self):
        """Test merging metadata without conflicts."""
        metadata1 = {
            'source': 'test1',
            'timestamp': '2023-01-01',
            'shape': (10, 2)
        }
        
        metadata2 = {
            'processing': 'normalization',
            'version': '1.0',
            'quality_score': 0.95
        }
        
        merged = self.manager.merge_metadata([metadata1, metadata2])
        
        assert 'source' in merged
        assert 'processing' in merged
        assert 'merge_timestamp' in merged
        assert merged['source_count'] == 2
        assert len(merged['merge_conflicts']) == 0
    
    def test_merge_metadata_with_conflicts(self):
        """Test merging metadata with conflicts."""
        metadata1 = {
            'version': '1.0',
            'timestamp': '2023-01-01T10:00:00'
        }
        
        metadata2 = {
            'version': '1.1',
            'timestamp': '2023-01-01T11:00:00'  # More recent
        }
        
        merged = self.manager.merge_metadata([metadata1, metadata2])
        
        assert len(merged['merge_conflicts']) == 2  # version and timestamp conflicts
        assert merged['timestamp'] == '2023-01-01T11:00:00'  # Should prefer more recent
    
    def test_transform_metadata_with_transformer(self):
        """Test metadata transformation with appropriate transformer."""
        original_metadata = {
            'data_format': 'pandas_dataframe',
            'columns': ['A', 'B'],
            'shape': (100, 2)
        }
        
        transformed = self.manager.transform_metadata(
            original_metadata,
            DataFormat.PANDAS_DATAFRAME,
            DataFormat.NUMPY_ARRAY,
            'test_adapter'
        )
        
        assert 'original_columns' in transformed  # Pandas transformer should have run
        assert 'transformation_history' in transformed
        assert len(transformed['transformation_history']) == 1
    
    def test_transform_metadata_with_lineage(self):
        """Test metadata transformation with lineage tracking."""
        original_metadata = {
            'lineage_id': 'test_lineage',
            'data_format': 'pandas_dataframe'
        }
        
        # Create lineage entry
        self.manager._lineage_tracking['test_lineage'] = MetadataLineage(
            original_source='test',
            current_format=DataFormat.PANDAS_DATAFRAME
        )
        
        transformed = self.manager.transform_metadata(
            original_metadata,
            DataFormat.PANDAS_DATAFRAME,
            DataFormat.NUMPY_ARRAY,
            'test_adapter'
        )
        
        # Check lineage was updated
        lineage = self.manager._lineage_tracking['test_lineage']
        assert len(lineage.transformation_history) == 1
        assert lineage.current_format == DataFormat.NUMPY_ARRAY
    
    def test_add_preservation_rule(self):
        """Test adding preservation rule."""
        rule = PreservationRule(
            metadata_key="test_rule",
            metadata_type=MetadataType.SEMANTIC,
            preservation_strategy=PreservationStrategy.STRICT
        )
        
        self.manager.add_preservation_rule(rule)
        
        assert 'test_rule' in self.manager._preservation_rules
        assert self.manager._preservation_rules['test_rule'] == rule
    
    def test_add_metadata_schema(self):
        """Test adding metadata schema."""
        schema = MetadataSchema(
            schema_name="test_schema",
            data_format=DataFormat.JSON,
            required_fields={'field1'},
            optional_fields={'field2'}
        )
        
        self.manager.add_metadata_schema(schema)
        
        assert DataFormat.JSON in self.manager._metadata_schemas
        assert self.manager._metadata_schemas[DataFormat.JSON] == schema
    
    def test_add_transformer(self):
        """Test adding metadata transformer."""
        class TestTransformer(PandasMetadataTransformer):
            def get_supported_metadata_types(self):
                return {MetadataType.CUSTOM}
        
        transformer = TestTransformer()
        initial_count = len(self.manager._transformers)
        
        self.manager.add_transformer(transformer)
        
        assert len(self.manager._transformers) == initial_count + 1
        assert transformer in self.manager._transformers
    
    def test_validate_metadata_valid(self):
        """Test validating valid metadata against schema."""
        metadata = {
            'data_format': 'pandas_dataframe',
            'extraction_timestamp': '2023-01-01T00:00:00',
            'shape': (100, 3),
            'columns': ['A', 'B', 'C']
        }
        
        result = self.manager.validate_metadata(metadata, DataFormat.PANDAS_DATAFRAME)
        
        assert isinstance(result, ValidationResult)
        assert result.is_valid is True
        assert result.score == 1.0
        assert len(result.errors) == 0
    
    def test_validate_metadata_missing_required(self):
        """Test validating metadata missing required fields."""
        metadata = {
            'shape': (100, 3)
            # Missing required 'data_format' and 'extraction_timestamp'
        }
        
        result = self.manager.validate_metadata(metadata, DataFormat.PANDAS_DATAFRAME)
        
        assert result.is_valid is False
        assert len(result.errors) >= 1  # At least one error for missing required fields
    
    def test_validate_metadata_no_schema(self):
        """Test validating metadata when no schema exists."""
        metadata = {'some_field': 'some_value'}
        
        result = self.manager.validate_metadata(metadata, DataFormat.JSON)  # No schema for JSON
        
        assert result.is_valid is True  # Should pass when no schema exists
        assert len(result.warnings) == 1  # Should warn about missing schema
    
    def test_get_lineage_info(self):
        """Test retrieving lineage information."""
        lineage = MetadataLineage(
            original_source='test_source',
            current_format=DataFormat.PANDAS_DATAFRAME
        )
        
        lineage_id = 'test_lineage_123'
        self.manager._lineage_tracking[lineage_id] = lineage
        
        retrieved = self.manager.get_lineage_info(lineage_id)
        
        assert retrieved is lineage
        assert retrieved.original_source == 'test_source'
    
    def test_get_nonexistent_lineage(self):
        """Test retrieving non-existent lineage information."""
        result = self.manager.get_lineage_info('nonexistent')
        assert result is None
    
    def test_preservation_rules_application(self):
        """Test application of preservation rules."""
        # Add a rule that removes semantic metadata in minimal strategy
        rule = PreservationRule(
            metadata_key="semantic_info",
            metadata_type=MetadataType.SEMANTIC,
            preservation_strategy=PreservationStrategy.MINIMAL
        )
        
        self.manager.add_preservation_rule(rule)
        
        metadata = {
            'semantic_info': 'should_be_removed',
            'shape': (100, 2),  # Structural - should be kept
            'extraction_timestamp': '2023-01-01'  # Operational - should be kept
        }
        
        transformed = self.manager._apply_preservation_rules(
            metadata,
            DataFormat.PANDAS_DATAFRAME,
            DataFormat.NUMPY_ARRAY
        )
        
        # Semantic metadata should be removed in minimal strategy
        assert 'semantic_info' not in transformed
        assert 'shape' in transformed  # Structural should remain
        assert 'extraction_timestamp' in transformed  # Operational should remain


class TestUtilityFunctions:
    """Test cases for utility functions."""
    
    def test_create_preservation_rule(self):
        """Test create_preservation_rule utility function."""
        rule = create_preservation_rule(
            'test_key',
            strategy=PreservationStrategy.STRICT,
            metadata_type=MetadataType.QUALITY,
            priority=50
        )
        
        assert isinstance(rule, PreservationRule)
        assert rule.metadata_key == 'test_key'
        assert rule.preservation_strategy == PreservationStrategy.STRICT
        assert rule.metadata_type == MetadataType.QUALITY
        assert rule.priority == 50
    
    def test_create_metadata_schema(self):
        """Test create_metadata_schema utility function."""
        schema = create_metadata_schema(
            'test_schema',
            DataFormat.TIME_SERIES,
            required_fields=['timestamp', 'value'],
            optional_fields=['category'],
            strict_validation=True
        )
        
        assert isinstance(schema, MetadataSchema)
        assert schema.schema_name == 'test_schema'
        assert schema.data_format == DataFormat.TIME_SERIES
        assert 'timestamp' in schema.required_fields
        assert 'category' in schema.optional_fields
        assert schema.strict_validation is True


class TestMetadataConflictResolution:
    """Test cases for metadata conflict resolution."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.manager = MetadataManager()
    
    def test_timestamp_conflict_resolution(self):
        """Test timestamp conflict resolution (prefer more recent)."""
        resolved = self.manager._resolve_merge_conflict(
            'timestamp',
            '2023-01-01T10:00:00',
            '2023-01-01T11:00:00'
        )
        
        assert resolved == '2023-01-01T11:00:00'  # Should prefer more recent
    
    def test_list_conflict_resolution(self):
        """Test list conflict resolution (merge and deduplicate)."""
        resolved = self.manager._resolve_merge_conflict(
            'items',
            ['a', 'b', 'c'],
            ['c', 'd', 'e']
        )
        
        assert resolved == ['a', 'b', 'c', 'd', 'e']  # Merged and deduplicated
    
    def test_dict_conflict_resolution(self):
        """Test dictionary conflict resolution (merge)."""
        resolved = self.manager._resolve_merge_conflict(
            'config',
            {'key1': 'value1', 'key2': 'value2'},
            {'key2': 'updated_value2', 'key3': 'value3'}
        )
        
        expected = {'key1': 'value1', 'key2': 'updated_value2', 'key3': 'value3'}
        assert resolved == expected
    
    def test_numeric_conflict_resolution(self):
        """Test numeric conflict resolution (average)."""
        resolved = self.manager._resolve_merge_conflict(
            'score',
            10.0,
            20.0
        )
        
        assert resolved == 15.0  # Average
    
    def test_default_conflict_resolution(self):
        """Test default conflict resolution (prefer new value)."""
        resolved = self.manager._resolve_merge_conflict(
            'unknown_key',
            'old_value',
            'new_value'
        )
        
        assert resolved == 'new_value'  # Should prefer new value


if __name__ == '__main__':
    pytest.main([__file__])