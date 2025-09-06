"""
Comprehensive tests for the Schema Validation System.

Tests cover:
- SchemaInferenceEngine functionality
- SchemaValidator with various data types
- SchemaEvolutionManager capabilities
- ValidationRule framework
- Error handling and edge cases
- Performance with large datasets
- Integration with existing framework components
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, date
from scipy import sparse
from typing import Dict, Any, List
import tempfile
import time

from src.localdata_mcp.pipeline.integration.schema_validation import (
    SchemaInferenceEngine,
    SchemaValidator,
    SchemaEvolutionManager,
    DataSchema,
    SchemaConstraint,
    ValidationError,
    ValidationRule,
    TypeValidationRule,
    RangeValidationRule,
    NullValidationRule,
    SchemaValidationLevel,
    SchemaConformanceLevel,
    ValidationRuleType,
    SchemaInferenceResult,
    SchemaValidationResult
)
from src.localdata_mcp.pipeline.integration.interfaces import DataFormat


class TestSchemaInferenceEngine:
    """Test cases for the SchemaInferenceEngine."""
    
    @pytest.fixture
    def inference_engine(self):
        """Create a SchemaInferenceEngine instance for testing."""
        return SchemaInferenceEngine()
    
    @pytest.fixture
    def sample_dataframe(self):
        """Create a sample DataFrame for testing."""
        return pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'name': ['Alice', 'Bob', 'Charlie', None, 'Eve'],
            'age': [25, 30, 35, 28, 22],
            'salary': [50000.0, 60000.0, 70000.0, 55000.0, 45000.0],
            'active': [True, True, False, True, True],
            'created_at': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'])
        })
    
    @pytest.fixture
    def sample_numpy_array(self):
        """Create a sample numpy array for testing."""
        return np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float64)
    
    @pytest.fixture
    def sample_sparse_matrix(self):
        """Create a sample sparse matrix for testing."""
        return sparse.csr_matrix([[1, 0, 2], [0, 3, 0], [4, 0, 5]], dtype=np.float32)
    
    def test_dataframe_schema_inference(self, inference_engine, sample_dataframe):
        """Test schema inference for pandas DataFrame."""
        result = inference_engine.infer_schema(sample_dataframe, DataFormat.PANDAS_DATAFRAME)
        
        assert isinstance(result, SchemaInferenceResult)
        assert result.confidence_score > 0.7
        assert result.inferred_schema.data_format == DataFormat.PANDAS_DATAFRAME
        
        # Check inferred fields
        schema = result.inferred_schema
        assert 'id' in schema.fields
        assert 'name' in schema.fields
        assert 'age' in schema.fields
        assert 'salary' in schema.fields
        assert 'active' in schema.fields
        assert 'created_at' in schema.fields
        
        # Check field types
        assert schema.fields['id']['type'] == 'int'
        assert schema.fields['name']['type'] == 'str'
        assert schema.fields['age']['type'] == 'int'
        assert schema.fields['salary']['type'] == 'float'
        assert schema.fields['active']['type'] == 'bool'
        assert schema.fields['created_at']['type'] == 'datetime'
        
        # Check nullable inference
        assert schema.fields['name']['nullable'] == True  # Has null values
        assert schema.fields['id']['nullable'] == False   # No null values
        
        # Check constraints were generated
        assert len(schema.constraints) > 0
        
        # Check inference details
        assert result.sample_size == len(sample_dataframe)
        assert 'column_count' in result.inference_details
        assert 'numeric_columns' in result.inference_details
    
    def test_numpy_schema_inference(self, inference_engine, sample_numpy_array):
        """Test schema inference for numpy array."""
        result = inference_engine.infer_schema(sample_numpy_array, DataFormat.NUMPY_ARRAY)
        
        assert isinstance(result, SchemaInferenceResult)
        assert result.confidence_score > 0.8
        assert result.inferred_schema.data_format == DataFormat.NUMPY_ARRAY
        
        schema = result.inferred_schema
        assert 'array_data' in schema.fields
        assert schema.fields['array_data']['type'] == 'float'
        assert schema.fields['array_data']['shape'] == (3, 3)
        assert schema.fields['array_data']['ndim'] == 2
        
        # Check inference details
        assert result.inference_details['shape'] == (3, 3)
        assert result.inference_details['element_type'] == 'float'
    
    def test_sparse_matrix_inference(self, inference_engine, sample_sparse_matrix):
        """Test schema inference for sparse matrix."""
        result = inference_engine.infer_schema(sample_sparse_matrix, DataFormat.SCIPY_SPARSE)
        
        assert isinstance(result, SchemaInferenceResult)
        assert result.confidence_score > 0.8
        assert result.inferred_schema.data_format == DataFormat.SCIPY_SPARSE
        
        schema = result.inferred_schema
        assert 'sparse_data' in schema.fields
        assert schema.fields['sparse_data']['type'] == 'float'
        assert schema.fields['sparse_data']['format'] == 'csr'
        assert schema.fields['sparse_data']['shape'] == (3, 3)
        
        # Check density calculation
        density = schema.fields['sparse_data']['density']
        expected_density = sample_sparse_matrix.nnz / (3 * 3)
        assert abs(density - expected_density) < 0.01
    
    def test_schema_inference_with_sampling(self, inference_engine):
        """Test schema inference with data sampling."""
        large_df = pd.DataFrame({
            'value': np.random.randn(10000),
            'category': np.random.choice(['A', 'B', 'C'], 10000)
        })
        
        result = inference_engine.infer_schema(large_df, sample_size=1000)
        
        assert result.sample_size == 1000
        assert result.confidence_score > 0.0
        assert len(result.inferred_schema.fields) == 2
    
    def test_low_confidence_warning(self, inference_engine):
        """Test that low confidence generates warnings."""
        # Create problematic data that should result in low confidence
        mixed_data = pd.DataFrame({
            'mixed_col': [1, 'two', 3.0, None, True, [1, 2, 3]]
        })
        
        result = inference_engine.infer_schema(mixed_data, confidence_threshold=0.8)
        
        # Should have warnings due to mixed types
        assert len(result.warnings) > 0 or result.confidence_score < 0.8
    
    def test_quality_metrics_calculation(self, inference_engine, sample_dataframe):
        """Test quality metrics calculation."""
        result = inference_engine.infer_schema(sample_dataframe)
        
        assert 0.0 <= result.data_quality_score <= 1.0
        assert 0.0 <= result.completeness_score <= 1.0
        assert 0.0 <= result.consistency_score <= 1.0
        
        # Completeness should be less than 1.0 due to null values in 'name' column
        assert result.completeness_score < 1.0
    
    def test_alternative_schemas(self, inference_engine):
        """Test alternative schema suggestions."""
        # Create ambiguous data that could have multiple interpretations
        ambiguous_data = pd.DataFrame({
            'numbers_as_strings': ['1', '2', '3', '4', '5']
        })
        
        result = inference_engine.infer_schema(ambiguous_data)
        
        # The engine might suggest alternative interpretations
        assert isinstance(result.alternative_schemas, list)


class TestSchemaValidator:
    """Test cases for the SchemaValidator."""
    
    @pytest.fixture
    def validator(self):
        """Create a SchemaValidator instance for testing."""
        return SchemaValidator()
    
    @pytest.fixture
    def valid_dataframe_schema(self):
        """Create a valid DataFrame schema for testing."""
        return DataSchema(
            schema_id="test_schema",
            schema_name="Test DataFrame Schema",
            data_format=DataFormat.PANDAS_DATAFRAME,
            fields={
                'id': {'type': 'int', 'nullable': False},
                'name': {'type': 'str', 'nullable': True},
                'age': {'type': 'int', 'nullable': False, 'min': 0, 'max': 150},
                'salary': {'type': 'float', 'nullable': False, 'min': 0.0}
            },
            constraints=[
                SchemaConstraint(
                    constraint_id="age_range",
                    constraint_type=ValidationRuleType.RANGE_CHECK,
                    field_name="age",
                    constraint_value=0,
                    description="Age must be non-negative"
                ),
                SchemaConstraint(
                    constraint_id="id_not_null",
                    constraint_type=ValidationRuleType.NULL_CHECK,
                    field_name="id",
                    constraint_value=False,
                    description="ID cannot be null"
                )
            ]
        )
    
    def test_valid_data_validation(self, validator, valid_dataframe_schema):
        """Test validation of valid data."""
        valid_data = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Alice', 'Bob', None],
            'age': [25, 30, 35],
            'salary': [50000.0, 60000.0, 70000.0]
        })
        
        result = validator.validate_data(valid_data, valid_dataframe_schema)
        
        assert isinstance(result, SchemaValidationResult)
        assert result.is_valid == True
        assert result.conformance_level == SchemaConformanceLevel.PERFECT
        assert result.validation_score == 1.0
        assert len(result.errors) == 0
        assert result.passed_checks > 0
    
    def test_invalid_data_validation(self, validator, valid_dataframe_schema):
        """Test validation of invalid data."""
        invalid_data = pd.DataFrame({
            'id': [1, None, 3],  # Null ID (violates constraint)
            'name': ['Alice', 'Bob', 'Charlie'],
            'age': [25, -5, 200],  # Negative age and age > 150
            'salary': [50000.0, 60000.0, -1000.0]  # Negative salary
        })
        
        result = validator.validate_data(invalid_data, valid_dataframe_schema)
        
        assert result.is_valid == False
        assert result.conformance_level != SchemaConformanceLevel.PERFECT
        assert result.validation_score < 1.0
        assert len(result.errors) > 0
        assert result.failed_checks > 0
        
        # Check specific error types
        error_types = {error.error_type for error in result.errors}
        assert ValidationRuleType.NULL_CHECK in error_types
        assert ValidationRuleType.RANGE_CHECK in error_types
    
    def test_type_validation_errors(self, validator, valid_dataframe_schema):
        """Test type validation errors."""
        type_mismatch_data = pd.DataFrame({
            'id': ['not_int', 2, 3],  # String instead of int
            'name': ['Alice', 'Bob', 'Charlie'],
            'age': [25, 30, 35],
            'salary': [50000.0, 60000.0, 70000.0]
        })
        
        result = validator.validate_data(type_mismatch_data, valid_dataframe_schema)
        
        assert result.is_valid == False
        assert len(result.errors) > 0
        
        # Find type validation errors
        type_errors = [e for e in result.errors if e.error_type == ValidationRuleType.TYPE_CHECK]
        assert len(type_errors) > 0
    
    def test_missing_column_validation(self, validator, valid_dataframe_schema):
        """Test validation with missing required columns."""
        incomplete_data = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Alice', 'Bob', 'Charlie']
            # Missing 'age' and 'salary' columns
        })
        
        result = validator.validate_data(incomplete_data, valid_dataframe_schema)
        
        assert result.is_valid == False
        assert len(result.errors) > 0
        
        # Check for missing column errors
        missing_column_errors = [e for e in result.errors 
                               if "missing" in e.error_message.lower()]
        assert len(missing_column_errors) > 0
    
    def test_extra_column_handling(self, validator):
        """Test handling of extra columns in strict mode."""
        strict_schema = DataSchema(
            schema_id="strict_test",
            schema_name="Strict Test Schema",
            data_format=DataFormat.PANDAS_DATAFRAME,
            fields={'id': {'type': 'int', 'nullable': False}},
            is_strict=True
        )
        
        data_with_extra = pd.DataFrame({
            'id': [1, 2, 3],
            'extra_column': ['A', 'B', 'C']  # Extra column
        })
        
        result = validator.validate_data(data_with_extra, strict_schema)
        
        # Should have warnings for extra columns in strict mode
        assert len(result.warnings) > 0
        extra_column_warnings = [w for w in result.warnings 
                               if "extra" in w.error_message.lower()]
        assert len(extra_column_warnings) > 0
    
    def test_numpy_array_validation(self, validator):
        """Test validation of numpy arrays."""
        numpy_schema = DataSchema(
            schema_id="numpy_test",
            schema_name="NumPy Test Schema",
            data_format=DataFormat.NUMPY_ARRAY,
            fields={'array_data': {'type': 'float', 'shape': (3, 3)}}
        )
        
        # Valid array
        valid_array = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        result = validator.validate_data(valid_array, numpy_schema)
        assert result.is_valid == True
        
        # Invalid shape
        invalid_array = np.array([[1.0, 2.0], [3.0, 4.0]])  # Wrong shape
        result = validator.validate_data(invalid_array, numpy_schema)
        assert result.is_valid == False
        assert len(result.errors) > 0
    
    def test_custom_validation_rule(self, validator, valid_dataframe_schema):
        """Test custom validation rules."""
        class CustomEmailValidation(ValidationRule):
            def __init__(self):
                super().__init__("email_validation", ValidationRuleType.PATTERN_CHECK, 
                               "Email format validation")
            
            def validate(self, field_name: str, value: Any, context: Dict[str, Any]) -> List[ValidationError]:
                if pd.isna(value):
                    return []
                
                errors = []
                if isinstance(value, str) and '@' not in value:
                    errors.append(ValidationError(
                        error_id=f"email_format_{field_name}",
                        error_type=ValidationRuleType.PATTERN_CHECK,
                        field_name=field_name,
                        expected_value="valid email format",
                        actual_value=value,
                        error_message=f"Invalid email format: {value}"
                    ))
                return errors
            
            def is_applicable(self, field_name: str, field_schema: Dict[str, Any]) -> bool:
                return field_name == 'email'
        
        # Add custom rule
        custom_rule = CustomEmailValidation()
        validator.add_custom_rule(custom_rule)
        
        # Create schema with email field
        email_schema = DataSchema(
            schema_id="email_test",
            schema_name="Email Test Schema",
            data_format=DataFormat.PANDAS_DATAFRAME,
            fields={'email': {'type': 'str', 'nullable': True}}
        )
        
        # Test with invalid email
        invalid_email_data = pd.DataFrame({
            'email': ['valid@email.com', 'invalid-email', 'another@valid.com']
        })
        
        result = validator.validate_data(invalid_email_data, email_schema)
        
        # Should have custom validation errors
        custom_errors = [e for e in result.errors 
                        if e.error_type == ValidationRuleType.PATTERN_CHECK]
        assert len(custom_errors) > 0
    
    def test_validation_performance(self, validator):
        """Test validation performance with large datasets."""
        # Create large dataset
        large_data = pd.DataFrame({
            'id': range(10000),
            'value': np.random.randn(10000),
            'category': np.random.choice(['A', 'B', 'C'], 10000)
        })
        
        schema = DataSchema(
            schema_id="perf_test",
            schema_name="Performance Test Schema",
            data_format=DataFormat.PANDAS_DATAFRAME,
            fields={
                'id': {'type': 'int', 'nullable': False},
                'value': {'type': 'float', 'nullable': False},
                'category': {'type': 'str', 'nullable': False}
            }
        )
        
        start_time = time.time()
        result = validator.validate_data(large_data, schema)
        validation_time = time.time() - start_time
        
        # Should complete within reasonable time
        assert validation_time < 10.0  # Less than 10 seconds
        assert result.validation_time > 0
        assert result.total_checks > 0
    
    def test_max_errors_limit(self, validator, valid_dataframe_schema):
        """Test maximum error limit functionality."""
        # Create data with many errors
        bad_data = pd.DataFrame({
            'id': [None] * 100,  # All null IDs
            'name': ['Name'] * 100,
            'age': [-1] * 100,   # All invalid ages
            'salary': [-1.0] * 100  # All invalid salaries
        })
        
        result = validator.validate_data(bad_data, valid_dataframe_schema, max_errors=10)
        
        # Should limit errors to max_errors
        assert len(result.errors) <= 10
        
        # But should still count all checks
        assert result.total_checks > 10


class TestSchemaEvolutionManager:
    """Test cases for the SchemaEvolutionManager."""
    
    @pytest.fixture
    def evolution_manager(self):
        """Create a SchemaEvolutionManager for testing."""
        return SchemaEvolutionManager()
    
    @pytest.fixture
    def base_schema(self):
        """Create a base schema for evolution testing."""
        return DataSchema(
            schema_id="base_v1",
            schema_name="Base Schema",
            schema_version="1.0",
            data_format=DataFormat.PANDAS_DATAFRAME,
            fields={
                'id': {'type': 'int', 'nullable': False},
                'name': {'type': 'str', 'nullable': True},
                'age': {'type': 'int', 'nullable': False}
            },
            constraints=[
                SchemaConstraint(
                    constraint_id="age_positive",
                    constraint_type=ValidationRuleType.RANGE_CHECK,
                    field_name="age",
                    constraint_value=0,
                    description="Age must be positive"
                )
            ]
        )
    
    @pytest.fixture
    def evolved_schema(self):
        """Create an evolved version of the base schema."""
        return DataSchema(
            schema_id="base_v2",
            schema_name="Base Schema",
            schema_version="2.0",
            data_format=DataFormat.PANDAS_DATAFRAME,
            fields={
                'id': {'type': 'int', 'nullable': False},
                'name': {'type': 'str', 'nullable': True},
                'age': {'type': 'int', 'nullable': False},
                'email': {'type': 'str', 'nullable': True},  # New field
                'salary': {'type': 'float', 'nullable': True}  # New field
            },
            constraints=[
                SchemaConstraint(
                    constraint_id="age_positive",
                    constraint_type=ValidationRuleType.RANGE_CHECK,
                    field_name="age",
                    constraint_value=18,  # Changed constraint value
                    description="Age must be at least 18"
                )
            ]
        )
    
    def test_schema_version_management(self, evolution_manager, base_schema, evolved_schema):
        """Test adding and retrieving schema versions."""
        evolution_manager.add_schema_version(base_schema)
        evolution_manager.add_schema_version(evolved_schema)
        
        # Test latest version retrieval
        latest = evolution_manager.get_latest_schema("Base Schema")
        assert latest is not None
        assert latest.schema_version == "2.0"
        
        # Test specific version retrieval
        v1 = evolution_manager.get_schema_version("Base Schema", "1.0")
        assert v1 is not None
        assert v1.schema_version == "1.0"
        
        v2 = evolution_manager.get_schema_version("Base Schema", "2.0")
        assert v2 is not None
        assert v2.schema_version == "2.0"
    
    def test_compatibility_score_calculation(self, evolution_manager, base_schema, evolved_schema):
        """Test compatibility score calculation between schemas."""
        score = evolution_manager.calculate_compatibility_score(base_schema, evolved_schema)
        
        assert 0.0 <= score <= 1.0
        # Should be reasonably compatible (added fields, minor constraint change)
        assert score > 0.5
        
        # Test with identical schemas
        identical_score = evolution_manager.calculate_compatibility_score(base_schema, base_schema)
        assert identical_score == 1.0
    
    def test_incompatible_schemas(self, evolution_manager):
        """Test compatibility scoring with incompatible schemas."""
        schema1 = DataSchema(
            schema_id="test1",
            schema_name="Test Schema 1",
            data_format=DataFormat.PANDAS_DATAFRAME,
            fields={'id': {'type': 'int', 'nullable': False}}
        )
        
        schema2 = DataSchema(
            schema_id="test2",
            schema_name="Test Schema 2",
            data_format=DataFormat.NUMPY_ARRAY,  # Different format
            fields={'array_data': {'type': 'float', 'shape': (10,)}}
        )
        
        score = evolution_manager.calculate_compatibility_score(schema1, schema2)
        
        # Should have low compatibility due to different formats and fields
        assert score < 0.7
    
    def test_schema_drift_detection(self, evolution_manager, base_schema):
        """Test schema drift detection functionality."""
        # Create data that represents drift from base schema
        drifted_data = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Alice', 'Bob', 'Charlie'],
            'age': [25, 30, 35],
            'new_field': ['A', 'B', 'C']  # New field indicates drift
        })
        
        drift_analysis = evolution_manager.detect_schema_drift(
            drifted_data, base_schema, drift_threshold=0.1
        )
        
        assert 'drift_detected' in drift_analysis
        assert 'drift_level' in drift_analysis
        assert 'compatibility_score' in drift_analysis
        assert 'field_changes' in drift_analysis
        assert 'recommendations' in drift_analysis
        
        # Should detect the addition of 'new_field'
        field_changes = drift_analysis['field_changes']
        assert len(field_changes['added_fields']) > 0
        assert 'new_field' in field_changes['added_fields']
    
    def test_field_change_detection(self, evolution_manager, base_schema, evolved_schema):
        """Test detailed field change detection."""
        drift_analysis = evolution_manager.detect_schema_drift(
            pd.DataFrame({'id': [1], 'name': ['Test'], 'age': [25], 
                         'email': ['test@test.com'], 'salary': [50000.0]}),
            base_schema
        )
        
        field_changes = drift_analysis['field_changes']
        
        # Should detect added fields
        assert len(field_changes['added_fields']) >= 2
        assert 'email' in field_changes['added_fields']
        assert 'salary' in field_changes['added_fields']
        
        # Should have no removed fields in this case
        assert len(field_changes['removed_fields']) == 0
    
    def test_constraint_change_detection(self, evolution_manager, base_schema):
        """Test constraint change detection."""
        # Create evolved schema with modified constraints
        modified_schema = DataSchema(
            schema_id="modified",
            schema_name="Base Schema",
            schema_version="1.1",
            data_format=DataFormat.PANDAS_DATAFRAME,
            fields=base_schema.fields.copy(),
            constraints=[
                SchemaConstraint(
                    constraint_id="age_positive",
                    constraint_type=ValidationRuleType.RANGE_CHECK,
                    field_name="age",
                    constraint_value=21,  # Changed from 0 to 21
                    description="Age must be at least 21"
                )
            ]
        )
        
        # Create manager and add schemas
        evolution_manager.add_schema_version(base_schema)
        evolution_manager.add_schema_version(modified_schema)
        
        # Test data that matches modified schema
        test_data = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Alice', 'Bob', 'Charlie'],
            'age': [25, 30, 35]
        })
        
        drift_analysis = evolution_manager.detect_schema_drift(test_data, base_schema)
        
        # Should detect constraint changes if inference picks up the higher minimum age
        constraint_changes = drift_analysis['constraint_changes']
        assert isinstance(constraint_changes, dict)
    
    def test_recommendations_generation(self, evolution_manager, base_schema):
        """Test drift recommendations generation."""
        # Data with significant changes
        changed_data = pd.DataFrame({
            'id': [1, 2, 3],
            'full_name': ['Alice Smith', 'Bob Jones', 'Charlie Brown'],  # Renamed field
            'age_years': [25, 30, 35],  # Renamed field
            'department': ['Engineering', 'Sales', 'Marketing']  # New field
        })
        
        drift_analysis = evolution_manager.detect_schema_drift(changed_data, base_schema)
        
        recommendations = drift_analysis['recommendations']
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        
        # Should have recommendations about field changes
        rec_text = ' '.join(recommendations).lower()
        assert any(keyword in rec_text for keyword in ['field', 'column', 'migration', 'transform'])
    
    def test_compatibility_caching(self, evolution_manager, base_schema, evolved_schema):
        """Test that compatibility scores are cached for performance."""
        # Calculate score twice
        score1 = evolution_manager.calculate_compatibility_score(base_schema, evolved_schema)
        score2 = evolution_manager.calculate_compatibility_score(base_schema, evolved_schema)
        
        assert score1 == score2
        
        # Check cache was used (cache should have one entry)
        assert len(evolution_manager.compatibility_cache) >= 1
    
    def test_drift_level_categorization(self, evolution_manager):
        """Test drift level categorization logic."""
        # Create schemas with different levels of compatibility
        base = DataSchema(
            schema_id="base",
            schema_name="Base",
            data_format=DataFormat.PANDAS_DATAFRAME,
            fields={'id': {'type': 'int', 'nullable': False}}
        )
        
        # Minimal drift - compatible schema
        minimal_drift = DataSchema(
            schema_id="minimal",
            schema_name="Base",
            data_format=DataFormat.PANDAS_DATAFRAME,
            fields={
                'id': {'type': 'int', 'nullable': False},
                'optional_field': {'type': 'str', 'nullable': True}  # Added optional field
            }
        )
        
        # Major drift - incompatible schema
        major_drift = DataSchema(
            schema_id="major",
            schema_name="Base",
            data_format=DataFormat.NUMPY_ARRAY,  # Different format
            fields={'array_data': {'type': 'float', 'shape': (10,)}}
        )
        
        minimal_score = evolution_manager.calculate_compatibility_score(base, minimal_drift)
        major_score = evolution_manager.calculate_compatibility_score(base, major_drift)
        
        assert minimal_score > major_score
        assert minimal_score > 0.8  # Should be high compatibility
        assert major_score < 0.5    # Should be low compatibility


class TestValidationRules:
    """Test cases for individual validation rules."""
    
    def test_type_validation_rule(self):
        """Test type validation rule functionality."""
        rule = TypeValidationRule('int')
        
        # Valid type
        errors = rule.validate('test_field', 42, {'nullable': False})
        assert len(errors) == 0
        
        # Invalid type
        errors = rule.validate('test_field', 'not_int', {'nullable': False})
        assert len(errors) == 1
        assert errors[0].error_type == ValidationRuleType.TYPE_CHECK
        
        # Null value with nullable=True
        errors = rule.validate('test_field', None, {'nullable': True})
        assert len(errors) == 0
        
        # Null value with nullable=False
        errors = rule.validate('test_field', None, {'nullable': False})
        assert len(errors) == 0  # Type rule doesn't handle null validation
    
    def test_type_coercion(self):
        """Test type coercion in type validation."""
        rule = TypeValidationRule(float, allow_coercion=True)
        
        # Should successfully coerce int to float
        errors = rule.validate('test_field', 42, {'nullable': False})
        assert len(errors) == 0
        
        # Should fail to coerce string to float
        errors = rule.validate('test_field', 'not_a_number', {'nullable': False})
        assert len(errors) == 1
    
    def test_range_validation_rule(self):
        """Test range validation rule functionality."""
        rule = RangeValidationRule(min_value=0, max_value=100, inclusive=True)
        
        # Valid range
        errors = rule.validate('test_field', 50, {})
        assert len(errors) == 0
        
        # Below minimum
        errors = rule.validate('test_field', -1, {})
        assert len(errors) == 1
        assert 'minimum' in errors[0].error_message
        
        # Above maximum
        errors = rule.validate('test_field', 101, {})
        assert len(errors) == 1
        assert 'maximum' in errors[0].error_message
        
        # Edge cases
        errors = rule.validate('test_field', 0, {})
        assert len(errors) == 0
        
        errors = rule.validate('test_field', 100, {})
        assert len(errors) == 0
    
    def test_null_validation_rule(self):
        """Test null validation rule functionality."""
        # Allow nulls
        rule = NullValidationRule(allow_nulls=True)
        errors = rule.validate('test_field', None, {})
        assert len(errors) == 0
        
        # Disallow nulls
        rule = NullValidationRule(allow_nulls=False)
        errors = rule.validate('test_field', None, {})
        assert len(errors) == 1
        assert errors[0].error_type == ValidationRuleType.NULL_CHECK
        
        # Non-null value
        errors = rule.validate('test_field', 'value', {})
        assert len(errors) == 0
    
    def test_rule_applicability(self):
        """Test validation rule applicability logic."""
        type_rule = TypeValidationRule('int')
        range_rule = RangeValidationRule(0, 100)
        null_rule = NullValidationRule(False)
        
        # Type rule applicability
        assert type_rule.is_applicable('field', {'type': 'int'}) == True
        assert type_rule.is_applicable('field', {'no_type': 'value'}) == False
        
        # Range rule applicability
        assert range_rule.is_applicable('field', {'type': 'int'}) == True
        assert range_rule.is_applicable('field', {'type': 'float'}) == True
        assert range_rule.is_applicable('field', {'type': 'str'}) == False
        assert range_rule.is_applicable('field', {'min': 0}) == True
        
        # Null rule applicability
        assert null_rule.is_applicable('field', {'nullable': True}) == True
        assert null_rule.is_applicable('field', {'required': True}) == True
        assert null_rule.is_applicable('field', {'type': 'int'}) == False


class TestSchemaValidationIntegration:
    """Integration tests for the complete schema validation system."""
    
    def test_end_to_end_workflow(self):
        """Test complete end-to-end schema validation workflow."""
        # 1. Create sample data
        data = pd.DataFrame({
            'user_id': [1, 2, 3, 4, 5],
            'username': ['alice', 'bob', 'charlie', 'diana', 'eve'],
            'age': [25, 30, 35, 28, 22],
            'balance': [100.50, 250.00, 75.25, 500.00, 10.00],
            'is_active': [True, True, False, True, True],
            'created_date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', 
                                          '2023-01-04', '2023-01-05'])
        })
        
        # 2. Infer schema from data
        inference_engine = SchemaInferenceEngine()
        inference_result = inference_engine.infer_schema(data)
        
        assert inference_result.confidence_score > 0.8
        inferred_schema = inference_result.inferred_schema
        
        # 3. Validate original data against inferred schema
        validator = SchemaValidator()
        validation_result = validator.validate_data(data, inferred_schema)
        
        assert validation_result.is_valid == True
        assert validation_result.validation_score == 1.0
        
        # 4. Test with modified data that should fail validation
        modified_data = data.copy()
        modified_data.loc[0, 'age'] = -5  # Invalid age
        modified_data.loc[1, 'user_id'] = None  # Null ID
        
        validation_result = validator.validate_data(modified_data, inferred_schema)
        
        assert validation_result.is_valid == False
        assert len(validation_result.errors) > 0
        
        # 5. Test schema evolution
        evolution_manager = SchemaEvolutionManager()
        evolution_manager.add_schema_version(inferred_schema)
        
        # Create evolved data with new field
        evolved_data = data.copy()
        evolved_data['email'] = ['alice@example.com', 'bob@example.com', 
                               'charlie@example.com', 'diana@example.com', 
                               'eve@example.com']
        
        drift_analysis = evolution_manager.detect_schema_drift(evolved_data, inferred_schema)
        
        assert drift_analysis['drift_detected'] == True
        assert 'email' in drift_analysis['field_changes']['added_fields']
    
    def test_performance_with_large_dataset(self):
        """Test system performance with large datasets."""
        # Create large dataset
        n_rows = 50000
        large_data = pd.DataFrame({
            'id': range(n_rows),
            'value1': np.random.randn(n_rows),
            'value2': np.random.randn(n_rows),
            'category': np.random.choice(['A', 'B', 'C', 'D', 'E'], n_rows),
            'flag': np.random.choice([True, False], n_rows),
            'timestamp': pd.date_range('2023-01-01', periods=n_rows, freq='1min')
        })
        
        # Test schema inference performance
        inference_engine = SchemaInferenceEngine()
        start_time = time.time()
        result = inference_engine.infer_schema(large_data, sample_size=5000)
        inference_time = time.time() - start_time
        
        assert inference_time < 30.0  # Should complete within 30 seconds
        assert result.confidence_score > 0.0
        
        # Test validation performance with subset
        validator = SchemaValidator()
        subset_data = large_data.head(10000)  # Use subset for validation test
        
        start_time = time.time()
        validation_result = validator.validate_data(subset_data, result.inferred_schema)
        validation_time = time.time() - start_time
        
        assert validation_time < 60.0  # Should complete within 60 seconds
        assert validation_result.total_checks > 0
    
    def test_memory_efficiency(self):
        """Test memory-efficient processing of large datasets."""
        # This test would ideally use memory profiling tools
        # For now, we test that the system can handle moderately large data
        
        # Create data that might cause memory issues if not handled properly
        n_rows = 100000
        data = pd.DataFrame({
            'text_col': ['long text string ' * 100] * n_rows,  # Large text data
            'numeric_col': np.random.randn(n_rows)
        })
        
        inference_engine = SchemaInferenceEngine()
        
        # Should handle large data without memory errors
        try:
            result = inference_engine.infer_schema(data, sample_size=1000)
            assert result is not None
            success = True
        except MemoryError:
            success = False
        
        assert success, "Schema inference should handle large datasets efficiently"
    
    def test_error_recovery(self):
        """Test error handling and recovery in edge cases."""
        inference_engine = SchemaInferenceEngine()
        validator = SchemaValidator()
        
        # Test with completely empty data
        empty_data = pd.DataFrame()
        result = inference_engine.infer_schema(empty_data)
        assert result.confidence_score >= 0.0  # Should not crash
        
        # Test with data containing only nulls
        null_data = pd.DataFrame({'col1': [None, None, None]})
        result = inference_engine.infer_schema(null_data)
        assert result.confidence_score >= 0.0
        
        # Test validation with mismatched data type
        simple_schema = DataSchema(
            schema_id="test",
            schema_name="Test Schema",
            data_format=DataFormat.PANDAS_DATAFRAME,
            fields={'id': {'type': 'int', 'nullable': False}}
        )
        
        # Pass wrong data type entirely
        wrong_data = "not a dataframe"
        result = validator.validate_data(wrong_data, simple_schema)
        assert result.is_valid == False
        assert len(result.errors) > 0
        
        # Test with None data
        result = validator.validate_data(None, simple_schema)
        assert result.is_valid == False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])