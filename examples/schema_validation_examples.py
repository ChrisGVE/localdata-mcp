"""
Schema Validation System - Real-World Usage Examples.

This module demonstrates practical applications of the Schema Validation System
for LocalData MCP v2.0, showing how to use schema inference, validation, and
evolution management in real data science workflows.

Examples covered:
1. Data Quality Pipeline with Schema Validation
2. Schema Evolution for Data Warehouse ETL
3. Multi-Format Data Integration with Validation
4. Custom Validation Rules for Domain-Specific Data
5. Streaming Data Validation
6. Schema Drift Detection in Production
7. Data Migration with Schema Compatibility
8. Scientific Data Validation
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from typing import Dict, Any, List
from scipy import sparse

from src.localdata_mcp.pipeline.integration.schema_validation import (
    SchemaInferenceEngine,
    SchemaValidator,
    SchemaEvolutionManager,
    DataSchema,
    SchemaConstraint,
    ValidationError,
    ValidationRule,
    ValidationRuleType,
    SchemaValidationLevel,
    SchemaConformanceLevel
)
from src.localdata_mcp.pipeline.integration.interfaces import DataFormat


def example_1_data_quality_pipeline():
    """
    Example 1: Data Quality Pipeline with Schema Validation
    
    Demonstrates how to build a data quality pipeline that automatically
    infers schemas and validates incoming data for quality issues.
    """
    print("\n=== Example 1: Data Quality Pipeline ===")
    
    # Simulate incoming customer data
    customer_data = pd.DataFrame({
        'customer_id': [1, 2, 3, 4, 5],
        'name': ['Alice Johnson', 'Bob Smith', 'Charlie Brown', '', 'Eve Davis'],
        'email': ['alice@email.com', 'bob@email.com', 'invalid-email', 'charlie@email.com', 'eve@email.com'],
        'age': [28, 35, -5, 45, 25],  # Note: negative age
        'annual_income': [55000.0, 75000.0, 85000.0, 65000.0, 45000.0],
        'credit_score': [720, 800, 650, 750, 680],
        'signup_date': pd.to_datetime(['2023-01-01', '2023-02-15', '2023-03-10', '2023-04-05', '2023-05-20'])
    })
    
    print("Original customer data:")
    print(customer_data)
    print(f"Data shape: {customer_data.shape}")
    
    # Step 1: Infer schema from clean reference data
    inference_engine = SchemaInferenceEngine()
    schema_result = inference_engine.infer_schema(customer_data)
    
    print(f"\nSchema inference confidence: {schema_result.confidence_score:.2f}")
    print(f"Inferred {len(schema_result.inferred_schema.fields)} fields")
    print(f"Data quality score: {schema_result.data_quality_score:.2f}")
    
    # Step 2: Create custom validation rules for business logic
    class EmailValidationRule(ValidationRule):
        def __init__(self):
            super().__init__("email_validation", ValidationRuleType.PATTERN_CHECK, 
                           "Email format validation")
        
        def validate(self, field_name: str, value: Any, context: Dict[str, Any]) -> List[ValidationError]:
            if pd.isna(value) or value == '':
                return []
            
            errors = []
            if isinstance(value, str) and '@' not in value:
                errors.append(ValidationError(
                    error_id=f"email_format_{field_name}_{id(value)}",
                    error_type=ValidationRuleType.PATTERN_CHECK,
                    field_name=field_name,
                    expected_value="valid email format",
                    actual_value=value,
                    error_message=f"Invalid email format: '{value}'"
                ))
            return errors
        
        def is_applicable(self, field_name: str, field_schema: Dict[str, Any]) -> bool:
            return 'email' in field_name.lower()
    
    class CreditScoreValidationRule(ValidationRule):
        def __init__(self):
            super().__init__("credit_score_validation", ValidationRuleType.RANGE_CHECK,
                           "Credit score range validation")
        
        def validate(self, field_name: str, value: Any, context: Dict[str, Any]) -> List[ValidationError]:
            if pd.isna(value):
                return []
            
            errors = []
            try:
                score = float(value)
                if not (300 <= score <= 850):
                    errors.append(ValidationError(
                        error_id=f"credit_score_range_{field_name}_{id(value)}",
                        error_type=ValidationRuleType.RANGE_CHECK,
                        field_name=field_name,
                        expected_value="300-850",
                        actual_value=score,
                        error_message=f"Credit score {score} outside valid range (300-850)"
                    ))
            except (ValueError, TypeError):
                errors.append(ValidationError(
                    error_id=f"credit_score_type_{field_name}_{id(value)}",
                    error_type=ValidationRuleType.TYPE_CHECK,
                    field_name=field_name,
                    expected_value="numeric value",
                    actual_value=str(value),
                    error_message=f"Credit score must be numeric, got: {value}"
                ))
            return errors
        
        def is_applicable(self, field_name: str, field_schema: Dict[str, Any]) -> bool:
            return 'credit_score' in field_name.lower()
    
    # Step 3: Set up validator with custom rules
    validator = SchemaValidator()
    validator.add_custom_rule(EmailValidationRule())
    validator.add_custom_rule(CreditScoreValidationRule())
    
    # Step 4: Validate the data
    validation_result = validator.validate_data(
        customer_data, 
        schema_result.inferred_schema,
        validation_level=SchemaValidationLevel.STRICT
    )
    
    print(f"\nValidation Results:")
    print(f"Is Valid: {validation_result.is_valid}")
    print(f"Conformance Level: {validation_result.conformance_level.value}")
    print(f"Validation Score: {validation_result.validation_score:.2f}")
    print(f"Total Checks: {validation_result.total_checks}")
    print(f"Passed: {validation_result.passed_checks}, Failed: {validation_result.failed_checks}")
    
    # Step 5: Report validation errors
    if validation_result.errors:
        print(f"\nFound {len(validation_result.errors)} validation errors:")
        for i, error in enumerate(validation_result.errors[:5], 1):  # Show first 5
            print(f"  {i}. {error.field_name}: {error.error_message}")
            if error.row_index is not None:
                print(f"     Row: {error.row_index}, Value: {error.actual_value}")
    
    # Step 6: Generate data quality report
    quality_report = {
        'total_records': len(customer_data),
        'validation_score': validation_result.validation_score,
        'error_count': len(validation_result.errors),
        'warning_count': len(validation_result.warnings),
        'conformance_level': validation_result.conformance_level.value,
        'data_quality_issues': []
    }
    
    # Categorize issues
    for error in validation_result.errors:
        quality_report['data_quality_issues'].append({
            'field': error.field_name,
            'issue_type': error.error_type.value,
            'severity': error.severity,
            'message': error.error_message
        })
    
    print(f"\nData Quality Report:")
    print(f"Records processed: {quality_report['total_records']}")
    print(f"Overall quality score: {quality_report['validation_score']:.1%}")
    print(f"Issues found: {quality_report['error_count']} errors, {quality_report['warning_count']} warnings")
    
    return quality_report


def example_2_schema_evolution_etl():
    """
    Example 2: Schema Evolution for Data Warehouse ETL
    
    Demonstrates managing schema changes in an ETL pipeline where
    source data schemas evolve over time.
    """
    print("\n=== Example 2: Schema Evolution in ETL ===")
    
    # Simulate version 1 of a data source
    orders_v1 = pd.DataFrame({
        'order_id': [1001, 1002, 1003],
        'customer_id': [101, 102, 103],
        'product_name': ['Widget A', 'Widget B', 'Widget C'],
        'quantity': [2, 1, 5],
        'unit_price': [19.99, 29.99, 9.99],
        'order_date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03'])
    })
    
    # Simulate version 2 with schema changes
    orders_v2 = pd.DataFrame({
        'order_id': [1004, 1005, 1006],
        'customer_id': [104, 105, 106],
        'product_id': ['P001', 'P002', 'P003'],  # Changed from product_name to product_id
        'product_name': ['Widget D', 'Widget E', 'Widget F'],  # Still present
        'quantity': [3, 2, 1],
        'unit_price': [24.99, 34.99, 14.99],
        'order_date': pd.to_datetime(['2023-01-04', '2023-01-05', '2023-01-06']),
        'discount_percent': [10.0, 5.0, 0.0],  # New field
        'shipping_method': ['Standard', 'Express', 'Standard']  # New field
    })
    
    print("Orders v1 data:")
    print(orders_v1)
    print(f"Shape: {orders_v1.shape}")
    
    print("\nOrders v2 data:")
    print(orders_v2)
    print(f"Shape: {orders_v2.shape}")
    
    # Set up schema evolution manager
    evolution_manager = SchemaEvolutionManager()
    inference_engine = SchemaInferenceEngine()
    
    # Infer and register v1 schema
    v1_result = inference_engine.infer_schema(orders_v1)
    v1_schema = v1_result.inferred_schema
    v1_schema.schema_version = "1.0"
    v1_schema.schema_name = "Orders Schema"
    evolution_manager.add_schema_version(v1_schema)
    
    print(f"\nRegistered v1 schema with {len(v1_schema.fields)} fields")
    
    # Detect schema drift with v2 data
    drift_analysis = evolution_manager.detect_schema_drift(
        orders_v2, v1_schema, drift_threshold=0.1
    )
    
    print(f"\nSchema Drift Analysis:")
    print(f"Drift detected: {drift_analysis['drift_detected']}")
    print(f"Drift level: {drift_analysis['drift_level']}")
    print(f"Compatibility score: {drift_analysis['compatibility_score']:.2f}")
    
    # Analyze field changes
    field_changes = drift_analysis['field_changes']
    print(f"\nField Changes:")
    print(f"Added fields: {field_changes['added_fields']}")
    print(f"Removed fields: {field_changes['removed_fields']}")
    print(f"Modified fields: {len(field_changes['modified_fields'])}")
    
    # Register v2 schema
    v2_result = inference_engine.infer_schema(orders_v2)
    v2_schema = v2_result.inferred_schema
    v2_schema.schema_version = "2.0"
    v2_schema.schema_name = "Orders Schema"
    evolution_manager.add_schema_version(v2_schema)
    
    # Generate migration recommendations
    recommendations = drift_analysis['recommendations']
    print(f"\nMigration Recommendations:")
    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. {rec}")
    
    # Test backward compatibility
    compatibility_score = evolution_manager.calculate_compatibility_score(v1_schema, v2_schema)
    print(f"\nBackward compatibility score: {compatibility_score:.2f}")
    
    # Create migration strategy
    migration_strategy = {
        'version_from': '1.0',
        'version_to': '2.0',
        'compatibility_score': compatibility_score,
        'required_transformations': [],
        'optional_enhancements': []
    }
    
    # Add transformation rules based on changes
    if 'product_id' in field_changes['added_fields']:
        migration_strategy['optional_enhancements'].append({
            'field': 'product_id',
            'action': 'derive_from_product_name',
            'description': 'Generate product_id from product_name using lookup table'
        })
    
    if 'discount_percent' in field_changes['added_fields']:
        migration_strategy['optional_enhancements'].append({
            'field': 'discount_percent',
            'action': 'default_value',
            'value': 0.0,
            'description': 'Set default discount to 0% for historical orders'
        })
    
    print(f"\nMigration Strategy:")
    print(f"From: {migration_strategy['version_from']} to {migration_strategy['version_to']}")
    print(f"Required transformations: {len(migration_strategy['required_transformations'])}")
    print(f"Optional enhancements: {len(migration_strategy['optional_enhancements'])}")
    
    return migration_strategy


def example_3_multi_format_integration():
    """
    Example 3: Multi-Format Data Integration with Validation
    
    Shows validation across different data formats (pandas, numpy, sparse)
    in a unified data processing pipeline.
    """
    print("\n=== Example 3: Multi-Format Data Integration ===")
    
    # Create different format datasets
    # 1. Customer demographics (DataFrame)
    demographics = pd.DataFrame({
        'customer_id': [1, 2, 3, 4, 5],
        'age': [25, 35, 45, 28, 32],
        'income': [50000, 75000, 95000, 62000, 58000],
        'region': ['North', 'South', 'East', 'West', 'Central']
    })
    
    # 2. Feature matrix (NumPy array)
    # Synthetic features for ML model
    feature_matrix = np.random.rand(5, 10)  # 5 customers, 10 features
    
    # 3. Interaction matrix (Sparse matrix)
    # Customer-product interactions
    interaction_data = sparse.csr_matrix([
        [1, 0, 3, 0, 2],  # Customer 1 interactions
        [0, 2, 1, 0, 0],  # Customer 2 interactions
        [2, 1, 0, 1, 3],  # Customer 3 interactions
        [0, 0, 2, 2, 1],  # Customer 4 interactions
        [1, 1, 0, 0, 2]   # Customer 5 interactions
    ])
    
    print(f"Demographics shape: {demographics.shape}")
    print(f"Features shape: {feature_matrix.shape}")
    print(f"Interactions shape: {interaction_data.shape}")
    
    # Set up validation for each format
    inference_engine = SchemaInferenceEngine()
    validator = SchemaValidator()
    
    # Validate DataFrame
    df_result = inference_engine.infer_schema(demographics, DataFormat.PANDAS_DATAFRAME)
    df_validation = validator.validate_data(demographics, df_result.inferred_schema)
    
    print(f"\nDataFrame Validation:")
    print(f"Valid: {df_validation.is_valid}, Score: {df_validation.validation_score:.2f}")
    print(f"Schema fields: {list(df_result.inferred_schema.fields.keys())}")
    
    # Validate NumPy array
    np_result = inference_engine.infer_schema(feature_matrix, DataFormat.NUMPY_ARRAY)
    np_validation = validator.validate_data(feature_matrix, np_result.inferred_schema)
    
    print(f"\nNumPy Array Validation:")
    print(f"Valid: {np_validation.is_valid}, Score: {np_validation.validation_score:.2f}")
    print(f"Array type: {np_result.inferred_schema.fields['array_data']['type']}")
    print(f"Array shape: {np_result.inferred_schema.fields['array_data']['shape']}")
    
    # Validate Sparse matrix
    sparse_result = inference_engine.infer_schema(interaction_data, DataFormat.SCIPY_SPARSE)
    sparse_validation = validator.validate_data(interaction_data, sparse_result.inferred_schema)
    
    print(f"\nSparse Matrix Validation:")
    print(f"Valid: {sparse_validation.is_valid}, Score: {sparse_validation.validation_score:.2f}")
    print(f"Matrix format: {sparse_result.inferred_schema.fields['sparse_data']['format']}")
    print(f"Density: {sparse_result.inferred_schema.fields['sparse_data']['density']:.3f}")
    
    # Create integration validation report
    integration_report = {
        'formats_validated': 3,
        'total_validation_score': (df_validation.validation_score + 
                                 np_validation.validation_score + 
                                 sparse_validation.validation_score) / 3,
        'format_details': {
            'dataframe': {
                'valid': df_validation.is_valid,
                'score': df_validation.validation_score,
                'fields': len(df_result.inferred_schema.fields),
                'rows': len(demographics)
            },
            'numpy': {
                'valid': np_validation.is_valid,
                'score': np_validation.validation_score,
                'shape': feature_matrix.shape,
                'dtype': str(feature_matrix.dtype)
            },
            'sparse': {
                'valid': sparse_validation.is_valid,
                'score': sparse_validation.validation_score,
                'format': interaction_data.format,
                'density': sparse_result.inferred_schema.fields['sparse_data']['density']
            }
        }
    }
    
    print(f"\nIntegration Report:")
    print(f"Overall validation score: {integration_report['total_validation_score']:.2f}")
    print(f"All formats valid: {all(details['valid'] for details in integration_report['format_details'].values())}")
    
    return integration_report


def example_4_custom_scientific_validation():
    """
    Example 4: Custom Validation Rules for Domain-Specific Scientific Data
    
    Demonstrates creating custom validation rules for scientific datasets
    with domain-specific constraints.
    """
    print("\n=== Example 4: Scientific Data Validation ===")
    
    # Simulate experimental results data
    experiment_data = pd.DataFrame({
        'experiment_id': ['EXP001', 'EXP002', 'EXP003', 'EXP004', 'EXP005'],
        'temperature_celsius': [25.5, 37.0, -196.0, 100.0, 22.5],  # Including liquid nitrogen temp
        'pressure_bar': [1.013, 2.5, 0.001, 10.0, 1.0],
        'ph_level': [7.0, 6.5, 8.2, 14.5, 7.4],  # Note: pH 14.5 is invalid (>14)
        'concentration_molarity': [0.1, 0.5, 1.0, -0.2, 0.75],  # Negative concentration invalid
        'reaction_time_minutes': [30, 45, 60, 15, 90],
        'yield_percentage': [85.5, 92.3, 78.9, 101.2, 88.7],  # >100% yield is suspicious
        'researcher': ['Dr. Smith', 'Dr. Johnson', 'Dr. Brown', 'Dr. Wilson', 'Dr. Davis']
    })
    
    print("Scientific experiment data:")
    print(experiment_data)
    
    # Create custom validation rules for scientific data
    class TemperatureValidationRule(ValidationRule):
        def __init__(self):
            super().__init__("temperature_validation", ValidationRuleType.RANGE_CHECK,
                           "Temperature validation for experimental conditions")
        
        def validate(self, field_name: str, value: Any, context: Dict[str, Any]) -> List[ValidationError]:
            if pd.isna(value):
                return []
            
            errors = []
            try:
                temp = float(value)
                # Absolute zero in Celsius is -273.15°C
                if temp < -273.15:
                    errors.append(ValidationError(
                        error_id=f"temp_absolute_zero_{field_name}_{id(value)}",
                        error_type=ValidationRuleType.RANGE_CHECK,
                        field_name=field_name,
                        expected_value="> -273.15°C",
                        actual_value=temp,
                        error_message=f"Temperature {temp}°C is below absolute zero (-273.15°C)"
                    ))
                # Extreme high temperature warning (above water boiling at normal pressure)
                elif temp > 1000:
                    errors.append(ValidationError(
                        error_id=f"temp_extreme_{field_name}_{id(value)}",
                        error_type=ValidationRuleType.RANGE_CHECK,
                        field_name=field_name,
                        expected_value="< 1000°C",
                        actual_value=temp,
                        error_message=f"Extremely high temperature {temp}°C - verify measurement",
                        severity="warning"
                    ))
            except (ValueError, TypeError):
                errors.append(ValidationError(
                    error_id=f"temp_type_{field_name}_{id(value)}",
                    error_type=ValidationRuleType.TYPE_CHECK,
                    field_name=field_name,
                    expected_value="numeric temperature",
                    actual_value=str(value),
                    error_message=f"Temperature must be numeric, got: {value}"
                ))
            return errors
        
        def is_applicable(self, field_name: str, field_schema: Dict[str, Any]) -> bool:
            return 'temperature' in field_name.lower()
    
    class PhValidationRule(ValidationRule):
        def __init__(self):
            super().__init__("ph_validation", ValidationRuleType.RANGE_CHECK,
                           "pH level validation (0-14 scale)")
        
        def validate(self, field_name: str, value: Any, context: Dict[str, Any]) -> List[ValidationError]:
            if pd.isna(value):
                return []
            
            errors = []
            try:
                ph = float(value)
                if ph < 0 or ph > 14:
                    errors.append(ValidationError(
                        error_id=f"ph_range_{field_name}_{id(value)}",
                        error_type=ValidationRuleType.RANGE_CHECK,
                        field_name=field_name,
                        expected_value="0-14",
                        actual_value=ph,
                        error_message=f"pH {ph} outside valid range (0-14)"
                    ))
            except (ValueError, TypeError):
                errors.append(ValidationError(
                    error_id=f"ph_type_{field_name}_{id(value)}",
                    error_type=ValidationRuleType.TYPE_CHECK,
                    field_name=field_name,
                    expected_value="numeric pH value",
                    actual_value=str(value),
                    error_message=f"pH must be numeric, got: {value}"
                ))
            return errors
        
        def is_applicable(self, field_name: str, field_schema: Dict[str, Any]) -> bool:
            return 'ph' in field_name.lower()
    
    class ConcentrationValidationRule(ValidationRule):
        def __init__(self):
            super().__init__("concentration_validation", ValidationRuleType.RANGE_CHECK,
                           "Chemical concentration validation (non-negative)")
        
        def validate(self, field_name: str, value: Any, context: Dict[str, Any]) -> List[ValidationError]:
            if pd.isna(value):
                return []
            
            errors = []
            try:
                conc = float(value)
                if conc < 0:
                    errors.append(ValidationError(
                        error_id=f"conc_negative_{field_name}_{id(value)}",
                        error_type=ValidationRuleType.RANGE_CHECK,
                        field_name=field_name,
                        expected_value=">= 0",
                        actual_value=conc,
                        error_message=f"Concentration cannot be negative: {conc}"
                    ))
                elif conc > 100:  # Very high concentration warning
                    errors.append(ValidationError(
                        error_id=f"conc_high_{field_name}_{id(value)}",
                        error_type=ValidationRuleType.RANGE_CHECK,
                        field_name=field_name,
                        expected_value="< 100 M",
                        actual_value=conc,
                        error_message=f"Very high concentration {conc} M - verify measurement",
                        severity="warning"
                    ))
            except (ValueError, TypeError):
                errors.append(ValidationError(
                    error_id=f"conc_type_{field_name}_{id(value)}",
                    error_type=ValidationRuleType.TYPE_CHECK,
                    field_name=field_name,
                    expected_value="numeric concentration",
                    actual_value=str(value),
                    error_message=f"Concentration must be numeric, got: {value}"
                ))
            return errors
        
        def is_applicable(self, field_name: str, field_schema: Dict[str, Any]) -> bool:
            return 'concentration' in field_name.lower()
    
    class YieldValidationRule(ValidationRule):
        def __init__(self):
            super().__init__("yield_validation", ValidationRuleType.RANGE_CHECK,
                           "Reaction yield validation (0-100%)")
        
        def validate(self, field_name: str, value: Any, context: Dict[str, Any]) -> List[ValidationError]:
            if pd.isna(value):
                return []
            
            errors = []
            try:
                yield_pct = float(value)
                if yield_pct < 0:
                    errors.append(ValidationError(
                        error_id=f"yield_negative_{field_name}_{id(value)}",
                        error_type=ValidationRuleType.RANGE_CHECK,
                        field_name=field_name,
                        expected_value=">= 0%",
                        actual_value=yield_pct,
                        error_message=f"Yield cannot be negative: {yield_pct}%"
                    ))
                elif yield_pct > 100:
                    errors.append(ValidationError(
                        error_id=f"yield_over100_{field_name}_{id(value)}",
                        error_type=ValidationRuleType.RANGE_CHECK,
                        field_name=field_name,
                        expected_value="<= 100%",
                        actual_value=yield_pct,
                        error_message=f"Yield >100% is suspicious: {yield_pct}% - check calculation",
                        severity="warning"
                    ))
            except (ValueError, TypeError):
                errors.append(ValidationError(
                    error_id=f"yield_type_{field_name}_{id(value)}",
                    error_type=ValidationRuleType.TYPE_CHECK,
                    field_name=field_name,
                    expected_value="numeric yield percentage",
                    actual_value=str(value),
                    error_message=f"Yield must be numeric, got: {value}"
                ))
            return errors
        
        def is_applicable(self, field_name: str, field_schema: Dict[str, Any]) -> bool:
            return 'yield' in field_name.lower()
    
    # Set up validator with scientific rules
    inference_engine = SchemaInferenceEngine()
    validator = SchemaValidator()
    
    # Add all custom scientific validation rules
    validator.add_custom_rule(TemperatureValidationRule())
    validator.add_custom_rule(PhValidationRule())
    validator.add_custom_rule(ConcentrationValidationRule())
    validator.add_custom_rule(YieldValidationRule())
    
    # Infer schema and validate
    schema_result = inference_engine.infer_schema(experiment_data)
    validation_result = validator.validate_data(
        experiment_data, 
        schema_result.inferred_schema,
        validation_level=SchemaValidationLevel.STRICT
    )
    
    print(f"\nScientific Data Validation Results:")
    print(f"Overall valid: {validation_result.is_valid}")
    print(f"Validation score: {validation_result.validation_score:.2f}")
    print(f"Errors: {len(validation_result.errors)}, Warnings: {len(validation_result.warnings)}")
    
    # Categorize scientific validation issues
    scientific_issues = {
        'temperature_issues': [],
        'ph_issues': [],
        'concentration_issues': [],
        'yield_issues': [],
        'other_issues': []
    }
    
    for error in validation_result.errors + validation_result.warnings:
        if 'temperature' in error.field_name.lower():
            scientific_issues['temperature_issues'].append(error)
        elif 'ph' in error.field_name.lower():
            scientific_issues['ph_issues'].append(error)
        elif 'concentration' in error.field_name.lower():
            scientific_issues['concentration_issues'].append(error)
        elif 'yield' in error.field_name.lower():
            scientific_issues['yield_issues'].append(error)
        else:
            scientific_issues['other_issues'].append(error)
    
    print(f"\nScientific Validation Issue Summary:")
    for issue_type, issues in scientific_issues.items():
        if issues:
            print(f"  {issue_type.replace('_', ' ').title()}: {len(issues)} issues")
            for issue in issues[:2]:  # Show first 2 of each type
                print(f"    - {issue.field_name}: {issue.error_message}")
    
    # Generate experimental data quality report
    quality_metrics = {
        'total_experiments': len(experiment_data),
        'valid_experiments': sum(1 for _, row in experiment_data.iterrows() 
                               if not any(error.row_index == row.name 
                                        for error in validation_result.errors)),
        'data_integrity_score': validation_result.validation_score,
        'critical_issues': len([e for e in validation_result.errors if e.severity == 'error']),
        'warnings': len([e for e in validation_result.errors if e.severity == 'warning']),
        'recommended_actions': []
    }
    
    # Generate recommendations
    if scientific_issues['ph_issues']:
        quality_metrics['recommended_actions'].append("Review pH measurements - values outside 0-14 range detected")
    if scientific_issues['concentration_issues']:
        quality_metrics['recommended_actions'].append("Check concentration calculations - negative values found")
    if scientific_issues['yield_issues']:
        quality_metrics['recommended_actions'].append("Verify yield calculations - values >100% detected")
    
    print(f"\nExperimental Data Quality Report:")
    print(f"Total experiments: {quality_metrics['total_experiments']}")
    print(f"Valid experiments: {quality_metrics['valid_experiments']}")
    print(f"Data integrity: {quality_metrics['data_integrity_score']:.1%}")
    print(f"Critical issues: {quality_metrics['critical_issues']}")
    print(f"Warnings: {quality_metrics['warnings']}")
    
    if quality_metrics['recommended_actions']:
        print(f"\nRecommended Actions:")
        for i, action in enumerate(quality_metrics['recommended_actions'], 1):
            print(f"  {i}. {action}")
    
    return quality_metrics


def example_5_streaming_validation():
    """
    Example 5: Streaming Data Validation
    
    Demonstrates validating data in streaming/batch scenarios where
    data arrives continuously and needs real-time validation.
    """
    print("\n=== Example 5: Streaming Data Validation ===")
    
    # Simulate streaming data arrival
    def generate_streaming_batch(batch_id: int, batch_size: int = 100):
        """Generate a batch of streaming data."""
        np.random.seed(batch_id)  # For reproducible results
        
        # Simulate some data quality issues in random batches
        error_rate = 0.05 if batch_id % 10 != 0 else 0.2  # Higher errors every 10th batch
        
        data = []
        for i in range(batch_size):
            record = {
                'timestamp': datetime.now() - timedelta(minutes=batch_size-i),
                'user_id': np.random.randint(1000, 9999),
                'sensor_value': np.random.normal(50, 10),
                'status': np.random.choice(['active', 'inactive'], p=[0.8, 0.2])
            }
            
            # Introduce errors randomly
            if np.random.random() < error_rate:
                if np.random.random() < 0.5:
                    record['sensor_value'] = np.random.choice([-999, 999, np.nan])  # Outlier or missing
                else:
                    record['user_id'] = None  # Missing user ID
            
            data.append(record)
        
        return pd.DataFrame(data)
    
    # Set up streaming validation pipeline
    inference_engine = SchemaInferenceEngine()
    validator = SchemaValidator()
    
    # Create reference schema from initial clean batch
    reference_batch = generate_streaming_batch(0, batch_size=500)
    reference_schema_result = inference_engine.infer_schema(reference_batch)
    reference_schema = reference_schema_result.inferred_schema
    
    print(f"Reference schema established from {len(reference_batch)} records")
    print(f"Schema confidence: {reference_schema_result.confidence_score:.2f}")
    
    # Process streaming batches
    batch_results = []
    validation_stats = {
        'total_batches': 0,
        'total_records': 0,
        'valid_records': 0,
        'error_count': 0,
        'avg_validation_time': 0.0,
        'quality_trend': []
    }
    
    print(f"\nProcessing streaming batches...")
    
    for batch_id in range(1, 11):  # Process 10 batches
        batch_data = generate_streaming_batch(batch_id, batch_size=50)
        
        # Validate batch
        start_time = time.time()
        batch_validation = validator.validate_data(
            batch_data, 
            reference_schema,
            validation_level=SchemaValidationLevel.STANDARD
        )
        validation_time = time.time() - start_time
        
        # Update statistics
        validation_stats['total_batches'] += 1
        validation_stats['total_records'] += len(batch_data)
        validation_stats['valid_records'] += batch_validation.passed_checks
        validation_stats['error_count'] += len(batch_validation.errors)
        validation_stats['avg_validation_time'] += validation_time
        validation_stats['quality_trend'].append(batch_validation.validation_score)
        
        batch_result = {
            'batch_id': batch_id,
            'record_count': len(batch_data),
            'validation_score': batch_validation.validation_score,
            'error_count': len(batch_validation.errors),
            'validation_time': validation_time,
            'conformance_level': batch_validation.conformance_level.value
        }
        batch_results.append(batch_result)
        
        # Real-time alerts for poor quality batches
        if batch_validation.validation_score < 0.8:
            print(f"  🚨 ALERT: Batch {batch_id} - Low quality score: {batch_validation.validation_score:.2f}")
        elif len(batch_validation.errors) > 5:
            print(f"  ⚠️  WARNING: Batch {batch_id} - High error count: {len(batch_validation.errors)}")
        else:
            print(f"  ✅ Batch {batch_id} - Quality: {batch_validation.validation_score:.2f}, Errors: {len(batch_validation.errors)}")
    
    # Calculate final statistics
    validation_stats['avg_validation_time'] /= validation_stats['total_batches']
    validation_stats['overall_quality'] = sum(validation_stats['quality_trend']) / len(validation_stats['quality_trend'])
    validation_stats['quality_variance'] = np.var(validation_stats['quality_trend'])
    
    print(f"\n📊 Streaming Validation Summary:")
    print(f"Total batches processed: {validation_stats['total_batches']}")
    print(f"Total records validated: {validation_stats['total_records']}")
    print(f"Overall data quality: {validation_stats['overall_quality']:.1%}")
    print(f"Total errors found: {validation_stats['error_count']}")
    print(f"Average validation time: {validation_stats['avg_validation_time']:.3f}s per batch")
    print(f"Quality consistency (low variance is good): {validation_stats['quality_variance']:.3f}")
    
    # Quality trend analysis
    trend_direction = "improving" if validation_stats['quality_trend'][-1] > validation_stats['quality_trend'][0] else "declining"
    print(f"Quality trend: {trend_direction}")
    
    # Performance metrics
    records_per_second = validation_stats['total_records'] / (validation_stats['avg_validation_time'] * validation_stats['total_batches'])
    print(f"Validation throughput: {records_per_second:.0f} records/second")
    
    return {
        'validation_stats': validation_stats,
        'batch_results': batch_results,
        'reference_schema': reference_schema
    }


def example_6_production_drift_monitoring():
    """
    Example 6: Schema Drift Detection in Production
    
    Simulates a production environment where data schemas may drift
    over time and need continuous monitoring.
    """
    print("\n=== Example 6: Production Schema Drift Monitoring ===")
    
    # Simulate production data over different time periods
    def generate_production_data(period: str) -> pd.DataFrame:
        """Generate production data for different time periods."""
        base_data = {
            'user_id': range(1000, 1100),
            'session_duration': np.random.exponential(300, 100),  # Average 5 minutes
            'pages_viewed': np.random.poisson(5, 100),
            'conversion': np.random.choice([True, False], 100, p=[0.15, 0.85])
        }
        
        if period == 'baseline':
            # Original schema
            data = pd.DataFrame(base_data)
        
        elif period == 'month_1':
            # Minor changes - new optional field
            data = pd.DataFrame(base_data)
            data['device_type'] = np.random.choice(['mobile', 'desktop', 'tablet'], 100, p=[0.6, 0.3, 0.1])
        
        elif period == 'month_2':
            # Schema drift - field type changes
            data = pd.DataFrame(base_data)
            data['device_type'] = np.random.choice(['mobile', 'desktop', 'tablet'], 100, p=[0.6, 0.3, 0.1])
            # Conversion becomes a probability score instead of boolean
            data['conversion'] = np.random.uniform(0, 1, 100)
        
        elif period == 'month_3':
            # Major drift - field removal and additions
            data = pd.DataFrame({
                'user_id': base_data['user_id'],
                'session_duration_seconds': base_data['session_duration'],  # Renamed field
                'pages_viewed': base_data['pages_viewed'],
                'device_type': np.random.choice(['mobile', 'desktop', 'tablet'], 100, p=[0.6, 0.3, 0.1]),
                'conversion_score': np.random.uniform(0, 1, 100),  # Renamed from conversion
                'user_segment': np.random.choice(['premium', 'standard', 'free'], 100),  # New field
                'referrer_source': np.random.choice(['organic', 'paid', 'social', 'email'], 100)  # New field
            })
        
        return data
    
    # Set up drift monitoring system
    evolution_manager = SchemaEvolutionManager()
    inference_engine = SchemaInferenceEngine()
    
    # Establish baseline schema
    baseline_data = generate_production_data('baseline')
    baseline_result = inference_engine.infer_schema(baseline_data)
    baseline_schema = baseline_result.inferred_schema
    baseline_schema.schema_version = "1.0"
    baseline_schema.schema_name = "Production Analytics Schema"
    evolution_manager.add_schema_version(baseline_schema)
    
    print(f"Baseline schema established: {len(baseline_schema.fields)} fields")
    
    # Monitor drift over time
    periods = ['month_1', 'month_2', 'month_3']
    drift_history = []
    
    for i, period in enumerate(periods, 1):
        print(f"\n--- Monitoring Period: {period.replace('_', ' ').title()} ---")
        
        current_data = generate_production_data(period)
        
        # Detect drift
        drift_analysis = evolution_manager.detect_schema_drift(
            current_data, 
            baseline_schema,
            drift_threshold=0.1
        )
        
        # Analyze drift severity
        compatibility_score = drift_analysis['compatibility_score']
        drift_level = drift_analysis['drift_level']
        field_changes = drift_analysis['field_changes']
        
        print(f"Drift Status: {'🚨 DETECTED' if drift_analysis['drift_detected'] else '✅ None'}")
        print(f"Drift Level: {drift_level}")
        print(f"Compatibility Score: {compatibility_score:.2f}")
        
        # Detail changes
        if field_changes['added_fields']:
            print(f"Added Fields: {field_changes['added_fields']}")
        if field_changes['removed_fields']:
            print(f"Removed Fields: {field_changes['removed_fields']}")
        if field_changes['modified_fields']:
            print(f"Modified Fields: {len(field_changes['modified_fields'])}")
        
        # Generate drift severity assessment
        if compatibility_score >= 0.9:
            severity = "LOW"
            action = "Continue monitoring"
        elif compatibility_score >= 0.7:
            severity = "MEDIUM"
            action = "Review changes, update documentation"
        elif compatibility_score >= 0.5:
            severity = "HIGH"
            action = "Plan schema migration, notify stakeholders"
        else:
            severity = "CRITICAL"
            action = "Immediate action required, potential data loss"
        
        print(f"Severity: {severity}")
        print(f"Recommended Action: {action}")
        
        # Register new schema version if significant drift
        if drift_analysis['drift_detected'] and compatibility_score < 0.8:
            new_result = inference_engine.infer_schema(current_data)
            new_schema = new_result.inferred_schema
            new_schema.schema_version = f"{1 + i}.0"
            new_schema.schema_name = "Production Analytics Schema"
            evolution_manager.add_schema_version(new_schema)
            print(f"Registered new schema version: {new_schema.schema_version}")
        
        # Store drift history
        drift_record = {
            'period': period,
            'compatibility_score': compatibility_score,
            'drift_level': drift_level,
            'drift_detected': drift_analysis['drift_detected'],
            'added_fields': len(field_changes['added_fields']),
            'removed_fields': len(field_changes['removed_fields']),
            'modified_fields': len(field_changes['modified_fields']),
            'severity': severity,
            'action': action
        }
        drift_history.append(drift_record)
        
        # Show recommendations
        recommendations = drift_analysis['recommendations']
        if recommendations:
            print("Recommendations:")
            for j, rec in enumerate(recommendations[:3], 1):  # Show top 3
                print(f"  {j}. {rec}")
    
    # Generate drift monitoring report
    print(f"\n📈 Drift Monitoring Report Summary:")
    print(f"Monitoring periods: {len(periods)}")
    
    # Calculate drift metrics
    avg_compatibility = sum(record['compatibility_score'] for record in drift_history) / len(drift_history)
    drift_incidents = sum(1 for record in drift_history if record['drift_detected'])
    schema_versions = len(evolution_manager.schema_versions.get("Production Analytics Schema", []))
    
    print(f"Average compatibility: {avg_compatibility:.2f}")
    print(f"Drift incidents: {drift_incidents}/{len(periods)}")
    print(f"Schema versions created: {schema_versions}")
    
    # Trend analysis
    compatibility_trend = [record['compatibility_score'] for record in drift_history]
    if len(compatibility_trend) >= 2:
        trend_direction = "deteriorating" if compatibility_trend[-1] < compatibility_trend[0] else "stable/improving"
        print(f"Compatibility trend: {trend_direction}")
    
    # Risk assessment
    latest_compatibility = drift_history[-1]['compatibility_score']
    if latest_compatibility < 0.5:
        risk_level = "HIGH - Immediate attention required"
    elif latest_compatibility < 0.7:
        risk_level = "MEDIUM - Plan migration soon"
    else:
        risk_level = "LOW - Continue monitoring"
    
    print(f"Current risk level: {risk_level}")
    
    return {
        'baseline_schema': baseline_schema,
        'drift_history': drift_history,
        'final_compatibility': latest_compatibility,
        'schema_versions': schema_versions
    }


def main():
    """
    Run all schema validation examples to demonstrate the system capabilities.
    """
    print("🚀 LocalData MCP v2.0 - Schema Validation System Examples")
    print("=" * 60)
    
    try:
        # Run all examples
        examples = [
            example_1_data_quality_pipeline,
            example_2_schema_evolution_etl,
            example_3_multi_format_integration,
            example_4_custom_scientific_validation,
            example_5_streaming_validation,
            example_6_production_drift_monitoring
        ]
        
        results = {}
        
        for example_func in examples:
            example_name = example_func.__name__
            print(f"\n{'='*60}")
            print(f"Running {example_name}...")
            
            try:
                start_time = time.time()
                result = example_func()
                execution_time = time.time() - start_time
                
                results[example_name] = {
                    'status': 'success',
                    'result': result,
                    'execution_time': execution_time
                }
                
                print(f"✅ {example_name} completed in {execution_time:.2f}s")
                
            except Exception as e:
                results[example_name] = {
                    'status': 'error',
                    'error': str(e),
                    'execution_time': 0
                }
                print(f"❌ {example_name} failed: {e}")
        
        # Summary
        print(f"\n{'='*60}")
        print("📊 EXECUTION SUMMARY")
        print(f"{'='*60}")
        
        successful = sum(1 for r in results.values() if r['status'] == 'success')
        total = len(results)
        total_time = sum(r['execution_time'] for r in results.values())
        
        print(f"Examples completed: {successful}/{total}")
        print(f"Total execution time: {total_time:.2f}s")
        print(f"Success rate: {successful/total:.1%}")
        
        if successful == total:
            print("🎉 All examples completed successfully!")
        else:
            print("⚠️  Some examples failed - check logs above")
            
        return results
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Execution interrupted by user")
        return None
    except Exception as e:
        print(f"\n\n💥 Unexpected error: {e}")
        return None


if __name__ == "__main__":
    main()