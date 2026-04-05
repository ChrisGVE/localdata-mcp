"""
SchemaValidator class.
"""

import time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

from ..interfaces import DataFormat
from ._types import (
    SchemaConformanceLevel,
    SchemaValidationLevel,
    SchemaValidationResult,
    ValidationError,
    ValidationRuleType,
)
from ._schema import DataSchema
from ._rules import (
    ValidationRule,
    TypeValidationRule,
    RangeValidationRule,
    NullValidationRule,
)
from ._types import SchemaConstraint
from ....logging_manager import get_logger

logger = get_logger(__name__)


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

    def validate_data(
        self,
        data: Any,
        schema: DataSchema,
        validation_level: SchemaValidationLevel = SchemaValidationLevel.STANDARD,
        max_errors: Optional[int] = None,
    ) -> SchemaValidationResult:
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
                errors, warnings, total_checks, passed_checks = (
                    self._validate_dataframe(data, schema, validation_level, max_errors)
                )
            elif schema.data_format == DataFormat.NUMPY_ARRAY:
                errors, warnings, total_checks, passed_checks = self._validate_numpy(
                    data, schema, validation_level, max_errors
                )
            else:
                errors, warnings, total_checks, passed_checks = self._validate_generic(
                    data, schema, validation_level, max_errors
                )

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
                schema_version=schema.schema_version,
            )

            logger.info(
                f"Schema validation completed: {validation_score:.2f} score, "
                f"{len(errors)} errors, {len(warnings)} warnings"
            )

            return result

        except Exception as e:
            logger.error(f"Schema validation failed with exception: {e}")

            # Return failed validation result
            return SchemaValidationResult(
                is_valid=False,
                conformance_level=SchemaConformanceLevel.INCOMPATIBLE,
                validation_score=0.0,
                errors=[
                    ValidationError(
                        error_id="validation_exception",
                        error_type=ValidationRuleType.CUSTOM,
                        field_name="__validation__",
                        expected_value="successful validation",
                        actual_value="exception",
                        error_message=str(e),
                        severity="error",
                    )
                ],
                validation_time=time.time() - start_time,
                validation_level=validation_level,
            )

    def _validate_dataframe(
        self,
        data: pd.DataFrame,
        schema: DataSchema,
        validation_level: SchemaValidationLevel,
        max_errors: Optional[int],
    ) -> Tuple[List[ValidationError], List[ValidationError], int, int]:
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
                error_message=f"Expected DataFrame, got {type(data).__name__}",
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
            if not field_schema.get("nullable", True):
                error = ValidationError(
                    error_id=f"missing_column_{col}",
                    error_type=ValidationRuleType.TYPE_CHECK,
                    field_name=col,
                    expected_value="column present",
                    actual_value="column missing",
                    error_message=f"Required column '{col}' is missing from data",
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
                    severity="warning",
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
                field_errors = self._validate_field_value(
                    col, value, field_schema, {"row_index": idx}
                )

                if field_errors:
                    errors.extend(field_errors)
                else:
                    passed_checks += 1

        # Validate schema constraints
        constraint_errors, constraint_warnings, constraint_total, constraint_passed = (
            self._validate_constraints(data, schema, max_errors)
        )

        errors.extend(constraint_errors)
        warnings.extend(constraint_warnings)
        total_checks += constraint_total
        passed_checks += constraint_passed

        return errors, warnings, total_checks, passed_checks

    def _validate_numpy(
        self,
        data: np.ndarray,
        schema: DataSchema,
        validation_level: SchemaValidationLevel,
        max_errors: Optional[int],
    ) -> Tuple[List[ValidationError], List[ValidationError], int, int]:
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
                error_message=f"Expected numpy array, got {type(data).__name__}",
            )
            errors.append(error)
            return errors, warnings, 1, 0

        # Validate array properties from schema
        if "array_data" in schema.fields:
            field_schema = schema.fields["array_data"]

            # Check shape if specified
            expected_shape = field_schema.get("shape")
            if expected_shape and data.shape != expected_shape:
                error = ValidationError(
                    error_id="shape_mismatch",
                    error_type=ValidationRuleType.TYPE_CHECK,
                    field_name="shape",
                    expected_value=expected_shape,
                    actual_value=data.shape,
                    error_message=f"Shape mismatch: expected {expected_shape}, got {data.shape}",
                )
                errors.append(error)

            total_checks += 1
            if not errors:
                passed_checks += 1

        return errors, warnings, total_checks, passed_checks

    def _validate_generic(
        self,
        data: Any,
        schema: DataSchema,
        validation_level: SchemaValidationLevel,
        max_errors: Optional[int],
    ) -> Tuple[List[ValidationError], List[ValidationError], int, int]:
        """Validate generic data against schema."""
        errors = []
        warnings = []
        total_checks = 1
        passed_checks = 0

        # Basic type validation
        if schema.fields:
            first_field = next(iter(schema.fields.values()))
            expected_type = first_field.get("python_type", "object")
            actual_type = type(data).__name__

            if expected_type != "object" and actual_type != expected_type:
                error = ValidationError(
                    error_id="type_mismatch",
                    error_type=ValidationRuleType.TYPE_CHECK,
                    field_name="__data__",
                    expected_value=expected_type,
                    actual_value=actual_type,
                    error_message=f"Type mismatch: expected {expected_type}, got {actual_type}",
                )
                errors.append(error)
            else:
                passed_checks += 1
        else:
            passed_checks += 1  # No schema to validate against

        return errors, warnings, total_checks, passed_checks

    def _validate_field_value(
        self,
        field_name: str,
        value: Any,
        field_schema: Dict[str, Any],
        context: Dict[str, Any],
    ) -> List[ValidationError]:
        """Validate a single field value against its schema definition."""
        errors = []

        # Type validation
        expected_type = field_schema.get("type")
        if expected_type:
            type_rule = TypeValidationRule(expected_type)
            type_errors = type_rule.validate(field_name, value, field_schema)
            errors.extend(type_errors)

        # Null validation
        nullable = field_schema.get("nullable", True)
        null_rule = NullValidationRule(allow_nulls=nullable)
        null_errors = null_rule.validate(field_name, value, field_schema)
        errors.extend(null_errors)

        # Range validation for numeric types
        if expected_type in ["int", "float", "numeric"]:
            min_val = field_schema.get("min")
            max_val = field_schema.get("max")
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

    def _validate_constraints(
        self, data: Any, schema: DataSchema, max_errors: Optional[int]
    ) -> Tuple[List[ValidationError], List[ValidationError], int, int]:
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
                    error_message=f"Constraint validation failed: {e}",
                )
                errors.append(error)

        return errors, warnings, total_checks, passed_checks

    def _validate_single_constraint(
        self, data: Any, constraint: SchemaConstraint
    ) -> List[ValidationError]:
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
                        error_message=constraint.error_message
                        or f"Value {value} violates minimum constraint",
                        row_index=idx,
                    )
                    errors.append(error)

        elif constraint.constraint_type == ValidationRuleType.NULL_CHECK:
            # Null constraint validation
            if isinstance(data, pd.DataFrame) and constraint.field_name in data.columns:
                column_data = data[constraint.field_name]
                null_values = column_data.isnull()

                if (
                    not constraint.constraint_value and null_values.any()
                ):  # constraint_value=False means no nulls allowed
                    null_indices = null_values[null_values].index
                    for idx in null_indices:
                        error = ValidationError(
                            error_id=f"null_violation_{constraint.constraint_id}_{idx}",
                            error_type=ValidationRuleType.NULL_CHECK,
                            field_name=constraint.field_name,
                            expected_value="non-null",
                            actual_value="null",
                            error_message=constraint.error_message
                            or f"Null value not allowed in {constraint.field_name}",
                            row_index=idx,
                        )
                        errors.append(error)

        return errors
