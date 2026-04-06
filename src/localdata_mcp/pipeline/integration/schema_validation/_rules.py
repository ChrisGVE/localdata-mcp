"""
Validation rule classes: ABC and concrete implementations.
"""

from abc import ABC, abstractmethod
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from ._types import ValidationError, ValidationRuleType


class ValidationRule(ABC):
    """Abstract base class for validation rules."""

    def __init__(
        self, rule_id: str, rule_type: ValidationRuleType, description: str = ""
    ):
        self.rule_id = rule_id
        self.rule_type = rule_type
        self.description = description
        self.is_enabled = True
        self.priority = 0

    @abstractmethod
    def validate(
        self, field_name: str, value: Any, context: Dict[str, Any]
    ) -> List[ValidationError]:
        """Execute the validation rule."""
        pass

    @abstractmethod
    def is_applicable(self, field_name: str, field_schema: Dict[str, Any]) -> bool:
        """Check if this rule applies to the given field."""
        pass


class TypeValidationRule(ValidationRule):
    """Validates field data types."""

    def __init__(self, expected_type: Union[type, str], allow_coercion: bool = True):
        super().__init__(
            "type_validation",
            ValidationRuleType.TYPE_CHECK,
            f"Type validation for {expected_type}",
        )
        self.expected_type = expected_type
        self.allow_coercion = allow_coercion

    def validate(
        self, field_name: str, value: Any, context: Dict[str, Any]
    ) -> List[ValidationError]:
        """Validate field type."""
        errors = []

        if pd.isna(value) and context.get("nullable", True):
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
                error_message=f"Expected {expected_type_name}, got {type(value).__name__}",
            )
            errors.append(error)

        return errors

    def is_applicable(self, field_name: str, field_schema: Dict[str, Any]) -> bool:
        """Check if type validation applies."""
        return "type" in field_schema

    def _check_string_type(self, value: Any, type_name: str) -> bool:
        """Check type based on string type name."""
        type_mapping = {
            "int": int,
            "float": float,
            "str": str,
            "bool": bool,
            "datetime": (datetime, date, pd.Timestamp),
            "numeric": (int, float, np.number),
        }

        expected_types = type_mapping.get(type_name, str)
        if isinstance(expected_types, tuple):
            return isinstance(value, expected_types)
        else:
            return isinstance(value, expected_types)


class RangeValidationRule(ValidationRule):
    """Validates numeric range constraints."""

    def __init__(
        self,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        inclusive: bool = True,
    ):
        super().__init__(
            "range_validation",
            ValidationRuleType.RANGE_CHECK,
            f"Range validation [{min_value}, {max_value}]",
        )
        self.min_value = min_value
        self.max_value = max_value
        self.inclusive = inclusive

    def validate(
        self, field_name: str, value: Any, context: Dict[str, Any]
    ) -> List[ValidationError]:
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
                errors.append(
                    ValidationError(
                        error_id=f"range_min_error_{field_name}_{id(value)}",
                        error_type=ValidationRuleType.RANGE_CHECK,
                        field_name=field_name,
                        expected_value=f">= {self.min_value}",
                        actual_value=numeric_value,
                        error_message=f"Value {numeric_value} is below minimum {self.min_value}",
                    )
                )
            elif not self.inclusive and numeric_value <= self.min_value:
                errors.append(
                    ValidationError(
                        error_id=f"range_min_error_{field_name}_{id(value)}",
                        error_type=ValidationRuleType.RANGE_CHECK,
                        field_name=field_name,
                        expected_value=f"> {self.min_value}",
                        actual_value=numeric_value,
                        error_message=f"Value {numeric_value} is not above minimum {self.min_value}",
                    )
                )

        # Check maximum value
        if self.max_value is not None:
            if self.inclusive and numeric_value > self.max_value:
                errors.append(
                    ValidationError(
                        error_id=f"range_max_error_{field_name}_{id(value)}",
                        error_type=ValidationRuleType.RANGE_CHECK,
                        field_name=field_name,
                        expected_value=f"<= {self.max_value}",
                        actual_value=numeric_value,
                        error_message=f"Value {numeric_value} is above maximum {self.max_value}",
                    )
                )
            elif not self.inclusive and numeric_value >= self.max_value:
                errors.append(
                    ValidationError(
                        error_id=f"range_max_error_{field_name}_{id(value)}",
                        error_type=ValidationRuleType.RANGE_CHECK,
                        field_name=field_name,
                        expected_value=f"< {self.max_value}",
                        actual_value=numeric_value,
                        error_message=f"Value {numeric_value} is not below maximum {self.max_value}",
                    )
                )

        return errors

    def is_applicable(self, field_name: str, field_schema: Dict[str, Any]) -> bool:
        """Check if range validation applies."""
        field_type = field_schema.get("type", "")
        return (
            field_type in ["int", "float", "numeric"]
            or "min" in field_schema
            or "max" in field_schema
        )


class NullValidationRule(ValidationRule):
    """Validates null/missing value constraints."""

    def __init__(self, allow_nulls: bool = True):
        super().__init__(
            "null_validation",
            ValidationRuleType.NULL_CHECK,
            f"Null validation (allow_nulls={allow_nulls})",
        )
        self.allow_nulls = allow_nulls

    def validate(
        self, field_name: str, value: Any, context: Dict[str, Any]
    ) -> List[ValidationError]:
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
                error_message=f"Field '{field_name}' cannot be null",
            )
            errors.append(error)

        return errors

    def is_applicable(self, field_name: str, field_schema: Dict[str, Any]) -> bool:
        """Check if null validation applies."""
        return "nullable" in field_schema or "required" in field_schema
