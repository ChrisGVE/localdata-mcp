"""
Metadata validation, preservation rules, and internal helpers for MetadataManager.

Provides the _MetadataValidationMixin with validation, preservation rule application,
merge conflict resolution, metadata change calculation, and lineage ID generation.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime

from ..interfaces import DataFormat, ValidationResult
from ._types import (
    MetadataSchema,
    MetadataType,
    PreservationRule,
    PreservationStrategy,
)
from ._transformers import (
    MetadataTransformer,
    NumpyMetadataTransformer,
    PandasMetadataTransformer,
)


class _MetadataValidationMixin:
    """Mixin providing validation, rules, and internal helper methods."""

    # Attributes expected from MetadataManager (for type checkers)
    _preservation_rules: Dict[str, PreservationRule]
    _metadata_schemas: Dict[DataFormat, MetadataSchema]
    _transformers: List[MetadataTransformer]

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
            required_fields={"data_format", "extraction_timestamp"},
            optional_fields={
                "shape",
                "columns",
                "dtypes",
                "memory_usage",
                "null_counts",
            },
            field_types={
                "shape": tuple,
                "columns": (list, dict),
                "dtypes": dict,
                "memory_usage": (int, float),
                "null_counts": dict,
            },
        )
        self.add_metadata_schema(pandas_schema)

        # Numpy Array schema
        numpy_schema = MetadataSchema(
            schema_name="numpy_array",
            data_format=DataFormat.NUMPY_ARRAY,
            required_fields={"data_format", "extraction_timestamp"},
            optional_fields={"shape", "dtype", "ndim", "size", "memory_bytes"},
            field_types={
                "shape": tuple,
                "dtype": str,
                "ndim": int,
                "size": int,
                "memory_bytes": int,
            },
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
            description="Preserve structural metadata with adaptation",
        )
        self.add_preservation_rule(structural_rule)

        # Medium priority rule for operational metadata
        operational_rule = PreservationRule(
            metadata_key="operational_metadata",
            metadata_type=MetadataType.OPERATIONAL,
            preservation_strategy=PreservationStrategy.STRICT,
            priority=50,
            description="Strictly preserve operational metadata",
        )
        self.add_preservation_rule(operational_rule)

    def _find_transformer(
        self, source_format: DataFormat, target_format: DataFormat
    ) -> Optional[MetadataTransformer]:
        """Find appropriate transformer for format conversion."""
        for transformer in self._transformers:
            if transformer.can_transform(source_format, target_format):
                return transformer
        return None

    def _apply_preservation_rules(
        self,
        metadata: Dict[str, Any],
        source_format: DataFormat,
        target_format: DataFormat,
    ) -> Dict[str, Any]:
        """Apply preservation rules to metadata."""
        modified_metadata = metadata.copy()

        # Sort rules by priority (higher priority first)
        sorted_rules = sorted(
            self._preservation_rules.values(), key=lambda r: r.priority, reverse=True
        )

        for rule in sorted_rules:
            if not rule.is_active:
                continue

            # Check if rule applies to this conversion
            if rule.source_formats and source_format not in rule.source_formats:
                continue
            if rule.target_formats and target_format not in rule.target_formats:
                continue

            # Apply rule
            if rule.metadata_key in modified_metadata:
                if rule.preservation_strategy == PreservationStrategy.STRICT:
                    # Keep as is
                    pass
                elif rule.preservation_strategy == PreservationStrategy.MINIMAL:
                    # Remove if not essential
                    if rule.metadata_type not in {
                        MetadataType.STRUCTURAL,
                        MetadataType.OPERATIONAL,
                    }:
                        del modified_metadata[rule.metadata_key]
                elif (
                    rule.preservation_strategy == PreservationStrategy.CUSTOM
                    and rule.transformer_func
                ):
                    # Apply custom transformation
                    modified_metadata[rule.metadata_key] = rule.transformer_func(
                        modified_metadata[rule.metadata_key]
                    )

        return modified_metadata

    def _validate_metadata(
        self, metadata: Dict[str, Any], format_type: DataFormat
    ) -> ValidationResult:
        """Validate metadata against schema."""
        if format_type not in self._metadata_schemas:
            return ValidationResult(
                is_valid=True,
                score=1.0,
                warnings=[f"No schema defined for {format_type.value}"],
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
                        errors.append(
                            f"Field '{field}' has incorrect type: "
                            f"expected {expected_type}, got {type(value)}"
                        )
                    else:
                        warnings.append(
                            f"Field '{field}' has unexpected type: "
                            f"expected {expected_type}, got {type(value)}"
                        )

        # Check for extra fields
        if not schema.allow_extra_fields:
            allowed_fields = schema.required_fields | schema.optional_fields
            extra_fields = set(metadata.keys()) - allowed_fields
            if extra_fields:
                warnings.extend(
                    [f"Extra field '{field}' not in schema" for field in extra_fields]
                )

        is_valid = len(errors) == 0
        score = 1.0 if is_valid else max(0.0, 1.0 - len(errors) * 0.2)

        return ValidationResult(
            is_valid=is_valid,
            score=score,
            errors=errors,
            warnings=warnings,
            details={
                "schema_name": schema.schema_name,
                "schema_version": schema.version,
                "validation_type": "strict" if schema.strict_validation else "lenient",
            },
        )

    def _resolve_merge_conflict(
        self, key: str, existing_value: Any, new_value: Any
    ) -> Any:
        """Resolve conflict when merging metadata."""
        # For timestamps, prefer the more recent one
        if "timestamp" in key.lower():
            try:
                existing_time = datetime.fromisoformat(
                    existing_value.replace("Z", "+00:00")
                )
                new_time = datetime.fromisoformat(new_value.replace("Z", "+00:00"))
                return new_value if new_time > existing_time else existing_value
            except:
                pass

        # For lists, merge them
        if isinstance(existing_value, list) and isinstance(new_value, list):
            combined = existing_value + new_value
            return list(
                dict.fromkeys(combined)
            )  # Remove duplicates while preserving order

        # For dicts, merge them
        if isinstance(existing_value, dict) and isinstance(new_value, dict):
            merged = existing_value.copy()
            merged.update(new_value)
            return merged

        # For numbers, take the average
        if isinstance(existing_value, (int, float)) and isinstance(
            new_value, (int, float)
        ):
            return (existing_value + new_value) / 2

        # Default: prefer new value
        return new_value

    def _calculate_metadata_changes(
        self, original: Dict[str, Any], transformed: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate changes between original and transformed metadata."""
        changes = {
            "added_keys": set(transformed.keys()) - set(original.keys()),
            "removed_keys": set(original.keys()) - set(transformed.keys()),
            "modified_keys": [],
        }

        for key in set(original.keys()) & set(transformed.keys()):
            if original[key] != transformed[key]:
                changes["modified_keys"].append(
                    {
                        "key": key,
                        "original_value": original[key],
                        "new_value": transformed[key],
                    }
                )

        return changes

    def _generate_lineage_id(self, data: Any) -> str:
        """Generate unique lineage ID for data."""
        import hashlib

        # Create hash based on data identity and timestamp
        components = [
            str(id(data)),
            str(type(data).__name__),
            str(datetime.now().timestamp()),
        ]

        lineage_string = "_".join(components)
        return hashlib.md5(lineage_string.encode()).hexdigest()[:16]
