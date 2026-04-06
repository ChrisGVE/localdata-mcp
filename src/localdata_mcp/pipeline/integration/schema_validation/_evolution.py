"""
SchemaEvolutionManager class.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from ....logging_manager import get_logger
from ._inference import SchemaInferenceEngine
from ._schema import DataSchema
from ._types import (
    SchemaConstraint,
    ValidationRuleType,
)

logger = get_logger(__name__)


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

    def get_schema_version(
        self, schema_name: str, version: str
    ) -> Optional[DataSchema]:
        """Get a specific version of a schema."""
        if schema_name in self.schema_versions:
            for schema in self.schema_versions[schema_name]:
                if schema.schema_version == version:
                    return schema
        return None

    def detect_schema_drift(
        self,
        current_data: Any,
        reference_schema: DataSchema,
        drift_threshold: float = 0.1,
    ) -> Dict[str, Any]:
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
        compatibility_score = self.calculate_compatibility_score(
            current_schema, reference_schema
        )

        # Detect specific changes
        field_changes = self._detect_field_changes(current_schema, reference_schema)
        constraint_changes = self._detect_constraint_changes(
            current_schema, reference_schema
        )

        # Determine drift level
        drift_detected = compatibility_score < (1.0 - drift_threshold)
        drift_level = self._categorize_drift_level(compatibility_score)

        drift_analysis = {
            "drift_detected": drift_detected,
            "drift_level": drift_level,
            "compatibility_score": compatibility_score,
            "current_schema": current_schema,
            "reference_schema": reference_schema,
            "field_changes": field_changes,
            "constraint_changes": constraint_changes,
            "analysis_timestamp": datetime.now(),
            "recommendations": self._generate_drift_recommendations(
                field_changes, constraint_changes
            ),
        }

        return drift_analysis

    def calculate_compatibility_score(
        self, schema1: DataSchema, schema2: DataSchema
    ) -> float:
        """Calculate compatibility score between two schemas (0.0 to 1.0)."""
        cache_key = f"{schema1.get_schema_hash()}_{schema2.get_schema_hash()}"
        if cache_key in self.compatibility_cache:
            return self.compatibility_cache[cache_key]

        score = 0.0
        total_weight = 0.0

        # Field compatibility (40% weight)
        field_score, field_weight = self._calculate_field_compatibility(
            schema1, schema2
        )
        score += field_score * field_weight
        total_weight += field_weight

        # Type compatibility (30% weight)
        type_score, type_weight = self._calculate_type_compatibility(schema1, schema2)
        score += type_score * type_weight
        total_weight += type_weight

        # Constraint compatibility (20% weight)
        constraint_score, constraint_weight = self._calculate_constraint_compatibility(
            schema1, schema2
        )
        score += constraint_score * constraint_weight
        total_weight += constraint_weight

        # Structure compatibility (10% weight)
        structure_score, structure_weight = self._calculate_structure_compatibility(
            schema1, schema2
        )
        score += structure_score * structure_weight
        total_weight += structure_weight

        final_score = score / total_weight if total_weight > 0 else 0.0

        # Cache the result
        self.compatibility_cache[cache_key] = final_score

        return final_score

    def _calculate_field_compatibility(
        self, schema1: DataSchema, schema2: DataSchema
    ) -> Tuple[float, float]:
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
        required_fields1 = {
            f
            for f, schema in schema1.fields.items()
            if not schema.get("nullable", True)
        }
        required_fields2 = {
            f
            for f, schema in schema2.fields.items()
            if not schema.get("nullable", True)
        }

        missing_required = (required_fields1 - fields2) | (required_fields2 - fields1)
        required_penalty = len(missing_required) * 0.2

        field_score = max(0.0, overlap_score - required_penalty)

        return field_score, 0.4

    def _calculate_type_compatibility(
        self, schema1: DataSchema, schema2: DataSchema
    ) -> Tuple[float, float]:
        """Calculate type-level compatibility score."""
        common_fields = set(schema1.fields.keys()) & set(schema2.fields.keys())

        if not common_fields:
            return 1.0, 0.3  # No common fields to compare

        compatible_types = 0
        total_comparisons = len(common_fields)

        for field in common_fields:
            type1 = schema1.fields[field].get("type", "object")
            type2 = schema2.fields[field].get("type", "object")

            if self._are_types_compatible(type1, type2):
                compatible_types += 1

        type_score = (
            compatible_types / total_comparisons if total_comparisons > 0 else 1.0
        )

        return type_score, 0.3

    def _calculate_constraint_compatibility(
        self, schema1: DataSchema, schema2: DataSchema
    ) -> Tuple[float, float]:
        """Calculate constraint-level compatibility score."""
        constraints1 = {
            (c.field_name, c.constraint_type): c for c in schema1.constraints
        }
        constraints2 = {
            (c.field_name, c.constraint_type): c for c in schema2.constraints
        }

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

        constraint_score = (
            compatible_constraints / len(all_constraint_keys)
            if all_constraint_keys
            else 1.0
        )

        return constraint_score, 0.2

    def _calculate_structure_compatibility(
        self, schema1: DataSchema, schema2: DataSchema
    ) -> Tuple[float, float]:
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
        numeric_types = {"int", "float", "numeric"}
        string_types = {"str", "string", "text"}

        if type1 in numeric_types and type2 in numeric_types:
            return True
        if type1 in string_types and type2 in string_types:
            return True

        # Special compatibility rules
        compatibility_map = {
            ("int", "float"): True,
            ("float", "int"): True,
            ("datetime", "str"): True,  # Datetime can be serialized as string
            ("bool", "int"): True,  # Boolean can be represented as integer
        }

        return compatibility_map.get((type1, type2), False)

    def _are_constraints_compatible(
        self, constraint1: SchemaConstraint, constraint2: SchemaConstraint
    ) -> bool:
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

    def _detect_field_changes(
        self, current_schema: DataSchema, reference_schema: DataSchema
    ) -> Dict[str, Any]:
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
                changes = self._detect_field_property_changes(
                    current_field, reference_field
                )
                if changes:
                    modified_fields.append({"field_name": field, "changes": changes})

        return {
            "added_fields": list(added_fields),
            "removed_fields": list(removed_fields),
            "modified_fields": modified_fields,
            "total_changes": len(added_fields)
            + len(removed_fields)
            + len(modified_fields),
        }

    def _detect_field_property_changes(
        self, current_field: Dict[str, Any], reference_field: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Detect changes in individual field properties."""
        changes = []

        all_props = set(current_field.keys()) | set(reference_field.keys())

        for prop in all_props:
            current_val = current_field.get(prop)
            reference_val = reference_field.get(prop)

            if current_val != reference_val:
                changes.append(
                    {
                        "property": prop,
                        "old_value": reference_val,
                        "new_value": current_val,
                    }
                )

        return changes

    def _detect_constraint_changes(
        self, current_schema: DataSchema, reference_schema: DataSchema
    ) -> Dict[str, Any]:
        """Detect changes in schema constraints."""
        current_constraints = {c.constraint_id: c for c in current_schema.constraints}
        reference_constraints = {
            c.constraint_id: c for c in reference_schema.constraints
        }

        added_constraints = set(current_constraints.keys()) - set(
            reference_constraints.keys()
        )
        removed_constraints = set(reference_constraints.keys()) - set(
            current_constraints.keys()
        )

        modified_constraints = []
        common_constraints = set(current_constraints.keys()) & set(
            reference_constraints.keys()
        )

        for constraint_id in common_constraints:
            current_constraint = current_constraints[constraint_id]
            reference_constraint = reference_constraints[constraint_id]

            if (
                current_constraint.constraint_value
                != reference_constraint.constraint_value
            ):
                modified_constraints.append(
                    {
                        "constraint_id": constraint_id,
                        "old_value": reference_constraint.constraint_value,
                        "new_value": current_constraint.constraint_value,
                    }
                )

        return {
            "added_constraints": list(added_constraints),
            "removed_constraints": list(removed_constraints),
            "modified_constraints": modified_constraints,
            "total_changes": len(added_constraints)
            + len(removed_constraints)
            + len(modified_constraints),
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

    def _generate_drift_recommendations(
        self, field_changes: Dict[str, Any], constraint_changes: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations for handling schema drift."""
        recommendations = []

        if field_changes["added_fields"]:
            recommendations.append(
                f"Consider adding default values for new fields: {', '.join(field_changes['added_fields'])}"
            )

        if field_changes["removed_fields"]:
            recommendations.append(
                f"Archive or migrate data for removed fields: {', '.join(field_changes['removed_fields'])}"
            )

        if field_changes["modified_fields"]:
            recommendations.append(
                f"Review type changes for modified fields - may require data transformation"
            )

        if constraint_changes["total_changes"] > 0:
            recommendations.append(
                "Review constraint changes to ensure data quality requirements are maintained"
            )

        if not recommendations:
            recommendations.append(
                "Schema compatibility is high - minimal migration effort required"
            )

        return recommendations
