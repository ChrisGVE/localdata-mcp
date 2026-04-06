"""
Core MetadataManager class for centralized metadata operations.

Implements the MetadataPreserver interface providing metadata extraction,
application, merging, transformation, validation, and lineage tracking.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ....logging_manager import get_logger
from ..interfaces import DataFormat, MetadataPreserver, ValidationResult
from ._extraction import _MetadataExtractionMixin
from ._transformers import MetadataTransformer
from ._types import (
    MetadataLineage,
    MetadataSchema,
    MetadataType,
    PreservationRule,
    PreservationStrategy,
)
from ._validation import _MetadataValidationMixin

logger = get_logger(__name__)


class MetadataManager(
    _MetadataExtractionMixin, _MetadataValidationMixin, MetadataPreserver
):
    """
    Comprehensive metadata management system for integration shims.

    Handles metadata preservation, transformation, validation, and lineage
    tracking across all supported data formats and conversion operations.
    """

    def __init__(
        self,
        default_strategy: PreservationStrategy = PreservationStrategy.ADAPTIVE,
        enable_lineage_tracking: bool = True,
        enable_validation: bool = True,
        max_lineage_history: int = 100,
    ):
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

        logger.info(
            "MetadataManager initialized",
            default_strategy=default_strategy.value,
            enable_lineage=enable_lineage_tracking,
            enable_validation=enable_validation,
        )

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
            "extraction_timestamp": datetime.now().isoformat(),
            "data_format": format_type.value,
            "extracted_by": "MetadataManager",
        }

        # Format-specific extraction
        if format_type == DataFormat.PANDAS_DATAFRAME and isinstance(
            data, pd.DataFrame
        ):
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
                    current_format=format_type,
                )
            metadata["lineage_id"] = lineage_id

        logger.debug(
            f"Extracted metadata for {format_type.value}",
            metadata_keys=list(metadata.keys()),
        )

        return metadata

    def apply_metadata(
        self, data: Any, metadata: Dict[str, Any], target_format: DataFormat
    ) -> Any:
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
                logger.warning(
                    "Metadata validation failed", errors=validation_result.errors
                )

        # Apply format-specific metadata
        enhanced_data = self._apply_format_specific_metadata(
            data, metadata, target_format
        )

        logger.debug(
            f"Applied metadata to {target_format.value}",
            applied_keys=list(metadata.keys()),
        )

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
                        conflict = {
                            "key": key,
                            "existing_value": merged[key],
                            "new_value": value,
                            "resolution": self._resolve_merge_conflict(
                                key, merged[key], value
                            ),
                        }
                        merge_conflicts.append(conflict)
                        merged[key] = conflict["resolution"]
                else:
                    merged[key] = value

        merged.update(
            {
                "merge_timestamp": datetime.now().isoformat(),
                "source_count": len(metadata_list),
                "merge_conflicts": merge_conflicts,
                "merged_by": "MetadataManager",
            }
        )

        if merge_conflicts:
            logger.warning(
                f"Resolved {len(merge_conflicts)} merge conflicts during metadata merge"
            )

        return merged

    def transform_metadata(
        self,
        metadata: Dict[str, Any],
        source_format: DataFormat,
        target_format: DataFormat,
        adapter_id: str,
    ) -> Dict[str, Any]:
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
            transformed = transformer.transform_metadata(
                transformed, source_format, target_format
            )

        # Apply preservation rules
        transformed = self._apply_preservation_rules(
            transformed, source_format, target_format
        )

        # Add transformation metadata
        transformation_info = {
            "transformation_timestamp": datetime.now().isoformat(),
            "source_format": source_format.value,
            "target_format": target_format.value,
            "adapter_id": adapter_id,
            "transformer_used": transformer.__class__.__name__ if transformer else None,
        }

        if "transformation_history" not in transformed:
            transformed["transformation_history"] = []
        transformed["transformation_history"].append(transformation_info)
        transformed["transformation_timestamp"] = transformation_info[
            "transformation_timestamp"
        ]
        transformed["target_format"] = transformation_info["target_format"]

        # Update lineage if tracking enabled
        if self.enable_lineage_tracking and "lineage_id" in metadata:
            lineage_id = metadata["lineage_id"]
            if lineage_id in self._lineage_tracking:
                self._lineage_tracking[lineage_id].add_transformation(
                    operation="metadata_transform",
                    source_format=source_format,
                    target_format=target_format,
                    adapter_id=adapter_id,
                    metadata_changes=self._calculate_metadata_changes(
                        metadata, transformed
                    ),
                )

        logger.debug(
            "Metadata transformed",
            source_format=source_format.value,
            target_format=target_format.value,
            transformer=transformer.__class__.__name__ if transformer else "none",
        )

        return transformed

    def add_preservation_rule(self, rule: PreservationRule) -> None:
        """Add a metadata preservation rule."""
        self._preservation_rules[rule.metadata_key] = rule
        logger.info(
            f"Added preservation rule for '{rule.metadata_key}'",
            strategy=rule.preservation_strategy.value,
            priority=rule.priority,
        )

    def add_metadata_schema(self, schema: MetadataSchema) -> None:
        """Add a metadata schema for validation."""
        self._metadata_schemas[schema.data_format] = schema
        logger.info(
            f"Added metadata schema for {schema.data_format.value}",
            required_fields=len(schema.required_fields),
            optional_fields=len(schema.optional_fields),
        )

    def add_transformer(self, transformer: MetadataTransformer) -> None:
        """Add a metadata transformer."""
        self._transformers.append(transformer)
        supported_types = transformer.get_supported_metadata_types()
        logger.info(
            f"Added metadata transformer {transformer.__class__.__name__}",
            supported_types=[t.value for t in supported_types],
        )

    def validate_metadata(
        self, metadata: Dict[str, Any], format_type: DataFormat
    ) -> ValidationResult:
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
