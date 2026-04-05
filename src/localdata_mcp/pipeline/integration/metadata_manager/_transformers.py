"""
Metadata transformers for format-specific metadata conversion.

Provides the abstract MetadataTransformer base class and concrete implementations
for pandas and numpy format conversions.
"""

from typing import Any, Dict, Set
from abc import ABC, abstractmethod
from datetime import datetime

from ..interfaces import DataFormat
from ._types import MetadataType


class MetadataTransformer(ABC):
    """Abstract base class for format-specific metadata transformers."""

    @abstractmethod
    def can_transform(
        self, source_format: DataFormat, target_format: DataFormat
    ) -> bool:
        """Check if this transformer can handle the format conversion."""
        pass

    @abstractmethod
    def transform_metadata(
        self,
        metadata: Dict[str, Any],
        source_format: DataFormat,
        target_format: DataFormat,
    ) -> Dict[str, Any]:
        """Transform metadata between formats."""
        pass

    @abstractmethod
    def get_supported_metadata_types(self) -> Set[MetadataType]:
        """Get metadata types supported by this transformer."""
        pass


class PandasMetadataTransformer(MetadataTransformer):
    """Metadata transformer for pandas DataFrame formats."""

    def can_transform(
        self, source_format: DataFormat, target_format: DataFormat
    ) -> bool:
        """Check if transformation is supported."""
        pandas_formats = {
            DataFormat.PANDAS_DATAFRAME,
            DataFormat.TIME_SERIES,
            DataFormat.CATEGORICAL,
        }
        return source_format in pandas_formats or target_format in pandas_formats

    def transform_metadata(
        self,
        metadata: Dict[str, Any],
        source_format: DataFormat,
        target_format: DataFormat,
    ) -> Dict[str, Any]:
        """Transform metadata for pandas-related conversions."""
        transformed = metadata.copy()

        # Handle DataFrame-specific metadata
        if source_format == DataFormat.PANDAS_DATAFRAME:
            if target_format == DataFormat.NUMPY_ARRAY:
                # Convert DataFrame metadata to array metadata
                if "columns" in transformed:
                    transformed["original_columns"] = transformed["columns"]
                    del transformed["columns"]
                if "dtypes" in transformed:
                    transformed["original_dtypes"] = transformed["dtypes"]
                    del transformed["dtypes"]
                if "shape" in transformed:
                    # Shape remains relevant for arrays
                    pass

        elif target_format == DataFormat.PANDAS_DATAFRAME:
            # Restore DataFrame metadata from other formats
            if "original_columns" in transformed:
                transformed["columns"] = transformed["original_columns"]
                del transformed["original_columns"]
            if "original_dtypes" in transformed:
                transformed["dtypes"] = transformed["original_dtypes"]
                del transformed["original_dtypes"]

        # Handle time series specific metadata
        if target_format == DataFormat.TIME_SERIES:
            transformed.update(
                {
                    "temporal_metadata": {
                        "is_time_series": True,
                        "conversion_timestamp": datetime.now().isoformat(),
                    }
                }
            )

        return transformed

    def get_supported_metadata_types(self) -> Set[MetadataType]:
        """Get supported metadata types."""
        return {
            MetadataType.STRUCTURAL,
            MetadataType.SEMANTIC,
            MetadataType.OPERATIONAL,
            MetadataType.QUALITY,
        }


class NumpyMetadataTransformer(MetadataTransformer):
    """Metadata transformer for numpy array formats."""

    def can_transform(
        self, source_format: DataFormat, target_format: DataFormat
    ) -> bool:
        """Check if transformation is supported."""
        numpy_formats = {DataFormat.NUMPY_ARRAY}
        return source_format in numpy_formats or target_format in numpy_formats

    def transform_metadata(
        self,
        metadata: Dict[str, Any],
        source_format: DataFormat,
        target_format: DataFormat,
    ) -> Dict[str, Any]:
        """Transform metadata for numpy-related conversions."""
        transformed = metadata.copy()

        if source_format == DataFormat.NUMPY_ARRAY:
            if target_format == DataFormat.PANDAS_DATAFRAME:
                # Preserve array-specific metadata
                if "shape" in transformed:
                    transformed["original_array_shape"] = transformed["shape"]
                if "dtype" in transformed:
                    transformed["original_array_dtype"] = transformed["dtype"]

        elif target_format == DataFormat.NUMPY_ARRAY:
            # Add array-specific metadata
            transformed.update(
                {
                    "array_metadata": {
                        "conversion_source": source_format.value,
                        "conversion_timestamp": datetime.now().isoformat(),
                    }
                }
            )

        return transformed

    def get_supported_metadata_types(self) -> Set[MetadataType]:
        """Get supported metadata types."""
        return {MetadataType.STRUCTURAL, MetadataType.OPERATIONAL, MetadataType.QUALITY}
