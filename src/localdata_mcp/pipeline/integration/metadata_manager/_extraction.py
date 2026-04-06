"""
Metadata extraction helpers for the MetadataManager.

Provides the _MetadataExtractionMixin with format-specific metadata extraction
and application methods for pandas DataFrames, numpy arrays, time series, and
generic data types.
"""

from typing import Any, Dict

import numpy as np
import pandas as pd

from ..interfaces import DataFormat


class _MetadataExtractionMixin:
    """Mixin providing metadata extraction and application helpers."""

    def _extract_dataframe_metadata(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Extract metadata from pandas DataFrame."""
        metadata = {
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes.to_dict(),
            "memory_usage": df.memory_usage(deep=True).sum(),
            "null_counts": df.isnull().sum().to_dict(),
            "index_type": type(df.index).__name__,
        }

        # Add additional DataFrame-specific metadata
        if hasattr(df.index, "freq") and df.index.freq:
            metadata["index_frequency"] = str(df.index.freq)

        return metadata

    def _extract_numpy_metadata(self, arr: np.ndarray) -> Dict[str, Any]:
        """Extract metadata from numpy array."""
        return {
            "shape": arr.shape,
            "dtype": str(arr.dtype),
            "ndim": arr.ndim,
            "size": arr.size,
            "memory_bytes": arr.nbytes,
            "is_contiguous": arr.flags.c_contiguous,
            "is_writeable": arr.flags.writeable,
        }

    def _extract_timeseries_metadata(self, data: Any) -> Dict[str, Any]:
        """Extract metadata from time series data."""
        metadata = {"is_time_series": True}

        if isinstance(data, pd.DataFrame):
            metadata.update(self._extract_dataframe_metadata(data))

            # Time series specific metadata
            if isinstance(data.index, pd.DatetimeIndex):
                metadata.update(
                    {
                        "temporal_range": (data.index.min(), data.index.max()),
                        "frequency": pd.infer_freq(data.index),
                        "has_regular_frequency": data.index.freq is not None,
                    }
                )

        return metadata

    def _extract_generic_metadata(self, data: Any) -> Dict[str, Any]:
        """Extract generic metadata from any data type."""
        metadata = {"data_type": type(data).__name__, "python_type": str(type(data))}

        # Add size information if available
        if hasattr(data, "__len__"):
            metadata["length"] = len(data)

        if hasattr(data, "size"):
            metadata["size"] = getattr(data, "size")

        return metadata

    def _apply_format_specific_metadata(
        self, data: Any, metadata: Dict[str, Any], format_type: DataFormat
    ) -> Any:
        """Apply format-specific metadata to data."""
        # For most formats, metadata is stored separately
        # Some formats might support embedded metadata

        if format_type == DataFormat.PANDAS_DATAFRAME and isinstance(
            data, pd.DataFrame
        ):
            # For DataFrames, we can store some metadata as attributes
            if hasattr(data, "attrs"):
                # Store compatible metadata in DataFrame.attrs (pandas >= 1.3)
                compatible_metadata = {
                    k: v
                    for k, v in metadata.items()
                    if isinstance(v, (str, int, float, bool, list, dict))
                }
                data.attrs.update(compatible_metadata)

        return data
