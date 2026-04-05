"""
Pandas converter methods: other formats to DataFrame.

Contains methods that convert TO pandas DataFrame from numpy, sparse,
dict, list, time series, and categorical formats.
"""

from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy import sparse

from ..interfaces import (
    DataFormat,
    ConversionError,
    ConversionContext,
)


class PandasToDataFrameMixin:
    """Mixin providing other-format-to-DataFrame conversion methods."""

    def _convert_to_dataframe(
        self, data: Any, source_format: DataFormat, context
    ) -> Any:
        """Convert other formats to DataFrame."""
        if source_format == DataFormat.NUMPY_ARRAY:
            return self._numpy_to_dataframe(data, context)
        elif source_format == DataFormat.SCIPY_SPARSE:
            return self._sparse_to_dataframe(data, context)
        elif source_format == DataFormat.PYTHON_DICT:
            return self._dict_to_dataframe(data, context)
        elif source_format == DataFormat.PYTHON_LIST:
            return self._list_to_dataframe(data, context)
        elif source_format == DataFormat.TIME_SERIES:
            return self._timeseries_to_dataframe(data, context)
        elif source_format == DataFormat.CATEGORICAL:
            return self._categorical_to_dataframe(data, context)
        else:
            raise ConversionError(
                ConversionError.Type.CONVERSION_FAILED,
                f"Unsupported source format: {source_format.value}",
            )

    def _numpy_to_dataframe(
        self, array: np.ndarray, context: ConversionContext
    ) -> pd.DataFrame:
        """Convert NumPy array to DataFrame."""
        try:
            # Generate column names if not preserved
            if "original_metadata" in context.intermediate_results:
                metadata = context.intermediate_results["original_metadata"]
                columns = metadata.get("columns", None)
            else:
                columns = None

            if columns is None or len(columns) != array.shape[1]:
                columns = [f"col_{i}" for i in range(array.shape[1])]
                context.warnings.append(
                    "Generated column names as original names not available"
                )

            # Create DataFrame
            df = pd.DataFrame(array, columns=columns)

            context.intermediate_results["columns_generated"] = columns
            context.performance_metrics["dataframe_shape"] = df.shape

            return df

        except Exception as e:
            raise ConversionError(
                ConversionError.Type.CONVERSION_FAILED,
                f"NumPy to DataFrame conversion failed: {str(e)}",
            )

    def _sparse_to_dataframe(
        self, sparse_matrix: sparse.spmatrix, context: ConversionContext
    ) -> pd.DataFrame:
        """Convert sparse matrix to DataFrame."""
        try:
            # Convert to dense first
            dense_array = sparse_matrix.toarray()

            # Use numpy conversion path
            df = self._numpy_to_dataframe(dense_array, context)

            context.intermediate_results["original_sparse_format"] = type(
                sparse_matrix
            ).__name__
            context.performance_metrics["sparsity_lost"] = True

            return df

        except Exception as e:
            raise ConversionError(
                ConversionError.Type.CONVERSION_FAILED,
                f"Sparse to DataFrame conversion failed: {str(e)}",
            )

    def _dict_to_dataframe(
        self, data: Dict[str, Any], context: ConversionContext
    ) -> pd.DataFrame:
        """Convert dictionary to DataFrame."""
        try:
            if "data" in data and "columns" in data:
                # Structured metadata format
                df = pd.DataFrame(data["data"])

                # Restore index if preserved
                if "index" in data and data["index"] is not None:
                    df.index = data["index"]

                context.intermediate_results["metadata_restored"] = True
            else:
                # Simple dictionary conversion
                df = pd.DataFrame.from_dict(data, orient="index").T
                context.warnings.append(
                    "No metadata structure found, used simple conversion"
                )

            context.performance_metrics["dict_keys"] = len(data)

            return df

        except Exception as e:
            raise ConversionError(
                ConversionError.Type.CONVERSION_FAILED,
                f"Dict to DataFrame conversion failed: {str(e)}",
            )

    def _list_to_dataframe(
        self, data: List[Any], context: ConversionContext
    ) -> pd.DataFrame:
        """Convert list to DataFrame."""
        try:
            if not data:
                # Empty list
                df = pd.DataFrame()
                context.warnings.append("Empty list provided, created empty DataFrame")
                return df

            # Check if first row might be headers
            if (
                len(data) > 1
                and isinstance(data[0], list)
                and all(isinstance(x, str) for x in data[0])
            ):
                # Assume first row is headers
                columns = data[0]
                data_rows = data[1:]
                df = pd.DataFrame(data_rows, columns=columns)
                context.intermediate_results["header_detected"] = True
            else:
                # No headers detected
                df = pd.DataFrame(data)
                context.warnings.append(
                    "No headers detected, used default column names"
                )

            context.performance_metrics["rows_processed"] = len(data)

            return df

        except Exception as e:
            raise ConversionError(
                ConversionError.Type.CONVERSION_FAILED,
                f"List to DataFrame conversion failed: {str(e)}",
            )

    def _timeseries_to_dataframe(
        self, ts_data: pd.DataFrame, context: ConversionContext
    ) -> pd.DataFrame:
        """Convert time series to regular DataFrame."""
        try:
            # Time series is already a DataFrame, just ensure proper formatting
            df = ts_data.copy()

            # Reset index if it's a DatetimeIndex and we want to preserve it as a column
            if (
                isinstance(df.index, pd.DatetimeIndex)
                and self.conversion_options.preserve_index
            ):
                df.reset_index(inplace=True)
                context.intermediate_results["datetime_index_preserved"] = True

            return df

        except Exception as e:
            raise ConversionError(
                ConversionError.Type.CONVERSION_FAILED,
                f"Time series to DataFrame conversion failed: {str(e)}",
            )

    def _categorical_to_dataframe(
        self, cat_data: pd.DataFrame, context: ConversionContext
    ) -> pd.DataFrame:
        """Convert categorical DataFrame to regular DataFrame."""
        try:
            df = cat_data.copy()

            # Convert categorical columns back to original types where sensible
            for col in df.columns:
                if pd.api.types.is_categorical_dtype(df[col]):
                    # Try to convert back to numeric if possible
                    try:
                        df[col] = pd.to_numeric(df[col].astype(str))
                        context.intermediate_results.setdefault(
                            "reconverted_numeric", []
                        ).append(col)
                    except (ValueError, TypeError):
                        # Keep as object type
                        df[col] = df[col].astype(str)
                        context.intermediate_results.setdefault(
                            "converted_to_string", []
                        ).append(col)

            return df

        except Exception as e:
            raise ConversionError(
                ConversionError.Type.CONVERSION_FAILED,
                f"Categorical to DataFrame conversion failed: {str(e)}",
            )
