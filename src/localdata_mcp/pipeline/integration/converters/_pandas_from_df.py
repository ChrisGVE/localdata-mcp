"""
Pandas converter methods: DataFrame to other formats.

Contains methods that convert FROM pandas DataFrame to numpy, sparse,
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


class PandasFromDataFrameMixin:
    """Mixin providing DataFrame-to-other-format conversion methods."""

    def _convert_from_dataframe(
        self,
        df,
        target_format: DataFormat,
        context,
    ) -> Any:
        """Convert DataFrame to other formats."""
        if not isinstance(df, pd.DataFrame):
            raise ConversionError(
                ConversionError.Type.TYPE_MISMATCH,
                f"Expected pandas DataFrame, got {type(df)}",
            )

        # Store metadata for preservation
        original_metadata = self._metadata_manager.extract_metadata(
            df, DataFormat.PANDAS_DATAFRAME
        )
        context.intermediate_results["original_metadata"] = original_metadata

        if target_format == DataFormat.NUMPY_ARRAY:
            return self._dataframe_to_numpy(df, context)
        elif target_format == DataFormat.SCIPY_SPARSE:
            return self._dataframe_to_sparse(df, context)
        elif target_format == DataFormat.PYTHON_DICT:
            return self._dataframe_to_dict(df, context)
        elif target_format == DataFormat.PYTHON_LIST:
            return self._dataframe_to_list(df, context)
        elif target_format == DataFormat.TIME_SERIES:
            return self._dataframe_to_timeseries(df, context)
        elif target_format == DataFormat.CATEGORICAL:
            return self._dataframe_to_categorical(df, context)
        else:
            raise ConversionError(
                ConversionError.Type.CONVERSION_FAILED,
                f"Unsupported target format: {target_format.value}",
            )

    def _dataframe_to_numpy(
        self, df: pd.DataFrame, context: ConversionContext
    ) -> np.ndarray:
        """Convert DataFrame to NumPy array."""
        try:
            # Handle empty DataFrame
            if df.empty:
                context.warnings.append(
                    "Empty DataFrame converted to empty NumPy array"
                )
                return np.empty((0, max(len(df.columns), 0)))

            # Handle mixed types by attempting numeric conversion
            if self.conversion_options.handle_mixed_types:
                numeric_df = df.select_dtypes(include=[np.number])

                if len(numeric_df.columns) == 0:
                    # No numeric columns - try to convert everything
                    context.warnings.append(
                        "No numeric columns found, attempting conversion of all data"
                    )
                    converted_df = pd.get_dummies(df, drop_first=True)
                elif len(numeric_df.columns) < len(df.columns):
                    # Mixed types - warn and use only numeric
                    non_numeric_cols = set(df.columns) - set(numeric_df.columns)
                    context.warnings.append(
                        f"Dropping non-numeric columns: {list(non_numeric_cols)}"
                    )
                    converted_df = numeric_df
                else:
                    # All numeric
                    converted_df = df
            else:
                converted_df = df

            # Convert to numpy array
            array_data = converted_df.values

            # Store conversion metadata
            context.intermediate_results["columns_preserved"] = list(
                converted_df.columns
            )
            context.intermediate_results["shape_change"] = (df.shape, array_data.shape)
            context.performance_metrics["rows_converted"] = len(df)

            self._conversion_stats["total_rows_processed"] += len(df)

            return array_data

        except Exception as e:
            raise ConversionError(
                ConversionError.Type.CONVERSION_FAILED,
                f"DataFrame to NumPy conversion failed: {str(e)}",
            )

    def _dataframe_to_sparse(
        self, df: pd.DataFrame, context: ConversionContext
    ) -> sparse.spmatrix:
        """Convert DataFrame to scipy sparse matrix."""
        try:
            # First convert to numpy
            array_data = self._dataframe_to_numpy(df, context)

            # Calculate density
            density = np.count_nonzero(array_data) / array_data.size
            context.performance_metrics["density"] = density

            if density > self.conversion_options.sparse_density_threshold:
                context.warnings.append(
                    f"Data density {density:.3f} above threshold "
                    f"{self.conversion_options.sparse_density_threshold}, but creating sparse matrix anyway"
                )

            # Create sparse matrix (CSR format for efficiency)
            sparse_matrix = sparse.csr_matrix(array_data)

            context.intermediate_results["sparsity_ratio"] = 1 - density
            context.intermediate_results["sparse_format"] = "csr"

            return sparse_matrix

        except Exception as e:
            raise ConversionError(
                ConversionError.Type.CONVERSION_FAILED,
                f"DataFrame to sparse matrix conversion failed: {str(e)}",
            )

    def _dataframe_to_dict(
        self, df: pd.DataFrame, context: ConversionContext
    ) -> Dict[str, Any]:
        """Convert DataFrame to dictionary."""
        try:
            # Use pandas to_dict with records orientation for row-based structure
            dict_data = df.to_dict("records")

            # Store additional metadata
            metadata_dict = {
                "data": dict_data,
                "columns": list(df.columns),
                "index": list(df.index)
                if self.conversion_options.preserve_index
                else None,
                "dtypes": df.dtypes.to_dict(),
                "shape": df.shape,
            }

            context.intermediate_results["preservation_mode"] = "full_metadata"
            context.performance_metrics["dict_entries"] = len(dict_data)

            return metadata_dict

        except Exception as e:
            raise ConversionError(
                ConversionError.Type.CONVERSION_FAILED,
                f"DataFrame to dict conversion failed: {str(e)}",
            )

    def _dataframe_to_list(
        self, df: pd.DataFrame, context: ConversionContext
    ) -> List[Any]:
        """Convert DataFrame to list."""
        try:
            # Convert to list of lists (rows as lists)
            list_data = df.values.tolist()

            # Optionally include column names as first row
            if self.conversion_options.preserve_columns:
                list_data.insert(0, list(df.columns))
                context.intermediate_results["header_included"] = True

            context.performance_metrics["list_length"] = len(list_data)

            return list_data

        except Exception as e:
            raise ConversionError(
                ConversionError.Type.CONVERSION_FAILED,
                f"DataFrame to list conversion failed: {str(e)}",
            )

    def _dataframe_to_timeseries(
        self, df: pd.DataFrame, context: ConversionContext
    ) -> pd.DataFrame:
        """Convert DataFrame to time series format."""
        try:
            # Check if already has datetime index
            if isinstance(df.index, pd.DatetimeIndex):
                ts_df = df.copy()
                context.intermediate_results["already_timeseries"] = True
            else:
                # Try to find datetime column to use as index
                datetime_cols = df.select_dtypes(include=["datetime64"]).columns

                if len(datetime_cols) > 0:
                    # Use first datetime column as index
                    ts_df = df.set_index(datetime_cols[0])
                    context.intermediate_results["datetime_column_used"] = (
                        datetime_cols[0]
                    )
                else:
                    # No datetime columns - create a simple integer index and warn
                    ts_df = df.copy()
                    ts_df.index = pd.date_range("2024-01-01", periods=len(df), freq="D")
                    context.warnings.append(
                        "No datetime columns found, created synthetic date range"
                    )

            # Sort by index for time series convention
            ts_df = ts_df.sort_index()

            context.intermediate_results["time_range"] = (
                ts_df.index.min(),
                ts_df.index.max(),
            )

            return ts_df

        except Exception as e:
            raise ConversionError(
                ConversionError.Type.CONVERSION_FAILED,
                f"DataFrame to time series conversion failed: {str(e)}",
            )

    def _dataframe_to_categorical(
        self, df: pd.DataFrame, context: ConversionContext
    ) -> pd.DataFrame:
        """Convert DataFrame to categorical format."""
        try:
            categorical_df = df.copy()
            converted_columns = []

            for col in df.columns:
                # Check if column should be categorical
                unique_ratio = df[col].nunique() / len(df) if len(df) > 0 else 0

                if (
                    unique_ratio <= self.conversion_options.categorical_threshold
                    and df[col].nunique() < 100
                ):  # Additional constraint for large cardinality
                    categorical_df[col] = df[col].astype("category")
                    converted_columns.append(col)

            if not converted_columns:
                context.warnings.append(
                    "No columns met criteria for categorical conversion"
                )

            context.intermediate_results["categorical_columns"] = converted_columns
            context.performance_metrics["categories_created"] = len(converted_columns)

            return categorical_df

        except Exception as e:
            raise ConversionError(
                ConversionError.Type.CONVERSION_FAILED,
                f"DataFrame to categorical conversion failed: {str(e)}",
            )
