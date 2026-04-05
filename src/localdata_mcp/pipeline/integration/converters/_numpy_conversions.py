"""
NumPy array conversion methods mixin.

Contains the individual format conversion methods used by NumpyConverter.
Split from _numpy.py for maintainability.
"""

from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy import sparse

from ..interfaces import (
    ConversionError,
    ConversionContext,
)


class NumpyConversionsMixin:
    """Mixin providing individual conversion methods for NumpyConverter."""

    def _numpy_to_dataframe(
        self, array: np.ndarray, context: ConversionContext
    ) -> pd.DataFrame:
        """Convert NumPy array to DataFrame."""
        try:
            if array.ndim == 1:
                # 1D array becomes single column
                df = pd.DataFrame(array, columns=["values"])
                context.intermediate_results["dimension_handling"] = "1d_to_column"
            elif array.ndim == 2:
                # 2D array becomes standard DataFrame
                columns = [f"col_{i}" for i in range(array.shape[1])]
                df = pd.DataFrame(array, columns=columns)
                context.intermediate_results["dimension_handling"] = "2d_standard"
            else:
                # Higher dimensions - flatten to 2D with warning
                reshaped = array.reshape(array.shape[0], -1)
                columns = [f"col_{i}" for i in range(reshaped.shape[1])]
                df = pd.DataFrame(reshaped, columns=columns)
                context.warnings.append(
                    f"Flattened {array.ndim}D array to 2D for DataFrame conversion"
                )
                context.intermediate_results["dimension_handling"] = (
                    f"{array.ndim}d_flattened"
                )

            context.performance_metrics["shape_preserved"] = array.shape
            return df

        except Exception as e:
            raise ConversionError(
                ConversionError.Type.CONVERSION_FAILED,
                f"NumPy to DataFrame conversion failed: {str(e)}",
            )

    def _numpy_to_sparse(
        self, array: np.ndarray, context: ConversionContext
    ) -> sparse.spmatrix:
        """Convert NumPy array to sparse matrix."""
        try:
            # Ensure 2D
            if array.ndim == 1:
                array = array.reshape(-1, 1)
                context.intermediate_results["reshaped_1d"] = True
            elif array.ndim > 2:
                original_shape = array.shape
                array = array.reshape(array.shape[0], -1)
                context.warnings.append(
                    f"Reshaped {original_shape} array to {array.shape} for sparse conversion"
                )

            # Calculate density
            density = np.count_nonzero(array) / array.size
            context.performance_metrics["density"] = density

            # Choose sparse format based on density and shape
            if density < 0.05:  # Very sparse
                sparse_matrix = sparse.coo_matrix(array)
                context.intermediate_results["sparse_format"] = "coo"
            elif array.shape[0] > array.shape[1]:  # Tall matrix
                sparse_matrix = sparse.csc_matrix(array)
                context.intermediate_results["sparse_format"] = "csc"
            else:  # Wide matrix or square
                sparse_matrix = sparse.csr_matrix(array)
                context.intermediate_results["sparse_format"] = "csr"

            return sparse_matrix

        except Exception as e:
            raise ConversionError(
                ConversionError.Type.CONVERSION_FAILED,
                f"NumPy to sparse conversion failed: {str(e)}",
            )

    def _numpy_to_list(
        self, array: np.ndarray, context: ConversionContext
    ) -> List[Any]:
        """Convert NumPy array to Python list."""
        try:
            # Convert to list preserving structure
            list_data = array.tolist()

            context.performance_metrics["original_shape"] = array.shape
            context.performance_metrics["list_nesting"] = array.ndim

            return list_data

        except Exception as e:
            raise ConversionError(
                ConversionError.Type.CONVERSION_FAILED,
                f"NumPy to list conversion failed: {str(e)}",
            )

    def _numpy_to_dict(
        self, array: np.ndarray, context: ConversionContext
    ) -> Dict[str, Any]:
        """Convert NumPy array to dictionary with metadata."""
        try:
            dict_data = {
                "data": array.tolist(),
                "shape": array.shape,
                "dtype": str(array.dtype),
                "ndim": array.ndim,
                "size": array.size,
            }

            # Add memory layout information
            dict_data["flags"] = {
                "c_contiguous": array.flags.c_contiguous,
                "f_contiguous": array.flags.f_contiguous,
                "writeable": array.flags.writeable,
            }

            context.performance_metrics["metadata_included"] = True

            return dict_data

        except Exception as e:
            raise ConversionError(
                ConversionError.Type.CONVERSION_FAILED,
                f"NumPy to dict conversion failed: {str(e)}",
            )

    def _dataframe_to_numpy(
        self, df: pd.DataFrame, context: ConversionContext
    ) -> np.ndarray:
        """Convert DataFrame to NumPy array."""
        try:
            # Select numeric columns only if mixed types
            numeric_df = df.select_dtypes(include=[np.number])

            if len(numeric_df.columns) == 0:
                # No numeric columns - try to convert
                try:
                    array = df.values.astype(float)
                    context.warnings.append(
                        "Forced conversion of non-numeric data to float"
                    )
                except (ValueError, TypeError):
                    # Use object array
                    array = df.values
                    context.warnings.append(
                        "Created object array due to mixed/non-numeric types"
                    )
            elif len(numeric_df.columns) < len(df.columns):
                # Mixed types - warn and use only numeric
                array = numeric_df.values
                dropped_cols = set(df.columns) - set(numeric_df.columns)
                context.warnings.append(
                    f"Dropped non-numeric columns: {list(dropped_cols)}"
                )
            else:
                # All numeric
                array = df.values

            context.performance_metrics["original_df_shape"] = df.shape
            context.performance_metrics["final_array_shape"] = array.shape

            return array

        except Exception as e:
            raise ConversionError(
                ConversionError.Type.CONVERSION_FAILED,
                f"DataFrame to NumPy conversion failed: {str(e)}",
            )

    def _sparse_to_numpy(
        self, sparse_matrix: sparse.spmatrix, context: ConversionContext
    ) -> np.ndarray:
        """Convert sparse matrix to NumPy array."""
        try:
            array = sparse_matrix.toarray()

            context.performance_metrics["original_sparse_format"] = type(
                sparse_matrix
            ).__name__
            context.performance_metrics["density_restored"] = (
                np.count_nonzero(array) / array.size
            )

            return array

        except Exception as e:
            raise ConversionError(
                ConversionError.Type.CONVERSION_FAILED,
                f"Sparse to NumPy conversion failed: {str(e)}",
            )

    def _list_to_numpy(self, data: List[Any], context: ConversionContext) -> np.ndarray:
        """Convert Python list to NumPy array."""
        try:
            array = np.array(data)

            context.performance_metrics["inferred_shape"] = array.shape
            context.performance_metrics["inferred_dtype"] = str(array.dtype)

            return array

        except Exception as e:
            raise ConversionError(
                ConversionError.Type.CONVERSION_FAILED,
                f"List to NumPy conversion failed: {str(e)}",
            )

    def _dict_to_numpy(
        self, data: Dict[str, Any], context: ConversionContext
    ) -> np.ndarray:
        """Convert dictionary to NumPy array."""
        try:
            if "data" in data and "shape" in data:
                # Structured format with metadata
                array = np.array(data["data"])

                # Try to restore original shape
                if "shape" in data:
                    try:
                        array = array.reshape(data["shape"])
                        context.intermediate_results["shape_restored"] = True
                    except ValueError:
                        context.warnings.append("Could not restore original shape")

                # Try to restore dtype
                if "dtype" in data:
                    try:
                        array = array.astype(data["dtype"])
                        context.intermediate_results["dtype_restored"] = True
                    except (ValueError, TypeError):
                        context.warnings.append(
                            f"Could not restore dtype {data['dtype']}"
                        )

            else:
                # Simple dictionary - try to convert values
                if all(isinstance(v, (list, tuple)) for v in data.values()):
                    # Dictionary of arrays
                    array = np.array(list(data.values()))
                else:
                    # Dictionary of scalars
                    array = np.array(list(data.values()))

                context.warnings.append(
                    "Converted simple dictionary, some structure may be lost"
                )

            return array

        except Exception as e:
            raise ConversionError(
                ConversionError.Type.CONVERSION_FAILED,
                f"Dict to NumPy conversion failed: {str(e)}",
            )
