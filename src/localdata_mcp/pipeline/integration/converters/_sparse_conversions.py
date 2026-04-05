"""
Sparse matrix conversion methods mixin.

Contains the individual format conversion methods used by SparseMatrixConverter.
Split from _sparse.py for maintainability.
"""

from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy import sparse

from ..interfaces import (
    ConversionError,
    ConversionContext,
)


class SparseConversionsMixin:
    """Mixin providing individual conversion methods for SparseMatrixConverter."""

    def _sparse_to_dataframe(
        self, sparse_matrix: sparse.spmatrix, context: ConversionContext
    ) -> pd.DataFrame:
        """Convert sparse matrix to DataFrame."""
        try:
            # Convert to dense array first
            dense_array = sparse_matrix.toarray()

            # Create DataFrame with generated column names
            columns = [f"col_{i}" for i in range(dense_array.shape[1])]
            df = pd.DataFrame(dense_array, columns=columns)

            context.performance_metrics["sparsity_lost"] = True
            context.performance_metrics["memory_increase"] = "significant"
            context.warnings.append(
                "Converted sparse to dense - significant memory increase possible"
            )

            return df

        except Exception as e:
            raise ConversionError(
                ConversionError.Type.CONVERSION_FAILED,
                f"Sparse to DataFrame conversion failed: {str(e)}",
            )

    def _sparse_to_numpy(
        self, sparse_matrix: sparse.spmatrix, context: ConversionContext
    ) -> np.ndarray:
        """Convert sparse matrix to NumPy array."""
        try:
            dense_array = sparse_matrix.toarray()

            context.performance_metrics["densification"] = True
            context.performance_metrics["original_nnz"] = sparse_matrix.nnz
            context.performance_metrics["final_size"] = dense_array.size

            return dense_array

        except Exception as e:
            raise ConversionError(
                ConversionError.Type.CONVERSION_FAILED,
                f"Sparse to NumPy conversion failed: {str(e)}",
            )

    def _sparse_to_dict(
        self, sparse_matrix: sparse.spmatrix, context: ConversionContext
    ) -> Dict[str, Any]:
        """Convert sparse matrix to dictionary."""
        try:
            # Store in COO format for easier serialization
            coo_matrix = sparse_matrix.tocoo()

            dict_data = {
                "format": "scipy_sparse_coo",
                "shape": sparse_matrix.shape,
                "data": coo_matrix.data.tolist(),
                "row": coo_matrix.row.tolist(),
                "col": coo_matrix.col.tolist(),
                "dtype": str(sparse_matrix.dtype),
                "nnz": sparse_matrix.nnz,
            }

            context.intermediate_results["preservation_format"] = "coo_coordinates"
            context.performance_metrics["compression_ratio"] = (
                sparse_matrix.nnz / sparse_matrix.size
            )

            return dict_data

        except Exception as e:
            raise ConversionError(
                ConversionError.Type.CONVERSION_FAILED,
                f"Sparse to dict conversion failed: {str(e)}",
            )

    def _sparse_to_list(
        self, sparse_matrix: sparse.spmatrix, context: ConversionContext
    ) -> List[Any]:
        """Convert sparse matrix to list."""
        try:
            # Convert to dense first, then to list
            dense_array = sparse_matrix.toarray()
            list_data = dense_array.tolist()

            context.warnings.append("Sparse structure lost in list conversion")

            return list_data

        except Exception as e:
            raise ConversionError(
                ConversionError.Type.CONVERSION_FAILED,
                f"Sparse to list conversion failed: {str(e)}",
            )

    def _dataframe_to_sparse(
        self, df: pd.DataFrame, context: ConversionContext
    ) -> sparse.spmatrix:
        """Convert DataFrame to sparse matrix."""
        try:
            # Convert to numpy first, handling mixed types
            numeric_df = df.select_dtypes(include=[np.number])

            if len(numeric_df.columns) == 0:
                # No numeric columns - try dummy encoding
                encoded_df = pd.get_dummies(df, drop_first=True)
                array_data = encoded_df.values
                context.warnings.append("Applied dummy encoding for non-numeric data")
            elif len(numeric_df.columns) < len(df.columns):
                # Mixed types
                array_data = numeric_df.values
                dropped_cols = set(df.columns) - set(numeric_df.columns)
                context.warnings.append(
                    f"Dropped non-numeric columns: {list(dropped_cols)}"
                )
            else:
                array_data = df.values

            # Calculate density to choose format
            density = np.count_nonzero(array_data) / array_data.size
            context.performance_metrics["density"] = density

            if density > self.conversion_options.sparse_density_threshold:
                context.warnings.append(
                    f"Data density {density:.3f} above threshold "
                    f"{self.conversion_options.sparse_density_threshold}"
                )

            # Choose sparse format based on shape and density
            if density < 0.01:  # Very sparse
                sparse_matrix = sparse.coo_matrix(array_data)
            elif array_data.shape[0] > array_data.shape[1]:  # Tall matrix
                sparse_matrix = sparse.csc_matrix(array_data)
            else:  # Wide matrix
                sparse_matrix = sparse.csr_matrix(array_data)

            context.intermediate_results["sparse_format_chosen"] = type(
                sparse_matrix
            ).__name__

            return sparse_matrix

        except Exception as e:
            raise ConversionError(
                ConversionError.Type.CONVERSION_FAILED,
                f"DataFrame to sparse conversion failed: {str(e)}",
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
                    f"Flattened {original_shape} to {array.shape} for sparse conversion"
                )

            # Calculate density and choose format
            density = np.count_nonzero(array) / array.size
            context.performance_metrics["density"] = density

            if density < 0.01:
                sparse_matrix = sparse.coo_matrix(array)
                context.intermediate_results["format_reason"] = "very_sparse"
            elif density < 0.1:
                sparse_matrix = sparse.csr_matrix(array)
                context.intermediate_results["format_reason"] = "moderately_sparse"
            else:
                sparse_matrix = sparse.csr_matrix(array)
                context.warnings.append(
                    f"Dense data (density={density:.3f}) converted to sparse"
                )
                context.intermediate_results["format_reason"] = "dense_but_requested"

            return sparse_matrix

        except Exception as e:
            raise ConversionError(
                ConversionError.Type.CONVERSION_FAILED,
                f"NumPy to sparse conversion failed: {str(e)}",
            )

    def _dict_to_sparse(
        self, data: Dict[str, Any], context: ConversionContext
    ) -> sparse.spmatrix:
        """Convert dictionary to sparse matrix."""
        try:
            if "format" in data and data["format"] == "scipy_sparse_coo":
                # Restore from COO format
                sparse_matrix = sparse.coo_matrix(
                    (data["data"], (data["row"], data["col"])),
                    shape=data["shape"],
                    dtype=data.get("dtype", "float64"),
                )

                # Convert to CSR for efficiency
                sparse_matrix = sparse_matrix.tocsr()

                context.intermediate_results["restored_from_coo"] = True
                context.performance_metrics["nnz_restored"] = data["nnz"]

            else:
                # Convert dictionary values to array first, then to sparse
                if isinstance(data, dict) and all(
                    isinstance(v, (list, np.ndarray)) for v in data.values()
                ):
                    # Dictionary of arrays
                    array = np.array(list(data.values()))
                else:
                    # Try to convert to array
                    array = np.array(list(data.values())).reshape(-1, 1)
                    context.warnings.append(
                        "Dictionary structure not preserved in sparse conversion"
                    )

                sparse_matrix = sparse.csr_matrix(array)

            return sparse_matrix

        except Exception as e:
            raise ConversionError(
                ConversionError.Type.CONVERSION_FAILED,
                f"Dict to sparse conversion failed: {str(e)}",
            )

    def _list_to_sparse(
        self, data: List[Any], context: ConversionContext
    ) -> sparse.spmatrix:
        """Convert list to sparse matrix."""
        try:
            # Convert list to numpy array first
            array = np.array(data)

            # Ensure 2D
            if array.ndim == 1:
                array = array.reshape(-1, 1)
                context.intermediate_results["reshaped_from_1d"] = True
            elif array.ndim > 2:
                original_shape = array.shape
                array = array.reshape(array.shape[0], -1)
                context.warnings.append(
                    f"Flattened shape from {original_shape} to {array.shape}"
                )

            # Create sparse matrix
            sparse_matrix = sparse.csr_matrix(array)

            density = sparse_matrix.nnz / sparse_matrix.size
            context.performance_metrics["density"] = density

            if density > 0.5:
                context.warnings.append(
                    f"High density ({density:.3f}) list converted to sparse - may be inefficient"
                )

            return sparse_matrix

        except Exception as e:
            raise ConversionError(
                ConversionError.Type.CONVERSION_FAILED,
                f"List to sparse conversion failed: {str(e)}",
            )
