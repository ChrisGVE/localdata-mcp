"""
Sparse matrix converter for bidirectional format conversion.

Handles conversions between sparse matrices and other formats with
intelligent density management and format optimization.
"""

import time
from typing import Any, Optional

from scipy import sparse

from ....logging_manager import get_logger
from ..base_adapters import BaseShimAdapter
from ..interfaces import (
    ConversionContext,
    ConversionError,
    ConversionRequest,
    ConversionResult,
    DataFormat,
)
from ..metadata_manager import MetadataManager
from ._common import ConversionContextInternal, ConversionOptions
from ._sparse_conversions import SparseConversionsMixin

logger = get_logger(__name__)


class SparseMatrixConverter(SparseConversionsMixin, BaseShimAdapter):
    """
    Bidirectional converter for scipy sparse matrix formats.

    Handles conversions between sparse matrices and other formats with
    intelligent density management and format optimization.
    """

    def __init__(
        self,
        adapter_id: str = "sparse_converter",
        conversion_options: Optional[ConversionOptions] = None,
        **kwargs,
    ):
        """Initialize SparseMatrixConverter."""

        supported_conversions = [
            # From sparse matrix
            (DataFormat.SCIPY_SPARSE, DataFormat.PANDAS_DATAFRAME),
            (DataFormat.SCIPY_SPARSE, DataFormat.NUMPY_ARRAY),
            (DataFormat.SCIPY_SPARSE, DataFormat.PYTHON_DICT),
            (DataFormat.SCIPY_SPARSE, DataFormat.PYTHON_LIST),
            # To sparse matrix
            (DataFormat.PANDAS_DATAFRAME, DataFormat.SCIPY_SPARSE),
            (DataFormat.NUMPY_ARRAY, DataFormat.SCIPY_SPARSE),
            (DataFormat.PYTHON_DICT, DataFormat.SCIPY_SPARSE),
            (DataFormat.PYTHON_LIST, DataFormat.SCIPY_SPARSE),
        ]

        super().__init__(
            adapter_id=adapter_id, supported_conversions=supported_conversions, **kwargs
        )

        self.conversion_options = conversion_options or ConversionOptions()
        self._metadata_manager = MetadataManager()

        # Sparse format preferences based on operations
        self._format_preferences = {
            "row_operations": sparse.csr_matrix,
            "column_operations": sparse.csc_matrix,
            "construction": sparse.coo_matrix,
            "general": sparse.csr_matrix,
        }

        logger.info(f"SparseMatrixConverter initialized", adapter_id=adapter_id)

    def convert(self, request: ConversionRequest) -> ConversionResult:
        """Override convert to use internal context."""
        internal_context = ConversionContextInternal(request_id=request.request_id)

        try:
            # Perform the actual conversion
            converted_data = self._perform_conversion(request, internal_context)

            # Calculate performance metrics
            execution_time = time.time() - internal_context.start_time

            # Create successful result
            result = ConversionResult(
                converted_data=converted_data,
                success=True,
                original_format=request.source_format,
                target_format=request.target_format,
                actual_format=request.target_format,
                metadata=request.metadata,
                performance_metrics={
                    "execution_time": execution_time,
                    "adapter_id": self.adapter_id,
                    **internal_context.performance_metrics,
                },
                quality_score=self._calculate_quality_score(
                    request, converted_data, internal_context
                ),
                warnings=internal_context.warnings,
                request_id=request.request_id,
                execution_time=execution_time,
            )

            return result

        except Exception as e:
            logger.error(f"Sparse conversion failed: {e}")
            return ConversionResult(
                converted_data=request.source_data,
                success=False,
                original_format=request.source_format,
                target_format=request.target_format,
                actual_format=request.source_format,
                errors=[str(e)],
                request_id=request.request_id,
                execution_time=time.time() - internal_context.start_time,
            )

    def _perform_conversion(
        self, request: ConversionRequest, context: ConversionContextInternal
    ) -> Any:
        """Perform sparse matrix conversion."""
        source_format = request.source_format
        target_format = request.target_format
        source_data = request.source_data

        try:
            if source_format == DataFormat.SCIPY_SPARSE:
                return self._convert_from_sparse(source_data, target_format, context)
            elif target_format == DataFormat.SCIPY_SPARSE:
                return self._convert_to_sparse(source_data, source_format, context)
            else:
                raise ConversionError(
                    ConversionError.Type.CONVERSION_FAILED,
                    f"Unsupported conversion path: {source_format.value} -> {target_format.value}",
                )

        except Exception as e:
            logger.error(f"Sparse matrix conversion failed: {e}")
            raise ConversionError(
                ConversionError.Type.CONVERSION_FAILED,
                f"Sparse conversion failed: {str(e)}",
            )

    def _convert_from_sparse(
        self,
        sparse_matrix: sparse.spmatrix,
        target_format: DataFormat,
        context: ConversionContext,
    ) -> Any:
        """Convert sparse matrix to other formats."""
        if not sparse.issparse(sparse_matrix):
            raise ConversionError(
                ConversionError.Type.TYPE_MISMATCH,
                f"Expected sparse matrix, got {type(sparse_matrix)}",
            )

        # Store sparse matrix metadata
        context.intermediate_results["original_sparse_info"] = {
            "format": type(sparse_matrix).__name__,
            "shape": sparse_matrix.shape,
            "nnz": sparse_matrix.nnz,
            "density": sparse_matrix.nnz
            / (sparse_matrix.shape[0] * sparse_matrix.shape[1]),
        }

        if target_format == DataFormat.PANDAS_DATAFRAME:
            return self._sparse_to_dataframe(sparse_matrix, context)
        elif target_format == DataFormat.NUMPY_ARRAY:
            return self._sparse_to_numpy(sparse_matrix, context)
        elif target_format == DataFormat.PYTHON_DICT:
            return self._sparse_to_dict(sparse_matrix, context)
        elif target_format == DataFormat.PYTHON_LIST:
            return self._sparse_to_list(sparse_matrix, context)
        else:
            raise ConversionError(
                ConversionError.Type.CONVERSION_FAILED,
                f"Unsupported target format: {target_format.value}",
            )

    def _convert_to_sparse(
        self, data: Any, source_format: DataFormat, context: ConversionContext
    ) -> sparse.spmatrix:
        """Convert other formats to sparse matrix."""
        if source_format == DataFormat.PANDAS_DATAFRAME:
            return self._dataframe_to_sparse(data, context)
        elif source_format == DataFormat.NUMPY_ARRAY:
            return self._numpy_to_sparse(data, context)
        elif source_format == DataFormat.PYTHON_DICT:
            return self._dict_to_sparse(data, context)
        elif source_format == DataFormat.PYTHON_LIST:
            return self._list_to_sparse(data, context)
        else:
            raise ConversionError(
                ConversionError.Type.CONVERSION_FAILED,
                f"Unsupported source format: {source_format.value}",
            )

    def _calculate_quality_score(
        self,
        request: ConversionRequest,
        converted_data: Any,
        context: ConversionContextInternal,
    ) -> float:
        """Calculate conversion quality score for sparse operations."""
        base_score = 1.0

        # Reduce score for warnings
        if context.warnings:
            base_score -= len(context.warnings) * 0.05

        # Reduce score for sparsity loss
        if context.performance_metrics.get("sparsity_lost", False):
            base_score -= 0.2

        # Reduce score for poor density match
        if "density" in context.performance_metrics:
            density = context.performance_metrics["density"]
            if density > 0.5 and request.target_format == DataFormat.SCIPY_SPARSE:
                base_score -= 0.1  # Dense data converted to sparse
            elif density < 0.1 and request.source_format == DataFormat.SCIPY_SPARSE:
                base_score -= 0.05  # Very sparse data densified

        return max(base_score, 0.0)
