"""
NumPy array converter for bidirectional format conversion.

Handles conversions between NumPy arrays and other formats while preserving
shape, data types, and array properties.
"""

import time
from typing import Any, Optional

import numpy as np

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
from ..type_detection import TypeDetectionEngine
from ._common import ConversionContextInternal, ConversionOptions
from ._numpy_conversions import NumpyConversionsMixin

logger = get_logger(__name__)


class NumpyConverter(NumpyConversionsMixin, BaseShimAdapter):
    """
    Bidirectional converter for NumPy array format.

    Handles conversions between NumPy arrays and other formats while preserving
    shape, data types, and array properties.
    """

    def __init__(
        self,
        adapter_id: str = "numpy_converter",
        conversion_options: Optional[ConversionOptions] = None,
        **kwargs,
    ):
        """Initialize NumpyConverter."""

        supported_conversions = [
            # From NumPy array
            (DataFormat.NUMPY_ARRAY, DataFormat.PANDAS_DATAFRAME),
            (DataFormat.NUMPY_ARRAY, DataFormat.SCIPY_SPARSE),
            (DataFormat.NUMPY_ARRAY, DataFormat.PYTHON_LIST),
            (DataFormat.NUMPY_ARRAY, DataFormat.PYTHON_DICT),
            # To NumPy array
            (DataFormat.PANDAS_DATAFRAME, DataFormat.NUMPY_ARRAY),
            (DataFormat.SCIPY_SPARSE, DataFormat.NUMPY_ARRAY),
            (DataFormat.PYTHON_LIST, DataFormat.NUMPY_ARRAY),
            (DataFormat.PYTHON_DICT, DataFormat.NUMPY_ARRAY),
        ]

        super().__init__(
            adapter_id=adapter_id, supported_conversions=supported_conversions, **kwargs
        )

        self.conversion_options = conversion_options or ConversionOptions()
        self._type_detector = TypeDetectionEngine()
        self._metadata_manager = MetadataManager()

        logger.info(f"NumpyConverter initialized", adapter_id=adapter_id)

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
                quality_score=1.0,  # Default quality score
                warnings=internal_context.warnings,
                request_id=request.request_id,
                execution_time=execution_time,
            )

            return result

        except Exception as e:
            logger.error(f"NumPy conversion failed: {e}")
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
        """Perform NumPy array conversion."""
        source_format = request.source_format
        target_format = request.target_format
        source_data = request.source_data

        try:
            if source_format == DataFormat.NUMPY_ARRAY:
                return self._convert_from_numpy(source_data, target_format, context)
            elif target_format == DataFormat.NUMPY_ARRAY:
                return self._convert_to_numpy(source_data, source_format, context)
            else:
                raise ConversionError(
                    ConversionError.Type.CONVERSION_FAILED,
                    f"Unsupported conversion path: {source_format.value} -> {target_format.value}",
                )

        except Exception as e:
            logger.error(f"NumPy conversion failed: {e}")
            raise ConversionError(
                ConversionError.Type.CONVERSION_FAILED,
                f"NumPy conversion failed: {str(e)}",
            )

    def _convert_from_numpy(
        self, array: np.ndarray, target_format: DataFormat, context: ConversionContext
    ) -> Any:
        """Convert NumPy array to other formats."""
        if not isinstance(array, np.ndarray):
            raise ConversionError(
                ConversionError.Type.TYPE_MISMATCH,
                f"Expected NumPy array, got {type(array)}",
            )

        # Store metadata
        original_metadata = self._metadata_manager.extract_metadata(
            array, DataFormat.NUMPY_ARRAY
        )
        context.intermediate_results["original_metadata"] = original_metadata

        if target_format == DataFormat.PANDAS_DATAFRAME:
            return self._numpy_to_dataframe(array, context)
        elif target_format == DataFormat.SCIPY_SPARSE:
            return self._numpy_to_sparse(array, context)
        elif target_format == DataFormat.PYTHON_LIST:
            return self._numpy_to_list(array, context)
        elif target_format == DataFormat.PYTHON_DICT:
            return self._numpy_to_dict(array, context)
        else:
            raise ConversionError(
                ConversionError.Type.CONVERSION_FAILED,
                f"Unsupported target format: {target_format.value}",
            )

    def _convert_to_numpy(
        self, data: Any, source_format: DataFormat, context: ConversionContext
    ) -> np.ndarray:
        """Convert other formats to NumPy array."""
        if source_format == DataFormat.PANDAS_DATAFRAME:
            return self._dataframe_to_numpy(data, context)
        elif source_format == DataFormat.SCIPY_SPARSE:
            return self._sparse_to_numpy(data, context)
        elif source_format == DataFormat.PYTHON_LIST:
            return self._list_to_numpy(data, context)
        elif source_format == DataFormat.PYTHON_DICT:
            return self._dict_to_numpy(data, context)
        else:
            raise ConversionError(
                ConversionError.Type.CONVERSION_FAILED,
                f"Unsupported source format: {source_format.value}",
            )
