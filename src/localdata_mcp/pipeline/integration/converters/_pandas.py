"""
Pandas DataFrame converter for bidirectional format conversion.

Handles conversions between DataFrame and other formats while preserving
index information, column names, data types, and categorical data.
"""

import time
from typing import Any, Dict, Optional

from ....logging_manager import get_logger
from ..base_adapters import BaseShimAdapter
from ..interfaces import (
    ConversionError,
    ConversionRequest,
    ConversionResult,
    DataFormat,
)
from ..metadata_manager import MetadataManager
from ..type_detection import TypeDetectionEngine
from ._common import ConversionContextInternal, ConversionOptions
from ._pandas_conversions import PandasConversionsMixin

logger = get_logger(__name__)


class PandasConverter(PandasConversionsMixin, BaseShimAdapter):
    """
    Bidirectional converter for pandas DataFrame format.

    Handles conversions between DataFrame and other formats while preserving
    index information, column names, data types, and categorical data.
    """

    def __init__(
        self,
        adapter_id: str = "pandas_converter",
        conversion_options: Optional[ConversionOptions] = None,
        **kwargs,
    ):
        """
        Initialize PandasConverter.

        Args:
            adapter_id: Unique identifier for this converter
            conversion_options: Options controlling conversion behavior
            **kwargs: Additional arguments passed to BaseShimAdapter
        """
        # Define supported conversions
        supported_conversions = [
            # From DataFrame
            (DataFormat.PANDAS_DATAFRAME, DataFormat.NUMPY_ARRAY),
            (DataFormat.PANDAS_DATAFRAME, DataFormat.SCIPY_SPARSE),
            (DataFormat.PANDAS_DATAFRAME, DataFormat.PYTHON_DICT),
            (DataFormat.PANDAS_DATAFRAME, DataFormat.PYTHON_LIST),
            (DataFormat.PANDAS_DATAFRAME, DataFormat.TIME_SERIES),
            (DataFormat.PANDAS_DATAFRAME, DataFormat.CATEGORICAL),
            # To DataFrame
            (DataFormat.NUMPY_ARRAY, DataFormat.PANDAS_DATAFRAME),
            (DataFormat.SCIPY_SPARSE, DataFormat.PANDAS_DATAFRAME),
            (DataFormat.PYTHON_DICT, DataFormat.PANDAS_DATAFRAME),
            (DataFormat.PYTHON_LIST, DataFormat.PANDAS_DATAFRAME),
            (DataFormat.TIME_SERIES, DataFormat.PANDAS_DATAFRAME),
            (DataFormat.CATEGORICAL, DataFormat.PANDAS_DATAFRAME),
        ]

        super().__init__(
            adapter_id=adapter_id, supported_conversions=supported_conversions, **kwargs
        )

        self.conversion_options = conversion_options or ConversionOptions()

        # Initialize components
        self._type_detector = TypeDetectionEngine()
        self._metadata_manager = MetadataManager()

        # Conversion statistics
        self._conversion_stats = {
            "successful_conversions": 0,
            "failed_conversions": 0,
            "total_rows_processed": 0,
            "average_conversion_time": 0.0,
        }

        logger.info(
            f"PandasConverter initialized",
            adapter_id=adapter_id,
            supported_conversions_count=len(supported_conversions),
        )

    def convert(self, request: ConversionRequest) -> ConversionResult:
        """Override convert to use internal context."""
        internal_context = ConversionContextInternal(request_id=request.request_id)

        try:
            # Validate request if enabled
            if self.enable_validation:
                validation_result = self.validate_request(request)
                if not validation_result.is_valid:
                    return self._create_error_result_internal(
                        request,
                        ConversionError.Type.SCHEMA_INVALID,
                        f"Request validation failed: {'; '.join(validation_result.errors)}",
                        internal_context,
                    )

            # Ensure adapter is fitted
            if not self._fitted:
                self.fit(request.source_data)

            # Perform the actual conversion
            converted_data = self._perform_conversion(request, internal_context)

            # Calculate performance metrics
            execution_time = time.time() - internal_context.start_time
            self._update_performance_stats(execution_time)

            # Create successful result
            result = ConversionResult(
                converted_data=converted_data,
                success=True,
                original_format=request.source_format,
                target_format=request.target_format,
                actual_format=request.target_format,
                metadata=self._preserve_metadata_internal(request, internal_context),
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

            logger.info(
                "Conversion completed successfully",
                request_id=request.request_id,
                adapter_id=self.adapter_id,
                execution_time=execution_time,
            )

            return result

        except ConversionError as e:
            logger.error(
                f"Conversion failed: {e}",
                request_id=request.request_id,
                adapter_id=self.adapter_id,
            )
            return self._create_error_result_internal(
                request,
                ConversionError.Type.CONVERSION_FAILED,
                str(e),
                internal_context,
            )
        except Exception as e:
            logger.error(
                f"Conversion failed unexpectedly: {e}",
                request_id=request.request_id,
                adapter_id=self.adapter_id,
            )
            return self._create_error_result_internal(
                request,
                ConversionError.Type.CONVERSION_FAILED,
                str(e),
                internal_context,
            )

    def _preserve_metadata_internal(
        self, request: ConversionRequest, context: ConversionContextInternal
    ) -> Dict[str, Any]:
        """Preserve and enhance metadata during conversion."""
        preserved_metadata = request.metadata.copy()

        # Store the original metadata for downstream access
        preserved_metadata["original_metadata"] = request.metadata.copy()

        # Add conversion metadata
        preserved_metadata.update(
            {
                "conversion_adapter": self.adapter_id,
                "conversion_timestamp": time.time(),
                "conversion_request_id": request.request_id,
                "source_format": request.source_format.value,
                "target_format": request.target_format.value,
            }
        )

        return preserved_metadata

    def _create_error_result_internal(
        self,
        request: ConversionRequest,
        error_type: ConversionError.Type,
        message: str,
        context: ConversionContextInternal,
    ) -> ConversionResult:
        """Create error result for failed conversions."""
        execution_time = time.time() - context.start_time

        return ConversionResult(
            converted_data=request.source_data,  # Return original data
            success=False,
            original_format=request.source_format,
            target_format=request.target_format,
            actual_format=request.source_format,  # No conversion happened
            errors=[f"{error_type.value}: {message}"],
            performance_metrics={
                "execution_time": execution_time,
                "adapter_id": self.adapter_id,
                "error_type": error_type.value,
            },
            quality_score=0.0,
            warnings=context.warnings,
            request_id=request.request_id,
            execution_time=execution_time,
        )

    def _perform_conversion(
        self, request: ConversionRequest, context: ConversionContextInternal
    ) -> Any:
        """
        Perform the actual data conversion.

        Args:
            request: Conversion request with source data and target format
            context: Conversion context for tracking

        Returns:
            Converted data
        """
        source_format = request.source_format
        target_format = request.target_format
        source_data = request.source_data

        logger.debug(f"Converting from {source_format.value} to {target_format.value}")

        try:
            # Route to appropriate conversion method
            if source_format == DataFormat.PANDAS_DATAFRAME:
                return self._convert_from_dataframe(source_data, target_format, context)
            elif target_format == DataFormat.PANDAS_DATAFRAME:
                return self._convert_to_dataframe(source_data, source_format, context)
            else:
                raise ConversionError(
                    ConversionError.Type.CONVERSION_FAILED,
                    f"Unsupported conversion path: {source_format.value} -> {target_format.value}",
                )

        except Exception as e:
            self._conversion_stats["failed_conversions"] += 1
            logger.error(f"Conversion failed: {e}")
            raise ConversionError(
                ConversionError.Type.CONVERSION_FAILED,
                f"Pandas conversion failed: {str(e)}",
            )

    def _calculate_quality_score(
        self,
        request: ConversionRequest,
        converted_data: Any,
        context: ConversionContextInternal,
    ) -> float:
        """Calculate conversion quality score."""
        base_score = 1.0

        # Reduce score for warnings
        if context.warnings:
            base_score -= len(context.warnings) * 0.05

        # Reduce score for data loss indicators
        if "shape_change" in context.intermediate_results:
            original_shape, new_shape = context.intermediate_results["shape_change"]
            if original_shape != new_shape:
                base_score -= 0.1

        # Reduce score for type conversions that might lose information
        if "sparsity_lost" in context.performance_metrics:
            base_score -= 0.1

        # Bonus for metadata preservation
        if context.intermediate_results.get("metadata_restored", False):
            base_score += 0.05

        return max(base_score, 0.0)
