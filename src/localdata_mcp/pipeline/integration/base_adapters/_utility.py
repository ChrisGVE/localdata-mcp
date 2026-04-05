"""
Utility adapter implementations for common conversion patterns.

Provides PassThroughAdapter for same-format conversions and
ValidationAdapter for adding comprehensive validation layers.
"""

from typing import Any, List, Optional, Tuple

from ..interfaces import (
    DataFormat,
    ConversionRequest,
    ConversionResult,
    ConversionCost,
    ShimAdapter,
    ConversionError,
    ValidationResult,
)
from ._core import BaseShimAdapter, ConversionContext
from ....logging_manager import get_logger

logger = get_logger(__name__)


class PassThroughAdapter(BaseShimAdapter):
    """
    Utility adapter for pass-through conversions (no actual conversion needed).

    Useful for same-format conversions or when data is already in target format.
    """

    def __init__(self, adapter_id: str = "pass_through"):
        """Initialize PassThroughAdapter."""
        # Support all format-to-same-format conversions
        all_formats = list(DataFormat)
        supported_conversions = [(fmt, fmt) for fmt in all_formats]

        super().__init__(
            adapter_id=adapter_id,
            supported_conversions=supported_conversions,
            enable_caching=False,  # No need to cache pass-through
        )

    def can_convert(self, request: ConversionRequest) -> float:
        """Return 1.0 confidence for same-format conversions."""
        if request.source_format == request.target_format:
            return 1.0
        return 0.0

    def _perform_conversion(
        self, request: ConversionRequest, context: ConversionContext
    ) -> Any:
        """Return data unchanged for pass-through conversion."""
        if request.source_format != request.target_format:
            raise ConversionError(
                ConversionError.Type.TYPE_MISMATCH,
                f"PassThroughAdapter cannot convert {request.source_format} to {request.target_format}",
            )

        return request.source_data

    def estimate_cost(self, request: ConversionRequest) -> ConversionCost:
        """Return minimal cost for pass-through."""
        return ConversionCost(
            computational_cost=0.01,
            memory_cost_mb=0.0,
            time_estimate_seconds=0.001,
            io_operations=0,
            network_operations=0,
            quality_impact=0.0,
        )


class ValidationAdapter(BaseShimAdapter):
    """
    Utility adapter that adds comprehensive validation to any conversion.

    Can wrap other adapters to add validation layers without changing
    their core functionality.
    """

    def __init__(
        self,
        adapter_id: str,
        wrapped_adapter: Optional[ShimAdapter] = None,
        strict_validation: bool = True,
        **kwargs,
    ):
        """
        Initialize ValidationAdapter.

        Args:
            adapter_id: Unique identifier for this adapter
            wrapped_adapter: Optional adapter to wrap with validation
            strict_validation: Enable strict validation mode
            **kwargs: Additional arguments passed to BaseShimAdapter
        """
        kwargs["enable_validation"] = True  # Force validation enabled
        super().__init__(adapter_id, **kwargs)

        self.wrapped_adapter = wrapped_adapter
        self.strict_validation = strict_validation

    def can_convert(self, request: ConversionRequest) -> float:
        """Delegate to wrapped adapter if available."""
        if self.wrapped_adapter:
            return self.wrapped_adapter.can_convert(request)
        return super().can_convert(request)

    def validate_request(self, request: ConversionRequest) -> ValidationResult:
        """Enhanced request validation."""
        # Start with base validation
        base_result = super().validate_request(request)

        errors = base_result.errors.copy()
        warnings = base_result.warnings.copy()

        # Additional validation checks
        if request.source_data is None:
            errors.append("Source data is None")

        # Validate format specifications
        if request.format_spec:
            if request.format_spec.memory_constraints:
                if (
                    request.format_spec.memory_constraints.max_memory_mb
                    and request.format_spec.memory_constraints.max_memory_mb <= 0
                ):
                    errors.append(
                        "Invalid memory constraint: max_memory_mb must be positive"
                    )

            if request.format_spec.performance_requirements:
                perf_req = request.format_spec.performance_requirements
                if (
                    perf_req.max_execution_time_seconds
                    and perf_req.max_execution_time_seconds <= 0
                ):
                    errors.append(
                        "Invalid performance requirement: max_execution_time_seconds must be positive"
                    )

        # Validate data compatibility with formats
        if self.strict_validation:
            # Additional strict validation logic
            data_size = self._estimate_data_size(request.source_data)
            if data_size > 1024 * 1024 * 1024:  # 1GB
                warnings.append(
                    "Large dataset detected - consider streaming processing"
                )

        return ValidationResult(
            is_valid=len(errors) == 0,
            score=1.0 if len(errors) == 0 else max(0.0, 1.0 - len(errors) * 0.2),
            errors=errors,
            warnings=warnings,
            details={
                "validation_type": "enhanced",
                "strict_mode": self.strict_validation,
                "wrapped_adapter": self.wrapped_adapter.adapter_id
                if self.wrapped_adapter
                else None,
            },
        )

    def _perform_conversion(
        self, request: ConversionRequest, context: ConversionContext
    ) -> Any:
        """Perform conversion with validation."""
        if self.wrapped_adapter:
            # Use wrapped adapter for conversion
            wrapped_result = self.wrapped_adapter.convert(request)
            if not wrapped_result.success:
                raise ConversionError(
                    ConversionError.Type.CONVERSION_FAILED,
                    f"Wrapped adapter conversion failed: {'; '.join(wrapped_result.errors)}",
                )

            # Validate the result
            self._validate_conversion_result(
                request, wrapped_result.converted_data, context
            )
            return wrapped_result.converted_data
        else:
            # Use base implementation
            result = super()._perform_conversion(request, context)
            self._validate_conversion_result(request, result, context)
            return result

    def _validate_conversion_result(
        self,
        request: ConversionRequest,
        converted_data: Any,
        context: ConversionContext,
    ):
        """Validate conversion result."""
        # Check if conversion actually changed the format as expected
        if request.source_format != request.target_format:
            # Perform format detection on result to verify conversion
            # This is a basic check - could be enhanced with more sophisticated detection
            if type(converted_data) == type(request.source_data):
                context.warnings.append(
                    "Converted data type matches source type - conversion may not have occurred"
                )

        # Check for data loss indicators
        if hasattr(request.source_data, "__len__") and hasattr(
            converted_data, "__len__"
        ):
            if len(converted_data) != len(request.source_data):
                context.warnings.append(
                    f"Data length changed during conversion: {len(request.source_data)} -> {len(converted_data)}"
                )

        # Memory usage validation
        source_size = self._estimate_data_size(request.source_data)
        converted_size = self._estimate_data_size(converted_data)

        if converted_size > source_size * 3:  # 3x increase threshold
            context.warnings.append(
                f"Significant memory increase during conversion: {source_size} -> {converted_size} bytes"
            )
