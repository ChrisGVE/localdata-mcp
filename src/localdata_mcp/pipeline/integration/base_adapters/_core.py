"""
Core base adapter implementation for the Integration Shims Framework.

Provides the BaseShimAdapter with sklearn-compatible transformer interface,
comprehensive validation, error handling, and performance monitoring.
"""

import time
import hashlib
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from ..interfaces import (
    DataFormat,
    ConversionRequest,
    ConversionResult,
    ConversionCost,
    ShimAdapter,
    ConversionError,
    ValidationResult,
    MemoryConstraints,
    PerformanceRequirements,
)
from ...type_conversion import TypeInferenceEngine
from ....logging_manager import get_logger

logger = get_logger(__name__)


@dataclass
class ConversionContext:
    """Extended context for conversion operations."""

    request_id: str
    start_time: float = field(default_factory=time.time)
    memory_usage_start: Optional[float] = None
    intermediate_results: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


class BaseShimAdapter(ShimAdapter, BaseEstimator, TransformerMixin):
    """
    Base implementation of ShimAdapter with sklearn-compatible interface.

    Provides common functionality including:
    - Sklearn-compatible fit/transform pattern
    - Comprehensive validation and error handling
    - Performance monitoring and logging
    - Context preservation throughout conversion
    """

    def __init__(
        self,
        adapter_id: str,
        supported_conversions: Optional[List[Tuple[DataFormat, DataFormat]]] = None,
        memory_constraints: Optional[MemoryConstraints] = None,
        performance_requirements: Optional[PerformanceRequirements] = None,
        enable_caching: bool = True,
        enable_validation: bool = True,
    ):
        """
        Initialize BaseShimAdapter.

        Args:
            adapter_id: Unique identifier for this adapter
            supported_conversions: List of (source, target) format tuples
            memory_constraints: Memory usage constraints
            performance_requirements: Performance requirements
            enable_caching: Enable result caching
            enable_validation: Enable input/output validation
        """
        super().__init__(adapter_id)
        self.supported_conversions = supported_conversions or []
        self.memory_constraints = memory_constraints
        self.performance_requirements = performance_requirements
        self.enable_caching = enable_caching
        self.enable_validation = enable_validation

        # Internal state
        self._fitted = False
        self._lifecycle_state = "uninitialized"
        self._fit_metadata: Dict[str, Any] = {}
        self._conversion_cache: Dict[str, ConversionResult] = {}
        self._performance_stats: Dict[str, List[float]] = {}

        # Initialize type inference engine for data analysis
        self._type_engine = TypeInferenceEngine()

        logger.info(
            f"Initialized {self.__class__.__name__}",
            adapter_id=adapter_id,
            enable_caching=enable_caching,
            enable_validation=enable_validation,
        )

    def initialize(self) -> bool:
        """
        Initialize the adapter for use within lifecycle-managed components.

        Provides compatibility with the EnhancedShimAdapter lifecycle protocol
        so that BaseShimAdapter subclasses (e.g., PandasConverter, NumpyConverter)
        can be used as components within domain shims.

        Returns:
            True (always succeeds for base adapters)
        """
        self._lifecycle_state = "initialized"
        logger.debug(
            f"BaseShimAdapter '{self.adapter_id}' initialized (lifecycle compat)"
        )
        return True

    def activate(self) -> bool:
        """
        Activate the adapter for processing within lifecycle-managed components.

        Provides compatibility with the EnhancedShimAdapter lifecycle protocol.

        Returns:
            True (always succeeds for base adapters)
        """
        self._lifecycle_state = "active"
        logger.debug(
            f"BaseShimAdapter '{self.adapter_id}' activated (lifecycle compat)"
        )
        return True

    @property
    def state(self):
        """
        Return current lifecycle state as an object with a value attribute.

        Provides compatibility with AdapterLifecycleState enum used in
        EnhancedShimAdapter and domain shims.
        """

        class _State:
            def __init__(self, value: str):
                self.value = value

            def __repr__(self):
                return f"LifecycleState({self.value})"

        return _State(self._lifecycle_state)

    def fit(self, X: Any, y: Optional[Any] = None) -> "BaseShimAdapter":
        """
        Fit the adapter by analyzing data characteristics.

        This method analyzes input data to optimize conversion strategies,
        cache common patterns, and validate adapter configuration.

        Args:
            X: Input data to analyze
            y: Target data (ignored, for sklearn compatibility)

        Returns:
            self (fitted adapter)
        """
        start_time = time.time()

        try:
            # Detect input format if needed
            if hasattr(X, "__iter__") and not isinstance(X, (str, bytes)):
                # Sample data for analysis
                sample_data = self._sample_data_for_analysis(X)

                # Analyze data characteristics
                self._fit_metadata = self._analyze_data_characteristics(sample_data)

                logger.info(
                    "Data analysis completed for fitting",
                    adapter_id=self.adapter_id,
                    data_size=len(sample_data)
                    if hasattr(sample_data, "__len__")
                    else "unknown",
                    analysis_time=time.time() - start_time,
                )

            self._fitted = True
            return self

        except Exception as e:
            logger.error(f"Failed to fit adapter {self.adapter_id}: {e}")
            raise ConversionError(
                ConversionError.Type.ADAPTER_NOT_FOUND, f"Adapter fitting failed: {e}"
            )

    def transform(self, X: Any) -> Any:
        """
        Transform input data using fitted adapter.

        Args:
            X: Input data to transform

        Returns:
            Transformed data
        """
        if not self._fitted:
            # Auto-fit if not already fitted
            self.fit(X)

        # This is a base implementation - subclasses should override
        # for specific transformation logic
        return X

    def can_convert(self, request: ConversionRequest) -> float:
        """
        Evaluate capability to handle conversion request.

        Args:
            request: Conversion request to evaluate

        Returns:
            Confidence score (0-1)
        """
        # Check if conversion is in supported list
        conversion_tuple = (request.source_format, request.target_format)
        if conversion_tuple in self.supported_conversions:
            base_confidence = 0.8
        elif request.source_format == request.target_format:
            # Pass-through conversion
            base_confidence = 1.0
        else:
            base_confidence = 0.0

        # Adjust based on data characteristics if fitted
        if self._fitted and base_confidence > 0:
            # Consider memory constraints
            if self.memory_constraints:
                data_size = self._estimate_data_size(request.source_data)
                if (
                    self.memory_constraints.max_memory_mb
                    and data_size > self.memory_constraints.max_memory_mb * 1024 * 1024
                ):
                    if not self.memory_constraints.prefer_streaming:
                        base_confidence *= 0.5

            # Consider performance requirements
            if self.performance_requirements:
                estimated_time = self._estimate_conversion_time(request)
                if (
                    self.performance_requirements.max_execution_time_seconds
                    and estimated_time
                    > self.performance_requirements.max_execution_time_seconds
                ):
                    base_confidence *= 0.7

        return min(base_confidence, 1.0)

    def convert(self, request: ConversionRequest) -> ConversionResult:
        """
        Perform data conversion with comprehensive error handling.

        Args:
            request: Conversion request with all parameters

        Returns:
            Conversion result with converted data and metadata
        """
        context = ConversionContext(request_id=request.request_id)

        try:
            # Validate request if enabled
            if self.enable_validation:
                validation_result = self.validate_request(request)
                if not validation_result.is_valid:
                    return self._create_error_result(
                        request,
                        ConversionError.Type.SCHEMA_INVALID,
                        f"Request validation failed: {'; '.join(validation_result.errors)}",
                        context,
                    )

            # Check cache if enabled
            if self.enable_caching:
                cache_key = self._generate_cache_key(request)
                if cache_key in self._conversion_cache:
                    cached_result = self._conversion_cache[cache_key]
                    logger.info(
                        "Returned cached conversion result",
                        request_id=request.request_id,
                    )
                    return cached_result

            # Ensure adapter is fitted
            if not self._fitted:
                self.fit(request.source_data)

            # Perform the actual conversion
            converted_data = self._perform_conversion(request, context)

            # Calculate performance metrics
            execution_time = time.time() - context.start_time
            self._update_performance_stats(execution_time)

            # Create successful result
            result = ConversionResult(
                converted_data=converted_data,
                success=True,
                original_format=request.source_format,
                target_format=request.target_format,
                actual_format=request.target_format,
                metadata=self._preserve_metadata(request, context),
                performance_metrics={
                    "execution_time": execution_time,
                    "adapter_id": self.adapter_id,
                    **context.performance_metrics,
                },
                quality_score=self._calculate_quality_score(
                    request, converted_data, context
                ),
                warnings=context.warnings,
                request_id=request.request_id,
                execution_time=execution_time,
            )

            # Cache result if enabled
            if self.enable_caching:
                self._conversion_cache[cache_key] = result

            logger.info(
                "Conversion completed successfully",
                request_id=request.request_id,
                adapter_id=self.adapter_id,
                execution_time=execution_time,
            )

            return result

        except ConversionError:
            raise
        except Exception as e:
            logger.error(
                f"Conversion failed unexpectedly: {e}",
                request_id=request.request_id,
                adapter_id=self.adapter_id,
            )
            return self._create_error_result(
                request, ConversionError.Type.CONVERSION_FAILED, str(e), context
            )

    def estimate_cost(self, request: ConversionRequest) -> ConversionCost:
        """
        Estimate computational cost of conversion.

        Args:
            request: Conversion request to estimate

        Returns:
            Estimated cost breakdown
        """
        data_size = self._estimate_data_size(request.source_data)

        # Base cost estimation
        computational_cost = 0.1  # Low base cost
        memory_cost_mb = data_size / (1024 * 1024) * 1.2  # 20% overhead
        time_estimate = data_size / (10 * 1024 * 1024)  # 10MB/sec processing rate

        # Adjust based on conversion complexity
        if request.source_format != request.target_format:
            computational_cost += 0.3
            time_estimate *= 2

        # Consider memory constraints
        if (
            self.memory_constraints
            and self.memory_constraints.prefer_streaming
            and data_size > 100 * 1024 * 1024
        ):  # 100MB threshold
            # Streaming reduces memory but increases time
            memory_cost_mb *= 0.3
            time_estimate *= 1.5
            computational_cost += 0.2

        return ConversionCost(
            computational_cost=min(computational_cost, 1.0),
            memory_cost_mb=memory_cost_mb,
            time_estimate_seconds=time_estimate,
            io_operations=1 if data_size > 1024 * 1024 else 0,  # Large data needs I/O
            network_operations=0,
            quality_impact=0.0,
        )

    def get_supported_conversions(self) -> List[Tuple[DataFormat, DataFormat]]:
        """Return list of supported conversion paths."""
        return self.supported_conversions.copy()

    # Protected methods for subclass implementation

    def _perform_conversion(
        self, request: ConversionRequest, context: ConversionContext
    ) -> Any:
        """
        Perform the actual data conversion.

        This is the main method that subclasses should override to implement
        specific conversion logic.

        Args:
            request: Conversion request
            context: Conversion context for tracking

        Returns:
            Converted data
        """
        # Base implementation is pass-through
        if request.source_format == request.target_format:
            return request.source_data

        raise NotImplementedError(
            f"Conversion from {request.source_format} to {request.target_format} "
            f"not implemented in {self.__class__.__name__}"
        )

    def _sample_data_for_analysis(self, data: Any, max_samples: int = 1000) -> Any:
        """Sample data for efficient analysis during fitting."""
        if isinstance(data, pd.DataFrame):
            return data.head(max_samples) if len(data) > max_samples else data
        elif isinstance(data, pd.Series):
            return data.head(max_samples) if len(data) > max_samples else data
        elif isinstance(data, np.ndarray):
            return data[:max_samples] if len(data) > max_samples else data
        elif hasattr(data, "__len__") and len(data) > max_samples:
            return data[:max_samples]
        else:
            return data

    def _analyze_data_characteristics(self, data: Any) -> Dict[str, Any]:
        """Analyze data characteristics for optimization."""
        characteristics = {
            "analysis_timestamp": time.time(),
            "data_type": type(data).__name__,
        }

        if isinstance(data, pd.DataFrame):
            dtypes_dict = data.dtypes.to_dict()
            characteristics.update(
                {
                    "shape": data.shape,
                    "dtypes": dtypes_dict,
                    "current_dtype": dtypes_dict,
                    "memory_usage": data.memory_usage(deep=True).sum(),
                    "null_counts": data.isnull().sum().to_dict(),
                }
            )
        elif isinstance(data, pd.Series):
            characteristics.update(
                {
                    "length": len(data),
                    "dtype": str(data.dtype),
                    "current_dtype": str(data.dtype),
                    "memory_usage": data.memory_usage(deep=True),
                    "null_count": data.isnull().sum(),
                }
            )
        elif isinstance(data, np.ndarray):
            characteristics.update(
                {
                    "shape": data.shape,
                    "dtype": str(data.dtype),
                    "current_dtype": str(data.dtype),
                    "memory_usage": data.nbytes,
                }
            )

        return characteristics

    def _estimate_data_size(self, data: Any) -> int:
        """Estimate data size in bytes."""
        if hasattr(data, "memory_usage"):
            return int(data.memory_usage(deep=True).sum())
        elif hasattr(data, "nbytes"):
            return int(data.nbytes)
        elif hasattr(data, "__len__"):
            # Rough estimation for other sequences
            return len(data) * 64  # Assume 64 bytes per item
        else:
            return 1024  # Default size

    def _estimate_conversion_time(self, request: ConversionRequest) -> float:
        """Estimate conversion time based on data and operation."""
        data_size = self._estimate_data_size(request.source_data)
        base_time = data_size / (50 * 1024 * 1024)  # 50MB/sec base rate

        # Adjust based on conversion complexity
        if request.source_format != request.target_format:
            base_time *= 2

        return max(base_time, 0.001)  # Minimum 1ms

    def _preserve_metadata(
        self, request: ConversionRequest, context: ConversionContext
    ) -> Dict[str, Any]:
        """Preserve and enhance metadata during conversion."""
        preserved_metadata = request.metadata.copy()

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

        # Add fit metadata if available
        if self._fit_metadata:
            preserved_metadata["fit_metadata"] = self._fit_metadata

        return preserved_metadata

    def _calculate_quality_score(
        self,
        request: ConversionRequest,
        converted_data: Any,
        context: ConversionContext,
    ) -> float:
        """Calculate conversion quality score."""
        base_score = 1.0

        # Reduce score for warnings
        if context.warnings:
            base_score -= len(context.warnings) * 0.1

        # Pass-through conversions are perfect
        if request.source_format == request.target_format:
            return 1.0

        # Reduce score based on potential data loss indicators
        # This is a basic implementation - subclasses can enhance

        return max(base_score, 0.0)

    def _generate_cache_key(self, request: ConversionRequest) -> str:
        """Generate cache key for conversion request."""
        # Create hash of key request components
        key_data = f"{request.source_format.value}_{request.target_format.value}_{id(request.source_data)}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def _update_performance_stats(self, execution_time: float):
        """Update performance statistics."""
        if "execution_times" not in self._performance_stats:
            self._performance_stats["execution_times"] = []

        self._performance_stats["execution_times"].append(execution_time)

        # Keep only recent stats
        if len(self._performance_stats["execution_times"]) > 100:
            self._performance_stats["execution_times"] = self._performance_stats[
                "execution_times"
            ][-100:]

    def _create_error_result(
        self,
        request: ConversionRequest,
        error_type: ConversionError.Type,
        message: str,
        context: ConversionContext,
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
