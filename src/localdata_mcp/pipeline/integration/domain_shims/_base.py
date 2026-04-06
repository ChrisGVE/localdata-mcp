"""
Base domain shim implementation.

Provides BaseDomainShim - the abstract base class for all domain-specific shims
with common functionality for parameter mapping, result normalization, and
semantic context preservation.
"""

import time
from abc import abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ....logging_manager import get_logger
from ..converters import NumpyConverter, PandasConverter
from ..interfaces import (
    ConversionContext,
    ConversionCost,
    ConversionError,
    ConversionRequest,
    ConversionResult,
    DataFormat,
    ValidationResult,
)
from ..metadata_manager import MetadataManager
from ..shim_registry import AdapterConfig, EnhancedShimAdapter
from ..type_detection import TypeDetectionEngine
from ._types import DomainMapping, DomainShimType, SemanticContext

logger = get_logger(__name__)


class BaseDomainShim(EnhancedShimAdapter):
    """
    Base class for all domain-specific shims.

    Provides common functionality for domain integration including parameter
    mapping, result normalization, and semantic context preservation.
    """

    def __init__(
        self,
        adapter_id: str,
        domain_type: DomainShimType,
        config: Optional[AdapterConfig] = None,
        **kwargs,
    ):
        """
        Initialize BaseDomainShim.

        Args:
            adapter_id: Unique identifier for this shim
            domain_type: Type of domain this shim handles
            config: Optional adapter configuration
            **kwargs: Additional arguments
        """
        super().__init__(adapter_id, config, **kwargs)

        self.domain_type = domain_type
        self.supported_mappings: List[DomainMapping] = []

        # Initialize domain-specific components
        self._pandas_converter = PandasConverter()
        self._numpy_converter = NumpyConverter()
        self._type_detector = TypeDetectionEngine()
        self._metadata_manager = MetadataManager()

        # Domain knowledge base
        self._domain_schemas = {}
        self._parameter_maps = {}
        self._result_normalizers = {}

        # Initialize domain-specific configurations
        self._initialize_domain_knowledge()
        self._load_domain_mappings()

        logger.info(
            f"BaseDomainShim initialized",
            adapter_id=adapter_id,
            domain_type=domain_type.value,
        )

    def _initialize_impl(self) -> bool:
        """Initialize domain-specific components."""
        try:
            # Initialize converters
            self._pandas_converter.initialize()
            self._numpy_converter.initialize()

            # Activate converters
            self._pandas_converter.activate()
            self._numpy_converter.activate()

            logger.info(f"Domain shim '{self.adapter_id}' initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize domain shim '{self.adapter_id}': {e}")
            return False

    def _activate_impl(self) -> bool:
        """Activate domain shim for processing."""
        try:
            # Ensure all components are active
            if (
                self._pandas_converter.state.value != "active"
                or self._numpy_converter.state.value != "active"
            ):
                return False

            logger.info(f"Domain shim '{self.adapter_id}' activated")
            return True

        except Exception as e:
            logger.error(f"Failed to activate domain shim '{self.adapter_id}': {e}")
            return False

    @abstractmethod
    def _initialize_domain_knowledge(self) -> None:
        """Initialize domain-specific knowledge and mappings."""
        pass

    @abstractmethod
    def _load_domain_mappings(self) -> None:
        """Load domain-specific conversion mappings."""
        pass

    def can_convert(self, request: ConversionRequest) -> float:
        """
        Evaluate conversion capability with semantic understanding.

        Args:
            request: Conversion request to evaluate

        Returns:
            Confidence score (0-1) for handling this conversion
        """
        confidence = 0.0

        try:
            # Check if this is a domain-relevant conversion
            source_domain = self._extract_domain_from_format(request.source_format)
            target_domain = self._extract_domain_from_format(request.target_format)

            # Higher confidence for domain-specific conversions
            if (
                source_domain == self.domain_type.value
                or target_domain == self.domain_type.value
            ):
                confidence += 0.7

            # Check for supported mapping
            mapping = self._find_domain_mapping(source_domain, target_domain)
            if mapping:
                confidence += 0.2

            # Consider data characteristics
            if hasattr(request, "context") and request.context:
                semantic_match = self._evaluate_semantic_match(request.context)
                confidence += semantic_match * 0.1

            logger.debug(
                f"Conversion confidence for '{self.adapter_id}': {confidence}",
                source_format=request.source_format.value,
                target_format=request.target_format.value,
            )

            return min(confidence, 1.0)

        except Exception as e:
            logger.error(f"Error evaluating conversion capability: {e}")
            return 0.0

    def _convert_impl(self, request: ConversionRequest) -> ConversionResult:
        """
        Implement domain-aware conversion with semantic preservation.

        Args:
            request: Conversion request

        Returns:
            Conversion result with domain-specific enhancements
        """
        start_time = time.time()

        try:
            # Reject None source data early
            if request.source_data is None:
                return self._create_error_result(
                    request, "Source data is None", time.time() - start_time
                )

            # Extract semantic context
            semantic_context = self._extract_semantic_context(request)

            # Find appropriate domain mapping
            source_domain = self._extract_domain_from_format(request.source_format)
            target_domain = self._extract_domain_from_format(request.target_format)
            mapping = self._find_domain_mapping(source_domain, target_domain)

            if not mapping:
                # Fallback to generic conversion
                return self._generic_conversion(request, semantic_context)

            # Perform domain-specific conversion
            converted_data = self._perform_domain_conversion(
                request, mapping, semantic_context
            )

            # Normalize results for target domain
            normalized_data = self._normalize_results(
                converted_data, mapping, semantic_context
            )

            # Preserve and enhance metadata
            enhanced_metadata = self._enhance_metadata(
                request, mapping, semantic_context
            )

            # Calculate quality metrics
            quality_score = self._calculate_conversion_quality(
                request, normalized_data, mapping, semantic_context
            )

            execution_time = time.time() - start_time

            result = ConversionResult(
                converted_data=normalized_data,
                success=True,
                original_format=request.source_format,
                target_format=request.target_format,
                actual_format=request.target_format,
                metadata=enhanced_metadata,
                performance_metrics={
                    "execution_time": execution_time,
                    "adapter_id": self.adapter_id,
                    "domain_mapping": mapping.source_domain
                    + "_to_"
                    + mapping.target_domain,
                    "semantic_context": semantic_context.analytical_goal,
                    "quality_preservation": mapping.quality_preservation,
                },
                quality_score=quality_score,
                request_id=request.request_id,
                execution_time=execution_time,
            )

            logger.info(
                "Domain conversion completed successfully",
                request_id=request.request_id,
                adapter_id=self.adapter_id,
                execution_time=execution_time,
                quality_score=quality_score,
            )

            return result

        except Exception as e:
            logger.error(f"Domain conversion failed: {e}")
            return self._create_error_result(request, str(e), time.time() - start_time)

    def _extract_domain_from_format(self, data_format: DataFormat) -> str:
        """Extract domain name from data format."""
        format_value = data_format.value.lower()

        if "statistical" in format_value:
            return "statistical"
        elif "regression" in format_value:
            return "regression"
        elif "time_series" in format_value or "forecast" in format_value:
            return "time_series"
        elif (
            "clustering" in format_value
            or "pattern" in format_value
            or "classification" in format_value
        ):
            return "pattern_recognition"
        else:
            return "generic"

    def _find_domain_mapping(
        self, source_domain: str, target_domain: str
    ) -> Optional[DomainMapping]:
        """Find appropriate domain mapping for conversion."""
        for mapping in self.supported_mappings:
            if (
                mapping.source_domain == source_domain
                and mapping.target_domain == target_domain
            ):
                return mapping
        return None

    def _extract_semantic_context(self, request: ConversionRequest) -> SemanticContext:
        """Extract semantic context from conversion request."""
        context = (
            request.context if hasattr(request, "context") and request.context else None
        )

        analytical_goal = "data_transformation"
        domain_context = self.domain_type.value
        target_use_case = "analysis"

        if context:
            analytical_goal = getattr(context, "user_intention", analytical_goal)
            domain_context = getattr(context, "source_domain", domain_context)
            target_use_case = getattr(context, "target_domain", target_use_case)

        return SemanticContext(
            analytical_goal=analytical_goal,
            domain_context=domain_context,
            target_use_case=target_use_case,
            data_characteristics=self._analyze_data_characteristics(
                request.source_data
            ),
            transformation_hints={},
            quality_requirements={"preservation": 0.9, "accuracy": 0.95},
        )

    def _analyze_data_characteristics(self, data: Any) -> Dict[str, Any]:
        """Analyze characteristics of source data."""
        characteristics = {}

        try:
            if isinstance(data, pd.DataFrame):
                characteristics.update(
                    {
                        "data_type": "dataframe",
                        "shape": data.shape,
                        "columns": list(data.columns),
                        "dtypes": data.dtypes.to_dict(),
                        "missing_values": data.isnull().sum().to_dict(),
                        "numeric_columns": len(
                            data.select_dtypes(include=[np.number]).columns
                        ),
                        "categorical_columns": len(
                            data.select_dtypes(include=["object", "category"]).columns
                        ),
                    }
                )
            elif isinstance(data, np.ndarray):
                characteristics.update(
                    {
                        "data_type": "numpy_array",
                        "shape": data.shape,
                        "dtype": str(data.dtype),
                        "dimensions": data.ndim,
                        "size": data.size,
                    }
                )
            elif isinstance(data, dict):
                characteristics.update(
                    {
                        "data_type": "dictionary",
                        "keys": list(data.keys()),
                        "values_types": [type(v).__name__ for v in data.values()],
                    }
                )
            else:
                characteristics["data_type"] = type(data).__name__

        except Exception as e:
            logger.warning(f"Failed to analyze data characteristics: {e}")
            characteristics["analysis_error"] = str(e)

        return characteristics

    def _evaluate_semantic_match(self, context: ConversionContext) -> float:
        """Evaluate how well this shim matches the semantic context."""
        # Base implementation - can be overridden by specific shims
        return 0.5

    @abstractmethod
    def _perform_domain_conversion(
        self,
        request: ConversionRequest,
        mapping: DomainMapping,
        semantic_context: SemanticContext,
    ) -> Any:
        """Perform domain-specific conversion logic."""
        pass

    @abstractmethod
    def _normalize_results(
        self, data: Any, mapping: DomainMapping, semantic_context: SemanticContext
    ) -> Any:
        """Normalize conversion results for target domain."""
        pass

    def _generic_conversion(
        self, request: ConversionRequest, semantic_context: SemanticContext
    ) -> ConversionResult:
        """Fallback to generic conversion when no domain mapping exists."""
        try:
            # Use pandas converter as fallback
            return self._pandas_converter.convert(request)
        except Exception as e:
            logger.error(f"Generic conversion failed: {e}")
            return self._create_error_result(
                request, f"Generic conversion failed: {e}", 0.0
            )

    def _enhance_metadata(
        self,
        request: ConversionRequest,
        mapping: DomainMapping,
        semantic_context: SemanticContext,
    ) -> Dict[str, Any]:
        """Enhance metadata with domain-specific information."""
        enhanced_metadata = request.metadata.copy()

        enhanced_metadata.update(
            {
                "domain_shim": {
                    "adapter_id": self.adapter_id,
                    "domain_type": self.domain_type.value,
                    "mapping_used": {
                        "source_domain": mapping.source_domain,
                        "target_domain": mapping.target_domain,
                        "quality_preservation": mapping.quality_preservation,
                    },
                    "semantic_context": {
                        "analytical_goal": semantic_context.analytical_goal,
                        "domain_context": semantic_context.domain_context,
                        "target_use_case": semantic_context.target_use_case,
                    },
                }
            }
        )

        return enhanced_metadata

    def _calculate_conversion_quality(
        self,
        request: ConversionRequest,
        converted_data: Any,
        mapping: DomainMapping,
        semantic_context: SemanticContext,
    ) -> float:
        """Calculate quality score for domain conversion."""
        base_quality = mapping.quality_preservation

        # Adjust based on data characteristics preservation
        if "shape" in semantic_context.data_characteristics:
            original_shape = semantic_context.data_characteristics["shape"]
            if hasattr(converted_data, "shape"):
                if original_shape == converted_data.shape:
                    base_quality += 0.05
                else:
                    base_quality -= 0.1

        # Adjust based on semantic context match
        if semantic_context.analytical_goal in mapping.semantic_hints:
            base_quality += 0.05

        return min(max(base_quality, 0.0), 1.0)

    def _create_error_result(
        self, request: ConversionRequest, error_message: str, execution_time: float
    ) -> ConversionResult:
        """Create error result for failed conversions."""
        return ConversionResult(
            converted_data=request.source_data,
            success=False,
            original_format=request.source_format,
            target_format=request.target_format,
            actual_format=request.source_format,
            errors=[f"Domain conversion error: {error_message}"],
            performance_metrics={
                "execution_time": execution_time,
                "adapter_id": self.adapter_id,
                "error": error_message,
            },
            quality_score=0.0,
            request_id=request.request_id,
            execution_time=execution_time,
        )

    def estimate_cost(self, request: ConversionRequest) -> ConversionCost:
        """Estimate computational cost of domain conversion."""
        # Base cost estimation
        base_cost = 0.3  # Medium computational cost
        memory_cost = 50.0  # MB
        time_estimate = 2.0  # seconds

        # Adjust based on data size
        if hasattr(request.source_data, "shape"):
            data_size = np.prod(request.source_data.shape)
            if data_size > 100000:
                base_cost += 0.2
                memory_cost += data_size / 1000
                time_estimate += data_size / 50000

        # Adjust based on domain complexity
        source_domain = self._extract_domain_from_format(request.source_format)
        target_domain = self._extract_domain_from_format(request.target_format)

        if source_domain != target_domain:
            base_cost += 0.1
            time_estimate += 0.5

        return ConversionCost(
            computational_cost=min(base_cost, 1.0),
            memory_cost_mb=memory_cost,
            time_estimate_seconds=time_estimate,
            io_operations=1,
            network_operations=0,
            quality_impact=0.05,  # Small quality impact from domain conversion
        )

    def get_supported_conversions(self) -> List[Tuple[DataFormat, DataFormat]]:
        """Get list of supported conversion paths for this domain shim."""
        conversions = []

        # Add domain-specific conversions based on mappings
        for mapping in self.supported_mappings:
            source_formats = self._get_formats_for_domain(mapping.source_domain)
            target_formats = self._get_formats_for_domain(mapping.target_domain)

            for source_format in source_formats:
                for target_format in target_formats:
                    conversions.append((source_format, target_format))

        return conversions

    def _get_formats_for_domain(self, domain: str) -> List[DataFormat]:
        """Get data formats associated with a domain."""
        domain_formats = {
            "statistical": [
                DataFormat.STATISTICAL_RESULT,
                DataFormat.PANDAS_DATAFRAME,
                DataFormat.NUMPY_ARRAY,
            ],
            "regression": [
                DataFormat.REGRESSION_MODEL,
                DataFormat.PANDAS_DATAFRAME,
                DataFormat.NUMPY_ARRAY,
            ],
            "time_series": [
                DataFormat.TIME_SERIES,
                DataFormat.FORECAST_RESULT,
                DataFormat.PANDAS_DATAFRAME,
            ],
            "pattern_recognition": [
                DataFormat.PATTERN_RECOGNITION_RESULT,
                DataFormat.CLUSTERING_RESULT,
                DataFormat.NUMPY_ARRAY,
            ],
            "generic": [
                DataFormat.PANDAS_DATAFRAME,
                DataFormat.NUMPY_ARRAY,
                DataFormat.PYTHON_DICT,
                DataFormat.PYTHON_LIST,
            ],
        }

        return domain_formats.get(domain, domain_formats["generic"])
