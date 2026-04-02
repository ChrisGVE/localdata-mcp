"""
Core interfaces and data structures for the Integration Shims Framework.

This module defines the fundamental interfaces, data structures, and abstract
base classes that form the foundation of the integration system.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union, Iterator
import time
from datetime import datetime

from ..base import PipelineError


class DataFormat(Enum):
    """Enumeration of all supported data formats across domains."""
    
    # Basic data structures
    PANDAS_DATAFRAME = "pandas_dataframe"
    NUMPY_ARRAY = "numpy_array"
    SCIPY_SPARSE = "scipy_sparse_matrix"
    PYTHON_LIST = "python_list"
    PYTHON_DICT = "python_dict"
    
    # Specialized scientific formats
    TIME_SERIES = "time_series"
    GEOSPATIAL = "geospatial"
    CATEGORICAL = "categorical"
    MIXED_TYPES = "mixed_types"
    HIERARCHICAL = "hierarchical"
    
    # Domain-specific result formats
    STATISTICAL_RESULT = "statistical_result"
    REGRESSION_MODEL = "regression_model"
    CLUSTERING_RESULT = "clustering_result"
    FORECAST_RESULT = "forecast_result"
    OPTIMIZATION_RESULT = "optimization_result"
    PATTERN_RECOGNITION_RESULT = "pattern_recognition_result"
    GEOSPATIAL_RESULT = "geospatial_result"
    BUSINESS_INTELLIGENCE_RESULT = "business_intelligence_result"
    
    # Complex multi-dimensional formats
    MULTI_INDEX = "multi_index"
    MULTI_LEVEL = "multi_level"
    STREAMING = "streaming"
    CHUNKED = "chunked"
    
    # File-like formats
    JSON = "json"
    CSV = "csv"
    PARQUET = "parquet"
    HDF5 = "hdf5"
    
    # Unknown or auto-detect
    UNKNOWN = "unknown"
    AUTO_DETECT = "auto_detect"


@dataclass
class MemoryConstraints:
    """Memory constraints for conversion operations."""
    max_memory_mb: Optional[int] = None
    prefer_streaming: bool = False
    chunk_size: Optional[int] = None
    memory_efficient: bool = True


@dataclass
class PerformanceRequirements:
    """Performance requirements for conversion operations."""
    max_execution_time_seconds: Optional[float] = None
    prefer_parallel: bool = False
    cpu_intensive_allowed: bool = True
    io_intensive_allowed: bool = True


@dataclass
class DataFormatSpec:
    """Specification of data format requirements and constraints."""
    format_type: DataFormat
    schema_requirements: Dict[str, Any] = field(default_factory=dict)
    memory_constraints: Optional[MemoryConstraints] = None
    performance_requirements: Optional[PerformanceRequirements] = None
    metadata_requirements: List[str] = field(default_factory=list)
    quality_requirements: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversionContext:
    """Context information for conversion operations."""
    source_domain: str
    target_domain: str
    user_intention: str = ""
    pipeline_context: Dict[str, Any] = field(default_factory=dict)
    performance_hints: Dict[str, Any] = field(default_factory=dict)
    debugging_enabled: bool = False


@dataclass
class ConversionRequest:
    """Request for data format conversion between domains."""
    source_data: Any
    source_format: DataFormat
    target_format: DataFormat
    metadata: Dict[str, Any] = field(default_factory=dict)
    context: ConversionContext = field(default_factory=lambda: ConversionContext(source_domain='unknown', target_domain='unknown'))
    format_spec: Optional[DataFormatSpec] = None
    request_id: str = field(default_factory=lambda: f"req_{int(time.time() * 1000)}")
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ConversionCost:
    """Estimated or actual cost of a conversion operation."""
    computational_cost: float  # Relative cost (0-1 scale)
    memory_cost_mb: float
    time_estimate_seconds: float
    io_operations: int = 0
    network_operations: int = 0
    quality_impact: float = 0.0  # Negative values indicate quality loss


@dataclass
class ConversionResult:
    """Result of a data format conversion operation."""
    converted_data: Any
    success: bool
    original_format: DataFormat
    target_format: DataFormat
    actual_format: DataFormat  # May differ from target if fallback used
    metadata: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    quality_score: float = 1.0  # 0-1 scale, 1 = perfect conversion
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    alternative_suggestions: List[str] = field(default_factory=list)
    conversion_cost: Optional[ConversionCost] = None
    request_id: str = ""
    execution_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ConversionStep:
    """Single step in a multi-step conversion path."""
    adapter_id: str
    source_format: DataFormat
    target_format: DataFormat
    estimated_cost: ConversionCost
    confidence: float  # 0-1 confidence in conversion success
    required: bool = True  # False for optional optimization steps


@dataclass
class ConversionPath:
    """Complete path for converting between two formats."""
    source_format: DataFormat
    target_format: DataFormat
    steps: List[ConversionStep]
    total_cost: ConversionCost
    success_probability: float
    path_id: str = field(default_factory=lambda: f"path_{int(time.time() * 1000)}")


class ConversionError(PipelineError):
    """Base class for conversion-related errors."""
    
    class Type(Enum):
        TYPE_MISMATCH = "type_mismatch"
        SCHEMA_INVALID = "schema_invalid"
        MEMORY_EXCEEDED = "memory_exceeded"
        TIMEOUT = "timeout"
        ADAPTER_NOT_FOUND = "adapter_not_found"
        CONVERSION_FAILED = "conversion_failed"
        METADATA_LOSS = "metadata_loss"
        QUALITY_DEGRADED = "quality_degraded"
    
    def __init__(self, error_type: Type, message: str, context: Optional[Dict[str, Any]] = None):
        self.error_type = error_type
        self.context = context or {}
        super().__init__(message)


@dataclass
class ValidationResult:
    """Result of validation operations."""
    is_valid: bool
    score: float  # 0-1 validation score
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)
    suggestions: List[str] = field(default_factory=list)


@dataclass
class CompatibilityScore:
    """Score representing compatibility between two components."""
    score: float  # 0-1 compatibility score
    direct_compatible: bool
    conversion_required: bool
    conversion_path: Optional[ConversionPath] = None
    compatibility_issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


# Abstract Base Classes

class ShimAdapter(ABC):
    """Abstract base class for all integration shim adapters."""
    
    def __init__(self, adapter_id: str):
        self.adapter_id = adapter_id
        self.supported_conversions: List[Tuple[DataFormat, DataFormat]] = []
        self.performance_profile: Dict[str, Any] = {}
        self.quality_profile: Dict[str, Any] = {}
    
    @abstractmethod
    def can_convert(self, request: ConversionRequest) -> float:
        """
        Return confidence score (0-1) for handling this conversion.
        
        Args:
            request: Conversion request to evaluate
            
        Returns:
            Confidence score (0 = cannot handle, 1 = perfect match)
        """
        pass
    
    @abstractmethod
    def convert(self, request: ConversionRequest) -> ConversionResult:
        """
        Perform the data conversion.
        
        Args:
            request: Conversion request with all parameters
            
        Returns:
            Conversion result with converted data and metadata
        """
        pass
    
    @abstractmethod
    def estimate_cost(self, request: ConversionRequest) -> ConversionCost:
        """
        Estimate computational cost of conversion.
        
        Args:
            request: Conversion request to estimate
            
        Returns:
            Estimated cost breakdown
        """
        pass
    
    @abstractmethod
    def get_supported_conversions(self) -> List[Tuple[DataFormat, DataFormat]]:
        """
        Return list of supported conversion paths.
        
        Returns:
            List of (source_format, target_format) tuples
        """
        pass
    
    def validate_request(self, request: ConversionRequest) -> ValidationResult:
        """
        Validate conversion request before processing.
        
        Args:
            request: Request to validate
            
        Returns:
            Validation result with errors and warnings
        """
        errors = []
        warnings = []
        
        # Check if conversion is supported
        supported_conversions = self.get_supported_conversions()
        conversion_supported = any(
            source == request.source_format and target == request.target_format
            for source, target in supported_conversions
        )
        
        if not conversion_supported:
            errors.append(f"Conversion from {request.source_format} to {request.target_format} not supported")
        
        # Validate source data
        if request.source_data is None:
            errors.append("Source data cannot be None")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            score=1.0 if len(errors) == 0 else 0.0,
            errors=errors,
            warnings=warnings
        )


class TypeDetector(ABC):
    """Abstract base class for data type detection."""
    
    @abstractmethod
    def detect_format(self, data: Any) -> 'FormatDetectionResult':
        """Detect the format of input data."""
        pass
    
    @abstractmethod
    def get_confidence_threshold(self) -> float:
        """Get minimum confidence threshold for detection."""
        pass


class MetadataPreserver(ABC):
    """Abstract base class for metadata preservation."""
    
    @abstractmethod
    def extract_metadata(self, data: Any, format_type: DataFormat) -> Dict[str, Any]:
        """Extract metadata from data."""
        pass
    
    @abstractmethod
    def apply_metadata(self, data: Any, metadata: Dict[str, Any], 
                      target_format: DataFormat) -> Any:
        """Apply metadata to converted data."""
        pass
    
    @abstractmethod
    def merge_metadata(self, metadata_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge metadata from multiple sources."""
        pass


class ConversionRegistry(ABC):
    """Abstract base class for conversion registry."""
    
    @abstractmethod
    def register_adapter(self, adapter: ShimAdapter) -> None:
        """Register a conversion adapter."""
        pass
    
    @abstractmethod
    def get_adapter(self, adapter_id: str) -> Optional[ShimAdapter]:
        """Get adapter by ID."""
        pass
    
    @abstractmethod
    def find_conversion_path(self, source_format: DataFormat, 
                           target_format: DataFormat) -> Optional[ConversionPath]:
        """Find optimal conversion path between formats."""
        pass
    
    @abstractmethod
    def get_compatible_adapters(self, request: ConversionRequest) -> List[Tuple[ShimAdapter, float]]:
        """Get adapters that can handle the request with confidence scores."""
        pass


class CompatibilityMatrix(ABC):
    """Abstract base class for compatibility matrix."""
    
    @abstractmethod
    def get_compatibility(self, source_format: DataFormat, 
                         target_format: DataFormat) -> CompatibilityScore:
        """Get compatibility score between formats."""
        pass
    
    @abstractmethod
    def register_domain_requirements(self, domain_name: str, 
                                   requirements: DataFormatSpec) -> None:
        """Register domain's format requirements."""
        pass
    
    @abstractmethod
    def validate_pipeline(self, pipeline_steps: List[str]) -> ValidationResult:
        """Validate pipeline compatibility."""
        pass


# Utility Classes

@dataclass
class DomainRequirements:
    """Requirements specification for a domain."""
    domain_name: str
    input_formats: List[DataFormat]
    output_formats: List[DataFormat]
    preferred_format: DataFormat
    metadata_requirements: List[str] = field(default_factory=list)
    performance_requirements: Optional[PerformanceRequirements] = None
    memory_constraints: Optional[MemoryConstraints] = None
    quality_requirements: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AdapterProfile:
    """Profile information for a shim adapter."""
    adapter_id: str
    name: str
    description: str
    supported_conversions: List[Tuple[DataFormat, DataFormat]]
    performance_characteristics: Dict[str, Any] = field(default_factory=dict)
    quality_characteristics: Dict[str, Any] = field(default_factory=dict)
    memory_requirements: Optional[MemoryConstraints] = None
    dependencies: List[str] = field(default_factory=list)
    version: str = "1.0.0"


@dataclass
class ConversionStats:
    """Statistics for conversion operations."""
    total_conversions: int = 0
    successful_conversions: int = 0
    failed_conversions: int = 0
    average_execution_time: float = 0.0
    total_data_processed_mb: float = 0.0
    cache_hit_rate: float = 0.0
    most_common_conversions: Dict[str, int] = field(default_factory=dict)
    error_distribution: Dict[str, int] = field(default_factory=dict)


# Factory Functions

def create_conversion_request(source_data: Any, 
                            source_format: DataFormat, 
                            target_format: DataFormat,
                            **kwargs) -> ConversionRequest:
    """Factory function to create a conversion request."""
    # Create context with appropriate domains if not provided
    if 'context' not in kwargs:
        context = ConversionContext(
            source_domain=source_format.value.split('_')[0],  # Use format prefix as domain
            target_domain=target_format.value.split('_')[0],
            user_intention=f"Convert {source_format.value} to {target_format.value}"
        )
        kwargs['context'] = context
    
    return ConversionRequest(
        source_data=source_data,
        source_format=source_format,
        target_format=target_format,
        **kwargs
    )


def create_format_spec(format_type: DataFormat, 
                      **requirements) -> DataFormatSpec:
    """Factory function to create a format specification."""
    return DataFormatSpec(
        format_type=format_type,
        **requirements
    )


# Constants

DEFAULT_MEMORY_CONSTRAINTS = MemoryConstraints(
    max_memory_mb=1024,
    prefer_streaming=False,
    chunk_size=10000,
    memory_efficient=True
)

DEFAULT_PERFORMANCE_REQUIREMENTS = PerformanceRequirements(
    max_execution_time_seconds=300,
    prefer_parallel=True,
    cpu_intensive_allowed=True,
    io_intensive_allowed=True
)

# Conversion quality thresholds
QUALITY_THRESHOLDS = {
    'excellent': 0.95,
    'good': 0.8,
    'acceptable': 0.6,
    'poor': 0.4,
    'unacceptable': 0.2
}