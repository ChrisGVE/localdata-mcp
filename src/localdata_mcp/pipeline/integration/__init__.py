"""
Integration Shims and Data Type Conversion Framework

This package provides comprehensive data type conversion and format adaptation
capabilities for seamless cross-domain pipeline composition in LocalData MCP v2.0.

Key Components:
- ShimAdapter: Base framework for format conversion adapters
- ConversionRegistry: Plugin system for dynamic converter discovery
- CompatibilityMatrix: Mapping system for domain compatibility
- TypeDetectionEngine: Advanced type detection and schema inference
- MetadataManager: Comprehensive metadata preservation system

The framework enables LLM agents to compose complex analytical workflows
spanning multiple domains without worrying about data format incompatibilities.
"""

from .interfaces import (
    # Core data structures
    DataFormat,
    ConversionRequest,
    ConversionResult,
    ConversionCost,
    ConversionPath,
    
    # Base interfaces
    ShimAdapter,
    TypeDetector,
    MetadataPreserver,
    
    # Registry and management
    ConversionRegistry,
    CompatibilityMatrix,
    
    # Error handling
    ConversionError,
    ValidationResult,
)

from .base_adapters import (
    # Base adapter implementations
    BaseShimAdapter,
    StreamingShimAdapter,
    CachingShimAdapter,
    
    # Utility adapters
    PassThroughAdapter,
    ValidationAdapter,
)

from .type_detection import (
    TypeDetectionEngine,
    FormatDetectionResult,
    SchemaInfo,
    detect_data_format,
)

from .metadata_manager import (
    MetadataManager,
    PreservationRule,
    MetadataSchema,
    PreservationStrategy,
    MetadataType,
    MetadataLineage,
    create_preservation_rule,
    create_metadata_schema,
)

from .converters import (
    # Core converters
    PandasConverter,
    NumpyConverter,
    SparseMatrixConverter,
    
    # Conversion options and utilities
    ConversionOptions,
    ConversionQuality,
    
    # Factory functions
    create_pandas_converter,
    create_numpy_converter,
    create_sparse_converter,
    create_memory_efficient_options,
    create_high_fidelity_options,
    create_streaming_options,
)

from .compatibility_matrix import (
    # Core compatibility system
    PipelineCompatibilityMatrix,
    CompatibilityLevel,
    DomainProfile,
    
    # Factory functions
    create_compatibility_matrix,
    create_minimal_compatibility_matrix,
    
    # Utility functions
    assess_pipeline_compatibility,
    find_optimal_format_for_domains,
    suggest_pipeline_improvements,
)

from .shim_registry import (
    # Enhanced ShimAdapter and Registry
    EnhancedShimAdapter,
    ShimRegistry,
    AdapterConfig,
    AdapterMetrics,
    HealthCheckResult,
    AdapterLifecycleState,
    
    # Factory functions
    create_shim_registry,
    create_adapter_config,
    
    # Utility functions
    validate_adapter_dependencies,
    monitor_adapter_performance,
)

__all__ = [
    # Core data structures
    'DataFormat',
    'ConversionRequest', 
    'ConversionResult',
    'ConversionCost',
    'ConversionPath',
    
    # Base interfaces
    'ShimAdapter',
    'TypeDetector',
    'MetadataPreserver',
    
    # Registry and management
    'ConversionRegistry',
    'CompatibilityMatrix',
    
    # Base adapter implementations
    'BaseShimAdapter',
    'StreamingShimAdapter', 
    'CachingShimAdapter',
    'PassThroughAdapter',
    'ValidationAdapter',
    
    # Type detection
    'TypeDetectionEngine',
    'FormatDetectionResult',
    'SchemaInfo',
    'detect_data_format',
    
    # Metadata management
    'MetadataManager',
    'PreservationRule',
    'MetadataSchema',
    'PreservationStrategy',
    'MetadataType',
    'MetadataLineage',
    'create_preservation_rule',
    'create_metadata_schema',
    
    # Core converters
    'PandasConverter',
    'NumpyConverter', 
    'SparseMatrixConverter',
    
    # Conversion options and utilities
    'ConversionOptions',
    'ConversionQuality',
    
    # Factory functions
    'create_pandas_converter',
    'create_numpy_converter',
    'create_sparse_converter',
    'create_memory_efficient_options',
    'create_high_fidelity_options',
    'create_streaming_options',
    
    # Compatibility matrix
    'PipelineCompatibilityMatrix',
    'CompatibilityLevel',
    'DomainProfile',
    'create_compatibility_matrix',
    'create_minimal_compatibility_matrix',
    'assess_pipeline_compatibility',
    'find_optimal_format_for_domains',
    'suggest_pipeline_improvements',
    
    # Error handling
    'ConversionError',
    'ValidationResult',
    
    # Enhanced ShimAdapter and Registry
    'EnhancedShimAdapter',
    'ShimRegistry', 
    'AdapterConfig',
    'AdapterMetrics',
    'HealthCheckResult',
    'AdapterLifecycleState',
    'create_shim_registry',
    'create_adapter_config',
    'validate_adapter_dependencies',
    'monitor_adapter_performance',
]

# Package metadata
__version__ = "2.0.0"
__author__ = "LocalData MCP Team"
__description__ = "Integration Shims and Data Type Conversion Framework"