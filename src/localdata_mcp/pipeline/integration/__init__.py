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
    
    # Error handling
    'ConversionError',
    'ValidationResult',
]

# Package metadata
__version__ = "2.0.0"
__author__ = "LocalData MCP Team"
__description__ = "Integration Shims and Data Type Conversion Framework"