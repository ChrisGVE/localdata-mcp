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

from .base_adapters import (  # Base adapter implementations; Utility adapters
    BaseShimAdapter,
    CachingShimAdapter,
    PassThroughAdapter,
    StreamingShimAdapter,
    ValidationAdapter,
)
from .compatibility_matrix import (  # Core compatibility system; Factory functions; Utility functions
    CompatibilityLevel,
    DomainProfile,
    PipelineCompatibilityMatrix,
    assess_pipeline_compatibility,
    create_compatibility_matrix,
    create_minimal_compatibility_matrix,
    find_optimal_format_for_domains,
    suggest_pipeline_improvements,
)
from .converters import (  # Core converters; Conversion options and utilities; Factory functions
    ConversionOptions,
    ConversionQuality,
    NumpyConverter,
    PandasConverter,
    SparseMatrixConverter,
    create_high_fidelity_options,
    create_memory_efficient_options,
    create_numpy_converter,
    create_pandas_converter,
    create_sparse_converter,
    create_streaming_options,
)
from .domain_shims import (  # Domain-specific shim classes; Domain configuration and mapping; Factory functions; Utility functions
    BaseDomainShim,
    DomainMapping,
    DomainShimType,
    PatternRecognitionShim,
    RegressionShim,
    SemanticContext,
    StatisticalShim,
    TimeSeriesShim,
    create_all_domain_shims,
    create_pattern_recognition_shim,
    create_regression_shim,
    create_statistical_shim,
    create_time_series_shim,
    get_compatible_domain_shims,
    validate_domain_shim_configuration,
)
from .error_recovery import (  # Core error recovery classes; Error recovery data structures; Factory functions
    AlternativePathwayEngine,
    ConversionErrorHandler,
    ErrorClassificationEnhanced,
    RecoveryResult,
    RecoveryStrategyEngine,
    RollbackManager,
    create_alternative_pathway_engine,
    create_complete_error_recovery_system,
    create_conversion_error_handler,
    create_error_recovery_framework,
    create_recovery_strategy_engine,
    create_rollback_manager,
)
from .interfaces import (  # Core data structures; Base interfaces; Registry and management; Error handling
    CompatibilityMatrix,
    ConversionCost,
    ConversionError,
    ConversionPath,
    ConversionRegistry,
    ConversionRequest,
    ConversionResult,
    DataFormat,
    MetadataPreserver,
    ShimAdapter,
    TypeDetector,
    ValidationResult,
)
from .metadata_manager import (
    MetadataLineage,
    MetadataManager,
    MetadataSchema,
    MetadataType,
    PreservationRule,
    PreservationStrategy,
    create_metadata_schema,
    create_preservation_rule,
)
from .performance_optimization import (  # Core performance optimization classes; Performance data structures
    CachedConversion,
    CacheEvictionPolicy,
    CacheStatistics,
    ConversionCache,
    LazyConversionState,
    LazyConverter,
    LazyLoadingManager,
)
from .pipeline_analyzer import (  # Core pipeline analysis classes; Data structures; Enums; Factory functions; Utility functions
    AnalysisType,
    IncompatibilityIssue,
    InjectionStrategy,
    OptimizationCriteria,
    PipelineAnalysisResult,
    PipelineAnalyzer,
    PipelineConnection,
    PipelineStep,
    PipelineValidator,
    ShimInjector,
    ShimRecommendation,
    analyze_and_fix_pipeline,
    create_optimization_criteria,
    create_pipeline_analyzer,
    create_pipeline_step,
    create_pipeline_validator,
    create_shim_injector,
)
from .schema_validation import (  # Schema validation core classes; Validation components; Results and enums
    DataSchema,
    NullValidationRule,
    RangeValidationRule,
    SchemaConformanceLevel,
    SchemaConstraint,
    SchemaEvolutionManager,
    SchemaInferenceEngine,
    SchemaInferenceResult,
    SchemaValidationLevel,
    SchemaValidationResult,
    SchemaValidator,
    TypeValidationRule,
    ValidationError,
    ValidationRule,
    ValidationRuleType,
)
from .shim_registry import (  # Enhanced ShimAdapter and Registry; Factory functions; Utility functions
    AdapterConfig,
    AdapterLifecycleState,
    AdapterMetrics,
    EnhancedShimAdapter,
    HealthCheckResult,
    ShimRegistry,
    create_adapter_config,
    create_shim_registry,
    monitor_adapter_performance,
    validate_adapter_dependencies,
)
from .type_detection import (
    FormatDetectionResult,
    SchemaInfo,
    TypeDetectionEngine,
    detect_data_format,
)

__all__ = [
    # Core data structures
    "DataFormat",
    "ConversionRequest",
    "ConversionResult",
    "ConversionCost",
    "ConversionPath",
    # Base interfaces
    "ShimAdapter",
    "TypeDetector",
    "MetadataPreserver",
    # Registry and management
    "ConversionRegistry",
    "CompatibilityMatrix",
    # Base adapter implementations
    "BaseShimAdapter",
    "StreamingShimAdapter",
    "CachingShimAdapter",
    "PassThroughAdapter",
    "ValidationAdapter",
    # Type detection
    "TypeDetectionEngine",
    "FormatDetectionResult",
    "SchemaInfo",
    "detect_data_format",
    # Metadata management
    "MetadataManager",
    "PreservationRule",
    "MetadataSchema",
    "PreservationStrategy",
    "MetadataType",
    "MetadataLineage",
    "create_preservation_rule",
    "create_metadata_schema",
    # Core converters
    "PandasConverter",
    "NumpyConverter",
    "SparseMatrixConverter",
    # Conversion options and utilities
    "ConversionOptions",
    "ConversionQuality",
    # Factory functions
    "create_pandas_converter",
    "create_numpy_converter",
    "create_sparse_converter",
    "create_memory_efficient_options",
    "create_high_fidelity_options",
    "create_streaming_options",
    # Compatibility matrix
    "PipelineCompatibilityMatrix",
    "CompatibilityLevel",
    "DomainProfile",
    "create_compatibility_matrix",
    "create_minimal_compatibility_matrix",
    "assess_pipeline_compatibility",
    "find_optimal_format_for_domains",
    "suggest_pipeline_improvements",
    # Error handling
    "ConversionError",
    "ValidationResult",
    # Enhanced ShimAdapter and Registry
    "EnhancedShimAdapter",
    "ShimRegistry",
    "AdapterConfig",
    "AdapterMetrics",
    "HealthCheckResult",
    "AdapterLifecycleState",
    "create_shim_registry",
    "create_adapter_config",
    "validate_adapter_dependencies",
    "monitor_adapter_performance",
    # Schema validation system
    "SchemaInferenceEngine",
    "SchemaValidator",
    "SchemaEvolutionManager",
    "DataSchema",
    "SchemaConstraint",
    "ValidationError",
    "ValidationRule",
    "TypeValidationRule",
    "RangeValidationRule",
    "NullValidationRule",
    "SchemaInferenceResult",
    "SchemaValidationResult",
    "SchemaValidationLevel",
    "SchemaConformanceLevel",
    "ValidationRuleType",
    # Automatic shim insertion system
    "PipelineAnalyzer",
    "ShimInjector",
    "PipelineValidator",
    "PipelineStep",
    "PipelineConnection",
    "IncompatibilityIssue",
    "ShimRecommendation",
    "PipelineAnalysisResult",
    "OptimizationCriteria",
    "AnalysisType",
    "InjectionStrategy",
    "create_pipeline_analyzer",
    "create_shim_injector",
    "create_pipeline_validator",
    "create_optimization_criteria",
    "create_pipeline_step",
    "analyze_and_fix_pipeline",
    # Pre-built domain shims
    "BaseDomainShim",
    "StatisticalShim",
    "RegressionShim",
    "TimeSeriesShim",
    "PatternRecognitionShim",
    "DomainShimType",
    "DomainMapping",
    "SemanticContext",
    "create_statistical_shim",
    "create_regression_shim",
    "create_time_series_shim",
    "create_pattern_recognition_shim",
    "create_all_domain_shims",
    "get_compatible_domain_shims",
    "validate_domain_shim_configuration",
    # Performance optimization system
    "ConversionCache",
    "LazyLoadingManager",
    "LazyConverter",
    "CachedConversion",
    "CacheStatistics",
    "CacheEvictionPolicy",
    "LazyConversionState",
    # Error recovery system
    "ConversionErrorHandler",
    "AlternativePathwayEngine",
    "RollbackManager",
    "RecoveryStrategyEngine",
    "ErrorClassificationEnhanced",
    "RecoveryResult",
    "create_conversion_error_handler",
    "create_alternative_pathway_engine",
    "create_rollback_manager",
    "create_recovery_strategy_engine",
    "create_error_recovery_framework",
    "create_complete_error_recovery_system",
]

# Package metadata
__version__ = "2.0.0"
__author__ = "LocalData MCP Team"
__description__ = "Integration Shims and Data Type Conversion Framework"
