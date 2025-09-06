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

from .schema_validation import (
    # Schema validation core classes
    SchemaInferenceEngine,
    SchemaValidator,
    SchemaEvolutionManager,
    DataSchema,
    
    # Validation components
    SchemaConstraint,
    ValidationError,
    ValidationRule,
    TypeValidationRule,
    RangeValidationRule,
    NullValidationRule,
    
    # Results and enums
    SchemaInferenceResult,
    SchemaValidationResult,
    SchemaValidationLevel,
    SchemaConformanceLevel,
    ValidationRuleType,
)

from .pipeline_analyzer import (
    # Core pipeline analysis classes
    PipelineAnalyzer,
    ShimInjector,
    PipelineValidator,
    
    # Data structures
    PipelineStep,
    PipelineConnection,
    IncompatibilityIssue,
    ShimRecommendation,
    PipelineAnalysisResult,
    OptimizationCriteria,
    
    # Enums
    AnalysisType,
    InjectionStrategy,
    
    # Factory functions
    create_pipeline_analyzer,
    create_shim_injector,
    create_pipeline_validator,
    create_optimization_criteria,
    create_pipeline_step,
    
    # Utility functions
    analyze_and_fix_pipeline,
)

from .domain_shims import (
    # Domain-specific shim classes
    BaseDomainShim,
    StatisticalShim,
    RegressionShim,
    TimeSeriesShim,
    PatternRecognitionShim,
    
    # Domain configuration and mapping
    DomainShimType,
    DomainMapping,
    SemanticContext,
    
    # Factory functions
    create_statistical_shim,
    create_regression_shim,
    create_time_series_shim,
    create_pattern_recognition_shim,
    create_all_domain_shims,
    
    # Utility functions
    get_compatible_domain_shims,
    validate_domain_shim_configuration,
)

from .performance_optimization import (
    # Core performance optimization classes
    ConversionCache,
    LazyLoadingManager,
    LazyConverter,
    
    # Performance data structures
    CachedConversion,
    CacheStatistics,
    CacheEvictionPolicy,
    LazyConversionState,
    
    # Factory functions
    create_conversion_cache,
    create_lazy_loading_manager,
    create_optimized_shim_adapter,
    create_performance_config,
)

from .error_recovery import (
    # Core error recovery classes
    ConversionErrorHandler,
    AlternativePathwayEngine,
    RollbackManager,
    RecoveryStrategyEngine,
    
    # Error recovery data structures
    ErrorClassificationEnhanced,
    PathwayDiscoveryResult,
    CheckpointManager,
    RecoveryResult,
    
    # Factory functions
    create_conversion_error_handler,
    create_alternative_pathway_engine,
    create_rollback_manager,
    create_recovery_strategy_engine,
    create_error_recovery_framework,
    create_complete_error_recovery_system,
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
    
    # Schema validation system
    'SchemaInferenceEngine',
    'SchemaValidator',
    'SchemaEvolutionManager',
    'DataSchema',
    'SchemaConstraint',
    'ValidationError',
    'ValidationRule',
    'TypeValidationRule',
    'RangeValidationRule',
    'NullValidationRule',
    'SchemaInferenceResult',
    'SchemaValidationResult',
    'SchemaValidationLevel',
    'SchemaConformanceLevel',
    'ValidationRuleType',
    
    # Automatic shim insertion system
    'PipelineAnalyzer',
    'ShimInjector',
    'PipelineValidator',
    'PipelineStep',
    'PipelineConnection',
    'IncompatibilityIssue',
    'ShimRecommendation',
    'PipelineAnalysisResult',
    'OptimizationCriteria',
    'AnalysisType',
    'InjectionStrategy',
    'create_pipeline_analyzer',
    'create_shim_injector',
    'create_pipeline_validator',
    'create_optimization_criteria',
    'create_pipeline_step',
    'analyze_and_fix_pipeline',
    
    # Pre-built domain shims
    'BaseDomainShim',
    'StatisticalShim',
    'RegressionShim', 
    'TimeSeriesShim',
    'PatternRecognitionShim',
    'DomainShimType',
    'DomainMapping',
    'SemanticContext',
    'create_statistical_shim',
    'create_regression_shim',
    'create_time_series_shim',
    'create_pattern_recognition_shim',
    'create_all_domain_shims',
    'get_compatible_domain_shims',
    'validate_domain_shim_configuration',
    
    # Performance optimization system
    'ConversionCache',
    'LazyLoadingManager',
    'LazyConverter',
    'CachedConversion',
    'CacheStatistics',
    'CacheEvictionPolicy',
    'LazyConversionState',
    'create_conversion_cache',
    'create_lazy_loading_manager',
    'create_optimized_shim_adapter',
    'create_performance_config',
    
    # Error recovery system
    'ConversionErrorHandler',
    'AlternativePathwayEngine',
    'RollbackManager',
    'RecoveryStrategyEngine',
    'ErrorClassificationEnhanced',
    'PathwayDiscoveryResult',
    'CheckpointManager',
    'RecoveryResult',
    'create_conversion_error_handler',
    'create_alternative_pathway_engine',
    'create_rollback_manager',
    'create_recovery_strategy_engine',
    'create_error_recovery_framework',
    'create_complete_error_recovery_system',
]

# Package metadata
__version__ = "2.0.0"
__author__ = "LocalData MCP Team"
__description__ = "Integration Shims and Data Type Conversion Framework"