"""
Error Handling and Recovery Framework for LocalData MCP v2.0 Integration Shims Framework.

This package provides comprehensive error handling, graceful degradation, and recovery
mechanisms for pipeline operations, ensuring robust data science workflows that can
adapt to and recover from various failure scenarios.

Key Components:
- ConversionErrorHandler: Graceful failure handling with retry strategies
- AlternativePathwayEngine: Intelligent alternative route suggestions
- RollbackManager: State management and transaction rollback
- Recovery Strategy Framework: Configurable recovery policies and patterns
"""

# Type definitions: enums and dataclasses
from ._types import (
    AlternativePathway,
    CheckpointId,
    CircuitBreakerEnhanced,
    CircuitBreakerState,
    ErrorAggregation,
    ErrorClassificationEnhanced,
    ErrorContext,
    ErrorHandlingResult,
    ErrorRecoverability,
    ErrorSeverity,
    PathwayCost,
    PerformanceImpact,
    PipelineCheckpoint,
    QualityAssessment,
    RecoveryPlan,
    RecoveryResult,
    RecoveryStrategy,
    RetryPolicy,
    RollbackPath,
)

# Circuit breaker
from ._circuit_breaker import CircuitBreaker

# Core classes
from ._error_handler import ConversionErrorHandler
from ._pathway_engine import AlternativePathwayEngine
from ._rollback_manager import RollbackManager
from ._recovery_engine import RecoveryStrategyEngine

# Factory and utility functions
from ._factories import (
    create_alternative_pathway_engine,
    create_complete_error_recovery_system,
    create_conversion_error_handler,
    create_error_recovery_framework,
    create_recovery_strategy_engine,
    create_rollback_manager,
    handle_pipeline_error_with_recovery,
)

# Alias for compatibility with test framework
RecoveryStrategyFramework = RecoveryStrategyEngine

__all__ = [
    # Enums
    "ErrorSeverity",
    "ErrorRecoverability",
    "RecoveryStrategy",
    "CircuitBreakerState",
    # Dataclasses
    "ErrorContext",
    "PipelineCheckpoint",
    "RecoveryPlan",
    "AlternativePathway",
    "ErrorClassificationEnhanced",
    "ErrorHandlingResult",
    "ErrorAggregation",
    "PerformanceImpact",
    "PathwayCost",
    "QualityAssessment",
    "CheckpointId",
    "RollbackPath",
    "RetryPolicy",
    "CircuitBreakerEnhanced",
    "RecoveryResult",
    # Classes
    "CircuitBreaker",
    "ConversionErrorHandler",
    "AlternativePathwayEngine",
    "RollbackManager",
    "RecoveryStrategyEngine",
    "RecoveryStrategyFramework",
    # Factory functions
    "create_conversion_error_handler",
    "create_alternative_pathway_engine",
    "create_rollback_manager",
    "create_recovery_strategy_engine",
    "create_complete_error_recovery_system",
    "create_error_recovery_framework",
    # Utility functions
    "handle_pipeline_error_with_recovery",
]
