"""
Error recovery type definitions: enums, dataclasses, and data structures.
"""

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from ..interfaces import (
    ConversionCost,
    ConversionError,
    ConversionPath,
    ConversionRequest,
    DataFormat,
    ShimAdapter,
)
from ...base import ErrorClassification


class ErrorSeverity(Enum):
    """Error severity levels for classification."""

    CRITICAL = "critical"  # System-threatening errors requiring immediate attention
    HIGH = "high"  # Errors that prevent core functionality
    MEDIUM = "medium"  # Errors that impact performance or quality
    LOW = "low"  # Minor issues with workarounds available
    INFO = "info"  # Informational warnings


class ErrorRecoverability(Enum):
    """Error recoverability assessment."""

    RECOVERABLE = "recoverable"  # Error can be recovered with alternative approach
    DEGRADABLE = "degradable"  # Can continue with reduced quality/functionality
    RETRYABLE = "retryable"  # Error may resolve with retry
    TERMINAL = "terminal"  # Error cannot be recovered, must fail


class RecoveryStrategy(Enum):
    """Available recovery strategies."""

    RETRY = "retry"  # Retry operation with backoff
    FALLBACK = "fallback"  # Use alternative conversion pathway
    DEGRADE = "degrade"  # Accept lower quality conversion
    FAIL_FAST = "fail_fast"  # Immediate failure for critical errors
    USER_INTERVENTION = "user_intervention"  # Escalate to user decision
    ROLLBACK = "rollback"  # Rollback to previous stable state
    SKIP = "skip"  # Skip problematic operation and continue


class CircuitBreakerState(Enum):
    """Circuit breaker states for failure management."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, block requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class ErrorContext:
    """Comprehensive context information for error analysis and recovery."""

    error_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)

    # Error details
    error_type: Union[ConversionError.Type, ErrorClassification, str] = "unknown"
    severity: ErrorSeverity = ErrorSeverity.HIGH
    message: str = ""
    exception: Optional[Exception] = None

    # Pipeline context
    pipeline_id: Optional[str] = None
    pipeline_step: Optional[str] = None
    conversion_request: Optional[ConversionRequest] = None

    # Execution context
    execution_environment: Dict[str, Any] = field(default_factory=dict)
    system_state: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)

    # Recovery context
    attempted_strategies: List[RecoveryStrategy] = field(default_factory=list)
    partial_results: Optional[Any] = None
    recovery_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineCheckpoint:
    """State checkpoint for pipeline rollback capabilities."""

    checkpoint_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)

    # Pipeline state
    pipeline_id: str = ""
    step_index: int = 0
    step_id: str = ""

    # Data state
    data_snapshot: Optional[Any] = None
    data_hash: Optional[str] = None
    temporary_files: List[str] = field(default_factory=list)

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_context: Dict[str, Any] = field(default_factory=dict)

    # Storage information
    storage_path: Optional[str] = None
    serialization_method: str = "pickle"


@dataclass
class RecoveryPlan:
    """Plan for executing error recovery strategies."""

    error_context: ErrorContext
    plan_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Recovery strategies in priority order
    strategies: List[RecoveryStrategy] = field(default_factory=list)
    strategy_configs: Dict[RecoveryStrategy, Dict[str, Any]] = field(
        default_factory=dict
    )

    # Alternative options
    alternative_paths: List[ConversionPath] = field(default_factory=list)
    fallback_adapters: List[ShimAdapter] = field(default_factory=list)

    # Execution metadata
    estimated_recovery_time: float = 0.0
    estimated_success_probability: float = 0.0
    recovery_cost: Optional[ConversionCost] = None


@dataclass
class AlternativePathway:
    """Alternative pathway suggestion for failed conversions."""

    pathway_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Pathway description
    description: str = ""
    alternative_formats: List[DataFormat] = field(default_factory=list)
    conversion_path: Optional[ConversionPath] = None

    # Analysis
    feasibility_score: float = 0.0
    cost_benefit_ratio: float = 0.0
    expected_quality_loss: float = 0.0

    # Implementation
    required_changes: List[str] = field(default_factory=list)
    implementation_complexity: str = "low"  # low, medium, high
    estimated_effort: float = 0.0  # hours


# Enhanced Error Handling Data Structures


@dataclass
class ErrorClassificationEnhanced:
    """Comprehensive error classification with recovery context."""

    error_type: ConversionError.Type
    severity: ErrorSeverity
    recoverability: ErrorRecoverability
    confidence: float  # 0-1 confidence in classification
    suggested_strategies: List[RecoveryStrategy] = field(default_factory=list)
    context_factors: Dict[str, Any] = field(default_factory=dict)
    diagnostic_info: Dict[str, Any] = field(default_factory=dict)
    performance_impact: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class ErrorHandlingResult:
    """Result of error handling operation."""

    handled: bool
    recovery_attempted: bool
    recovery_successful: bool
    classification: ErrorClassificationEnhanced
    recovery_actions: List[str] = field(default_factory=list)
    alternative_suggestions: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    partial_results: Optional[Any] = None
    next_recommended_action: Optional[str] = None


@dataclass
class ErrorAggregation:
    """Aggregated error information for batch operations."""

    total_errors: int
    error_distribution: Dict[str, int] = field(default_factory=dict)
    severity_distribution: Dict[str, int] = field(default_factory=dict)
    recoverability_distribution: Dict[str, int] = field(default_factory=dict)
    most_common_errors: List[Tuple[str, int]] = field(default_factory=list)
    suggested_batch_strategies: List[RecoveryStrategy] = field(default_factory=list)
    aggregate_confidence: float = 0.0


@dataclass
class PerformanceImpact:
    """Assessment of error's impact on system performance."""

    execution_time_increase_percent: float = 0.0
    memory_overhead_mb: float = 0.0
    throughput_reduction_percent: float = 0.0
    resource_contention: Dict[str, float] = field(default_factory=dict)
    cascade_risk: float = 0.0  # Risk of error cascading to other operations
    recovery_cost_estimate: Optional[ConversionCost] = None


@dataclass
class PathwayCost:
    """Cost assessment for alternative conversion pathways."""

    computational_cost: float
    quality_degradation: float
    time_overhead: float
    memory_overhead: float
    reliability_score: float  # 0-1 score of pathway reliability
    confidence: float  # 0-1 confidence in cost estimates


@dataclass
class QualityAssessment:
    """Quality assessment for alternative pathways."""

    expected_quality_score: float  # 0-1 expected conversion quality
    quality_degradation: float  # Expected quality loss vs optimal path
    metadata_preservation: float  # 0-1 metadata preservation score
    data_fidelity: float  # 0-1 data fidelity score
    risk_factors: List[str] = field(default_factory=list)


@dataclass
class CheckpointId:
    """Unique identifier for rollback checkpoints."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    description: str = ""

    def __str__(self) -> str:
        return f"Checkpoint({self.id[:8]}@{self.description})"


@dataclass
class RollbackPath:
    """Path for rolling back from one state to another."""

    from_checkpoint: CheckpointId
    to_checkpoint: CheckpointId
    rollback_steps: List[Callable] = field(default_factory=list)
    estimated_cost: ConversionCost = None
    risk_assessment: Dict[str, float] = field(default_factory=dict)


@dataclass
class RetryPolicy:
    """Configuration for retry behavior."""

    max_attempts: int = 3
    base_delay_seconds: float = 1.0
    exponential_backoff: bool = True
    backoff_multiplier: float = 2.0
    max_delay_seconds: float = 60.0
    jitter: bool = True  # Add randomization to delays
    retry_on_types: Set[ConversionError.Type] = field(default_factory=set)
    abort_on_types: Set[ConversionError.Type] = field(default_factory=set)


@dataclass
class CircuitBreakerEnhanced:
    """Enhanced circuit breaker pattern for recurring failures."""

    failure_threshold: int = 5
    success_threshold: int = 3
    timeout_seconds: float = 60.0
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: float = 0.0
    state: str = "closed"  # closed, open, half-open


@dataclass
class RecoveryResult:
    """Result of recovery strategy execution."""

    strategy_used: RecoveryStrategy
    success: bool
    recovered_data: Optional[Any] = None
    execution_time: float = 0.0
    quality_score: float = 0.0
    warnings: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    next_strategy_recommendation: Optional[RecoveryStrategy] = None
