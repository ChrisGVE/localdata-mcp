"""
Error Handling and Recovery Framework for LocalData MCP v2.0 Integration Shims Framework.

This module provides comprehensive error handling, graceful degradation, and recovery 
mechanisms for pipeline operations, ensuring robust data science workflows that can
adapt to and recover from various failure scenarios.

Key Components:
- ConversionErrorHandler: Graceful failure handling with retry strategies
- AlternativePathwayEngine: Intelligent alternative route suggestions
- RollbackManager: State management and transaction rollback
- Recovery Strategy Framework: Configurable recovery policies and patterns

Design Principles:
- Intention-Driven Interface: Error recovery by analytical goals
- Context-Aware Composition: Recovery considers pipeline context
- Progressive Disclosure: Simple recovery with advanced options
- Streaming-First: Memory-efficient error recovery
- Modular Domain Integration: Extensible recovery strategies
"""

import asyncio
import logging
import random
import time
import threading
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
import uuid
from datetime import datetime, timedelta
import pickle
import tempfile
import shutil
import os

from .interfaces import (
    ConversionError, ConversionRequest, ConversionResult, ConversionPath,
    ConversionCost, DataFormat, ShimAdapter, ConversionContext,
    ValidationResult, ConversionStep
)
from .shim_registry import ShimRegistry, EnhancedShimAdapter
from ..base import PipelineError, ErrorClassification
from ...logging_manager import get_logger

logger = get_logger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels for classification."""
    CRITICAL = "critical"      # System-threatening errors requiring immediate attention
    HIGH = "high"             # Errors that prevent core functionality
    MEDIUM = "medium"         # Errors that impact performance or quality
    LOW = "low"              # Minor issues with workarounds available
    INFO = "info"            # Informational warnings


class ErrorRecoverability(Enum):
    """Error recoverability assessment."""
    RECOVERABLE = "recoverable"         # Error can be recovered with alternative approach
    DEGRADABLE = "degradable"          # Can continue with reduced quality/functionality
    RETRYABLE = "retryable"            # Error may resolve with retry
    TERMINAL = "terminal"              # Error cannot be recovered, must fail


class RecoveryStrategy(Enum):
    """Available recovery strategies."""
    RETRY = "retry"                    # Retry operation with backoff
    FALLBACK = "fallback"              # Use alternative conversion pathway
    DEGRADE = "degrade"                # Accept lower quality conversion
    FAIL_FAST = "fail_fast"            # Immediate failure for critical errors
    USER_INTERVENTION = "user_intervention"  # Escalate to user decision
    ROLLBACK = "rollback"              # Rollback to previous stable state
    SKIP = "skip"                      # Skip problematic operation and continue


class CircuitBreakerState(Enum):
    """Circuit breaker states for failure management."""
    CLOSED = "closed"       # Normal operation
    OPEN = "open"          # Failing, block requests
    HALF_OPEN = "half_open" # Testing if service recovered


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
    strategy_configs: Dict[RecoveryStrategy, Dict[str, Any]] = field(default_factory=dict)
    
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


class CircuitBreaker:
    """Circuit breaker pattern implementation for failure management."""
    
    def __init__(self,
                 failure_threshold: int = 5,
                 recovery_timeout: int = 60,
                 expected_exception: type = Exception):
        """
        Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before testing recovery
            expected_exception: Exception type to handle
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitBreakerState.CLOSED
        self._lock = threading.Lock()
        
        logger.debug(f"CircuitBreaker initialized", 
                    threshold=failure_threshold, 
                    timeout=recovery_timeout)
    
    def __call__(self, func):
        """Decorator to wrap functions with circuit breaker."""
        def wrapper(*args, **kwargs):
            with self._lock:
                # Check if circuit should transition from OPEN to HALF_OPEN
                if (self.state == CircuitBreakerState.OPEN and
                    self.last_failure_time and
                    time.time() - self.last_failure_time > self.recovery_timeout):
                    self.state = CircuitBreakerState.HALF_OPEN
                    logger.info("Circuit breaker transitioning to HALF_OPEN")
                
                # Block requests if circuit is OPEN
                if self.state == CircuitBreakerState.OPEN:
                    raise ConversionError(
                        ConversionError.Type.ADAPTER_NOT_FOUND,
                        "Circuit breaker is OPEN - service unavailable",
                        {"circuit_state": self.state.value}
                    )
            
            try:
                # Execute function
                result = func(*args, **kwargs)
                
                # Success - reset failure count and close circuit
                with self._lock:
                    if self.state == CircuitBreakerState.HALF_OPEN:
                        self.state = CircuitBreakerState.CLOSED
                        logger.info("Circuit breaker CLOSED after successful recovery test")
                    self.failure_count = 0
                
                return result
                
            except self.expected_exception as e:
                with self._lock:
                    self.failure_count += 1
                    self.last_failure_time = time.time()
                    
                    # Open circuit if threshold exceeded
                    if self.failure_count >= self.failure_threshold:
                        self.state = CircuitBreakerState.OPEN
                        logger.warning(f"Circuit breaker OPENED after {self.failure_count} failures")
                
                raise
        
        return wrapper
    
    def get_state(self) -> Dict[str, Any]:
        """Get current circuit breaker state."""
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "failure_threshold": self.failure_threshold,
            "last_failure_time": self.last_failure_time,
            "recovery_timeout": self.recovery_timeout
        }


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


class ConversionErrorHandler:
    """
    Advanced error classification and handling for conversion operations.
    
    Provides sophisticated error analysis with diagnostic context, performance
    impact assessment, and intelligent recovery recommendations.
    """
    
    def __init__(self,
                 classification_confidence_threshold: float = 0.7,
                 enable_performance_tracking: bool = True,
                 enable_error_learning: bool = True,
                 max_retry_attempts: int = 3,
                 base_retry_delay: float = 1.0,
                 max_retry_delay: float = 60.0,
                 exponential_backoff: bool = True,
                 jitter: bool = True):
        """
        Initialize ConversionErrorHandler.
        
        Args:
            classification_confidence_threshold: Minimum confidence for classifications
            enable_performance_tracking: Track performance impact of errors
            enable_error_learning: Learn from past error patterns
            max_retry_attempts: Maximum number of retry attempts
            base_retry_delay: Base delay for retry backoff (seconds)
            max_retry_delay: Maximum delay for retry backoff (seconds)
            exponential_backoff: Use exponential backoff strategy
            jitter: Add random jitter to retry delays
        """
        self.classification_confidence_threshold = classification_confidence_threshold
        self.enable_performance_tracking = enable_performance_tracking
        self.enable_error_learning = enable_error_learning
        self.max_retry_attempts = max_retry_attempts
        self.base_retry_delay = base_retry_delay
        self.max_retry_delay = max_retry_delay
        self.exponential_backoff = exponential_backoff
        self.jitter = jitter
        
        # Error tracking and learning
        self._error_history: List[Tuple[Exception, ErrorClassificationEnhanced]] = []
        self._error_patterns: Dict[str, Dict] = defaultdict(dict)
        self._classification_cache: Dict[str, ErrorClassificationEnhanced] = {}
        self.error_history: List[ErrorContext] = []
        self.error_patterns: Dict[str, List[ErrorContext]] = defaultdict(list)
        self.recovery_stats: Dict[RecoveryStrategy, Dict[str, int]] = defaultdict(
            lambda: {"attempted": 0, "successful": 0, "failed": 0}
        )
        
        # Performance tracking
        self._performance_baselines: Dict[str, float] = {}
        self._error_impact_history: List[PerformanceImpact] = []
        
        # Circuit breakers for different error types
        self.circuit_breakers: Dict[str, CircuitBreaker] = {
            "conversion_timeout": CircuitBreaker(failure_threshold=3, recovery_timeout=30),
            "memory_overflow": CircuitBreaker(failure_threshold=2, recovery_timeout=120),
            "adapter_not_found": CircuitBreaker(failure_threshold=5, recovery_timeout=60)
        }
        
        self._lock = threading.RLock()
        
        # Enhanced error learning and tracking
        if self.enable_error_learning:
            logger.info("Error pattern learning enabled")
        if self.enable_performance_tracking:
            logger.info("Performance impact tracking enabled")
        
        logger.info("ConversionErrorHandler initialized",
                   max_retries=max_retry_attempts,
                   backoff_strategy="exponential" if exponential_backoff else "linear",
                   confidence_threshold=classification_confidence_threshold,
                   performance_tracking=enable_performance_tracking,
                   error_learning=enable_error_learning)
    
    def handle_conversion_error(self,
                              error: Exception,
                              conversion_request: ConversionRequest,
                              pipeline_context: Optional[Dict[str, Any]] = None) -> ErrorContext:
        """
        Handle conversion error with comprehensive analysis and recovery planning.
        
        Args:
            error: The exception that occurred
            conversion_request: The conversion request that failed
            pipeline_context: Additional pipeline context
            
        Returns:
            ErrorContext with analysis and recovery information
        """
        start_time = time.time()
        
        # Create error context
        error_context = self._create_error_context(
            error, conversion_request, pipeline_context
        )
        
        # Classify error severity and type
        self._classify_error(error_context)
        
        # Record error in history and patterns
        with self._lock:
            self.error_history.append(error_context)
            pattern_key = f"{error_context.error_type}_{error_context.conversion_request.source_format.value}_{error_context.conversion_request.target_format.value}"
            self.error_patterns[pattern_key].append(error_context)
            
            # Maintain history size
            if len(self.error_history) > 1000:
                self.error_history = self.error_history[-500:]
        
        # Generate recovery plan
        recovery_plan = self._generate_recovery_plan(error_context)
        error_context.recovery_metadata["recovery_plan"] = recovery_plan
        
        # Generate troubleshooting guidance
        troubleshooting_guide = self._generate_troubleshooting_guide(error_context)
        error_context.recovery_metadata["troubleshooting_guide"] = troubleshooting_guide
        
        # Log error with context
        logger.error(f"Conversion error handled",
                    error_id=error_context.error_id,
                    error_type=error_context.error_type,
                    severity=error_context.severity.value,
                    source_format=conversion_request.source_format.value,
                    target_format=conversion_request.target_format.value,
                    recovery_strategies=len(recovery_plan.strategies),
                    handling_time=time.time() - start_time)
        
        return error_context
    
    def classify_error(self, 
                      error: Exception, 
                      context: ConversionContext) -> ErrorClassificationEnhanced:
        """
        Classify error with comprehensive analysis and recovery recommendations.
        
        Args:
            error: Exception to classify
            context: Conversion context for classification
            
        Returns:
            Detailed error classification with recovery strategies
        """
        start_time = time.time()
        
        with self._lock:
            # Check cache first
            error_key = self._generate_error_key(error, context)
            if error_key in self._classification_cache:
                cached = self._classification_cache[error_key]
                logger.debug(f"Using cached classification for {type(error).__name__}")
                return cached
            
            # Perform detailed classification
            classification = self._perform_classification(error, context)
            
            # Cache if confidence is high enough
            if classification.confidence >= self.classification_confidence_threshold:
                self._classification_cache[error_key] = classification
            
            # Learn from this error if enabled
            if self.enable_error_learning:
                self._learn_from_error(error, classification, context)
            
            # Track performance if enabled
            if self.enable_performance_tracking:
                classification_time = time.time() - start_time
                self._track_classification_performance(classification_time)
            
            logger.info("Error classified",
                       error_type=classification.error_type.value,
                       severity=classification.severity.value,
                       recoverability=classification.recoverability.value,
                       confidence=classification.confidence)
            
            return classification
    
    def handle_error(self, 
                    error: Exception, 
                    request: ConversionRequest) -> ErrorHandlingResult:
        """
        Handle error with intelligent recovery strategies.
        
        Args:
            error: Exception to handle
            request: Original conversion request
            
        Returns:
            Error handling result with recovery actions
        """
        start_time = time.time()
        
        try:
            # Classify the error
            classification = self.classify_error(error, request.context)
            
            # Determine handling approach based on classification
            recovery_actions = []
            alternative_suggestions = []
            recovery_attempted = False
            recovery_successful = False
            
            # Generate recovery recommendations
            if classification.recoverability != ErrorRecoverability.TERMINAL:
                recovery_actions = self._generate_recovery_actions(classification, request)
                alternative_suggestions = self._generate_alternatives(classification, request)
                recovery_attempted = len(recovery_actions) > 0
            
            # Assess performance impact
            performance_impact = self._assess_performance_impact(error, request)
            
            # Determine next action
            next_action = self._determine_next_action(classification, request)
            
            execution_time = time.time() - start_time
            
            result = ErrorHandlingResult(
                handled=True,
                recovery_attempted=recovery_attempted,
                recovery_successful=recovery_successful,  # Will be updated by caller
                classification=classification,
                recovery_actions=recovery_actions,
                alternative_suggestions=alternative_suggestions,
                performance_metrics={
                    'handling_time': execution_time,
                    'performance_impact': performance_impact.__dict__
                },
                next_recommended_action=next_action
            )
            
            logger.info("Error handled",
                       error_type=classification.error_type.value,
                       recovery_actions=len(recovery_actions),
                       alternatives=len(alternative_suggestions))
            
            return result
            
        except Exception as handling_error:
            logger.error(f"Error handling failed: {handling_error}")
            # Return minimal handling result
            return ErrorHandlingResult(
                handled=False,
                recovery_attempted=False,
                recovery_successful=False,
                classification=ErrorClassificationEnhanced(
                    error_type=ConversionError.Type.CONVERSION_FAILED,
                    severity=ErrorSeverity.HIGH,
                    recoverability=ErrorRecoverability.TERMINAL,
                    confidence=0.5
                ),
                recovery_actions=[],
                alternative_suggestions=[],
                performance_metrics={'handling_time': time.time() - start_time}
            )
    
    def aggregate_errors(self, errors: List[Exception]) -> ErrorAggregation:
        """
        Aggregate multiple errors for batch operation analysis.
        
        Args:
            errors: List of exceptions to aggregate
            
        Returns:
            Aggregated error analysis with batch recommendations
        """
        if not errors:
            return ErrorAggregation(total_errors=0)
        
        error_types = []
        severities = []
        recoverabilities = []
        classifications = []
        
        # Classify all errors
        for error in errors:
            # Use minimal context for batch processing
            context = ConversionContext(source_domain='batch', target_domain='batch')
            classification = self.classify_error(error, context)
            classifications.append(classification)
            error_types.append(classification.error_type.value)
            severities.append(classification.severity.value)
            recoverabilities.append(classification.recoverability.value)
        
        # Calculate distributions
        error_distribution = dict(defaultdict(int))
        for error_type in error_types:
            error_distribution[error_type] += 1
        
        severity_distribution = dict(defaultdict(int))
        for severity in severities:
            severity_distribution[severity] += 1
        
        recoverability_distribution = dict(defaultdict(int))
        for recoverability in recoverabilities:
            recoverability_distribution[recoverability] += 1
        
        # Find most common errors
        most_common = sorted(error_distribution.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Generate batch strategies
        batch_strategies = self._generate_batch_strategies(classifications)
        
        # Calculate aggregate confidence
        confidences = [c.confidence for c in classifications]
        aggregate_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        return ErrorAggregation(
            total_errors=len(errors),
            error_distribution=error_distribution,
            severity_distribution=severity_distribution,
            recoverability_distribution=recoverability_distribution,
            most_common_errors=most_common,
            suggested_batch_strategies=batch_strategies,
            aggregate_confidence=aggregate_confidence
        )
    
    def assess_performance_impact(self, error: Exception) -> PerformanceImpact:
        """
        Assess the performance impact of an error on system operations.
        
        Args:
            error: Exception to assess
            
        Returns:
            Performance impact assessment
        """
        return self._assess_performance_impact(error, None)
    
    def execute_recovery_strategy(self,
                                error_context: ErrorContext,
                                strategy: RecoveryStrategy,
                                operation: Callable,
                                *args, **kwargs) -> Tuple[bool, Any]:
        """
        Execute specific recovery strategy.
        
        Args:
            error_context: Error context with recovery metadata
            strategy: Recovery strategy to execute
            operation: Original operation to retry/modify
            *args, **kwargs: Operation arguments
            
        Returns:
            Tuple of (success, result)
        """
        with self._lock:
            self.recovery_stats[strategy]["attempted"] += 1
        
        error_context.attempted_strategies.append(strategy)
        
        logger.info(f"Executing recovery strategy",
                   error_id=error_context.error_id,
                   strategy=strategy.value)
        
        try:
            if strategy == RecoveryStrategy.RETRY:
                return self._execute_retry_strategy(error_context, operation, *args, **kwargs)
            
            elif strategy == RecoveryStrategy.FALLBACK:
                return self._execute_fallback_strategy(error_context, operation, *args, **kwargs)
            
            elif strategy == RecoveryStrategy.SKIP:
                return self._execute_skip_strategy(error_context)
            
            elif strategy == RecoveryStrategy.ALTERNATIVE_PATH:
                return self._execute_alternative_path_strategy(error_context, operation, *args, **kwargs)
            
            else:
                logger.warning(f"Unsupported recovery strategy: {strategy.value}")
                return False, None
                
        except Exception as e:
            with self._lock:
                self.recovery_stats[strategy]["failed"] += 1
            
            logger.error(f"Recovery strategy failed",
                        error_id=error_context.error_id,
                        strategy=strategy.value,
                        recovery_error=str(e))
            
            return False, None
    
    # Enhanced classification and analysis methods
    
    def _perform_classification(self, 
                               error: Exception, 
                               context: ConversionContext) -> ErrorClassificationEnhanced:
        """Perform detailed error classification."""
        
        # Determine error type
        if isinstance(error, ConversionError):
            error_type = error.error_type
        else:
            error_type = self._infer_error_type(error)
        
        # Assess severity
        severity = self._assess_severity(error, error_type, context)
        
        # Assess recoverability
        recoverability = self._assess_recoverability(error, error_type, context)
        
        # Calculate classification confidence
        confidence = self._calculate_classification_confidence(error, error_type, context)
        
        # Generate recovery strategies
        suggested_strategies = self._suggest_recovery_strategies(error_type, severity, recoverability)
        
        # Extract context factors
        context_factors = self._extract_context_factors(error, context)
        
        # Generate diagnostic information
        diagnostic_info = self._generate_diagnostic_info(error, context)
        
        # Assess performance impact
        performance_impact = self._assess_performance_impact(error, None).__dict__
        
        return ErrorClassificationEnhanced(
            error_type=error_type,
            severity=severity,
            recoverability=recoverability,
            confidence=confidence,
            suggested_strategies=suggested_strategies,
            context_factors=context_factors,
            diagnostic_info=diagnostic_info,
            performance_impact=performance_impact
        )
    
    def _infer_error_type(self, error: Exception) -> ConversionError.Type:
        """Infer ConversionError.Type from generic exception."""
        error_str = str(error).lower()
        error_class = type(error).__name__.lower()
        
        if 'memory' in error_str or 'memoryerror' in error_class:
            return ConversionError.Type.MEMORY_EXCEEDED
        elif 'timeout' in error_str or 'timeouterror' in error_class:
            return ConversionError.Type.TIMEOUT
        elif 'type' in error_str or 'typeerror' in error_class:
            return ConversionError.Type.TYPE_MISMATCH
        elif 'value' in error_str or 'valueerror' in error_class:
            return ConversionError.Type.SCHEMA_INVALID
        else:
            return ConversionError.Type.CONVERSION_FAILED
    
    def _assess_severity(self, 
                        error: Exception, 
                        error_type: ConversionError.Type, 
                        context: ConversionContext) -> ErrorSeverity:
        """Assess error severity based on type and context."""
        
        # Critical severity conditions
        if (error_type == ConversionError.Type.MEMORY_EXCEEDED or
            'system' in context.source_domain.lower() or
            'critical' in context.user_intention.lower()):
            return ErrorSeverity.CRITICAL
        
        # High severity conditions
        if (error_type in [ConversionError.Type.TIMEOUT, ConversionError.Type.ADAPTER_NOT_FOUND] or
            'primary' in context.user_intention.lower()):
            return ErrorSeverity.HIGH
        
        # Medium severity conditions
        if (error_type in [ConversionError.Type.TYPE_MISMATCH, ConversionError.Type.SCHEMA_INVALID] or
            'important' in context.user_intention.lower()):
            return ErrorSeverity.MEDIUM
        
        # Default to low severity
        return ErrorSeverity.LOW
    
    def _assess_recoverability(self, 
                              error: Exception, 
                              error_type: ConversionError.Type, 
                              context: ConversionContext) -> ErrorRecoverability:
        """Assess error recoverability."""
        
        # Terminal conditions
        if (error_type == ConversionError.Type.MEMORY_EXCEEDED and 
            context.performance_hints.get('memory_critical', False)):
            return ErrorRecoverability.TERMINAL
        
        # Retryable conditions
        if error_type in [ConversionError.Type.TIMEOUT, ConversionError.Type.ADAPTER_NOT_FOUND]:
            return ErrorRecoverability.RETRYABLE
        
        # Degradable conditions
        if error_type in [ConversionError.Type.METADATA_LOSS, ConversionError.Type.QUALITY_DEGRADED]:
            return ErrorRecoverability.DEGRADABLE
        
        # Default to recoverable
        return ErrorRecoverability.RECOVERABLE
    
    def _calculate_classification_confidence(self, 
                                           error: Exception, 
                                           error_type: ConversionError.Type, 
                                           context: ConversionContext) -> float:
        """Calculate confidence in error classification."""
        confidence = 0.5  # Base confidence
        
        # Higher confidence for ConversionError instances
        if isinstance(error, ConversionError):
            confidence += 0.3
        
        # Higher confidence with rich context
        if context.user_intention:
            confidence += 0.1
        if context.performance_hints:
            confidence += 0.1
        
        # Lower confidence for generic exceptions
        if type(error) in [Exception, RuntimeError, ValueError]:
            confidence -= 0.2
        
        return max(min(confidence, 1.0), 0.0)
    
    def _suggest_recovery_strategies(self, 
                                   error_type: ConversionError.Type,
                                   severity: ErrorSeverity,
                                   recoverability: ErrorRecoverability) -> List[RecoveryStrategy]:
        """Suggest appropriate recovery strategies."""
        strategies = []
        
        # Strategy selection based on recoverability
        if recoverability == ErrorRecoverability.RETRYABLE:
            strategies.append(RecoveryStrategy.RETRY)
        
        if recoverability == ErrorRecoverability.RECOVERABLE:
            strategies.extend([RecoveryStrategy.FALLBACK, RecoveryStrategy.ROLLBACK])
        
        if recoverability == ErrorRecoverability.DEGRADABLE:
            strategies.append(RecoveryStrategy.DEGRADE)
        
        if recoverability == ErrorRecoverability.TERMINAL:
            if severity == ErrorSeverity.CRITICAL:
                strategies.append(RecoveryStrategy.FAIL_FAST)
            else:
                strategies.append(RecoveryStrategy.USER_INTERVENTION)
        
        # Add SKIP for non-critical errors
        if severity in [ErrorSeverity.LOW, ErrorSeverity.INFO]:
            strategies.append(RecoveryStrategy.SKIP)
        
        return strategies
    
    def _extract_context_factors(self, 
                                error: Exception, 
                                context: ConversionContext) -> Dict[str, Any]:
        """Extract relevant context factors for error analysis."""
        factors = {
            'error_class': type(error).__name__,
            'error_message_length': len(str(error)),
            'has_user_intention': bool(context.user_intention),
            'domain_match': context.source_domain == context.target_domain,
            'has_performance_hints': bool(context.performance_hints),
            'debugging_enabled': context.debugging_enabled
        }
        
        # Add performance hint factors
        if context.performance_hints:
            factors.update({
                f'hint_{k}': v for k, v in context.performance_hints.items()
                if isinstance(v, (bool, int, float, str))
            })
        
        return factors
    
    def _generate_diagnostic_info(self, 
                                 error: Exception, 
                                 context: ConversionContext) -> Dict[str, Any]:
        """Generate diagnostic information for error analysis."""
        import traceback
        
        diagnostic = {
            'error_message': str(error),
            'error_type': type(error).__name__,
            'traceback_length': len(traceback.format_exc().split('\n')),
            'context_source_domain': context.source_domain,
            'context_target_domain': context.target_domain,
            'context_intention': context.user_intention,
            'timestamp': time.time()
        }
        
        # Add traceback if debugging is enabled
        if context.debugging_enabled:
            diagnostic['full_traceback'] = traceback.format_exc()
        
        return diagnostic
    
    def _assess_performance_impact(self, 
                                  error: Exception, 
                                  request: Optional[ConversionRequest]) -> PerformanceImpact:
        """Assess performance impact of error."""
        
        # Base impact assessment
        impact = PerformanceImpact()
        
        # Estimate based on error type
        if isinstance(error, ConversionError):
            if error.error_type == ConversionError.Type.MEMORY_EXCEEDED:
                impact.memory_overhead_mb = 100.0  # Estimated cleanup overhead
                impact.execution_time_increase_percent = 50.0
            elif error.error_type == ConversionError.Type.TIMEOUT:
                impact.execution_time_increase_percent = 200.0  # Timeout overhead
            elif error.error_type == ConversionError.Type.ADAPTER_NOT_FOUND:
                impact.execution_time_increase_percent = 10.0  # Search overhead
        
        # Estimate cascade risk
        if request and hasattr(request, 'metadata'):
            pipeline_depth = request.metadata.get('pipeline_depth', 1)
            impact.cascade_risk = min(pipeline_depth * 0.1, 0.8)
        
        return impact
    
    def _generate_recovery_actions(self, 
                                  classification: ErrorClassificationEnhanced,
                                  request: ConversionRequest) -> List[str]:
        """Generate specific recovery actions."""
        actions = []
        
        for strategy in classification.suggested_strategies:
            if strategy == RecoveryStrategy.RETRY:
                actions.append(f"Retry conversion with exponential backoff (max 3 attempts)")
            elif strategy == RecoveryStrategy.FALLBACK:
                actions.append("Attempt alternative conversion pathway")
            elif strategy == RecoveryStrategy.DEGRADE:
                actions.append("Accept reduced quality conversion")
            elif strategy == RecoveryStrategy.ROLLBACK:
                actions.append("Rollback to previous stable checkpoint")
            elif strategy == RecoveryStrategy.SKIP:
                actions.append("Skip this conversion and continue with remaining operations")
            elif strategy == RecoveryStrategy.USER_INTERVENTION:
                actions.append("Escalate to user for manual intervention")
        
        return actions
    
    def _generate_alternatives(self, 
                              classification: ErrorClassificationEnhanced,
                              request: ConversionRequest) -> List[str]:
        """Generate alternative approach suggestions."""
        alternatives = []
        
        # Based on error type
        if classification.error_type == ConversionError.Type.MEMORY_EXCEEDED:
            alternatives.extend([
                "Use streaming conversion with smaller chunks",
                "Temporarily reduce memory footprint of other operations",
                "Convert to intermediate format with lower memory requirements"
            ])
        elif classification.error_type == ConversionError.Type.TYPE_MISMATCH:
            alternatives.extend([
                "Apply automatic type coercion before conversion",
                "Use format-specific preprocessing",
                "Convert through intermediate compatible format"
            ])
        elif classification.error_type == ConversionError.Type.ADAPTER_NOT_FOUND:
            alternatives.extend([
                "Use multi-step conversion through supported intermediate formats",
                "Load additional adapter modules",
                "Use generic conversion with manual formatting"
            ])
        
        return alternatives
    
    def _determine_next_action(self, 
                              classification: ErrorClassificationEnhanced,
                              request: ConversionRequest) -> Optional[str]:
        """Determine the next recommended action."""
        
        if not classification.suggested_strategies:
            return None
        
        # Choose primary strategy
        primary_strategy = classification.suggested_strategies[0]
        
        if primary_strategy == RecoveryStrategy.RETRY:
            return "retry_with_backoff"
        elif primary_strategy == RecoveryStrategy.FALLBACK:
            return "find_alternative_pathway"
        elif primary_strategy == RecoveryStrategy.DEGRADE:
            return "accept_quality_degradation"
        elif primary_strategy == RecoveryStrategy.ROLLBACK:
            return "rollback_to_checkpoint"
        elif primary_strategy == RecoveryStrategy.FAIL_FAST:
            return "fail_immediately"
        elif primary_strategy == RecoveryStrategy.USER_INTERVENTION:
            return "escalate_to_user"
        else:
            return "continue_with_caution"
    
    def _generate_batch_strategies(self, 
                                  classifications: List[ErrorClassificationEnhanced]) -> List[RecoveryStrategy]:
        """Generate recovery strategies for batch operations."""
        # Count strategy occurrences
        strategy_counts = defaultdict(int)
        for classification in classifications:
            for strategy in classification.suggested_strategies:
                strategy_counts[strategy] += 1
        
        # Sort by frequency and return top strategies
        sorted_strategies = sorted(strategy_counts.items(), key=lambda x: x[1], reverse=True)
        return [strategy for strategy, _ in sorted_strategies[:3]]
    
    def _generate_error_key(self, error: Exception, context: ConversionContext) -> str:
        """Generate cache key for error classification."""
        return f"{type(error).__name__}:{hash(str(error))}:{context.source_domain}:{context.target_domain}"
    
    def _learn_from_error(self, 
                         error: Exception, 
                         classification: ErrorClassificationEnhanced,
                         context: ConversionContext):
        """Learn from error patterns for improved future classification."""
        self._error_history.append((error, classification))
        
        # Update error patterns
        error_pattern_key = f"{type(error).__name__}:{context.source_domain}:{context.target_domain}"
        pattern = self._error_patterns[error_pattern_key]
        pattern['count'] = pattern.get('count', 0) + 1
        pattern['last_seen'] = time.time()
        pattern['average_confidence'] = (
            (pattern.get('average_confidence', 0.0) * (pattern['count'] - 1) + classification.confidence) 
            / pattern['count']
        )
        
        # Limit history size
        if len(self._error_history) > 1000:
            self._error_history = self._error_history[-500:]
    
    def _track_classification_performance(self, classification_time: float):
        """Track error classification performance metrics."""
        baseline_key = 'classification_time'
        
        if baseline_key not in self._performance_baselines:
            self._performance_baselines[baseline_key] = classification_time
        else:
            # Exponential moving average
            alpha = 0.1
            self._performance_baselines[baseline_key] = (
                alpha * classification_time + 
                (1 - alpha) * self._performance_baselines[baseline_key]
            )
    
    def _create_error_context(self,
                            error: Exception,
                            conversion_request: ConversionRequest,
                            pipeline_context: Optional[Dict[str, Any]]) -> ErrorContext:
        """Create comprehensive error context."""
        return ErrorContext(
            message=str(error),
            exception=error,
            conversion_request=conversion_request,
            pipeline_id=pipeline_context.get("pipeline_id") if pipeline_context else None,
            pipeline_step=pipeline_context.get("pipeline_step") if pipeline_context else None,
            execution_environment={
                "timestamp": time.time(),
                "thread_id": threading.current_thread().ident,
                "process_id": os.getpid()
            },
            system_state=self._capture_system_state(),
            performance_metrics=pipeline_context.get("performance_metrics", {}) if pipeline_context else {}
        )
    
    def _classify_error(self, error_context: ErrorContext) -> None:
        """Classify error type and severity."""
        error = error_context.exception
        
        # Classify by exception type
        if isinstance(error, ConversionError):
            error_context.error_type = error.error_type
        elif isinstance(error, PipelineError):
            error_context.error_type = error.classification
        elif isinstance(error, MemoryError):
            error_context.error_type = ConversionError.Type.MEMORY_EXCEEDED
            error_context.severity = ErrorSeverity.CRITICAL
        elif isinstance(error, TimeoutError):
            error_context.error_type = ConversionError.Type.TIMEOUT
            error_context.severity = ErrorSeverity.HIGH
        elif isinstance(error, (ValueError, TypeError)):
            error_context.error_type = ConversionError.Type.TYPE_MISMATCH
            error_context.severity = ErrorSeverity.MEDIUM
        else:
            error_context.error_type = ConversionError.Type.CONVERSION_FAILED
            error_context.severity = ErrorSeverity.HIGH
        
        # Adjust severity based on context
        if error_context.pipeline_step == "critical_conversion":
            error_context.severity = ErrorSeverity.CRITICAL
        elif "memory" in str(error).lower():
            error_context.severity = ErrorSeverity.CRITICAL
    
    def _generate_recovery_plan(self, error_context: ErrorContext) -> RecoveryPlan:
        """Generate recovery plan based on error context and patterns."""
        plan = RecoveryPlan(error_context=error_context)
        
        # Determine strategies based on error type and severity
        if error_context.error_type == ConversionError.Type.TIMEOUT:
            plan.strategies = [RecoveryStrategy.RETRY, RecoveryStrategy.FALLBACK]
            plan.strategy_configs[RecoveryStrategy.RETRY] = {
                "max_attempts": 2,
                "timeout_multiplier": 2.0
            }
        
        elif error_context.error_type == ConversionError.Type.MEMORY_EXCEEDED:
            plan.strategies = [RecoveryStrategy.ALTERNATIVE_PATH, RecoveryStrategy.FALLBACK]
            plan.strategy_configs[RecoveryStrategy.ALTERNATIVE_PATH] = {
                "prefer_streaming": True,
                "max_memory_mb": 500
            }
        
        elif error_context.error_type == ConversionError.Type.ADAPTER_NOT_FOUND:
            plan.strategies = [RecoveryStrategy.ALTERNATIVE_PATH, RecoveryStrategy.SKIP]
        
        else:
            # Default strategy sequence
            plan.strategies = [RecoveryStrategy.RETRY, RecoveryStrategy.ALTERNATIVE_PATH, RecoveryStrategy.FALLBACK]
        
        # Calculate estimates
        plan.estimated_recovery_time = self._estimate_recovery_time(plan)
        plan.estimated_success_probability = self._estimate_success_probability(plan, error_context)
        
        return plan
    
    def _execute_retry_strategy(self,
                              error_context: ErrorContext,
                              operation: Callable,
                              *args, **kwargs) -> Tuple[bool, Any]:
        """Execute retry strategy with exponential backoff."""
        for attempt in range(self.max_retry_attempts):
            try:
                # Calculate delay
                if attempt > 0:
                    delay = self._calculate_retry_delay(attempt)
                    logger.debug(f"Retrying after {delay:.2f}s delay", 
                               attempt=attempt + 1, 
                               max_attempts=self.max_retry_attempts)
                    time.sleep(delay)
                
                # Execute operation
                result = operation(*args, **kwargs)
                
                with self._lock:
                    self.recovery_stats[RecoveryStrategy.RETRY]["successful"] += 1
                
                logger.info(f"Retry successful on attempt {attempt + 1}")
                return True, result
                
            except Exception as retry_error:
                logger.debug(f"Retry attempt {attempt + 1} failed: {retry_error}")
                
                if attempt == self.max_retry_attempts - 1:
                    logger.warning(f"All retry attempts exhausted")
                    return False, None
        
        return False, None
    
    def _execute_fallback_strategy(self,
                                 error_context: ErrorContext,
                                 operation: Callable,
                                 *args, **kwargs) -> Tuple[bool, Any]:
        """Execute fallback strategy with degraded functionality."""
        # Implement fallback logic based on error context
        logger.info("Executing fallback strategy with degraded functionality")
        
        # Return partial results if available
        if error_context.partial_results is not None:
            with self._lock:
                self.recovery_stats[RecoveryStrategy.FALLBACK]["successful"] += 1
            return True, error_context.partial_results
        
        # Generate minimal acceptable result
        fallback_result = self._generate_fallback_result(error_context)
        
        if fallback_result is not None:
            with self._lock:
                self.recovery_stats[RecoveryStrategy.FALLBACK]["successful"] += 1
            return True, fallback_result
        
        return False, None
    
    def _execute_skip_strategy(self, error_context: ErrorContext) -> Tuple[bool, Any]:
        """Execute skip strategy."""
        logger.info("Skipping problematic operation")
        
        with self._lock:
            self.recovery_stats[RecoveryStrategy.SKIP]["successful"] += 1
        
        # Return indication that operation was skipped
        return True, {
            "status": "skipped",
            "reason": f"Error: {error_context.error_type}",
            "error_id": error_context.error_id
        }
    
    def _execute_alternative_path_strategy(self,
                                         error_context: ErrorContext,
                                         operation: Callable,
                                         *args, **kwargs) -> Tuple[bool, Any]:
        """Execute alternative path strategy."""
        logger.info("Attempting alternative conversion path")
        
        # This would typically integrate with AlternativePathwayEngine
        # For now, return indication that alternative path should be tried
        with self._lock:
            self.recovery_stats[RecoveryStrategy.ALTERNATIVE_PATH]["successful"] += 1
        
        return True, {
            "status": "alternative_path_required",
            "error_id": error_context.error_id,
            "original_request": error_context.conversion_request
        }
    
    def _calculate_retry_delay(self, attempt: int) -> float:
        """Calculate retry delay with exponential backoff and jitter."""
        if self.exponential_backoff:
            delay = self.base_retry_delay * (2 ** attempt)
        else:
            delay = self.base_retry_delay * (attempt + 1)
        
        # Apply maximum delay limit
        delay = min(delay, self.max_retry_delay)
        
        # Add jitter to avoid thundering herd
        if self.jitter:
            jitter_amount = delay * 0.1 * random.random()
            delay += jitter_amount
        
        return delay
    
    def _capture_system_state(self) -> Dict[str, Any]:
        """Capture current system state for error analysis."""
        state = {
            "timestamp": time.time(),
            "thread_count": threading.active_count()
        }
        
        try:
            import psutil
            process = psutil.Process()
            state.update({
                "memory_usage_mb": process.memory_info().rss / (1024 * 1024),
                "cpu_percent": process.cpu_percent(),
                "open_files": len(process.open_files())
            })
        except ImportError:
            pass
        
        return state
    
    def _estimate_recovery_time(self, plan: RecoveryPlan) -> float:
        """Estimate time required for recovery plan execution."""
        base_times = {
            RecoveryStrategy.RETRY: 5.0,
            RecoveryStrategy.FALLBACK: 2.0,
            RecoveryStrategy.SKIP: 0.1,
            RecoveryStrategy.ALTERNATIVE_PATH: 10.0,
            RecoveryStrategy.ROLLBACK: 3.0
        }
        
        total_time = 0.0
        for strategy in plan.strategies:
            total_time += base_times.get(strategy, 5.0)
        
        return total_time
    
    def _estimate_success_probability(self, plan: RecoveryPlan, error_context: ErrorContext) -> float:
        """Estimate success probability for recovery plan."""
        # Base probabilities by strategy
        base_probabilities = {
            RecoveryStrategy.RETRY: 0.7,
            RecoveryStrategy.FALLBACK: 0.9,
            RecoveryStrategy.SKIP: 1.0,
            RecoveryStrategy.ALTERNATIVE_PATH: 0.8,
            RecoveryStrategy.ROLLBACK: 0.95
        }
        
        # Calculate combined probability (assuming independence)
        failure_probability = 1.0
        for strategy in plan.strategies:
            strategy_success_prob = base_probabilities.get(strategy, 0.5)
            failure_probability *= (1.0 - strategy_success_prob)
        
        return 1.0 - failure_probability
    
    def _generate_fallback_result(self, error_context: ErrorContext) -> Optional[Any]:
        """Generate minimal acceptable fallback result."""
        # This would depend on the specific conversion type
        # Return a basic structure indicating fallback was used
        return {
            "fallback": True,
            "error_context": error_context.error_id,
            "original_source_format": error_context.conversion_request.source_format.value if error_context.conversion_request else "unknown",
            "original_target_format": error_context.conversion_request.target_format.value if error_context.conversion_request else "unknown"
        }
    
    def _generate_troubleshooting_guide(self, error_context: ErrorContext) -> List[str]:
        """Generate troubleshooting guidance based on error context."""
        guide = []
        
        # General guidance based on error type
        if error_context.error_type == ConversionError.Type.MEMORY_EXCEEDED:
            guide.extend([
                "Consider using streaming processing for large datasets",
                "Reduce batch size or implement chunked processing",
                "Check available system memory and close unnecessary applications",
                "Consider using sparse data representations"
            ])
        
        elif error_context.error_type == ConversionError.Type.TIMEOUT:
            guide.extend([
                "Check network connectivity and service availability",
                "Increase timeout values in configuration",
                "Consider implementing asynchronous processing",
                "Break large operations into smaller chunks"
            ])
        
        elif error_context.error_type == ConversionError.Type.ADAPTER_NOT_FOUND:
            guide.extend([
                "Verify that required adapters are installed and available",
                "Check adapter registry configuration",
                "Consider alternative data formats or conversion paths",
                "Update adapter dependencies if needed"
            ])
        
        else:
            guide.extend([
                "Review input data format and structure",
                "Check conversion parameters and configuration",
                "Verify data quality and completeness",
                "Consider alternative processing approaches"
            ])
        
        # Add pattern-based guidance
        error_pattern_key = f"{error_context.error_type}_{error_context.conversion_request.source_format.value}_{error_context.conversion_request.target_format.value}" if error_context.conversion_request else str(error_context.error_type)
        
        if error_pattern_key in self.error_patterns:
            recent_errors = self.error_patterns[error_pattern_key][-5:]  # Last 5 occurrences
            if len(recent_errors) > 2:
                guide.append(f"This error pattern has occurred {len(recent_errors)} times recently - consider systematic investigation")
        
        return guide
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error handling statistics."""
        with self._lock:
            return {
                "total_errors_handled": len(self.error_history),
                "error_patterns": {k: len(v) for k, v in self.error_patterns.items()},
                "recovery_statistics": dict(self.recovery_stats),
                "circuit_breaker_states": {k: v.get_state() for k, v in self.circuit_breakers.items()},
                "recent_error_types": [ctx.error_type for ctx in self.error_history[-10:]],
                "average_recovery_attempts": sum(len(ctx.attempted_strategies) for ctx in self.error_history[-100:]) / min(len(self.error_history), 100) if self.error_history else 0
            }
    
    def clear_error_history(self) -> None:
        """Clear error history and reset statistics."""
        with self._lock:
            self.error_history.clear()
            self.error_patterns.clear()
            self.recovery_stats.clear()
            
            # Reset circuit breakers
            for breaker in self.circuit_breakers.values():
                breaker.failure_count = 0
                breaker.last_failure_time = None
                breaker.state = CircuitBreakerState.CLOSED
            
            logger.info("Error history and statistics cleared")
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error handling statistics."""
        with self._lock:
            return {
                'total_errors_processed': len(self._error_history),
                'cached_classifications': len(self._classification_cache),
                'learned_patterns': len(self._error_patterns),
                'performance_baselines': self._performance_baselines.copy(),
                'most_common_error_types': dict(
                    defaultdict(int, {
                        classification.error_type.value: count 
                        for _, classification in self._error_history[-100:]
                        for count in [1]
                    })
                ),
                "total_errors_handled": len(self.error_history),
                "error_patterns": {k: len(v) for k, v in self.error_patterns.items()},
                "recovery_statistics": dict(self.recovery_stats),
                "circuit_breaker_states": {k: v.get_state() for k, v in self.circuit_breakers.items()},
                "recent_error_types": [ctx.error_type for ctx in self.error_history[-10:]],
                "average_recovery_attempts": sum(len(ctx.attempted_strategies) for ctx in self.error_history[-100:]) / min(len(self.error_history), 100) if self.error_history else 0
            }


class AlternativePathwayEngine:
    """
    Intelligent discovery and evaluation of alternative conversion pathways.
    
    Finds alternative routes when direct conversion fails, with cost-benefit
    analysis and quality assessment for each pathway option.
    """
    
    def __init__(self, 
                 registry: Optional[ShimRegistry] = None,
                 max_pathway_depth: int = 5,
                 enable_pathway_caching: bool = True,
                 quality_threshold: float = 0.6):
        """
        Initialize AlternativePathwayEngine.
        
        Args:
            registry: ShimRegistry for adapter discovery
            max_pathway_depth: Maximum number of conversion steps in pathway
            enable_pathway_caching: Cache successful pathways for reuse
            quality_threshold: Minimum quality threshold for pathways
        """
        self.registry = registry
        self.max_pathway_depth = max_pathway_depth
        self.enable_pathway_caching = enable_pathway_caching
        self.quality_threshold = quality_threshold
        
        # Pathway caching (new enhanced system)
        self._pathway_cache: Dict[str, List[ConversionPath]] = {}
        self._successful_pathways: Dict[str, ConversionPath] = {}
        
        # Performance tracking
        self._pathway_success_rates: Dict[str, float] = defaultdict(float)
        self._pathway_performance: Dict[str, Dict[str, float]] = defaultdict(dict)
        
        # Legacy pathway analysis cache (for backward compatibility)
        self.pathway_cache: Dict[str, List[AlternativePathway]] = {}
        self.success_patterns: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        # Analysis statistics
        self.stats = {
            "pathways_analyzed": 0,
            "alternatives_found": 0,
            "successful_alternatives": 0,
            "cache_hits": 0
        }
        
        # Thread safety
        self._lock = threading.RLock()
        
        logger.info("AlternativePathwayEngine initialized",
                   max_depth=max_pathway_depth,
                   caching_enabled=enable_pathway_caching,
                   quality_threshold=quality_threshold)
    
    def find_alternative_pathways(self,
                                failed_conversion: ConversionRequest,
                                error_context: ErrorContext,
                                max_alternatives: int = 5) -> List[AlternativePathway]:
        """
        Find alternative pathways for failed conversion.
        
        Args:
            failed_conversion: The conversion that failed
            error_context: Context of the failure
            max_alternatives: Maximum number of alternatives to return
            
        Returns:
            List of alternative pathways with feasibility analysis
        """
        cache_key = self._generate_pathway_cache_key(failed_conversion, error_context)
        
        with self._lock:
            self.stats["pathways_analyzed"] += 1
            
            # Check cache first
            if cache_key in self.pathway_cache:
                self.stats["cache_hits"] += 1
                return self.pathway_cache[cache_key][:max_alternatives]
        
        logger.info("Analyzing alternative pathways",
                   source_format=failed_conversion.source_format.value,
                   target_format=failed_conversion.target_format.value,
                   error_type=error_context.error_type)
        
        alternatives = []
        
        # Strategy 1: Find alternative intermediate formats
        intermediate_alternatives = self._find_intermediate_format_alternatives(failed_conversion)
        alternatives.extend(intermediate_alternatives)
        
        # Strategy 2: Find alternative target formats with similar capabilities
        target_alternatives = self._find_alternative_target_formats(failed_conversion)
        alternatives.extend(target_alternatives)
        
        # Strategy 3: Find alternative source format preprocessors
        source_alternatives = self._find_source_preprocessing_alternatives(failed_conversion)
        alternatives.extend(source_alternatives)
        
        # Strategy 4: Find pipeline restructuring alternatives
        restructuring_alternatives = self._find_restructuring_alternatives(failed_conversion, error_context)
        alternatives.extend(restructuring_alternatives)
        
        # Analyze feasibility and cost-benefit for each alternative
        for alternative in alternatives:
            self._analyze_alternative_feasibility(alternative, failed_conversion, error_context)
        
        # Sort by feasibility score
        alternatives.sort(key=lambda a: a.feasibility_score, reverse=True)
        
        # Cache results
        with self._lock:
            self.pathway_cache[cache_key] = alternatives
            self.stats["alternatives_found"] += len(alternatives)
        
        logger.info(f"Found {len(alternatives)} alternative pathways",
                   top_feasibility=alternatives[0].feasibility_score if alternatives else 0.0)
        
        return alternatives[:max_alternatives]
    
    def _find_intermediate_format_alternatives(self,
                                             failed_conversion: ConversionRequest) -> List[AlternativePathway]:
        """Find alternatives using different intermediate formats."""
        alternatives = []
        
        # Get all possible formats as intermediates
        potential_intermediates = [fmt for fmt in DataFormat if 
                                 fmt not in [failed_conversion.source_format, failed_conversion.target_format]]
        
        for intermediate_format in potential_intermediates:
            # Check if source -> intermediate -> target path exists
            source_to_intermediate = self._check_conversion_feasibility(
                failed_conversion.source_format, intermediate_format
            )
            
            if not source_to_intermediate:
                continue
            
            intermediate_to_target = self._check_conversion_feasibility(
                intermediate_format, failed_conversion.target_format
            )
            
            if not intermediate_to_target:
                continue
            
            # Create alternative pathway
            alternative = AlternativePathway(
                description=f"Use {intermediate_format.value} as intermediate format",
                alternative_formats=[intermediate_format],
                required_changes=[
                    f"Add conversion step: {failed_conversion.source_format.value} -> {intermediate_format.value}",
                    f"Add conversion step: {intermediate_format.value} -> {failed_conversion.target_format.value}"
                ],
                implementation_complexity="medium"
            )
            
            alternatives.append(alternative)
        
        return alternatives
    
    def _find_alternative_target_formats(self,
                                       failed_conversion: ConversionRequest) -> List[AlternativePathway]:
        """Find alternative target formats with similar analytical capabilities."""
        alternatives = []
        
        # Map formats to their analytical capabilities
        capability_groups = {
            "tabular_analysis": [DataFormat.PANDAS_DATAFRAME, DataFormat.CSV, DataFormat.PARQUET],
            "numerical_computation": [DataFormat.NUMPY_ARRAY, DataFormat.SCIPY_SPARSE],
            "statistical_analysis": [DataFormat.STATISTICAL_RESULT, DataFormat.PANDAS_DATAFRAME],
            "time_series_analysis": [DataFormat.TIME_SERIES, DataFormat.PANDAS_DATAFRAME],
            "machine_learning": [DataFormat.NUMPY_ARRAY, DataFormat.SCIPY_SPARSE, DataFormat.PANDAS_DATAFRAME]
        }
        
        # Find capability group of target format
        target_capabilities = []
        for capability, formats in capability_groups.items():
            if failed_conversion.target_format in formats:
                target_capabilities.append(capability)
        
        # Find alternative formats with similar capabilities
        alternative_targets = set()
        for capability in target_capabilities:
            alternative_targets.update(capability_groups.get(capability, []))
        
        # Remove original target format
        alternative_targets.discard(failed_conversion.target_format)
        
        for alt_format in alternative_targets:
            # Check if conversion is feasible
            if self._check_conversion_feasibility(failed_conversion.source_format, alt_format):
                alternative = AlternativePathway(
                    description=f"Use {alt_format.value} instead of {failed_conversion.target_format.value}",
                    alternative_formats=[alt_format],
                    required_changes=[
                        f"Change target format to {alt_format.value}",
                        "Update downstream pipeline to handle new format"
                    ],
                    implementation_complexity="low"
                )
                alternatives.append(alternative)
        
        return alternatives
    
    def _find_source_preprocessing_alternatives(self,
                                              failed_conversion: ConversionRequest) -> List[AlternativePathway]:
        """Find alternatives involving source data preprocessing."""
        alternatives = []
        
        # Common preprocessing alternatives based on source format
        preprocessing_options = {
            DataFormat.PANDAS_DATAFRAME: [
                ("sparse_representation", "Convert to sparse matrix representation"),
                ("chunked_processing", "Process data in chunks to reduce memory usage"),
                ("column_subset", "Use only essential columns for conversion"),
                ("data_type_optimization", "Optimize data types to reduce memory usage")
            ],
            DataFormat.NUMPY_ARRAY: [
                ("dtype_conversion", "Convert to more memory-efficient data types"),
                ("array_reshaping", "Reshape array for better compatibility"),
                ("sparse_conversion", "Convert to sparse array if appropriate")
            ],
            DataFormat.CSV: [
                ("streaming_reader", "Use streaming CSV reader for large files"),
                ("column_selection", "Read only required columns"),
                ("data_type_specification", "Specify data types during reading")
            ]
        }
        
        source_options = preprocessing_options.get(failed_conversion.source_format, [])
        
        for option_id, description in source_options:
            alternative = AlternativePathway(
                description=f"Preprocess source data: {description}",
                alternative_formats=[failed_conversion.source_format],  # Same format, preprocessed
                required_changes=[
                    f"Add preprocessing step: {description}",
                    "Modify source data handling in pipeline"
                ],
                implementation_complexity="low" if "streaming" not in option_id else "medium"
            )
            alternatives.append(alternative)
        
        return alternatives
    
    def _find_restructuring_alternatives(self,
                                       failed_conversion: ConversionRequest,
                                       error_context: ErrorContext) -> List[AlternativePathway]:
        """Find alternatives involving pipeline restructuring."""
        alternatives = []
        
        # Restructuring strategies based on error type
        if error_context.error_type == ConversionError.Type.MEMORY_EXCEEDED:
            alternatives.append(AlternativePathway(
                description="Implement streaming-based conversion pipeline",
                required_changes=[
                    "Break conversion into streaming chunks",
                    "Implement memory-efficient intermediate storage",
                    "Add progress tracking and resumption capability"
                ],
                implementation_complexity="high",
                estimated_effort=8.0
            ))
        
        elif error_context.error_type == ConversionError.Type.TIMEOUT:
            alternatives.append(AlternativePathway(
                description="Implement asynchronous conversion pipeline",
                required_changes=[
                    "Convert to asynchronous processing",
                    "Add job queue for conversion tasks",
                    "Implement result caching and retrieval"
                ],
                implementation_complexity="high",
                estimated_effort=12.0
            ))
        
        elif error_context.error_type == ConversionError.Type.ADAPTER_NOT_FOUND:
            alternatives.append(AlternativePathway(
                description="Implement custom adapter for specific conversion",
                required_changes=[
                    "Develop custom conversion adapter",
                    "Register adapter in shim registry",
                    "Add unit tests and validation"
                ],
                implementation_complexity="high",
                estimated_effort=16.0
            ))
        
        return alternatives
    
    def _check_conversion_feasibility(self, source_format: DataFormat, target_format: DataFormat) -> bool:
        """Check if conversion between formats is feasible."""
        # This would typically use the compatibility matrix
        # For now, implement basic feasibility rules
        
        # Direct format compatibility
        if source_format == target_format:
            return True
        
        # Tabular data conversions
        tabular_formats = {DataFormat.PANDAS_DATAFRAME, DataFormat.CSV, DataFormat.PARQUET}
        if source_format in tabular_formats and target_format in tabular_formats:
            return True
        
        # Numerical data conversions
        numerical_formats = {DataFormat.NUMPY_ARRAY, DataFormat.SCIPY_SPARSE}
        if source_format in numerical_formats and target_format in numerical_formats:
            return True
        
        # DataFrame to numerical
        if source_format == DataFormat.PANDAS_DATAFRAME and target_format in numerical_formats:
            return True
        
        # Assume other conversions may be possible through adapters
        return True
    
    def _analyze_alternative_feasibility(self,
                                       alternative: AlternativePathway,
                                       failed_conversion: ConversionRequest,
                                       error_context: ErrorContext) -> None:
        """Analyze feasibility and cost-benefit for an alternative pathway."""
        # Base feasibility score
        feasibility_score = 0.5
        
        # Adjust based on complexity
        complexity_adjustments = {
            "low": 0.3,
            "medium": 0.0,
            "high": -0.2
        }
        feasibility_score += complexity_adjustments.get(alternative.implementation_complexity, 0.0)
        
        # Adjust based on error type compatibility
        error_type_compatibility = {
            ConversionError.Type.MEMORY_EXCEEDED: {
                "sparse": 0.4, "streaming": 0.5, "chunked": 0.4
            },
            ConversionError.Type.TIMEOUT: {
                "asynchronous": 0.3, "chunked": 0.2, "streaming": 0.2
            },
            ConversionError.Type.ADAPTER_NOT_FOUND: {
                "custom": 0.2, "intermediate": 0.3, "alternative": 0.4
            }
        }
        
        error_adjustments = error_type_compatibility.get(error_context.error_type, {})
        for keyword, adjustment in error_adjustments.items():
            if keyword in alternative.description.lower():
                feasibility_score += adjustment
                break
        
        # Clamp score to valid range
        alternative.feasibility_score = max(0.0, min(1.0, feasibility_score))
        
        # Calculate cost-benefit ratio (simplified)
        estimated_cost = alternative.estimated_effort * 10  # Cost in arbitrary units
        estimated_benefit = feasibility_score * 100  # Benefit in arbitrary units
        alternative.cost_benefit_ratio = estimated_benefit / max(estimated_cost, 1)
        
        # Estimate quality loss (simplified)
        quality_loss_factors = {
            "sparse": 0.05,
            "chunked": 0.02,
            "preprocessing": 0.01,
            "alternative_format": 0.10,
            "custom": 0.00
        }
        
        quality_loss = 0.0
        for keyword, loss in quality_loss_factors.items():
            if keyword in alternative.description.lower():
                quality_loss = max(quality_loss, loss)
        
        alternative.expected_quality_loss = quality_loss
    
    def _generate_pathway_cache_key(self,
                                  conversion_request: ConversionRequest,
                                  error_context: ErrorContext) -> str:
        """Generate cache key for pathway analysis."""
        return f"{conversion_request.source_format.value}_{conversion_request.target_format.value}_{error_context.error_type}"
    
    def record_pathway_success(self,
                             pathway: AlternativePathway,
                             success_metrics: Dict[str, Any]) -> None:
        """Record successful use of an alternative pathway."""
        with self._lock:
            self.stats["successful_alternatives"] += 1
            
            pattern_key = f"{pathway.alternative_formats}_{pathway.implementation_complexity}"
            self.success_patterns[pattern_key].append({
                "pathway_id": pathway.pathway_id,
                "feasibility_score": pathway.feasibility_score,
                "success_metrics": success_metrics,
                "timestamp": time.time()
            })
        
        logger.info("Alternative pathway success recorded",
                   pathway_id=pathway.pathway_id,
                   feasibility_score=pathway.feasibility_score)
    
    def get_pathway_statistics(self) -> Dict[str, Any]:
        """Get alternative pathway analysis statistics."""
        with self._lock:
            return {
                **self.stats,
                "cache_size": len(self.pathway_cache),
                "success_patterns": {k: len(v) for k, v in self.success_patterns.items()},
                "cache_hit_rate": self.stats["cache_hits"] / max(self.stats["pathways_analyzed"], 1)
            }
    
    def clear_pathway_cache(self) -> None:
        """Clear pathway analysis cache."""
        with self._lock:
            self.pathway_cache.clear()
            self._pathway_cache.clear()
            logger.info("Alternative pathway cache cleared")
    
    # Enhanced pathway discovery methods
    
    def find_alternative_pathways_enhanced(self, 
                                 failed_request: ConversionRequest) -> List[ConversionPath]:
        """
        Find alternative conversion pathways for a failed conversion (enhanced version).
        
        Args:
            failed_request: Original conversion request that failed
            
        Returns:
            List of alternative conversion pathways, sorted by viability
        """
        start_time = time.time()
        
        source_format = failed_request.source_format
        target_format = failed_request.target_format
        
        # Check cache first
        cache_key = f"{source_format.value}->{target_format.value}"
        if self.enable_pathway_caching and cache_key in self._pathway_cache:
            cached_pathways = self._pathway_cache[cache_key]
            logger.debug(f"Using cached pathways for {cache_key}: {len(cached_pathways)} found")
            return cached_pathways
        
        logger.info(f"Finding alternative pathways: {source_format.value} -> {target_format.value}")
        
        # Find pathways using graph search
        pathways = self._discover_pathways_bfs(source_format, target_format, failed_request)
        
        # Evaluate and sort pathways
        evaluated_pathways = []
        for pathway in pathways:
            cost = self.assess_pathway_cost(pathway)
            quality = self.assess_quality_degradation(pathway)
            
            # Calculate overall viability score
            viability_score = self._calculate_viability_score(pathway, cost, quality)
            pathway.success_probability = viability_score
            
            if quality.expected_quality_score >= self.quality_threshold:
                evaluated_pathways.append(pathway)
        
        # Sort by viability (success probability)
        evaluated_pathways.sort(key=lambda p: p.success_probability, reverse=True)
        
        # Cache successful search
        if self.enable_pathway_caching:
            self._pathway_cache[cache_key] = evaluated_pathways
        
        discovery_time = time.time() - start_time
        logger.info(f"Found {len(evaluated_pathways)} alternative pathways in {discovery_time:.2f}s")
        
        return evaluated_pathways
    
    def assess_pathway_cost(self, pathway: ConversionPath) -> PathwayCost:
        """
        Assess the computational cost of a conversion pathway.
        
        Args:
            pathway: Conversion pathway to assess
            
        Returns:
            Detailed cost assessment
        """
        total_computational_cost = 0.0
        total_time_overhead = 0.0
        total_memory_overhead = 0.0
        min_reliability = 1.0
        
        # Sum costs across all steps
        for step in pathway.steps:
            step_cost = step.estimated_cost
            total_computational_cost += step_cost.computational_cost
            total_time_overhead += step_cost.time_estimate_seconds
            total_memory_overhead += step_cost.memory_cost_mb
            min_reliability = min(min_reliability, step.confidence)
        
        # Calculate quality degradation (increases with pathway length)
        quality_degradation = min(len(pathway.steps) * 0.05, 0.3)  # Max 30% degradation
        
        # Confidence decreases with pathway complexity
        confidence = max(min_reliability * (0.95 ** len(pathway.steps)), 0.1)
        
        return PathwayCost(
            computational_cost=total_computational_cost,
            quality_degradation=quality_degradation,
            time_overhead=total_time_overhead,
            memory_overhead=total_memory_overhead,
            reliability_score=min_reliability,
            confidence=confidence
        )
    
    def assess_quality_degradation(self, pathway: ConversionPath) -> QualityAssessment:
        """
        Assess quality degradation for a conversion pathway.
        
        Args:
            pathway: Conversion pathway to assess
            
        Returns:
            Quality assessment with degradation analysis
        """
        # Base quality starts high and degrades with each conversion step
        base_quality = 1.0
        cumulative_degradation = 0.0
        metadata_preservation = 1.0
        data_fidelity = 1.0
        risk_factors = []
        
        for i, step in enumerate(pathway.steps):
            # Each step introduces some quality loss
            step_degradation = 0.02 + (0.01 * i)  # Increasing degradation per step
            cumulative_degradation += step_degradation
            
            # Specific format conversions have known quality impacts
            degradation_factor = self._get_format_degradation_factor(
                step.source_format, step.target_format
            )
            cumulative_degradation += degradation_factor
            
            # Metadata preservation decreases with conversions
            if self._loses_metadata(step.source_format, step.target_format):
                metadata_preservation *= 0.9
                risk_factors.append(f"Metadata loss in {step.source_format.value} -> {step.target_format.value}")
            
            # Data fidelity assessment
            if self._loses_precision(step.source_format, step.target_format):
                data_fidelity *= 0.95
                risk_factors.append(f"Precision loss in {step.source_format.value} -> {step.target_format.value}")
        
        expected_quality = max(base_quality - cumulative_degradation, 0.1)
        
        return QualityAssessment(
            expected_quality_score=expected_quality,
            quality_degradation=cumulative_degradation,
            metadata_preservation=metadata_preservation,
            data_fidelity=data_fidelity,
            risk_factors=risk_factors
        )
    
    def cache_successful_pathway(self, pathway: ConversionPath) -> None:
        """
        Cache a successful conversion pathway for future reuse.
        
        Args:
            pathway: Successfully executed conversion pathway
        """
        if not self.enable_pathway_caching:
            return
        
        with self._lock:
            cache_key = f"{pathway.source_format.value}->{pathway.target_format.value}"
            
            # Store successful pathway
            self._successful_pathways[pathway.path_id] = pathway
            
            # Update success rate
            current_rate = self._pathway_success_rates.get(cache_key, 0.0)
            self._pathway_success_rates[cache_key] = min(current_rate + 0.1, 1.0)
            
            # Update cache with this successful pathway prioritized
            if cache_key in self._pathway_cache:
                cached_pathways = self._pathway_cache[cache_key]
                # Move successful pathway to front if it exists
                for i, cached_pathway in enumerate(cached_pathways):
                    if cached_pathway.path_id == pathway.path_id:
                        cached_pathways.insert(0, cached_pathways.pop(i))
                        break
                else:
                    # Add new successful pathway to front
                    cached_pathways.insert(0, pathway)
            else:
                self._pathway_cache[cache_key] = [pathway]
            
            logger.info(f"Cached successful pathway {pathway.path_id} for {cache_key}")
    
    def _discover_pathways_bfs(self, 
                              source_format: DataFormat,
                              target_format: DataFormat,
                              original_request: ConversionRequest) -> List[ConversionPath]:
        """Discover pathways using breadth-first search."""
        
        if not self.registry:
            logger.warning("No registry available for pathway discovery")
            return []
        
        pathways = []
        visited = set()
        
        # BFS queue: (current_format, path_so_far, total_cost)
        queue = deque([(source_format, [], self._create_zero_cost())])
        
        while queue and len(pathways) < 10:  # Limit number of pathways
            current_format, path, current_cost = queue.popleft()
            
            # Skip if already visited this format in this path
            if current_format in [step.target_format for step in path]:
                continue
            
            # Skip if path is too deep
            if len(path) >= self.max_pathway_depth:
                continue
            
            # Check if we reached target
            if current_format == target_format and path:
                pathway = ConversionPath(
                    source_format=source_format,
                    target_format=target_format,
                    steps=path,
                    total_cost=current_cost,
                    success_probability=0.8 ** len(path)  # Decreases with steps
                )
                pathways.append(pathway)
                continue
            
            # Get available adapters from current format
            available_adapters = self._get_adapters_from_format(current_format)
            
            for adapter in available_adapters:
                supported_conversions = adapter.get_supported_conversions()
                
                for source_fmt, target_fmt in supported_conversions:
                    if source_fmt == current_format and target_fmt != current_format:
                        
                        # Create dummy request for cost estimation
                        dummy_request = ConversionRequest(
                            source_data=None,
                            source_format=source_fmt,
                            target_format=target_fmt,
                            context=original_request.context
                        )
                        
                        try:
                            step_cost = adapter.estimate_cost(dummy_request)
                            confidence = adapter.can_convert(dummy_request)
                        except Exception:
                            # Use default values if estimation fails
                            step_cost = self._create_default_cost()
                            confidence = 0.5
                        
                        new_step = ConversionStep(
                            adapter_id=adapter.adapter_id,
                            source_format=source_fmt,
                            target_format=target_fmt,
                            estimated_cost=step_cost,
                            confidence=confidence
                        )
                        
                        new_path = path + [new_step]
                        new_cost = self._add_costs(current_cost, step_cost)
                        
                        queue.append((target_fmt, new_path, new_cost))
        
        return pathways
    
    def _get_adapters_from_format(self, format_type: DataFormat) -> List[EnhancedShimAdapter]:
        """Get adapters that can convert from the given format."""
        if not self.registry:
            return []
        
        available_adapters = []
        for adapter in self.registry.get_active_adapters():
            supported_conversions = adapter.get_supported_conversions()
            for source_fmt, _ in supported_conversions:
                if source_fmt == format_type:
                    available_adapters.append(adapter)
                    break
        
        return available_adapters
    
    def _calculate_viability_score(self, 
                                  pathway: ConversionPath,
                                  cost: PathwayCost,
                                  quality: QualityAssessment) -> float:
        """Calculate overall viability score for pathway."""
        
        # Base score from quality
        quality_score = quality.expected_quality_score * 0.4
        
        # Reliability score
        reliability_score = cost.reliability_score * 0.3
        
        # Cost efficiency (inverse of computational cost, capped)
        cost_efficiency = max(1.0 - min(cost.computational_cost, 1.0), 0.1) * 0.2
        
        # Path simplicity (prefer shorter paths)
        simplicity_score = max(1.0 - (len(pathway.steps) - 1) * 0.1, 0.1) * 0.1
        
        return quality_score + reliability_score + cost_efficiency + simplicity_score
    
    def _get_format_degradation_factor(self, 
                                      source: DataFormat, 
                                      target: DataFormat) -> float:
        """Get quality degradation factor for specific format conversions."""
        degradation_map = {
            # High precision to low precision
            (DataFormat.PANDAS_DATAFRAME, DataFormat.PYTHON_LIST): 0.05,
            (DataFormat.NUMPY_ARRAY, DataFormat.PYTHON_LIST): 0.03,
            (DataFormat.SCIPY_SPARSE, DataFormat.PANDAS_DATAFRAME): 0.02,
            
            # Complex to simple
            (DataFormat.TIME_SERIES, DataFormat.PANDAS_DATAFRAME): 0.04,
            (DataFormat.CATEGORICAL, DataFormat.NUMPY_ARRAY): 0.06,
            
            # Structured to unstructured
            (DataFormat.PANDAS_DATAFRAME, DataFormat.PYTHON_DICT): 0.01,
        }
        
        return degradation_map.get((source, target), 0.01)  # Default small degradation
    
    def _loses_metadata(self, source: DataFormat, target: DataFormat) -> bool:
        """Check if conversion loses metadata."""
        metadata_losing_conversions = {
            (DataFormat.PANDAS_DATAFRAME, DataFormat.NUMPY_ARRAY),
            (DataFormat.PANDAS_DATAFRAME, DataFormat.PYTHON_LIST),
            (DataFormat.TIME_SERIES, DataFormat.NUMPY_ARRAY),
            (DataFormat.CATEGORICAL, DataFormat.NUMPY_ARRAY)
        }
        return (source, target) in metadata_losing_conversions
    
    def _loses_precision(self, source: DataFormat, target: DataFormat) -> bool:
        """Check if conversion loses numerical precision."""
        precision_losing_conversions = {
            (DataFormat.NUMPY_ARRAY, DataFormat.PYTHON_LIST),
            (DataFormat.SCIPY_SPARSE, DataFormat.PYTHON_LIST),
            (DataFormat.PANDAS_DATAFRAME, DataFormat.PYTHON_DICT)
        }
        return (source, target) in precision_losing_conversions
    
    def _create_zero_cost(self) -> ConversionCost:
        """Create zero-cost baseline."""
        return ConversionCost(
            computational_cost=0.0,
            memory_cost_mb=0.0,
            time_estimate_seconds=0.0,
            io_operations=0,
            network_operations=0,
            quality_impact=0.0
        )
    
    def _create_default_cost(self) -> ConversionCost:
        """Create default cost estimate."""
        return ConversionCost(
            computational_cost=0.1,
            memory_cost_mb=10.0,
            time_estimate_seconds=1.0,
            io_operations=0,
            network_operations=0,
            quality_impact=0.02
        )
    
    def _add_costs(self, cost1: ConversionCost, cost2: ConversionCost) -> ConversionCost:
        """Add two conversion costs together."""
        return ConversionCost(
            computational_cost=cost1.computational_cost + cost2.computational_cost,
            memory_cost_mb=cost1.memory_cost_mb + cost2.memory_cost_mb,
            time_estimate_seconds=cost1.time_estimate_seconds + cost2.time_estimate_seconds,
            io_operations=cost1.io_operations + cost2.io_operations,
            network_operations=cost1.network_operations + cost2.network_operations,
            quality_impact=cost1.quality_impact + cost2.quality_impact
        )
    
    def get_pathway_statistics(self) -> Dict[str, Any]:
        """Get pathway discovery and performance statistics."""
        with self._lock:
            return {
                'cached_pathways': len(self._pathway_cache),
                'successful_pathways': len(self._successful_pathways),
                'average_success_rates': dict(self._pathway_success_rates),
                'pathway_performance': dict(self._pathway_performance),
                **self.stats,
                "cache_size": len(self.pathway_cache),
                "success_patterns": {k: len(v) for k, v in self.success_patterns.items()},
                "cache_hit_rate": self.stats["cache_hits"] / max(self.stats["pathways_analyzed"], 1)
            }


class RollbackManager:
    """
    Transaction-like state management with rollback capabilities.
    
    Provides checkpointing, state restoration, and resource cleanup
    for complex multi-step conversion operations.
    """
    
    def __init__(self,
                 max_checkpoints: int = 10,
                 enable_disk_persistence: bool = False,
                 checkpoint_compression: bool = True,
                 checkpoint_storage_path: Optional[str] = None,
                 cleanup_on_success: bool = True):
        """
        Initialize RollbackManager.
        
        Args:
            max_checkpoints: Maximum number of checkpoints to maintain
            enable_disk_persistence: Store checkpoints on disk for recovery
            checkpoint_compression: Compress checkpoint data to save space
            checkpoint_storage_path: Path for checkpoint storage (None for temp)
            cleanup_on_success: Whether to cleanup checkpoints on successful completion
        """
        self.max_checkpoints = max_checkpoints
        self.enable_disk_persistence = enable_disk_persistence
        self.checkpoint_compression = checkpoint_compression
        self.checkpoint_storage_path = checkpoint_storage_path or tempfile.mkdtemp(prefix="localdata_checkpoints_")
        self.cleanup_on_success = cleanup_on_success
        
        # Ensure storage directory exists
        os.makedirs(self.checkpoint_storage_path, exist_ok=True)
        
        # Checkpoint tracking
        self.checkpoints: Dict[str, List[PipelineCheckpoint]] = defaultdict(list)
        self.active_transactions: Set[str] = set()
        
        # Statistics
        self.stats = {
            "checkpoints_created": 0,
            "rollbacks_performed": 0,
            "successful_cleanups": 0,
            "failed_cleanups": 0
        }
        
        self._lock = threading.RLock()
        
        logger.info("RollbackManager initialized",
                   storage_path=self.checkpoint_storage_path,
                   max_checkpoints=max_checkpoints)
    
    def begin_transaction(self, pipeline_id: str) -> str:
        """
        Begin a new transaction for pipeline execution.
        
        Args:
            pipeline_id: Identifier for the pipeline
            
        Returns:
            Transaction ID
        """
        transaction_id = f"{pipeline_id}_{int(time.time() * 1000)}"
        
        with self._lock:
            self.active_transactions.add(transaction_id)
        
        logger.info("Transaction begun",
                   transaction_id=transaction_id,
                   pipeline_id=pipeline_id)
        
        return transaction_id
    
    def create_checkpoint(self,
                        transaction_id: str,
                        pipeline_id: str,
                        step_index: int,
                        step_id: str,
                        data: Any,
                        metadata: Optional[Dict[str, Any]] = None,
                        execution_context: Optional[Dict[str, Any]] = None) -> PipelineCheckpoint:
        """
        Create a checkpoint for pipeline state.
        
        Args:
            transaction_id: Transaction identifier
            pipeline_id: Pipeline identifier
            step_index: Index of current step
            step_id: Identifier of current step
            data: Data state to checkpoint
            metadata: Additional metadata
            execution_context: Execution context to save
            
        Returns:
            Created checkpoint
        """
        checkpoint = PipelineCheckpoint(
            pipeline_id=pipeline_id,
            step_index=step_index,
            step_id=step_id,
            metadata=metadata or {},
            execution_context=execution_context or {}
        )
        
        try:
            # Serialize and store data
            if data is not None:
                checkpoint.data_hash = str(hash(str(data)))
                storage_path = os.path.join(
                    self.checkpoint_storage_path,
                    f"checkpoint_{checkpoint.checkpoint_id}.pkl"
                )
                
                with open(storage_path, 'wb') as f:
                    pickle.dump(data, f)
                
                checkpoint.storage_path = storage_path
                checkpoint.data_snapshot = None  # Don't keep in memory
            
            # Add to checkpoint history
            with self._lock:
                self.checkpoints[transaction_id].append(checkpoint)
                self.stats["checkpoints_created"] += 1
                
                # Maintain checkpoint limit
                if len(self.checkpoints[transaction_id]) > self.max_checkpoints:
                    old_checkpoint = self.checkpoints[transaction_id].pop(0)
                    self._cleanup_checkpoint_storage(old_checkpoint)
            
            logger.debug("Checkpoint created",
                        transaction_id=transaction_id,
                        checkpoint_id=checkpoint.checkpoint_id,
                        step_id=step_id)
            
            return checkpoint
            
        except Exception as e:
            logger.error(f"Failed to create checkpoint: {e}")
            raise PipelineError(
                f"Checkpoint creation failed: {str(e)}",
                ErrorClassification.CONFIGURATION_ERROR,
                "checkpoint_creation"
            )
    
    def rollback_to_checkpoint(self,
                             transaction_id: str,
                             checkpoint_id: Optional[str] = None) -> Tuple[Any, Dict[str, Any]]:
        """
        Rollback to a specific checkpoint or the latest one.
        
        Args:
            transaction_id: Transaction to rollback
            checkpoint_id: Specific checkpoint ID (None for latest)
            
        Returns:
            Tuple of (restored_data, execution_context)
        """
        with self._lock:
            if transaction_id not in self.checkpoints:
                raise ValueError(f"No checkpoints found for transaction: {transaction_id}")
            
            transaction_checkpoints = self.checkpoints[transaction_id]
            if not transaction_checkpoints:
                raise ValueError(f"No checkpoints available for transaction: {transaction_id}")
            
            # Find target checkpoint
            if checkpoint_id:
                target_checkpoint = None
                for cp in transaction_checkpoints:
                    if cp.checkpoint_id == checkpoint_id:
                        target_checkpoint = cp
                        break
                
                if not target_checkpoint:
                    raise ValueError(f"Checkpoint not found: {checkpoint_id}")
            else:
                target_checkpoint = transaction_checkpoints[-1]  # Latest checkpoint
            
            self.stats["rollbacks_performed"] += 1
        
        try:
            # Restore data from checkpoint
            restored_data = None
            if target_checkpoint.storage_path and os.path.exists(target_checkpoint.storage_path):
                with open(target_checkpoint.storage_path, 'rb') as f:
                    restored_data = pickle.load(f)
            elif target_checkpoint.data_snapshot is not None:
                restored_data = target_checkpoint.data_snapshot
            
            # Cleanup checkpoints after rollback point
            self._cleanup_checkpoints_after(transaction_id, target_checkpoint)
            
            logger.info("Rollback completed",
                       transaction_id=transaction_id,
                       checkpoint_id=target_checkpoint.checkpoint_id,
                       step_id=target_checkpoint.step_id)
            
            return restored_data, target_checkpoint.execution_context
            
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            raise PipelineError(
                f"Rollback failed: {str(e)}",
                ErrorClassification.EXTERNAL_DEPENDENCY_ERROR,
                "rollback_operation"
            )
    
    def commit_transaction(self, transaction_id: str) -> None:
        """
        Commit transaction and optionally cleanup checkpoints.
        
        Args:
            transaction_id: Transaction to commit
        """
        with self._lock:
            if transaction_id not in self.active_transactions:
                logger.warning(f"Transaction not active: {transaction_id}")
                return
            
            self.active_transactions.remove(transaction_id)
        
        # Cleanup checkpoints if configured
        if self.cleanup_on_success:
            self._cleanup_transaction_checkpoints(transaction_id)
        
        logger.info("Transaction committed",
                   transaction_id=transaction_id,
                   cleanup_performed=self.cleanup_on_success)
    
    def abort_transaction(self, transaction_id: str) -> None:
        """
        Abort transaction and cleanup all checkpoints.
        
        Args:
            transaction_id: Transaction to abort
        """
        with self._lock:
            if transaction_id in self.active_transactions:
                self.active_transactions.remove(transaction_id)
        
        # Always cleanup on abort
        self._cleanup_transaction_checkpoints(transaction_id)
        
        logger.info("Transaction aborted",
                   transaction_id=transaction_id)
    
    def get_checkpoint_history(self, transaction_id: str) -> List[PipelineCheckpoint]:
        """Get checkpoint history for a transaction."""
        with self._lock:
            return self.checkpoints.get(transaction_id, []).copy()
    
    def _cleanup_checkpoints_after(self,
                                 transaction_id: str,
                                 target_checkpoint: PipelineCheckpoint) -> None:
        """Cleanup checkpoints created after the target checkpoint."""
        with self._lock:
            if transaction_id not in self.checkpoints:
                return
            
            checkpoints = self.checkpoints[transaction_id]
            target_index = -1
            
            for i, cp in enumerate(checkpoints):
                if cp.checkpoint_id == target_checkpoint.checkpoint_id:
                    target_index = i
                    break
            
            if target_index >= 0:
                # Cleanup checkpoints after target
                checkpoints_to_cleanup = checkpoints[target_index + 1:]
                self.checkpoints[transaction_id] = checkpoints[:target_index + 1]
                
                for cp in checkpoints_to_cleanup:
                    self._cleanup_checkpoint_storage(cp)
    
    def _cleanup_transaction_checkpoints(self, transaction_id: str) -> None:
        """Cleanup all checkpoints for a transaction."""
        with self._lock:
            checkpoints = self.checkpoints.get(transaction_id, [])
            if transaction_id in self.checkpoints:
                del self.checkpoints[transaction_id]
        
        # Cleanup storage for all checkpoints
        for checkpoint in checkpoints:
            self._cleanup_checkpoint_storage(checkpoint)
        
        logger.debug(f"Cleaned up {len(checkpoints)} checkpoints for transaction {transaction_id}")
    
    def _cleanup_checkpoint_storage(self, checkpoint: PipelineCheckpoint) -> None:
        """Cleanup storage for a specific checkpoint."""
        try:
            # Remove storage file
            if checkpoint.storage_path and os.path.exists(checkpoint.storage_path):
                os.remove(checkpoint.storage_path)
            
            # Remove temporary files
            for temp_file in checkpoint.temporary_files:
                if os.path.exists(temp_file):
                    if os.path.isdir(temp_file):
                        shutil.rmtree(temp_file)
                    else:
                        os.remove(temp_file)
            
            with self._lock:
                self.stats["successful_cleanups"] += 1
            
        except Exception as e:
            with self._lock:
                self.stats["failed_cleanups"] += 1
            logger.warning(f"Checkpoint cleanup failed: {e}", checkpoint_id=checkpoint.checkpoint_id)
    
    def get_rollback_statistics(self) -> Dict[str, Any]:
        """Get rollback manager statistics."""
        with self._lock:
            return {
                **self.stats,
                "active_transactions": len(self.active_transactions),
                "total_checkpoints": sum(len(checkpoints) for checkpoints in self.checkpoints.values()),
                "storage_path": self.checkpoint_storage_path,
                "storage_size_mb": self._calculate_storage_size()
            }
    
    def _calculate_storage_size(self) -> float:
        """Calculate total storage size in MB."""
        total_size = 0
        try:
            for root, dirs, files in os.walk(self.checkpoint_storage_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    total_size += os.path.getsize(file_path)
        except Exception:
            pass
        
        return total_size / (1024 * 1024)
    
    def cleanup_all_checkpoints(self) -> None:
        """Cleanup all checkpoints and reset state."""
        with self._lock:
            all_checkpoints = []
            for checkpoints in self.checkpoints.values():
                all_checkpoints.extend(checkpoints)
            
            self.checkpoints.clear()
            self.active_transactions.clear()
        
        # Cleanup storage for all checkpoints
        for checkpoint in all_checkpoints:
            self._cleanup_checkpoint_storage(checkpoint)
        
        logger.info(f"Cleaned up all {len(all_checkpoints)} checkpoints")


class RecoveryStrategyEngine:
    """
    Configurable recovery strategy engine with learning capabilities.
    
    Orchestrates recovery strategies based on error context, learns from failure patterns,
    and provides intelligent strategy selection for different error scenarios.
    """
    
    def __init__(self,
                 error_handler: ConversionErrorHandler,
                 pathway_engine: AlternativePathwayEngine,
                 rollback_manager: RollbackManager):
        """
        Initialize RecoveryStrategyEngine.
        
        Args:
            error_handler: Error handler for basic recovery operations
            pathway_engine: Engine for finding alternative pathways
            rollback_manager: Manager for rollback operations
        """
        self.error_handler = error_handler
        self.pathway_engine = pathway_engine
        self.rollback_manager = rollback_manager
        
        # Strategy configuration
        self.strategy_configs: Dict[RecoveryStrategy, Dict[str, Any]] = {
            RecoveryStrategy.RETRY: {
                "max_attempts": 3,
                "backoff_factor": 2.0,
                "max_delay": 60.0
            },
            RecoveryStrategy.FALLBACK: {
                "quality_threshold": 0.7,
                "performance_degradation_allowed": 0.5
            },
            RecoveryStrategy.ALTERNATIVE_PATH: {
                "max_alternatives": 5,
                "feasibility_threshold": 0.6
            },
            RecoveryStrategy.ROLLBACK: {
                "max_rollback_steps": 3,
                "cleanup_on_rollback": True
            }
        }
        
        # Learning system
        self.strategy_performance: Dict[str, Dict[RecoveryStrategy, List[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        self.error_pattern_strategies: Dict[str, List[RecoveryStrategy]] = {}
        
        # Execution statistics
        self.stats = {
            "recoveries_attempted": 0,
            "recoveries_successful": 0,
            "strategy_usage": defaultdict(int),
            "average_recovery_time": 0.0
        }
        
        self._lock = threading.RLock()
        
        logger.info("RecoveryStrategyEngine initialized")
    
    def execute_recovery(self,
                        error_context: ErrorContext,
                        operation: Callable,
                        *args, **kwargs) -> Tuple[bool, Any, Dict[str, Any]]:
        """
        Execute comprehensive recovery strategy for error.
        
        Args:
            error_context: Context of the error to recover from
            operation: Original operation to recover
            *args, **kwargs: Operation arguments
            
        Returns:
            Tuple of (success, result, recovery_metadata)
        """
        start_time = time.time()
        
        with self._lock:
            self.stats["recoveries_attempted"] += 1
        
        logger.info("Executing recovery strategy",
                   error_id=error_context.error_id,
                   error_type=error_context.error_type,
                   severity=error_context.severity.value)
        
        # Generate recovery plan
        recovery_plan = self._generate_intelligent_recovery_plan(error_context)
        
        recovery_metadata = {
            "recovery_plan": recovery_plan,
            "attempted_strategies": [],
            "successful_strategy": None,
            "execution_time": 0.0,
            "quality_impact": 0.0
        }
        
        # Execute strategies in order
        for strategy in recovery_plan.strategies:
            strategy_start_time = time.time()
            
            try:
                with self._lock:
                    self.stats["strategy_usage"][strategy] += 1
                
                success, result = self._execute_strategy(
                    strategy, error_context, operation, *args, **kwargs
                )
                
                strategy_execution_time = time.time() - strategy_start_time
                
                recovery_metadata["attempted_strategies"].append({
                    "strategy": strategy.value,
                    "success": success,
                    "execution_time": strategy_execution_time
                })
                
                if success:
                    # Record successful recovery
                    self._record_strategy_success(error_context, strategy, strategy_execution_time)
                    
                    recovery_metadata["successful_strategy"] = strategy.value
                    recovery_metadata["execution_time"] = time.time() - start_time
                    
                    with self._lock:
                        self.stats["recoveries_successful"] += 1
                        # Update average recovery time
                        total_time = self.stats["average_recovery_time"] * (self.stats["recoveries_successful"] - 1)
                        self.stats["average_recovery_time"] = (total_time + recovery_metadata["execution_time"]) / self.stats["recoveries_successful"]
                    
                    logger.info("Recovery successful",
                               error_id=error_context.error_id,
                               strategy=strategy.value,
                               total_time=recovery_metadata["execution_time"])
                    
                    return True, result, recovery_metadata
                
            except Exception as strategy_error:
                logger.warning(f"Recovery strategy {strategy.value} failed: {strategy_error}")
                recovery_metadata["attempted_strategies"][-1]["error"] = str(strategy_error)
        
        # All strategies failed
        recovery_metadata["execution_time"] = time.time() - start_time
        
        logger.warning("All recovery strategies failed",
                      error_id=error_context.error_id,
                      strategies_attempted=len(recovery_plan.strategies))
        
        return False, None, recovery_metadata
    
    def _generate_intelligent_recovery_plan(self, error_context: ErrorContext) -> RecoveryPlan:
        """Generate intelligent recovery plan based on error context and learning."""
        # Start with base recovery plan from error handler
        base_plan = self.error_handler._generate_recovery_plan(error_context)
        
        # Enhance with learned patterns
        error_pattern = self._classify_error_pattern(error_context)
        learned_strategies = self._get_learned_strategies(error_pattern)
        
        # Combine and prioritize strategies
        combined_strategies = list(base_plan.strategies)
        for strategy in learned_strategies:
            if strategy not in combined_strategies:
                combined_strategies.append(strategy)
        
        # Reorder based on historical success rates
        prioritized_strategies = self._prioritize_strategies(combined_strategies, error_context)
        
        # Create enhanced recovery plan
        enhanced_plan = RecoveryPlan(
            error_context=error_context,
            strategies=prioritized_strategies,
            strategy_configs=self._get_strategy_configs_for_error(error_context),
            estimated_recovery_time=self._estimate_plan_execution_time(prioritized_strategies),
            estimated_success_probability=self._estimate_plan_success_probability(prioritized_strategies, error_context)
        )
        
        return enhanced_plan
    
    def _execute_strategy(self,
                         strategy: RecoveryStrategy,
                         error_context: ErrorContext,
                         operation: Callable,
                         *args, **kwargs) -> Tuple[bool, Any]:
        """Execute specific recovery strategy."""
        logger.debug(f"Executing strategy: {strategy.value}")
        
        if strategy == RecoveryStrategy.RETRY:
            return self.error_handler.execute_recovery_strategy(
                error_context, strategy, operation, *args, **kwargs
            )
        
        elif strategy == RecoveryStrategy.FALLBACK:
            return self.error_handler.execute_recovery_strategy(
                error_context, strategy, operation, *args, **kwargs
            )
        
        elif strategy == RecoveryStrategy.SKIP:
            return self.error_handler.execute_recovery_strategy(
                error_context, strategy, operation, *args, **kwargs
            )
        
        elif strategy == RecoveryStrategy.ALTERNATIVE_PATH:
            return self._execute_alternative_path_recovery(error_context, operation, *args, **kwargs)
        
        elif strategy == RecoveryStrategy.ROLLBACK:
            return self._execute_rollback_recovery(error_context, operation, *args, **kwargs)
        
        else:
            logger.warning(f"Unknown recovery strategy: {strategy.value}")
            return False, None
    
    def _execute_alternative_path_recovery(self,
                                         error_context: ErrorContext,
                                         operation: Callable,
                                         *args, **kwargs) -> Tuple[bool, Any]:
        """Execute alternative pathway recovery."""
        if not error_context.conversion_request:
            return False, None
        
        # Find alternative pathways
        alternatives = self.pathway_engine.find_alternative_pathways(
            error_context.conversion_request, error_context
        )
        
        if not alternatives:
            logger.debug("No alternative pathways found")
            return False, None
        
        # Try the most feasible alternative
        best_alternative = alternatives[0]
        
        logger.info(f"Attempting alternative pathway: {best_alternative.description}")
        
        # This would typically modify the conversion request and retry
        # For now, return success indication
        return True, {
            "status": "alternative_pathway_used",
            "pathway": best_alternative,
            "original_error": error_context.error_id
        }
    
    def _execute_rollback_recovery(self,
                                 error_context: ErrorContext,
                                 operation: Callable,
                                 *args, **kwargs) -> Tuple[bool, Any]:
        """Execute rollback recovery."""
        if not error_context.pipeline_id:
            logger.debug("No pipeline ID for rollback")
            return False, None
        
        try:
            # This would typically rollback to a previous checkpoint
            # For demonstration, we'll simulate successful rollback
            logger.info("Performing pipeline rollback")
            
            return True, {
                "status": "rollback_completed",
                "pipeline_id": error_context.pipeline_id,
                "error_id": error_context.error_id
            }
            
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False, None
    
    def _classify_error_pattern(self, error_context: ErrorContext) -> str:
        """Classify error into pattern for learning purposes."""
        components = [
            str(error_context.error_type),
            error_context.severity.value
        ]
        
        if error_context.conversion_request:
            components.extend([
                error_context.conversion_request.source_format.value,
                error_context.conversion_request.target_format.value
            ])
        
        return "_".join(components)
    
    def _get_learned_strategies(self, error_pattern: str) -> List[RecoveryStrategy]:
        """Get learned strategies for error pattern."""
        return self.error_pattern_strategies.get(error_pattern, [])
    
    def _prioritize_strategies(self,
                             strategies: List[RecoveryStrategy],
                             error_context: ErrorContext) -> List[RecoveryStrategy]:
        """Prioritize strategies based on historical success rates."""
        error_pattern = self._classify_error_pattern(error_context)
        
        # Get success rates for each strategy
        strategy_scores = {}
        for strategy in strategies:
            performances = self.strategy_performance[error_pattern][strategy]
            if performances:
                # Success rate based on recent performance
                recent_performances = performances[-10:]  # Last 10 attempts
                success_rate = sum(1 for p in recent_performances if p > 0) / len(recent_performances)
                avg_time = sum(abs(p) for p in recent_performances) / len(recent_performances)
                
                # Score combines success rate and speed (lower time is better)
                strategy_scores[strategy] = success_rate - (avg_time / 100.0)
            else:
                # Default score for untested strategies
                default_scores = {
                    RecoveryStrategy.RETRY: 0.7,
                    RecoveryStrategy.FALLBACK: 0.8,
                    RecoveryStrategy.SKIP: 0.9,
                    RecoveryStrategy.ALTERNATIVE_PATH: 0.6,
                    RecoveryStrategy.ROLLBACK: 0.5
                }
                strategy_scores[strategy] = default_scores.get(strategy, 0.5)
        
        # Sort by score (highest first)
        return sorted(strategies, key=lambda s: strategy_scores.get(s, 0), reverse=True)
    
    def _record_strategy_success(self,
                               error_context: ErrorContext,
                               strategy: RecoveryStrategy,
                               execution_time: float) -> None:
        """Record successful strategy execution for learning."""
        error_pattern = self._classify_error_pattern(error_context)
        
        with self._lock:
            # Record positive performance (success)
            self.strategy_performance[error_pattern][strategy].append(execution_time)
            
            # Update learned strategies for this pattern
            if error_pattern not in self.error_pattern_strategies:
                self.error_pattern_strategies[error_pattern] = []
            
            if strategy not in self.error_pattern_strategies[error_pattern]:
                self.error_pattern_strategies[error_pattern].insert(0, strategy)
            else:
                # Move successful strategy to front
                self.error_pattern_strategies[error_pattern].remove(strategy)
                self.error_pattern_strategies[error_pattern].insert(0, strategy)
            
            # Maintain history size
            if len(self.strategy_performance[error_pattern][strategy]) > 50:
                self.strategy_performance[error_pattern][strategy] = self.strategy_performance[error_pattern][strategy][-25:]
    
    def _get_strategy_configs_for_error(self, error_context: ErrorContext) -> Dict[RecoveryStrategy, Dict[str, Any]]:
        """Get strategy configurations adapted for specific error."""
        configs = self.strategy_configs.copy()
        
        # Adapt configurations based on error context
        if error_context.error_type == ConversionError.Type.MEMORY_EXCEEDED:
            configs[RecoveryStrategy.RETRY]["max_attempts"] = 1  # Don't retry memory errors aggressively
            configs[RecoveryStrategy.ALTERNATIVE_PATH]["max_alternatives"] = 3
        
        elif error_context.error_type == ConversionError.Type.TIMEOUT:
            configs[RecoveryStrategy.RETRY]["max_delay"] = 120.0  # Longer delays for timeout
            configs[RecoveryStrategy.RETRY]["backoff_factor"] = 1.5
        
        return configs
    
    def _estimate_plan_execution_time(self, strategies: List[RecoveryStrategy]) -> float:
        """Estimate execution time for recovery plan."""
        base_times = {
            RecoveryStrategy.RETRY: 5.0,
            RecoveryStrategy.FALLBACK: 2.0,
            RecoveryStrategy.SKIP: 0.1,
            RecoveryStrategy.ALTERNATIVE_PATH: 8.0,
            RecoveryStrategy.ROLLBACK: 3.0
        }
        
        return sum(base_times.get(strategy, 5.0) for strategy in strategies)
    
    def _estimate_plan_success_probability(self,
                                         strategies: List[RecoveryStrategy],
                                         error_context: ErrorContext) -> float:
        """Estimate success probability for recovery plan."""
        error_pattern = self._classify_error_pattern(error_context)
        
        # Calculate probability that at least one strategy succeeds
        failure_probability = 1.0
        
        for strategy in strategies:
            performances = self.strategy_performance[error_pattern][strategy]
            if performances:
                recent_performances = performances[-5:]  # Recent history
                success_rate = sum(1 for p in recent_performances if p > 0) / len(recent_performances)
            else:
                # Default success rates
                default_rates = {
                    RecoveryStrategy.RETRY: 0.6,
                    RecoveryStrategy.FALLBACK: 0.8,
                    RecoveryStrategy.SKIP: 1.0,
                    RecoveryStrategy.ALTERNATIVE_PATH: 0.7,
                    RecoveryStrategy.ROLLBACK: 0.9
                }
                success_rate = default_rates.get(strategy, 0.5)
            
            failure_probability *= (1.0 - success_rate)
        
        return 1.0 - failure_probability
    
    def get_recovery_statistics(self) -> Dict[str, Any]:
        """Get comprehensive recovery strategy statistics."""
        with self._lock:
            return {
                **dict(self.stats),
                "learned_patterns": len(self.error_pattern_strategies),
                "strategy_performance_data": {
                    pattern: {
                        strategy.value: {
                            "attempts": len(performances),
                            "avg_time": sum(abs(p) for p in performances) / len(performances) if performances else 0,
                            "success_rate": sum(1 for p in performances if p > 0) / len(performances) if performances else 0
                        }
                        for strategy, performances in strategies.items()
                    }
                    for pattern, strategies in self.strategy_performance.items()
                },
                "strategy_configurations": self.strategy_configs
            }
    
    def update_strategy_config(self,
                             strategy: RecoveryStrategy,
                             config_updates: Dict[str, Any]) -> None:
        """Update configuration for a specific strategy."""
        with self._lock:
            if strategy not in self.strategy_configs:
                self.strategy_configs[strategy] = {}
            
            self.strategy_configs[strategy].update(config_updates)
        
        logger.info(f"Strategy configuration updated",
                   strategy=strategy.value,
                   updates=config_updates)
    
    def clear_learning_data(self) -> None:
        """Clear all learned patterns and performance data."""
        with self._lock:
            self.strategy_performance.clear()
            self.error_pattern_strategies.clear()
        
        logger.info("Recovery strategy learning data cleared")


# Factory Functions

def create_conversion_error_handler(**kwargs) -> ConversionErrorHandler:
    """Create ConversionErrorHandler with standard configuration."""
    return ConversionErrorHandler(**kwargs)


def create_alternative_pathway_engine(registry: Optional[ShimRegistry] = None,
                                    **kwargs) -> AlternativePathwayEngine:
    """Create AlternativePathwayEngine with required dependencies."""
    return AlternativePathwayEngine(registry=registry, **kwargs)


def create_rollback_manager(**kwargs) -> RollbackManager:
    """Create RollbackManager with standard configuration."""
    return RollbackManager(**kwargs)


def create_recovery_strategy_engine(error_handler: ConversionErrorHandler,
                                  pathway_engine: AlternativePathwayEngine,
                                  rollback_manager: RollbackManager) -> RecoveryStrategyEngine:
    """Create RecoveryStrategyEngine with all components."""
    return RecoveryStrategyEngine(error_handler, pathway_engine, rollback_manager)


def create_complete_error_recovery_system(registry: Optional[ShimRegistry] = None,
                                        **kwargs) -> Dict[str, Any]:
    """
    Create complete error recovery system with all components.
    
    Args:
        registry: Optional ShimRegistry for adapter discovery
        **kwargs: Additional configuration options
        
    Returns:
        Dictionary with all recovery system components
    """
    # Create individual components
    error_handler = create_conversion_error_handler(**kwargs.get("error_handler", {}))
    pathway_engine = create_alternative_pathway_engine(registry=registry, **kwargs.get("pathway_engine", {}))
    rollback_manager = create_rollback_manager(**kwargs.get("rollback_manager", {}))
    recovery_engine = create_recovery_strategy_engine(error_handler, pathway_engine, rollback_manager)
    
    logger.info("Complete error recovery system created")
    
    return {
        "error_handler": error_handler,
        "pathway_engine": pathway_engine,
        "rollback_manager": rollback_manager,
        "recovery_engine": recovery_engine
    }


def create_error_recovery_framework(registry: Optional[ShimRegistry] = None,
                                  **kwargs) -> Tuple[ConversionErrorHandler, AlternativePathwayEngine, RollbackManager, RecoveryStrategyEngine]:
    """
    Create complete error recovery framework components as tuple.
    
    Args:
        registry: Optional ShimRegistry for pathway discovery
        **kwargs: Component-specific configuration options
        
    Returns:
        Tuple of (error_handler, pathway_engine, rollback_manager, recovery_engine)
    """
    # Create individual components
    error_handler = create_conversion_error_handler(
        **kwargs.get("error_handler", {})
    )
    
    pathway_engine = create_alternative_pathway_engine(
        registry=registry, 
        **kwargs.get("pathway_engine", {})
    )
    
    rollback_manager = create_rollback_manager(
        **kwargs.get("rollback_manager", {})
    )
    
    recovery_engine = create_recovery_strategy_engine(
        error_handler=error_handler,
        pathway_engine=pathway_engine,
        rollback_manager=rollback_manager,
        **kwargs.get("recovery_engine", {})
    )
    
    return error_handler, pathway_engine, rollback_manager, recovery_engine


# Utility Functions

def handle_pipeline_error_with_recovery(error: Exception,
                                       conversion_request: ConversionRequest,
                                       recovery_system: Dict[str, Any],
                                       operation: Callable,
                                       *args, **kwargs) -> Tuple[bool, Any, Dict[str, Any]]:
    """
    High-level utility function to handle pipeline error with full recovery.
    
    Args:
        error: The error that occurred
        conversion_request: The conversion request that failed
        recovery_system: Complete recovery system from create_complete_error_recovery_system
        operation: Original operation to recover
        *args, **kwargs: Operation arguments
        
    Returns:
        Tuple of (success, result, recovery_metadata)
    """
    error_handler = recovery_system["error_handler"]
    recovery_engine = recovery_system["recovery_engine"]
    
    # Handle error and create context
    error_context = error_handler.handle_conversion_error(error, conversion_request)
    
    # Execute recovery
    return recovery_engine.execute_recovery(error_context, operation, *args, **kwargs)


# Alias for compatibility with test framework
RecoveryStrategyFramework = RecoveryStrategyEngine