"""
ConversionErrorHandler: advanced error classification and handling for conversion operations.
"""

import logging
import os
import random
import threading
import time
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..interfaces import (
    ConversionContext,
    ConversionError,
    ConversionRequest,
)
from ...base import ErrorClassification, PipelineError
from ....logging_manager import get_logger
from ._circuit_breaker import CircuitBreaker
from ._types import (
    CircuitBreakerState,
    ErrorAggregation,
    ErrorClassificationEnhanced,
    ErrorContext,
    ErrorHandlingResult,
    ErrorRecoverability,
    ErrorSeverity,
    PerformanceImpact,
    RecoveryPlan,
    RecoveryStrategy,
)

logger = get_logger(__name__)


class ConversionErrorHandler:
    """
    Advanced error classification and handling for conversion operations.

    Provides sophisticated error analysis with diagnostic context, performance
    impact assessment, and intelligent recovery recommendations.
    """

    def __init__(
        self,
        classification_confidence_threshold: float = 0.7,
        enable_performance_tracking: bool = True,
        enable_error_learning: bool = True,
        max_retry_attempts: int = 3,
        base_retry_delay: float = 1.0,
        max_retry_delay: float = 60.0,
        exponential_backoff: bool = True,
        jitter: bool = True,
    ):
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
            "conversion_timeout": CircuitBreaker(
                failure_threshold=3, recovery_timeout=30
            ),
            "memory_overflow": CircuitBreaker(
                failure_threshold=2, recovery_timeout=120
            ),
            "adapter_not_found": CircuitBreaker(
                failure_threshold=5, recovery_timeout=60
            ),
        }

        self._lock = threading.RLock()

        # Enhanced error learning and tracking
        if self.enable_error_learning:
            logger.info("Error pattern learning enabled")
        if self.enable_performance_tracking:
            logger.info("Performance impact tracking enabled")

        logger.info(
            "ConversionErrorHandler initialized",
            max_retries=max_retry_attempts,
            backoff_strategy="exponential" if exponential_backoff else "linear",
            confidence_threshold=classification_confidence_threshold,
            performance_tracking=enable_performance_tracking,
            error_learning=enable_error_learning,
        )

    def handle_conversion_error(
        self,
        error: Exception,
        conversion_request: ConversionRequest,
        pipeline_context: Optional[Dict[str, Any]] = None,
    ) -> ErrorContext:
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
        logger.error(
            f"Conversion error handled",
            error_id=error_context.error_id,
            error_type=error_context.error_type,
            severity=error_context.severity.value,
            source_format=conversion_request.source_format.value,
            target_format=conversion_request.target_format.value,
            recovery_strategies=len(recovery_plan.strategies),
            handling_time=time.time() - start_time,
        )

        return error_context

    def classify_error(
        self, error: Exception, context: ConversionContext
    ) -> ErrorClassificationEnhanced:
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

            logger.info(
                "Error classified",
                error_type=classification.error_type.value,
                severity=classification.severity.value,
                recoverability=classification.recoverability.value,
                confidence=classification.confidence,
            )

            return classification

    def handle_error(
        self, error: Exception, request: ConversionRequest
    ) -> ErrorHandlingResult:
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
                recovery_actions = self._generate_recovery_actions(
                    classification, request
                )
                alternative_suggestions = self._generate_alternatives(
                    classification, request
                )
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
                    "handling_time": execution_time,
                    "performance_impact": performance_impact.__dict__,
                },
                next_recommended_action=next_action,
            )

            logger.info(
                "Error handled",
                error_type=classification.error_type.value,
                recovery_actions=len(recovery_actions),
                alternatives=len(alternative_suggestions),
            )

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
                    confidence=0.5,
                ),
                recovery_actions=[],
                alternative_suggestions=[],
                performance_metrics={"handling_time": time.time() - start_time},
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
            context = ConversionContext(source_domain="batch", target_domain="batch")
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
        most_common = sorted(
            error_distribution.items(), key=lambda x: x[1], reverse=True
        )[:5]

        # Generate batch strategies
        batch_strategies = self._generate_batch_strategies(classifications)

        # Calculate aggregate confidence
        confidences = [c.confidence for c in classifications]
        aggregate_confidence = (
            sum(confidences) / len(confidences) if confidences else 0.0
        )

        return ErrorAggregation(
            total_errors=len(errors),
            error_distribution=error_distribution,
            severity_distribution=severity_distribution,
            recoverability_distribution=recoverability_distribution,
            most_common_errors=most_common,
            suggested_batch_strategies=batch_strategies,
            aggregate_confidence=aggregate_confidence,
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

    def execute_recovery_strategy(
        self,
        error_context: ErrorContext,
        strategy: RecoveryStrategy,
        operation: Callable,
        *args,
        **kwargs,
    ) -> Tuple[bool, Any]:
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

        logger.info(
            f"Executing recovery strategy",
            error_id=error_context.error_id,
            strategy=strategy.value,
        )

        try:
            if strategy == RecoveryStrategy.RETRY:
                return self._execute_retry_strategy(
                    error_context, operation, *args, **kwargs
                )

            elif strategy == RecoveryStrategy.FALLBACK:
                return self._execute_fallback_strategy(
                    error_context, operation, *args, **kwargs
                )

            elif strategy == RecoveryStrategy.SKIP:
                return self._execute_skip_strategy(error_context)

            elif strategy == RecoveryStrategy.ALTERNATIVE_PATH:
                return self._execute_alternative_path_strategy(
                    error_context, operation, *args, **kwargs
                )

            else:
                logger.warning(f"Unsupported recovery strategy: {strategy.value}")
                return False, None

        except Exception as e:
            with self._lock:
                self.recovery_stats[strategy]["failed"] += 1

            logger.error(
                f"Recovery strategy failed",
                error_id=error_context.error_id,
                strategy=strategy.value,
                recovery_error=str(e),
            )

            return False, None

    # Enhanced classification and analysis methods

    def _perform_classification(
        self, error: Exception, context: ConversionContext
    ) -> ErrorClassificationEnhanced:
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
        confidence = self._calculate_classification_confidence(
            error, error_type, context
        )

        # Generate recovery strategies
        suggested_strategies = self._suggest_recovery_strategies(
            error_type, severity, recoverability
        )

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
            performance_impact=performance_impact,
        )

    def _infer_error_type(self, error: Exception) -> ConversionError.Type:
        """Infer ConversionError.Type from generic exception."""
        error_str = str(error).lower()
        error_class = type(error).__name__.lower()

        if "memory" in error_str or "memoryerror" in error_class:
            return ConversionError.Type.MEMORY_EXCEEDED
        elif "timeout" in error_str or "timeouterror" in error_class:
            return ConversionError.Type.TIMEOUT
        elif "type" in error_str or "typeerror" in error_class:
            return ConversionError.Type.TYPE_MISMATCH
        elif "value" in error_str or "valueerror" in error_class:
            return ConversionError.Type.SCHEMA_INVALID
        else:
            return ConversionError.Type.CONVERSION_FAILED

    def _assess_severity(
        self,
        error: Exception,
        error_type: ConversionError.Type,
        context: ConversionContext,
    ) -> ErrorSeverity:
        """Assess error severity based on type and context."""

        # Critical severity conditions
        if (
            error_type == ConversionError.Type.MEMORY_EXCEEDED
            or "system" in context.source_domain.lower()
            or "critical" in context.user_intention.lower()
        ):
            return ErrorSeverity.CRITICAL

        # High severity conditions
        if (
            error_type
            in [ConversionError.Type.TIMEOUT, ConversionError.Type.ADAPTER_NOT_FOUND]
            or "primary" in context.user_intention.lower()
        ):
            return ErrorSeverity.HIGH

        # Medium severity conditions
        if (
            error_type
            in [ConversionError.Type.TYPE_MISMATCH, ConversionError.Type.SCHEMA_INVALID]
            or "important" in context.user_intention.lower()
        ):
            return ErrorSeverity.MEDIUM

        # Default to low severity
        return ErrorSeverity.LOW

    def _assess_recoverability(
        self,
        error: Exception,
        error_type: ConversionError.Type,
        context: ConversionContext,
    ) -> ErrorRecoverability:
        """Assess error recoverability."""

        # Terminal conditions
        if (
            error_type == ConversionError.Type.MEMORY_EXCEEDED
            and context.performance_hints.get("memory_critical", False)
        ):
            return ErrorRecoverability.TERMINAL

        # Retryable conditions
        if error_type in [
            ConversionError.Type.TIMEOUT,
            ConversionError.Type.ADAPTER_NOT_FOUND,
        ]:
            return ErrorRecoverability.RETRYABLE

        # Degradable conditions
        if error_type in [
            ConversionError.Type.METADATA_LOSS,
            ConversionError.Type.QUALITY_DEGRADED,
        ]:
            return ErrorRecoverability.DEGRADABLE

        # Default to recoverable
        return ErrorRecoverability.RECOVERABLE

    def _calculate_classification_confidence(
        self,
        error: Exception,
        error_type: ConversionError.Type,
        context: ConversionContext,
    ) -> float:
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

    def _suggest_recovery_strategies(
        self,
        error_type: ConversionError.Type,
        severity: ErrorSeverity,
        recoverability: ErrorRecoverability,
    ) -> List[RecoveryStrategy]:
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

    def _extract_context_factors(
        self, error: Exception, context: ConversionContext
    ) -> Dict[str, Any]:
        """Extract relevant context factors for error analysis."""
        factors = {
            "error_class": type(error).__name__,
            "error_message_length": len(str(error)),
            "has_user_intention": bool(context.user_intention),
            "domain_match": context.source_domain == context.target_domain,
            "has_performance_hints": bool(context.performance_hints),
            "debugging_enabled": context.debugging_enabled,
        }

        # Add performance hint factors
        if context.performance_hints:
            factors.update(
                {
                    f"hint_{k}": v
                    for k, v in context.performance_hints.items()
                    if isinstance(v, (bool, int, float, str))
                }
            )

        return factors

    def _generate_diagnostic_info(
        self, error: Exception, context: ConversionContext
    ) -> Dict[str, Any]:
        """Generate diagnostic information for error analysis."""
        import traceback

        diagnostic = {
            "error_message": str(error),
            "error_type": type(error).__name__,
            "traceback_length": len(traceback.format_exc().split("\n")),
            "context_source_domain": context.source_domain,
            "context_target_domain": context.target_domain,
            "context_intention": context.user_intention,
            "timestamp": time.time(),
        }

        # Add traceback if debugging is enabled
        if context.debugging_enabled:
            diagnostic["full_traceback"] = traceback.format_exc()

        return diagnostic

    def _assess_performance_impact(
        self, error: Exception, request: Optional[ConversionRequest]
    ) -> PerformanceImpact:
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
        if request and hasattr(request, "metadata"):
            pipeline_depth = request.metadata.get("pipeline_depth", 1)
            impact.cascade_risk = min(pipeline_depth * 0.1, 0.8)

        return impact

    def _generate_recovery_actions(
        self, classification: ErrorClassificationEnhanced, request: ConversionRequest
    ) -> List[str]:
        """Generate specific recovery actions."""
        actions = []

        for strategy in classification.suggested_strategies:
            if strategy == RecoveryStrategy.RETRY:
                actions.append(
                    f"Retry conversion with exponential backoff (max 3 attempts)"
                )
            elif strategy == RecoveryStrategy.FALLBACK:
                actions.append("Attempt alternative conversion pathway")
            elif strategy == RecoveryStrategy.DEGRADE:
                actions.append("Accept reduced quality conversion")
            elif strategy == RecoveryStrategy.ROLLBACK:
                actions.append("Rollback to previous stable checkpoint")
            elif strategy == RecoveryStrategy.SKIP:
                actions.append(
                    "Skip this conversion and continue with remaining operations"
                )
            elif strategy == RecoveryStrategy.USER_INTERVENTION:
                actions.append("Escalate to user for manual intervention")

        return actions

    def _generate_alternatives(
        self, classification: ErrorClassificationEnhanced, request: ConversionRequest
    ) -> List[str]:
        """Generate alternative approach suggestions."""
        alternatives = []

        # Based on error type
        if classification.error_type == ConversionError.Type.MEMORY_EXCEEDED:
            alternatives.extend(
                [
                    "Use streaming conversion with smaller chunks",
                    "Temporarily reduce memory footprint of other operations",
                    "Convert to intermediate format with lower memory requirements",
                ]
            )
        elif classification.error_type == ConversionError.Type.TYPE_MISMATCH:
            alternatives.extend(
                [
                    "Apply automatic type coercion before conversion",
                    "Use format-specific preprocessing",
                    "Convert through intermediate compatible format",
                ]
            )
        elif classification.error_type == ConversionError.Type.ADAPTER_NOT_FOUND:
            alternatives.extend(
                [
                    "Use multi-step conversion through supported intermediate formats",
                    "Load additional adapter modules",
                    "Use generic conversion with manual formatting",
                ]
            )

        return alternatives

    def _determine_next_action(
        self, classification: ErrorClassificationEnhanced, request: ConversionRequest
    ) -> Optional[str]:
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

    def _generate_batch_strategies(
        self, classifications: List[ErrorClassificationEnhanced]
    ) -> List[RecoveryStrategy]:
        """Generate recovery strategies for batch operations."""
        # Count strategy occurrences
        strategy_counts = defaultdict(int)
        for classification in classifications:
            for strategy in classification.suggested_strategies:
                strategy_counts[strategy] += 1

        # Sort by frequency and return top strategies
        sorted_strategies = sorted(
            strategy_counts.items(), key=lambda x: x[1], reverse=True
        )
        return [strategy for strategy, _ in sorted_strategies[:3]]

    def _generate_error_key(self, error: Exception, context: ConversionContext) -> str:
        """Generate cache key for error classification."""
        return f"{type(error).__name__}:{hash(str(error))}:{context.source_domain}:{context.target_domain}"

    def _learn_from_error(
        self,
        error: Exception,
        classification: ErrorClassificationEnhanced,
        context: ConversionContext,
    ):
        """Learn from error patterns for improved future classification."""
        self._error_history.append((error, classification))

        # Update error patterns
        error_pattern_key = (
            f"{type(error).__name__}:{context.source_domain}:{context.target_domain}"
        )
        pattern = self._error_patterns[error_pattern_key]
        pattern["count"] = pattern.get("count", 0) + 1
        pattern["last_seen"] = time.time()
        pattern["average_confidence"] = (
            pattern.get("average_confidence", 0.0) * (pattern["count"] - 1)
            + classification.confidence
        ) / pattern["count"]

        # Limit history size
        if len(self._error_history) > 1000:
            self._error_history = self._error_history[-500:]

    def _track_classification_performance(self, classification_time: float):
        """Track error classification performance metrics."""
        baseline_key = "classification_time"

        if baseline_key not in self._performance_baselines:
            self._performance_baselines[baseline_key] = classification_time
        else:
            # Exponential moving average
            alpha = 0.1
            self._performance_baselines[baseline_key] = (
                alpha * classification_time
                + (1 - alpha) * self._performance_baselines[baseline_key]
            )

    def _create_error_context(
        self,
        error: Exception,
        conversion_request: ConversionRequest,
        pipeline_context: Optional[Dict[str, Any]],
    ) -> ErrorContext:
        """Create comprehensive error context."""
        return ErrorContext(
            message=str(error),
            exception=error,
            conversion_request=conversion_request,
            pipeline_id=pipeline_context.get("pipeline_id")
            if pipeline_context
            else None,
            pipeline_step=pipeline_context.get("pipeline_step")
            if pipeline_context
            else None,
            execution_environment={
                "timestamp": time.time(),
                "thread_id": threading.current_thread().ident,
                "process_id": os.getpid(),
            },
            system_state=self._capture_system_state(),
            performance_metrics=pipeline_context.get("performance_metrics", {})
            if pipeline_context
            else {},
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
                "timeout_multiplier": 2.0,
            }

        elif error_context.error_type == ConversionError.Type.MEMORY_EXCEEDED:
            plan.strategies = [
                RecoveryStrategy.ALTERNATIVE_PATH,
                RecoveryStrategy.FALLBACK,
            ]
            plan.strategy_configs[RecoveryStrategy.ALTERNATIVE_PATH] = {
                "prefer_streaming": True,
                "max_memory_mb": 500,
            }

        elif error_context.error_type == ConversionError.Type.ADAPTER_NOT_FOUND:
            plan.strategies = [RecoveryStrategy.ALTERNATIVE_PATH, RecoveryStrategy.SKIP]

        else:
            # Default strategy sequence
            plan.strategies = [
                RecoveryStrategy.RETRY,
                RecoveryStrategy.ALTERNATIVE_PATH,
                RecoveryStrategy.FALLBACK,
            ]

        # Calculate estimates
        plan.estimated_recovery_time = self._estimate_recovery_time(plan)
        plan.estimated_success_probability = self._estimate_success_probability(
            plan, error_context
        )

        return plan

    def _execute_retry_strategy(
        self, error_context: ErrorContext, operation: Callable, *args, **kwargs
    ) -> Tuple[bool, Any]:
        """Execute retry strategy with exponential backoff."""
        for attempt in range(self.max_retry_attempts):
            try:
                # Calculate delay
                if attempt > 0:
                    delay = self._calculate_retry_delay(attempt)
                    logger.debug(
                        f"Retrying after {delay:.2f}s delay",
                        attempt=attempt + 1,
                        max_attempts=self.max_retry_attempts,
                    )
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

    def _execute_fallback_strategy(
        self, error_context: ErrorContext, operation: Callable, *args, **kwargs
    ) -> Tuple[bool, Any]:
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
            "error_id": error_context.error_id,
        }

    def _execute_alternative_path_strategy(
        self, error_context: ErrorContext, operation: Callable, *args, **kwargs
    ) -> Tuple[bool, Any]:
        """Execute alternative path strategy."""
        logger.info("Attempting alternative conversion path")

        # This would typically integrate with AlternativePathwayEngine
        # For now, return indication that alternative path should be tried
        with self._lock:
            self.recovery_stats[RecoveryStrategy.ALTERNATIVE_PATH]["successful"] += 1

        return True, {
            "status": "alternative_path_required",
            "error_id": error_context.error_id,
            "original_request": error_context.conversion_request,
        }

    def _calculate_retry_delay(self, attempt: int) -> float:
        """Calculate retry delay with exponential backoff and jitter."""
        if self.exponential_backoff:
            delay = self.base_retry_delay * (2**attempt)
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
        state = {"timestamp": time.time(), "thread_count": threading.active_count()}

        try:
            import psutil

            process = psutil.Process()
            state.update(
                {
                    "memory_usage_mb": process.memory_info().rss / (1024 * 1024),
                    "cpu_percent": process.cpu_percent(),
                    "open_files": len(process.open_files()),
                }
            )
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
            RecoveryStrategy.ROLLBACK: 3.0,
        }

        total_time = 0.0
        for strategy in plan.strategies:
            total_time += base_times.get(strategy, 5.0)

        return total_time

    def _estimate_success_probability(
        self, plan: RecoveryPlan, error_context: ErrorContext
    ) -> float:
        """Estimate success probability for recovery plan."""
        # Base probabilities by strategy
        base_probabilities = {
            RecoveryStrategy.RETRY: 0.7,
            RecoveryStrategy.FALLBACK: 0.9,
            RecoveryStrategy.SKIP: 1.0,
            RecoveryStrategy.ALTERNATIVE_PATH: 0.8,
            RecoveryStrategy.ROLLBACK: 0.95,
        }

        # Calculate combined probability (assuming independence)
        failure_probability = 1.0
        for strategy in plan.strategies:
            strategy_success_prob = base_probabilities.get(strategy, 0.5)
            failure_probability *= 1.0 - strategy_success_prob

        return 1.0 - failure_probability

    def _generate_fallback_result(self, error_context: ErrorContext) -> Optional[Any]:
        """Generate minimal acceptable fallback result."""
        # This would depend on the specific conversion type
        # Return a basic structure indicating fallback was used
        return {
            "fallback": True,
            "error_context": error_context.error_id,
            "original_source_format": error_context.conversion_request.source_format.value
            if error_context.conversion_request
            else "unknown",
            "original_target_format": error_context.conversion_request.target_format.value
            if error_context.conversion_request
            else "unknown",
        }

    def _generate_troubleshooting_guide(self, error_context: ErrorContext) -> List[str]:
        """Generate troubleshooting guidance based on error context."""
        guide = []

        # General guidance based on error type
        if error_context.error_type == ConversionError.Type.MEMORY_EXCEEDED:
            guide.extend(
                [
                    "Consider using streaming processing for large datasets",
                    "Reduce batch size or implement chunked processing",
                    "Check available system memory and close unnecessary applications",
                    "Consider using sparse data representations",
                ]
            )

        elif error_context.error_type == ConversionError.Type.TIMEOUT:
            guide.extend(
                [
                    "Check network connectivity and service availability",
                    "Increase timeout values in configuration",
                    "Consider implementing asynchronous processing",
                    "Break large operations into smaller chunks",
                ]
            )

        elif error_context.error_type == ConversionError.Type.ADAPTER_NOT_FOUND:
            guide.extend(
                [
                    "Verify that required adapters are installed and available",
                    "Check adapter registry configuration",
                    "Consider alternative data formats or conversion paths",
                    "Update adapter dependencies if needed",
                ]
            )

        else:
            guide.extend(
                [
                    "Review input data format and structure",
                    "Check conversion parameters and configuration",
                    "Verify data quality and completeness",
                    "Consider alternative processing approaches",
                ]
            )

        # Add pattern-based guidance
        error_pattern_key = (
            f"{error_context.error_type}_{error_context.conversion_request.source_format.value}_{error_context.conversion_request.target_format.value}"
            if error_context.conversion_request
            else str(error_context.error_type)
        )

        if error_pattern_key in self.error_patterns:
            recent_errors = self.error_patterns[error_pattern_key][
                -5:
            ]  # Last 5 occurrences
            if len(recent_errors) > 2:
                guide.append(
                    f"This error pattern has occurred {len(recent_errors)} times recently - consider systematic investigation"
                )

        return guide

    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error handling statistics."""
        with self._lock:
            return {
                "total_errors_handled": len(self.error_history),
                "error_patterns": {k: len(v) for k, v in self.error_patterns.items()},
                "recovery_statistics": dict(self.recovery_stats),
                "circuit_breaker_states": {
                    k: v.get_state() for k, v in self.circuit_breakers.items()
                },
                "recent_error_types": [
                    ctx.error_type for ctx in self.error_history[-10:]
                ],
                "average_recovery_attempts": sum(
                    len(ctx.attempted_strategies) for ctx in self.error_history[-100:]
                )
                / min(len(self.error_history), 100)
                if self.error_history
                else 0,
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
                "total_errors_processed": len(self._error_history),
                "cached_classifications": len(self._classification_cache),
                "learned_patterns": len(self._error_patterns),
                "performance_baselines": self._performance_baselines.copy(),
                "most_common_error_types": dict(
                    defaultdict(
                        int,
                        {
                            classification.error_type.value: count
                            for _, classification in self._error_history[-100:]
                            for count in [1]
                        },
                    )
                ),
                "total_errors_handled": len(self.error_history),
                "error_patterns": {k: len(v) for k, v in self.error_patterns.items()},
                "recovery_statistics": dict(self.recovery_stats),
                "circuit_breaker_states": {
                    k: v.get_state() for k, v in self.circuit_breakers.items()
                },
                "recent_error_types": [
                    ctx.error_type for ctx in self.error_history[-10:]
                ],
                "average_recovery_attempts": sum(
                    len(ctx.attempted_strategies) for ctx in self.error_history[-100:]
                )
                / min(len(self.error_history), 100)
                if self.error_history
                else 0,
            }
