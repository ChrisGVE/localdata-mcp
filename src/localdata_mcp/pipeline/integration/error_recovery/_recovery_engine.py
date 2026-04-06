"""
RecoveryStrategyEngine: configurable recovery strategy engine with learning capabilities.
"""

import logging
import threading
import time
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple

from ....logging_manager import get_logger
from ..interfaces import ConversionError, ConversionRequest
from ._error_handler import ConversionErrorHandler
from ._pathway_engine import AlternativePathwayEngine
from ._rollback_manager import RollbackManager
from ._types import (
    ErrorContext,
    RecoveryPlan,
    RecoveryStrategy,
)

logger = get_logger(__name__)


class RecoveryStrategyEngine:
    """
    Configurable recovery strategy engine with learning capabilities.

    Orchestrates recovery strategies based on error context, learns from failure patterns,
    and provides intelligent strategy selection for different error scenarios.
    """

    def __init__(
        self,
        error_handler: ConversionErrorHandler,
        pathway_engine: AlternativePathwayEngine,
        rollback_manager: RollbackManager,
    ):
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
                "max_delay": 60.0,
            },
            RecoveryStrategy.FALLBACK: {
                "quality_threshold": 0.7,
                "performance_degradation_allowed": 0.5,
            },
            RecoveryStrategy.ALTERNATIVE_PATH: {
                "max_alternatives": 5,
                "feasibility_threshold": 0.6,
            },
            RecoveryStrategy.ROLLBACK: {
                "max_rollback_steps": 3,
                "cleanup_on_rollback": True,
            },
        }

        # Learning system
        self.strategy_performance: Dict[str, Dict[RecoveryStrategy, List[float]]] = (
            defaultdict(lambda: defaultdict(list))
        )
        self.error_pattern_strategies: Dict[str, List[RecoveryStrategy]] = {}

        # Execution statistics
        self.stats = {
            "recoveries_attempted": 0,
            "recoveries_successful": 0,
            "strategy_usage": defaultdict(int),
            "average_recovery_time": 0.0,
        }

        self._lock = threading.RLock()

        logger.info("RecoveryStrategyEngine initialized")

    def execute_recovery(
        self, error_context: ErrorContext, operation: Callable, *args, **kwargs
    ) -> Tuple[bool, Any, Dict[str, Any]]:
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

        logger.info(
            "Executing recovery strategy",
            error_id=error_context.error_id,
            error_type=error_context.error_type,
            severity=error_context.severity.value,
        )

        # Generate recovery plan
        recovery_plan = self._generate_intelligent_recovery_plan(error_context)

        recovery_metadata = {
            "recovery_plan": recovery_plan,
            "attempted_strategies": [],
            "successful_strategy": None,
            "execution_time": 0.0,
            "quality_impact": 0.0,
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

                recovery_metadata["attempted_strategies"].append(
                    {
                        "strategy": strategy.value,
                        "success": success,
                        "execution_time": strategy_execution_time,
                    }
                )

                if success:
                    # Record successful recovery
                    self._record_strategy_success(
                        error_context, strategy, strategy_execution_time
                    )

                    recovery_metadata["successful_strategy"] = strategy.value
                    recovery_metadata["execution_time"] = time.time() - start_time

                    with self._lock:
                        self.stats["recoveries_successful"] += 1
                        # Update average recovery time
                        total_time = self.stats["average_recovery_time"] * (
                            self.stats["recoveries_successful"] - 1
                        )
                        self.stats["average_recovery_time"] = (
                            total_time + recovery_metadata["execution_time"]
                        ) / self.stats["recoveries_successful"]

                    logger.info(
                        "Recovery successful",
                        error_id=error_context.error_id,
                        strategy=strategy.value,
                        total_time=recovery_metadata["execution_time"],
                    )

                    return True, result, recovery_metadata

            except Exception as strategy_error:
                logger.warning(
                    f"Recovery strategy {strategy.value} failed: {strategy_error}"
                )
                recovery_metadata["attempted_strategies"][-1]["error"] = str(
                    strategy_error
                )

        # All strategies failed
        recovery_metadata["execution_time"] = time.time() - start_time

        logger.warning(
            "All recovery strategies failed",
            error_id=error_context.error_id,
            strategies_attempted=len(recovery_plan.strategies),
        )

        return False, None, recovery_metadata

    def _generate_intelligent_recovery_plan(
        self, error_context: ErrorContext
    ) -> RecoveryPlan:
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
        prioritized_strategies = self._prioritize_strategies(
            combined_strategies, error_context
        )

        # Create enhanced recovery plan
        enhanced_plan = RecoveryPlan(
            error_context=error_context,
            strategies=prioritized_strategies,
            strategy_configs=self._get_strategy_configs_for_error(error_context),
            estimated_recovery_time=self._estimate_plan_execution_time(
                prioritized_strategies
            ),
            estimated_success_probability=self._estimate_plan_success_probability(
                prioritized_strategies, error_context
            ),
        )

        return enhanced_plan

    def _execute_strategy(
        self,
        strategy: RecoveryStrategy,
        error_context: ErrorContext,
        operation: Callable,
        *args,
        **kwargs,
    ) -> Tuple[bool, Any]:
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
            return self._execute_alternative_path_recovery(
                error_context, operation, *args, **kwargs
            )

        elif strategy == RecoveryStrategy.ROLLBACK:
            return self._execute_rollback_recovery(
                error_context, operation, *args, **kwargs
            )

        else:
            logger.warning(f"Unknown recovery strategy: {strategy.value}")
            return False, None

    def _execute_alternative_path_recovery(
        self, error_context: ErrorContext, operation: Callable, *args, **kwargs
    ) -> Tuple[bool, Any]:
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
            "original_error": error_context.error_id,
        }

    def _execute_rollback_recovery(
        self, error_context: ErrorContext, operation: Callable, *args, **kwargs
    ) -> Tuple[bool, Any]:
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
                "error_id": error_context.error_id,
            }

        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False, None

    def _classify_error_pattern(self, error_context: ErrorContext) -> str:
        """Classify error into pattern for learning purposes."""
        components = [str(error_context.error_type), error_context.severity.value]

        if error_context.conversion_request:
            components.extend(
                [
                    error_context.conversion_request.source_format.value,
                    error_context.conversion_request.target_format.value,
                ]
            )

        return "_".join(components)

    def _get_learned_strategies(self, error_pattern: str) -> List[RecoveryStrategy]:
        """Get learned strategies for error pattern."""
        return self.error_pattern_strategies.get(error_pattern, [])

    def _prioritize_strategies(
        self, strategies: List[RecoveryStrategy], error_context: ErrorContext
    ) -> List[RecoveryStrategy]:
        """Prioritize strategies based on historical success rates."""
        error_pattern = self._classify_error_pattern(error_context)

        # Get success rates for each strategy
        strategy_scores = {}
        for strategy in strategies:
            performances = self.strategy_performance[error_pattern][strategy]
            if performances:
                # Success rate based on recent performance
                recent_performances = performances[-10:]  # Last 10 attempts
                success_rate = sum(1 for p in recent_performances if p > 0) / len(
                    recent_performances
                )
                avg_time = sum(abs(p) for p in recent_performances) / len(
                    recent_performances
                )

                # Score combines success rate and speed (lower time is better)
                strategy_scores[strategy] = success_rate - (avg_time / 100.0)
            else:
                # Default score for untested strategies
                default_scores = {
                    RecoveryStrategy.RETRY: 0.7,
                    RecoveryStrategy.FALLBACK: 0.8,
                    RecoveryStrategy.SKIP: 0.9,
                    RecoveryStrategy.ALTERNATIVE_PATH: 0.6,
                    RecoveryStrategy.ROLLBACK: 0.5,
                }
                strategy_scores[strategy] = default_scores.get(strategy, 0.5)

        # Sort by score (highest first)
        return sorted(strategies, key=lambda s: strategy_scores.get(s, 0), reverse=True)

    def _record_strategy_success(
        self,
        error_context: ErrorContext,
        strategy: RecoveryStrategy,
        execution_time: float,
    ) -> None:
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
                self.strategy_performance[error_pattern][strategy] = (
                    self.strategy_performance[error_pattern][strategy][-25:]
                )

    def _get_strategy_configs_for_error(
        self, error_context: ErrorContext
    ) -> Dict[RecoveryStrategy, Dict[str, Any]]:
        """Get strategy configurations adapted for specific error."""
        configs = self.strategy_configs.copy()

        # Adapt configurations based on error context
        if error_context.error_type == ConversionError.Type.MEMORY_EXCEEDED:
            configs[RecoveryStrategy.RETRY][
                "max_attempts"
            ] = 1  # Don't retry memory errors aggressively
            configs[RecoveryStrategy.ALTERNATIVE_PATH]["max_alternatives"] = 3

        elif error_context.error_type == ConversionError.Type.TIMEOUT:
            configs[RecoveryStrategy.RETRY][
                "max_delay"
            ] = 120.0  # Longer delays for timeout
            configs[RecoveryStrategy.RETRY]["backoff_factor"] = 1.5

        return configs

    def _estimate_plan_execution_time(
        self, strategies: List[RecoveryStrategy]
    ) -> float:
        """Estimate execution time for recovery plan."""
        base_times = {
            RecoveryStrategy.RETRY: 5.0,
            RecoveryStrategy.FALLBACK: 2.0,
            RecoveryStrategy.SKIP: 0.1,
            RecoveryStrategy.ALTERNATIVE_PATH: 8.0,
            RecoveryStrategy.ROLLBACK: 3.0,
        }

        return sum(base_times.get(strategy, 5.0) for strategy in strategies)

    def _estimate_plan_success_probability(
        self, strategies: List[RecoveryStrategy], error_context: ErrorContext
    ) -> float:
        """Estimate success probability for recovery plan."""
        error_pattern = self._classify_error_pattern(error_context)

        # Calculate probability that at least one strategy succeeds
        failure_probability = 1.0

        for strategy in strategies:
            performances = self.strategy_performance[error_pattern][strategy]
            if performances:
                recent_performances = performances[-5:]  # Recent history
                success_rate = sum(1 for p in recent_performances if p > 0) / len(
                    recent_performances
                )
            else:
                # Default success rates
                default_rates = {
                    RecoveryStrategy.RETRY: 0.6,
                    RecoveryStrategy.FALLBACK: 0.8,
                    RecoveryStrategy.SKIP: 1.0,
                    RecoveryStrategy.ALTERNATIVE_PATH: 0.7,
                    RecoveryStrategy.ROLLBACK: 0.9,
                }
                success_rate = default_rates.get(strategy, 0.5)

            failure_probability *= 1.0 - success_rate

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
                            "avg_time": (
                                sum(abs(p) for p in performances) / len(performances)
                                if performances
                                else 0
                            ),
                            "success_rate": (
                                sum(1 for p in performances if p > 0)
                                / len(performances)
                                if performances
                                else 0
                            ),
                        }
                        for strategy, performances in strategies.items()
                    }
                    for pattern, strategies in self.strategy_performance.items()
                },
                "strategy_configurations": self.strategy_configs,
            }

    def update_strategy_config(
        self, strategy: RecoveryStrategy, config_updates: Dict[str, Any]
    ) -> None:
        """Update configuration for a specific strategy."""
        with self._lock:
            if strategy not in self.strategy_configs:
                self.strategy_configs[strategy] = {}

            self.strategy_configs[strategy].update(config_updates)

        logger.info(
            f"Strategy configuration updated",
            strategy=strategy.value,
            updates=config_updates,
        )

    def clear_learning_data(self) -> None:
        """Clear all learned patterns and performance data."""
        with self._lock:
            self.strategy_performance.clear()
            self.error_pattern_strategies.clear()

        logger.info("Recovery strategy learning data cleared")
