"""Error recovery strategies and execution.

Provides RecoveryStrategy, RecoveryAction, and ErrorRecoveryManager.
"""

import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..logging_manager import get_logging_manager, get_logger
from .exceptions import (
    ErrorCategory,
    LocalDataError,
)

# Get structured logger
logger = get_logger(__name__)


# ============================================================================
# Error Recovery Strategies
# ============================================================================


class RecoveryStrategy(Enum):
    """Types of recovery strategies."""

    CONNECTION_RESET = "connection_reset"
    QUERY_SIMPLIFICATION = "query_simplification"
    RESULT_PAGINATION = "result_pagination"
    CACHE_FALLBACK = "cache_fallback"
    READ_REPLICA = "read_replica"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    PARTIAL_RESULTS = "partial_results"


@dataclass
class RecoveryAction:
    """Represents a recovery action that can be taken."""

    strategy: RecoveryStrategy
    description: str
    action_function: Callable
    metadata: Dict[str, Any] = field(default_factory=dict)
    success_probability: float = 0.5
    estimated_time: float = 0.0

    def execute(self, context: Dict[str, Any]) -> Tuple[bool, Any, Optional[str]]:
        """Execute the recovery action.

        Returns:
            Tuple[bool, Any, Optional[str]]: (success, result, error_message)
        """
        try:
            result = self.action_function(context)
            return True, result, None
        except Exception as e:
            return False, None, str(e)


class ErrorRecoveryManager:
    """Manages error recovery strategies and execution."""

    def __init__(self):
        self._recovery_strategies: Dict[ErrorCategory, List[RecoveryAction]] = (
            defaultdict(list)
        )
        self._recovery_history: deque = deque(maxlen=1000)
        self._lock = threading.RLock()
        self._register_default_strategies()

    def register_strategy(self, category: ErrorCategory, action: RecoveryAction):
        """Register a recovery strategy for an error category."""
        with self._lock:
            self._recovery_strategies[category].append(action)

    def get_recovery_options(self, error: LocalDataError) -> List[RecoveryAction]:
        """Get applicable recovery options for an error."""
        with self._lock:
            return self._recovery_strategies.get(error.category, []).copy()

    def attempt_recovery(
        self, error: LocalDataError, context: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, Any, List[str]]:
        """Attempt recovery from an error using available strategies.

        Returns:
            Tuple[bool, Any, List[str]]: (recovered, result, attempted_strategies)
        """
        context = context or {}
        context.update(
            {
                "error": error,
                "database_name": error.database_name,
                "query": error.query,
                "metadata": error.metadata,
            }
        )

        recovery_options = self.get_recovery_options(error)
        attempted_strategies = []

        # Sort by success probability (highest first)
        recovery_options.sort(key=lambda x: x.success_probability, reverse=True)

        for action in recovery_options:
            attempted_strategies.append(action.strategy.value)

            try:
                success, result, error_msg = action.execute(context)

                # Record recovery attempt
                recovery_record = {
                    "timestamp": time.time(),
                    "error_code": error.error_code,
                    "error_category": error.category.value,
                    "strategy": action.strategy.value,
                    "success": success,
                    "error_message": error_msg,
                    "database_name": error.database_name,
                }

                with self._lock:
                    self._recovery_history.append(recovery_record)

                if success:
                    logging_manager = get_logging_manager()
                    with logging_manager.context(
                        operation="error_recovery_success",
                        component="error_handler",
                        error_code=error.error_code,
                    ):
                        logger.info(
                            "Error recovery successful",
                            strategy=action.strategy.value,
                            attempt_duration=recovery_record["duration"],
                            attempts_before_success=len(attempted_strategies),
                        )
                    return True, result, attempted_strategies
                else:
                    logging_manager = get_logging_manager()
                    with logging_manager.context(
                        operation="error_recovery_attempt",
                        component="error_handler",
                        error_code=error.error_code,
                    ):
                        logger.warning(
                            "Error recovery attempt failed",
                            strategy=action.strategy.value,
                            error_message=error_msg,
                            attempt_duration=recovery_record["duration"],
                        )

            except Exception as recovery_error:
                logging_manager = get_logging_manager()
                logging_manager.log_error(
                    recovery_error,
                    "error_handler",
                    strategy=action.strategy.value,
                    original_error=error.error_code,
                )

        logging_manager = get_logging_manager()
        logging_manager.log_error(
            Exception(f"All recovery strategies exhausted for {error.error_code}"),
            "error_handler",
            error_code=error.error_code,
            attempted_strategies=attempted_strategies,
            strategy_count=len(attempted_strategies),
        )
        return False, None, attempted_strategies

    def get_recovery_statistics(self) -> Dict[str, Any]:
        """Get statistics about recovery attempts."""
        with self._lock:
            if not self._recovery_history:
                return {
                    "total_attempts": 0,
                    "success_rate": 0.0,
                    "strategy_stats": {},
                    "category_stats": {},
                }

            total_attempts = len(self._recovery_history)
            successful_attempts = sum(
                1 for record in self._recovery_history if record["success"]
            )
            success_rate = (successful_attempts / total_attempts) * 100

            # Strategy statistics
            strategy_stats = defaultdict(lambda: {"attempts": 0, "successes": 0})
            category_stats = defaultdict(lambda: {"attempts": 0, "successes": 0})

            for record in self._recovery_history:
                strategy = record["strategy"]
                category = record["error_category"]

                strategy_stats[strategy]["attempts"] += 1
                category_stats[category]["attempts"] += 1

                if record["success"]:
                    strategy_stats[strategy]["successes"] += 1
                    category_stats[category]["successes"] += 1

            # Calculate success rates
            for stats in strategy_stats.values():
                if stats["attempts"] > 0:
                    stats["success_rate"] = (
                        stats["successes"] / stats["attempts"]
                    ) * 100
                else:
                    stats["success_rate"] = 0.0

            for stats in category_stats.values():
                if stats["attempts"] > 0:
                    stats["success_rate"] = (
                        stats["successes"] / stats["attempts"]
                    ) * 100
                else:
                    stats["success_rate"] = 0.0

            return {
                "total_attempts": total_attempts,
                "successful_attempts": successful_attempts,
                "success_rate": success_rate,
                "strategy_stats": dict(strategy_stats),
                "category_stats": dict(category_stats),
            }

    def _register_default_strategies(self):
        """Register default recovery strategies."""

        # Connection recovery strategies
        def reset_connection(context: Dict[str, Any]) -> str:
            """Reset database connection."""
            database_name = context.get("database_name")
            if not database_name:
                raise ValueError("Database name required for connection reset")

            # This would integrate with ConnectionManager
            # For now, return a placeholder message
            return f"Connection reset initiated for database: {database_name}"

        self.register_strategy(
            ErrorCategory.CONNECTION,
            RecoveryAction(
                strategy=RecoveryStrategy.CONNECTION_RESET,
                description="Reset database connection and retry",
                action_function=reset_connection,
                success_probability=0.7,
                estimated_time=5.0,
            ),
        )

        # Query simplification for resource exhaustion
        def simplify_query(context: Dict[str, Any]) -> str:
            """Suggest query simplification."""
            query = context.get("query", "")
            suggestions = []

            if "ORDER BY" in query.upper():
                suggestions.append("Remove or simplify ORDER BY clause")
            if "GROUP BY" in query.upper():
                suggestions.append("Consider pre-aggregated tables")
            if "JOIN" in query.upper():
                suggestions.append("Reduce number of JOINs or use EXISTS instead")
            if "DISTINCT" in query.upper():
                suggestions.append("Remove DISTINCT if possible")

            if not suggestions:
                suggestions = ["Add LIMIT clause to reduce result set size"]

            return f"Query simplification suggestions: {'; '.join(suggestions)}"

        self.register_strategy(
            ErrorCategory.RESOURCE_EXHAUSTION,
            RecoveryAction(
                strategy=RecoveryStrategy.QUERY_SIMPLIFICATION,
                description="Suggest query optimizations to reduce resource usage",
                action_function=simplify_query,
                success_probability=0.6,
                estimated_time=0.1,
            ),
        )

        # Result pagination for large datasets
        def suggest_pagination(context: Dict[str, Any]) -> Dict[str, Any]:
            """Suggest result set pagination."""
            query = context.get("query", "")

            # Simple heuristic for pagination
            suggested_limit = 1000
            if "LIMIT" not in query.upper():
                pagination_query = f"{query.rstrip(';')} LIMIT {suggested_limit}"
            else:
                pagination_query = query  # Already has LIMIT

            return {
                "strategy": "pagination",
                "suggested_query": pagination_query,
                "suggested_limit": suggested_limit,
                "message": f"Consider paginating results with LIMIT {suggested_limit}",
            }

        self.register_strategy(
            ErrorCategory.RESOURCE_EXHAUSTION,
            RecoveryAction(
                strategy=RecoveryStrategy.RESULT_PAGINATION,
                description="Suggest result pagination to reduce memory usage",
                action_function=suggest_pagination,
                success_probability=0.8,
                estimated_time=0.1,
            ),
        )

        # Partial results for timeout errors
        def partial_results_strategy(context: Dict[str, Any]) -> Dict[str, Any]:
            """Suggest partial results approach."""
            return {
                "strategy": "partial_results",
                "message": "Consider using streaming execution or reducing query scope",
                "suggestions": [
                    "Use streaming query execution for large results",
                    "Add time-based filters to reduce data scope",
                    "Consider pre-computed aggregations",
                    "Use sampling for approximate results",
                ],
            }

        self.register_strategy(
            ErrorCategory.TIMEOUT,
            RecoveryAction(
                strategy=RecoveryStrategy.PARTIAL_RESULTS,
                description="Suggest partial results approaches for timeout issues",
                action_function=partial_results_strategy,
                success_probability=0.5,
                estimated_time=0.1,
            ),
        )
