"""
Missing Value Handler - Strategy Selection

Intelligent strategy selection, rationale generation, and strategy dispatch.
"""

from typing import Any, Dict, Tuple

import pandas as pd

from ...logging_manager import get_logger
from ._types import MissingValuePattern

logger = get_logger(__name__)


class StrategySelectionMixin:
    """Mixin providing strategy selection and rationale methods."""

    def _intelligent_strategy_selection(
        self, data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Intelligently select optimal imputation strategy based on data characteristics."""
        if self._missing_pattern is None:
            # Run pattern analysis if not already done
            _, _ = self._analyze_missing_patterns(data)

        selected_strategy = self._select_optimal_strategy(data, self._missing_pattern)

        metadata = {
            "selected_strategy": selected_strategy,
            "selection_rationale": self._get_strategy_rationale(
                selected_strategy, self._missing_pattern
            ),
            "confidence": self._calculate_strategy_confidence(
                selected_strategy, self._missing_pattern
            ),
        }

        # Store selected strategy for next step
        self.custom_parameters["auto_selected_strategy"] = selected_strategy

        return data.copy(), metadata

    def _select_optimal_strategy(
        self, data: pd.DataFrame, pattern: MissingValuePattern
    ) -> str:
        """Select optimal imputation strategy based on missing value pattern analysis."""

        # Factor 1: Missing percentage
        overall_missing = pattern.missing_percentage

        # Factor 2: Data size and complexity
        n_rows, n_cols = data.shape
        numeric_cols = len(data.select_dtypes(include=["number"]).columns)

        # Factor 3: Pattern type
        pattern_type = pattern.pattern_type

        # Factor 4: Computational constraints
        is_large_dataset = n_rows > 50000 or n_cols > 100

        # Decision logic
        if overall_missing < 5 and not is_large_dataset:
            return "simple"
        elif pattern_type == "MCAR" and overall_missing < 15:
            return "knn" if not is_large_dataset else "simple"
        elif pattern_type == "MAR" and numeric_cols > 2 and not is_large_dataset:
            return "iterative"
        elif pattern_type == "MNAR" or overall_missing > 30:
            return "custom"  # Requires domain knowledge
        elif is_large_dataset:
            return "knn"  # Good balance of quality and speed
        else:
            return "iterative"  # Default advanced method

    def _get_strategy_rationale(
        self, strategy: str, pattern: MissingValuePattern
    ) -> Dict[str, Any]:
        """Get rationale for strategy selection."""
        rationale = {
            "strategy": strategy,
            "factors": {
                "missing_percentage": pattern.missing_percentage,
                "pattern_type": pattern.pattern_type,
                "pattern_confidence": pattern.confidence_score,
                "recommendations_followed": [],
            },
        }

        # Match selected strategy to recommendations
        for rec in pattern.recommendations:
            if strategy.upper() in rec.upper() or strategy in rec.lower():
                rationale["factors"]["recommendations_followed"].append(rec)

        return rationale

    def _calculate_strategy_confidence(
        self, strategy: str, pattern: MissingValuePattern
    ) -> float:
        """Calculate confidence in strategy selection."""
        base_confidence = pattern.confidence_score

        # Adjust based on strategy-pattern alignment
        pattern_type = pattern.pattern_type
        missing_pct = pattern.missing_percentage

        # Strategy-specific confidence adjustments
        adjustments = {
            "simple": 0.9 if missing_pct < 5 else 0.6,
            "knn": 0.8 if pattern_type in ["MCAR", "MAR"] else 0.5,
            "iterative": 0.9 if pattern_type == "MAR" else 0.7,
            "custom": 0.6,  # Always uncertain without domain knowledge
        }

        strategy_confidence = adjustments.get(strategy, 0.5)

        # Combine pattern confidence with strategy confidence
        combined_confidence = (base_confidence + strategy_confidence) / 2

        return max(0.0, min(1.0, combined_confidence))

    def _apply_selected_strategy(
        self, data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Apply the selected imputation strategy."""
        selected_strategy = self.custom_parameters.get(
            "auto_selected_strategy", "simple"
        )

        if selected_strategy == "simple":
            return self._simple_imputation(data)
        elif selected_strategy == "knn":
            return self._knn_imputation(data)
        elif selected_strategy == "iterative":
            return self._iterative_imputation(data)
        elif selected_strategy == "custom":
            return self._custom_imputation(data)
        else:
            # Fallback to simple
            return self._simple_imputation(data)
