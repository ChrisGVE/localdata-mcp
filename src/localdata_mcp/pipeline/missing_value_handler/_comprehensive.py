"""
Missing Value Handler - Comprehensive Assessment

Cross-validation of strategies, ensemble imputation, comprehensive quality
metrics, coverage statistics, data integrity, and artifact detection.
"""

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from ...logging_manager import get_logger

logger = get_logger(__name__)


class ComprehensiveAssessmentMixin:
    """Mixin providing comprehensive analysis and cross-validation methods."""

    def _evaluate_all_strategies(
        self, data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Evaluate all available imputation strategies."""
        strategies = ["simple", "knn", "iterative"]
        evaluation_results = {}

        for strategy in strategies:
            try:
                # Apply strategy
                if strategy == "simple":
                    imputed_data, strategy_metadata = self._simple_imputation(data)
                elif strategy == "knn":
                    imputed_data, strategy_metadata = self._knn_imputation(data)
                elif strategy == "iterative":
                    imputed_data, strategy_metadata = self._iterative_imputation(data)

                # Assess quality
                _, quality_metadata = self._assess_imputation_quality(imputed_data)

                evaluation_results[strategy] = {
                    "strategy_metadata": strategy_metadata,
                    "quality_metadata": quality_metadata,
                    "overall_score": quality_metadata.get("overall_quality_score", 0),
                }

            except Exception as e:
                evaluation_results[strategy] = {"error": str(e), "overall_score": 0}

        # Select best strategy
        best_strategy = max(
            evaluation_results.keys(),
            key=lambda k: evaluation_results[k].get("overall_score", 0),
        )

        metadata = {
            "evaluation_results": evaluation_results,
            "best_strategy": best_strategy,
            "best_score": evaluation_results[best_strategy].get("overall_score", 0),
        }

        # Store best strategy selection
        self.custom_parameters["evaluated_best_strategy"] = best_strategy

        return data.copy(), metadata

    def _cross_validate_strategies(
        self, data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Cross-validate imputation strategies for robust quality assessment."""
        if not self.cross_validation:
            return data.copy(), {"cross_validation": "disabled"}

        numeric_cols = data.select_dtypes(include=["number"]).columns
        cv_results = {}

        # Only cross-validate on numeric columns with sufficient data
        testable_cols = [
            col
            for col in numeric_cols
            if data[col].isnull().sum() > 0
            and data[col].isnull().sum() < len(data) * 0.5
        ]

        if len(testable_cols) == 0:
            return data.copy(), {"cross_validation": "no_suitable_columns"}

        strategies = ["simple", "knn", "iterative"]

        for strategy in strategies:
            strategy_scores = []

            for col in testable_cols[:3]:  # Limit to first 3 columns for performance
                try:
                    # Create artificial missing values for testing
                    col_scores = self._cross_validate_column(data, col, strategy)
                    strategy_scores.extend(col_scores)
                except Exception as e:
                    logger.warning(
                        f"Cross-validation failed for {strategy} on {col}: {e}"
                    )

            if strategy_scores:
                cv_results[strategy] = {
                    "mean_score": np.mean(strategy_scores),
                    "std_score": np.std(strategy_scores),
                    "scores": strategy_scores,
                }

        metadata = {
            "cross_validation_results": cv_results,
            "tested_columns": testable_cols[:3],
            "best_cv_strategy": (
                max(cv_results.keys(), key=lambda k: cv_results[k]["mean_score"])
                if cv_results
                else None
            ),
        }

        return data.copy(), metadata

    def _cross_validate_column(
        self, data: pd.DataFrame, column: str, strategy: str
    ) -> List[float]:
        """Cross-validate imputation for a single column."""
        # Get complete cases for this column
        complete_data = data[data[column].notna()]

        if len(complete_data) < 20:  # Need minimum data for CV
            return [0.5]

        # 5-fold cross-validation
        kf = KFold(
            n_splits=min(5, len(complete_data) // 4), shuffle=True, random_state=42
        )
        scores = []

        for train_idx, test_idx in kf.split(complete_data):
            try:
                # Create train/test split
                train_data = complete_data.iloc[train_idx].copy()
                test_data = complete_data.iloc[test_idx].copy()

                # Artificially remove values from test set
                test_values = test_data[column].copy()
                test_data[column] = np.nan

                # Combine train and test data for imputation
                cv_data = pd.concat([train_data, test_data])

                # Apply imputation strategy
                if strategy == "simple":
                    imputed_data, _ = self._simple_imputation(cv_data)
                elif strategy == "knn":
                    imputed_data, _ = self._knn_imputation(cv_data)
                elif strategy == "iterative":
                    imputed_data, _ = self._iterative_imputation(cv_data)

                # Extract imputed values for test set
                imputed_test_values = imputed_data[column].iloc[len(train_data) :]

                # Calculate accuracy (correlation with true values)
                if len(imputed_test_values) > 1 and len(test_values) > 1:
                    correlation = np.corrcoef(imputed_test_values, test_values)[0, 1]
                    scores.append(max(0, correlation))

            except Exception as e:
                scores.append(0.0)  # Failed fold

        return scores

    def _ensemble_imputation(
        self, data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Ensemble imputation combining multiple strategies."""
        # Get evaluation results
        eval_strategy = self.custom_parameters.get("evaluated_best_strategy", "simple")
        cv_strategy = self.custom_parameters.get("best_cv_strategy")

        # Use the best validated strategy
        final_strategy = cv_strategy if cv_strategy else eval_strategy

        # Apply the selected strategy
        if final_strategy == "simple":
            return self._simple_imputation(data)
        elif final_strategy == "knn":
            return self._knn_imputation(data)
        elif final_strategy == "iterative":
            return self._iterative_imputation(data)
        else:
            return self._simple_imputation(data)  # Fallback

    def _comprehensive_quality_assessment(
        self, data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Comprehensive quality assessment with detailed metrics."""
        _, basic_quality = self._assess_imputation_quality(data)

        # Additional comprehensive metrics
        comprehensive_metrics = {
            "basic_quality": basic_quality,
            "imputation_coverage": self._calculate_imputation_coverage(data),
            "data_integrity": self._assess_data_integrity(data),
            "imputation_artifacts": self._detect_imputation_artifacts(data),
        }

        return data.copy(), comprehensive_metrics

    def _calculate_imputation_coverage(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate imputation coverage statistics."""
        if self._original_data is None:
            return {"coverage": "no_original_data"}

        original_missing = self._original_data.isnull().sum().sum()
        current_missing = data.isnull().sum().sum()

        coverage = {
            "original_missing_values": original_missing,
            "remaining_missing_values": current_missing,
            "imputation_rate": (
                (original_missing - current_missing) / original_missing
                if original_missing > 0
                else 1.0
            ),
            "complete_imputation": current_missing == 0,
        }

        return coverage

    def _assess_data_integrity(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Assess data integrity after imputation."""
        integrity_checks = {}

        # Check for reasonable value ranges
        numeric_cols = data.select_dtypes(include=["number"]).columns
        for col in numeric_cols:
            if self._original_data is not None and col in self._original_data.columns:
                orig_min, orig_max = (
                    self._original_data[col].min(),
                    self._original_data[col].max(),
                )
                curr_min, curr_max = data[col].min(), data[col].max()

                integrity_checks[col] = {
                    "values_within_original_range": curr_min >= orig_min
                    and curr_max <= orig_max,
                    "range_expansion": (
                        (curr_max - curr_min) / (orig_max - orig_min)
                        if orig_max != orig_min
                        else 1.0
                    ),
                }

        return integrity_checks

    def _detect_imputation_artifacts(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect potential artifacts from imputation."""
        artifacts = {}

        # Check for repeated values (potential over-imputation)
        for col in data.columns:
            if self._original_data is not None and col in self._original_data.columns:
                if self._original_data[col].isnull().sum() > 0:
                    # Check for artificial concentration of values
                    value_counts = data[col].value_counts()
                    most_common_freq = (
                        value_counts.iloc[0] / len(data) if len(value_counts) > 0 else 0
                    )

                    artifacts[col] = {
                        "high_concentration_detected": most_common_freq > 0.3,
                        "most_common_frequency": most_common_freq,
                        "unique_values_ratio": data[col].nunique() / len(data),
                    }

        return artifacts
