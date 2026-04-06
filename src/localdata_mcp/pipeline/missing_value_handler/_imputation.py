"""
Missing Value Handler - Concrete Imputation Implementations

Simple, KNN, iterative, and custom imputation strategy implementations.
"""

import time
import tracemalloc
from typing import Any, Dict, Tuple

import pandas as pd
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer

from ...logging_manager import get_logger

logger = get_logger(__name__)


class ImputationStrategyMixin:
    """Mixin providing concrete imputation strategy implementations."""

    def _simple_imputation(
        self, data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Simple imputation using median/mode strategies."""
        result_data = data.copy()
        imputation_log = {}
        start_time = time.time()

        # Start memory tracking
        tracemalloc.start()

        # Numeric columns - median imputation
        numeric_cols = data.select_dtypes(include=["number"]).columns
        for col in numeric_cols:
            if data[col].isnull().sum() > 0:
                imputer = SimpleImputer(strategy="median")
                result_data[col] = imputer.fit_transform(data[[col]]).ravel()
                self._fitted_imputers[f"{col}_simple"] = imputer
                imputation_log[col] = {
                    "strategy": "simple_median",
                    "missing_before": data[col].isnull().sum(),
                    "missing_after": result_data[col].isnull().sum(),
                    "imputed_value": imputer.statistics_[0],
                }

        # Categorical columns - most frequent imputation
        categorical_cols = data.select_dtypes(include=["object", "category"]).columns
        for col in categorical_cols:
            if data[col].isnull().sum() > 0:
                imputer = SimpleImputer(strategy="most_frequent")
                result_data[col] = imputer.fit_transform(data[[col]]).ravel()
                self._fitted_imputers[f"{col}_simple"] = imputer
                imputation_log[col] = {
                    "strategy": "simple_mode",
                    "missing_before": data[col].isnull().sum(),
                    "missing_after": result_data[col].isnull().sum(),
                    "imputed_value": imputer.statistics_[0],
                }

        # Calculate memory usage
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        execution_time = time.time() - start_time

        metadata = {
            "strategy": "simple",
            "imputation_log": imputation_log,
            "execution_time": execution_time,
            "memory_usage_mb": peak / (1024 * 1024),
            "total_imputed_values": sum(
                log["missing_before"] - log["missing_after"]
                for log in imputation_log.values()
            ),
        }

        return result_data, metadata

    def _knn_imputation(
        self, data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """KNN-based imputation for multivariate missing value handling."""
        result_data = data.copy()
        imputation_log = {}
        start_time = time.time()

        tracemalloc.start()

        # Handle numeric columns with KNN
        numeric_cols = data.select_dtypes(include=["number"]).columns
        n_neighbors = self.strategy_configs["knn"]["n_neighbors"]
        if len(numeric_cols) > 0 and any(
            data[col].isnull().sum() > 0 for col in numeric_cols
        ):
            # Determine optimal number of neighbors
            n_neighbors = min(
                self.strategy_configs["knn"]["n_neighbors"], len(data) // 10 + 1
            )
            n_neighbors = max(1, n_neighbors)  # Ensure at least 1 neighbor

            try:
                knn_imputer = KNNImputer(
                    n_neighbors=n_neighbors,
                    weights=self.strategy_configs["knn"]["weights"],
                )

                numeric_data = data[numeric_cols]
                imputed_numeric = knn_imputer.fit_transform(numeric_data)

                for i, col in enumerate(numeric_cols):
                    if numeric_data[col].isnull().sum() > 0:
                        result_data[col] = imputed_numeric[:, i]
                        self._fitted_imputers[f"{col}_knn"] = knn_imputer
                        imputation_log[col] = {
                            "strategy": "knn",
                            "n_neighbors": n_neighbors,
                            "missing_before": data[col].isnull().sum(),
                            "missing_after": result_data[col].isnull().sum(),
                        }

            except Exception as e:
                logger.warning(f"KNN imputation failed, falling back to median: {e}")
                # Fallback to simple imputation
                for col in numeric_cols:
                    if data[col].isnull().sum() > 0:
                        result_data[col].fillna(data[col].median(), inplace=True)
                        imputation_log[col] = {
                            "strategy": "median_fallback",
                            "missing_before": data[col].isnull().sum(),
                            "missing_after": result_data[col].isnull().sum(),
                            "fallback_reason": str(e),
                        }

        # Handle categorical columns with most frequent
        categorical_cols = data.select_dtypes(include=["object", "category"]).columns
        for col in categorical_cols:
            if data[col].isnull().sum() > 0:
                most_frequent = (
                    data[col].mode().iloc[0] if len(data[col].mode()) > 0 else "unknown"
                )
                result_data[col].fillna(most_frequent, inplace=True)
                imputation_log[col] = {
                    "strategy": "most_frequent",
                    "missing_before": data[col].isnull().sum(),
                    "missing_after": result_data[col].isnull().sum(),
                    "imputed_value": most_frequent,
                }

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        execution_time = time.time() - start_time

        metadata = {
            "strategy": "knn",
            "parameters": {"n_neighbors": n_neighbors},
            "imputation_log": imputation_log,
            "execution_time": execution_time,
            "memory_usage_mb": peak / (1024 * 1024),
            "total_imputed_values": sum(
                log["missing_before"] - log["missing_after"]
                for log in imputation_log.values()
            ),
        }

        return result_data, metadata

    def _iterative_imputation(
        self, data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Iterative imputation using machine learning models."""
        result_data = data.copy()
        imputation_log = {}
        start_time = time.time()

        tracemalloc.start()

        # Prepare data for iterative imputation
        numeric_cols = data.select_dtypes(include=["number"]).columns
        categorical_cols = data.select_dtypes(include=["object", "category"]).columns

        # Handle numeric columns with IterativeImputer
        if len(numeric_cols) > 1 and any(
            data[col].isnull().sum() > 0 for col in numeric_cols
        ):
            try:
                iterative_imputer = IterativeImputer(
                    estimator=self.strategy_configs["iterative"]["estimator"],
                    random_state=self.strategy_configs["iterative"]["random_state"],
                    max_iter=self.strategy_configs["iterative"]["max_iter"],
                )

                numeric_data = data[numeric_cols]
                imputed_numeric = iterative_imputer.fit_transform(numeric_data)

                for i, col in enumerate(numeric_cols):
                    if numeric_data[col].isnull().sum() > 0:
                        result_data[col] = imputed_numeric[:, i]
                        self._fitted_imputers[f"{col}_iterative"] = iterative_imputer
                        imputation_log[col] = {
                            "strategy": "iterative",
                            "estimator": str(iterative_imputer.estimator),
                            "n_iter": iterative_imputer.n_iter_,
                            "missing_before": data[col].isnull().sum(),
                            "missing_after": result_data[col].isnull().sum(),
                        }

            except Exception as e:
                logger.warning(
                    f"Iterative imputation failed, falling back to median: {e}"
                )
                # Fallback to simple imputation
                for col in numeric_cols:
                    if data[col].isnull().sum() > 0:
                        result_data[col].fillna(data[col].median(), inplace=True)
                        imputation_log[col] = {
                            "strategy": "median_fallback",
                            "missing_before": data[col].isnull().sum(),
                            "missing_after": result_data[col].isnull().sum(),
                            "fallback_reason": str(e),
                        }

        # Handle categorical columns
        for col in categorical_cols:
            if data[col].isnull().sum() > 0:
                # Use frequency-based imputation with threshold
                value_counts = data[col].value_counts()
                if len(value_counts) > 0 and value_counts.iloc[0] / len(data) > 0.05:
                    most_frequent = value_counts.index[0]
                    strategy_name = "most_frequent"
                else:
                    most_frequent = "unknown"
                    strategy_name = "unknown_substitution"

                result_data[col].fillna(most_frequent, inplace=True)
                imputation_log[col] = {
                    "strategy": strategy_name,
                    "missing_before": data[col].isnull().sum(),
                    "missing_after": result_data[col].isnull().sum(),
                    "imputed_value": most_frequent,
                }

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        execution_time = time.time() - start_time

        metadata = {
            "strategy": "iterative",
            "parameters": self.strategy_configs["iterative"],
            "imputation_log": imputation_log,
            "execution_time": execution_time,
            "memory_usage_mb": peak / (1024 * 1024),
            "total_imputed_values": sum(
                log["missing_before"] - log["missing_after"]
                for log in imputation_log.values()
            ),
        }

        return result_data, metadata

    def _custom_imputation(
        self, data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Custom imputation based on user-defined parameters."""
        # For now, implement domain-specific strategies
        result_data = data.copy()
        imputation_log = {}

        # Add missing value indicators for high-missing columns
        high_missing_cols = []
        for col in data.columns:
            missing_pct = (data[col].isnull().sum() / len(data)) * 100
            if missing_pct > 30:  # High missing rate
                indicator_col = f"{col}_was_missing"
                result_data[indicator_col] = data[col].isnull().astype(int)
                high_missing_cols.append(col)
                imputation_log[indicator_col] = {
                    "strategy": "missing_indicator",
                    "original_column": col,
                    "missing_percentage": missing_pct,
                }

        # Apply conservative imputation to original columns
        simple_data, simple_metadata = self._simple_imputation(data)
        result_data[data.columns] = simple_data[data.columns]

        # Combine logs
        imputation_log.update(simple_metadata.get("imputation_log", {}))

        metadata = {
            "strategy": "custom",
            "high_missing_columns": high_missing_cols,
            "missing_indicators_added": len(high_missing_cols),
            "imputation_log": imputation_log,
            "execution_time": simple_metadata.get("execution_time", 0),
        }

        return result_data, metadata
