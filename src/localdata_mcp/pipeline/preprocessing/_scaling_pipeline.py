"""
Feature Scaling Pipeline - Sklearn-based feature scaling and normalization.

Implements the FeatureScalingPipeline class with automatic strategy selection
based on data distribution analysis, streaming compatibility, and metadata
preservation for inverse transforms.
"""

import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
from sklearn.preprocessing import (
    StandardScaler,
    RobustScaler,
    MinMaxScaler,
    QuantileTransformer,
    PowerTransformer,
)

from ..base import (
    AnalysisPipelineBase,
    StreamingConfig,
)
from ...logging_manager import get_logger

logger = get_logger(__name__)


class FeatureScalingPipeline(AnalysisPipelineBase):
    """
    Feature scaling and normalization pipeline using sklearn preprocessing transformers.

    Provides comprehensive feature scaling capabilities with automatic strategy selection
    based on data distribution analysis, streaming compatibility, and metadata preservation.
    """

    def __init__(
        self,
        analytical_intention: str = "scale features for analysis",
        scaling_strategy: str = "auto",  # "auto", "standard", "minmax", "robust", "quantile", "power"
        column_specific_scaling: Optional[Dict[str, str]] = None,
        streaming_config: Optional[StreamingConfig] = None,
        custom_parameters: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize feature scaling pipeline.

        Args:
            analytical_intention: Natural language description of scaling goal
            scaling_strategy: Scaling method to use or "auto" for automatic selection
            column_specific_scaling: Per-column scaling strategies
            streaming_config: Configuration for streaming execution
            custom_parameters: Additional custom parameters
        """
        super().__init__(
            analytical_intention=analytical_intention,
            streaming_config=streaming_config or StreamingConfig(),
            progressive_complexity="auto",
            composition_aware=True,
            custom_parameters=custom_parameters or {},
        )

        self.scaling_strategy = scaling_strategy
        self.column_specific_scaling = column_specific_scaling or {}

        # Scaling state for streaming compatibility
        self._scalers: Dict[str, Any] = {}
        self._scaling_metadata: Dict[str, Any] = {}

        logger.info(
            "FeatureScalingPipeline initialized",
            intention=analytical_intention,
            strategy=scaling_strategy,
        )

    def get_analysis_type(self) -> str:
        """Get the analysis type - feature scaling."""
        return "feature_scaling"

    def _configure_analysis_pipeline(self) -> List[Callable]:
        """Configure scaling pipeline steps."""
        return [
            self._analyze_data_distributions,
            self._select_scaling_strategies,
            self._apply_feature_scaling,
            self._validate_scaling_results,
        ]

    def _execute_analysis_step(
        self, step: Callable, data: pd.DataFrame, context: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Execute individual scaling step with error handling."""
        step_name = step.__name__
        start_time = time.time()

        try:
            # Execute the scaling step
            scaled_data, step_metadata = step(data)

            execution_time = time.time() - start_time

            metadata = {
                "step": step_name,
                "execution_time": execution_time,
                "success": True,
                "step_metadata": step_metadata,
            }

            logger.info(
                f"Scaling step {step_name} completed successfully",
                execution_time=execution_time,
            )

            return scaled_data, metadata

        except Exception as e:
            execution_time = time.time() - start_time

            logger.error(f"Scaling step {step_name} failed: {e}")

            metadata = {
                "step": step_name,
                "execution_time": execution_time,
                "success": False,
                "error": str(e),
            }

            return data, metadata  # Return original data on failure

    def _execute_streaming_analysis(
        self, data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Execute scaling with streaming support."""
        processed_data = data.copy()

        # Apply each scaling step in the pipeline
        for scale_func in self._analysis_pipeline:
            processed_data, step_metadata = self._execute_analysis_step(
                scale_func, processed_data, self.get_execution_context()
            )

        metadata = self._build_scaling_metadata(processed_data, streaming_enabled=True)
        return processed_data, metadata

    def _execute_standard_analysis(
        self, data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Execute scaling on full dataset in memory."""
        processed_data = data.copy()

        # Apply each scaling step in the pipeline
        for scale_func in self._analysis_pipeline:
            processed_data, step_metadata = self._execute_analysis_step(
                scale_func, processed_data, self.get_execution_context()
            )

        metadata = self._build_scaling_metadata(processed_data, streaming_enabled=False)
        return processed_data, metadata

    # Scaling method implementations
    def _analyze_data_distributions(
        self, data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Analyze data distributions to inform scaling strategy selection."""
        numeric_cols = data.select_dtypes(include=["number"]).columns
        distribution_analysis = {}

        for col in numeric_cols:
            col_data = data[col].dropna()
            if len(col_data) == 0:
                continue

            # Basic statistics
            stats = {
                "mean": col_data.mean(),
                "median": col_data.median(),
                "std": col_data.std(),
                "min": col_data.min(),
                "max": col_data.max(),
                "skewness": col_data.skew(),
                "kurtosis": col_data.kurtosis(),
            }

            # Distribution characteristics
            stats["has_outliers"] = self._detect_outliers_iqr(col_data)
            stats["is_normal"] = (
                abs(stats["skewness"]) < 0.5 and abs(stats["kurtosis"]) < 3
            )
            stats["has_negative"] = stats["min"] < 0
            stats["wide_range"] = (stats["max"] - stats["min"]) > 1000

            distribution_analysis[col] = stats

        metadata = {
            "distribution_analysis": distribution_analysis,
            "numeric_columns_analyzed": len(numeric_cols),
        }

        return data, metadata

    def _select_scaling_strategies(
        self, data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Select optimal scaling strategies for each column."""
        numeric_cols = data.select_dtypes(include=["number"]).columns
        selected_strategies = {}

        for col in numeric_cols:
            if col in self.column_specific_scaling:
                # Use user-specified strategy
                selected_strategies[col] = self.column_specific_scaling[col]
            elif self.scaling_strategy != "auto":
                # Use global strategy
                selected_strategies[col] = self.scaling_strategy
            else:
                # Auto-select based on data characteristics
                col_data = data[col].dropna()
                if len(col_data) == 0:
                    continue

                # Decision logic based on data characteristics
                skewness = abs(col_data.skew())
                has_outliers = self._detect_outliers_iqr(col_data)
                wide_range = (col_data.max() - col_data.min()) > 1000

                if skewness > 2:
                    selected_strategies[col] = "power"  # Handle highly skewed data
                elif has_outliers:
                    selected_strategies[col] = "robust"  # Robust to outliers
                elif wide_range:
                    selected_strategies[col] = "minmax"  # Scale to fixed range
                else:
                    selected_strategies[col] = "standard"  # Standard normalization

        self._scaling_metadata["selected_strategies"] = selected_strategies

        metadata = {
            "selected_strategies": selected_strategies,
            "total_columns": len(selected_strategies),
        }

        return data, metadata

    def _apply_feature_scaling(
        self, data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Apply the selected scaling transformations."""
        result_data = data.copy()
        scaling_log = {}

        strategies = self._scaling_metadata.get("selected_strategies", {})

        for col, strategy in strategies.items():
            try:
                # Create and fit scaler
                scaler = self._create_scaler(strategy)

                # Fit and transform the column
                original_data = data[[col]]
                scaled_data = scaler.fit_transform(original_data)
                result_data[col] = scaled_data.flatten()

                # Store scaler for potential inverse transformation or streaming
                self._scalers[col] = scaler

                # Log scaling operation
                scaling_log[col] = {
                    "strategy": strategy,
                    "scaler_type": type(scaler).__name__,
                    "original_range": [data[col].min(), data[col].max()],
                    "scaled_range": [result_data[col].min(), result_data[col].max()],
                }

            except Exception as e:
                logger.warning(f"Failed to scale column {col} with {strategy}: {e}")
                scaling_log[col] = {
                    "strategy": strategy,
                    "error": str(e),
                    "fallback": "no_scaling",
                }

        metadata = {
            "scaling_log": scaling_log,
            "successful_scalings": sum(
                1 for log in scaling_log.values() if "error" not in log
            ),
            "failed_scalings": sum(1 for log in scaling_log.values() if "error" in log),
        }

        return result_data, metadata

    def _validate_scaling_results(
        self, data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Validate scaling results and generate quality metrics."""
        numeric_cols = data.select_dtypes(include=["number"]).columns
        validation_results = {}

        for col in numeric_cols:
            if col in self._scalers:
                col_data = data[col].dropna()

                validation_results[col] = {
                    "mean": col_data.mean(),
                    "std": col_data.std(),
                    "min": col_data.min(),
                    "max": col_data.max(),
                    "has_invalid_values": (
                        np.isinf(col_data) | np.isnan(col_data)
                    ).any(),
                    "scaling_quality": "good"
                    if abs(col_data.mean()) < 2 and col_data.std() > 0
                    else "check_needed",
                }

        metadata = {
            "validation_results": validation_results,
            "overall_quality": "good"
            if all(v["scaling_quality"] == "good" for v in validation_results.values())
            else "needs_review",
        }

        return data, metadata

    def _create_scaler(self, strategy: str) -> Any:
        """Create scaler instance based on strategy."""
        if strategy == "standard":
            return StandardScaler()
        elif strategy == "minmax":
            return MinMaxScaler()
        elif strategy == "robust":
            return RobustScaler()
        elif strategy == "quantile":
            return QuantileTransformer(n_quantiles=100, random_state=42)
        elif strategy == "power":
            return PowerTransformer(method="yeo-johnson", standardize=True)
        else:
            return StandardScaler()  # Default fallback

    def _detect_outliers_iqr(self, data: pd.Series) -> bool:
        """Detect outliers using IQR method."""
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = (data < lower_bound) | (data > upper_bound)
        return outliers.sum() > 0.05 * len(data)  # More than 5% outliers

    def _build_scaling_metadata(
        self, scaled_data: pd.DataFrame, streaming_enabled: bool
    ) -> Dict[str, Any]:
        """Build comprehensive metadata for scaling results."""
        return {
            "scaling_pipeline": {
                "analytical_intention": self.analytical_intention,
                "scaling_strategy": self.scaling_strategy,
                "streaming_enabled": streaming_enabled,
                "scalers_fitted": len(self._scalers),
            },
            "scaling_results": self._scaling_metadata,
            "data_characteristics": {
                "shape": scaled_data.shape,
                "numeric_columns": scaled_data.select_dtypes(
                    include=["number"]
                ).columns.tolist(),
                "memory_usage_mb": scaled_data.memory_usage(deep=True).sum()
                / (1024 * 1024),
            },
            "composition_context": {
                "ready_for_ml": True,
                "scaling_artifacts": {
                    "fitted_scalers": list(self._scalers.keys()),
                    "inverse_transform_available": True,
                },
                "suggested_next_steps": [
                    {
                        "analysis_type": "machine_learning",
                        "reason": "Features scaled for ML algorithms",
                        "confidence": 0.9,
                    },
                    {
                        "analysis_type": "statistical_analysis",
                        "reason": "Normalized features for statistical comparison",
                        "confidence": 0.8,
                    },
                ],
            },
        }

    # Public utility methods
    def inverse_transform(self, scaled_data: pd.DataFrame) -> pd.DataFrame:
        """Inverse transform scaled data back to original scale."""
        result_data = scaled_data.copy()

        for col, scaler in self._scalers.items():
            if col in result_data.columns:
                try:
                    original_data = scaler.inverse_transform(result_data[[col]])
                    result_data[col] = original_data.flatten()
                except Exception as e:
                    logger.warning(f"Failed to inverse transform {col}: {e}")

        return result_data

    def get_scaler(self, column: str) -> Optional[Any]:
        """Get fitted scaler for specific column."""
        return self._scalers.get(column)
