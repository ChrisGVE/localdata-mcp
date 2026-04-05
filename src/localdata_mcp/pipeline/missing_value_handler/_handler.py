"""
Missing Value Handler - Core Handler Class

The MissingValueHandler class that composes all mixins and provides
the main pipeline orchestration and execution methods.
"""

import time
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

from ..base import AnalysisPipelineBase, StreamingConfig
from ._types import MissingValuePattern, ImputationMetadata
from ._pattern_analysis import PatternAnalysisMixin
from ._strategies import StrategySelectionMixin
from ._imputation import ImputationStrategyMixin
from ._quality import QualityAssessmentMixin
from ._comprehensive import ComprehensiveAssessmentMixin
from ._metadata import MetadataUtilityMixin
from ...logging_manager import get_logger

logger = get_logger(__name__)

# Suppress sklearn warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


class MissingValueHandler(
    PatternAnalysisMixin,
    StrategySelectionMixin,
    ImputationStrategyMixin,
    QualityAssessmentMixin,
    ComprehensiveAssessmentMixin,
    MetadataUtilityMixin,
    AnalysisPipelineBase,
):
    """
    Sophisticated missing value handling with sklearn.impute integration.

    This handler provides:
    - Multiple imputation strategies with automatic selection
    - Missing value pattern analysis (MCAR, MAR, MNAR)
    - Cross-validation assessment of imputation quality
    - Progressive disclosure from simple to expert-level control
    - Full transparency and reversibility
    """

    def __init__(
        self,
        analytical_intention: str = "handle missing values intelligently",
        strategy: str = "auto",  # "auto", "simple", "knn", "iterative", "custom"
        complexity: str = "auto",  # "minimal", "auto", "comprehensive", "custom"
        cross_validation: bool = True,
        metadata_tracking: bool = True,
        streaming_config: Optional[StreamingConfig] = None,
        custom_parameters: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize missing value handler with intelligent strategy selection.

        Args:
            analytical_intention: Natural language description of imputation goal
            strategy: Imputation strategy selection approach
            complexity: Level of analysis complexity
            cross_validation: Enable quality assessment via cross-validation
            metadata_tracking: Enable detailed imputation metadata tracking
            streaming_config: Configuration for streaming execution
            custom_parameters: Additional custom parameters
        """
        super().__init__(
            analytical_intention=analytical_intention,
            streaming_config=streaming_config or StreamingConfig(),
            progressive_complexity=complexity,
            composition_aware=True,
            custom_parameters=custom_parameters or {},
        )

        self.strategy = strategy
        self.complexity = complexity
        self.cross_validation = cross_validation
        self.metadata_tracking = metadata_tracking

        # Strategy configurations
        self.strategy_configs = {
            "simple": {"numeric": "median", "categorical": "most_frequent"},
            "knn": {"n_neighbors": 5, "weights": "uniform"},
            "iterative": {
                "estimator": RandomForestRegressor(n_estimators=10, random_state=42),
                "random_state": 42,
                "max_iter": 10,
            },
            "custom": self.custom_parameters.get("strategy_config", {}),
        }

        # Quality thresholds
        self.quality_thresholds = {
            "min_accuracy": 0.7,
            "max_mse_increase": 0.2,
            "min_correlation_preservation": 0.8,
            "max_distribution_deviation": 0.1,
        }

        # Imputation tracking
        self._missing_pattern: Optional[MissingValuePattern] = None
        self._imputation_metadata: Optional[ImputationMetadata] = None
        self._fitted_imputers: Dict[str, Any] = {}
        self._original_data: Optional[pd.DataFrame] = None

        logger.info(
            "MissingValueHandler initialized", strategy=strategy, complexity=complexity
        )

    def get_analysis_type(self) -> str:
        """Get the analysis type - missing value handling."""
        return "missing_value_handling"

    def _configure_analysis_pipeline(self) -> List[Callable]:
        """Configure missing value handling pipeline based on complexity level."""
        pipeline_steps = []

        # Always start with pattern analysis
        pipeline_steps.append(self._analyze_missing_patterns)

        if self.complexity == "minimal":
            pipeline_steps.extend([self._simple_imputation])

        elif self.complexity == "auto":
            pipeline_steps.extend(
                [
                    self._intelligent_strategy_selection,
                    self._apply_selected_strategy,
                    self._assess_imputation_quality,
                ]
            )

        elif self.complexity == "comprehensive":
            pipeline_steps.extend(
                [
                    self._evaluate_all_strategies,
                    self._cross_validate_strategies,
                    self._ensemble_imputation,
                    self._comprehensive_quality_assessment,
                ]
            )

        elif self.complexity == "custom":
            # Load custom pipeline from parameters
            custom_steps = self.custom_parameters.get("pipeline_steps", [])
            pipeline_steps.extend(custom_steps)

        # Always end with metadata compilation
        if self.metadata_tracking:
            pipeline_steps.append(self._compile_imputation_metadata)

        logger.info(f"Configured imputation pipeline with {len(pipeline_steps)} steps")
        return pipeline_steps

    def _execute_analysis_step(
        self, step: Callable, data: pd.DataFrame, context: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Execute individual imputation step with comprehensive logging."""
        step_name = step.__name__
        start_time = time.time()

        # Store original data for quality assessment
        if self._original_data is None:
            self._original_data = data.copy()

        try:
            # Execute the imputation step
            processed_data, step_metadata = step(data)

            execution_time = time.time() - start_time

            metadata = {
                "step": step_name,
                "execution_time": execution_time,
                "success": True,
                "step_metadata": step_metadata,
                "missing_values_before": data.isnull().sum().sum(),
                "missing_values_after": processed_data.isnull().sum().sum(),
            }

            logger.info(
                f"Imputation step {step_name} completed successfully",
                execution_time=execution_time,
                missing_reduced=metadata["missing_values_before"]
                - metadata["missing_values_after"],
            )

            return processed_data, metadata

        except Exception as e:
            execution_time = time.time() - start_time

            logger.error(f"Imputation step {step_name} failed: {e}")

            metadata = {
                "step": step_name,
                "execution_time": execution_time,
                "success": False,
                "error": str(e),
            }

            return data, metadata  # Return original data

    def _execute_streaming_analysis(
        self, data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Execute imputation with streaming support for large datasets."""
        processed_data = data.copy()

        # Apply each imputation step in the pipeline
        for impute_func in self._analysis_pipeline:
            processed_data, step_metadata = self._execute_analysis_step(
                impute_func, processed_data, self.get_execution_context()
            )

        # Build comprehensive metadata
        metadata = self._build_imputation_metadata(
            processed_data, streaming_enabled=True
        )
        return processed_data, metadata

    def _execute_standard_analysis(
        self, data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Execute imputation on full dataset in memory."""
        processed_data = data.copy()

        # Apply each imputation step in the pipeline
        for impute_func in self._analysis_pipeline:
            processed_data, step_metadata = self._execute_analysis_step(
                impute_func, processed_data, self.get_execution_context()
            )

        # Build comprehensive metadata
        metadata = self._build_imputation_metadata(
            processed_data, streaming_enabled=False
        )
        return processed_data, metadata
