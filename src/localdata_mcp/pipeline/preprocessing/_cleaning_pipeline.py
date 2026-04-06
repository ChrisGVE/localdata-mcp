"""
Data Cleaning Pipeline - Advanced data cleaning with sklearn preprocessing.

Implements the DataCleaningPipeline class with intention-driven configuration,
advanced outlier detection, fuzzy duplicate matching, and comprehensive
data validation with progressive complexity levels.
"""

import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from ...logging_manager import get_logger
from ..base import (
    AnalysisPipelineBase,
    StreamingConfig,
)
from ._dataclasses import CleaningOperation, DataQualityMetrics

logger = get_logger(__name__)


class DataCleaningPipeline(AnalysisPipelineBase):
    """
    Advanced data cleaning pipeline using sklearn preprocessing with intention-driven configuration.

    This pipeline provides comprehensive data cleaning capabilities including:
    - Advanced outlier detection using IsolationForest and LocalOutlierFactor
    - Sophisticated duplicate detection with fuzzy matching
    - Comprehensive data validation with configurable business rules
    - Progressive complexity levels from minimal to expert
    - Full transparency and reversibility of operations
    """

    def __init__(
        self,
        analytical_intention: str = "clean data for analysis",
        cleaning_intensity: str = "auto",  # "minimal", "auto", "comprehensive", "custom"
        quality_thresholds: Optional[Dict[str, float]] = None,
        business_rules: Optional[List[Dict[str, Any]]] = None,
        streaming_config: Optional[StreamingConfig] = None,
        custom_parameters: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize data cleaning pipeline with intention-driven configuration.

        Args:
            analytical_intention: Natural language description of cleaning goal
            cleaning_intensity: Level of cleaning complexity
            quality_thresholds: Custom quality score thresholds
            business_rules: Custom business validation rules
            streaming_config: Configuration for streaming execution
            custom_parameters: Additional custom parameters
        """
        super().__init__(
            analytical_intention=analytical_intention,
            streaming_config=streaming_config or StreamingConfig(),
            progressive_complexity=cleaning_intensity,
            composition_aware=True,
            custom_parameters=custom_parameters or {},
        )

        self.cleaning_intensity = cleaning_intensity
        self.quality_thresholds = quality_thresholds or {
            "completeness_threshold": 0.95,
            "consistency_threshold": 0.98,
            "validity_threshold": 0.90,
            "accuracy_threshold": 0.85,
            "overall_threshold": 0.90,
        }
        self.business_rules = business_rules or []

        # Cleaning operation tracking
        self._cleaning_operations: List[CleaningOperation] = []
        self._quality_metrics_before: Optional[DataQualityMetrics] = None
        self._quality_metrics_after: Optional[DataQualityMetrics] = None

        # Advanced cleaning components
        self._outlier_detectors = {
            "isolation_forest": None,
            "local_outlier_factor": None,
        }

        logger.info(
            "DataCleaningPipeline initialized",
            intention=analytical_intention,
            intensity=cleaning_intensity,
        )

    def get_analysis_type(self) -> str:
        """Get the analysis type - data cleaning."""
        return "data_cleaning"

    def _configure_analysis_pipeline(self) -> List[Callable]:
        """Configure cleaning pipeline based on intensity level and intention."""
        pipeline_steps = []

        # Always start with data profiling
        pipeline_steps.append(self._assess_initial_quality)

        if self.cleaning_intensity == "minimal":
            pipeline_steps.extend(
                [
                    self._basic_type_inference,
                    self._handle_basic_missing_values,
                    self._remove_exact_duplicates,
                ]
            )

        elif self.cleaning_intensity == "auto":
            pipeline_steps.extend(
                [
                    self._comprehensive_type_inference,
                    self._intelligent_missing_value_handling,
                    self._advanced_outlier_detection,
                    self._sophisticated_duplicate_detection,
                    self._basic_data_validation,
                ]
            )

        elif self.cleaning_intensity == "comprehensive":
            pipeline_steps.extend(
                [
                    self._comprehensive_type_inference,
                    self._intelligent_missing_value_handling,
                    self._advanced_outlier_detection,
                    self._sophisticated_duplicate_detection,
                    self._comprehensive_data_validation,
                    self._data_consistency_enhancement,
                    self._feature_engineering_cleanup,
                    self._final_quality_optimization,
                ]
            )

        elif self.cleaning_intensity == "custom":
            # Load custom cleaning steps from parameters
            custom_steps = self.custom_parameters.get("cleaning_steps", [])
            pipeline_steps.extend(custom_steps)

        # Always end with quality assessment
        pipeline_steps.append(self._assess_final_quality)

        logger.info(f"Configured cleaning pipeline with {len(pipeline_steps)} steps")
        return pipeline_steps

    def _execute_analysis_step(
        self, step: Callable, data: pd.DataFrame, context: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Execute individual cleaning step with comprehensive logging."""
        step_name = step.__name__
        start_time = time.time()

        try:
            # Capture before statistics
            before_stats = self._capture_data_statistics(data)

            # Execute the cleaning step
            cleaned_data, step_metadata = step(data)

            # Capture after statistics
            after_stats = self._capture_data_statistics(cleaned_data)

            execution_time = time.time() - start_time

            # Record the cleaning operation
            operation = CleaningOperation(
                operation_type=step_name,
                parameters=step_metadata.get("parameters", {}),
                records_affected=step_metadata.get("records_affected", 0),
                execution_time=execution_time,
                success=True,
                before_stats=before_stats,
                after_stats=after_stats,
                reversibility_data=step_metadata.get("reversibility_data", {}),
            )

            self._cleaning_operations.append(operation)

            metadata = {
                "step": step_name,
                "execution_time": execution_time,
                "success": True,
                "step_metadata": step_metadata,
                "data_impact": {
                    "rows_before": len(data),
                    "rows_after": len(cleaned_data),
                    "columns_before": len(data.columns),
                    "columns_after": len(cleaned_data.columns),
                },
            }

            logger.info(
                f"Cleaning step {step_name} completed successfully",
                execution_time=execution_time,
                records_affected=operation.records_affected,
            )

            return cleaned_data, metadata

        except Exception as e:
            execution_time = time.time() - start_time

            # Record failed operation
            operation = CleaningOperation(
                operation_type=step_name,
                execution_time=execution_time,
                success=False,
                error_message=str(e),
            )

            self._cleaning_operations.append(operation)

            logger.error(f"Cleaning step {step_name} failed: {e}")

            # Return original data for graceful degradation
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
        """Execute cleaning with streaming support for large datasets."""
        processed_data = data.copy()

        # Apply each cleaning step in the pipeline
        for clean_func in self._analysis_pipeline:
            processed_data, step_metadata = self._execute_analysis_step(
                clean_func, processed_data, self.get_execution_context()
            )

        # Build comprehensive metadata
        metadata = self._build_cleaning_metadata(processed_data, streaming_enabled=True)
        return processed_data, metadata

    def _execute_standard_analysis(
        self, data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Execute cleaning on full dataset in memory."""
        processed_data = data.copy()

        # Apply each cleaning step in the pipeline
        for clean_func in self._analysis_pipeline:
            processed_data, step_metadata = self._execute_analysis_step(
                clean_func, processed_data, self.get_execution_context()
            )

        # Build comprehensive metadata
        metadata = self._build_cleaning_metadata(
            processed_data, streaming_enabled=False
        )
        return processed_data, metadata

    # ===========================================
    # IMPORT CLEANING METHOD IMPLEMENTATIONS
    # ===========================================

    # Import method implementations from preprocessing sub-package modules
    from ._cleaning_methods import (
        _assess_initial_quality,
        _basic_type_inference,
        _comprehensive_type_inference,
        _handle_basic_missing_values,
        _intelligent_missing_value_handling,
    )
    from ._cleaning_methods_part2 import (
        _advanced_outlier_detection,
        _basic_data_validation,
        _remove_exact_duplicates,
        _sophisticated_duplicate_detection,
    )
    from ._cleaning_methods_part3 import (
        _assess_final_quality,
        _comprehensive_data_validation,
        _data_consistency_enhancement,
        _feature_engineering_cleanup,
        _final_quality_optimization,
    )

    # ===========================================
    # UTILITY METHODS
    # ===========================================

    def _calculate_comprehensive_quality_metrics(
        self, data: pd.DataFrame
    ) -> DataQualityMetrics:
        """Calculate comprehensive data quality metrics."""
        metrics = DataQualityMetrics()

        # Completeness - percentage of non-null values
        total_cells = len(data) * len(data.columns)
        non_null_cells = total_cells - data.isnull().sum().sum()
        metrics.completeness_score = (non_null_cells / total_cells) * 100
        metrics.missing_value_percentage = (
            (total_cells - non_null_cells) / total_cells
        ) * 100

        # Consistency - no duplicates
        total_rows = len(data)
        unique_rows = len(data.drop_duplicates())
        metrics.consistency_score = (unique_rows / total_rows) * 100
        metrics.duplicate_percentage = ((total_rows - unique_rows) / total_rows) * 100

        # Validity - appropriate data types and ranges
        validity_scores = []

        # Check numeric columns for reasonable ranges
        numeric_cols = data.select_dtypes(include=["number"]).columns
        for col in numeric_cols:
            # Simple validity check - no infinite values
            infinite_count = np.isinf(data[col]).sum()
            validity_scores.append((len(data) - infinite_count) / len(data) * 100)

        # Check datetime columns
        datetime_cols = data.select_dtypes(include=["datetime64"]).columns
        for col in datetime_cols:
            # Simple validity check - reasonable date range
            invalid_dates = data[col].isna().sum()
            validity_scores.append((len(data) - invalid_dates) / len(data) * 100)

        metrics.validity_score = np.mean(validity_scores) if validity_scores else 100
        metrics.type_conformity_percentage = metrics.validity_score

        # Accuracy - outlier assessment (inverse of outlier percentage)
        try:
            if len(numeric_cols) > 0:
                numeric_data = data[numeric_cols].fillna(data[numeric_cols].median())
                if len(numeric_data) > 10:  # Need minimum data for outlier detection
                    iso_forest = IsolationForest(contamination=0.1, random_state=42)
                    outliers = iso_forest.fit_predict(numeric_data)
                    outlier_percentage = (outliers == -1).sum() / len(outliers) * 100
                    metrics.outlier_percentage = outlier_percentage
                    metrics.accuracy_score = max(0, 100 - outlier_percentage)
                else:
                    metrics.accuracy_score = 95  # Default for small datasets
            else:
                metrics.accuracy_score = 100  # No numeric columns
        except Exception:
            metrics.accuracy_score = 90  # Default fallback

        # Business rules compliance (default 100 if no rules specified)
        metrics.business_rules_compliance = 100

        # Calculate overall score
        metrics.calculate_overall_score()

        # Data profile
        metrics.data_profile = {
            "shape": data.shape,
            "dtypes": dict(data.dtypes),
            "memory_usage_mb": data.memory_usage(deep=True).sum() / (1024 * 1024),
            "numeric_columns": len(numeric_cols),
            "categorical_columns": len(
                data.select_dtypes(include=["object", "category"]).columns
            ),
            "datetime_columns": len(datetime_cols),
        }

        return metrics

    def _capture_data_statistics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Capture comprehensive data statistics for before/after comparison."""
        stats = {
            "shape": data.shape,
            "dtypes": dict(data.dtypes),
            "missing_values": data.isnull().sum().sum(),
            "duplicates": data.duplicated().sum(),
            "memory_usage_mb": data.memory_usage(deep=True).sum() / (1024 * 1024),
        }

        # Numeric statistics
        numeric_cols = data.select_dtypes(include=["number"]).columns
        if len(numeric_cols) > 0:
            stats["numeric_summary"] = data[numeric_cols].describe().to_dict()

        return stats

    def _build_cleaning_metadata(
        self, cleaned_data: pd.DataFrame, streaming_enabled: bool
    ) -> Dict[str, Any]:
        """Build comprehensive metadata for cleaning results."""
        metadata = {
            "cleaning_pipeline": {
                "analytical_intention": self.analytical_intention,
                "cleaning_intensity": self.cleaning_intensity,
                "streaming_enabled": streaming_enabled,
                "total_operations": len(self._cleaning_operations),
                "successful_operations": sum(
                    1 for op in self._cleaning_operations if op.success
                ),
                "failed_operations": sum(
                    1 for op in self._cleaning_operations if not op.success
                ),
            },
            "quality_assessment": {
                "before": (
                    self._quality_metrics_before.__dict__
                    if self._quality_metrics_before
                    else {}
                ),
                "after": (
                    self._quality_metrics_after.__dict__
                    if self._quality_metrics_after
                    else {}
                ),
                "improvement": {
                    "overall_score": (
                        self._quality_metrics_after.overall_quality_score
                        - self._quality_metrics_before.overall_quality_score
                        if self._quality_metrics_before and self._quality_metrics_after
                        else 0
                    )
                },
            },
            "operations_log": [op.__dict__ for op in self._cleaning_operations],
            "data_transformation": {
                "original_shape": (
                    self._quality_metrics_before.data_profile.get("shape", (0, 0))
                    if self._quality_metrics_before
                    else (0, 0)
                ),
                "cleaned_shape": cleaned_data.shape,
                "total_records_affected": sum(
                    op.records_affected
                    for op in self._cleaning_operations
                    if op.success
                ),
            },
            "composition_context": {
                "ready_for_analysis": (
                    self._quality_metrics_after.overall_quality_score
                    >= self.quality_thresholds["overall_threshold"]
                    if self._quality_metrics_after
                    else False
                ),
                "data_characteristics": self._analyze_cleaned_data(cleaned_data),
                "suggested_next_steps": self._suggest_post_cleaning_steps(cleaned_data),
                "cleaning_artifacts": self._extract_cleaning_artifacts(),
            },
        }

        return metadata

    def _analyze_cleaned_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze characteristics of cleaned data."""
        return {
            "shape": data.shape,
            "dtypes": dict(data.dtypes),
            "quality_score": (
                self._quality_metrics_after.overall_quality_score
                if self._quality_metrics_after
                else 0
            ),
            "numeric_columns": data.select_dtypes(include=["number"]).columns.tolist(),
            "categorical_columns": data.select_dtypes(
                include=["object", "category"]
            ).columns.tolist(),
            "datetime_columns": data.select_dtypes(
                include=["datetime64"]
            ).columns.tolist(),
            "missing_values": data.isnull().sum().sum(),
            "memory_usage_mb": data.memory_usage(deep=True).sum() / (1024 * 1024),
            "ready_for_analysis": (
                self._quality_metrics_after.overall_quality_score
                >= self.quality_thresholds["overall_threshold"]
                if self._quality_metrics_after
                else False
            ),
        }

    def _suggest_post_cleaning_steps(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Suggest next steps after data cleaning."""
        suggestions = []

        numeric_cols = data.select_dtypes(include=["number"]).columns.tolist()
        categorical_cols = data.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()
        datetime_cols = data.select_dtypes(include=["datetime64"]).columns.tolist()

        # Statistical analysis suggestion
        if len(numeric_cols) >= 2:
            suggestions.append(
                {
                    "analysis_type": "statistical_analysis",
                    "reason": "Multiple numeric columns available for correlation and statistical tests",
                    "confidence": 0.9,
                    "next_tool": "statistical_analyzer",
                }
            )

        # Time series analysis
        if datetime_cols and numeric_cols:
            suggestions.append(
                {
                    "analysis_type": "time_series_analysis",
                    "reason": "Datetime and numeric columns available for temporal analysis",
                    "confidence": 0.8,
                    "next_tool": "time_series_analyzer",
                }
            )

        # Machine learning readiness
        if categorical_cols and numeric_cols:
            suggestions.append(
                {
                    "analysis_type": "machine_learning",
                    "reason": "Mixed data types suitable for supervised/unsupervised learning",
                    "confidence": 0.7,
                    "next_tool": "ml_preprocessor",
                }
            )

        # Data visualization
        suggestions.append(
            {
                "analysis_type": "data_visualization",
                "reason": "Clean data ready for exploratory visualization",
                "confidence": 0.9,
                "next_tool": "visualization_generator",
            }
        )

        return suggestions

    def _extract_cleaning_artifacts(self) -> Dict[str, Any]:
        """Extract cleaning artifacts for potential reuse or analysis."""
        artifacts = {
            "transformation_history": [
                op.__dict__ for op in self._cleaning_operations if op.success
            ],
            "quality_thresholds": self.quality_thresholds,
            "business_rules_applied": self.business_rules,
            "outlier_detection_parameters": {
                "methods_used": ["isolation_forest", "local_outlier_factor"],
                "contamination_rate": 0.1,
            },
            "type_inference_results": {
                # Extract from operation logs
                op.operation_type: op.after_stats
                for op in self._cleaning_operations
                if "type_inference" in op.operation_type and op.success
            },
        }

        return artifacts

    # Public utility methods
    def get_quality_report(self) -> Dict[str, Any]:
        """Get comprehensive quality report before and after cleaning."""
        return {
            "before_cleaning": (
                self._quality_metrics_before.__dict__
                if self._quality_metrics_before
                else {}
            ),
            "after_cleaning": (
                self._quality_metrics_after.__dict__
                if self._quality_metrics_after
                else {}
            ),
            "operations_performed": len(self._cleaning_operations),
            "successful_operations": sum(
                1 for op in self._cleaning_operations if op.success
            ),
            "failed_operations": sum(
                1 for op in self._cleaning_operations if not op.success
            ),
            "total_records_affected": sum(
                op.records_affected for op in self._cleaning_operations if op.success
            ),
        }

    def get_cleaning_summary(self) -> str:
        """Get human-readable summary of cleaning operations."""
        if not self._cleaning_operations:
            return "No cleaning operations performed yet."

        successful_ops = [op for op in self._cleaning_operations if op.success]
        total_affected = sum(op.records_affected for op in successful_ops)

        summary_parts = [
            f"Data cleaning completed with {len(successful_ops)} successful operations.",
            f"Total records affected: {total_affected:,}",
        ]

        if self._quality_metrics_before and self._quality_metrics_after:
            improvement = (
                self._quality_metrics_after.overall_quality_score
                - self._quality_metrics_before.overall_quality_score
            )
            summary_parts.append(
                f"Overall quality improvement: {improvement:.1f} points"
            )
            summary_parts.append(
                f"Final quality score: {self._quality_metrics_after.overall_quality_score:.1f}/100"
            )

        return "\n".join(summary_parts)

    def is_ready_for_analysis(self) -> bool:
        """Check if data meets quality thresholds for analysis."""
        if not self._quality_metrics_after:
            return False

        return (
            self._quality_metrics_after.overall_quality_score
            >= self.quality_thresholds["overall_threshold"]
        )
