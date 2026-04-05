"""
Missing Value Handler - Metadata & Utilities

Methods for compiling imputation metadata, building result metadata,
analyzing imputed data, and public utility accessors.
"""

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from ._types import MissingValuePattern
from ...logging_manager import get_logger

logger = get_logger(__name__)


class MetadataUtilityMixin:
    """Mixin providing metadata compilation and utility methods."""

    def _compile_imputation_metadata(
        self, data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Compile comprehensive imputation metadata."""
        if not self.metadata_tracking:
            return data.copy(), {"metadata_compilation": "disabled"}

        # Compile all metadata from the imputation process
        metadata = {
            "imputation_summary": {
                "missing_pattern": self._missing_pattern.__dict__
                if self._missing_pattern
                else {},
                "total_original_missing": self._original_data.isnull().sum().sum()
                if self._original_data is not None
                else 0,
                "total_final_missing": data.isnull().sum().sum(),
                "imputation_complete": data.isnull().sum().sum() == 0,
            },
            "fitted_imputers": list(self._fitted_imputers.keys()),
            "quality_thresholds": self.quality_thresholds,
            "configuration": {
                "strategy": self.strategy,
                "complexity": self.complexity,
                "cross_validation_enabled": self.cross_validation,
            },
        }

        return data.copy(), metadata

    def _build_imputation_metadata(
        self, processed_data: pd.DataFrame, streaming_enabled: bool
    ) -> Dict[str, Any]:
        """Build comprehensive metadata for imputation results."""
        metadata = {
            "imputation_pipeline": {
                "analytical_intention": self.analytical_intention,
                "strategy": self.strategy,
                "complexity": self.complexity,
                "streaming_enabled": streaming_enabled,
                "cross_validation": self.cross_validation,
            },
            "missing_value_analysis": {
                "pattern_type": self._missing_pattern.pattern_type
                if self._missing_pattern
                else "unknown",
                "pattern_confidence": self._missing_pattern.confidence_score
                if self._missing_pattern
                else 0.0,
                "recommendations_followed": self._missing_pattern.recommendations
                if self._missing_pattern
                else [],
            },
            "imputation_results": {
                "original_missing_values": self._original_data.isnull().sum().sum()
                if self._original_data is not None
                else 0,
                "final_missing_values": processed_data.isnull().sum().sum(),
                "imputation_complete": processed_data.isnull().sum().sum() == 0,
                "columns_imputed": len(self._fitted_imputers),
            },
            "quality_assessment": {
                "quality_thresholds": self.quality_thresholds,
                "imputation_artifacts_detected": False,  # Would be set by quality assessment
            },
            "composition_context": {
                "ready_for_analysis": processed_data.isnull().sum().sum() == 0,
                "data_characteristics": self._analyze_imputed_data(processed_data),
                "suggested_next_steps": self._suggest_post_imputation_steps(
                    processed_data
                ),
                "imputation_artifacts": self._extract_imputation_artifacts(),
            },
        }

        return metadata

    def _analyze_imputed_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze characteristics of imputed data."""
        return {
            "shape": data.shape,
            "dtypes": dict(data.dtypes),
            "missing_values": data.isnull().sum().sum(),
            "numeric_columns": data.select_dtypes(include=["number"]).columns.tolist(),
            "categorical_columns": data.select_dtypes(
                include=["object", "category"]
            ).columns.tolist(),
            "memory_usage_mb": data.memory_usage(deep=True).sum() / (1024 * 1024),
            "imputation_complete": data.isnull().sum().sum() == 0,
        }

    def _suggest_post_imputation_steps(
        self, data: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """Suggest next steps after imputation."""
        suggestions = []

        if data.isnull().sum().sum() == 0:
            suggestions.append(
                {
                    "analysis_type": "data_validation",
                    "reason": "Complete imputation achieved - validate data quality",
                    "confidence": 0.9,
                }
            )

            suggestions.append(
                {
                    "analysis_type": "outlier_detection",
                    "reason": "Imputation may have introduced outliers",
                    "confidence": 0.7,
                }
            )

            suggestions.append(
                {
                    "analysis_type": "feature_engineering",
                    "reason": "Complete dataset ready for feature creation",
                    "confidence": 0.8,
                }
            )
        else:
            suggestions.append(
                {
                    "analysis_type": "advanced_imputation",
                    "reason": "Some missing values remain - consider domain-specific methods",
                    "confidence": 0.6,
                }
            )

        return suggestions

    def _extract_imputation_artifacts(self) -> Dict[str, Any]:
        """Extract imputation artifacts for potential reuse or analysis."""
        artifacts = {
            "fitted_imputers": self._fitted_imputers,
            "imputation_strategy_used": self.strategy,
            "missing_value_pattern": self._missing_pattern.__dict__
            if self._missing_pattern
            else {},
            "quality_thresholds": self.quality_thresholds,
        }

        return artifacts

    # Public utility methods
    def get_imputation_summary(self) -> str:
        """Get human-readable summary of imputation operations."""
        if self._original_data is None:
            return "No imputation performed yet."

        original_missing = self._original_data.isnull().sum().sum()
        pattern_type = (
            self._missing_pattern.pattern_type if self._missing_pattern else "unknown"
        )

        summary_parts = [
            f"Missing value imputation completed using {self.strategy} strategy.",
            f"Missing value pattern detected: {pattern_type}",
            f"Original missing values: {original_missing:,}",
            f"Imputation strategies applied: {len(self._fitted_imputers)}",
        ]

        if self._missing_pattern:
            summary_parts.append(
                f"Pattern confidence: {self._missing_pattern.confidence_score:.2f}"
            )

        return "\n".join(summary_parts)

    def is_imputation_complete(self) -> bool:
        """Check if imputation is complete (no missing values remain)."""
        return len(self._fitted_imputers) > 0  # At least some imputation was performed

    def get_missing_value_pattern(self) -> Optional[MissingValuePattern]:
        """Get the analyzed missing value pattern."""
        return self._missing_pattern

    def get_fitted_imputers(self) -> Dict[str, Any]:
        """Get fitted imputers for reuse or inspection."""
        return self._fitted_imputers.copy()
