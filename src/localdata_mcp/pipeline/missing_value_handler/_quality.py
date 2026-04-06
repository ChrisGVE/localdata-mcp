"""
Missing Value Handler - Quality Assessment

Methods for assessing imputation quality: distribution preservation,
correlation preservation, and overall quality scoring.
"""

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from ...logging_manager import get_logger

logger = get_logger(__name__)


class QualityAssessmentMixin:
    """Mixin providing imputation quality assessment methods."""

    def _assess_imputation_quality(
        self, data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Assess quality of imputation using various metrics."""
        if self._original_data is None:
            return data.copy(), {"quality_assessment": "no_original_data"}

        quality_metrics = {}

        # Compare distributions before and after imputation
        numeric_cols = self._original_data.select_dtypes(include=["number"]).columns

        for col in numeric_cols:
            if self._original_data[col].isnull().sum() > 0:  # Column had missing values
                original_values = self._original_data[col].dropna()
                imputed_values = data[col]

                # Distribution preservation
                try:
                    ks_stat, p_value = stats.ks_2samp(original_values, imputed_values)
                    distribution_preservation = 1 - ks_stat  # Higher is better
                except:
                    distribution_preservation = 0.5

                # Correlation preservation (if possible)
                correlation_preservation = self._assess_correlation_preservation(
                    self._original_data, data, col
                )

                quality_metrics[col] = {
                    "distribution_preservation": distribution_preservation,
                    "correlation_preservation": correlation_preservation,
                    "ks_statistic": ks_stat if "ks_stat" in locals() else None,
                    "ks_p_value": p_value if "p_value" in locals() else None,
                }

        # Overall quality score
        if quality_metrics:
            overall_quality = np.mean(
                [
                    metrics["distribution_preservation"]
                    for metrics in quality_metrics.values()
                ]
            )
        else:
            overall_quality = 1.0  # No imputation needed

        metadata = {
            "quality_assessment": quality_metrics,
            "overall_quality_score": overall_quality,
            "quality_threshold_met": overall_quality
            >= self.quality_thresholds.get("min_correlation_preservation", 0.8),
        }

        return data.copy(), metadata

    def _assess_correlation_preservation(
        self, original_data: pd.DataFrame, imputed_data: pd.DataFrame, target_col: str
    ) -> float:
        """Assess how well correlations are preserved after imputation."""
        numeric_cols = original_data.select_dtypes(include=["number"]).columns
        other_cols = [col for col in numeric_cols if col != target_col]

        if len(other_cols) == 0:
            return 1.0

        try:
            # Calculate correlations before imputation (using only complete cases)
            complete_cases = original_data[list(numeric_cols)].dropna()
            if len(complete_cases) < 10:
                return 0.5  # Not enough data

            original_corrs = complete_cases.corr()[target_col].drop(target_col)

            # Calculate correlations after imputation
            imputed_corrs = (
                imputed_data[list(numeric_cols)].corr()[target_col].drop(target_col)
            )

            # Compare correlations
            correlation_diff = np.abs(original_corrs - imputed_corrs).mean()
            correlation_preservation = max(0.0, 1.0 - correlation_diff)

            return correlation_preservation

        except:
            return 0.5  # Default if calculation fails
