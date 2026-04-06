"""
Time Series Analysis - Quality validation transformer.

Contains the TimeSeriesQualityValidator for assessing time series data quality
and providing preprocessing recommendations.
"""

import time
from typing import Any, Dict, List, Optional

import pandas as pd

from ...logging_manager import get_logger
from ._base import TimeSeriesAnalysisResult
from ._pipeline import validate_time_series_continuity
from ._transformer import TimeSeriesTransformer

logger = get_logger(__name__)


class TimeSeriesQualityValidator(TimeSeriesTransformer):
    """
    sklearn-compatible transformer for time series data quality validation.

    Validates data quality and provides recommendations for preprocessing.
    """

    def __init__(
        self,
        min_observations=10,
        max_missing_percentage=0.2,
        require_regular_frequency=False,
        check_stationarity=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.min_observations = min_observations
        self.max_missing_percentage = max_missing_percentage
        self.require_regular_frequency = require_regular_frequency
        self.check_stationarity = check_stationarity

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fit the quality validator."""
        if self.validate_input:
            X, y = self._validate_time_series(X, y)
        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> TimeSeriesAnalysisResult:
        """Validate time series data quality."""
        start_time = time.time()

        if self.validate_input:
            X, _ = self._validate_time_series(X, None)

        try:
            quality_assessment = validate_time_series_continuity(X)
            warnings_list: List[str] = []
            recommendations_list: List[str] = []
            quality_issues: List[str] = []

            self._check_observations(X, quality_issues, recommendations_list)
            self._check_missing_values(
                quality_assessment, quality_issues, recommendations_list
            )
            self._check_frequency(
                quality_assessment, quality_issues, recommendations_list
            )

            stationarity_info = self._check_stationarity_if_needed(
                X, warnings_list, recommendations_list
            )

            overall_quality = quality_assessment["data_quality_score"]
            interpretation = self._build_interpretation(overall_quality, quality_issues)

            recommendations_list.extend(quality_assessment.get("recommendations", []))

            return TimeSeriesAnalysisResult(
                analysis_type="quality_validation",
                interpretation=interpretation,
                data_quality_score=overall_quality,
                frequency=quality_assessment["frequency"],
                model_diagnostics={
                    "total_observations": len(X),
                    "missing_values": quality_assessment["missing_value_count"],
                    "missing_percentage": quality_assessment[
                        "missing_value_percentage"
                    ],
                    "has_datetime_index": quality_assessment["has_datetime_index"],
                    "is_monotonic": quality_assessment["is_monotonic"],
                    "time_gaps": len(quality_assessment["gaps"]),
                    "stationarity_info": stationarity_info,
                },
                recommendations=recommendations_list,
                warnings=warnings_list
                + [f"Quality issue: {issue}" for issue in quality_issues],
                processing_time=time.time() - start_time,
            )

        except Exception as e:
            logger.error(f"Error in quality validation: {e}")
            return TimeSeriesAnalysisResult(
                analysis_type="quality_validation_error",
                interpretation=f"Error during quality validation: {str(e)}",
                warnings=[str(e)],
                processing_time=time.time() - start_time,
            )

    def _check_observations(
        self,
        X: pd.DataFrame,
        quality_issues: List[str],
        recommendations: List[str],
    ) -> None:
        """Check minimum observation count."""
        if len(X) < self.min_observations:
            quality_issues.append(
                f"Insufficient data: {len(X)} < {self.min_observations} required"
            )
            recommendations.append("Collect more historical data")

    def _check_missing_values(
        self,
        assessment: Dict[str, Any],
        quality_issues: List[str],
        recommendations: List[str],
    ) -> None:
        """Check missing value percentage."""
        missing_pct = assessment["missing_value_percentage"]
        if missing_pct > self.max_missing_percentage * 100:
            quality_issues.append(
                f"High missing values: {missing_pct:.1f}% > "
                f"{self.max_missing_percentage * 100:.1f}% threshold"
            )
            recommendations.append("Apply missing value imputation")

    def _check_frequency(
        self,
        assessment: Dict[str, Any],
        quality_issues: List[str],
        recommendations: List[str],
    ) -> None:
        """Check frequency regularity."""
        if self.require_regular_frequency and assessment["frequency"] is None:
            quality_issues.append("Irregular time frequency detected")
            recommendations.append("Resample to regular frequency")

    def _check_stationarity_if_needed(
        self,
        X: pd.DataFrame,
        warnings_list: List[str],
        recommendations: List[str],
    ) -> Dict[str, Any]:
        """Perform basic stationarity check if configured."""
        if not self.check_stationarity or len(X.columns) == 0:
            return {}

        try:
            from statsmodels.tsa.stattools import adfuller

            first_col = X.iloc[:, 0].dropna()
            if len(first_col) <= 10:
                return {}

            adf_result = adfuller(first_col)
            info = {
                "adf_statistic": adf_result[0],
                "adf_pvalue": adf_result[1],
                "is_likely_stationary": adf_result[1] < 0.05,
            }
            if not info["is_likely_stationary"]:
                recommendations.append(
                    "Consider differencing or transformation for stationarity"
                )
            return info

        except Exception as e:
            warnings_list.append(f"Could not perform stationarity check: {e}")
            return {}

    @staticmethod
    def _build_interpretation(overall_quality: float, quality_issues: List[str]) -> str:
        """Build the quality interpretation string."""
        if len(quality_issues) == 0:
            return f"Time series data quality: GOOD (score: {overall_quality:.2f})"
        elif overall_quality > 0.7:
            return (
                f"Time series data quality: ACCEPTABLE "
                f"(score: {overall_quality:.2f}) with minor issues"
            )
        return (
            f"Time series data quality: POOR "
            f"(score: {overall_quality:.2f}) - significant issues detected"
        )
