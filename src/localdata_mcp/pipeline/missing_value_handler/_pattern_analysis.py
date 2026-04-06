"""
Missing Value Handler - Pattern Analysis

Methods for analyzing missing value patterns (MCAR, MAR, MNAR),
column-level patterns, temporal patterns, and pattern confidence scoring.
"""

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from ...logging_manager import get_logger
from ._types import MissingValuePattern

logger = get_logger(__name__)


class PatternAnalysisMixin:
    """Mixin providing missing value pattern analysis methods."""

    def _analyze_missing_patterns(
        self, data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Analyze missing value patterns to guide imputation strategy."""
        missing_info = {}

        # Calculate missing percentages per column
        missing_counts = data.isnull().sum()
        missing_percentages = (missing_counts / len(data)) * 100

        # Analyze missing patterns across columns
        missing_pattern_matrix = data.isnull()
        pattern_correlations = missing_pattern_matrix.corr()

        # Classify missing pattern type
        pattern_type = self._classify_missing_pattern(data, pattern_correlations)

        # Column-specific patterns
        column_patterns = {}
        for col in data.columns:
            if missing_counts[col] > 0:
                column_patterns[col] = {
                    "missing_count": missing_counts[col],
                    "missing_percentage": missing_percentages[col],
                    "data_type": str(data[col].dtype),
                    "unique_values": data[col].nunique(),
                    "pattern_with_others": self._analyze_column_pattern(data, col),
                }

        # Temporal patterns (if datetime columns exist)
        temporal_patterns = self._analyze_temporal_patterns(data)

        # Generate recommendations based on patterns
        recommendations = self._generate_pattern_recommendations(
            pattern_type, missing_percentages, column_patterns
        )

        # Store pattern analysis
        self._missing_pattern = MissingValuePattern(
            pattern_type=pattern_type,
            missing_percentage=missing_percentages.mean(),
            column_patterns=column_patterns,
            correlation_matrix=pattern_correlations,
            temporal_patterns=temporal_patterns,
            confidence_score=self._calculate_pattern_confidence(pattern_correlations),
            recommendations=recommendations,
        )

        metadata = {
            "pattern_analysis": {
                "pattern_type": pattern_type,
                "overall_missing_percentage": missing_percentages.mean(),
                "columns_with_missing": len(column_patterns),
                "recommendations": recommendations,
                "confidence_score": self._missing_pattern.confidence_score,
            }
        }

        return data.copy(), metadata

    def _classify_missing_pattern(
        self, data: pd.DataFrame, pattern_correlations: pd.DataFrame
    ) -> str:
        """Classify the type of missing data pattern."""
        # Analyze correlation strengths
        strong_correlations = (pattern_correlations.abs() > 0.5).sum().sum() - len(
            pattern_correlations
        )
        moderate_correlations = (
            ((pattern_correlations.abs() > 0.3) & (pattern_correlations.abs() <= 0.5))
            .sum()
            .sum()
        )

        total_possible = len(pattern_correlations) * (len(pattern_correlations) - 1)

        if strong_correlations > total_possible * 0.1:
            return "MNAR"  # Missing Not At Random
        elif moderate_correlations > total_possible * 0.2:
            return "MAR"  # Missing At Random
        else:
            return "MCAR"  # Missing Completely At Random

    def _analyze_column_pattern(
        self, data: pd.DataFrame, column: str
    ) -> Dict[str, Any]:
        """Analyze missing pattern for a specific column."""
        missing_mask = data[column].isnull()

        pattern_info = {
            "correlates_with": [],
            "distribution_bias": None,
            "temporal_pattern": None,
        }

        # Check correlation with other columns' missing patterns
        for other_col in data.columns:
            if other_col != column and data[other_col].isnull().sum() > 0:
                correlation = missing_mask.corr(data[other_col].isnull())
                if abs(correlation) > 0.3:
                    pattern_info["correlates_with"].append(
                        {"column": other_col, "correlation": correlation}
                    )

        # Check for distribution bias
        numeric_cols = data.select_dtypes(include=["number"]).columns
        for num_col in numeric_cols:
            if (
                num_col != column
                and len(data[~missing_mask]) > 10
                and len(data[missing_mask]) > 10
            ):
                present_data = data[~missing_mask][num_col].dropna()
                all_data = data[num_col].dropna()

                if len(present_data) > 0 and len(all_data) > 0:
                    try:
                        # Compare distributions
                        stat, p_value = stats.ks_2samp(present_data, all_data)
                        if p_value < 0.05:  # Significant difference
                            pattern_info["distribution_bias"] = {
                                "affected_column": num_col,
                                "p_value": p_value,
                                "mean_difference": present_data.mean()
                                - all_data.mean(),
                            }
                    except:
                        pass

        return pattern_info

    def _analyze_temporal_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze temporal patterns in missing values."""
        temporal_patterns = {}

        # Check for datetime columns
        datetime_cols = data.select_dtypes(include=["datetime64"]).columns

        for dt_col in datetime_cols:
            if data[dt_col].isnull().sum() == 0:  # Use as time reference
                for col in data.columns:
                    if col != dt_col and data[col].isnull().sum() > 0:
                        try:
                            # Analyze missing pattern over time
                            missing_by_time = (
                                data.groupby(pd.Grouper(key=dt_col, freq="D"))[col]
                                .apply(
                                    lambda x: (
                                        x.isnull().sum() / len(x) if len(x) > 0 else 0
                                    )
                                )
                                .fillna(0)
                            )

                            if (
                                len(missing_by_time) > 1
                                and missing_by_time.var() > 0.01
                            ):  # Significant temporal variation
                                temporal_patterns[col] = {
                                    "time_reference": dt_col,
                                    "temporal_variance": missing_by_time.var(),
                                    "peak_missing_periods": missing_by_time.nlargest(
                                        min(3, len(missing_by_time))
                                    ).index.tolist(),
                                }
                        except:
                            pass

        return temporal_patterns

    def _generate_pattern_recommendations(
        self,
        pattern_type: str,
        missing_percentages: pd.Series,
        column_patterns: Dict[str, Dict[str, Any]],
    ) -> List[str]:
        """Generate imputation strategy recommendations based on missing patterns."""
        recommendations = []

        # Overall missing percentage recommendations
        overall_missing = missing_percentages.mean()

        if overall_missing < 5:
            recommendations.append(
                "Low missing data rate - simple imputation methods suitable"
            )
        elif overall_missing < 20:
            recommendations.append(
                "Moderate missing data - consider KNN or iterative imputation"
            )
        else:
            recommendations.append(
                "High missing data rate - advanced methods and careful validation needed"
            )

        # Pattern-specific recommendations
        if pattern_type == "MCAR":
            recommendations.append(
                "MCAR pattern detected - any imputation method appropriate"
            )
        elif pattern_type == "MAR":
            recommendations.append(
                "MAR pattern detected - multivariate methods recommended (KNN, Iterative)"
            )
        elif pattern_type == "MNAR":
            recommendations.append(
                "MNAR pattern detected - domain knowledge required, consider explicit missing indicators"
            )

        # Column-specific recommendations
        high_missing_cols = [
            col for col, perc in missing_percentages.items() if perc > 50
        ]
        if high_missing_cols:
            recommendations.append(
                f"Columns with >50% missing: {high_missing_cols} - consider removal or domain-specific imputation"
            )

        return recommendations

    def _calculate_pattern_confidence(
        self, pattern_correlations: pd.DataFrame
    ) -> float:
        """Calculate confidence score for pattern classification."""
        # Base confidence on correlation strength and consistency
        abs_correlations = pattern_correlations.abs().fillna(0)

        # Remove diagonal (self-correlations)
        mask = np.eye(len(abs_correlations), dtype=bool)
        off_diagonal = abs_correlations.values[~mask]

        if len(off_diagonal) == 0:
            return 0.5

        # Higher correlations = higher confidence in pattern classification
        mean_correlation = np.mean(off_diagonal)
        correlation_variance = np.var(off_diagonal)

        # Confidence increases with mean correlation but decreases with high variance
        confidence = mean_correlation * (1 - correlation_variance)

        return max(0.0, min(1.0, confidence))
