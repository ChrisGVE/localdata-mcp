"""Data quality assessment for response metadata.

Provides functions for evaluating data completeness, consistency, validity,
and overall quality of DataFrames.
"""

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from .models import (
    DataQualityLevel,
    DataQualityMetrics,
)


def assess_data_quality(df: pd.DataFrame) -> DataQualityMetrics:
    """Assess comprehensive data quality metrics for a DataFrame.

    Args:
        df: DataFrame to assess.

    Returns:
        Data quality metrics dataclass.
    """
    if df.empty:
        return DataQualityMetrics(
            overall_quality=DataQualityLevel.UNKNOWN,
            quality_score=0.0,
            completeness=0.0,
            consistency=0.0,
            validity=0.0,
            accuracy=0.0,
        )

    completeness = _compute_completeness(df)
    consistency = _compute_consistency(df)
    validity, issues = _compute_validity(df)
    accuracy = (completeness + consistency + validity) / 3.0

    quality_score = (
        completeness * 0.4 + consistency * 0.2 + validity * 0.2 + accuracy * 0.2
    )

    overall_quality = _quality_level_from_score(quality_score)

    recommendations: List[str] = []
    if completeness < 0.8:
        recommendations.append("Consider handling missing values")
    if consistency < 0.7:
        recommendations.append("Check data format consistency")
    if len(issues) > 0:
        recommendations.append("Review data for outliers and anomalies")

    return DataQualityMetrics(
        overall_quality=overall_quality,
        quality_score=quality_score,
        completeness=completeness,
        consistency=consistency,
        validity=validity,
        accuracy=accuracy,
        issues=issues,
        recommendations=recommendations,
    )


def _compute_completeness(df: pd.DataFrame) -> float:
    """Compute completeness as ratio of non-null values."""
    total_values = df.size
    non_null_values = df.count().sum()
    return (non_null_values / total_values) if total_values > 0 else 0.0


def _compute_consistency(df: pd.DataFrame) -> float:
    """Compute data format consistency across columns."""
    consistency_scores = []
    for col in df.columns:
        if df[col].dtype == "object":
            non_null_values = df[col].dropna()
            if len(non_null_values) > 0:
                lengths = non_null_values.astype(str).str.len()
                if len(lengths) > 1:
                    cv = lengths.std() / lengths.mean() if lengths.mean() > 0 else 1.0
                    consistency_scores.append(max(0.0, 1.0 - min(1.0, cv)))
                else:
                    consistency_scores.append(1.0)
        else:
            consistency_scores.append(1.0)

    return float(np.mean(consistency_scores)) if consistency_scores else 0.0


def _compute_validity(df: pd.DataFrame) -> Tuple[float, List[Dict[str, Any]]]:
    """Compute validity scores and collect quality issues."""
    validity_scores: list[float] = []
    issues: list[Dict[str, Any]] = []

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_bool_dtype(
            df[col]
        ):
            q1, q3 = df[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            if iqr > 0:
                outliers = (
                    (df[col] < (q1 - 3 * iqr)) | (df[col] > (q3 + 3 * iqr))
                ).sum()
                validity_score = max(0.0, 1.0 - (outliers / len(df)))
                validity_scores.append(validity_score)
                if outliers > len(df) * 0.05:
                    issues.append(
                        {
                            "type": "outliers",
                            "column": col,
                            "count": int(outliers),
                            "percentage": outliers / len(df) * 100,
                        }
                    )
            else:
                validity_scores.append(1.0)
        else:
            validity_scores.append(1.0)

    validity = float(np.mean(validity_scores)) if validity_scores else 1.0
    return validity, issues


def _quality_level_from_score(quality_score: float) -> DataQualityLevel:
    """Map a numeric quality score to a DataQualityLevel."""
    if quality_score >= 0.9:
        return DataQualityLevel.EXCELLENT
    if quality_score >= 0.7:
        return DataQualityLevel.GOOD
    if quality_score >= 0.5:
        return DataQualityLevel.FAIR
    return DataQualityLevel.POOR
