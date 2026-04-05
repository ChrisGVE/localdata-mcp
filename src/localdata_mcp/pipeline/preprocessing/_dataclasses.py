"""
Preprocessing dataclasses and transformation strategies.

Contains shared data structures and strategy selectors used across
preprocessing pipeline modules.
"""

from typing import Any, Dict, Optional
from dataclasses import dataclass, field

import pandas as pd
import numpy as np
import re


@dataclass
class DataQualityMetrics:
    """Comprehensive data quality assessment metrics."""

    # Completeness metrics
    completeness_score: float = 0.0
    missing_value_percentage: float = 0.0

    # Consistency metrics
    consistency_score: float = 0.0
    duplicate_percentage: float = 0.0

    # Validity metrics
    validity_score: float = 0.0
    type_conformity_percentage: float = 0.0

    # Accuracy metrics (outlier detection)
    accuracy_score: float = 0.0
    outlier_percentage: float = 0.0

    # Overall quality score
    overall_quality_score: float = 0.0

    # Business rules compliance
    business_rules_compliance: float = 0.0

    # Data profile summary
    data_profile: Dict[str, Any] = field(default_factory=dict)

    def calculate_overall_score(self) -> float:
        """Calculate overall data quality score from component metrics."""
        scores = [
            self.completeness_score,
            self.consistency_score,
            self.validity_score,
            self.accuracy_score,
            self.business_rules_compliance,
        ]
        self.overall_quality_score = np.mean([s for s in scores if s > 0])
        return self.overall_quality_score


@dataclass
class CleaningOperation:
    """Record of a data cleaning operation for transparency and reversibility."""

    operation_type: str
    column: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    records_affected: int = 0
    execution_time: float = 0.0
    success: bool = True
    error_message: Optional[str] = None
    before_stats: Dict[str, Any] = field(default_factory=dict)
    after_stats: Dict[str, Any] = field(default_factory=dict)
    reversibility_data: Dict[str, Any] = field(default_factory=dict)


class TransformationStrategy:
    """Strategies for different preprocessing transformations."""

    @staticmethod
    def missing_values_auto(data: pd.DataFrame) -> str:
        """Automatically determine missing value strategy."""
        numeric_cols = data.select_dtypes(include=["number"]).columns
        categorical_cols = data.select_dtypes(include=["object", "category"]).columns

        strategies = []
        if len(numeric_cols) > 0:
            strategies.append("numeric_median")
        if len(categorical_cols) > 0:
            strategies.append("categorical_mode")

        return (
            "mixed" if len(strategies) > 1 else strategies[0] if strategies else "none"
        )

    @staticmethod
    def outlier_detection_auto(data: pd.DataFrame) -> str:
        """Automatically determine outlier detection strategy."""
        numeric_cols = data.select_dtypes(include=["number"]).columns

        if len(numeric_cols) == 0:
            return "none"

        # Use IQR method for most cases, Z-score for large datasets
        if len(data) > 10000:
            return "zscore"
        else:
            return "iqr"

    @staticmethod
    def encoding_strategy_auto(data: pd.DataFrame) -> str:
        """Automatically determine categorical encoding strategy."""
        categorical_cols = data.select_dtypes(include=["object", "category"]).columns

        if len(categorical_cols) == 0:
            return "none"

        # Check cardinality to decide between label encoding and one-hot encoding
        high_cardinality_threshold = 10
        strategies = []

        for col in categorical_cols:
            cardinality = data[col].nunique()
            if cardinality <= high_cardinality_threshold:
                strategies.append("onehot")
            else:
                strategies.append("label")

        # Return most common strategy
        return max(set(strategies), key=strategies.count) if strategies else "none"

    @staticmethod
    def duplicate_detection_strategy(data: pd.DataFrame) -> str:
        """Automatically determine duplicate detection strategy."""
        # For small datasets, use exact matching
        if len(data) < 1000:
            return "exact"
        # For larger datasets, use hash-based detection for efficiency
        elif len(data) < 50000:
            return "hash_based"
        else:
            return "sampling_based"

    @staticmethod
    def data_type_inference_strategy(data: pd.DataFrame) -> Dict[str, str]:
        """Automatically determine data type inference strategies per column."""
        strategies = {}

        for col in data.columns:
            if data[col].dtype == "object":
                # Check if it might be datetime
                sample_values = data[col].dropna().astype(str).head(100)
                if len(sample_values) > 0:
                    datetime_patterns = [
                        r"\d{4}-\d{2}-\d{2}",  # YYYY-MM-DD
                        r"\d{2}/\d{2}/\d{4}",  # MM/DD/YYYY
                        r"\d{2}-\d{2}-\d{4}",  # MM-DD-YYYY
                    ]
                    is_datetime = any(
                        re.search(pattern, str(val))
                        for val in sample_values[:10]
                        for pattern in datetime_patterns
                    )

                    if is_datetime:
                        strategies[col] = "datetime"
                    else:
                        # Try numeric conversion
                        try:
                            pd.to_numeric(sample_values, errors="raise")
                            strategies[col] = "numeric"
                        except:
                            strategies[col] = "categorical"
                else:
                    strategies[col] = "categorical"
            else:
                strategies[col] = "preserve"

        return strategies
