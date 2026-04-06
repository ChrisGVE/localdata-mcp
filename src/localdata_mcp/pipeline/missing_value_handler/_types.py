"""
Missing Value Handler - Data Types

Dataclasses used across the missing value handling pipeline.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


@dataclass
class MissingValuePattern:
    """Analysis of missing value patterns in the dataset."""

    pattern_type: str  # "MCAR", "MAR", "MNAR"
    missing_percentage: float
    column_patterns: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    correlation_matrix: Optional[pd.DataFrame] = None
    temporal_patterns: Dict[str, Any] = field(default_factory=dict)
    confidence_score: float = 0.0
    recommendations: List[str] = field(default_factory=list)


@dataclass
class ImputationQuality:
    """Quality metrics for imputation methods."""

    strategy_name: str
    accuracy_score: float
    mse: float
    mae: float
    distribution_preservation: float
    correlation_preservation: float
    confidence_interval: Tuple[float, float]
    execution_time: float
    memory_usage: float


@dataclass
class ImputationMetadata:
    """Comprehensive metadata for imputation operations."""

    selected_strategy: str
    strategy_confidence: float
    missing_pattern: MissingValuePattern
    quality_assessment: Dict[str, ImputationQuality]
    cross_validation_results: Dict[str, Any] = field(default_factory=dict)
    imputation_log: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    reversibility_data: Dict[str, Any] = field(default_factory=dict)
