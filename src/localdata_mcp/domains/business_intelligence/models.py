"""
Business Intelligence Domain - Data models, enums, and result structures.

This module defines the core data structures used across the business
intelligence domain including enums for attribution models and experiment
status, and dataclass result structures for all BI analysis types.
"""

from typing import Any, Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum

import pandas as pd


class AttributionModel(Enum):
    """Attribution model types for marketing analysis."""

    FIRST_TOUCH = "first_touch"
    LAST_TOUCH = "last_touch"
    LINEAR = "linear"
    TIME_DECAY = "time_decay"
    POSITION_BASED = "position_based"


class ExperimentStatus(Enum):
    """Status of A/B test experiments."""

    PLANNING = "planning"
    RUNNING = "running"
    COMPLETED = "completed"
    STOPPED = "stopped"


@dataclass
class RFMResult:
    """Result structure for RFM analysis."""

    rfm_scores: pd.DataFrame
    segments: pd.DataFrame
    segment_summary: pd.DataFrame
    quartile_boundaries: Dict[str, List[float]]

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary format."""
        return {
            "rfm_scores": self.rfm_scores.to_dict("records"),
            "segments": self.segments.to_dict("records"),
            "segment_summary": self.segment_summary.to_dict("records"),
            "quartile_boundaries": self.quartile_boundaries,
        }


@dataclass
class CohortAnalysisResult:
    """Result structure for cohort analysis."""

    cohort_table: pd.DataFrame
    cohort_sizes: pd.DataFrame
    retention_rates: pd.DataFrame
    period_summary: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary format."""
        return {
            "cohort_table": self.cohort_table.to_dict(),
            "cohort_sizes": self.cohort_sizes.to_dict(),
            "retention_rates": self.retention_rates.to_dict(),
            "period_summary": self.period_summary,
        }


@dataclass
class CLVResult:
    """Result structure for Customer Lifetime Value analysis."""

    clv_scores: pd.DataFrame
    model_metrics: Dict[str, float]
    clv_distribution: Dict[str, float]
    segment_clv: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary format."""
        return {
            "clv_scores": self.clv_scores.to_dict("records"),
            "model_metrics": self.model_metrics,
            "clv_distribution": self.clv_distribution,
            "segment_clv": self.segment_clv,
        }


@dataclass
class ABTestResult:
    """Result structure for A/B test analysis."""

    test_name: str
    test_statistic: float
    p_value: float
    confidence_interval: Tuple[float, float]
    effect_size: float
    power: float
    conclusion: str
    sample_sizes: Dict[str, int]
    conversion_rates: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary format."""
        return {
            "test_name": self.test_name,
            "test_statistic": self.test_statistic,
            "p_value": self.p_value,
            "confidence_interval": self.confidence_interval,
            "effect_size": self.effect_size,
            "power": self.power,
            "conclusion": self.conclusion,
            "sample_sizes": self.sample_sizes,
            "conversion_rates": self.conversion_rates,
        }


@dataclass
class AttributionResult:
    """Result structure for attribution modeling."""

    attribution_weights: pd.DataFrame
    channel_attribution: pd.DataFrame
    model_comparison: Dict[str, Dict[str, float]]
    conversion_paths: pd.DataFrame

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary format."""
        return {
            "attribution_weights": self.attribution_weights.to_dict("records"),
            "channel_attribution": self.channel_attribution.to_dict("records"),
            "model_comparison": self.model_comparison,
            "conversion_paths": self.conversion_paths.to_dict("records"),
        }


@dataclass
class FunnelAnalysisResult:
    """Result structure for funnel analysis."""

    funnel_steps: pd.DataFrame
    conversion_rates: Dict[str, float]
    drop_off_rates: Dict[str, float]
    bottlenecks: List[str]
    optimization_recommendations: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary format."""
        return {
            "funnel_steps": self.funnel_steps.to_dict("records"),
            "conversion_rates": self.conversion_rates,
            "drop_off_rates": self.drop_off_rates,
            "bottlenecks": self.bottlenecks,
            "optimization_recommendations": self.optimization_recommendations,
        }
