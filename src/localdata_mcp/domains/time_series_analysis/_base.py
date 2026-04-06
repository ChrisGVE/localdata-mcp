"""
Time Series Analysis - Base types.

Contains the foundational types: exception class and result dataclass
used by all time series analysis sub-modules.
"""

import warnings
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

# Suppress specific warnings that are not critical for time series analysis
warnings.filterwarnings("ignore", category=RuntimeWarning, module="statsmodels")
warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")
warnings.filterwarnings("ignore", category=FutureWarning, module="statsmodels")


class TimeSeriesValidationError(ValueError):
    """Custom exception for time series data validation errors."""

    pass


@dataclass
class TimeSeriesAnalysisResult:
    """Standardized result structure for time series analysis operations."""

    analysis_type: str
    timestamp: datetime = field(default_factory=datetime.now)

    # Core analysis results
    statistic: Optional[float] = None
    p_value: Optional[float] = None
    critical_values: Optional[Dict[str, float]] = None

    # Time series specific results
    frequency: Optional[str] = None
    seasonality_period: Optional[int] = None
    trend_component: Optional[pd.Series] = None
    seasonal_component: Optional[pd.Series] = None
    residual_component: Optional[pd.Series] = None

    # Forecasting results
    forecast_values: Optional[pd.Series] = None
    forecast_confidence_intervals: Optional[pd.DataFrame] = None
    forecast_horizon: Optional[int] = None

    # Model parameters and diagnostics
    model_parameters: Dict[str, Any] = field(default_factory=dict)
    model_diagnostics: Dict[str, Any] = field(default_factory=dict)
    fit_statistics: Dict[str, float] = field(default_factory=dict)

    # Interpretation and recommendations
    interpretation: str = ""
    recommendations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    # Additional metadata
    processing_time: float = 0.0
    data_quality_score: float = 0.0
    confidence_level: float = 0.95

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary format for JSON serialization."""
        result_dict = {
            "analysis_type": self.analysis_type,
            "timestamp": self.timestamp.isoformat(),
            "interpretation": self.interpretation,
            "processing_time": self.processing_time,
            "data_quality_score": self.data_quality_score,
            "confidence_level": self.confidence_level,
        }

        # Add non-None numeric results
        for field_name in [
            "statistic",
            "p_value",
            "frequency",
            "seasonality_period",
            "forecast_horizon",
        ]:
            value = getattr(self, field_name)
            if value is not None:
                result_dict[field_name] = value

        # Add dictionary fields
        for dict_field in [
            "critical_values",
            "model_parameters",
            "model_diagnostics",
            "fit_statistics",
        ]:
            value = getattr(self, dict_field)
            if value:
                result_dict[dict_field] = value

        # Add list fields
        for list_field in ["recommendations", "warnings"]:
            value = getattr(self, list_field)
            if value:
                result_dict[list_field] = value

        # Handle pandas Series/DataFrame fields
        if self.trend_component is not None:
            result_dict["trend_component"] = self.trend_component.to_dict()
        if self.seasonal_component is not None:
            result_dict["seasonal_component"] = self.seasonal_component.to_dict()
        if self.residual_component is not None:
            result_dict["residual_component"] = self.residual_component.to_dict()
        if self.forecast_values is not None:
            result_dict["forecast_values"] = self.forecast_values.to_dict()
        if self.forecast_confidence_intervals is not None:
            result_dict["forecast_confidence_intervals"] = (
                self.forecast_confidence_intervals.to_dict()
            )

        return result_dict
