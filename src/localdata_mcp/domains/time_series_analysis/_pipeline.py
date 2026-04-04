"""
Time Series Analysis - Pipeline and utility functions.

Contains the TimeSeriesPipeline orchestration class and standalone
utility functions for time series validation and preprocessing.
"""

import time
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.utils.validation import check_is_fitted

from ...logging_manager import get_logger
from ...pipeline.base import (
    AnalysisPipelineBase,
    CompositionMetadata,
    StreamingConfig,
    PipelineState,
)
from ._base import TimeSeriesAnalysisResult, TimeSeriesValidationError
from ._transformer import TimeSeriesTransformer

logger = get_logger(__name__)


class TimeSeriesPipeline(AnalysisPipelineBase):
    """
    Main pipeline class for time series analysis workflows.

    Orchestrates time series analysis operations with streaming support,
    temporal data validation, and comprehensive result formatting.
    """

    def __init__(
        self,
        transformers: Optional[List[TimeSeriesTransformer]] = None,
        streaming_config: Optional[StreamingConfig] = None,
        validate_temporal_data: bool = True,
    ):
        super().__init__()
        self.transformers = transformers or []
        self.streaming_config = streaming_config
        self.validate_temporal_data = validate_temporal_data
        self._fitted_transformers: List[TimeSeriesTransformer] = []

    def add_transformer(
        self, transformer: TimeSeriesTransformer
    ) -> "TimeSeriesPipeline":
        """Add a time series transformer to the pipeline."""
        if not isinstance(transformer, TimeSeriesTransformer):
            raise ValueError("Transformer must be an instance of TimeSeriesTransformer")
        self.transformers.append(transformer)
        return self

    def fit(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> "TimeSeriesPipeline":
        """Fit all transformers in the pipeline."""
        start_time = time.time()

        try:
            if self.validate_temporal_data:
                X, y = self._validate_pipeline_input(X, y)

            self._fitted_transformers = []
            current_X = X.copy()

            for i, transformer in enumerate(self.transformers):
                logger.debug(
                    f"Fitting transformer {i + 1}/{len(self.transformers)}: "
                    f"{type(transformer).__name__}"
                )
                fitted_transformer = transformer.fit(current_X, y)
                self._fitted_transformers.append(fitted_transformer)

            self._state = PipelineState.FITTED
            self._fit_time = time.time() - start_time
            logger.info(f"Time series pipeline fitted in {self._fit_time:.2f}s")
            self.is_fitted_ = True
            return self

        except Exception as e:
            self._state = PipelineState.ERROR
            logger.error(f"Error fitting time series pipeline: {e}")
            raise

    def transform(self, X: pd.DataFrame) -> List[TimeSeriesAnalysisResult]:
        """Apply all fitted transformers to the input data."""
        check_is_fitted(self, ["_fitted_transformers"])

        if self.validate_temporal_data:
            X, _ = self._validate_pipeline_input(X, None)

        results: List[TimeSeriesAnalysisResult] = []
        current_X = X.copy()

        for transformer in self._fitted_transformers:
            try:
                result = transformer.transform(current_X)
                if isinstance(result, TimeSeriesAnalysisResult):
                    results.append(result)
            except Exception as e:
                logger.error(f"Error in transformer {type(transformer).__name__}: {e}")
                error_result = TimeSeriesAnalysisResult(
                    analysis_type=f"{type(transformer).__name__}_error",
                    interpretation=f"Error in analysis: {str(e)}",
                    warnings=[str(e)],
                )
                results.append(error_result)

        return results

    def fit_transform(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> List[TimeSeriesAnalysisResult]:
        """Fit the pipeline and transform the data."""
        return self.fit(X, y).transform(X)

    def _validate_pipeline_input(
        self, X: pd.DataFrame, y: Optional[pd.Series]
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """Validate input data for the time series pipeline."""
        if not isinstance(X, pd.DataFrame):
            raise TimeSeriesValidationError("Input X must be a pandas DataFrame")

        dummy_transformer = type(
            "DummyTransformer",
            (TimeSeriesTransformer,),
            {"fit": lambda self, X, y=None: self, "transform": lambda self, X: X},
        )()
        return dummy_transformer._validate_time_series(X, y)

    def get_composition_metadata(self) -> CompositionMetadata:
        """Get metadata for downstream tool composition."""
        return CompositionMetadata(
            domain="time_series",
            analysis_type="pipeline",
            result_type="time_series_analysis_results",
            compatible_tools=[
                "statistical_analysis",
                "regression_modeling",
                "forecast_time_series",
                "decompose_series",
                "test_stationarity",
            ],
            suggested_compositions=[
                {
                    "name": "Time Series Forecasting Workflow",
                    "tools": [
                        "test_stationarity",
                        "forecast_time_series",
                        "evaluate_forecast",
                    ],
                    "description": "Complete forecasting pipeline with stationarity testing",
                },
                {
                    "name": "Time Series EDA Workflow",
                    "tools": [
                        "decompose_series",
                        "test_stationarity",
                        "analyze_autocorrelation",
                    ],
                    "description": "Exploratory analysis of time series characteristics",
                },
            ],
            confidence_level=0.85,
            quality_score=0.9,
        )


def validate_datetime_index(data: Union[pd.DataFrame, pd.Series]) -> bool:
    """Validate that data has a proper datetime index."""
    try:
        if not isinstance(data.index, pd.DatetimeIndex):
            return False
        if data.index.isnull().any():
            return False
        if not data.index.is_monotonic_increasing:
            return False
        return True
    except Exception:
        return False


def infer_time_series_frequency(
    data: Union[pd.DataFrame, pd.Series],
) -> Optional[str]:
    """Infer the frequency of a time series."""
    try:
        return pd.infer_freq(data.index)
    except Exception:
        return None


def detect_time_series_gaps(
    data: Union[pd.DataFrame, pd.Series], expected_frequency: Optional[str] = None
) -> List[Tuple[datetime, datetime]]:
    """Detect gaps in time series data."""
    gaps: List[Tuple[datetime, datetime]] = []

    try:
        if expected_frequency is None:
            expected_frequency = pd.infer_freq(data.index)

        if expected_frequency is None:
            return gaps

        expected_index = pd.date_range(
            start=data.index.min(), end=data.index.max(), freq=expected_frequency
        )
        missing_dates = expected_index.difference(data.index)

        if len(missing_dates) > 0:
            missing_dates = missing_dates.sort_values()
            current_gap_start = missing_dates[0]
            current_gap_end = missing_dates[0]

            for i in range(1, len(missing_dates)):
                if missing_dates[i] - current_gap_end <= pd.Timedelta(
                    expected_frequency
                ):
                    current_gap_end = missing_dates[i]
                else:
                    gaps.append((current_gap_start, current_gap_end))
                    current_gap_start = missing_dates[i]
                    current_gap_end = missing_dates[i]

            gaps.append((current_gap_start, current_gap_end))

    except Exception as e:
        logger.warning(f"Could not detect time series gaps: {e}")

    return gaps


def validate_time_series_continuity(
    data: Union[pd.DataFrame, pd.Series], max_gap_tolerance: Optional[str] = None
) -> Dict[str, Any]:
    """Validate time series data continuity and quality."""
    result: Dict[str, Any] = {
        "is_continuous": True,
        "has_datetime_index": False,
        "is_monotonic": False,
        "frequency": None,
        "gaps": [],
        "missing_value_count": 0,
        "missing_value_percentage": 0.0,
        "data_quality_score": 0.0,
        "recommendations": [],
    }

    try:
        result["has_datetime_index"] = isinstance(data.index, pd.DatetimeIndex)
        if not result["has_datetime_index"]:
            result["is_continuous"] = False
            result["recommendations"].append("Convert index to datetime format")

        result["is_monotonic"] = data.index.is_monotonic_increasing
        if not result["is_monotonic"]:
            result["is_continuous"] = False
            result["recommendations"].append("Sort data by datetime index")

        result["frequency"] = infer_time_series_frequency(data)
        if result["frequency"] is None:
            result["recommendations"].append("Consider resampling to regular frequency")

        if result["frequency"] is not None:
            gaps = detect_time_series_gaps(data, result["frequency"])
            result["gaps"] = [(gap[0].isoformat(), gap[1].isoformat()) for gap in gaps]
            if len(gaps) > 0:
                result["is_continuous"] = False
                result["recommendations"].append(
                    f"Found {len(gaps)} time gaps that may need interpolation"
                )

        if isinstance(data, pd.DataFrame):
            missing_count = data.isnull().sum().sum()
            total_values = data.size
        else:
            missing_count = data.isnull().sum()
            total_values = len(data)

        result["missing_value_count"] = int(missing_count)
        result["missing_value_percentage"] = (
            float(missing_count / total_values * 100) if total_values > 0 else 0.0
        )

        if result["missing_value_percentage"] > 5.0:
            result["recommendations"].append(
                "High percentage of missing values - consider imputation"
            )

        quality_factors = [
            1.0 if result["has_datetime_index"] else 0.0,
            1.0 if result["is_monotonic"] else 0.5,
            1.0 if result["frequency"] is not None else 0.7,
            1.0 if len(result["gaps"]) == 0 else 0.8,
            max(0.0, 1.0 - result["missing_value_percentage"] / 50.0),
        ]
        result["data_quality_score"] = float(np.mean(quality_factors))

    except Exception as e:
        logger.error(f"Error validating time series continuity: {e}")
        result["recommendations"].append(f"Validation error: {str(e)}")

    return result
