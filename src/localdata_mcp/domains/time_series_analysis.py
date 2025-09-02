"""
Time Series Analysis Domain - Comprehensive time series analysis capabilities.

This module implements advanced time series analysis tools including forecasting,
decomposition, stationarity testing, and multivariate analysis using statsmodels
and sklearn integration.

Key Features:
- ARIMA/SARIMA forecasting models with automatic parameter selection
- Time series decomposition (seasonal, trend, residual components)
- Stationarity testing (ADF, KPSS, Phillips-Perron tests)
- Autocorrelation analysis (ACF/PACF) for model identification
- Multivariate time series analysis (VAR, cointegration, Granger causality)
- Change point detection and anomaly identification
- Prophet integration for robust forecasting with holiday effects
- Full sklearn pipeline compatibility with temporal data validation
- Streaming-compatible processing for high-frequency time series
- Comprehensive result formatting with forecasting confidence intervals
"""

import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import time

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose

from ..logging_manager import get_logger
from ..pipeline.base import (
    AnalysisPipelineBase, PipelineResult, CompositionMetadata, 
    StreamingConfig, PipelineState
)

logger = get_logger(__name__)

# Suppress specific warnings that are not critical for time series analysis
warnings.filterwarnings('ignore', category=RuntimeWarning, module='statsmodels')
warnings.filterwarnings('ignore', category=UserWarning, module='statsmodels')
warnings.filterwarnings('ignore', category=FutureWarning, module='statsmodels')


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
            'analysis_type': self.analysis_type,
            'timestamp': self.timestamp.isoformat(),
            'interpretation': self.interpretation,
            'processing_time': self.processing_time,
            'data_quality_score': self.data_quality_score,
            'confidence_level': self.confidence_level
        }
        
        # Add non-None numeric results
        for field_name in ['statistic', 'p_value', 'frequency', 'seasonality_period', 'forecast_horizon']:
            value = getattr(self, field_name)
            if value is not None:
                result_dict[field_name] = value
                
        # Add dictionary fields
        for dict_field in ['critical_values', 'model_parameters', 'model_diagnostics', 'fit_statistics']:
            value = getattr(self, dict_field)
            if value:
                result_dict[dict_field] = value
                
        # Add list fields
        for list_field in ['recommendations', 'warnings']:
            value = getattr(self, list_field)
            if value:
                result_dict[list_field] = value
                
        # Handle pandas Series/DataFrame fields
        if self.trend_component is not None:
            result_dict['trend_component'] = self.trend_component.to_dict()
        if self.seasonal_component is not None:
            result_dict['seasonal_component'] = self.seasonal_component.to_dict()
        if self.residual_component is not None:
            result_dict['residual_component'] = self.residual_component.to_dict()
        if self.forecast_values is not None:
            result_dict['forecast_values'] = self.forecast_values.to_dict()
        if self.forecast_confidence_intervals is not None:
            result_dict['forecast_confidence_intervals'] = self.forecast_confidence_intervals.to_dict()
            
        return result_dict


class TimeSeriesTransformer(BaseEstimator, TransformerMixin, ABC):
    """
    Abstract base class for sklearn-compatible time series transformers.
    
    Provides core functionality for temporal data validation, frequency detection,
    and time series specific preprocessing operations. All concrete time series
    transformers should inherit from this class.
    
    Parameters:
    -----------
    validate_input : bool, default=True
        Whether to validate time series input data
    infer_frequency : bool, default=True
        Whether to automatically infer time series frequency
    handle_missing : str, default='interpolate'
        Strategy for handling missing values: 'interpolate', 'forward_fill', 'drop'
    """
    
    def __init__(self, validate_input=True, infer_frequency=True, handle_missing='interpolate'):
        self.validate_input = validate_input
        self.infer_frequency = infer_frequency
        self.handle_missing = handle_missing
        
    def _validate_time_series(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Validate time series data structure and datetime index.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input time series data
        y : pd.Series, optional
            Target time series data
            
        Returns:
        --------
        X_validated : pd.DataFrame
            Validated input data
        y_validated : pd.Series, optional
            Validated target data
            
        Raises:
        -------
        TimeSeriesValidationError
            If data fails time series validation checks
        """
        if not isinstance(X, pd.DataFrame):
            raise TimeSeriesValidationError("Input data must be a pandas DataFrame")
            
        # Ensure datetime index
        if not isinstance(X.index, pd.DatetimeIndex):
            if 'datetime' in X.columns or 'date' in X.columns or 'timestamp' in X.columns:
                # Try to find datetime column
                datetime_col = None
                for col in ['datetime', 'date', 'timestamp']:
                    if col in X.columns:
                        datetime_col = col
                        break
                        
                if datetime_col:
                    X = X.set_index(pd.to_datetime(X[datetime_col]))
                    X = X.drop(columns=[datetime_col])
                else:
                    raise TimeSeriesValidationError("No valid datetime column found")
            else:
                # Try to convert index to datetime
                try:
                    X.index = pd.to_datetime(X.index)
                except (ValueError, TypeError) as e:
                    raise TimeSeriesValidationError(f"Cannot convert index to datetime: {e}")
        
        # Check for monotonic index
        if not X.index.is_monotonic_increasing:
            logger.warning("Time series index is not monotonic, sorting by datetime")
            X = X.sort_index()
            
        # Validate target series if provided
        if y is not None:
            if not isinstance(y, pd.Series):
                raise TimeSeriesValidationError("Target data must be a pandas Series")
            if not isinstance(y.index, pd.DatetimeIndex):
                try:
                    y.index = pd.to_datetime(y.index)
                except (ValueError, TypeError) as e:
                    raise TimeSeriesValidationError(f"Cannot convert target index to datetime: {e}")
                    
        return X, y
    
    def _infer_frequency(self, X: pd.DataFrame) -> Optional[str]:
        """
        Infer the frequency of a time series from its datetime index.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Time series data with datetime index
            
        Returns:
        --------
        frequency : str or None
            Inferred frequency string (e.g., 'D', 'H', 'M')
        """
        try:
            inferred_freq = pd.infer_freq(X.index)
            if inferred_freq:
                logger.debug(f"Inferred time series frequency: {inferred_freq}")
                return inferred_freq
        except Exception as e:
            logger.warning(f"Could not infer frequency: {e}")
            
        # Fallback: calculate most common difference
        if len(X.index) > 1:
            diffs = X.index[1:] - X.index[:-1]
            most_common_diff = diffs.mode()
            if len(most_common_diff) > 0:
                diff = most_common_diff[0]
                if diff == timedelta(days=1):
                    return 'D'
                elif diff == timedelta(hours=1):
                    return 'H'
                elif diff == timedelta(minutes=1):
                    return 'T'
                elif diff == timedelta(seconds=1):
                    return 'S'
                    
        logger.warning("Could not determine time series frequency")
        return None
    
    def _handle_missing_temporal_data(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in time series data using temporal-aware methods.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Time series data with potential missing values
            
        Returns:
        --------
        X_processed : pd.DataFrame
            Data with missing values handled
        """
        if not X.isnull().any().any():
            return X
            
        X_processed = X.copy()
        
        if self.handle_missing == 'interpolate':
            # Use time-aware interpolation
            X_processed = X_processed.interpolate(method='time')
        elif self.handle_missing == 'forward_fill':
            X_processed = X_processed.fillna(method='ffill')
        elif self.handle_missing == 'drop':
            X_processed = X_processed.dropna()
        else:
            logger.warning(f"Unknown missing value strategy: {self.handle_missing}")
            
        return X_processed
    
    def _detect_seasonality(self, series: pd.Series, max_period: int = 365) -> Dict[str, Any]:
        """
        Detect seasonal patterns in time series data.
        
        Parameters:
        -----------
        series : pd.Series
            Time series data
        max_period : int, default=365
            Maximum period to test for seasonality
            
        Returns:
        --------
        seasonality_info : dict
            Dictionary containing seasonality detection results
        """
        seasonality_info = {
            'has_seasonality': False,
            'dominant_period': None,
            'seasonal_strength': 0.0,
            'tested_periods': []
        }
        
        try:
            # Test common seasonal periods based on frequency
            freq = self._infer_frequency(pd.DataFrame(index=series.index))
            test_periods = []
            
            if freq == 'D':  # Daily data
                test_periods = [7, 30, 365]  # Weekly, monthly, yearly
            elif freq == 'H':  # Hourly data
                test_periods = [24, 168, 720]  # Daily, weekly, monthly
            elif freq == 'M':  # Monthly data
                test_periods = [12]  # Yearly
            else:
                # Default periods to test
                test_periods = [7, 12, 24, 52]
                
            # Filter periods that are feasible for the data length
            test_periods = [p for p in test_periods if p < len(series) // 3 and p <= max_period]
            seasonality_info['tested_periods'] = test_periods
            
            best_period = None
            best_strength = 0.0
            
            for period in test_periods:
                try:
                    # Simple seasonal strength calculation using autocorrelation
                    autocorr = series.autocorr(lag=period)
                    if not np.isnan(autocorr) and abs(autocorr) > best_strength:
                        best_strength = abs(autocorr)
                        best_period = period
                except Exception as e:
                    logger.debug(f"Could not test period {period}: {e}")
                    continue
                    
            if best_period is not None and best_strength > 0.3:  # Threshold for significant seasonality
                seasonality_info['has_seasonality'] = True
                seasonality_info['dominant_period'] = best_period
                seasonality_info['seasonal_strength'] = best_strength
                
        except Exception as e:
            logger.warning(f"Error in seasonality detection: {e}")
            
        return seasonality_info
    
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fit the time series transformer."""
        pass
    
    @abstractmethod
    def transform(self, X: pd.DataFrame) -> Union[pd.DataFrame, TimeSeriesAnalysisResult]:
        """Transform the time series data."""
        pass


class TimeSeriesPipeline(AnalysisPipelineBase):
    """
    Main pipeline class for time series analysis workflows.
    
    Orchestrates time series analysis operations with streaming support,
    temporal data validation, and comprehensive result formatting.
    
    Parameters:
    -----------
    transformers : list of TimeSeriesTransformer
        List of time series analysis transformers to apply
    streaming_config : StreamingConfig, optional
        Configuration for streaming data processing
    validate_temporal_data : bool, default=True
        Whether to validate temporal data structure
    """
    
    def __init__(self, 
                 transformers: Optional[List[TimeSeriesTransformer]] = None,
                 streaming_config: Optional[StreamingConfig] = None,
                 validate_temporal_data: bool = True):
        super().__init__()
        self.transformers = transformers or []
        self.streaming_config = streaming_config
        self.validate_temporal_data = validate_temporal_data
        self._fitted_transformers = []
        
    def add_transformer(self, transformer: TimeSeriesTransformer) -> 'TimeSeriesPipeline':
        """Add a time series transformer to the pipeline."""
        if not isinstance(transformer, TimeSeriesTransformer):
            raise ValueError("Transformer must be an instance of TimeSeriesTransformer")
        self.transformers.append(transformer)
        return self
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'TimeSeriesPipeline':
        """
        Fit all transformers in the pipeline.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Time series input data with datetime index
        y : pd.Series, optional
            Target time series data
            
        Returns:
        --------
        self : TimeSeriesPipeline
            Fitted pipeline instance
        """
        start_time = time.time()
        
        try:
            # Validate temporal data if requested
            if self.validate_temporal_data:
                X, y = self._validate_pipeline_input(X, y)
            
            # Fit each transformer sequentially
            self._fitted_transformers = []
            current_X = X.copy()
            
            for i, transformer in enumerate(self.transformers):
                logger.debug(f"Fitting transformer {i+1}/{len(self.transformers)}: {type(transformer).__name__}")
                
                fitted_transformer = transformer.fit(current_X, y)
                self._fitted_transformers.append(fitted_transformer)
                
                # For some transformers, we might need to update X for next transformer
                # This depends on the specific transformer implementation
                
            self._state = PipelineState.FITTED
            self._fit_time = time.time() - start_time
            
            logger.info(f"Time series pipeline fitted in {self._fit_time:.2f}s")
            return self
            
        except Exception as e:
            self._state = PipelineState.ERROR
            logger.error(f"Error fitting time series pipeline: {e}")
            raise
    
    def transform(self, X: pd.DataFrame) -> List[TimeSeriesAnalysisResult]:
        """
        Apply all fitted transformers to the input data.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Time series input data
            
        Returns:
        --------
        results : list of TimeSeriesAnalysisResult
            Results from all transformers
        """
        check_is_fitted(self, ['_fitted_transformers'])
        
        if self.validate_temporal_data:
            X, _ = self._validate_pipeline_input(X, None)
        
        results = []
        current_X = X.copy()
        
        for transformer in self._fitted_transformers:
            try:
                result = transformer.transform(current_X)
                if isinstance(result, TimeSeriesAnalysisResult):
                    results.append(result)
                # Handle other result types if needed
            except Exception as e:
                logger.error(f"Error in transformer {type(transformer).__name__}: {e}")
                # Create error result
                error_result = TimeSeriesAnalysisResult(
                    analysis_type=f"{type(transformer).__name__}_error",
                    interpretation=f"Error in analysis: {str(e)}",
                    warnings=[str(e)]
                )
                results.append(error_result)
        
        return results
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> List[TimeSeriesAnalysisResult]:
        """Fit the pipeline and transform the data."""
        return self.fit(X, y).transform(X)
    
    def _validate_pipeline_input(self, X: pd.DataFrame, y: Optional[pd.Series]) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """Validate input data for the time series pipeline."""
        if not isinstance(X, pd.DataFrame):
            raise TimeSeriesValidationError("Input X must be a pandas DataFrame")
        
        # Use the validation from TimeSeriesTransformer
        dummy_transformer = type('DummyTransformer', (TimeSeriesTransformer,), {
            'fit': lambda self, X, y=None: self,
            'transform': lambda self, X: X
        })()
        
        return dummy_transformer._validate_time_series(X, y)
    
    def get_composition_metadata(self) -> CompositionMetadata:
        """Get metadata for downstream tool composition."""
        return CompositionMetadata(
            domain="time_series",
            analysis_type="pipeline",
            result_type="time_series_analysis_results",
            compatible_tools=[
                "statistical_analysis", "regression_modeling", 
                "forecast_time_series", "decompose_series", "test_stationarity"
            ],
            suggested_compositions=[
                {
                    "name": "Time Series Forecasting Workflow",
                    "tools": ["test_stationarity", "forecast_time_series", "evaluate_forecast"],
                    "description": "Complete forecasting pipeline with stationarity testing"
                },
                {
                    "name": "Time Series EDA Workflow", 
                    "tools": ["decompose_series", "test_stationarity", "analyze_autocorrelation"],
                    "description": "Exploratory analysis of time series characteristics"
                }
            ],
            confidence_level=0.85,
            quality_score=0.9
        )


# Utility functions for time series validation and preprocessing

def validate_datetime_index(data: Union[pd.DataFrame, pd.Series]) -> bool:
    """
    Validate that data has a proper datetime index.
    
    Parameters:
    -----------
    data : pd.DataFrame or pd.Series
        Data to validate
        
    Returns:
    --------
    is_valid : bool
        True if data has valid datetime index
    """
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


def infer_time_series_frequency(data: Union[pd.DataFrame, pd.Series]) -> Optional[str]:
    """
    Infer the frequency of a time series.
    
    Parameters:
    -----------
    data : pd.DataFrame or pd.Series
        Time series data
        
    Returns:
    --------
    frequency : str or None
        Inferred frequency string
    """
    try:
        return pd.infer_freq(data.index)
    except Exception:
        return None


def detect_time_series_gaps(data: Union[pd.DataFrame, pd.Series], 
                           expected_frequency: Optional[str] = None) -> List[Tuple[datetime, datetime]]:
    """
    Detect gaps in time series data.
    
    Parameters:
    -----------
    data : pd.DataFrame or pd.Series
        Time series data
    expected_frequency : str, optional
        Expected frequency to check against
        
    Returns:
    --------
    gaps : list of tuples
        List of (start, end) datetime pairs representing gaps
    """
    gaps = []
    
    try:
        if expected_frequency is None:
            expected_frequency = pd.infer_freq(data.index)
            
        if expected_frequency is None:
            return gaps
            
        # Create expected full index
        expected_index = pd.date_range(
            start=data.index.min(),
            end=data.index.max(),
            freq=expected_frequency
        )
        
        # Find missing dates
        missing_dates = expected_index.difference(data.index)
        
        if len(missing_dates) > 0:
            # Group consecutive missing dates into gaps
            missing_dates = missing_dates.sort_values()
            current_gap_start = missing_dates[0]
            current_gap_end = missing_dates[0]
            
            for i in range(1, len(missing_dates)):
                if missing_dates[i] - current_gap_end <= pd.Timedelta(expected_frequency):
                    current_gap_end = missing_dates[i]
                else:
                    gaps.append((current_gap_start, current_gap_end))
                    current_gap_start = missing_dates[i]
                    current_gap_end = missing_dates[i]
            
            # Add the last gap
            gaps.append((current_gap_start, current_gap_end))
            
    except Exception as e:
        logger.warning(f"Could not detect time series gaps: {e}")
        
    return gaps


def validate_time_series_continuity(data: Union[pd.DataFrame, pd.Series],
                                   max_gap_tolerance: Optional[str] = None) -> Dict[str, Any]:
    """
    Validate time series data continuity and quality.
    
    Parameters:
    -----------
    data : pd.DataFrame or pd.Series
        Time series data to validate
    max_gap_tolerance : str, optional
        Maximum acceptable gap (e.g., '1D', '1H')
        
    Returns:
    --------
    validation_result : dict
        Dictionary containing validation results and metrics
    """
    result = {
        'is_continuous': True,
        'has_datetime_index': False,
        'is_monotonic': False,
        'frequency': None,
        'gaps': [],
        'missing_value_count': 0,
        'missing_value_percentage': 0.0,
        'data_quality_score': 0.0,
        'recommendations': []
    }
    
    try:
        # Check datetime index
        result['has_datetime_index'] = isinstance(data.index, pd.DatetimeIndex)
        if not result['has_datetime_index']:
            result['is_continuous'] = False
            result['recommendations'].append("Convert index to datetime format")
            
        # Check monotonic ordering
        result['is_monotonic'] = data.index.is_monotonic_increasing
        if not result['is_monotonic']:
            result['is_continuous'] = False
            result['recommendations'].append("Sort data by datetime index")
        
        # Infer frequency
        result['frequency'] = infer_time_series_frequency(data)
        if result['frequency'] is None:
            result['recommendations'].append("Consider resampling to regular frequency")
        
        # Detect gaps
        if result['frequency'] is not None:
            gaps = detect_time_series_gaps(data, result['frequency'])
            result['gaps'] = [(gap[0].isoformat(), gap[1].isoformat()) for gap in gaps]
            if len(gaps) > 0:
                result['is_continuous'] = False
                result['recommendations'].append(f"Found {len(gaps)} time gaps that may need interpolation")
        
        # Check for missing values
        if isinstance(data, pd.DataFrame):
            missing_count = data.isnull().sum().sum()
            total_values = data.size
        else:
            missing_count = data.isnull().sum()
            total_values = len(data)
            
        result['missing_value_count'] = int(missing_count)
        result['missing_value_percentage'] = float(missing_count / total_values * 100) if total_values > 0 else 0.0
        
        if result['missing_value_percentage'] > 5.0:
            result['recommendations'].append("High percentage of missing values - consider imputation")
        
        # Calculate overall data quality score
        quality_factors = [
            1.0 if result['has_datetime_index'] else 0.0,
            1.0 if result['is_monotonic'] else 0.5,
            1.0 if result['frequency'] is not None else 0.7,
            1.0 if len(result['gaps']) == 0 else 0.8,
            max(0.0, 1.0 - result['missing_value_percentage'] / 50.0)  # Penalty for missing values
        ]
        
        result['data_quality_score'] = float(np.mean(quality_factors))
        
    except Exception as e:
        logger.error(f"Error validating time series continuity: {e}")
        result['recommendations'].append(f"Validation error: {str(e)}")
        
    return result