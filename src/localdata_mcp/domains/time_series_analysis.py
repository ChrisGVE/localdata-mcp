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
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf, pp_test
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy import stats
import itertools

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


class TimeSeriesResamplingTransformer(TimeSeriesTransformer):
    """
    sklearn-compatible transformer for time series resampling operations.
    
    Resamples time series data to a specified frequency with aggregation functions.
    Useful for converting high-frequency data to lower frequencies or handling
    irregular time series data.
    
    Parameters:
    -----------
    target_frequency : str
        Target frequency for resampling (e.g., 'D', 'H', 'W', 'M')
    aggregation_method : str or dict, default='mean'
        Aggregation method: 'mean', 'sum', 'min', 'max', 'first', 'last', or
        dict mapping column names to aggregation functions
    interpolate_missing : bool, default=True
        Whether to interpolate missing values after resampling
    """
    
    def __init__(self, target_frequency: str, aggregation_method='mean', 
                 interpolate_missing=True, **kwargs):
        super().__init__(**kwargs)
        self.target_frequency = target_frequency
        self.aggregation_method = aggregation_method
        self.interpolate_missing = interpolate_missing
        self.original_frequency_ = None
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Fit the resampling transformer.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Time series data with datetime index
        y : pd.Series, optional
            Target time series data (not used)
            
        Returns:
        --------
        self : TimeSeriesResamplingTransformer
            Fitted transformer
        """
        if self.validate_input:
            X, y = self._validate_time_series(X, y)
            
        # Store original frequency for metadata
        if self.infer_frequency:
            self.original_frequency_ = self._infer_frequency(X)
            
        logger.debug(f"Fitted resampling transformer: {self.original_frequency_} -> {self.target_frequency}")
        return self
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Resample the time series data.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Time series data to resample
            
        Returns:
        --------
        X_resampled : pd.DataFrame
            Resampled time series data
        """
        check_is_fitted(self, ['original_frequency_'])
        
        if self.validate_input:
            X, _ = self._validate_time_series(X, None)
            
        try:
            # Perform resampling
            if isinstance(self.aggregation_method, dict):
                X_resampled = X.resample(self.target_frequency).agg(self.aggregation_method)
            else:
                resampler = X.resample(self.target_frequency)
                if self.aggregation_method == 'mean':
                    X_resampled = resampler.mean()
                elif self.aggregation_method == 'sum':
                    X_resampled = resampler.sum()
                elif self.aggregation_method == 'min':
                    X_resampled = resampler.min()
                elif self.aggregation_method == 'max':
                    X_resampled = resampler.max()
                elif self.aggregation_method == 'first':
                    X_resampled = resampler.first()
                elif self.aggregation_method == 'last':
                    X_resampled = resampler.last()
                else:
                    raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")
            
            # Handle missing values after resampling if requested
            if self.interpolate_missing and X_resampled.isnull().any().any():
                X_resampled = X_resampled.interpolate(method='time')
                
            logger.debug(f"Resampled from {len(X)} to {len(X_resampled)} observations")
            return X_resampled
            
        except Exception as e:
            logger.error(f"Error in resampling transformation: {e}")
            raise


class TimeSeriesFeatureExtractor(TimeSeriesTransformer):
    """
    sklearn-compatible transformer for extracting temporal features from time series data.
    
    Creates features like lag variables, rolling statistics, datetime components,
    and seasonality indicators to enhance time series analysis and forecasting.
    
    Parameters:
    -----------
    lag_features : list of int, default=[1, 7, 30]
        Lag periods to create lagged features
    rolling_windows : list of int, default=[7, 30]
        Window sizes for rolling statistics (mean, std, min, max)
    datetime_features : list of str, default=['hour', 'dayofweek', 'month', 'quarter']
        Datetime components to extract
    seasonal_features : bool, default=True
        Whether to create seasonal indicator features
    cyclical_encoding : bool, default=True
        Whether to use cyclical encoding for periodic features
    """
    
    def __init__(self, lag_features=[1, 7, 30], rolling_windows=[7, 30], 
                 datetime_features=['hour', 'dayofweek', 'month', 'quarter'],
                 seasonal_features=True, cyclical_encoding=True, **kwargs):
        super().__init__(**kwargs)
        self.lag_features = lag_features
        self.rolling_windows = rolling_windows
        self.datetime_features = datetime_features
        self.seasonal_features = seasonal_features
        self.cyclical_encoding = cyclical_encoding
        self.feature_names_ = []
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fit the feature extraction transformer."""
        if self.validate_input:
            X, y = self._validate_time_series(X, y)
            
        # Determine feature names that will be created
        self.feature_names_ = []
        
        # Original columns
        self.feature_names_.extend(X.columns.tolist())
        
        # Lag features
        for col in X.columns:
            for lag in self.lag_features:
                self.feature_names_.append(f'{col}_lag_{lag}')
                
        # Rolling features
        for col in X.columns:
            for window in self.rolling_windows:
                self.feature_names_.extend([
                    f'{col}_rolling_mean_{window}',
                    f'{col}_rolling_std_{window}',
                    f'{col}_rolling_min_{window}',
                    f'{col}_rolling_max_{window}'
                ])
        
        # Datetime features
        for dt_feature in self.datetime_features:
            if self.cyclical_encoding and dt_feature in ['hour', 'dayofweek', 'month']:
                self.feature_names_.extend([f'{dt_feature}_sin', f'{dt_feature}_cos'])
            else:
                self.feature_names_.append(dt_feature)
        
        logger.debug(f"Will create {len(self.feature_names_)} features")
        return self
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Extract temporal features from time series data."""
        check_is_fitted(self, ['feature_names_'])
        
        if self.validate_input:
            X, _ = self._validate_time_series(X, None)
            
        try:
            X_features = X.copy()
            
            # Create lag features
            for col in X.columns:
                for lag in self.lag_features:
                    X_features[f'{col}_lag_{lag}'] = X[col].shift(lag)
            
            # Create rolling features
            for col in X.columns:
                for window in self.rolling_windows:
                    rolling = X[col].rolling(window=window, min_periods=1)
                    X_features[f'{col}_rolling_mean_{window}'] = rolling.mean()
                    X_features[f'{col}_rolling_std_{window}'] = rolling.std()
                    X_features[f'{col}_rolling_min_{window}'] = rolling.min()
                    X_features[f'{col}_rolling_max_{window}'] = rolling.max()
            
            # Create datetime features
            for dt_feature in self.datetime_features:
                if dt_feature == 'hour':
                    values = X.index.hour
                    if self.cyclical_encoding:
                        X_features['hour_sin'] = np.sin(2 * np.pi * values / 24)
                        X_features['hour_cos'] = np.cos(2 * np.pi * values / 24)
                    else:
                        X_features['hour'] = values
                        
                elif dt_feature == 'dayofweek':
                    values = X.index.dayofweek
                    if self.cyclical_encoding:
                        X_features['dayofweek_sin'] = np.sin(2 * np.pi * values / 7)
                        X_features['dayofweek_cos'] = np.cos(2 * np.pi * values / 7)
                    else:
                        X_features['dayofweek'] = values
                        
                elif dt_feature == 'month':
                    values = X.index.month
                    if self.cyclical_encoding:
                        X_features['month_sin'] = np.sin(2 * np.pi * values / 12)
                        X_features['month_cos'] = np.cos(2 * np.pi * values / 12)
                    else:
                        X_features['month'] = values
                        
                elif dt_feature == 'quarter':
                    X_features['quarter'] = X.index.quarter
                elif dt_feature == 'year':
                    X_features['year'] = X.index.year
                elif dt_feature == 'dayofyear':
                    X_features['dayofyear'] = X.index.dayofyear
                elif dt_feature == 'weekofyear':
                    X_features['weekofyear'] = X.index.isocalendar().week
            
            # Create seasonal features if requested
            if self.seasonal_features:
                seasonality_info = self._detect_seasonality(X.iloc[:, 0])  # Use first column
                if seasonality_info['has_seasonality']:
                    period = seasonality_info['dominant_period']
                    position_in_season = np.arange(len(X)) % period
                    X_features['seasonal_position_sin'] = np.sin(2 * np.pi * position_in_season / period)
                    X_features['seasonal_position_cos'] = np.cos(2 * np.pi * position_in_season / period)
            
            logger.debug(f"Created {len(X_features.columns)} total features")
            return X_features
            
        except Exception as e:
            logger.error(f"Error in feature extraction: {e}")
            raise


class TimeSeriesImputationTransformer(TimeSeriesTransformer):
    """
    sklearn-compatible transformer for time series missing value imputation.
    
    Provides specialized imputation methods that respect temporal structure,
    including forward fill, backward fill, interpolation, and seasonal imputation.
    
    Parameters:
    -----------
    method : str, default='interpolate'
        Imputation method: 'forward_fill', 'backward_fill', 'interpolate',
        'seasonal', 'mean', or 'median'
    interpolation_method : str, default='time'
        Interpolation method when method='interpolate'
    limit_direction : str, default='forward'
        Direction for fill operations
    seasonal_period : int, optional
        Period for seasonal imputation
    """
    
    def __init__(self, method='interpolate', interpolation_method='time',
                 limit_direction='forward', seasonal_period=None, **kwargs):
        super().__init__(**kwargs)
        self.method = method
        self.interpolation_method = interpolation_method
        self.limit_direction = limit_direction
        self.seasonal_period = seasonal_period
        self.seasonal_patterns_ = {}
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fit the imputation transformer."""
        if self.validate_input:
            X, y = self._validate_time_series(X, y)
            
        # For seasonal imputation, learn seasonal patterns
        if self.method == 'seasonal':
            for col in X.columns:
                series = X[col].dropna()
                if len(series) > 0:
                    # Detect seasonality if period not specified
                    if self.seasonal_period is None:
                        seasonality_info = self._detect_seasonality(series)
                        if seasonality_info['has_seasonality']:
                            period = seasonality_info['dominant_period']
                        else:
                            period = min(12, len(series) // 4)  # Default fallback
                    else:
                        period = self.seasonal_period
                    
                    # Calculate seasonal averages
                    seasonal_means = {}
                    for i in range(period):
                        season_values = series.iloc[i::period]
                        if len(season_values) > 0:
                            seasonal_means[i] = season_values.mean()
                        else:
                            seasonal_means[i] = series.mean()
                    
                    self.seasonal_patterns_[col] = {
                        'period': period,
                        'seasonal_means': seasonal_means
                    }
                    
        return self
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values in time series data."""
        if self.validate_input:
            X, _ = self._validate_time_series(X, None)
            
        X_imputed = X.copy()
        
        try:
            if self.method == 'forward_fill':
                X_imputed = X_imputed.fillna(method='ffill')
                
            elif self.method == 'backward_fill':
                X_imputed = X_imputed.fillna(method='bfill')
                
            elif self.method == 'interpolate':
                X_imputed = X_imputed.interpolate(method=self.interpolation_method)
                
            elif self.method == 'mean':
                X_imputed = X_imputed.fillna(X_imputed.mean())
                
            elif self.method == 'median':
                X_imputed = X_imputed.fillna(X_imputed.median())
                
            elif self.method == 'seasonal':
                check_is_fitted(self, ['seasonal_patterns_'])
                
                for col in X.columns:
                    if col in self.seasonal_patterns_:
                        pattern = self.seasonal_patterns_[col]
                        period = pattern['period']
                        seasonal_means = pattern['seasonal_means']
                        
                        # Fill missing values with seasonal pattern
                        missing_mask = X_imputed[col].isnull()
                        for idx in missing_mask[missing_mask].index:
                            position_in_period = len(X_imputed.loc[:idx]) % period
                            if position_in_period in seasonal_means:
                                X_imputed.loc[idx, col] = seasonal_means[position_in_period]
            
            # Log imputation results
            original_missing = X.isnull().sum().sum()
            final_missing = X_imputed.isnull().sum().sum()
            imputed_values = original_missing - final_missing
            
            if imputed_values > 0:
                logger.debug(f"Imputed {imputed_values} missing values using {self.method}")
                
            return X_imputed
            
        except Exception as e:
            logger.error(f"Error in imputation: {e}")
            raise


class TimeSeriesQualityValidator(TimeSeriesTransformer):
    """
    sklearn-compatible transformer for time series data quality validation.
    
    Validates time series data quality and provides recommendations for
    preprocessing and analysis approach.
    
    Parameters:
    -----------
    min_observations : int, default=10
        Minimum number of observations required
    max_missing_percentage : float, default=0.2
        Maximum percentage of missing values allowed
    require_regular_frequency : bool, default=False
        Whether to require regular time frequency
    check_stationarity : bool, default=False
        Whether to perform basic stationarity checks
    """
    
    def __init__(self, min_observations=10, max_missing_percentage=0.2,
                 require_regular_frequency=False, check_stationarity=False, **kwargs):
        super().__init__(**kwargs)
        self.min_observations = min_observations
        self.max_missing_percentage = max_missing_percentage
        self.require_regular_frequency = require_regular_frequency
        self.check_stationarity = check_stationarity
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fit the quality validator."""
        if self.validate_input:
            X, y = self._validate_time_series(X, y)
        return self
        
    def transform(self, X: pd.DataFrame) -> TimeSeriesAnalysisResult:
        """
        Validate time series data quality.
        
        Returns:
        --------
        result : TimeSeriesAnalysisResult
            Quality validation results and recommendations
        """
        start_time = time.time()
        
        if self.validate_input:
            X, _ = self._validate_time_series(X, None)
        
        try:
            # Perform comprehensive quality assessment
            quality_assessment = validate_time_series_continuity(X)
            
            # Additional checks
            warnings_list = []
            recommendations_list = []
            quality_issues = []
            
            # Check minimum observations
            if len(X) < self.min_observations:
                quality_issues.append(f"Insufficient data: {len(X)} < {self.min_observations} required")
                recommendations_list.append("Collect more historical data")
            
            # Check missing value percentage
            missing_pct = quality_assessment['missing_value_percentage']
            if missing_pct > self.max_missing_percentage * 100:
                quality_issues.append(f"High missing values: {missing_pct:.1f}% > {self.max_missing_percentage*100:.1f}% threshold")
                recommendations_list.append("Apply missing value imputation")
            
            # Check frequency regularity
            if self.require_regular_frequency and quality_assessment['frequency'] is None:
                quality_issues.append("Irregular time frequency detected")
                recommendations_list.append("Resample to regular frequency")
            
            # Basic stationarity check if requested
            stationarity_info = {}
            if self.check_stationarity and len(X.columns) > 0:
                try:
                    from statsmodels.tsa.stattools import adfuller
                    first_col = X.iloc[:, 0].dropna()
                    if len(first_col) > 10:
                        adf_result = adfuller(first_col)
                        stationarity_info = {
                            'adf_statistic': adf_result[0],
                            'adf_pvalue': adf_result[1],
                            'is_likely_stationary': adf_result[1] < 0.05
                        }
                        if not stationarity_info['is_likely_stationary']:
                            recommendations_list.append("Consider differencing or transformation for stationarity")
                except Exception as e:
                    warnings_list.append(f"Could not perform stationarity check: {e}")
            
            # Overall assessment
            overall_quality = quality_assessment['data_quality_score']
            if len(quality_issues) == 0:
                interpretation = f"Time series data quality: GOOD (score: {overall_quality:.2f})"
            elif overall_quality > 0.7:
                interpretation = f"Time series data quality: ACCEPTABLE (score: {overall_quality:.2f}) with minor issues"
            else:
                interpretation = f"Time series data quality: POOR (score: {overall_quality:.2f}) - significant issues detected"
            
            # Combine recommendations
            recommendations_list.extend(quality_assessment.get('recommendations', []))
            
            result = TimeSeriesAnalysisResult(
                analysis_type="quality_validation",
                interpretation=interpretation,
                data_quality_score=overall_quality,
                frequency=quality_assessment['frequency'],
                model_diagnostics={
                    'total_observations': len(X),
                    'missing_values': quality_assessment['missing_value_count'],
                    'missing_percentage': missing_pct,
                    'has_datetime_index': quality_assessment['has_datetime_index'],
                    'is_monotonic': quality_assessment['is_monotonic'],
                    'time_gaps': len(quality_assessment['gaps']),
                    'stationarity_info': stationarity_info
                },
                recommendations=recommendations_list,
                warnings=warnings_list + [f"Quality issue: {issue}" for issue in quality_issues],
                processing_time=time.time() - start_time
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error in quality validation: {e}")
            return TimeSeriesAnalysisResult(
                analysis_type="quality_validation_error",
                interpretation=f"Error during quality validation: {str(e)}",
                warnings=[str(e)],
                processing_time=time.time() - start_time
            )


class StationarityTestTransformer(TimeSeriesTransformer):
    """
    sklearn-compatible transformer for time series stationarity testing.
    
    Performs multiple stationarity tests including Augmented Dickey-Fuller (ADF),
    Kwiatkowski-Phillips-Schmidt-Shin (KPSS), and Phillips-Perron tests to 
    assess whether a time series is stationary.
    
    Parameters:
    -----------
    tests : list of str, default=['adf', 'kpss', 'pp']
        List of tests to perform: 'adf', 'kpss', 'pp' (Phillips-Perron)
    alpha : float, default=0.05
        Significance level for hypothesis tests
    auto_differencing : bool, default=False
        Whether to automatically suggest differencing for non-stationary series
    max_diff_order : int, default=2
        Maximum order of differencing to test if auto_differencing is True
    """
    
    def __init__(self, tests=['adf', 'kpss', 'pp'], alpha=0.05, 
                 auto_differencing=False, max_diff_order=2, **kwargs):
        super().__init__(**kwargs)
        self.tests = tests
        self.alpha = alpha
        self.auto_differencing = auto_differencing
        self.max_diff_order = max_diff_order
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fit the stationarity test transformer."""
        if self.validate_input:
            X, y = self._validate_time_series(X, y)
        return self
        
    def transform(self, X: pd.DataFrame) -> TimeSeriesAnalysisResult:
        """
        Perform stationarity tests on time series data.
        
        Returns:
        --------
        result : TimeSeriesAnalysisResult
            Stationarity test results and recommendations
        """
        start_time = time.time()
        
        if self.validate_input:
            X, _ = self._validate_time_series(X, None)
            
        try:
            # Use first column for testing (can be extended for multivariate)
            if len(X.columns) == 0:
                raise ValueError("No columns found in time series data")
                
            series = X.iloc[:, 0].dropna()
            if len(series) < 10:
                raise ValueError("Insufficient data points for stationarity testing")
            
            test_results = {}
            overall_stationary = True
            recommendations = []
            warnings_list = []
            
            # Perform Augmented Dickey-Fuller test
            if 'adf' in self.tests:
                try:
                    adf_result = adfuller(series, autolag='AIC')
                    test_results['adf'] = {
                        'statistic': adf_result[0],
                        'p_value': adf_result[1],
                        'critical_values': adf_result[4],
                        'used_lag': adf_result[2],
                        'n_observations': adf_result[3],
                        'is_stationary': adf_result[1] <= self.alpha,
                        'interpretation': 'Stationary' if adf_result[1] <= self.alpha else 'Non-stationary'
                    }
                    if not test_results['adf']['is_stationary']:
                        overall_stationary = False
                except Exception as e:
                    warnings_list.append(f"ADF test failed: {e}")
                    
            # Perform KPSS test
            if 'kpss' in self.tests:
                try:
                    kpss_result = kpss(series, regression='c', nlags='auto')
                    test_results['kpss'] = {
                        'statistic': kpss_result[0],
                        'p_value': kpss_result[1],
                        'critical_values': kpss_result[3],
                        'used_lag': kpss_result[2],
                        'is_stationary': kpss_result[1] > self.alpha,  # Note: KPSS null hypothesis is stationarity
                        'interpretation': 'Stationary' if kpss_result[1] > self.alpha else 'Non-stationary'
                    }
                    if not test_results['kpss']['is_stationary']:
                        overall_stationary = False
                except Exception as e:
                    warnings_list.append(f"KPSS test failed: {e}")
                    
            # Perform Phillips-Perron test
            if 'pp' in self.tests:
                try:
                    pp_result = pp_test(series, lags='auto')
                    test_results['pp'] = {
                        'statistic': pp_result[0],
                        'p_value': pp_result[1],
                        'used_lag': pp_result[2],
                        'n_observations': pp_result[3],
                        'critical_values': pp_result[4],
                        'is_stationary': pp_result[1] <= self.alpha,
                        'interpretation': 'Stationary' if pp_result[1] <= self.alpha else 'Non-stationary'
                    }
                    if not test_results['pp']['is_stationary']:
                        overall_stationary = False
                except Exception as e:
                    warnings_list.append(f"Phillips-Perron test failed: {e}")
            
            # Generate recommendations
            if not overall_stationary:
                recommendations.append("Time series appears to be non-stationary")
                recommendations.append("Consider applying differencing or transformation")
                
                if self.auto_differencing:
                    diff_recommendations = self._suggest_differencing(series)
                    recommendations.extend(diff_recommendations)
            else:
                recommendations.append("Time series appears to be stationary")
                recommendations.append("Suitable for ARMA modeling without differencing")
            
            # Generate interpretation summary
            stationary_tests = sum([result.get('is_stationary', False) for result in test_results.values()])
            total_tests = len(test_results)
            
            if total_tests == 0:
                interpretation = "No stationarity tests could be performed"
            elif stationary_tests == total_tests:
                interpretation = f"All {total_tests} stationarity tests indicate STATIONARY series"
            elif stationary_tests == 0:
                interpretation = f"All {total_tests} stationarity tests indicate NON-STATIONARY series"
            else:
                interpretation = f"Mixed results: {stationary_tests}/{total_tests} tests indicate stationarity"
            
            result = TimeSeriesAnalysisResult(
                analysis_type="stationarity_testing",
                interpretation=interpretation,
                model_diagnostics=test_results,
                recommendations=recommendations,
                warnings=warnings_list,
                processing_time=time.time() - start_time
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error in stationarity testing: {e}")
            return TimeSeriesAnalysisResult(
                analysis_type="stationarity_testing_error",
                interpretation=f"Error during stationarity testing: {str(e)}",
                warnings=[str(e)],
                processing_time=time.time() - start_time
            )
    
    def _suggest_differencing(self, series: pd.Series) -> List[str]:
        """
        Suggest appropriate differencing order for non-stationary series.
        
        Parameters:
        -----------
        series : pd.Series
            Time series data
            
        Returns:
        --------
        suggestions : list of str
            List of differencing recommendations
        """
        suggestions = []
        
        try:
            for diff_order in range(1, self.max_diff_order + 1):
                # Apply differencing
                diff_series = series.diff(diff_order).dropna()
                
                if len(diff_series) < 10:
                    break
                    
                # Test stationarity of differenced series
                try:
                    adf_result = adfuller(diff_series, autolag='AIC')
                    if adf_result[1] <= self.alpha:
                        suggestions.append(f"First-order differencing (d={diff_order}) achieves stationarity")
                        break
                except Exception:
                    continue
            
            if not suggestions:
                suggestions.append("Consider seasonal differencing or transformation (log, sqrt)")
                
        except Exception as e:
            suggestions.append(f"Could not test differencing: {e}")
            
        return suggestions


class UnitRootTestTransformer(TimeSeriesTransformer):
    """
    sklearn-compatible transformer for comprehensive unit root testing.
    
    Performs various unit root tests to detect the presence of unit roots
    in time series data, which is essential for determining integration order.
    
    Parameters:
    -----------
    test_type : str, default='adf'
        Primary test type: 'adf', 'kpss', 'pp', or 'dfgls'
    regression_type : str, default='c'
        Regression type: 'c' (constant), 'ct' (constant and trend), 'ctt' (constant, trend, and trend^2), 'nc' (no constant)
    max_lags : int, optional
        Maximum number of lags to use in the test
    autolag : str, default='AIC'
        Method to automatically determine lag length: 'AIC', 'BIC', 't-stat', or None
    """
    
    def __init__(self, test_type='adf', regression_type='c', max_lags=None, autolag='AIC', **kwargs):
        super().__init__(**kwargs)
        self.test_type = test_type
        self.regression_type = regression_type
        self.max_lags = max_lags
        self.autolag = autolag
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fit the unit root test transformer."""
        if self.validate_input:
            X, y = self._validate_time_series(X, y)
        return self
        
    def transform(self, X: pd.DataFrame) -> TimeSeriesAnalysisResult:
        """
        Perform unit root tests on time series data.
        
        Returns:
        --------
        result : TimeSeriesAnalysisResult
            Unit root test results and integration order recommendations
        """
        start_time = time.time()
        
        if self.validate_input:
            X, _ = self._validate_time_series(X, None)
            
        try:
            # Use first column for testing
            if len(X.columns) == 0:
                raise ValueError("No columns found in time series data")
                
            series = X.iloc[:, 0].dropna()
            if len(series) < 10:
                raise ValueError("Insufficient data points for unit root testing")
            
            test_results = {}
            recommendations = []
            warnings_list = []
            
            # Prepare test parameters
            test_kwargs = {'regression': self.regression_type}
            if self.max_lags is not None:
                test_kwargs['maxlag'] = self.max_lags
            if self.autolag is not None:
                test_kwargs['autolag'] = self.autolag
                
            # Perform the specified unit root test
            if self.test_type == 'adf':
                try:
                    result = adfuller(series, **test_kwargs)
                    test_results['adf'] = {
                        'test_statistic': result[0],
                        'p_value': result[1],
                        'lags_used': result[2],
                        'n_observations': result[3],
                        'critical_values': result[4],
                        'ic_best': result[5] if len(result) > 5 else None
                    }
                    
                    has_unit_root = result[1] > 0.05
                    interpretation = "Series has unit root (non-stationary)" if has_unit_root else "Series does not have unit root (stationary)"
                    
                except Exception as e:
                    warnings_list.append(f"ADF test failed: {e}")
                    interpretation = "Unit root test failed"
                    
            elif self.test_type == 'kpss':
                try:
                    result = kpss(series, regression=self.regression_type, nlags=self.autolag)
                    test_results['kpss'] = {
                        'test_statistic': result[0],
                        'p_value': result[1],
                        'lags_used': result[2],
                        'critical_values': result[3]
                    }
                    
                    has_unit_root = result[1] <= 0.05  # KPSS null hypothesis is stationarity
                    interpretation = "Series has unit root (non-stationary)" if has_unit_root else "Series does not have unit root (stationary)"
                    
                except Exception as e:
                    warnings_list.append(f"KPSS test failed: {e}")
                    interpretation = "Unit root test failed"
                    
            elif self.test_type == 'pp':
                try:
                    result = pp_test(series, lags=self.autolag, regression=self.regression_type)
                    test_results['pp'] = {
                        'test_statistic': result[0],
                        'p_value': result[1],
                        'lags_used': result[2],
                        'n_observations': result[3],
                        'critical_values': result[4]
                    }
                    
                    has_unit_root = result[1] > 0.05
                    interpretation = "Series has unit root (non-stationary)" if has_unit_root else "Series does not have unit root (stationary)"
                    
                except Exception as e:
                    warnings_list.append(f"Phillips-Perron test failed: {e}")
                    interpretation = "Unit root test failed"
            else:
                raise ValueError(f"Unsupported test type: {self.test_type}")
            
            # Generate recommendations based on results
            if 'has_unit_root' in locals() and has_unit_root:
                recommendations.extend([
                    "Series contains unit root - requires differencing",
                    "Consider first differencing: y(t) - y(t-1)",
                    "Test differenced series for stationarity"
                ])
                
                # Test if first differencing makes it stationary
                try:
                    diff_series = series.diff().dropna()
                    if len(diff_series) >= 10:
                        diff_result = adfuller(diff_series)
                        if diff_result[1] <= 0.05:
                            recommendations.append("First differencing should achieve stationarity (I(1) process)")
                        else:
                            recommendations.append("May require second differencing or seasonal differencing (I(2) or seasonal process)")
                except Exception:
                    pass
            else:
                recommendations.extend([
                    "Series appears stationary - suitable for ARMA modeling",
                    "No differencing required"
                ])
            
            result = TimeSeriesAnalysisResult(
                analysis_type="unit_root_testing",
                interpretation=interpretation,
                model_diagnostics=test_results,
                recommendations=recommendations,
                warnings=warnings_list,
                processing_time=time.time() - start_time
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error in unit root testing: {e}")
            return TimeSeriesAnalysisResult(
                analysis_type="unit_root_testing_error",
                interpretation=f"Error during unit root testing: {str(e)}",
                warnings=[str(e)],
                processing_time=time.time() - start_time
            )


class AutocorrelationAnalysisTransformer(TimeSeriesTransformer):
    """
    sklearn-compatible transformer for autocorrelation function (ACF) analysis.
    
    Computes and analyzes the autocorrelation function of time series data,
    including significance testing and lag selection recommendations.
    
    Parameters:
    -----------
    max_lags : int, optional
        Maximum number of lags to compute. If None, uses min(40, len(series)//4)
    alpha : float, default=0.05
        Significance level for correlation tests
    fft : bool, default=True
        Whether to use FFT for computation (faster for long series)
    missing : str, default='none'
        How to handle missing values: 'none', 'drop', 'conservative'
    """
    
    def __init__(self, max_lags=None, alpha=0.05, fft=True, missing='none', **kwargs):
        super().__init__(**kwargs)
        self.max_lags = max_lags
        self.alpha = alpha
        self.fft = fft
        self.missing = missing
        self.acf_values_ = None
        self.confidence_intervals_ = None
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fit the ACF analysis transformer."""
        if self.validate_input:
            X, y = self._validate_time_series(X, y)
        return self
        
    def transform(self, X: pd.DataFrame) -> TimeSeriesAnalysisResult:
        """
        Compute autocorrelation function analysis.
        
        Returns:
        --------
        result : TimeSeriesAnalysisResult
            ACF analysis results with lag recommendations
        """
        start_time = time.time()
        
        if self.validate_input:
            X, _ = self._validate_time_series(X, None)
            
        try:
            # Use first column for analysis
            if len(X.columns) == 0:
                raise ValueError("No columns found in time series data")
                
            series = X.iloc[:, 0].dropna()
            if len(series) < 10:
                raise ValueError("Insufficient data points for autocorrelation analysis")
            
            # Determine number of lags
            if self.max_lags is None:
                nlags = min(40, len(series) // 4)
            else:
                nlags = min(self.max_lags, len(series) - 1)
            
            # Compute ACF
            acf_values, confint = acf(series, nlags=nlags, alpha=self.alpha, 
                                     fft=self.fft, missing=self.missing)
            
            self.acf_values_ = acf_values
            self.confidence_intervals_ = confint
            
            # Find significant lags
            significant_lags = []
            for lag in range(1, len(acf_values)):  # Skip lag 0 (always 1.0)
                if abs(acf_values[lag]) > abs(confint[lag, 1] - acf_values[lag]):
                    significant_lags.append({
                        'lag': lag,
                        'correlation': acf_values[lag],
                        'is_significant': True,
                        'confidence_lower': confint[lag, 0],
                        'confidence_upper': confint[lag, 1]
                    })
                else:
                    significant_lags.append({
                        'lag': lag,
                        'correlation': acf_values[lag],
                        'is_significant': False,
                        'confidence_lower': confint[lag, 0],
                        'confidence_upper': confint[lag, 1]
                    })
            
            # Identify patterns and generate recommendations
            recommendations = []
            warnings_list = []
            pattern_analysis = self._analyze_acf_patterns(acf_values, significant_lags)
            
            # Generate interpretation
            num_significant = sum(1 for lag_info in significant_lags if lag_info['is_significant'])
            
            if num_significant == 0:
                interpretation = "No significant autocorrelations detected - series appears to be white noise"
                recommendations.append("Series may be suitable for simple forecasting methods")
            elif num_significant <= 3:
                interpretation = f"Few significant autocorrelations detected ({num_significant} lags)"
                recommendations.append("Consider MA or low-order ARMA models")
            else:
                interpretation = f"Multiple significant autocorrelations detected ({num_significant} lags)"
                recommendations.append("Consider higher-order ARMA models or seasonal patterns")
            
            # Add pattern-specific recommendations
            recommendations.extend(pattern_analysis['recommendations'])
            warnings_list.extend(pattern_analysis.get('warnings', []))
            
            result = TimeSeriesAnalysisResult(
                analysis_type="autocorrelation_analysis",
                interpretation=interpretation,
                model_diagnostics={
                    'acf_values': acf_values.tolist(),
                    'confidence_intervals': confint.tolist(),
                    'significant_lags': significant_lags,
                    'pattern_analysis': pattern_analysis,
                    'max_lags_computed': nlags,
                    'series_length': len(series)
                },
                recommendations=recommendations,
                warnings=warnings_list,
                processing_time=time.time() - start_time
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error in autocorrelation analysis: {e}")
            return TimeSeriesAnalysisResult(
                analysis_type="autocorrelation_analysis_error",
                interpretation=f"Error during autocorrelation analysis: {str(e)}",
                warnings=[str(e)],
                processing_time=time.time() - start_time
            )
    
    def _analyze_acf_patterns(self, acf_values: np.ndarray, significant_lags: List[Dict]) -> Dict[str, Any]:
        """
        Analyze patterns in autocorrelation function.
        
        Parameters:
        -----------
        acf_values : np.ndarray
            Autocorrelation function values
        significant_lags : list of dict
            Information about significant lags
            
        Returns:
        --------
        pattern_info : dict
            Analysis of ACF patterns and recommendations
        """
        analysis = {
            'pattern_type': 'unknown',
            'decay_rate': 'unknown',
            'seasonal_pattern': False,
            'seasonal_period': None,
            'recommendations': [],
            'warnings': []
        }
        
        try:
            # Analyze decay pattern
            sig_correlations = [lag['correlation'] for lag in significant_lags if lag['is_significant']]
            
            if len(sig_correlations) == 0:
                analysis['pattern_type'] = 'white_noise'
                analysis['decay_rate'] = 'immediate'
                analysis['recommendations'].append("Series appears to be white noise - no temporal dependence")
                
            elif len(sig_correlations) <= 2:
                analysis['pattern_type'] = 'short_memory'
                analysis['decay_rate'] = 'fast'
                analysis['recommendations'].append("Short-term temporal dependence - consider MA(1) or MA(2) model")
                
            else:
                # Check for exponential decay (AR pattern)
                if self._has_exponential_decay(acf_values[1:6]):  # Check first 5 lags
                    analysis['pattern_type'] = 'autoregressive'
                    analysis['decay_rate'] = 'exponential'
                    analysis['recommendations'].append("Exponential decay pattern - consider AR model")
                else:
                    analysis['pattern_type'] = 'complex'
                    analysis['decay_rate'] = 'slow'
                    analysis['recommendations'].append("Complex autocorrelation pattern - consider ARMA model")
            
            # Check for seasonal patterns
            seasonal_info = self._detect_seasonal_acf(acf_values, significant_lags)
            if seasonal_info['has_seasonal']:
                analysis['seasonal_pattern'] = True
                analysis['seasonal_period'] = seasonal_info['period']
                analysis['recommendations'].append(f"Seasonal pattern detected (period ≈ {seasonal_info['period']}) - consider seasonal ARIMA")
            
            # Check for problematic patterns
            if max(abs(acf_values[1:6])) > 0.9:
                analysis['warnings'].append("Very high autocorrelations suggest possible non-stationarity")
                analysis['recommendations'].append("Consider differencing the series")
                
        except Exception as e:
            analysis['warnings'].append(f"Error in pattern analysis: {e}")
            
        return analysis
    
    def _has_exponential_decay(self, correlations: np.ndarray) -> bool:
        """Check if correlations show exponential decay pattern."""
        if len(correlations) < 3:
            return False
            
        try:
            # Fit exponential decay: y = a * exp(-b * x)
            # Use log transformation: log(|y|) = log(a) - b * x
            x = np.arange(1, len(correlations) + 1)
            y = np.abs(correlations)
            
            # Filter out very small values to avoid log issues
            valid_mask = y > 0.01
            if np.sum(valid_mask) < 3:
                return False
                
            x_valid = x[valid_mask]
            y_valid = y[valid_mask]
            
            log_y = np.log(y_valid)
            
            # Simple linear regression on log scale
            slope = np.polyfit(x_valid, log_y, 1)[0]
            
            # Exponential decay should have negative slope and good fit
            return slope < -0.1  # Threshold for significant decay
            
        except Exception:
            return False
    
    def _detect_seasonal_acf(self, acf_values: np.ndarray, significant_lags: List[Dict]) -> Dict[str, Any]:
        """Detect seasonal patterns in autocorrelation function."""
        seasonal_info = {'has_seasonal': False, 'period': None, 'strength': 0.0}
        
        try:
            # Look for periodic peaks in significant lags
            sig_lags = [lag['lag'] for lag in significant_lags if lag['is_significant']]
            
            if len(sig_lags) >= 3:
                # Check for regular spacing (seasonal periods)
                for period in [4, 7, 12, 24, 52]:  # Common seasonal periods
                    if period >= len(acf_values):
                        continue
                        
                    # Check if there are significant correlations at multiples of this period
                    seasonal_lags = [lag for lag in sig_lags if lag % period == 0 and lag > 0]
                    
                    if len(seasonal_lags) >= 2:
                        # Calculate average correlation at seasonal lags
                        seasonal_correlations = [abs(acf_values[lag]) for lag in seasonal_lags if lag < len(acf_values)]
                        avg_seasonal_corr = np.mean(seasonal_correlations)
                        
                        if avg_seasonal_corr > 0.2:  # Threshold for seasonal strength
                            seasonal_info['has_seasonal'] = True
                            seasonal_info['period'] = period
                            seasonal_info['strength'] = avg_seasonal_corr
                            break
                            
        except Exception as e:
            logger.debug(f"Error in seasonal ACF detection: {e}")
            
        return seasonal_info


class PartialAutocorrelationAnalysisTransformer(TimeSeriesTransformer):
    """
    sklearn-compatible transformer for partial autocorrelation function (PACF) analysis.
    
    Computes and analyzes the partial autocorrelation function of time series data,
    useful for identifying the order of autoregressive models.
    
    Parameters:
    -----------
    max_lags : int, optional
        Maximum number of lags to compute. If None, uses min(40, len(series)//4)
    alpha : float, default=0.05
        Significance level for correlation tests
    method : str, default='ywmle'
        Method for PACF computation: 'ywmle', 'ols'
    """
    
    def __init__(self, max_lags=None, alpha=0.05, method='ywmle', **kwargs):
        super().__init__(**kwargs)
        self.max_lags = max_lags
        self.alpha = alpha
        self.method = method
        self.pacf_values_ = None
        self.confidence_intervals_ = None
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fit the PACF analysis transformer."""
        if self.validate_input:
            X, y = self._validate_time_series(X, y)
        return self
        
    def transform(self, X: pd.DataFrame) -> TimeSeriesAnalysisResult:
        """
        Compute partial autocorrelation function analysis.
        
        Returns:
        --------
        result : TimeSeriesAnalysisResult
            PACF analysis results with AR order recommendations
        """
        start_time = time.time()
        
        if self.validate_input:
            X, _ = self._validate_time_series(X, None)
            
        try:
            # Use first column for analysis
            if len(X.columns) == 0:
                raise ValueError("No columns found in time series data")
                
            series = X.iloc[:, 0].dropna()
            if len(series) < 10:
                raise ValueError("Insufficient data points for partial autocorrelation analysis")
            
            # Determine number of lags
            if self.max_lags is None:
                nlags = min(40, len(series) // 4)
            else:
                nlags = min(self.max_lags, len(series) - 1)
            
            # Compute PACF
            pacf_values, confint = pacf(series, nlags=nlags, alpha=self.alpha, method=self.method)
            
            self.pacf_values_ = pacf_values
            self.confidence_intervals_ = confint
            
            # Find significant lags
            significant_lags = []
            for lag in range(1, len(pacf_values)):  # Skip lag 0 (always 1.0)
                is_significant = abs(pacf_values[lag]) > abs(confint[lag, 1] - pacf_values[lag])
                significant_lags.append({
                    'lag': lag,
                    'partial_correlation': pacf_values[lag],
                    'is_significant': is_significant,
                    'confidence_lower': confint[lag, 0],
                    'confidence_upper': confint[lag, 1]
                })
            
            # Analyze PACF for AR order suggestion
            ar_order_analysis = self._analyze_ar_order(pacf_values, significant_lags)
            
            # Generate recommendations
            recommendations = []
            warnings_list = []
            
            if ar_order_analysis['suggested_order'] == 0:
                interpretation = "No significant partial autocorrelations - series may be MA or white noise"
                recommendations.append("Consider MA model or simple forecasting methods")
            else:
                interpretation = f"Significant partial autocorrelations up to lag {ar_order_analysis['suggested_order']}"
                recommendations.append(f"Consider AR({ar_order_analysis['suggested_order']}) model")
                
                if ar_order_analysis['seasonal_order'] > 0:
                    recommendations.append(f"Seasonal AR component suggested: P={ar_order_analysis['seasonal_order']}")
            
            # Add analysis-specific recommendations
            recommendations.extend(ar_order_analysis.get('recommendations', []))
            warnings_list.extend(ar_order_analysis.get('warnings', []))
            
            result = TimeSeriesAnalysisResult(
                analysis_type="partial_autocorrelation_analysis",
                interpretation=interpretation,
                model_diagnostics={
                    'pacf_values': pacf_values.tolist(),
                    'confidence_intervals': confint.tolist(),
                    'significant_lags': significant_lags,
                    'ar_order_analysis': ar_order_analysis,
                    'max_lags_computed': nlags,
                    'series_length': len(series)
                },
                recommendations=recommendations,
                warnings=warnings_list,
                processing_time=time.time() - start_time
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error in partial autocorrelation analysis: {e}")
            return TimeSeriesAnalysisResult(
                analysis_type="partial_autocorrelation_analysis_error",
                interpretation=f"Error during partial autocorrelation analysis: {str(e)}",
                warnings=[str(e)],
                processing_time=time.time() - start_time
            )
    
    def _analyze_ar_order(self, pacf_values: np.ndarray, significant_lags: List[Dict]) -> Dict[str, Any]:
        """
        Analyze PACF to suggest AR model order.
        
        Parameters:
        -----------
        pacf_values : np.ndarray
            Partial autocorrelation function values
        significant_lags : list of dict
            Information about significant lags
            
        Returns:
        --------
        ar_analysis : dict
            AR order analysis and recommendations
        """
        analysis = {
            'suggested_order': 0,
            'seasonal_order': 0,
            'cutoff_pattern': 'none',
            'recommendations': [],
            'warnings': []
        }
        
        try:
            # Find the last significant lag (AR order suggestion)
            significant_lag_numbers = [lag['lag'] for lag in significant_lags if lag['is_significant']]
            
            if len(significant_lag_numbers) == 0:
                analysis['suggested_order'] = 0
                analysis['cutoff_pattern'] = 'immediate'
                analysis['recommendations'].append("No AR component suggested")
            else:
                # Traditional approach: find where PACF cuts off
                # Look for clear cutoff pattern
                last_significant = max(significant_lag_numbers)
                
                # Check if there's a clear cutoff (most recent significant lags)
                recent_significant = [lag for lag in significant_lag_numbers if lag <= 10]
                
                if len(recent_significant) > 0:
                    # Suggest order based on last consecutive significant lag
                    consecutive_order = self._find_consecutive_cutoff(significant_lags)
                    analysis['suggested_order'] = consecutive_order
                    
                    if consecutive_order <= 3:
                        analysis['cutoff_pattern'] = 'clear'
                        analysis['recommendations'].append(f"Clear PACF cutoff suggests AR({consecutive_order})")
                    else:
                        analysis['cutoff_pattern'] = 'gradual'
                        analysis['recommendations'].append("Gradual PACF decay - consider ARMA model")
                        analysis['suggested_order'] = min(3, consecutive_order)  # Cap at reasonable order
                else:
                    analysis['suggested_order'] = last_significant
                    analysis['cutoff_pattern'] = 'unclear'
                    analysis['warnings'].append("PACF pattern unclear - validate model selection")
                
                # Check for seasonal patterns
                seasonal_lags = [lag for lag in significant_lag_numbers if lag >= 4]
                seasonal_periods = self._detect_seasonal_pacf_periods(seasonal_lags)
                
                if seasonal_periods:
                    analysis['seasonal_order'] = 1
                    analysis['recommendations'].append(f"Seasonal patterns detected at lags: {seasonal_periods}")
                    
        except Exception as e:
            analysis['warnings'].append(f"Error in AR order analysis: {e}")
            
        return analysis
    
    def _find_consecutive_cutoff(self, significant_lags: List[Dict]) -> int:
        """Find the order where PACF cuts off based on consecutive significant lags."""
        # Find the longest sequence of consecutive significant lags starting from lag 1
        order = 0
        
        for lag_info in sorted(significant_lags, key=lambda x: x['lag']):
            if lag_info['lag'] == order + 1 and lag_info['is_significant']:
                order = lag_info['lag']
            else:
                break
                
        return order
    
    def _detect_seasonal_pacf_periods(self, seasonal_lags: List[int]) -> List[int]:
        """Detect seasonal periods in PACF significant lags."""
        periods = []
        
        for period in [4, 7, 12, 24, 52]:
            seasonal_matches = [lag for lag in seasonal_lags if lag % period == 0]
            if len(seasonal_matches) >= 1:
                periods.append(period)
                
        return periods


class LagSelectionTransformer(TimeSeriesTransformer):
    """
    sklearn-compatible transformer for automated lag selection using information criteria.
    
    Determines optimal lag structures for time series models using AIC, BIC, and
    other information criteria, combining ACF/PACF analysis with statistical tests.
    
    Parameters:
    -----------
    max_ar_order : int, default=5
        Maximum AR order to consider
    max_ma_order : int, default=5
        Maximum MA order to consider
    information_criteria : list of str, default=['aic', 'bic']
        Information criteria to use: 'aic', 'bic', 'hqic'
    seasonal : bool, default=False
        Whether to consider seasonal components
    seasonal_periods : list of int, optional
        Seasonal periods to test if seasonal=True
    """
    
    def __init__(self, max_ar_order=5, max_ma_order=5, 
                 information_criteria=['aic', 'bic'], seasonal=False,
                 seasonal_periods=None, **kwargs):
        super().__init__(**kwargs)
        self.max_ar_order = max_ar_order
        self.max_ma_order = max_ma_order
        self.information_criteria = information_criteria
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods or []
        self.best_orders_ = {}
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fit the lag selection transformer."""
        if self.validate_input:
            X, y = self._validate_time_series(X, y)
        return self
        
    def transform(self, X: pd.DataFrame) -> TimeSeriesAnalysisResult:
        """
        Perform automated lag selection analysis.
        
        Returns:
        --------
        result : TimeSeriesAnalysisResult
            Lag selection results with optimal model orders
        """
        start_time = time.time()
        
        if self.validate_input:
            X, _ = self._validate_time_series(X, None)
            
        try:
            # Use first column for analysis
            if len(X.columns) == 0:
                raise ValueError("No columns found in time series data")
                
            series = X.iloc[:, 0].dropna()
            if len(series) < 20:
                raise ValueError("Insufficient data points for lag selection (need at least 20)")
            
            # Grid search over model orders
            results_grid = self._grid_search_orders(series)
            
            # Select best models for each criterion
            best_models = {}
            for criterion in self.information_criteria:
                if criterion in results_grid and len(results_grid[criterion]) > 0:
                    best_model = min(results_grid[criterion], key=lambda x: x[criterion])
                    best_models[criterion] = {
                        'ar_order': best_model['ar_order'],
                        'ma_order': best_model['ma_order'],
                        'criterion_value': best_model[criterion],
                        'seasonal_order': best_model.get('seasonal_order', 0)
                    }
            
            self.best_orders_ = best_models
            
            # Generate recommendations
            recommendations = []
            warnings_list = []
            
            if len(best_models) == 0:
                interpretation = "Could not determine optimal model orders"
                recommendations.append("Use default ARIMA(1,0,1) or consult domain expert")
                warnings_list.append("Model selection failed - check data quality")
            else:
                # Get consensus recommendation
                consensus = self._get_consensus_recommendation(best_models)
                
                interpretation = f"Optimal model orders determined: ARIMA({consensus['ar']},{consensus['d']},{consensus['ma']})"
                recommendations.append(f"Recommended model: ARIMA({consensus['ar']},{consensus['d']},{consensus['ma']})")
                
                if consensus.get('seasonal_order', 0) > 0:
                    recommendations.append(f"Seasonal component suggested: ({consensus['seasonal_order']},{consensus['seasonal_d']},{consensus['seasonal_ma']})")
                
                # Add criterion-specific information
                for criterion, model_info in best_models.items():
                    recommendations.append(f"{criterion.upper()} suggests: AR({model_info['ar_order']}) MA({model_info['ma_order']})")
            
            result = TimeSeriesAnalysisResult(
                analysis_type="lag_selection",
                interpretation=interpretation,
                model_diagnostics={
                    'best_models': best_models,
                    'grid_search_results': results_grid,
                    'consensus_recommendation': consensus if len(best_models) > 0 else None,
                    'series_length': len(series)
                },
                recommendations=recommendations,
                warnings=warnings_list,
                processing_time=time.time() - start_time
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error in lag selection: {e}")
            return TimeSeriesAnalysisResult(
                analysis_type="lag_selection_error",
                interpretation=f"Error during lag selection: {str(e)}",
                warnings=[str(e)],
                processing_time=time.time() - start_time
            )
    
    def _grid_search_orders(self, series: pd.Series) -> Dict[str, List[Dict]]:
        """
        Perform grid search over ARIMA orders.
        
        Parameters:
        -----------
        series : pd.Series
            Time series data
            
        Returns:
        --------
        results : dict
            Dictionary of results for each information criterion
        """
        results = {criterion: [] for criterion in self.information_criteria}
        
        try:
            from statsmodels.tsa.arima.model import ARIMA
            
            # Test different AR and MA orders
            for ar_order in range(self.max_ar_order + 1):
                for ma_order in range(self.max_ma_order + 1):
                    if ar_order == 0 and ma_order == 0:
                        continue  # Skip (0,0,0) model
                        
                    try:
                        # Fit ARIMA model
                        model = ARIMA(series, order=(ar_order, 0, ma_order))
                        fitted_model = model.fit()
                        
                        # Collect information criteria
                        model_result = {
                            'ar_order': ar_order,
                            'ma_order': ma_order,
                            'aic': fitted_model.aic,
                            'bic': fitted_model.bic,
                            'hqic': fitted_model.hqic,
                            'converged': fitted_model.mle_retvals['converged'] if hasattr(fitted_model, 'mle_retvals') else True
                        }
                        
                        # Add to results for each criterion
                        for criterion in self.information_criteria:
                            if criterion in model_result:
                                results[criterion].append(model_result)
                                
                    except Exception as e:
                        logger.debug(f"ARIMA({ar_order},0,{ma_order}) failed: {e}")
                        continue
                        
        except ImportError:
            logger.warning("statsmodels ARIMA not available - using simplified lag selection")
            
        return results
    
    def _get_consensus_recommendation(self, best_models: Dict[str, Dict]) -> Dict[str, int]:
        """Get consensus recommendation from different information criteria."""
        if len(best_models) == 0:
            return {'ar': 1, 'd': 0, 'ma': 1, 'seasonal_order': 0, 'seasonal_d': 0, 'seasonal_ma': 0}
        
        # Get the most common orders
        ar_orders = [model['ar_order'] for model in best_models.values()]
        ma_orders = [model['ma_order'] for model in best_models.values()]
        
        # Use mode or median
        consensus_ar = int(np.round(np.median(ar_orders)))
        consensus_ma = int(np.round(np.median(ma_orders)))
        
        return {
            'ar': consensus_ar,
            'd': 0,  # Differencing order (would be determined by stationarity tests)
            'ma': consensus_ma,
            'seasonal_order': 0,  # Simplified for now
            'seasonal_d': 0,
            'seasonal_ma': 0
        }


class TimeSeriesDecompositionTransformer(TimeSeriesTransformer):
    """
    sklearn-compatible transformer for time series decomposition analysis.
    
    Performs seasonal decomposition to separate time series into trend, seasonal, 
    and residual components using various decomposition methods.
    
    Parameters:
    -----------
    model : str, default='additive'
        Type of decomposition: 'additive' or 'multiplicative'
    period : int, optional
        Period for seasonal decomposition. If None, attempts to detect automatically
    extrapolate_trend : str or int, default='freq'
        How to extrapolate trend at ends: 'freq', integer, or None
    two_sided : bool, default=True
        Whether to use centered moving average for trend estimation
    method : str, default='seasonal_decompose'
        Decomposition method: 'seasonal_decompose', 'stl', 'x13' (if available)
    """
    
    def __init__(self, model='additive', period=None, extrapolate_trend='freq',
                 two_sided=True, method='seasonal_decompose', **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.period = period
        self.extrapolate_trend = extrapolate_trend
        self.two_sided = two_sided
        self.method = method
        self.decomposition_result_ = None
        self.detected_period_ = None
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fit the decomposition transformer."""
        if self.validate_input:
            X, y = self._validate_time_series(X, y)
        return self
        
    def transform(self, X: pd.DataFrame) -> TimeSeriesAnalysisResult:
        """
        Perform time series decomposition analysis.
        
        Returns:
        --------
        result : TimeSeriesAnalysisResult
            Decomposition results with trend, seasonal, and residual components
        """
        start_time = time.time()
        
        if self.validate_input:
            X, _ = self._validate_time_series(X, None)
            
        try:
            # Use first column for decomposition
            if len(X.columns) == 0:
                raise ValueError("No columns found in time series data")
                
            series = X.iloc[:, 0].dropna()
            if len(series) < 20:
                raise ValueError("Insufficient data points for decomposition (need at least 20)")
            
            # Detect period if not provided
            detected_period = self.period
            if detected_period is None:
                detected_period = self._detect_seasonal_period(series)
                
            if detected_period is None or detected_period < 2:
                # Fallback to simple trend extraction without seasonality
                decomp_result = self._simple_trend_decomposition(series)
                interpretation = "No clear seasonal pattern detected - performed trend-only decomposition"
                has_seasonality = False
            else:
                # Perform full seasonal decomposition
                if self.method == 'stl':
                    decomp_result = self._stl_decomposition(series, detected_period)
                elif self.method == 'x13':
                    decomp_result = self._x13_decomposition(series, detected_period)
                else:
                    decomp_result = self._seasonal_decomposition(series, detected_period)
                
                interpretation = f"Decomposition completed with period {detected_period}"
                has_seasonality = True
                
            self.decomposition_result_ = decomp_result
            self.detected_period_ = detected_period
            
            # Analyze decomposition components
            component_analysis = self._analyze_components(decomp_result, has_seasonality)
            
            # Generate recommendations
            recommendations = []
            warnings_list = []
            
            if has_seasonality:
                recommendations.append(f"Strong seasonal pattern detected with period {detected_period}")
                recommendations.append("Consider seasonal ARIMA or seasonal forecasting models")
            else:
                recommendations.append("No significant seasonal pattern - focus on trend modeling")
                recommendations.append("Consider non-seasonal ARIMA or trend-based forecasting")
            
            # Add component-specific recommendations
            recommendations.extend(component_analysis.get('recommendations', []))
            warnings_list.extend(component_analysis.get('warnings', []))
            
            result = TimeSeriesAnalysisResult(
                analysis_type="time_series_decomposition",
                interpretation=interpretation,
                trend_component=decomp_result.get('trend'),
                seasonal_component=decomp_result.get('seasonal'),
                residual_component=decomp_result.get('resid'),
                seasonality_period=detected_period,
                model_diagnostics={
                    'decomposition_method': self.method,
                    'model_type': self.model,
                    'detected_period': detected_period,
                    'has_seasonality': has_seasonality,
                    'component_analysis': component_analysis,
                    'series_length': len(series)
                },
                recommendations=recommendations,
                warnings=warnings_list,
                processing_time=time.time() - start_time
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error in time series decomposition: {e}")
            return TimeSeriesAnalysisResult(
                analysis_type="decomposition_error",
                interpretation=f"Error during time series decomposition: {str(e)}",
                warnings=[str(e)],
                processing_time=time.time() - start_time
            )
    
    def _detect_seasonal_period(self, series: pd.Series) -> Optional[int]:
        """Detect seasonal period using autocorrelation analysis."""
        try:
            # Use the seasonality detection from base class
            seasonality_info = self._detect_seasonality(series)
            
            if seasonality_info['has_seasonality']:
                return seasonality_info['dominant_period']
            
            # Fallback: try common periods based on data frequency
            freq = self._infer_frequency(pd.DataFrame(index=series.index))
            
            if freq == 'D':  # Daily data
                test_periods = [7, 30, 365]
            elif freq in ['H', 'T']:  # Hourly or minutely
                test_periods = [24, 168]  # Daily, weekly
            elif freq == 'M':  # Monthly
                test_periods = [12]
            else:
                test_periods = [4, 7, 12, 24]
                
            # Test periods with sufficient data
            for period in test_periods:
                if len(series) >= 3 * period:
                    # Simple test: check autocorrelation at the period
                    try:
                        autocorr_at_period = series.autocorr(lag=period)
                        if not pd.isna(autocorr_at_period) and abs(autocorr_at_period) > 0.3:
                            return period
                    except Exception:
                        continue
                        
            return None
            
        except Exception as e:
            logger.debug(f"Error detecting seasonal period: {e}")
            return None
    
    def _seasonal_decomposition(self, series: pd.Series, period: int) -> Dict[str, pd.Series]:
        """Perform seasonal decomposition using statsmodels."""
        try:
            decomp = seasonal_decompose(
                series, 
                model=self.model,
                period=period,
                extrapolate_trend=self.extrapolate_trend,
                two_sided=self.two_sided
            )
            
            return {
                'trend': decomp.trend,
                'seasonal': decomp.seasonal,
                'resid': decomp.resid,
                'observed': series
            }
            
        except Exception as e:
            logger.error(f"Seasonal decomposition failed: {e}")
            # Fallback to simple trend extraction
            return self._simple_trend_decomposition(series)
    
    def _stl_decomposition(self, series: pd.Series, period: int) -> Dict[str, pd.Series]:
        """Perform STL decomposition for robust trend/seasonal extraction."""
        try:
            from statsmodels.tsa.seasonal import STL
            
            # STL parameters
            stl = STL(
                series,
                seasonal=7,  # Seasonal smoothing parameter
                trend=None,   # Trend smoothing (auto)
                period=period,
                robust=True   # Robust to outliers
            )
            
            result = stl.fit()
            
            return {
                'trend': result.trend,
                'seasonal': result.seasonal,
                'resid': result.resid,
                'observed': series
            }
            
        except Exception as e:
            logger.warning(f"STL decomposition failed: {e}, falling back to seasonal_decompose")
            return self._seasonal_decomposition(series, period)
    
    def _x13_decomposition(self, series: pd.Series, period: int) -> Dict[str, pd.Series]:
        """Perform X-13ARIMA-SEATS decomposition if available."""
        try:
            from statsmodels.tsa.x13 import x13_arima_analysis
            
            # X-13 requires specific frequency
            x13_result = x13_arima_analysis(series)
            
            return {
                'trend': x13_result.trend,
                'seasonal': x13_result.seasonal,
                'resid': x13_result.irregular,
                'observed': series
            }
            
        except Exception as e:
            logger.warning(f"X-13 decomposition not available: {e}, falling back to seasonal_decompose")
            return self._seasonal_decomposition(series, period)
    
    def _simple_trend_decomposition(self, series: pd.Series) -> Dict[str, pd.Series]:
        """Simple trend extraction without seasonal component."""
        try:
            # Use rolling mean as trend
            window_size = max(3, len(series) // 20)  # Adaptive window size
            trend = series.rolling(window=window_size, center=True, min_periods=1).mean()
            residual = series - trend
            
            # No seasonal component
            seasonal = pd.Series(0, index=series.index)
            
            return {
                'trend': trend,
                'seasonal': seasonal,
                'resid': residual,
                'observed': series
            }
            
        except Exception as e:
            logger.error(f"Simple trend decomposition failed: {e}")
            # Ultimate fallback
            return {
                'trend': pd.Series(series.mean(), index=series.index),
                'seasonal': pd.Series(0, index=series.index),
                'resid': series - series.mean(),
                'observed': series
            }
    
    def _analyze_components(self, decomp_result: Dict[str, pd.Series], has_seasonality: bool) -> Dict[str, Any]:
        """Analyze decomposition components for insights."""
        analysis = {
            'trend_strength': 0.0,
            'seasonal_strength': 0.0,
            'residual_variance': 0.0,
            'decomposition_quality': 0.0,
            'recommendations': [],
            'warnings': []
        }
        
        try:
            observed = decomp_result['observed']
            trend = decomp_result['trend'].dropna()
            seasonal = decomp_result['seasonal']
            residual = decomp_result['resid'].dropna()
            
            # Calculate component strengths
            if len(trend) > 0:
                trend_var = trend.var()
                total_var = observed.var()
                if total_var > 0:
                    analysis['trend_strength'] = min(1.0, trend_var / total_var)
            
            if has_seasonality and len(seasonal) > 0:
                seasonal_var = seasonal.var()
                total_var = observed.var()
                if total_var > 0:
                    analysis['seasonal_strength'] = min(1.0, seasonal_var / total_var)
            
            # Residual analysis
            if len(residual) > 0:
                analysis['residual_variance'] = residual.var()
                
                # Check residual properties
                if analysis['residual_variance'] < 0.1 * observed.var():
                    analysis['recommendations'].append("Low residual variance - good decomposition quality")
                else:
                    analysis['warnings'].append("High residual variance - decomposition may be incomplete")
            
            # Overall decomposition quality
            explained_variance = analysis['trend_strength'] + analysis['seasonal_strength']
            analysis['decomposition_quality'] = min(1.0, explained_variance)
            
            # Generate insights
            if analysis['trend_strength'] > 0.7:
                analysis['recommendations'].append("Strong trend component - trend modeling is important")
            elif analysis['trend_strength'] < 0.2:
                analysis['recommendations'].append("Weak trend component - focus on seasonal/residual patterns")
                
            if has_seasonality:
                if analysis['seasonal_strength'] > 0.5:
                    analysis['recommendations'].append("Strong seasonal component - seasonal modeling essential")
                elif analysis['seasonal_strength'] < 0.2:
                    analysis['warnings'].append("Weak seasonal pattern - verify seasonal period")
                    
        except Exception as e:
            analysis['warnings'].append(f"Error in component analysis: {e}")
            
        return analysis


class TrendAnalysisTransformer(TimeSeriesTransformer):
    """
    sklearn-compatible transformer for trend analysis and extraction.
    
    Analyzes trend characteristics including direction, strength, and changepoints
    using various trend detection methods.
    
    Parameters:
    -----------
    method : str, default='linear'
        Trend detection method: 'linear', 'polynomial', 'lowess', 'hodrick_prescott'
    degree : int, default=1
        Polynomial degree (for polynomial method)
    alpha : float, default=0.05
        Significance level for trend tests
    changepoint_detection : bool, default=True
        Whether to detect trend changepoints
    min_segment_length : int, default=10
        Minimum segment length for changepoint detection
    """
    
    def __init__(self, method='linear', degree=1, alpha=0.05, 
                 changepoint_detection=True, min_segment_length=10, **kwargs):
        super().__init__(**kwargs)
        self.method = method
        self.degree = degree
        self.alpha = alpha
        self.changepoint_detection = changepoint_detection
        self.min_segment_length = min_segment_length
        self.trend_parameters_ = {}
        self.changepoints_ = []
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fit the trend analysis transformer."""
        if self.validate_input:
            X, y = self._validate_time_series(X, y)
        return self
        
    def transform(self, X: pd.DataFrame) -> TimeSeriesAnalysisResult:
        """
        Perform trend analysis on time series data.
        
        Returns:
        --------
        result : TimeSeriesAnalysisResult
            Trend analysis results with direction, strength, and changepoints
        """
        start_time = time.time()
        
        if self.validate_input:
            X, _ = self._validate_time_series(X, None)
            
        try:
            # Use first column for analysis
            if len(X.columns) == 0:
                raise ValueError("No columns found in time series data")
                
            series = X.iloc[:, 0].dropna()
            if len(series) < 10:
                raise ValueError("Insufficient data points for trend analysis")
            
            # Extract trend using specified method
            trend_result = self._extract_trend(series)
            
            # Analyze trend properties
            trend_analysis = self._analyze_trend_properties(series, trend_result['trend'])
            
            # Detect changepoints if requested
            changepoints = []
            if self.changepoint_detection:
                changepoints = self._detect_changepoints(series)
                
            self.trend_parameters_ = trend_result
            self.changepoints_ = changepoints
            
            # Generate interpretation
            trend_direction = trend_analysis['direction']
            trend_strength = trend_analysis['strength']
            
            if trend_strength < 0.1:
                interpretation = "No significant trend detected - series appears stationary around mean"
            elif trend_strength < 0.3:
                interpretation = f"Weak {trend_direction} trend detected (strength: {trend_strength:.2f})"
            elif trend_strength < 0.7:
                interpretation = f"Moderate {trend_direction} trend detected (strength: {trend_strength:.2f})"
            else:
                interpretation = f"Strong {trend_direction} trend detected (strength: {trend_strength:.2f})"
            
            if len(changepoints) > 0:
                interpretation += f" with {len(changepoints)} trend changepoint(s)"
            
            # Generate recommendations
            recommendations = []
            warnings_list = []
            
            if trend_strength > 0.5:
                recommendations.append("Strong trend detected - detrending may be necessary for stationary modeling")
                recommendations.append("Consider trend-aware forecasting methods")
            else:
                recommendations.append("Weak or no trend - focus on seasonal or residual patterns")
            
            if len(changepoints) > 0:
                recommendations.append(f"Trend changepoints detected - consider structural break models")
                for i, cp in enumerate(changepoints[:3]):  # Show first 3
                    cp_date = series.index[cp] if cp < len(series.index) else "unknown"
                    recommendations.append(f"Changepoint {i+1} at position {cp} ({cp_date})")
            
            # Add analysis-specific recommendations
            recommendations.extend(trend_analysis.get('recommendations', []))
            warnings_list.extend(trend_analysis.get('warnings', []))
            
            result = TimeSeriesAnalysisResult(
                analysis_type="trend_analysis",
                interpretation=interpretation,
                trend_component=trend_result['trend'],
                model_diagnostics={
                    'trend_method': self.method,
                    'trend_direction': trend_direction,
                    'trend_strength': trend_strength,
                    'trend_parameters': trend_result.get('parameters', {}),
                    'changepoints': changepoints,
                    'trend_analysis': trend_analysis,
                    'series_length': len(series)
                },
                recommendations=recommendations,
                warnings=warnings_list,
                processing_time=time.time() - start_time
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error in trend analysis: {e}")
            return TimeSeriesAnalysisResult(
                analysis_type="trend_analysis_error",
                interpretation=f"Error during trend analysis: {str(e)}",
                warnings=[str(e)],
                processing_time=time.time() - start_time
            )
    
    def _extract_trend(self, series: pd.Series) -> Dict[str, Any]:
        """Extract trend using the specified method."""
        try:
            x = np.arange(len(series))
            y = series.values
            
            if self.method == 'linear':
                # Linear regression
                coeffs = np.polyfit(x, y, 1)
                trend_values = np.polyval(coeffs, x)
                
                return {
                    'trend': pd.Series(trend_values, index=series.index),
                    'parameters': {'slope': coeffs[0], 'intercept': coeffs[1]},
                    'method': 'linear'
                }
                
            elif self.method == 'polynomial':
                # Polynomial regression
                coeffs = np.polyfit(x, y, self.degree)
                trend_values = np.polyval(coeffs, x)
                
                return {
                    'trend': pd.Series(trend_values, index=series.index),
                    'parameters': {'coefficients': coeffs.tolist(), 'degree': self.degree},
                    'method': 'polynomial'
                }
                
            elif self.method == 'lowess':
                # LOWESS smoothing
                from statsmodels.nonparametric.smoothers_lowess import lowess
                
                smoothed = lowess(y, x, frac=0.3, return_sorted=False)
                
                return {
                    'trend': pd.Series(smoothed, index=series.index),
                    'parameters': {'method': 'lowess', 'frac': 0.3},
                    'method': 'lowess'
                }
                
            elif self.method == 'hodrick_prescott':
                # Hodrick-Prescott filter
                try:
                    from statsmodels.tsa.filters.hp_filter import hpfilter
                    
                    cycle, trend_values = hpfilter(series, lamb=1600)  # Standard lambda for monthly data
                    
                    return {
                        'trend': trend_values,
                        'cycle': cycle,
                        'parameters': {'lambda': 1600},
                        'method': 'hodrick_prescott'
                    }
                except ImportError:
                    logger.warning("Hodrick-Prescott filter not available, using linear trend")
                    return self._extract_trend_linear(series)
                    
            else:
                raise ValueError(f"Unknown trend method: {self.method}")
                
        except Exception as e:
            logger.error(f"Error extracting trend: {e}")
            # Fallback to linear trend
            return self._extract_trend_linear(series)
    
    def _extract_trend_linear(self, series: pd.Series) -> Dict[str, Any]:
        """Fallback linear trend extraction."""
        x = np.arange(len(series))
        y = series.values
        coeffs = np.polyfit(x, y, 1)
        trend_values = np.polyval(coeffs, x)
        
        return {
            'trend': pd.Series(trend_values, index=series.index),
            'parameters': {'slope': coeffs[0], 'intercept': coeffs[1]},
            'method': 'linear_fallback'
        }
    
    def _analyze_trend_properties(self, series: pd.Series, trend: pd.Series) -> Dict[str, Any]:
        """Analyze properties of the extracted trend."""
        analysis = {
            'direction': 'unknown',
            'strength': 0.0,
            'significance': 0.0,
            'recommendations': [],
            'warnings': []
        }
        
        try:
            # Determine trend direction
            if len(trend) > 1:
                overall_change = trend.iloc[-1] - trend.iloc[0]
                if overall_change > 0:
                    analysis['direction'] = 'upward'
                elif overall_change < 0:
                    analysis['direction'] = 'downward'
                else:
                    analysis['direction'] = 'flat'
            
            # Calculate trend strength (correlation with time)
            if len(series) > 2:
                time_index = np.arange(len(series))
                correlation = np.corrcoef(series.values, time_index)[0, 1]
                analysis['strength'] = abs(correlation) if not np.isnan(correlation) else 0.0
            
            # Statistical significance test (simple t-test on slope)
            if self.method in ['linear', 'polynomial'] and 'slope' in self.trend_parameters_.get('parameters', {}):
                try:
                    from scipy import stats
                    
                    x = np.arange(len(series))
                    y = series.values
                    
                    # Simple linear regression t-test
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                    analysis['significance'] = 1.0 - p_value  # Convert to significance measure
                    
                    if p_value <= self.alpha:
                        analysis['recommendations'].append(f"Trend is statistically significant (p={p_value:.4f})")
                    else:
                        analysis['warnings'].append(f"Trend is not statistically significant (p={p_value:.4f})")
                        
                except Exception as e:
                    analysis['warnings'].append(f"Could not test trend significance: {e}")
                    
        except Exception as e:
            analysis['warnings'].append(f"Error in trend property analysis: {e}")
            
        return analysis
    
    def _detect_changepoints(self, series: pd.Series) -> List[int]:
        """Detect trend changepoints using simple methods."""
        changepoints = []
        
        try:
            # Simple changepoint detection using moving averages
            window_size = max(5, len(series) // 20)
            
            if len(series) < 2 * window_size:
                return changepoints  # Not enough data
                
            # Calculate moving averages
            ma_short = series.rolling(window=window_size, center=True).mean()
            ma_long = series.rolling(window=window_size * 2, center=True).mean()
            
            # Find crossover points
            diff = ma_short - ma_long
            sign_changes = np.diff(np.sign(diff.dropna()))
            
            # Get changepoint indices
            change_indices = np.where(sign_changes != 0)[0]
            
            # Filter changepoints by minimum segment length
            filtered_changepoints = []
            last_cp = 0
            
            for cp in change_indices:
                if cp - last_cp >= self.min_segment_length:
                    filtered_changepoints.append(cp)
                    last_cp = cp
                    
            changepoints = filtered_changepoints
            
        except Exception as e:
            logger.debug(f"Error in changepoint detection: {e}")
            
        return changepoints


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


# ============================================================================
# ARIMA Forecasting Models (Subtask 37.6)
# ============================================================================

class ARIMAForecastTransformer(TimeSeriesTransformer):
    """
    sklearn-compatible transformer for ARIMA time series forecasting.
    
    Implements AutoRegressive Integrated Moving Average (ARIMA) models
    for time series prediction with confidence intervals.
    
    Parameters:
    -----------
    order : tuple of int, default=(1, 1, 1)
        The (p, d, q) order of the model for the number of AR parameters,
        differences, and MA parameters
    seasonal_order : tuple of int, default=(0, 0, 0, 0)
        The (P, D, Q, s) order of the seasonal component
    trend : str, default='c'
        Parameter controlling the deterministic trend polynomial
    enforce_stationarity : bool, default=True
        Whether or not to transform AR parameters to enforce stationarity
    enforce_invertibility : bool, default=True
        Whether or not to transform MA parameters to enforce invertibility
    forecast_steps : int, default=10
        Number of steps to forecast ahead
    alpha : float, default=0.05
        The significance level for confidence intervals (1 - alpha = confidence level)
    """
    
    def __init__(self, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0), 
                 trend='c', enforce_stationarity=True, enforce_invertibility=True,
                 forecast_steps=10, alpha=0.05, **kwargs):
        super().__init__(**kwargs)
        self.order = order
        self.seasonal_order = seasonal_order
        self.trend = trend
        self.enforce_stationarity = enforce_stationarity
        self.enforce_invertibility = enforce_invertibility
        self.forecast_steps = forecast_steps
        self.alpha = alpha
        self.model_ = None
        self.fitted_model_ = None
        self.training_data_ = None
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fit the ARIMA model to the time series data."""
        if self.validate_input:
            X, y = self._validate_time_series(X, y)
            
        try:
            # Use the first column if X is a DataFrame, otherwise use y
            if isinstance(X, pd.DataFrame):
                if X.shape[1] == 1:
                    data = X.iloc[:, 0]
                else:
                    raise ValueError("ARIMA model requires univariate time series data")
            elif y is not None:
                data = y
            else:
                raise ValueError("No target variable provided for ARIMA modeling")
                
            # Store training data for forecasting
            self.training_data_ = data
            
            # Create and fit ARIMA model
            self.model_ = ARIMA(
                data,
                order=self.order,
                seasonal_order=self.seasonal_order,
                trend=self.trend,
                enforce_stationarity=self.enforce_stationarity,
                enforce_invertibility=self.enforce_invertibility
            )
            
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                self.fitted_model_ = self.model_.fit()
                
        except Exception as e:
            logger = get_logger(__name__)
            logger.error(f"Error fitting ARIMA model: {e}")
            raise
            
        return self
        
    def transform(self, X: pd.DataFrame) -> TimeSeriesAnalysisResult:
        """
        Generate ARIMA forecasts and model diagnostics.
        
        Returns:
        --------
        result : TimeSeriesAnalysisResult
            ARIMA model results with forecasts, confidence intervals, and diagnostics
        """
        start_time = time.time()
        logger = get_logger(__name__)
        
        try:
            check_is_fitted(self, 'fitted_model_')
            
            if self.validate_input:
                X, _ = self._validate_time_series(X)
                
            # Generate forecasts
            forecast_result = self.fitted_model_.get_forecast(steps=self.forecast_steps)
            forecast_values = forecast_result.predicted_mean
            forecast_ci = forecast_result.conf_int(alpha=self.alpha)
            
            # Create forecast index
            if hasattr(self.training_data_, 'index') and isinstance(self.training_data_.index, pd.DatetimeIndex):
                last_date = self.training_data_.index[-1]
                freq = self.training_data_.index.freq or pd.infer_freq(self.training_data_.index)
                if freq is not None:
                    forecast_index = pd.date_range(
                        start=last_date + pd.Timedelta(freq),
                        periods=self.forecast_steps,
                        freq=freq
                    )
                else:
                    forecast_index = range(len(self.training_data_), 
                                         len(self.training_data_) + self.forecast_steps)
            else:
                forecast_index = range(len(self.training_data_), 
                                     len(self.training_data_) + self.forecast_steps)
            
            # Model diagnostics
            aic = self.fitted_model_.aic
            bic = self.fitted_model_.bic
            loglikelihood = self.fitted_model_.llf
            
            # Residual diagnostics
            residuals = self.fitted_model_.resid
            ljung_box = acorr_ljungbox(residuals, lags=min(10, len(residuals)//5), return_df=True)
            
            # In-sample predictions for evaluation
            in_sample_pred = self.fitted_model_.fittedvalues
            
            result_data = {
                'model_type': 'ARIMA',
                'order': self.order,
                'seasonal_order': self.seasonal_order,
                'forecast_values': forecast_values.tolist(),
                'forecast_index': forecast_index.tolist() if hasattr(forecast_index, 'tolist') else list(forecast_index),
                'forecast_lower_ci': forecast_ci.iloc[:, 0].tolist(),
                'forecast_upper_ci': forecast_ci.iloc[:, 1].tolist(),
                'confidence_level': 1 - self.alpha,
                'in_sample_predictions': in_sample_pred.tolist(),
                'residuals': residuals.tolist(),
                'model_fit': {
                    'aic': float(aic),
                    'bic': float(bic),
                    'log_likelihood': float(loglikelihood),
                    'num_params': self.fitted_model_.params.shape[0]
                },
                'residual_diagnostics': {
                    'ljung_box_stat': float(ljung_box.iloc[-1]['lb_stat']),
                    'ljung_box_pvalue': float(ljung_box.iloc[-1]['lb_pvalue']),
                    'residual_autocorrelation': 'No significant autocorrelation' if ljung_box.iloc[-1]['lb_pvalue'] > 0.05 else 'Significant autocorrelation detected'
                },
                'model_params': {
                    param: float(value) for param, value in self.fitted_model_.params.items()
                }
            }
            
            # Add recommendations
            recommendations = []
            if ljung_box.iloc[-1]['lb_pvalue'] <= 0.05:
                recommendations.append("Residuals show significant autocorrelation - consider adjusting model parameters")
            if aic > 1000:  # Arbitrary threshold for demonstration
                recommendations.append("High AIC value - model may be overfitting")
                
            result_data['recommendations'] = recommendations
            
            return TimeSeriesAnalysisResult(
                data=result_data,
                execution_time=time.time() - start_time,
                metadata=CompositionMetadata(
                    timestamp=datetime.now(),
                    data_shape=(len(self.training_data_), 1) if hasattr(self.training_data_, '__len__') else (0, 1),
                    parameters={
                        'order': self.order,
                        'seasonal_order': self.seasonal_order,
                        'forecast_steps': self.forecast_steps
                    }
                )
            )
            
        except Exception as e:
            logger.error(f"Error in ARIMA forecasting: {e}")
            raise TimeSeriesValidationError(f"ARIMA forecasting failed: {str(e)}")


class SARIMAForecastTransformer(TimeSeriesTransformer):
    """
    sklearn-compatible transformer for Seasonal ARIMA time series forecasting.
    
    Implements Seasonal AutoRegressive Integrated Moving Average (SARIMA) models
    for time series with seasonal patterns.
    
    Parameters:
    -----------
    order : tuple of int, default=(1, 1, 1)
        The (p, d, q) order of the non-seasonal part
    seasonal_order : tuple of int, default=(1, 1, 1, 12)
        The (P, D, Q, s) order of the seasonal part
    trend : str, default='c'
        Parameter controlling the deterministic trend polynomial
    enforce_stationarity : bool, default=True
        Whether or not to transform AR parameters to enforce stationarity
    enforce_invertibility : bool, default=True
        Whether or not to transform MA parameters to enforce invertibility
    forecast_steps : int, default=10
        Number of steps to forecast ahead
    alpha : float, default=0.05
        The significance level for confidence intervals
    """
    
    def __init__(self, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12),
                 trend='c', enforce_stationarity=True, enforce_invertibility=True,
                 forecast_steps=10, alpha=0.05, **kwargs):
        super().__init__(**kwargs)
        self.order = order
        self.seasonal_order = seasonal_order
        self.trend = trend
        self.enforce_stationarity = enforce_stationarity
        self.enforce_invertibility = enforce_invertibility
        self.forecast_steps = forecast_steps
        self.alpha = alpha
        self.model_ = None
        self.fitted_model_ = None
        self.training_data_ = None
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fit the SARIMA model to the time series data."""
        if self.validate_input:
            X, y = self._validate_time_series(X, y)
            
        try:
            # Use the first column if X is a DataFrame, otherwise use y
            if isinstance(X, pd.DataFrame):
                if X.shape[1] == 1:
                    data = X.iloc[:, 0]
                else:
                    raise ValueError("SARIMA model requires univariate time series data")
            elif y is not None:
                data = y
            else:
                raise ValueError("No target variable provided for SARIMA modeling")
                
            # Store training data for forecasting
            self.training_data_ = data
            
            # Create and fit SARIMA model using SARIMAX
            self.model_ = SARIMAX(
                data,
                order=self.order,
                seasonal_order=self.seasonal_order,
                trend=self.trend,
                enforce_stationarity=self.enforce_stationarity,
                enforce_invertibility=self.enforce_invertibility
            )
            
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                self.fitted_model_ = self.model_.fit()
                
        except Exception as e:
            logger = get_logger(__name__)
            logger.error(f"Error fitting SARIMA model: {e}")
            raise
            
        return self
        
    def transform(self, X: pd.DataFrame) -> TimeSeriesAnalysisResult:
        """
        Generate SARIMA forecasts and model diagnostics.
        
        Returns:
        --------
        result : TimeSeriesAnalysisResult
            SARIMA model results with seasonal forecasts and diagnostics
        """
        start_time = time.time()
        logger = get_logger(__name__)
        
        try:
            check_is_fitted(self, 'fitted_model_')
            
            if self.validate_input:
                X, _ = self._validate_time_series(X)
                
            # Generate forecasts
            forecast_result = self.fitted_model_.get_forecast(steps=self.forecast_steps)
            forecast_values = forecast_result.predicted_mean
            forecast_ci = forecast_result.conf_int(alpha=self.alpha)
            
            # Create forecast index
            if hasattr(self.training_data_, 'index') and isinstance(self.training_data_.index, pd.DatetimeIndex):
                last_date = self.training_data_.index[-1]
                freq = self.training_data_.index.freq or pd.infer_freq(self.training_data_.index)
                if freq is not None:
                    forecast_index = pd.date_range(
                        start=last_date + pd.Timedelta(freq),
                        periods=self.forecast_steps,
                        freq=freq
                    )
                else:
                    forecast_index = range(len(self.training_data_), 
                                         len(self.training_data_) + self.forecast_steps)
            else:
                forecast_index = range(len(self.training_data_), 
                                     len(self.training_data_) + self.forecast_steps)
            
            # Model diagnostics
            aic = self.fitted_model_.aic
            bic = self.fitted_model_.bic
            loglikelihood = self.fitted_model_.llf
            
            # Residual diagnostics
            residuals = self.fitted_model_.resid
            ljung_box = acorr_ljungbox(residuals, lags=min(10, len(residuals)//5), return_df=True)
            
            # In-sample predictions
            in_sample_pred = self.fitted_model_.fittedvalues
            
            result_data = {
                'model_type': 'SARIMA',
                'order': self.order,
                'seasonal_order': self.seasonal_order,
                'seasonal_period': self.seasonal_order[3],
                'forecast_values': forecast_values.tolist(),
                'forecast_index': forecast_index.tolist() if hasattr(forecast_index, 'tolist') else list(forecast_index),
                'forecast_lower_ci': forecast_ci.iloc[:, 0].tolist(),
                'forecast_upper_ci': forecast_ci.iloc[:, 1].tolist(),
                'confidence_level': 1 - self.alpha,
                'in_sample_predictions': in_sample_pred.tolist(),
                'residuals': residuals.tolist(),
                'model_fit': {
                    'aic': float(aic),
                    'bic': float(bic),
                    'log_likelihood': float(loglikelihood),
                    'num_params': self.fitted_model_.params.shape[0]
                },
                'residual_diagnostics': {
                    'ljung_box_stat': float(ljung_box.iloc[-1]['lb_stat']),
                    'ljung_box_pvalue': float(ljung_box.iloc[-1]['lb_pvalue']),
                    'residual_autocorrelation': 'No significant autocorrelation' if ljung_box.iloc[-1]['lb_pvalue'] > 0.05 else 'Significant autocorrelation detected'
                },
                'model_params': {
                    param: float(value) for param, value in self.fitted_model_.params.items()
                }
            }
            
            # Add recommendations
            recommendations = []
            if ljung_box.iloc[-1]['lb_pvalue'] <= 0.05:
                recommendations.append("Residuals show significant autocorrelation - consider adjusting seasonal parameters")
            if self.seasonal_order[3] > 24:
                recommendations.append("Large seasonal period detected - ensure sufficient data for reliable estimation")
                
            result_data['recommendations'] = recommendations
            
            return TimeSeriesAnalysisResult(
                data=result_data,
                execution_time=time.time() - start_time,
                metadata=CompositionMetadata(
                    timestamp=datetime.now(),
                    data_shape=(len(self.training_data_), 1) if hasattr(self.training_data_, '__len__') else (0, 1),
                    parameters={
                        'order': self.order,
                        'seasonal_order': self.seasonal_order,
                        'forecast_steps': self.forecast_steps
                    }
                )
            )
            
        except Exception as e:
            logger.error(f"Error in SARIMA forecasting: {e}")
            raise TimeSeriesValidationError(f"SARIMA forecasting failed: {str(e)}")


class AutoARIMATransformer(TimeSeriesTransformer):
    """
    sklearn-compatible transformer for automatic ARIMA parameter selection.
    
    Automatically selects optimal ARIMA parameters using information criteria
    (AIC/BIC) through grid search over specified parameter ranges.
    
    Parameters:
    -----------
    max_p : int, default=5
        Maximum number of AR terms to test
    max_d : int, default=2
        Maximum number of differences to test
    max_q : int, default=5
        Maximum number of MA terms to test
    max_P : int, default=2
        Maximum number of seasonal AR terms to test
    max_D : int, default=1
        Maximum number of seasonal differences to test
    max_Q : int, default=2
        Maximum number of seasonal MA terms to test
    seasonal_period : int, default=None
        Seasonal period (auto-detected if None)
    information_criterion : str, default='aic'
        Information criterion for model selection ('aic' or 'bic')
    seasonal : bool, default=True
        Whether to consider seasonal models
    stepwise : bool, default=True
        Whether to use stepwise search (faster) or exhaustive search
    forecast_steps : int, default=10
        Number of steps to forecast ahead
    alpha : float, default=0.05
        Significance level for confidence intervals
    """
    
    def __init__(self, max_p=5, max_d=2, max_q=5, max_P=2, max_D=1, max_Q=2,
                 seasonal_period=None, information_criterion='aic', seasonal=True,
                 stepwise=True, forecast_steps=10, alpha=0.05, **kwargs):
        super().__init__(**kwargs)
        self.max_p = max_p
        self.max_d = max_d
        self.max_q = max_q
        self.max_P = max_P
        self.max_D = max_D
        self.max_Q = max_Q
        self.seasonal_period = seasonal_period
        self.information_criterion = information_criterion
        self.seasonal = seasonal
        self.stepwise = stepwise
        self.forecast_steps = forecast_steps
        self.alpha = alpha
        self.best_model_ = None
        self.best_order_ = None
        self.best_seasonal_order_ = None
        self.model_results_ = []
        self.training_data_ = None
        
    def _detect_seasonal_period(self, data):
        """Detect seasonal period from data frequency."""
        if hasattr(data, 'index') and isinstance(data.index, pd.DatetimeIndex):
            freq = data.index.freq or pd.infer_freq(data.index)
            if freq:
                # Common seasonal periods based on frequency
                freq_str = str(freq).upper()
                if 'H' in freq_str:  # Hourly
                    return 24
                elif 'D' in freq_str:  # Daily
                    return 7
                elif 'W' in freq_str:  # Weekly
                    return 52
                elif 'M' in freq_str:  # Monthly
                    return 12
                elif 'Q' in freq_str:  # Quarterly
                    return 4
        
        # Default seasonal period if cannot detect
        return 12
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fit the Auto-ARIMA model by searching optimal parameters."""
        if self.validate_input:
            X, y = self._validate_time_series(X, y)
            
        logger = get_logger(__name__)
        
        try:
            # Use the first column if X is a DataFrame, otherwise use y
            if isinstance(X, pd.DataFrame):
                if X.shape[1] == 1:
                    data = X.iloc[:, 0]
                else:
                    raise ValueError("Auto-ARIMA requires univariate time series data")
            elif y is not None:
                data = y
            else:
                raise ValueError("No target variable provided for Auto-ARIMA modeling")
                
            self.training_data_ = data
            
            # Detect seasonal period if not provided
            if self.seasonal and self.seasonal_period is None:
                self.seasonal_period = self._detect_seasonal_period(data)
            
            # Generate parameter combinations
            if self.stepwise:
                # Stepwise search - start with simple model and expand
                param_combinations = self._stepwise_search()
            else:
                # Exhaustive search
                param_combinations = self._exhaustive_search()
            
            best_ic = np.inf
            self.model_results_ = []
            
            logger.info(f"Testing {len(param_combinations)} ARIMA parameter combinations")
            
            for i, (order, seasonal_order) in enumerate(param_combinations):
                try:
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore")
                        
                        if self.seasonal and seasonal_order != (0, 0, 0, 0):
                            model = SARIMAX(
                                data,
                                order=order,
                                seasonal_order=seasonal_order,
                                enforce_stationarity=True,
                                enforce_invertibility=True
                            )
                        else:
                            model = ARIMA(
                                data,
                                order=order,
                                enforce_stationarity=True,
                                enforce_invertibility=True
                            )
                        
                        fitted_model = model.fit()
                        
                        # Calculate information criterion
                        ic_value = fitted_model.aic if self.information_criterion == 'aic' else fitted_model.bic
                        
                        model_result = {
                            'order': order,
                            'seasonal_order': seasonal_order,
                            'aic': fitted_model.aic,
                            'bic': fitted_model.bic,
                            'log_likelihood': fitted_model.llf,
                            'selected_ic': ic_value
                        }
                        
                        self.model_results_.append(model_result)
                        
                        if ic_value < best_ic:
                            best_ic = ic_value
                            self.best_model_ = fitted_model
                            self.best_order_ = order
                            self.best_seasonal_order_ = seasonal_order
                            
                except Exception as e:
                    logger.debug(f"Failed to fit model {order}, {seasonal_order}: {e}")
                    continue
            
            if self.best_model_ is None:
                raise ValueError("No valid ARIMA model could be fitted to the data")
                
            logger.info(f"Best model found: ARIMA{self.best_order_} x {self.best_seasonal_order_} "
                       f"with {self.information_criterion.upper()}={best_ic:.2f}")
                
        except Exception as e:
            logger.error(f"Error in Auto-ARIMA fitting: {e}")
            raise
            
        return self
        
    def _stepwise_search(self):
        """Generate parameter combinations using stepwise approach."""
        combinations = []
        
        # Start with simple models and expand
        for d in range(self.max_d + 1):
            for p in range(min(3, self.max_p + 1)):
                for q in range(min(3, self.max_q + 1)):
                    if self.seasonal:
                        for D in range(self.max_D + 1):
                            for P in range(min(2, self.max_P + 1)):
                                for Q in range(min(2, self.max_Q + 1)):
                                    seasonal_order = (P, D, Q, self.seasonal_period)
                                    combinations.append(((p, d, q), seasonal_order))
                    else:
                        combinations.append(((p, d, q), (0, 0, 0, 0)))
        
        return combinations[:50]  # Limit to first 50 combinations for efficiency
        
    def _exhaustive_search(self):
        """Generate all parameter combinations for exhaustive search."""
        combinations = []
        
        for p in range(self.max_p + 1):
            for d in range(self.max_d + 1):
                for q in range(self.max_q + 1):
                    if self.seasonal:
                        for P in range(self.max_P + 1):
                            for D in range(self.max_D + 1):
                                for Q in range(self.max_Q + 1):
                                    seasonal_order = (P, D, Q, self.seasonal_period)
                                    combinations.append(((p, d, q), seasonal_order))
                    else:
                        combinations.append(((p, d, q), (0, 0, 0, 0)))
        
        return combinations
        
    def transform(self, X: pd.DataFrame) -> TimeSeriesAnalysisResult:
        """
        Generate forecasts using the best Auto-ARIMA model.
        
        Returns:
        --------
        result : TimeSeriesAnalysisResult
            Auto-ARIMA results with optimal model parameters and forecasts
        """
        start_time = time.time()
        logger = get_logger(__name__)
        
        try:
            check_is_fitted(self, 'best_model_')
            
            if self.validate_input:
                X, _ = self._validate_time_series(X)
                
            # Generate forecasts using best model
            forecast_result = self.best_model_.get_forecast(steps=self.forecast_steps)
            forecast_values = forecast_result.predicted_mean
            forecast_ci = forecast_result.conf_int(alpha=self.alpha)
            
            # Create forecast index
            if hasattr(self.training_data_, 'index') and isinstance(self.training_data_.index, pd.DatetimeIndex):
                last_date = self.training_data_.index[-1]
                freq = self.training_data_.index.freq or pd.infer_freq(self.training_data_.index)
                if freq is not None:
                    forecast_index = pd.date_range(
                        start=last_date + pd.Timedelta(freq),
                        periods=self.forecast_steps,
                        freq=freq
                    )
                else:
                    forecast_index = range(len(self.training_data_), 
                                         len(self.training_data_) + self.forecast_steps)
            else:
                forecast_index = range(len(self.training_data_), 
                                     len(self.training_data_) + self.forecast_steps)
            
            # Model diagnostics
            residuals = self.best_model_.resid
            ljung_box = acorr_ljungbox(residuals, lags=min(10, len(residuals)//5), return_df=True)
            in_sample_pred = self.best_model_.fittedvalues
            
            # Sort model results by IC for comparison
            sorted_models = sorted(self.model_results_, key=lambda x: x['selected_ic'])
            
            result_data = {
                'model_type': 'Auto-ARIMA',
                'best_order': self.best_order_,
                'best_seasonal_order': self.best_seasonal_order_,
                'selection_criterion': self.information_criterion.upper(),
                'models_tested': len(self.model_results_),
                'forecast_values': forecast_values.tolist(),
                'forecast_index': forecast_index.tolist() if hasattr(forecast_index, 'tolist') else list(forecast_index),
                'forecast_lower_ci': forecast_ci.iloc[:, 0].tolist(),
                'forecast_upper_ci': forecast_ci.iloc[:, 1].tolist(),
                'confidence_level': 1 - self.alpha,
                'in_sample_predictions': in_sample_pred.tolist(),
                'residuals': residuals.tolist(),
                'best_model_fit': {
                    'aic': float(self.best_model_.aic),
                    'bic': float(self.best_model_.bic),
                    'log_likelihood': float(self.best_model_.llf),
                    'num_params': self.best_model_.params.shape[0]
                },
                'residual_diagnostics': {
                    'ljung_box_stat': float(ljung_box.iloc[-1]['lb_stat']),
                    'ljung_box_pvalue': float(ljung_box.iloc[-1]['lb_pvalue']),
                    'residual_autocorrelation': 'No significant autocorrelation' if ljung_box.iloc[-1]['lb_pvalue'] > 0.05 else 'Significant autocorrelation detected'
                },
                'model_comparison': sorted_models[:10],  # Top 10 models
                'model_params': {
                    param: float(value) for param, value in self.best_model_.params.items()
                }
            }
            
            # Add recommendations
            recommendations = []
            if len(self.model_results_) < 10:
                recommendations.append("Limited parameter search - consider expanding search ranges")
            if ljung_box.iloc[-1]['lb_pvalue'] <= 0.05:
                recommendations.append("Residuals show autocorrelation - model may need refinement")
                
            result_data['recommendations'] = recommendations
            
            return TimeSeriesAnalysisResult(
                data=result_data,
                execution_time=time.time() - start_time,
                metadata=CompositionMetadata(
                    timestamp=datetime.now(),
                    data_shape=(len(self.training_data_), 1) if hasattr(self.training_data_, '__len__') else (0, 1),
                    parameters={
                        'max_p': self.max_p,
                        'max_d': self.max_d,
                        'max_q': self.max_q,
                        'seasonal': self.seasonal,
                        'information_criterion': self.information_criterion
                    }
                )
            )
            
        except Exception as e:
            logger.error(f"Error in Auto-ARIMA forecasting: {e}")
            raise TimeSeriesValidationError(f"Auto-ARIMA forecasting failed: {str(e)}")


class AdvancedForecastingTransformer(TimeSeriesTransformer):
    """
    sklearn-compatible transformer for advanced forecasting methods.
    
    Provides access to Prophet, Exponential Smoothing, and Ensemble forecasting
    with automatic method selection based on data characteristics and user preferences.
    
    Parameters:
    -----------
    method : str, default='auto'
        Forecasting method: 'prophet', 'exponential_smoothing', 'ensemble', 'auto'
    forecast_steps : int, default=10
        Number of steps to forecast ahead
    confidence_level : float, default=0.95
        Confidence level for prediction intervals
    ensemble_weights : dict, optional
        Weights for ensemble methods (auto-computed if None)
    prophet_params : dict, optional
        Additional parameters for Prophet model
    holt_winters_params : dict, optional
        Additional parameters for Holt-Winters model
    """
    
    def __init__(self, method='auto', forecast_steps=10, confidence_level=0.95,
                 ensemble_weights=None, prophet_params=None, holt_winters_params=None, **kwargs):
        super().__init__(**kwargs)
        self.method = method
        self.forecast_steps = forecast_steps
        self.confidence_level = confidence_level
        self.ensemble_weights = ensemble_weights or {}
        self.prophet_params = prophet_params or {}
        self.holt_winters_params = holt_winters_params or {}
        
        # Model instances
        self.prophet_forecaster_ = None
        self.exponential_smoothing_forecaster_ = None
        self.ensemble_forecaster_ = None
        self.evaluator_ = None
        self.selected_method_ = None
        self.training_data_ = None
        
    def _select_method(self, data: pd.Series) -> str:
        """
        Automatically select the best forecasting method based on data characteristics.
        
        Parameters:
        -----------
        data : pd.Series
            Time series data for analysis
            
        Returns:
        --------
        method : str
            Selected forecasting method
        """
        if self.method != 'auto':
            return self.method
            
        # Analyze data characteristics
        n_observations = len(data)
        frequency = self._infer_frequency(pd.DataFrame({'value': data}))
        
        # Check for strong seasonality
        has_seasonality = self._detect_seasonality(data)
        
        # Selection logic
        if n_observations < 50:
            return 'exponential_smoothing'  # Better for small datasets
        elif has_seasonality and n_observations >= 100:
            # Prophet is good for seasonal data with sufficient observations
            try:
                import prophet
                return 'prophet'
            except ImportError:
                logger.warning("Prophet not available, falling back to exponential smoothing")
                return 'exponential_smoothing'
        elif n_observations >= 200:
            return 'ensemble'  # Use ensemble for large datasets
        else:
            return 'exponential_smoothing'
            
    def _detect_seasonality(self, data: pd.Series) -> bool:
        """
        Detect if the time series has significant seasonality.
        
        Parameters:
        -----------
        data : pd.Series
            Time series data
            
        Returns:
        --------
        has_seasonality : bool
            True if seasonality is detected
        """
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose
            
            # Need at least 2 full cycles for seasonality detection
            min_periods = 24  # Default assumption
            if len(data) < 2 * min_periods:
                return False
                
            decomposition = seasonal_decompose(data, model='additive', period=min_periods)
            seasonal_strength = np.var(decomposition.seasonal) / np.var(data)
            
            return seasonal_strength > 0.1  # Threshold for significant seasonality
        except Exception:
            return False
            
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fit the advanced forecasting model."""
        if self.validate_input:
            X, y = self._validate_time_series(X, y)
            
        logger = get_logger(__name__)
        
        try:
            # Extract univariate time series
            if isinstance(X, pd.DataFrame):
                if X.shape[1] == 1:
                    data = X.iloc[:, 0]
                else:
                    raise ValueError("Advanced forecasting requires univariate time series data")
            elif y is not None:
                data = y
            else:
                raise ValueError("No target variable provided for forecasting")
                
            self.training_data_ = data
            
            # Select forecasting method
            self.selected_method_ = self._select_method(data)
            logger.info(f"Selected forecasting method: {self.selected_method_}")
            
            # Initialize selected method(s)
            if self.selected_method_ == 'prophet':
                self.prophet_forecaster_ = ProphetForecaster(
                    forecast_steps=self.forecast_steps,
                    confidence_level=self.confidence_level,
                    **self.prophet_params
                )
                self.prophet_forecaster_.fit(X, y)
                
            elif self.selected_method_ == 'exponential_smoothing':
                self.exponential_smoothing_forecaster_ = ExponentialSmoothingForecaster(
                    forecast_steps=self.forecast_steps,
                    confidence_level=self.confidence_level,
                    **self.holt_winters_params
                )
                self.exponential_smoothing_forecaster_.fit(X, y)
                
            elif self.selected_method_ == 'ensemble':
                self.ensemble_forecaster_ = EnsembleForecaster(
                    forecast_steps=self.forecast_steps,
                    confidence_level=self.confidence_level,
                    weights=self.ensemble_weights
                )
                self.ensemble_forecaster_.fit(X, y)
                
            # Initialize evaluator
            self.evaluator_ = ForecastEvaluator()
            
        except Exception as e:
            logger.error(f"Error in advanced forecasting fit: {e}")
            raise TimeSeriesValidationError(f"Advanced forecasting fit failed: {str(e)}")
            
        return self
        
    def transform(self, X: pd.DataFrame) -> PipelineResult:
        """Generate forecasts using the fitted model."""
        check_is_fitted(self, ['selected_method_', 'training_data_'])
        
        start_time = time.time()
        logger = get_logger(__name__)
        
        try:
            # Generate forecasts based on selected method
            if self.selected_method_ == 'prophet' and self.prophet_forecaster_:
                result = self.prophet_forecaster_.transform(X)
            elif self.selected_method_ == 'exponential_smoothing' and self.exponential_smoothing_forecaster_:
                result = self.exponential_smoothing_forecaster_.transform(X)
            elif self.selected_method_ == 'ensemble' and self.ensemble_forecaster_:
                result = self.ensemble_forecaster_.transform(X)
            else:
                raise ValueError(f"No fitted model available for method: {self.selected_method_}")
                
            # Add method information to result
            result.data['selected_method'] = self.selected_method_
            result.data['available_methods'] = ['prophet', 'exponential_smoothing', 'ensemble']
            
            return result
            
        except Exception as e:
            logger.error(f"Error in advanced forecasting transform: {e}")
            raise TimeSeriesValidationError(f"Advanced forecasting transform failed: {str(e)}")


class ProphetForecaster(TimeSeriesTransformer):
    """
    sklearn-compatible transformer for Facebook Prophet forecasting.
    
    Provides robust forecasting with automatic handling of trends, seasonality,
    holidays, and special events. Prophet is particularly effective for time series
    with strong seasonal effects and several seasons of historical data.
    
    Parameters:
    -----------
    forecast_steps : int, default=10
        Number of steps to forecast ahead
    confidence_level : float, default=0.95
        Confidence level for prediction intervals
    growth : str, default='linear'
        Type of trend: 'linear' or 'logistic'
    seasonality_mode : str, default='additive'
        Seasonality mode: 'additive' or 'multiplicative'
    daily_seasonality : bool or str, default='auto'
        Whether to include daily seasonality
    weekly_seasonality : bool or str, default='auto'
        Whether to include weekly seasonality
    yearly_seasonality : bool or str, default='auto'
        Whether to include yearly seasonality
    holidays : pd.DataFrame, optional
        Holiday dataframe with columns 'holiday' and 'ds'
    changepoint_prior_scale : float, default=0.05
        Flexibility of automatic changepoint selection
    seasonality_prior_scale : float, default=10.0
        Strength of seasonality model
    """
    
    def __init__(self, forecast_steps=10, confidence_level=0.95, growth='linear',
                 seasonality_mode='additive', daily_seasonality='auto',
                 weekly_seasonality='auto', yearly_seasonality='auto',
                 holidays=None, changepoint_prior_scale=0.05,
                 seasonality_prior_scale=10.0, **kwargs):
        super().__init__(**kwargs)
        self.forecast_steps = forecast_steps
        self.confidence_level = confidence_level
        self.growth = growth
        self.seasonality_mode = seasonality_mode
        self.daily_seasonality = daily_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.yearly_seasonality = yearly_seasonality
        self.holidays = holidays
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale
        
        self.model_ = None
        self.forecast_ = None
        self.training_data_ = None
        self._prophet_available = self._check_prophet_availability()
        
    def _check_prophet_availability(self) -> bool:
        """
        Check if Prophet is available and can be imported.
        
        Returns:
        --------
        available : bool
            True if Prophet can be imported
        """
        try:
            import prophet
            return True
        except ImportError:
            logger.warning("Prophet package not available. Install with: pip install prophet")
            return False
            
    def _prepare_prophet_data(self, data: pd.Series) -> pd.DataFrame:
        """
        Prepare data in Prophet's required format.
        
        Parameters:
        -----------
        data : pd.Series
            Time series data with datetime index
            
        Returns:
        --------
        df : pd.DataFrame
            DataFrame with columns 'ds' (dates) and 'y' (values)
        """
        df = pd.DataFrame({
            'ds': data.index,
            'y': data.values
        })
        return df
        
    def _create_default_holidays(self, data: pd.Series) -> Optional[pd.DataFrame]:
        """
        Create a basic holidays dataframe if none provided.
        
        Parameters:
        -----------
        data : pd.Series
            Time series data to analyze date range
            
        Returns:
        --------
        holidays : pd.DataFrame or None
            Basic holidays dataframe or None if not applicable
        """
        # For now, return None - users can provide custom holidays
        # Future enhancement: could include common holidays for different countries
        return None
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fit the Prophet forecasting model."""
        if not self._prophet_available:
            raise ImportError("Prophet package is required but not available. Install with: pip install prophet")
            
        if self.validate_input:
            X, y = self._validate_time_series(X, y)
            
        logger = get_logger(__name__)
        
        try:
            # Extract univariate time series
            if isinstance(X, pd.DataFrame):
                if X.shape[1] == 1:
                    data = X.iloc[:, 0]
                else:
                    raise ValueError("Prophet forecasting requires univariate time series data")
            elif y is not None:
                data = y
            else:
                raise ValueError("No target variable provided for Prophet forecasting")
                
            self.training_data_ = data
            
            # Prepare data for Prophet
            prophet_data = self._prepare_prophet_data(data)
            
            # Import Prophet (after checking availability)
            from prophet import Prophet
            
            # Initialize Prophet model with parameters
            model_params = {
                'growth': self.growth,
                'seasonality_mode': self.seasonality_mode,
                'daily_seasonality': self.daily_seasonality,
                'weekly_seasonality': self.weekly_seasonality,
                'yearly_seasonality': self.yearly_seasonality,
                'changepoint_prior_scale': self.changepoint_prior_scale,
                'seasonality_prior_scale': self.seasonality_prior_scale
            }
            
            if self.holidays is not None:
                model_params['holidays'] = self.holidays
            elif len(data) > 365:  # Only add holidays for data spanning more than a year
                default_holidays = self._create_default_holidays(data)
                if default_holidays is not None:
                    model_params['holidays'] = default_holidays
                    
            # Suppress Prophet's verbose output
            import logging
            logging.getLogger('prophet').setLevel(logging.WARNING)
            
            self.model_ = Prophet(**model_params)
            self.model_.fit(prophet_data)
            
            logger.info("Prophet model fitted successfully")
            
        except Exception as e:
            logger.error(f"Error in Prophet fitting: {e}")
            raise TimeSeriesValidationError(f"Prophet fitting failed: {str(e)}")
            
        return self
        
    def transform(self, X: pd.DataFrame) -> PipelineResult:
        """Generate forecasts using the fitted Prophet model."""
        check_is_fitted(self, ['model_', 'training_data_'])
        
        start_time = time.time()
        logger = get_logger(__name__)
        
        try:
            # Create future dataframe for forecasting
            future = self.model_.make_future_dataframe(
                periods=self.forecast_steps,
                freq=self._infer_frequency(pd.DataFrame({'value': self.training_data_}))
            )
            
            # Generate forecast
            forecast = self.model_.predict(future)
            self.forecast_ = forecast
            
            # Extract forecast values (only future periods)
            forecast_values = forecast[['ds', 'yhat']].tail(self.forecast_steps)
            forecast_values = forecast_values.set_index('ds')['yhat']
            
            # Extract confidence intervals
            confidence_intervals = forecast[['ds', 'yhat_lower', 'yhat_upper']].tail(self.forecast_steps)
            confidence_intervals = confidence_intervals.set_index('ds')[['yhat_lower', 'yhat_upper']]
            
            # Calculate prediction intervals width
            interval_width = (confidence_intervals['yhat_upper'] - confidence_intervals['yhat_lower']).mean()
            
            # Prepare result data
            result_data = {
                'forecast_method': 'Prophet',
                'forecast_values': forecast_values.to_dict(),
                'confidence_intervals': confidence_intervals.to_dict(),
                'forecast_horizon': self.forecast_steps,
                'confidence_level': self.confidence_level,
                'model_components': {
                    'trend': forecast[['ds', 'trend']].set_index('ds')['trend'].to_dict(),
                    'seasonal': {},  # Will be populated with seasonal components
                },
                'model_parameters': {
                    'growth': self.growth,
                    'seasonality_mode': self.seasonality_mode,
                    'changepoint_prior_scale': self.changepoint_prior_scale,
                    'seasonality_prior_scale': self.seasonality_prior_scale
                },
                'diagnostics': {
                    'mean_interval_width': float(interval_width),
                    'number_of_changepoints': len(self.model_.changepoints),
                    'training_data_points': len(self.training_data_)
                }
            }
            
            # Add seasonal components if available
            for component in ['weekly', 'yearly', 'daily']:
                if component in forecast.columns:
                    result_data['model_components']['seasonal'][component] = \
                        forecast[['ds', component]].set_index('ds')[component].to_dict()
                        
            # Add interpretation
            interpretation = self._generate_interpretation(forecast_values, confidence_intervals)
            result_data['interpretation'] = interpretation
            
            # Add recommendations
            recommendations = self._generate_recommendations()
            result_data['recommendations'] = recommendations
            
            return TimeSeriesAnalysisResult(
                data=result_data,
                execution_time=time.time() - start_time,
                metadata=CompositionMetadata(
                    timestamp=datetime.now(),
                    data_shape=(len(self.training_data_), 1),
                    parameters={
                        'forecast_steps': self.forecast_steps,
                        'growth': self.growth,
                        'seasonality_mode': self.seasonality_mode
                    }
                )
            )
            
        except Exception as e:
            logger.error(f"Error in Prophet forecasting: {e}")
            raise TimeSeriesValidationError(f"Prophet forecasting failed: {str(e)}")
            
    def _generate_interpretation(self, forecast_values: pd.Series, 
                               confidence_intervals: pd.DataFrame) -> str:
        """
        Generate human-readable interpretation of Prophet forecast results.
        
        Parameters:
        -----------
        forecast_values : pd.Series
            Forecasted values
        confidence_intervals : pd.DataFrame
            Confidence intervals for forecasts
            
        Returns:
        --------
        interpretation : str
            Human-readable interpretation
        """
        avg_forecast = forecast_values.mean()
        forecast_trend = 'increasing' if forecast_values.iloc[-1] > forecast_values.iloc[0] else 'decreasing'
        avg_interval_width = (confidence_intervals['yhat_upper'] - confidence_intervals['yhat_lower']).mean()
        
        interpretation = (
            f"Prophet forecast shows {forecast_trend} trend over {self.forecast_steps} periods. "
            f"Average predicted value: {avg_forecast:.2f}. "
            f"Average prediction interval width: {avg_interval_width:.2f}, "
            f"indicating {'high' if avg_interval_width > abs(avg_forecast) * 0.2 else 'moderate'} uncertainty."
        )
        
        return interpretation
        
    def _generate_recommendations(self) -> List[str]:
        """
        Generate recommendations based on the Prophet model.
        
        Returns:
        --------
        recommendations : List[str]
            List of recommendations for the user
        """
        recommendations = []
        
        if len(self.training_data_) < 100:
            recommendations.append("Consider collecting more historical data for improved forecast accuracy")
            
        if self.model_ and len(self.model_.changepoints) > len(self.training_data_) * 0.1:
            recommendations.append("High number of changepoints detected - consider adjusting changepoint_prior_scale")
            
        if self.seasonality_mode == 'additive' and self.training_data_.std() > self.training_data_.mean() * 0.5:
            recommendations.append("High variability detected - consider using multiplicative seasonality")
            
        recommendations.append("Prophet works best with at least one year of data and clear seasonal patterns")
        
        return recommendations


class ExponentialSmoothingForecaster(TimeSeriesTransformer):
    """
    sklearn-compatible transformer for Exponential Smoothing (Holt-Winters) forecasting.
    
    Implements simple, double (Holt's), and triple (Holt-Winters) exponential smoothing
    methods for time series forecasting. Automatically selects the best model
    configuration based on data characteristics.
    
    Parameters:
    -----------
    forecast_steps : int, default=10
        Number of steps to forecast ahead
    confidence_level : float, default=0.95
        Confidence level for prediction intervals
    trend : str or None, default='auto'
        Type of trend: None, 'add', 'mul', or 'auto'
    seasonal : str or None, default='auto'
        Type of seasonality: None, 'add', 'mul', or 'auto'
    seasonal_periods : int, default=None
        Number of periods in a complete seasonal cycle (auto-detected if None)
    damped_trend : bool, default=False
        Whether to use a damped trend
    smoothing_level : float, default=None
        Smoothing parameter for level (alpha). Auto-optimized if None
    smoothing_trend : float, default=None
        Smoothing parameter for trend (beta). Auto-optimized if None
    smoothing_seasonal : float, default=None
        Smoothing parameter for seasonal component (gamma). Auto-optimized if None
    use_boxcox : bool or str, default=False
        Whether to use Box-Cox transformation
    """
    
    def __init__(self, forecast_steps=10, confidence_level=0.95, trend='auto',
                 seasonal='auto', seasonal_periods=None, damped_trend=False,
                 smoothing_level=None, smoothing_trend=None, smoothing_seasonal=None,
                 use_boxcox=False, **kwargs):
        super().__init__(**kwargs)
        self.forecast_steps = forecast_steps
        self.confidence_level = confidence_level
        self.trend = trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self.damped_trend = damped_trend
        self.smoothing_level = smoothing_level
        self.smoothing_trend = smoothing_trend
        self.smoothing_seasonal = smoothing_seasonal
        self.use_boxcox = use_boxcox
        
        self.model_ = None
        self.fitted_model_ = None
        self.training_data_ = None
        self.selected_trend_ = None
        self.selected_seasonal_ = None
        self.selected_seasonal_periods_ = None
        
    def _detect_seasonality_periods(self, data: pd.Series) -> int:
        """
        Detect the number of periods in a seasonal cycle.
        
        Parameters:
        -----------
        data : pd.Series
            Time series data with datetime index
            
        Returns:
        --------
        periods : int
            Number of periods in seasonal cycle
        """
        # Try to infer from frequency
        freq = self._infer_frequency(pd.DataFrame({'value': data}))
        
        if freq:
            freq_map = {
                'H': 24,    # Hourly -> daily seasonality
                'D': 7,     # Daily -> weekly seasonality  
                'W': 52,    # Weekly -> yearly seasonality
                'M': 12,    # Monthly -> yearly seasonality
                'Q': 4,     # Quarterly -> yearly seasonality
                'T': 60,    # Minutes -> hourly seasonality
                'S': 60     # Seconds -> minute seasonality
            }
            
            for f, periods in freq_map.items():
                if f in str(freq).upper():
                    return periods
                    
        # Fallback: use autocorrelation to detect seasonality
        try:
            max_lag = min(len(data) // 3, 50)
            autocorr = [data.autocorr(lag=i) for i in range(1, max_lag + 1)]
            
            # Find the lag with maximum autocorrelation
            if autocorr:
                best_lag = np.argmax(np.abs(autocorr)) + 1
                if abs(autocorr[best_lag - 1]) > 0.3:  # Significant autocorrelation
                    return best_lag
        except Exception as e:
            logger.debug(f"Could not detect seasonality via autocorrelation: {e}")
            
        # Default fallback
        return 12
        
    def _select_model_components(self, data: pd.Series) -> Tuple[str, str, int]:
        """
        Automatically select trend and seasonality components.
        
        Parameters:
        -----------
        data : pd.Series
            Time series data
            
        Returns:
        --------
        trend : str
            Selected trend component
        seasonal : str
            Selected seasonal component
        seasonal_periods : int
            Selected seasonal periods
        """
        if self.trend != 'auto' and self.seasonal != 'auto':
            return self.trend, self.seasonal, self.seasonal_periods or self._detect_seasonality_periods(data)
            
        # Detect trend
        if self.trend == 'auto':
            # Simple trend detection using linear regression
            x = np.arange(len(data))
            slope, _, r_value, _, _ = stats.linregress(x, data.values)
            
            if abs(r_value) > 0.7 and abs(slope) > data.std() * 0.01:
                # Check if multiplicative trend is better
                if data.min() > 0 and (data.max() / data.min()) > 2:
                    selected_trend = 'mul'
                else:
                    selected_trend = 'add'
            else:
                selected_trend = None
        else:
            selected_trend = self.trend
            
        # Detect seasonality
        if self.seasonal == 'auto':
            seasonal_periods = self.seasonal_periods or self._detect_seasonality_periods(data)
            
            if len(data) >= 2 * seasonal_periods:
                try:
                    # Try seasonal decomposition to check for seasonality
                    decomposition = seasonal_decompose(data, model='additive', period=seasonal_periods)
                    seasonal_strength = np.var(decomposition.seasonal) / np.var(data)
                    
                    if seasonal_strength > 0.1:
                        # Check if multiplicative seasonality is better
                        if data.min() > 0:
                            # Compare additive vs multiplicative seasonal patterns
                            try:
                                decomp_mul = seasonal_decompose(data, model='multiplicative', period=seasonal_periods)
                                seasonal_strength_mul = np.var(decomp_mul.seasonal) / np.var(data)
                                
                                if seasonal_strength_mul > seasonal_strength * 1.1:
                                    selected_seasonal = 'mul'
                                else:
                                    selected_seasonal = 'add'
                            except:
                                selected_seasonal = 'add'
                        else:
                            selected_seasonal = 'add'
                    else:
                        selected_seasonal = None
                        seasonal_periods = None
                except Exception as e:
                    logger.debug(f"Seasonality detection failed: {e}")
                    selected_seasonal = None
                    seasonal_periods = None
            else:
                selected_seasonal = None
                seasonal_periods = None
        else:
            selected_seasonal = self.seasonal
            seasonal_periods = self.seasonal_periods or self._detect_seasonality_periods(data)
            
        return selected_trend, selected_seasonal, seasonal_periods
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fit the Exponential Smoothing forecasting model."""
        if self.validate_input:
            X, y = self._validate_time_series(X, y)
            
        logger = get_logger(__name__)
        
        try:
            # Import here to handle potential ImportError gracefully
            from scipy import stats
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
            
            # Extract univariate time series
            if isinstance(X, pd.DataFrame):
                if X.shape[1] == 1:
                    data = X.iloc[:, 0]
                else:
                    raise ValueError("Exponential Smoothing requires univariate time series data")
            elif y is not None:
                data = y
            else:
                raise ValueError("No target variable provided for Exponential Smoothing")
                
            self.training_data_ = data
            
            # Select model components automatically
            self.selected_trend_, self.selected_seasonal_, self.selected_seasonal_periods_ = \
                self._select_model_components(data)
                
            logger.info(f"Selected components - Trend: {self.selected_trend_}, "
                       f"Seasonal: {self.selected_seasonal_}, Periods: {self.selected_seasonal_periods_}")
            
            # Initialize Exponential Smoothing model
            model_params = {
                'trend': self.selected_trend_,
                'seasonal': self.selected_seasonal_,
                'seasonal_periods': self.selected_seasonal_periods_,
                'damped_trend': self.damped_trend,
                'use_boxcox': self.use_boxcox
            }
            
            # Remove None values
            model_params = {k: v for k, v in model_params.items() if v is not None}
            
            self.model_ = ExponentialSmoothing(data, **model_params)
            
            # Fit model with optional smoothing parameters
            fit_params = {}
            if self.smoothing_level is not None:
                fit_params['smoothing_level'] = self.smoothing_level
            if self.smoothing_trend is not None:
                fit_params['smoothing_trend'] = self.smoothing_trend
            if self.smoothing_seasonal is not None:
                fit_params['smoothing_seasonal'] = self.smoothing_seasonal
                
            self.fitted_model_ = self.model_.fit(**fit_params)
            
            logger.info("Exponential Smoothing model fitted successfully")
            
        except Exception as e:
            logger.error(f"Error in Exponential Smoothing fitting: {e}")
            raise TimeSeriesValidationError(f"Exponential Smoothing fitting failed: {str(e)}")
            
        return self
        
    def transform(self, X: pd.DataFrame) -> PipelineResult:
        """Generate forecasts using the fitted Exponential Smoothing model."""
        check_is_fitted(self, ['fitted_model_', 'training_data_'])
        
        start_time = time.time()
        logger = get_logger(__name__)
        
        try:
            # Generate forecast
            forecast_result = self.fitted_model_.forecast(steps=self.forecast_steps)
            
            # Generate prediction intervals if possible
            try:
                # Try to get prediction intervals
                alpha = 1 - self.confidence_level
                forecast_summary = self.fitted_model_.get_forecast(steps=self.forecast_steps)
                confidence_intervals = forecast_summary.conf_int(alpha=alpha)
                
                # Convert to DataFrame with proper column names
                confidence_intervals.columns = ['lower', 'upper']
                
            except Exception as e:
                logger.warning(f"Could not generate prediction intervals: {e}")
                # Create dummy intervals based on historical residuals
                residuals = self.fitted_model_.resid
                std_residual = np.std(residuals)
                z_score = stats.norm.ppf(1 - alpha/2) if 'stats' in locals() else 1.96
                
                confidence_intervals = pd.DataFrame({
                    'lower': forecast_result - z_score * std_residual,
                    'upper': forecast_result + z_score * std_residual
                }, index=forecast_result.index)
                
            # Calculate prediction intervals width
            interval_width = (confidence_intervals['upper'] - confidence_intervals['lower']).mean()
            
            # Prepare result data
            result_data = {
                'forecast_method': 'Exponential Smoothing (Holt-Winters)',
                'forecast_values': forecast_result.to_dict(),
                'confidence_intervals': confidence_intervals.to_dict(),
                'forecast_horizon': self.forecast_steps,
                'confidence_level': self.confidence_level,
                'model_components': {
                    'trend': self.selected_trend_,
                    'seasonal': self.selected_seasonal_,
                    'seasonal_periods': self.selected_seasonal_periods_,
                    'damped_trend': self.damped_trend
                },
                'model_parameters': {
                    'smoothing_level': float(self.fitted_model_.params['smoothing_level']),
                    'aic': float(self.fitted_model_.aic),
                    'bic': float(self.fitted_model_.bic),
                    'sse': float(self.fitted_model_.sse)
                },
                'diagnostics': {
                    'mean_interval_width': float(interval_width),
                    'training_data_points': len(self.training_data_),
                    'model_fit_quality': 'good' if self.fitted_model_.aic < len(self.training_data_) * 2 else 'moderate'
                }
            }
            
            # Add trend and seasonal smoothing parameters if available
            if self.selected_trend_ and 'smoothing_trend' in self.fitted_model_.params:
                result_data['model_parameters']['smoothing_trend'] = float(self.fitted_model_.params['smoothing_trend'])
            if self.selected_seasonal_ and 'smoothing_seasonal' in self.fitted_model_.params:
                result_data['model_parameters']['smoothing_seasonal'] = float(self.fitted_model_.params['smoothing_seasonal'])
                
            # Add interpretation
            interpretation = self._generate_interpretation(forecast_result, confidence_intervals)
            result_data['interpretation'] = interpretation
            
            # Add recommendations
            recommendations = self._generate_recommendations()
            result_data['recommendations'] = recommendations
            
            return TimeSeriesAnalysisResult(
                data=result_data,
                execution_time=time.time() - start_time,
                metadata=CompositionMetadata(
                    timestamp=datetime.now(),
                    data_shape=(len(self.training_data_), 1),
                    parameters={
                        'forecast_steps': self.forecast_steps,
                        'trend': self.selected_trend_,
                        'seasonal': self.selected_seasonal_,
                        'seasonal_periods': self.selected_seasonal_periods_
                    }
                )
            )
            
        except Exception as e:
            logger.error(f"Error in Exponential Smoothing forecasting: {e}")
            raise TimeSeriesValidationError(f"Exponential Smoothing forecasting failed: {str(e)}")
            
    def _generate_interpretation(self, forecast_values: pd.Series, 
                               confidence_intervals: pd.DataFrame) -> str:
        """
        Generate human-readable interpretation of Exponential Smoothing forecast results.
        
        Parameters:
        -----------
        forecast_values : pd.Series
            Forecasted values
        confidence_intervals : pd.DataFrame
            Confidence intervals for forecasts
            
        Returns:
        --------
        interpretation : str
            Human-readable interpretation
        """
        avg_forecast = forecast_values.mean()
        forecast_trend = 'increasing' if forecast_values.iloc[-1] > forecast_values.iloc[0] else 'decreasing'
        avg_interval_width = (confidence_intervals['upper'] - confidence_intervals['lower']).mean()
        
        model_type = "Simple"
        if self.selected_trend_ and self.selected_seasonal_:
            model_type = "Triple (Holt-Winters)"
        elif self.selected_trend_:
            model_type = "Double (Holt's)"
            
        interpretation = (
            f"{model_type} Exponential Smoothing forecast shows {forecast_trend} trend over {self.forecast_steps} periods. "
            f"Average predicted value: {avg_forecast:.2f}. "
            f"Prediction uncertainty: {'high' if avg_interval_width > abs(avg_forecast) * 0.2 else 'moderate'} "
            f"(interval width: {avg_interval_width:.2f})."
        )
        
        if self.selected_seasonal_:
            interpretation += f" Seasonal component detected with period {self.selected_seasonal_periods_}."
            
        return interpretation
        
    def _generate_recommendations(self) -> List[str]:
        """
        Generate recommendations based on the Exponential Smoothing model.
        
        Returns:
        --------
        recommendations : List[str]
            List of recommendations for the user
        """
        recommendations = []
        
        if len(self.training_data_) < 30:
            recommendations.append("Consider collecting more historical data for improved forecast accuracy")
            
        if self.fitted_model_.aic > len(self.training_data_) * 3:
            recommendations.append("High AIC value suggests model may be overfitting - consider simpler configuration")
            
        if self.selected_seasonal_ is None and len(self.training_data_) > 24:
            recommendations.append("No seasonality detected - verify if seasonal patterns exist in your data")
            
        if self.selected_trend_ is None:
            recommendations.append("No trend detected - consider if the data has underlying growth patterns")
            
        recommendations.append("Exponential smoothing works well for data with clear trends and seasonal patterns")
        
        return recommendations


class EnsembleForecaster(TimeSeriesTransformer):
    """
    sklearn-compatible transformer for ensemble forecasting methods.
    
    Combines multiple forecasting models (Prophet, Exponential Smoothing, ARIMA)
    to create robust ensemble predictions. Uses weighted averaging or more
    sophisticated combination methods.
    
    Parameters:
    -----------
    forecast_steps : int, default=10
        Number of steps to forecast ahead
    confidence_level : float, default=0.95
        Confidence level for prediction intervals
    methods : list, default=['exponential_smoothing', 'arima']
        List of forecasting methods to combine
    weights : dict, optional
        Weights for each method (auto-computed if None)
    combination_method : str, default='weighted_average'
        Method for combining forecasts: 'weighted_average', 'median', 'best_performer'
    validation_split : float, default=0.2
        Fraction of data to use for model validation and weight optimization
    """
    
    def __init__(self, forecast_steps=10, confidence_level=0.95, 
                 methods=None, weights=None, combination_method='weighted_average',
                 validation_split=0.2, **kwargs):
        super().__init__(**kwargs)
        self.forecast_steps = forecast_steps
        self.confidence_level = confidence_level
        self.methods = methods or ['exponential_smoothing', 'arima']
        self.weights = weights or {}
        self.combination_method = combination_method
        self.validation_split = validation_split
        
        # Model instances
        self.models_ = {}
        self.fitted_models_ = {}
        self.model_performance_ = {}
        self.computed_weights_ = {}
        self.training_data_ = None
        self.validation_data_ = None
        
    def _split_data(self, data: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """
        Split data into training and validation sets.
        
        Parameters:
        -----------
        data : pd.Series
            Full time series data
            
        Returns:
        --------
        train_data : pd.Series
            Training data
        val_data : pd.Series
            Validation data
        """
        split_point = int(len(data) * (1 - self.validation_split))
        train_data = data.iloc[:split_point]
        val_data = data.iloc[split_point:]
        return train_data, val_data
        
    def _initialize_models(self) -> Dict[str, TimeSeriesTransformer]:
        """
        Initialize the forecasting models for the ensemble.
        
        Returns:
        --------
        models : dict
            Dictionary of initialized model instances
        """
        models = {}
        
        for method in self.methods:
            if method == 'prophet':
                try:
                    models[method] = ProphetForecaster(
                        forecast_steps=len(self.validation_data_) if hasattr(self, 'validation_data_') else self.forecast_steps,
                        confidence_level=self.confidence_level
                    )
                except Exception as e:
                    logger.warning(f"Could not initialize Prophet: {e}")
                    continue
                    
            elif method == 'exponential_smoothing':
                models[method] = ExponentialSmoothingForecaster(
                    forecast_steps=len(self.validation_data_) if hasattr(self, 'validation_data_') else self.forecast_steps,
                    confidence_level=self.confidence_level
                )
                
            elif method == 'arima':
                models[method] = ARIMAForecastTransformer(
                    forecast_steps=len(self.validation_data_) if hasattr(self, 'validation_data_') else self.forecast_steps,
                    confidence_level=self.confidence_level
                )
                
            elif method == 'auto_arima':
                models[method] = AutoARIMATransformer(
                    forecast_steps=len(self.validation_data_) if hasattr(self, 'validation_data_') else self.forecast_steps,
                    stepwise=True  # Faster for ensemble
                )
                
        return models
        
    def _evaluate_model_performance(self, model: TimeSeriesTransformer, 
                                   method: str, train_data: pd.DataFrame, 
                                   val_data: pd.Series) -> Dict[str, float]:
        """
        Evaluate model performance on validation data.
        
        Parameters:
        -----------
        model : TimeSeriesTransformer
            Fitted model to evaluate
        method : str
            Method name
        train_data : pd.DataFrame
            Training data
        val_data : pd.Series
            Validation data for evaluation
            
        Returns:
        --------
        performance : dict
            Performance metrics
        """
        try:
            # Fit model on training data
            model.fit(train_data)
            
            # Generate forecasts for validation period
            result = model.transform(train_data)
            
            # Extract forecast values
            if hasattr(result, 'data') and 'forecast_values' in result.data:
                forecast_dict = result.data['forecast_values']
                if isinstance(forecast_dict, dict):
                    forecast_values = pd.Series(forecast_dict)
                else:
                    forecast_values = forecast_dict
            else:
                raise ValueError(f"Could not extract forecasts from {method} model")
                
            # Align forecast and actual values
            min_length = min(len(forecast_values), len(val_data))
            forecast_aligned = forecast_values.iloc[:min_length]
            actual_aligned = val_data.iloc[:min_length]
            
            # Calculate metrics
            mae = np.mean(np.abs(forecast_aligned - actual_aligned))
            mse = np.mean((forecast_aligned - actual_aligned) ** 2)
            rmse = np.sqrt(mse)
            
            # MAPE (handle zero values)
            mape_values = np.abs((actual_aligned - forecast_aligned) / actual_aligned)
            mape_values = mape_values[np.isfinite(mape_values)]  # Remove inf/nan
            mape = np.mean(mape_values) * 100 if len(mape_values) > 0 else float('inf')
            
            return {
                'mae': mae,
                'mse': mse,
                'rmse': rmse,
                'mape': mape,
                'score': 1.0 / (1.0 + rmse)  # Higher is better
            }
            
        except Exception as e:
            logger.warning(f"Could not evaluate {method}: {e}")
            return {
                'mae': float('inf'),
                'mse': float('inf'), 
                'rmse': float('inf'),
                'mape': float('inf'),
                'score': 0.0
            }
            
    def _compute_weights(self) -> Dict[str, float]:
        """
        Compute weights for ensemble combination based on validation performance.
        
        Returns:
        --------
        weights : dict
            Computed weights for each method
        """
        if self.weights:
            # Use provided weights, normalize to sum to 1
            total_weight = sum(self.weights.values())
            return {k: v / total_weight for k, v in self.weights.items()}
            
        # Compute weights based on performance scores
        if not self.model_performance_:
            # Equal weights as fallback
            num_models = len(self.fitted_models_)
            return {method: 1.0 / num_models for method in self.fitted_models_}
            
        # Inverse error weighting (better performance = higher weight)
        scores = {method: perf['score'] for method, perf in self.model_performance_.items()}
        total_score = sum(scores.values())
        
        if total_score == 0:
            # Equal weights fallback
            num_models = len(scores)
            return {method: 1.0 / num_models for method in scores}
            
        return {method: score / total_score for method, score in scores.items()}
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fit the ensemble forecasting models."""
        if self.validate_input:
            X, y = self._validate_time_series(X, y)
            
        logger = get_logger(__name__)
        
        try:
            # Extract univariate time series
            if isinstance(X, pd.DataFrame):
                if X.shape[1] == 1:
                    data = X.iloc[:, 0]
                else:
                    raise ValueError("Ensemble forecasting requires univariate time series data")
            elif y is not None:
                data = y
            else:
                raise ValueError("No target variable provided for ensemble forecasting")
                
            self.training_data_ = data
            
            # Split data for validation if we have enough data points
            if len(data) > 50 and self.validation_split > 0:
                train_data, val_data = self._split_data(data)
                self.validation_data_ = val_data
                logger.info(f"Split data: {len(train_data)} training, {len(val_data)} validation")
            else:
                train_data = data
                self.validation_data_ = None
                logger.info("Using full dataset for training (insufficient data for validation)")
                
            # Initialize models
            self.models_ = self._initialize_models()
            logger.info(f"Initialized {len(self.models_)} models: {list(self.models_.keys())}")
            
            # Fit models and evaluate performance
            train_df = pd.DataFrame({'value': train_data})
            
            for method, model in self.models_.items():
                try:
                    logger.info(f"Fitting {method} model...")
                    
                    if self.validation_data_ is not None:
                        # Evaluate on validation set
                        performance = self._evaluate_model_performance(
                            model, method, train_df, self.validation_data_
                        )
                        self.model_performance_[method] = performance
                        logger.info(f"{method} validation RMSE: {performance['rmse']:.4f}")
                    else:
                        # Fit on full data
                        model.fit(train_df)
                        
                    self.fitted_models_[method] = model
                    
                except Exception as e:
                    logger.warning(f"Failed to fit {method}: {e}")
                    continue
                    
            if not self.fitted_models_:
                raise ValueError("No models could be fitted successfully")
                
            # Now fit models on full training data for final predictions
            full_train_df = pd.DataFrame({'value': self.training_data_})
            for method in self.fitted_models_:
                try:
                    # Update forecast steps for final fitting
                    self.fitted_models_[method].forecast_steps = self.forecast_steps
                    self.fitted_models_[method].fit(full_train_df)
                except Exception as e:
                    logger.warning(f"Failed to refit {method} on full data: {e}")
                    
            # Compute ensemble weights
            self.computed_weights_ = self._compute_weights()
            logger.info(f"Computed weights: {self.computed_weights_}")
            
        except Exception as e:
            logger.error(f"Error in ensemble forecasting fit: {e}")
            raise TimeSeriesValidationError(f"Ensemble forecasting fit failed: {str(e)}")
            
        return self
        
    def transform(self, X: pd.DataFrame) -> PipelineResult:
        """Generate ensemble forecasts using fitted models."""
        check_is_fitted(self, ['fitted_models_', 'computed_weights_', 'training_data_'])
        
        start_time = time.time()
        logger = get_logger(__name__)
        
        try:
            # Generate forecasts from each model
            individual_forecasts = {}
            individual_intervals = {}
            
            for method, model in self.fitted_models_.items():
                try:
                    result = model.transform(X)
                    
                    # Extract forecast values
                    if hasattr(result, 'data') and 'forecast_values' in result.data:
                        forecast_dict = result.data['forecast_values']
                        if isinstance(forecast_dict, dict):
                            forecast_values = pd.Series(forecast_dict)
                        else:
                            forecast_values = forecast_dict
                            
                        individual_forecasts[method] = forecast_values
                        
                        # Extract confidence intervals if available
                        if 'confidence_intervals' in result.data:
                            intervals_dict = result.data['confidence_intervals']
                            if isinstance(intervals_dict, dict):
                                intervals = pd.DataFrame(intervals_dict)
                            else:
                                intervals = intervals_dict
                            individual_intervals[method] = intervals
                            
                except Exception as e:
                    logger.warning(f"Failed to get forecast from {method}: {e}")
                    continue
                    
            if not individual_forecasts:
                raise ValueError("No individual forecasts could be generated")
                
            # Combine forecasts based on combination method
            if self.combination_method == 'weighted_average':
                ensemble_forecast = self._weighted_average_combination(individual_forecasts)
            elif self.combination_method == 'median':
                ensemble_forecast = self._median_combination(individual_forecasts)
            elif self.combination_method == 'best_performer':
                ensemble_forecast = self._best_performer_combination(individual_forecasts)
            else:
                raise ValueError(f"Unknown combination method: {self.combination_method}")
                
            # Combine confidence intervals
            ensemble_intervals = self._combine_confidence_intervals(individual_intervals)
            
            # Calculate ensemble statistics
            forecast_std = np.std([f.values for f in individual_forecasts.values()], axis=0)
            forecast_agreement = 1.0 - (np.mean(forecast_std) / np.mean(np.abs(ensemble_forecast)))
            
            # Prepare result data
            result_data = {
                'forecast_method': 'Ensemble',
                'forecast_values': ensemble_forecast.to_dict(),
                'confidence_intervals': ensemble_intervals.to_dict() if ensemble_intervals is not None else None,
                'forecast_horizon': self.forecast_steps,
                'confidence_level': self.confidence_level,
                'ensemble_details': {
                    'methods': list(self.fitted_models_.keys()),
                    'weights': self.computed_weights_,
                    'combination_method': self.combination_method,
                    'individual_forecasts': {method: f.to_dict() for method, f in individual_forecasts.items()}
                },
                'ensemble_statistics': {
                    'forecast_agreement': float(forecast_agreement),
                    'method_count': len(individual_forecasts),
                    'forecast_std_dev': forecast_std.tolist() if hasattr(forecast_std, 'tolist') else float(forecast_std)
                },
                'model_performance': self.model_performance_
            }
            
            # Add interpretation
            interpretation = self._generate_interpretation(ensemble_forecast, individual_forecasts)
            result_data['interpretation'] = interpretation
            
            # Add recommendations
            recommendations = self._generate_recommendations()
            result_data['recommendations'] = recommendations
            
            return TimeSeriesAnalysisResult(
                data=result_data,
                execution_time=time.time() - start_time,
                metadata=CompositionMetadata(
                    timestamp=datetime.now(),
                    data_shape=(len(self.training_data_), 1),
                    parameters={
                        'forecast_steps': self.forecast_steps,
                        'methods': self.methods,
                        'combination_method': self.combination_method
                    }
                )
            )
            
        except Exception as e:
            logger.error(f"Error in ensemble forecasting transform: {e}")
            raise TimeSeriesValidationError(f"Ensemble forecasting transform failed: {str(e)}")
            
    def _weighted_average_combination(self, forecasts: Dict[str, pd.Series]) -> pd.Series:
        """
        Combine forecasts using weighted average.
        
        Parameters:
        -----------
        forecasts : dict
            Dictionary of individual forecasts
            
        Returns:
        --------
        combined_forecast : pd.Series
            Weighted average forecast
        """
        # Align all forecasts to same length
        min_length = min(len(f) for f in forecasts.values())
        aligned_forecasts = {method: f.iloc[:min_length] for method, f in forecasts.items()}
        
        # Calculate weighted average
        combined = None
        for method, forecast in aligned_forecasts.items():
            weight = self.computed_weights_.get(method, 0.0)
            if combined is None:
                combined = forecast * weight
            else:
                combined += forecast * weight
                
        return combined
        
    def _median_combination(self, forecasts: Dict[str, pd.Series]) -> pd.Series:
        """
        Combine forecasts using median.
        
        Parameters:
        -----------
        forecasts : dict
            Dictionary of individual forecasts
            
        Returns:
        --------
        combined_forecast : pd.Series
            Median forecast
        """
        # Align all forecasts to same length
        min_length = min(len(f) for f in forecasts.values())
        forecast_array = np.array([f.iloc[:min_length].values for f in forecasts.values()])
        
        # Calculate median across methods
        median_values = np.median(forecast_array, axis=0)
        
        # Use index from first forecast
        first_forecast = list(forecasts.values())[0]
        return pd.Series(median_values, index=first_forecast.index[:min_length])
        
    def _best_performer_combination(self, forecasts: Dict[str, pd.Series]) -> pd.Series:
        """
        Use forecast from best-performing model.
        
        Parameters:
        -----------
        forecasts : dict
            Dictionary of individual forecasts
            
        Returns:
        --------
        combined_forecast : pd.Series
            Best performer's forecast
        """
        if not self.model_performance_:
            # Fallback to first available forecast
            return list(forecasts.values())[0]
            
        # Find best performing method
        best_method = max(self.model_performance_.keys(), 
                         key=lambda x: self.model_performance_[x]['score'])
        
        return forecasts.get(best_method, list(forecasts.values())[0])
        
    def _combine_confidence_intervals(self, intervals: Dict[str, pd.DataFrame]) -> Optional[pd.DataFrame]:
        """
        Combine confidence intervals from multiple models.
        
        Parameters:
        -----------
        intervals : dict
            Dictionary of individual confidence intervals
            
        Returns:
        --------
        combined_intervals : pd.DataFrame or None
            Combined confidence intervals
        """
        if not intervals:
            return None
            
        try:
            # Align all intervals
            min_length = min(len(ci) for ci in intervals.values())
            aligned_intervals = {method: ci.iloc[:min_length] for method, ci in intervals.items()}
            
            # Use the widest intervals (most conservative approach)
            all_lowers = [ci['lower'] if 'lower' in ci.columns else ci.iloc[:, 0] 
                         for ci in aligned_intervals.values()]
            all_uppers = [ci['upper'] if 'upper' in ci.columns else ci.iloc[:, 1] 
                         for ci in aligned_intervals.values()]
            
            combined_lower = np.min(all_lowers, axis=0)
            combined_upper = np.max(all_uppers, axis=0)
            
            # Use index from first interval
            first_interval = list(intervals.values())[0]
            return pd.DataFrame({
                'lower': combined_lower,
                'upper': combined_upper
            }, index=first_interval.index[:min_length])
            
        except Exception as e:
            logger.warning(f"Could not combine confidence intervals: {e}")
            return None
            
    def _generate_interpretation(self, ensemble_forecast: pd.Series, 
                               individual_forecasts: Dict[str, pd.Series]) -> str:
        """
        Generate human-readable interpretation of ensemble forecast results.
        
        Parameters:
        -----------
        ensemble_forecast : pd.Series
            Combined ensemble forecast
        individual_forecasts : dict
            Dictionary of individual model forecasts
            
        Returns:
        --------
        interpretation : str
            Human-readable interpretation
        """
        avg_forecast = ensemble_forecast.mean()
        forecast_trend = 'increasing' if ensemble_forecast.iloc[-1] > ensemble_forecast.iloc[0] else 'decreasing'
        
        # Calculate forecast agreement
        forecast_std = np.std([f.values for f in individual_forecasts.values()], axis=0)
        avg_std = np.mean(forecast_std)
        agreement = 'high' if avg_std < abs(avg_forecast) * 0.1 else ('moderate' if avg_std < abs(avg_forecast) * 0.2 else 'low')
        
        interpretation = (
            f"Ensemble of {len(individual_forecasts)} models predicts {forecast_trend} trend over {self.forecast_steps} periods. "
            f"Average predicted value: {avg_forecast:.2f}. "
            f"Model agreement: {agreement} (std dev: {avg_std:.2f}). "
            f"Best performing method: {max(self.model_performance_.keys(), key=lambda x: self.model_performance_[x]['score']) if self.model_performance_ else 'N/A'}."
        )
        
        return interpretation
        
    def _generate_recommendations(self) -> List[str]:
        """
        Generate recommendations based on the ensemble model performance.
        
        Returns:
        --------
        recommendations : List[str]
            List of recommendations for the user
        """
        recommendations = []
        
        if len(self.fitted_models_) < 3:
            recommendations.append("Consider adding more forecasting methods to the ensemble for better robustness")
            
        if self.model_performance_:
            # Check performance spread
            scores = [perf['score'] for perf in self.model_performance_.values()]
            score_std = np.std(scores)
            if score_std > 0.3:
                recommendations.append("High performance variation between models - consider investigating data characteristics")
                
            # Check if one model dominates
            max_weight = max(self.computed_weights_.values())
            if max_weight > 0.8:
                best_method = max(self.computed_weights_.keys(), key=lambda x: self.computed_weights_[x])
                recommendations.append(f"Single model ({best_method}) dominates ensemble - consider using it individually")
                
        if len(self.training_data_) < 100:
            recommendations.append("More historical data would improve ensemble forecast reliability")
            
        recommendations.append("Ensemble forecasting combines multiple methods to reduce individual model biases")
        
        return recommendations


class ForecastEvaluator(BaseEstimator, TransformerMixin):
    """
    sklearn-compatible transformer for evaluating forecast accuracy.
    
    Provides comprehensive evaluation metrics for comparing forecasted values
    against actual observations, including MAE, MAPE, RMSE, and other
    time series specific metrics.
    
    Parameters:
    -----------
    metrics : list, default=['mae', 'mape', 'rmse', 'mase']
        List of metrics to compute
    seasonal_period : int, optional
        Seasonal period for MASE calculation
    """
    
    def __init__(self, metrics=None, seasonal_period=None):
        self.metrics = metrics or ['mae', 'mape', 'rmse', 'mase']
        self.seasonal_period = seasonal_period
        
    def fit(self, X, y=None):
        """ForecastEvaluator doesn't require fitting."""
        return self
        
    def transform(self, X: pd.DataFrame) -> PipelineResult:
        """This method is not used - use evaluate_forecast instead."""
        raise NotImplementedError("Use evaluate_forecast method instead")
        
    def evaluate_forecast(self, actual: Union[pd.Series, np.ndarray], 
                         predicted: Union[pd.Series, np.ndarray],
                         historical_data: Optional[Union[pd.Series, np.ndarray]] = None) -> Dict[str, float]:
        """
        Evaluate forecast accuracy using multiple metrics.
        
        Parameters:
        -----------
        actual : pd.Series or np.ndarray
            Actual observed values
        predicted : pd.Series or np.ndarray
            Predicted/forecasted values
        historical_data : pd.Series or np.ndarray, optional
            Historical data for MASE calculation
            
        Returns:
        --------
        metrics : dict
            Dictionary of evaluation metrics
        """
        # Convert to numpy arrays for calculation
        actual_array = actual.values if hasattr(actual, 'values') else np.array(actual)
        predicted_array = predicted.values if hasattr(predicted, 'values') else np.array(predicted)
        
        # Ensure same length
        min_length = min(len(actual_array), len(predicted_array))
        actual_array = actual_array[:min_length]
        predicted_array = predicted_array[:min_length]
        
        if len(actual_array) == 0:
            raise ValueError("No data points to evaluate")
            
        results = {}
        
        # Calculate requested metrics
        for metric in self.metrics:
            if metric == 'mae':
                results['mae'] = self._calculate_mae(actual_array, predicted_array)
            elif metric == 'mape':
                results['mape'] = self._calculate_mape(actual_array, predicted_array)
            elif metric == 'rmse':
                results['rmse'] = self._calculate_rmse(actual_array, predicted_array)
            elif metric == 'mase':
                results['mase'] = self._calculate_mase(actual_array, predicted_array, historical_data)
            elif metric == 'smape':
                results['smape'] = self._calculate_smape(actual_array, predicted_array)
            elif metric == 'mse':
                results['mse'] = self._calculate_mse(actual_array, predicted_array)
            elif metric == 'r2':
                results['r2'] = self._calculate_r2(actual_array, predicted_array)
            elif metric == 'directional_accuracy':
                results['directional_accuracy'] = self._calculate_directional_accuracy(actual_array, predicted_array)
                
        # Add additional summary statistics
        results['forecast_bias'] = np.mean(predicted_array - actual_array)
        results['mean_actual'] = np.mean(actual_array)
        results['mean_predicted'] = np.mean(predicted_array)
        results['std_actual'] = np.std(actual_array)
        results['std_predicted'] = np.std(predicted_array)
        results['n_observations'] = len(actual_array)
        
        return results
        
    def _calculate_mae(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        """
        Calculate Mean Absolute Error.
        
        Parameters:
        -----------
        actual : np.ndarray
            Actual values
        predicted : np.ndarray
            Predicted values
            
        Returns:
        --------
        mae : float
            Mean Absolute Error
        """
        return float(np.mean(np.abs(actual - predicted)))
        
    def _calculate_mape(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        """
        Calculate Mean Absolute Percentage Error.
        
        Parameters:
        -----------
        actual : np.ndarray
            Actual values
        predicted : np.ndarray
            Predicted values
            
        Returns:
        --------
        mape : float
            Mean Absolute Percentage Error (as percentage)
        """
        # Handle zero values in actual
        mask = actual != 0
        if not np.any(mask):
            return float('inf')  # All actual values are zero
            
        mape_values = np.abs((actual[mask] - predicted[mask]) / actual[mask]) * 100
        return float(np.mean(mape_values))
        
    def _calculate_rmse(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        """
        Calculate Root Mean Square Error.
        
        Parameters:
        -----------
        actual : np.ndarray
            Actual values
        predicted : np.ndarray
            Predicted values
            
        Returns:
        --------
        rmse : float
            Root Mean Square Error
        """
        return float(np.sqrt(np.mean((actual - predicted) ** 2)))
        
    def _calculate_mse(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        """
        Calculate Mean Square Error.
        
        Parameters:
        -----------
        actual : np.ndarray
            Actual values
        predicted : np.ndarray
            Predicted values
            
        Returns:
        --------
        mse : float
            Mean Square Error
        """
        return float(np.mean((actual - predicted) ** 2))
        
    def _calculate_mase(self, actual: np.ndarray, predicted: np.ndarray, 
                       historical_data: Optional[np.ndarray] = None) -> float:
        """
        Calculate Mean Absolute Scaled Error.
        
        Parameters:
        -----------
        actual : np.ndarray
            Actual values
        predicted : np.ndarray
            Predicted values
        historical_data : np.ndarray, optional
            Historical data for scaling factor calculation
            
        Returns:
        --------
        mase : float
            Mean Absolute Scaled Error
        """
        if historical_data is None:
            # Cannot calculate MASE without historical data
            return float('nan')
            
        historical_array = historical_data.values if hasattr(historical_data, 'values') else np.array(historical_data)
        
        # Calculate seasonal naive forecast errors
        seasonal_period = self.seasonal_period or 1
        
        if len(historical_array) <= seasonal_period:
            # Use simple naive forecast (lag-1) if insufficient data for seasonal naive
            naive_errors = np.abs(np.diff(historical_array))
        else:
            # Use seasonal naive forecast
            naive_forecast = historical_array[:-seasonal_period]
            naive_actual = historical_array[seasonal_period:]
            naive_errors = np.abs(naive_actual - naive_forecast)
            
        if len(naive_errors) == 0 or np.mean(naive_errors) == 0:
            return float('inf')
            
        mae = np.mean(np.abs(actual - predicted))
        mean_naive_error = np.mean(naive_errors)
        
        return float(mae / mean_naive_error)
        
    def _calculate_smape(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        """
        Calculate Symmetric Mean Absolute Percentage Error.
        
        Parameters:
        -----------
        actual : np.ndarray
            Actual values
        predicted : np.ndarray
            Predicted values
            
        Returns:
        --------
        smape : float
            Symmetric Mean Absolute Percentage Error (as percentage)
        """
        denominator = (np.abs(actual) + np.abs(predicted)) / 2.0
        
        # Handle zero denominators
        mask = denominator != 0
        if not np.any(mask):
            return 0.0  # Perfect forecast when both actual and predicted are zero
            
        smape_values = np.abs(actual[mask] - predicted[mask]) / denominator[mask] * 100
        return float(np.mean(smape_values))
        
    def _calculate_r2(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        """
        Calculate R-squared (coefficient of determination).
        
        Parameters:
        -----------
        actual : np.ndarray
            Actual values
        predicted : np.ndarray
            Predicted values
            
        Returns:
        --------
        r2 : float
            R-squared value
        """
        ss_res = np.sum((actual - predicted) ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)
        
        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0
            
        return float(1 - (ss_res / ss_tot))
        
    def _calculate_directional_accuracy(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        """
        Calculate directional accuracy (percentage of correct direction predictions).
        
        Parameters:
        -----------
        actual : np.ndarray
            Actual values
        predicted : np.ndarray
            Predicted values
            
        Returns:
        --------
        directional_accuracy : float
            Directional accuracy as percentage
        """
        if len(actual) < 2 or len(predicted) < 2:
            return float('nan')
            
        # Calculate direction changes
        actual_direction = np.diff(actual) >= 0
        predicted_direction = np.diff(predicted) >= 0
        
        # Calculate accuracy
        correct_directions = actual_direction == predicted_direction
        return float(np.mean(correct_directions) * 100)
        
    def create_evaluation_report(self, actual: Union[pd.Series, np.ndarray], 
                               predicted: Union[pd.Series, np.ndarray],
                               historical_data: Optional[Union[pd.Series, np.ndarray]] = None,
                               model_name: str = "Model") -> Dict[str, Any]:
        """
        Create a comprehensive evaluation report.
        
        Parameters:
        -----------
        actual : pd.Series or np.ndarray
            Actual observed values
        predicted : pd.Series or np.ndarray
            Predicted/forecasted values
        historical_data : pd.Series or np.ndarray, optional
            Historical data for MASE calculation
        model_name : str, default='Model'
            Name of the model being evaluated
            
        Returns:
        --------
        report : dict
            Comprehensive evaluation report
        """
        metrics = self.evaluate_forecast(actual, predicted, historical_data)
        
        # Create performance categorization
        performance_category = self._categorize_performance(metrics)
        
        # Generate interpretation
        interpretation = self._generate_evaluation_interpretation(metrics, model_name)
        
        # Generate recommendations
        recommendations = self._generate_evaluation_recommendations(metrics)
        
        report = {
            'model_name': model_name,
            'evaluation_metrics': metrics,
            'performance_category': performance_category,
            'interpretation': interpretation,
            'recommendations': recommendations,
            'timestamp': datetime.now().isoformat()
        }
        
        return report
        
    def _categorize_performance(self, metrics: Dict[str, float]) -> str:
        """
        Categorize model performance based on metrics.
        
        Parameters:
        -----------
        metrics : dict
            Evaluation metrics
            
        Returns:
        --------
        category : str
            Performance category
        """
        # Use MAPE as primary metric for categorization
        mape = metrics.get('mape', float('inf'))
        
        if mape <= 5:
            return 'excellent'
        elif mape <= 15:
            return 'good'
        elif mape <= 25:
            return 'moderate'
        else:
            return 'poor'
            
    def _generate_evaluation_interpretation(self, metrics: Dict[str, float], model_name: str) -> str:
        """
        Generate human-readable interpretation of evaluation results.
        
        Parameters:
        -----------
        metrics : dict
            Evaluation metrics
        model_name : str
            Name of the model
            
        Returns:
        --------
        interpretation : str
            Human-readable interpretation
        """
        mae = metrics.get('mae', 0)
        mape = metrics.get('mape', 0)
        rmse = metrics.get('rmse', 0)
        r2 = metrics.get('r2', 0)
        n_obs = metrics.get('n_observations', 0)
        
        interpretation = (
            f"{model_name} evaluation on {n_obs} observations: "
            f"MAE = {mae:.3f}, MAPE = {mape:.1f}%, RMSE = {rmse:.3f}. "
        )
        
        if r2 != 0:
            interpretation += f"Variance explained (R²) = {r2:.3f}. "
            
        # Add performance assessment
        if mape <= 10:
            interpretation += "Forecast accuracy is excellent."
        elif mape <= 20:
            interpretation += "Forecast accuracy is good."
        elif mape <= 30:
            interpretation += "Forecast accuracy is moderate."
        else:
            interpretation += "Forecast accuracy needs improvement."
            
        return interpretation
        
    def _generate_evaluation_recommendations(self, metrics: Dict[str, float]) -> List[str]:
        """
        Generate recommendations based on evaluation metrics.
        
        Parameters:
        -----------
        metrics : dict
            Evaluation metrics
            
        Returns:
        --------
        recommendations : List[str]
            List of recommendations
        """
        recommendations = []
        
        mape = metrics.get('mape', 0)
        mae = metrics.get('mae', 0)
        rmse = metrics.get('rmse', 0)
        bias = abs(metrics.get('forecast_bias', 0))
        directional_accuracy = metrics.get('directional_accuracy', 0)
        
        if mape > 30:
            recommendations.append("High forecast error - consider alternative models or additional features")
            
        if rmse > mae * 2:
            recommendations.append("High RMSE relative to MAE suggests presence of large forecast errors")
            
        if bias > mae * 0.5:
            recommendations.append("Significant forecast bias detected - model systematically over/under-predicts")
            
        if directional_accuracy and directional_accuracy < 50:
            recommendations.append("Poor directional accuracy - model struggles to predict trend changes")
            
        if metrics.get('n_observations', 0) < 20:
            recommendations.append("Limited evaluation data - collect more out-of-sample observations for robust evaluation")
            
        if not recommendations:
            recommendations.append("Forecast performance is satisfactory - continue monitoring")
            
        return recommendations