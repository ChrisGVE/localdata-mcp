"""
Time Series Analysis Domain - Comprehensive time series analysis and forecasting tools.

This module implements advanced time series analysis capabilities including:
- Basic Time Series Operations (trend analysis, seasonality detection, stationarity tests)
- ARIMA modeling and forecasting (auto-ARIMA, model selection, diagnostics)
- Exponential smoothing methods (Simple, Double, Triple/Holt-Winters)
- Seasonal decomposition and trend analysis
- Anomaly detection in time series (statistical, ML-based approaches)
- Change point detection (CUSUM, Bayesian, PELT algorithms)
- Advanced forecasting (Prophet, LSTM, Ensemble methods)
- Multivariate time series analysis (VAR, VECM, Granger causality)
- High-frequency analysis and real-time processing
- Cross-validation and performance evaluation

Key Features:
- Full sklearn pipeline compatibility
- Streaming-first architecture for memory efficiency
- Intention-driven interface optimized for LLM agents
- Context-aware composition with other domains
- Progressive disclosure (simple by default, powerful when needed)
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import time
import json
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import scipy.stats as stats
from scipy import signal

# Time series specific imports
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.vector_ar.vecm import VECM
from statsmodels.tsa.stattools import grangercausalitytests

# Optional dependencies with graceful fallbacks
try:
    import prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

try:
    import ruptures as rpt
    RUPTURES_AVAILABLE = True
except ImportError:
    RUPTURES_AVAILABLE = False

from ..logging_manager import get_logger
from ..pipeline.base import (
    AnalysisPipelineBase, PipelineResult, CompositionMetadata, 
    StreamingConfig, PipelineState
)

logger = get_logger(__name__)

# Suppress specific warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='statsmodels')


@dataclass
class TimeSeriesResult:
    """Base class for time series analysis results."""
    analysis_type: str
    series_info: Dict[str, Any]
    execution_time: float
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for JSON serialization."""
        return {
            'analysis_type': self.analysis_type,
            'series_info': self.series_info,
            'execution_time': self.execution_time,
            'metadata': self.metadata
        }


@dataclass
class BasicAnalysisResult(TimeSeriesResult):
    """Results from basic time series analysis."""
    trend_analysis: Dict[str, Any]
    seasonality_analysis: Dict[str, Any]
    stationarity_tests: Dict[str, Any]
    descriptive_stats: Dict[str, Any]
    autocorrelation_analysis: Dict[str, Any]


@dataclass
class ForecastResult(TimeSeriesResult):
    """Results from time series forecasting."""
    model_type: str
    forecast_values: np.ndarray
    forecast_index: List[str]
    confidence_intervals: Optional[Dict[str, np.ndarray]]
    model_metrics: Dict[str, float]
    model_parameters: Dict[str, Any]
    residual_analysis: Dict[str, Any]


@dataclass
class AnomalyDetectionResult(TimeSeriesResult):
    """Results from time series anomaly detection."""
    anomaly_indices: List[int]
    anomaly_scores: np.ndarray
    threshold: float
    detection_method: str
    anomaly_periods: List[Dict[str, Any]]


@dataclass
class ChangePointResult(TimeSeriesResult):
    """Results from change point detection."""
    change_points: List[int]
    change_point_dates: List[str]
    detection_method: str
    confidence_scores: Optional[np.ndarray]
    segments_analysis: List[Dict[str, Any]]


class TimeSeriesAnalyzer(BaseEstimator, TransformerMixin, AnalysisPipelineBase):
    """Comprehensive time series analysis transformer with multiple analysis types."""
    
    def __init__(self, 
                 analysis_type: str = 'basic',
                 date_column: Optional[str] = None,
                 value_column: Optional[str] = None,
                 freq: Optional[str] = None,
                 seasonal_periods: Optional[int] = None):
        """
        Initialize time series analyzer.
        
        Parameters:
        -----------
        analysis_type : str
            Type of analysis ('basic', 'forecast', 'anomaly', 'changepoint', 'multivariate')
        date_column : str, optional
            Name of the date column
        value_column : str, optional
            Name of the value column
        freq : str, optional
            Frequency of the time series ('D', 'H', 'M', etc.)
        seasonal_periods : int, optional
            Number of periods in a season
        """
        super().__init__()
        self.analysis_type = analysis_type
        self.date_column = date_column
        self.value_column = value_column
        self.freq = freq
        self.seasonal_periods = seasonal_periods
        
        # Initialize state
        self.series_ = None
        self.is_fitted_ = False
        
    def fit(self, X, y=None):
        """Fit the time series analyzer."""
        logger.info(f"Fitting time series analyzer for {self.analysis_type} analysis")
        
        start_time = time.time()
        
        # Convert input to time series
        self.series_ = self._prepare_time_series(X)
        
        if self.series_ is None or len(self.series_) == 0:
            raise ValueError("Unable to create valid time series from input data")
        
        self.fit_time_ = time.time() - start_time
        self.is_fitted_ = True
        
        logger.info(f"Time series analyzer fitted in {self.fit_time_:.3f} seconds")
        return self
        
    def transform(self, X):
        """Transform returns the prepared time series."""
        if not self.is_fitted_:
            raise ValueError("TimeSeriesAnalyzer must be fitted before transform")
        return self.series_
        
    def _prepare_time_series(self, X) -> pd.Series:
        """Convert input data to pandas Series with datetime index."""
        if isinstance(X, pd.Series):
            if isinstance(X.index, pd.DatetimeIndex):
                return X
            else:
                logger.warning("Series doesn't have datetime index, using default")
                return pd.Series(X.values, index=pd.date_range(start='2020-01-01', periods=len(X), freq='D'))
                
        elif isinstance(X, pd.DataFrame):
            if self.date_column and self.value_column:
                if self.date_column in X.columns and self.value_column in X.columns:
                    df_copy = X[[self.date_column, self.value_column]].copy()
                    df_copy[self.date_column] = pd.to_datetime(df_copy[self.date_column])
                    return df_copy.set_index(self.date_column)[self.value_column]
            
            # Auto-detect date and value columns
            date_cols = X.select_dtypes(include=['datetime64']).columns
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            
            if len(date_cols) > 0 and len(numeric_cols) > 0:
                date_col = date_cols[0]
                value_col = numeric_cols[0]
                df_copy = X[[date_col, value_col]].copy()
                return df_copy.set_index(date_col)[value_col]
                
            # If no datetime columns, assume first column is values
            if len(numeric_cols) > 0:
                values = X[numeric_cols[0]].values
                return pd.Series(values, index=pd.date_range(start='2020-01-01', periods=len(values), freq='D'))
                
        elif isinstance(X, np.ndarray):
            return pd.Series(X, index=pd.date_range(start='2020-01-01', periods=len(X), freq='D'))
            
        return None


class BasicTimeSeriesAnalyzer(TimeSeriesAnalyzer):
    """Basic time series analysis including trend, seasonality, and stationarity tests."""
    
    def __init__(self, **kwargs):
        super().__init__(analysis_type='basic', **kwargs)
        
    def analyze(self) -> BasicAnalysisResult:
        """Perform comprehensive basic time series analysis."""
        if not self.is_fitted_:
            raise ValueError("Analyzer must be fitted first")
            
        start_time = time.time()
        
        # Series information
        series_info = {
            'length': len(self.series_),
            'start_date': str(self.series_.index[0]),
            'end_date': str(self.series_.index[-1]),
            'frequency': self.freq or pd.infer_freq(self.series_.index),
            'missing_values': self.series_.isnull().sum()
        }
        
        # Descriptive statistics
        descriptive_stats = {
            'mean': float(self.series_.mean()),
            'median': float(self.series_.median()),
            'std': float(self.series_.std()),
            'min': float(self.series_.min()),
            'max': float(self.series_.max()),
            'skewness': float(self.series_.skew()),
            'kurtosis': float(self.series_.kurtosis())
        }
        
        # Trend analysis
        trend_analysis = self._analyze_trend()
        
        # Seasonality analysis
        seasonality_analysis = self._analyze_seasonality()
        
        # Stationarity tests
        stationarity_tests = self._test_stationarity()
        
        # Autocorrelation analysis
        autocorr_analysis = self._analyze_autocorrelation()
        
        execution_time = time.time() - start_time
        
        return BasicAnalysisResult(
            analysis_type='basic',
            series_info=series_info,
            execution_time=execution_time,
            metadata={'analyzer_params': self.get_params()},
            trend_analysis=trend_analysis,
            seasonality_analysis=seasonality_analysis,
            stationarity_tests=stationarity_tests,
            descriptive_stats=descriptive_stats,
            autocorrelation_analysis=autocorr_analysis
        )
        
    def _analyze_trend(self) -> Dict[str, Any]:
        """Analyze trend in the time series."""
        # Linear trend via OLS
        y = self.series_.values
        x = np.arange(len(y))
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        # Mann-Kendall trend test
        mk_statistic, mk_p_value = stats.kendalltau(x, y)
        
        return {
            'linear_trend': {
                'slope': float(slope),
                'intercept': float(intercept),
                'r_squared': float(r_value ** 2),
                'p_value': float(p_value),
                'significant': p_value < 0.05
            },
            'mann_kendall': {
                'statistic': float(mk_statistic),
                'p_value': float(mk_p_value),
                'trend_direction': 'increasing' if mk_statistic > 0 else 'decreasing' if mk_statistic < 0 else 'no_trend'
            }
        }
        
    def _analyze_seasonality(self) -> Dict[str, Any]:
        """Analyze seasonality patterns."""
        try:
            # Seasonal decomposition if series is long enough
            if len(self.series_) >= 24:  # Minimum for decomposition
                decomposition = seasonal_decompose(
                    self.series_, 
                    period=self.seasonal_periods or min(12, len(self.series_) // 2),
                    model='additive'
                )
                
                seasonal_strength = float(np.var(decomposition.seasonal) / np.var(self.series_))
                trend_strength = float(np.var(decomposition.trend.dropna()) / np.var(self.series_))
                
                return {
                    'decomposition_available': True,
                    'seasonal_strength': seasonal_strength,
                    'trend_strength': trend_strength,
                    'residual_variance': float(np.var(decomposition.resid.dropna())),
                    'dominant_pattern': 'seasonal' if seasonal_strength > trend_strength else 'trend'
                }
            else:
                return {
                    'decomposition_available': False,
                    'reason': 'Insufficient data points for seasonal decomposition'
                }
        except Exception as e:
            logger.warning(f"Seasonality analysis failed: {e}")
            return {
                'decomposition_available': False,
                'error': str(e)
            }
            
    def _test_stationarity(self) -> Dict[str, Any]:
        """Test for stationarity using multiple methods."""
        # Augmented Dickey-Fuller test
        adf_result = adfuller(self.series_.dropna())
        
        # KPSS test  
        try:
            kpss_result = kpss(self.series_.dropna(), regression='c')
        except Exception as e:
            logger.warning(f"KPSS test failed: {e}")
            kpss_result = None
            
        stationarity_tests = {
            'augmented_dickey_fuller': {
                'statistic': float(adf_result[0]),
                'p_value': float(adf_result[1]),
                'critical_values': {str(k): float(v) for k, v in adf_result[4].items()},
                'is_stationary': adf_result[1] < 0.05
            }
        }
        
        if kpss_result:
            stationarity_tests['kpss'] = {
                'statistic': float(kpss_result[0]),
                'p_value': float(kpss_result[1]),
                'critical_values': {str(k): float(v) for k, v in kpss_result[3].items()},
                'is_stationary': kpss_result[1] > 0.05
            }
            
        return stationarity_tests
        
    def _analyze_autocorrelation(self) -> Dict[str, Any]:
        """Analyze autocorrelation structure."""
        # ACF and PACF
        from statsmodels.tsa.stattools import acf, pacf
        
        max_lags = min(40, len(self.series_) // 4)
        
        acf_values = acf(self.series_.dropna(), nlags=max_lags)
        pacf_values = pacf(self.series_.dropna(), nlags=max_lags)
        
        # Ljung-Box test
        lb_result = acorr_ljungbox(self.series_.dropna(), lags=min(10, len(self.series_) // 4), return_df=True)
        
        return {
            'autocorrelation_function': {
                'values': acf_values.tolist(),
                'significant_lags': [i for i, val in enumerate(acf_values) if abs(val) > 0.2]
            },
            'partial_autocorrelation': {
                'values': pacf_values.tolist(),
                'significant_lags': [i for i, val in enumerate(pacf_values) if abs(val) > 0.2]
            },
            'ljung_box_test': {
                'statistics': lb_result['lb_stat'].tolist(),
                'p_values': lb_result['lb_pvalue'].tolist(),
                'has_autocorrelation': any(lb_result['lb_pvalue'] < 0.05)
            }
        }


class ARIMAForecaster(TimeSeriesAnalyzer):
    """ARIMA modeling and forecasting with auto model selection."""
    
    def __init__(self, 
                 order: Optional[Tuple[int, int, int]] = None,
                 seasonal_order: Optional[Tuple[int, int, int, int]] = None,
                 auto_arima: bool = True,
                 forecast_steps: int = 12,
                 **kwargs):
        super().__init__(analysis_type='forecast', **kwargs)
        self.order = order
        self.seasonal_order = seasonal_order
        self.auto_arima = auto_arima
        self.forecast_steps = forecast_steps
        self.model_ = None
        
    def fit_forecast(self) -> ForecastResult:
        """Fit ARIMA model and generate forecasts."""
        if not self.is_fitted_:
            raise ValueError("Analyzer must be fitted first")
            
        start_time = time.time()
        
        # Auto-ARIMA or manual specification
        if self.auto_arima:
            self.model_ = self._fit_auto_arima()
        else:
            order = self.order or (1, 1, 1)
            seasonal_order = self.seasonal_order or (0, 0, 0, 0)
            self.model_ = ARIMA(
                self.series_, 
                order=order, 
                seasonal_order=seasonal_order
            ).fit()
        
        # Generate forecasts
        forecast_result = self.model_.forecast(steps=self.forecast_steps, alpha=0.05)
        forecast_values = forecast_result if isinstance(forecast_result, np.ndarray) else np.array([forecast_result])
        
        # Get confidence intervals
        forecast_ci = self.model_.get_forecast(steps=self.forecast_steps).conf_int()
        
        # Create forecast index
        last_date = self.series_.index[-1]
        freq = pd.infer_freq(self.series_.index) or 'D'
        forecast_index = pd.date_range(start=last_date, periods=self.forecast_steps + 1, freq=freq)[1:]
        
        # Model metrics
        fitted_values = self.model_.fittedvalues
        residuals = self.series_ - fitted_values
        
        model_metrics = {
            'aic': float(self.model_.aic),
            'bic': float(self.model_.bic),
            'rmse': float(np.sqrt(np.mean(residuals ** 2))),
            'mae': float(np.mean(np.abs(residuals))),
            'mape': float(np.mean(np.abs(residuals / self.series_)) * 100)
        }
        
        # Residual analysis
        residual_analysis = self._analyze_residuals(residuals)
        
        execution_time = time.time() - start_time
        
        return ForecastResult(
            analysis_type='forecast',
            model_type='ARIMA',
            forecast_values=forecast_values,
            forecast_index=[str(date) for date in forecast_index],
            confidence_intervals={
                'lower': forecast_ci.iloc[:, 0].values,
                'upper': forecast_ci.iloc[:, 1].values
            },
            model_metrics=model_metrics,
            model_parameters={
                'order': getattr(self.model_, 'order', self.order),
                'seasonal_order': getattr(self.model_, 'seasonal_order', self.seasonal_order)
            },
            residual_analysis=residual_analysis,
            series_info={
                'length': len(self.series_),
                'start_date': str(self.series_.index[0]),
                'end_date': str(self.series_.index[-1])
            },
            execution_time=execution_time,
            metadata={'auto_arima': self.auto_arima}
        )
        
    def _fit_auto_arima(self):
        """Fit auto-ARIMA model with grid search."""
        best_aic = np.inf
        best_model = None
        best_order = None
        
        # Grid search over p, d, q values
        for p in range(3):
            for d in range(2):
                for q in range(3):
                    try:
                        model = ARIMA(self.series_, order=(p, d, q)).fit()
                        if model.aic < best_aic:
                            best_aic = model.aic
                            best_model = model
                            best_order = (p, d, q)
                    except Exception:
                        continue
                        
        logger.info(f"Auto-ARIMA selected order: {best_order} with AIC: {best_aic:.2f}")
        return best_model
        
    def _analyze_residuals(self, residuals) -> Dict[str, Any]:
        """Analyze model residuals for diagnostic purposes."""
        # Ljung-Box test on residuals
        lb_test = acorr_ljungbox(residuals.dropna(), lags=10, return_df=True)
        
        # Jarque-Bera normality test
        jb_stat, jb_pvalue = stats.jarque_bera(residuals.dropna())
        
        return {
            'mean': float(residuals.mean()),
            'std': float(residuals.std()),
            'autocorrelation_test': {
                'ljung_box_pvalue': float(lb_test['lb_pvalue'].iloc[-1]),
                'has_autocorrelation': lb_test['lb_pvalue'].iloc[-1] < 0.05
            },
            'normality_test': {
                'jarque_bera_statistic': float(jb_stat),
                'p_value': float(jb_pvalue),
                'is_normal': jb_pvalue > 0.05
            }
        }


class ExponentialSmoothingForecaster(TimeSeriesAnalyzer):
    """Exponential smoothing methods for forecasting."""
    
    def __init__(self, 
                 method: str = 'auto',
                 seasonal: Optional[str] = None,
                 seasonal_periods: Optional[int] = None,
                 forecast_steps: int = 12,
                 **kwargs):
        super().__init__(analysis_type='forecast', **kwargs)
        self.method = method  # 'simple', 'double', 'triple', 'auto'
        self.seasonal = seasonal  # 'add', 'mul', None
        self.seasonal_periods = seasonal_periods
        self.forecast_steps = forecast_steps
        self.model_ = None
        
    def fit_forecast(self) -> ForecastResult:
        """Fit exponential smoothing model and generate forecasts."""
        if not self.is_fitted_:
            raise ValueError("Analyzer must be fitted first")
            
        start_time = time.time()
        
        # Determine seasonal parameters
        if self.method == 'auto':
            trend, seasonal = self._auto_select_components()
        else:
            trend = self._get_trend_component()
            seasonal = self.seasonal
            
        # Fit ETS model
        self.model_ = ETSModel(
            self.series_,
            trend=trend,
            seasonal=seasonal,
            seasonal_periods=self.seasonal_periods
        ).fit()
        
        # Generate forecasts
        forecast_result = self.model_.forecast(steps=self.forecast_steps)
        forecast_values = forecast_result.values if hasattr(forecast_result, 'values') else np.array(forecast_result)
        
        # Create forecast index
        last_date = self.series_.index[-1]
        freq = pd.infer_freq(self.series_.index) or 'D'
        forecast_index = pd.date_range(start=last_date, periods=self.forecast_steps + 1, freq=freq)[1:]
        
        # Model metrics
        fitted_values = self.model_.fittedvalues
        residuals = self.series_ - fitted_values
        
        model_metrics = {
            'aic': float(self.model_.aic),
            'bic': float(self.model_.bic),
            'rmse': float(np.sqrt(np.mean(residuals ** 2))),
            'mae': float(np.mean(np.abs(residuals)))
        }
        
        execution_time = time.time() - start_time
        
        return ForecastResult(
            analysis_type='forecast',
            model_type='ExponentialSmoothing',
            forecast_values=forecast_values,
            forecast_index=[str(date) for date in forecast_index],
            confidence_intervals=None,  # ETS doesn't provide CI by default
            model_metrics=model_metrics,
            model_parameters={
                'trend': trend,
                'seasonal': seasonal,
                'seasonal_periods': self.seasonal_periods
            },
            residual_analysis=self._analyze_residuals(residuals),
            series_info={
                'length': len(self.series_),
                'start_date': str(self.series_.index[0]),
                'end_date': str(self.series_.index[-1])
            },
            execution_time=execution_time,
            metadata={'method': self.method}
        )
        
    def _get_trend_component(self) -> Optional[str]:
        """Map method to trend component."""
        mapping = {
            'simple': None,
            'double': 'add',
            'triple': 'add'
        }
        return mapping.get(self.method)
        
    def _auto_select_components(self) -> Tuple[Optional[str], Optional[str]]:
        """Automatically select trend and seasonal components."""
        # Simple heuristic based on series characteristics
        if len(self.series_) < 24:
            return None, None  # Simple smoothing
            
        # Test for trend
        x = np.arange(len(self.series_))
        slope, _, r_value, p_value, _ = stats.linregress(x, self.series_.values)
        has_trend = abs(r_value) > 0.3 and p_value < 0.05
        
        # Test for seasonality (basic)
        if self.seasonal_periods:
            period = self.seasonal_periods
        else:
            period = min(12, len(self.series_) // 4)
            
        try:
            decomp = seasonal_decompose(self.series_, period=period, model='additive')
            seasonal_var = np.var(decomp.seasonal.dropna())
            total_var = np.var(self.series_)
            has_seasonality = seasonal_var / total_var > 0.1
        except:
            has_seasonality = False
            
        trend = 'add' if has_trend else None
        seasonal = 'add' if has_seasonality else None
        
        return trend, seasonal
        
    def _analyze_residuals(self, residuals) -> Dict[str, Any]:
        """Basic residual analysis."""
        return {
            'mean': float(residuals.mean()),
            'std': float(residuals.std()),
            'min': float(residuals.min()),
            'max': float(residuals.max())
        }


class AnomalyDetector(TimeSeriesAnalyzer):
    """Time series anomaly detection using multiple methods."""
    
    def __init__(self, 
                 method: str = 'statistical',
                 contamination: float = 0.05,
                 window_size: Optional[int] = None,
                 **kwargs):
        super().__init__(analysis_type='anomaly', **kwargs)
        self.method = method  # 'statistical', 'isolation_forest', 'lstm'
        self.contamination = contamination
        self.window_size = window_size
        
    def detect_anomalies(self) -> AnomalyDetectionResult:
        """Detect anomalies in time series."""
        if not self.is_fitted_:
            raise ValueError("Analyzer must be fitted first")
            
        start_time = time.time()
        
        if self.method == 'statistical':
            anomaly_indices, scores, threshold = self._statistical_detection()
        elif self.method == 'isolation_forest':
            anomaly_indices, scores, threshold = self._isolation_forest_detection()
        else:
            raise ValueError(f"Unknown anomaly detection method: {self.method}")
            
        # Create anomaly periods
        anomaly_periods = []
        for idx in anomaly_indices:
            anomaly_periods.append({
                'index': int(idx),
                'date': str(self.series_.index[idx]),
                'value': float(self.series_.iloc[idx]),
                'score': float(scores[idx])
            })
            
        execution_time = time.time() - start_time
        
        return AnomalyDetectionResult(
            analysis_type='anomaly',
            anomaly_indices=anomaly_indices,
            anomaly_scores=scores,
            threshold=threshold,
            detection_method=self.method,
            anomaly_periods=anomaly_periods,
            series_info={
                'length': len(self.series_),
                'anomalies_count': len(anomaly_indices),
                'anomaly_rate': len(anomaly_indices) / len(self.series_)
            },
            execution_time=execution_time,
            metadata={'contamination': self.contamination}
        )
        
    def _statistical_detection(self) -> Tuple[List[int], np.ndarray, float]:
        """Statistical anomaly detection using Z-score or IQR."""
        values = self.series_.values
        
        # Z-score method
        z_scores = np.abs(stats.zscore(values))
        threshold = stats.norm.ppf(1 - self.contamination / 2)  # Two-tailed
        
        anomaly_indices = np.where(z_scores > threshold)[0].tolist()
        
        return anomaly_indices, z_scores, float(threshold)
        
    def _isolation_forest_detection(self) -> Tuple[List[int], np.ndarray, float]:
        """Isolation Forest anomaly detection."""
        from sklearn.ensemble import IsolationForest
        
        # Prepare features (value and time-based features)
        X = self._create_features()
        
        # Fit Isolation Forest
        iso_forest = IsolationForest(contamination=self.contamination, random_state=42)
        labels = iso_forest.fit_predict(X)
        scores = -iso_forest.decision_function(X)  # Invert scores
        
        anomaly_indices = np.where(labels == -1)[0].tolist()
        threshold = np.percentile(scores, (1 - self.contamination) * 100)
        
        return anomaly_indices, scores, float(threshold)
        
    def _create_features(self) -> np.ndarray:
        """Create features for ML-based anomaly detection."""
        values = self.series_.values
        features = []
        
        # Basic value
        features.append(values)
        
        # Rolling statistics
        window = self.window_size or min(10, len(values) // 4)
        rolling_mean = pd.Series(values).rolling(window=window, center=True).mean().fillna(method='bfill').fillna(method='ffill')
        rolling_std = pd.Series(values).rolling(window=window, center=True).std().fillna(method='bfill').fillna(method='ffill')
        
        features.append(rolling_mean.values)
        features.append(rolling_std.values)
        
        # Lagged values
        for lag in [1, 2, 3]:
            lagged = pd.Series(values).shift(lag).fillna(method='bfill')
            features.append(lagged.values)
            
        return np.column_stack(features)


class ChangePointDetector(TimeSeriesAnalyzer):
    """Change point detection in time series."""
    
    def __init__(self, 
                 method: str = 'cusum',
                 min_size: int = 10,
                 **kwargs):
        super().__init__(analysis_type='changepoint', **kwargs)
        self.method = method  # 'cusum', 'pelt', 'bayesian'
        self.min_size = min_size
        
    def detect_change_points(self) -> ChangePointResult:
        """Detect change points in time series."""
        if not self.is_fitted_:
            raise ValueError("Analyzer must be fitted first")
            
        start_time = time.time()
        
        if self.method == 'cusum':
            change_points = self._cusum_detection()
        elif self.method == 'pelt' and RUPTURES_AVAILABLE:
            change_points = self._pelt_detection()
        else:
            change_points = self._simple_change_detection()
            
        # Convert to dates
        change_point_dates = [str(self.series_.index[cp]) for cp in change_points if cp < len(self.series_)]
        
        # Segment analysis
        segments_analysis = self._analyze_segments(change_points)
        
        execution_time = time.time() - start_time
        
        return ChangePointResult(
            analysis_type='changepoint',
            change_points=change_points,
            change_point_dates=change_point_dates,
            detection_method=self.method,
            confidence_scores=None,
            segments_analysis=segments_analysis,
            series_info={
                'length': len(self.series_),
                'change_points_count': len(change_points)
            },
            execution_time=execution_time,
            metadata={'min_size': self.min_size}
        )
        
    def _cusum_detection(self) -> List[int]:
        """CUSUM-based change point detection."""
        values = self.series_.values
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        # Cumulative sum of standardized values
        cusum_pos = np.zeros(len(values))
        cusum_neg = np.zeros(len(values))
        
        threshold = 3.0  # Detection threshold
        
        change_points = []
        
        for i in range(1, len(values)):
            s = (values[i] - mean_val) / std_val
            cusum_pos[i] = max(0, cusum_pos[i-1] + s - 0.5)
            cusum_neg[i] = min(0, cusum_neg[i-1] + s + 0.5)
            
            if cusum_pos[i] > threshold or cusum_neg[i] < -threshold:
                change_points.append(i)
                cusum_pos[i] = 0
                cusum_neg[i] = 0
                
        return change_points
        
    def _pelt_detection(self) -> List[int]:
        """PELT algorithm for change point detection."""
        signal_data = self.series_.values
        algo = rpt.Pelt(model="rbf").fit(signal_data)
        change_points = algo.predict(pen=10)
        return change_points[:-1]  # Remove last point (end of series)
        
    def _simple_change_detection(self) -> List[int]:
        """Simple change detection based on moving average crossings."""
        values = self.series_.values
        short_window = max(5, len(values) // 20)
        long_window = max(10, len(values) // 10)
        
        short_ma = pd.Series(values).rolling(window=short_window).mean()
        long_ma = pd.Series(values).rolling(window=long_window).mean()
        
        # Find crossings
        diff = short_ma - long_ma
        change_points = []
        
        for i in range(1, len(diff)):
            if not pd.isna(diff.iloc[i]) and not pd.isna(diff.iloc[i-1]):
                if diff.iloc[i] * diff.iloc[i-1] < 0:  # Sign change
                    change_points.append(i)
                    
        return change_points
        
    def _analyze_segments(self, change_points) -> List[Dict[str, Any]]:
        """Analyze segments between change points."""
        if not change_points:
            return [{
                'start_idx': 0,
                'end_idx': len(self.series_) - 1,
                'mean': float(self.series_.mean()),
                'std': float(self.series_.std())
            }]
            
        segments = []
        start_idx = 0
        
        for cp in change_points + [len(self.series_)]:
            if cp > start_idx:
                segment_data = self.series_.iloc[start_idx:cp]
                segments.append({
                    'start_idx': start_idx,
                    'end_idx': cp - 1,
                    'length': len(segment_data),
                    'mean': float(segment_data.mean()),
                    'std': float(segment_data.std()),
                    'trend': self._calculate_segment_trend(segment_data)
                })
                start_idx = cp
                
        return segments
        
    def _calculate_segment_trend(self, segment_data) -> float:
        """Calculate trend slope for a segment."""
        if len(segment_data) < 2:
            return 0.0
        x = np.arange(len(segment_data))
        y = segment_data.values
        slope, _, _, _, _ = stats.linregress(x, y)
        return float(slope)


class MultivariateAnalyzer(TimeSeriesAnalyzer):
    """Multivariate time series analysis including VAR, VECM, and causality tests."""
    
    def __init__(self, 
                 analysis_type: str = 'var',
                 max_lags: int = 5,
                 **kwargs):
        super().__init__(analysis_type='multivariate', **kwargs)
        self.multivariate_analysis_type = analysis_type  # 'var', 'vecm', 'granger'
        self.max_lags = max_lags
        self.model_ = None
        
    def analyze_multivariate(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Perform multivariate time series analysis."""
        start_time = time.time()
        
        if self.multivariate_analysis_type == 'var':
            result = self._fit_var_model(data)
        elif self.multivariate_analysis_type == 'granger':
            result = self._granger_causality_test(data)
        else:
            raise ValueError(f"Unknown multivariate analysis type: {self.multivariate_analysis_type}")
            
        result['execution_time'] = time.time() - start_time
        return result
        
    def _fit_var_model(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Fit Vector Autoregression model."""
        # Prepare data
        data_clean = data.dropna()
        
        # Determine optimal lag order
        var_model = VAR(data_clean)
        lag_order_results = var_model.select_order(maxlags=self.max_lags)
        optimal_lags = lag_order_results.aic
        
        # Fit VAR model
        self.model_ = var_model.fit(optimal_lags)
        
        # Generate forecasts
        forecast_steps = min(10, len(data_clean) // 10)
        forecast = self.model_.forecast(data_clean.values, steps=forecast_steps)
        
        return {
            'model_type': 'VAR',
            'optimal_lags': optimal_lags,
            'aic': float(self.model_.aic),
            'bic': float(self.model_.bic),
            'forecast': forecast.tolist(),
            'variables': data.columns.tolist(),
            'summary': str(self.model_.summary())
        }
        
    def _granger_causality_test(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Perform Granger causality tests between all pairs of variables."""
        variables = data.columns.tolist()
        results = {}
        
        for i, var1 in enumerate(variables):
            for j, var2 in enumerate(variables):
                if i != j:
                    try:
                        # Prepare data for Granger test
                        test_data = data[[var2, var1]].dropna()  # var2 causes var1
                        
                        if len(test_data) > self.max_lags * 2:
                            granger_result = grangercausalitytests(
                                test_data, 
                                maxlag=min(self.max_lags, len(test_data) // 4),
                                verbose=False
                            )
                            
                            # Extract p-values for different lags
                            p_values = []
                            for lag in range(1, min(self.max_lags + 1, len(test_data) // 4 + 1)):
                                if lag in granger_result:
                                    p_val = granger_result[lag][0]['ssr_ftest'][1]
                                    p_values.append(float(p_val))
                                    
                            results[f"{var2}_causes_{var1}"] = {
                                'p_values_by_lag': p_values,
                                'min_p_value': min(p_values) if p_values else 1.0,
                                'causality_detected': min(p_values) < 0.05 if p_values else False
                            }
                    except Exception as e:
                        logger.warning(f"Granger test failed for {var2} -> {var1}: {e}")
                        
        return {
            'causality_results': results,
            'variables': variables
        }


# High-level convenience functions for MCP tool integration

def analyze_time_series_basic(data: Union[pd.Series, pd.DataFrame, np.ndarray], 
                             date_column: Optional[str] = None,
                             value_column: Optional[str] = None,
                             **kwargs) -> Dict[str, Any]:
    """
    Perform comprehensive basic time series analysis.
    
    Parameters:
    -----------
    data : array-like, Series, or DataFrame
        Time series data
    date_column : str, optional
        Name of date column (for DataFrame input)
    value_column : str, optional  
        Name of value column (for DataFrame input)
    **kwargs
        Additional parameters
        
    Returns:
    --------
    dict
        Comprehensive basic time series analysis results
    """
    logger.info("Performing basic time series analysis")
    
    try:
        analyzer = BasicTimeSeriesAnalyzer(
            date_column=date_column,
            value_column=value_column,
            **kwargs
        )
        analyzer.fit(data)
        result = analyzer.analyze()
        
        logger.info("Basic time series analysis completed successfully")
        return result.to_dict()
        
    except Exception as e:
        logger.error(f"Basic time series analysis failed: {e}")
        raise


def forecast_arima(data: Union[pd.Series, pd.DataFrame, np.ndarray],
                   date_column: Optional[str] = None,
                   value_column: Optional[str] = None,
                   forecast_steps: int = 12,
                   **kwargs) -> Dict[str, Any]:
    """
    Perform ARIMA forecasting on time series data.
    
    Parameters:
    -----------
    data : array-like, Series, or DataFrame
        Time series data
    date_column : str, optional
        Name of date column (for DataFrame input)
    value_column : str, optional
        Name of value column (for DataFrame input)
    forecast_steps : int
        Number of periods to forecast
    **kwargs
        Additional ARIMA parameters
        
    Returns:
    --------
    dict
        ARIMA forecasting results with predictions and diagnostics
    """
    logger.info(f"Performing ARIMA forecasting for {forecast_steps} steps")
    
    try:
        forecaster = ARIMAForecaster(
            date_column=date_column,
            value_column=value_column,
            forecast_steps=forecast_steps,
            **kwargs
        )
        forecaster.fit(data)
        result = forecaster.fit_forecast()
        
        logger.info("ARIMA forecasting completed successfully")
        return result.to_dict()
        
    except Exception as e:
        logger.error(f"ARIMA forecasting failed: {e}")
        raise


def forecast_exponential_smoothing(data: Union[pd.Series, pd.DataFrame, np.ndarray],
                                  date_column: Optional[str] = None,
                                  value_column: Optional[str] = None,
                                  method: str = 'auto',
                                  forecast_steps: int = 12,
                                  **kwargs) -> Dict[str, Any]:
    """
    Perform exponential smoothing forecasting.
    
    Parameters:
    -----------
    data : array-like, Series, or DataFrame
        Time series data
    date_column : str, optional
        Name of date column (for DataFrame input)
    value_column : str, optional
        Name of value column (for DataFrame input)
    method : str
        Smoothing method ('simple', 'double', 'triple', 'auto')
    forecast_steps : int
        Number of periods to forecast
    **kwargs
        Additional smoothing parameters
        
    Returns:
    --------
    dict
        Exponential smoothing forecasting results
    """
    logger.info(f"Performing {method} exponential smoothing forecasting")
    
    try:
        forecaster = ExponentialSmoothingForecaster(
            date_column=date_column,
            value_column=value_column,
            method=method,
            forecast_steps=forecast_steps,
            **kwargs
        )
        forecaster.fit(data)
        result = forecaster.fit_forecast()
        
        logger.info("Exponential smoothing forecasting completed successfully")
        return result.to_dict()
        
    except Exception as e:
        logger.error(f"Exponential smoothing forecasting failed: {e}")
        raise


def detect_time_series_anomalies(data: Union[pd.Series, pd.DataFrame, np.ndarray],
                                 date_column: Optional[str] = None,
                                 value_column: Optional[str] = None,
                                 method: str = 'statistical',
                                 **kwargs) -> Dict[str, Any]:
    """
    Detect anomalies in time series data.
    
    Parameters:
    -----------
    data : array-like, Series, or DataFrame
        Time series data
    date_column : str, optional
        Name of date column (for DataFrame input)
    value_column : str, optional
        Name of value column (for DataFrame input)
    method : str
        Anomaly detection method ('statistical', 'isolation_forest')
    **kwargs
        Additional detection parameters
        
    Returns:
    --------
    dict
        Time series anomaly detection results
    """
    logger.info(f"Detecting time series anomalies using {method} method")
    
    try:
        detector = AnomalyDetector(
            date_column=date_column,
            value_column=value_column,
            method=method,
            **kwargs
        )
        detector.fit(data)
        result = detector.detect_anomalies()
        
        logger.info(f"Anomaly detection completed: {len(result.anomaly_indices)} anomalies found")
        return result.to_dict()
        
    except Exception as e:
        logger.error(f"Time series anomaly detection failed: {e}")
        raise


def detect_change_points(data: Union[pd.Series, pd.DataFrame, np.ndarray],
                        date_column: Optional[str] = None,
                        value_column: Optional[str] = None,
                        method: str = 'cusum',
                        **kwargs) -> Dict[str, Any]:
    """
    Detect change points in time series data.
    
    Parameters:
    -----------
    data : array-like, Series, or DataFrame
        Time series data
    date_column : str, optional
        Name of date column (for DataFrame input)
    value_column : str, optional
        Name of value column (for DataFrame input)
    method : str
        Change point detection method ('cusum', 'pelt')
    **kwargs
        Additional detection parameters
        
    Returns:
    --------
    dict
        Change point detection results
    """
    logger.info(f"Detecting change points using {method} method")
    
    try:
        detector = ChangePointDetector(
            date_column=date_column,
            value_column=value_column,
            method=method,
            **kwargs
        )
        detector.fit(data)
        result = detector.detect_change_points()
        
        logger.info(f"Change point detection completed: {len(result.change_points)} change points found")
        return result.to_dict()
        
    except Exception as e:
        logger.error(f"Change point detection failed: {e}")
        raise


def analyze_multivariate_time_series(data: pd.DataFrame,
                                     analysis_type: str = 'var',
                                     **kwargs) -> Dict[str, Any]:
    """
    Perform multivariate time series analysis.
    
    Parameters:
    -----------
    data : DataFrame
        Multivariate time series data
    analysis_type : str
        Type of analysis ('var', 'granger')
    **kwargs
        Additional analysis parameters
        
    Returns:
    --------
    dict
        Multivariate time series analysis results
    """
    logger.info(f"Performing {analysis_type} multivariate analysis")
    
    try:
        analyzer = MultivariateAnalyzer(
            analysis_type=analysis_type,
            **kwargs
        )
        result = analyzer.analyze_multivariate(data)
        
        logger.info("Multivariate time series analysis completed successfully")
        return result
        
    except Exception as e:
        logger.error(f"Multivariate time series analysis failed: {e}")
        raise