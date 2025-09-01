"""
Time Series Integration Adapter - LocalData MCP v2.0

Specialized adapter for time series analysis libraries:
- statsmodels: Advanced statistical modeling (ARIMA, VAR, etc.)
- Prophet: Facebook's forecasting library with trend/seasonality
- sktime: Time series machine learning (has some sklearn compatibility)
- scipy.stats: Statistical tests and distributions

Key Integration Challenges:
- Different API patterns: fit/predict vs model().fit().summary()
- Time index handling and frequency detection
- Seasonal decomposition and forecasting horizons
- Multi-variate vs univariate time series
- Streaming compatibility for continuous forecasting
"""

from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from dataclasses import dataclass
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings

# Import base integration architecture
from library_integration_shims import (
    BaseLibraryAdapter,
    LibraryCategory,
    LibraryDependency, 
    IntegrationStrategy,
    IntegrationMetadata,
    LibraryIntegrationResult,
    requires_library,
    CompositionError
)

logger = logging.getLogger(__name__)


# ============================================================================
# Time Series-Specific Data Structures
# ============================================================================

@dataclass
class TimeSeriesContext:
    """Context for time series operations."""
    datetime_column: str = 'datetime'
    target_column: Optional[str] = None
    frequency: Optional[str] = None  # 'D', 'H', 'M', etc.
    seasonal_periods: Optional[int] = None
    forecast_horizon: int = 30
    confidence_level: float = 0.95


@dataclass 
class TimeSeriesMetadata(IntegrationMetadata):
    """Extended metadata for time series operations."""
    model_type: Optional[str] = None
    frequency_detected: Optional[str] = None
    seasonal_decomposition: Optional[Dict[str, Any]] = None
    forecast_horizon: Optional[int] = None
    fit_metrics: Optional[Dict[str, float]] = None
    residual_diagnostics: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.seasonal_decomposition is None:
            self.seasonal_decomposition = {}
        if self.fit_metrics is None:
            self.fit_metrics = {}
        if self.residual_diagnostics is None:
            self.residual_diagnostics = {}


# ============================================================================
# Time Series Adapter Implementation
# ============================================================================

class TimeSeriesAdapter(BaseLibraryAdapter):
    """
    Integration adapter for time series analysis libraries.
    
    Handles:
    - Statistical time series models (ARIMA, VAR, etc.)
    - Forecasting with Prophet and seasonal patterns
    - Time series preprocessing and feature engineering
    - Model evaluation with proper time series validation
    - Streaming forecasting with model updating
    """
    
    def __init__(self):
        dependencies = [
            LibraryDependency(
                name="statsmodels",
                import_path="statsmodels.api",
                min_version="0.13.0",
                sklearn_equivalent="sklearn.linear_model.LinearRegression",
                installation_hint="pip install statsmodels"
            ),
            LibraryDependency(
                name="prophet",
                import_path="prophet",
                sklearn_equivalent="sklearn.linear_model.LinearRegression", 
                installation_hint="pip install prophet"
            ),
            LibraryDependency(
                name="sktime",
                import_path="sktime.forecasting.arima",
                sklearn_equivalent="sklearn.linear_model.LinearRegression",
                installation_hint="pip install sktime"
            ),
            LibraryDependency(
                name="scipy",
                import_path="scipy.stats",
                min_version="1.7.0",
                is_optional=False,  # scipy is usually available
                installation_hint="pip install scipy"
            )
        ]
        
        super().__init__(LibraryCategory.TIME_SERIES, dependencies)
    
    def get_supported_functions(self) -> Dict[str, Callable]:
        """Return supported time series functions."""
        return {
            # Data Preparation
            'prepare_timeseries': self.prepare_timeseries,
            'detect_frequency': self.detect_frequency,
            'handle_missing_values': self.handle_missing_values,
            'seasonal_decomposition': self.seasonal_decomposition,
            
            # Forecasting Models
            'arima_forecast': self.arima_forecast,
            'prophet_forecast': self.prophet_forecast,
            'exponential_smoothing': self.exponential_smoothing,
            'linear_trend_forecast': self.linear_trend_forecast,
            
            # Statistical Tests
            'stationarity_test': self.stationarity_test,
            'seasonality_test': self.seasonality_test,
            'autocorrelation_analysis': self.autocorrelation_analysis,
            
            # Model Evaluation
            'forecast_accuracy': self.forecast_accuracy,
            'residual_diagnostics': self.residual_diagnostics,
            'cross_validation_timeseries': self.cross_validation_timeseries,
            
            # Feature Engineering
            'lag_features': self.create_lag_features,
            'rolling_features': self.create_rolling_features,
            'time_features': self.create_time_features
        }
    
    def adapt_function_call(self,
                          function_name: str, 
                          data: Any,
                          parameters: Dict[str, Any]) -> Tuple[Any, TimeSeriesMetadata]:
        """Adapt function call to time series library APIs."""
        
        if function_name not in self.get_supported_functions():
            raise CompositionError(
                f"Unsupported time series function: {function_name}",
                error_type="unsupported_function"
            )
        
        func = self.get_supported_functions()[function_name]
        
        # Ensure data has proper time series structure
        prepared_data, prep_transformations = self._prepare_timeseries_data(data, parameters)
        
        # Execute the function
        try:
            result = func(prepared_data, **parameters)
            
            # Convert result to standard format
            output_result, output_transformations = self.convert_output_data(result)
            
            # Create metadata
            metadata = TimeSeriesMetadata(
                library_used=self._detect_primary_library(function_name),
                integration_strategy=self._get_integration_strategy(function_name),
                data_transformations=prep_transformations + output_transformations,
                streaming_compatible=self._is_streaming_compatible(function_name),
                original_parameters=parameters
            )
            
            return output_result, metadata
            
        except Exception as e:
            return self._handle_timeseries_error(function_name, data, parameters, e)
    
    # ========================================================================
    # Data Preparation Functions
    # ========================================================================
    
    def prepare_timeseries(self, data: pd.DataFrame, **params) -> pd.DataFrame:
        """Prepare DataFrame for time series analysis."""
        datetime_col = params.get('datetime_column', 'datetime')
        target_col = params.get('target_column', 'value')
        frequency = params.get('frequency', None)
        
        # Ensure datetime column is properly formatted
        if datetime_col not in data.columns:
            raise ValueError(f"Datetime column '{datetime_col}' not found in data")
        
        df = data.copy()
        df[datetime_col] = pd.to_datetime(df[datetime_col])
        
        # Set as index for time series operations
        if df.index.name != datetime_col:
            df = df.set_index(datetime_col)
        
        # Sort by datetime
        df = df.sort_index()
        
        # Infer or set frequency
        if frequency:
            df = df.asfreq(frequency)
        else:
            # Try to infer frequency
            try:
                df.index.freq = pd.infer_freq(df.index)
            except:
                logger.warning("Could not infer frequency from datetime index")
        
        return df
    
    def detect_frequency(self, data: pd.DataFrame, **params) -> Dict[str, Any]:
        """Detect the frequency of time series data."""
        datetime_col = params.get('datetime_column', 'datetime')
        
        if datetime_col in data.columns:
            datetime_series = pd.to_datetime(data[datetime_col])
        elif isinstance(data.index, pd.DatetimeIndex):
            datetime_series = data.index
        else:
            raise ValueError("No datetime column or index found")
        
        # Calculate time differences
        time_diffs = datetime_series.diff().dropna()
        most_common_diff = time_diffs.mode()[0] if len(time_diffs) > 0 else None
        
        # Infer frequency
        try:
            inferred_freq = pd.infer_freq(datetime_series)
        except:
            inferred_freq = None
        
        return pd.DataFrame([{
            'inferred_frequency': inferred_freq,
            'most_common_interval': str(most_common_diff),
            'min_interval': str(time_diffs.min()),
            'max_interval': str(time_diffs.max()),
            'irregular_intervals': len(time_diffs.unique()) > 1
        }])
    
    def handle_missing_values(self, data: pd.DataFrame, **params) -> pd.DataFrame:
        """Handle missing values in time series data."""
        method = params.get('method', 'interpolate')
        limit = params.get('limit', None)
        
        result = data.copy()
        
        if method == 'interpolate':
            result = result.interpolate(method='time', limit=limit)
        elif method == 'forward_fill':
            result = result.fillna(method='ffill', limit=limit)
        elif method == 'backward_fill':
            result = result.fillna(method='bfill', limit=limit)
        elif method == 'drop':
            result = result.dropna()
        
        return result
    
    @requires_library("statsmodels")
    def seasonal_decomposition(self, data: pd.DataFrame, **params) -> pd.DataFrame:
        """Perform seasonal decomposition of time series."""
        import statsmodels.api as sm
        
        target_col = params.get('target_column', data.columns[0])
        model = params.get('model', 'additive')  # 'additive' or 'multiplicative'
        period = params.get('period', None)
        
        if period is None:
            # Auto-detect seasonality
            period = self._detect_seasonality(data[target_col])
        
        decomposition = sm.tsa.seasonal_decompose(
            data[target_col],
            model=model,
            period=period
        )
        
        result = pd.DataFrame({
            'observed': decomposition.observed,
            'trend': decomposition.trend, 
            'seasonal': decomposition.seasonal,
            'residual': decomposition.resid
        })
        
        return result
    
    # ========================================================================
    # Forecasting Models
    # ========================================================================
    
    @requires_library("statsmodels")
    def arima_forecast(self, data: pd.DataFrame, **params) -> pd.DataFrame:
        """ARIMA forecasting using statsmodels."""
        import statsmodels.api as sm
        
        target_col = params.get('target_column', data.columns[0])
        order = params.get('order', (1, 1, 1))  # (p, d, q)
        forecast_periods = params.get('forecast_periods', 30)
        confidence_level = params.get('confidence_level', 0.95)
        
        # Fit ARIMA model
        model = sm.tsa.ARIMA(data[target_col], order=order)
        fitted_model = model.fit()
        
        # Generate forecasts
        forecast = fitted_model.forecast(steps=forecast_periods)
        conf_int = fitted_model.get_prediction(
            start=len(data),
            end=len(data) + forecast_periods - 1,
            dynamic=False
        ).conf_int(alpha=1-confidence_level)
        
        # Create result DataFrame
        forecast_index = pd.date_range(
            start=data.index[-1] + pd.Timedelta(days=1),
            periods=forecast_periods,
            freq=data.index.freq
        )
        
        result = pd.DataFrame({
            'forecast': forecast,
            'lower_bound': conf_int.iloc[:, 0],
            'upper_bound': conf_int.iloc[:, 1]
        }, index=forecast_index)
        
        # Add model diagnostics
        result.attrs['model_summary'] = str(fitted_model.summary())
        result.attrs['aic'] = fitted_model.aic
        result.attrs['bic'] = fitted_model.bic
        
        return result
    
    @requires_library("prophet")
    def prophet_forecast(self, data: pd.DataFrame, **params) -> pd.DataFrame:
        """Prophet forecasting with trend and seasonality."""
        from prophet import Prophet
        
        datetime_col = params.get('datetime_column', 'ds')
        target_col = params.get('target_column', 'y')
        forecast_periods = params.get('forecast_periods', 30)
        seasonality_mode = params.get('seasonality_mode', 'additive')
        
        # Prepare data in Prophet format
        prophet_data = data.rename(columns={
            data.index.name or datetime_col: 'ds',
            target_col: 'y'
        }).reset_index()
        
        if 'ds' not in prophet_data.columns:
            prophet_data['ds'] = prophet_data.index
        
        # Initialize and fit Prophet model
        model = Prophet(seasonality_mode=seasonality_mode)
        model.fit(prophet_data)
        
        # Create future DataFrame
        future = model.make_future_dataframe(periods=forecast_periods)
        
        # Generate forecasts
        forecast = model.predict(future)
        
        # Return only future forecasts
        forecast_only = forecast.tail(forecast_periods).set_index('ds')
        
        result = pd.DataFrame({
            'forecast': forecast_only['yhat'],
            'lower_bound': forecast_only['yhat_lower'], 
            'upper_bound': forecast_only['yhat_upper'],
            'trend': forecast_only['trend'],
            'weekly': forecast_only.get('weekly', 0),
            'yearly': forecast_only.get('yearly', 0)
        })
        
        return result
    
    def exponential_smoothing(self, data: pd.DataFrame, **params) -> pd.DataFrame:
        """Exponential smoothing forecasting."""
        if self.is_library_available("statsmodels"):
            return self._statsmodels_exponential_smoothing(data, **params)
        else:
            return self._simple_exponential_smoothing_fallback(data, **params)
    
    def linear_trend_forecast(self, data: pd.DataFrame, **params) -> pd.DataFrame:
        """Simple linear trend forecasting as fallback."""
        from sklearn.linear_model import LinearRegression
        
        target_col = params.get('target_column', data.columns[0])
        forecast_periods = params.get('forecast_periods', 30)
        
        # Convert datetime index to numeric for regression
        X = np.arange(len(data)).reshape(-1, 1)
        y = data[target_col].values
        
        # Fit linear regression
        model = LinearRegression()
        model.fit(X, y)
        
        # Generate forecasts
        future_X = np.arange(len(data), len(data) + forecast_periods).reshape(-1, 1)
        forecast_values = model.predict(future_X)
        
        # Create result DataFrame
        forecast_index = pd.date_range(
            start=data.index[-1] + pd.Timedelta(days=1),
            periods=forecast_periods,
            freq=data.index.freq
        )
        
        result = pd.DataFrame({
            'forecast': forecast_values,
            'lower_bound': forecast_values * 0.95,  # Simple confidence interval
            'upper_bound': forecast_values * 1.05
        }, index=forecast_index)
        
        return result
    
    # ========================================================================
    # Statistical Tests
    # ========================================================================
    
    def stationarity_test(self, data: pd.DataFrame, **params) -> pd.DataFrame:
        """Test for stationarity using ADF test."""
        target_col = params.get('target_column', data.columns[0])
        
        if self.is_library_available("statsmodels"):
            from statsmodels.tsa.stattools import adfuller
            
            result = adfuller(data[target_col].dropna())
            
            return pd.DataFrame([{
                'adf_statistic': result[0],
                'p_value': result[1],
                'critical_value_1%': result[4]['1%'],
                'critical_value_5%': result[4]['5%'],
                'critical_value_10%': result[4]['10%'],
                'is_stationary': result[1] < 0.05
            }])
        else:
            # Fallback: Simple statistical test
            series = data[target_col].dropna()
            rolling_mean = series.rolling(window=12).mean()
            rolling_std = series.rolling(window=12).std()
            
            mean_stability = rolling_mean.std() / series.mean()
            variance_stability = rolling_std.std() / series.std()
            
            return pd.DataFrame([{
                'mean_stability': mean_stability,
                'variance_stability': variance_stability,
                'is_stationary_estimate': mean_stability < 0.1 and variance_stability < 0.1
            }])
    
    def seasonality_test(self, data: pd.DataFrame, **params) -> pd.DataFrame:
        """Test for seasonality patterns."""
        target_col = params.get('target_column', data.columns[0])
        
        # Simple seasonality detection
        series = data[target_col].dropna()
        
        # Test for different seasonal periods
        periods_to_test = [7, 12, 24, 365]  # Daily, monthly, bi-annual, yearly
        seasonality_scores = {}
        
        for period in periods_to_test:
            if len(series) > 2 * period:
                # Calculate autocorrelation at seasonal lag
                autocorr = series.autocorr(lag=period)
                seasonality_scores[f'period_{period}'] = abs(autocorr)
        
        # Find strongest seasonality
        if seasonality_scores:
            best_period = max(seasonality_scores.items(), key=lambda x: x[1])
            
            return pd.DataFrame([{
                'detected_period': int(best_period[0].split('_')[1]),
                'seasonality_strength': best_period[1],
                'has_seasonality': best_period[1] > 0.3,
                **seasonality_scores
            }])
        else:
            return pd.DataFrame([{'has_seasonality': False}])
    
    def autocorrelation_analysis(self, data: pd.DataFrame, **params) -> pd.DataFrame:
        """Analyze autocorrelation structure.""" 
        target_col = params.get('target_column', data.columns[0])
        max_lags = params.get('max_lags', 20)
        
        series = data[target_col].dropna()
        
        # Calculate autocorrelations
        autocorrelations = []
        for lag in range(1, min(max_lags + 1, len(series))):
            autocorr = series.autocorr(lag=lag)
            autocorrelations.append({
                'lag': lag,
                'autocorrelation': autocorr,
                'significant': abs(autocorr) > 2 / np.sqrt(len(series))  # Rough significance
            })
        
        return pd.DataFrame(autocorrelations)
    
    # ========================================================================
    # Helper Methods
    # ========================================================================
    
    def _prepare_timeseries_data(self, data: Any, parameters: Dict[str, Any]) -> Tuple[pd.DataFrame, List[str]]:
        """Prepare data for time series operations."""
        transformations = []
        
        # Convert to DataFrame if needed
        if not isinstance(data, pd.DataFrame):
            if isinstance(data, pd.Series):
                data = data.to_frame()
                transformations.append("series_to_dataframe")
            else:
                data = pd.DataFrame(data)
                transformations.append("array_to_dataframe")
        
        # Ensure datetime index if specified
        datetime_col = parameters.get('datetime_column')
        if datetime_col and datetime_col in data.columns:
            data = data.set_index(datetime_col)
            transformations.append("set_datetime_index")
        
        return data, transformations
    
    def _detect_primary_library(self, function_name: str) -> str:
        """Detect which library will be used for a function."""
        if function_name in ['arima_forecast', 'seasonal_decomposition']:
            return "statsmodels" if self.is_library_available("statsmodels") else "fallback"
        elif function_name == 'prophet_forecast':
            return "prophet" if self.is_library_available("prophet") else "fallback"
        else:
            return "scipy" if self.is_library_available("scipy") else "pandas"
    
    def _get_integration_strategy(self, function_name: str) -> IntegrationStrategy:
        """Get integration strategy for a function."""
        if function_name in ['arima_forecast', 'prophet_forecast']:
            return IntegrationStrategy.SKLEARN_WRAPPER
        else:
            return IntegrationStrategy.FUNCTION_ADAPTER
    
    def _is_streaming_compatible(self, function_name: str) -> bool:
        """Check if function supports streaming execution."""
        streaming_functions = [
            'prepare_timeseries', 'handle_missing_values',
            'lag_features', 'rolling_features', 'time_features'
        ]
        return function_name in streaming_functions
    
    def _detect_seasonality(self, series: pd.Series) -> int:
        """Auto-detect seasonal period."""
        # Try common periods
        common_periods = [7, 12, 24, 365]
        
        best_period = 12  # Default
        best_score = 0
        
        for period in common_periods:
            if len(series) > 2 * period:
                autocorr = series.autocorr(lag=period)
                if abs(autocorr) > best_score:
                    best_score = abs(autocorr)
                    best_period = period
        
        return best_period
    
    def _statsmodels_exponential_smoothing(self, data: pd.DataFrame, **params) -> pd.DataFrame:
        """Exponential smoothing using statsmodels."""
        import statsmodels.api as sm
        
        target_col = params.get('target_column', data.columns[0])
        forecast_periods = params.get('forecast_periods', 30)
        
        model = sm.tsa.ExponentialSmoothing(data[target_col])
        fitted_model = model.fit()
        forecast = fitted_model.forecast(forecast_periods)
        
        forecast_index = pd.date_range(
            start=data.index[-1] + pd.Timedelta(days=1),
            periods=forecast_periods,
            freq=data.index.freq
        )
        
        return pd.DataFrame({
            'forecast': forecast
        }, index=forecast_index)
    
    def _simple_exponential_smoothing_fallback(self, data: pd.DataFrame, **params) -> pd.DataFrame:
        """Simple exponential smoothing fallback implementation."""
        target_col = params.get('target_column', data.columns[0])
        forecast_periods = params.get('forecast_periods', 30)
        alpha = params.get('alpha', 0.3)  # Smoothing parameter
        
        series = data[target_col].dropna()
        
        # Calculate exponentially weighted moving average
        smoothed = series.ewm(alpha=alpha).mean()
        
        # Simple forecast: repeat last smoothed value
        last_value = smoothed.iloc[-1]
        forecast_values = [last_value] * forecast_periods
        
        forecast_index = pd.date_range(
            start=data.index[-1] + pd.Timedelta(days=1),
            periods=forecast_periods,
            freq=data.index.freq
        )
        
        return pd.DataFrame({
            'forecast': forecast_values
        }, index=forecast_index)
    
    def _handle_timeseries_error(self, 
                                function_name: str,
                                data: Any, 
                                parameters: Dict[str, Any],
                                error: Exception) -> Tuple[Any, TimeSeriesMetadata]:
        """Handle time series operation errors with fallbacks."""
        logger.error(f"Time series operation {function_name} failed: {error}")
        
        # Attempt fallback for forecasting functions
        if function_name in ['arima_forecast', 'prophet_forecast'] and 'linear_trend' not in str(error):
            try:
                fallback_result = self.linear_trend_forecast(data, **parameters)
                
                metadata = TimeSeriesMetadata(
                    library_used="sklearn_fallback",
                    integration_strategy=IntegrationStrategy.FALLBACK_CHAIN,
                    fallback_used=True,
                    original_parameters=parameters
                )
                
                return fallback_result, metadata
            except Exception as fallback_error:
                logger.error(f"Fallback also failed: {fallback_error}")
        
        # Re-raise original error if no fallback available
        raise CompositionError(
            f"Time series operation {function_name} failed: {error}",
            error_type="timeseries_operation_failed"
        )


if __name__ == "__main__":
    # Example usage
    adapter = TimeSeriesAdapter()
    
    # Create sample time series data
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    values = np.cumsum(np.random.randn(100)) + 10  # Random walk with drift
    sample_data = pd.DataFrame({
        'datetime': dates,
        'value': values
    }).set_index('datetime')
    
    try:
        # Test time series preparation
        result, metadata = adapter.adapt_function_call(
            'prepare_timeseries',
            sample_data,
            {'target_column': 'value'}
        )
        print(f"Prepared time series with frequency: {result.index.freq}")
        print(f"Library used: {metadata.library_used}")
        
        # Test forecasting
        forecast_result, forecast_metadata = adapter.adapt_function_call(
            'linear_trend_forecast',
            result,
            {'target_column': 'value', 'forecast_periods': 10}
        )
        print(f"Generated forecast for {len(forecast_result)} periods")
        
    except Exception as e:
        print(f"Time series adapter test failed: {e}")