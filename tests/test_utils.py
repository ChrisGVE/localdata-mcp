"""
Test utilities for Time Series Integration Testing.

This module provides comprehensive utilities for creating synthetic time series data,
validating results, measuring performance, and supporting integration tests.
"""

import numpy as np
import pandas as pd
import sqlite3
import tempfile
import time
import json
from typing import Dict, Any, List, Tuple, Optional, Union
from pathlib import Path
from datetime import datetime, timedelta


def create_synthetic_time_series(
    start_date: str = '2020-01-01',
    end_date: str = '2023-12-31', 
    freq: str = 'D',
    trend: float = 0.0,
    seasonality: bool = True,
    seasonal_periods: int = 365,
    noise_level: float = 0.1,
    anomalies: float = 0.02,
    change_points: Optional[List[Tuple[str, str]]] = None
) -> pd.DataFrame:
    """
    Create synthetic time series data with configurable characteristics.
    
    Parameters:
    -----------
    start_date : str
        Start date for the time series
    end_date : str
        End date for the time series  
    freq : str
        Frequency of the time series ('D', 'H', 'M', etc.)
    trend : float
        Linear trend component (slope per period)
    seasonality : bool
        Whether to include seasonal component
    seasonal_periods : int
        Number of periods in a season
    noise_level : float
        Standard deviation of random noise
    anomalies : float
        Proportion of points to make anomalous (0.0-1.0)
    change_points : List[Tuple[str, str]], optional
        List of (date, type) tuples for change points.
        Types: 'level', 'trend', 'variance'
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with 'date' and 'value' columns
    """
    
    # Generate date range
    date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
    n_points = len(date_range)
    
    # Initialize time series components
    trend_component = np.arange(n_points) * trend
    seasonal_component = np.zeros(n_points)
    noise_component = np.random.normal(0, noise_level, n_points)
    
    # Add seasonality
    if seasonality and seasonal_periods > 0:
        seasonal_amplitude = 2.0
        seasonal_component = seasonal_amplitude * np.sin(
            2 * np.pi * np.arange(n_points) / seasonal_periods
        )
        
        # Add harmonic for more realistic seasonality
        if seasonal_periods >= 12:
            seasonal_component += 0.5 * seasonal_amplitude * np.sin(
                4 * np.pi * np.arange(n_points) / seasonal_periods
            )
    
    # Combine components
    values = 100 + trend_component + seasonal_component + noise_component
    
    # Apply change points
    if change_points:
        for change_date, change_type in change_points:
            change_idx = date_range.get_loc(pd.to_datetime(change_date))
            
            if change_type == 'level':
                # Level shift
                values[change_idx:] += np.random.uniform(5, 15)
            elif change_type == 'trend':
                # Trend change
                new_trend = trend + np.random.uniform(-0.2, 0.2)
                trend_adjustment = np.arange(len(values) - change_idx) * new_trend
                values[change_idx:] += trend_adjustment
            elif change_type == 'variance':
                # Variance change
                new_noise = noise_level * np.random.uniform(2, 5)
                values[change_idx:] += np.random.normal(0, new_noise, len(values) - change_idx)
    
    # Add anomalies
    if anomalies > 0:
        n_anomalies = int(n_points * anomalies)
        anomaly_indices = np.random.choice(n_points, size=n_anomalies, replace=False)
        
        for idx in anomaly_indices:
            # Create different types of anomalies
            anomaly_type = np.random.choice(['spike', 'dip', 'outlier'])
            
            if anomaly_type == 'spike':
                values[idx] += np.random.uniform(3, 8) * noise_level * 10
            elif anomaly_type == 'dip':
                values[idx] -= np.random.uniform(3, 8) * noise_level * 10
            else:  # outlier
                values[idx] = np.random.uniform(values.min() - 20, values.max() + 20)
    
    # Create DataFrame
    ts_data = pd.DataFrame({
        'date': date_range,
        'value': values
    })
    
    return ts_data


def create_multivariate_time_series(
    start_date: str = '2020-01-01',
    end_date: str = '2023-12-31',
    freq: str = 'D',
    n_variables: int = 4,
    cross_correlations: bool = True,
    causality_patterns: bool = True
) -> pd.DataFrame:
    """
    Create synthetic multivariate time series with cross-correlations and causality.
    
    Parameters:
    -----------
    start_date : str
        Start date for the time series
    end_date : str
        End date for the time series
    freq : str
        Frequency of the time series
    n_variables : int
        Number of variables to generate
    cross_correlations : bool
        Whether to include cross-correlations between variables
    causality_patterns : bool
        Whether to include Granger causality patterns
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with 'date' column and multiple variable columns
    """
    
    # Generate date range
    date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
    n_points = len(date_range)
    
    # Initialize variables
    variables = {}
    
    # Generate base variable (independent)
    variables['var1'] = create_synthetic_time_series(
        start_date, end_date, freq,
        trend=0.05, seasonality=True, seasonal_periods=365,
        noise_level=0.1, anomalies=0.01
    )['value'].values
    
    # Generate correlated variables
    for i in range(2, n_variables + 1):
        var_name = f'var{i}'
        
        # Base independent component
        base_component = create_synthetic_time_series(
            start_date, end_date, freq,
            trend=np.random.uniform(-0.02, 0.08),
            seasonality=True, seasonal_periods=365,
            noise_level=0.15, anomalies=0.01
        )['value'].values
        
        variable_values = base_component.copy()
        
        # Add cross-correlation with previous variables
        if cross_correlations:
            for j in range(1, i):
                prev_var = variables[f'var{j}']
                correlation_strength = np.random.uniform(0.2, 0.7)
                
                # Add lagged correlation
                lag = np.random.choice([0, 1, 2])
                if lag > 0:
                    lagged_var = np.concatenate([np.zeros(lag), prev_var[:-lag]])
                else:
                    lagged_var = prev_var
                
                variable_values += correlation_strength * lagged_var
        
        # Add causality patterns (Granger causality)
        if causality_patterns and i > 2:
            # Variable i is caused by variable i-1 with some lags
            causing_var = variables[f'var{i-1}']
            causality_strength = np.random.uniform(0.3, 0.8)
            
            # Add multiple lags for causality
            for lag in [1, 2]:
                if lag < len(causing_var):
                    lagged_causing = np.concatenate([np.zeros(lag), causing_var[:-lag]])
                    variable_values += causality_strength * lagged_causing / (lag + 1)
        
        variables[var_name] = variable_values
    
    # Create DataFrame
    multi_data = pd.DataFrame({'date': date_range})
    for var_name, var_values in variables.items():
        multi_data[var_name] = var_values
    
    return multi_data


def generate_high_frequency_data(
    start_date: str = '2023-01-01',
    end_date: str = '2023-12-31',
    freq: str = 'H',
    n_series: int = 1,
    complexity: str = 'medium'
) -> pd.DataFrame:
    """
    Generate high-frequency time series data for performance testing.
    
    Parameters:
    -----------
    start_date : str
        Start date for the time series
    end_date : str
        End date for the time series
    freq : str
        Frequency ('H' for hourly, '15min' for 15-minute, etc.)
    n_series : int
        Number of time series to generate
    complexity : str
        Complexity level ('low', 'medium', 'high')
        
    Returns:
    --------
    pd.DataFrame
        High-frequency time series data
    """
    
    # Set complexity parameters
    complexity_params = {
        'low': {'noise': 0.05, 'seasonal_components': 1, 'anomaly_rate': 0.001},
        'medium': {'noise': 0.1, 'seasonal_components': 2, 'anomaly_rate': 0.005},
        'high': {'noise': 0.15, 'seasonal_components': 3, 'anomaly_rate': 0.01}
    }
    
    params = complexity_params.get(complexity, complexity_params['medium'])
    
    # Generate date range
    date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
    n_points = len(date_range)
    
    # Generate base time series
    if n_series == 1:
        # Single series
        values = np.random.randn(n_points).cumsum() * params['noise']
        
        # Add multiple seasonal components for complexity
        for i in range(params['seasonal_components']):
            period = np.random.choice([24, 168, 24*7, 24*30])  # Hourly, weekly, monthly
            amplitude = np.random.uniform(1, 3)
            phase = np.random.uniform(0, 2*np.pi)
            
            seasonal = amplitude * np.sin(
                2 * np.pi * np.arange(n_points) / period + phase
            )
            values += seasonal
        
        # Add trend
        trend = np.random.uniform(-0.001, 0.001)
        values += np.arange(n_points) * trend
        
        # Add anomalies
        n_anomalies = int(n_points * params['anomaly_rate'])
        if n_anomalies > 0:
            anomaly_indices = np.random.choice(n_points, size=n_anomalies, replace=False)
            anomaly_magnitudes = np.random.uniform(5, 15, size=n_anomalies)
            anomaly_signs = np.random.choice([-1, 1], size=n_anomalies)
            values[anomaly_indices] += anomaly_magnitudes * anomaly_signs
        
        return pd.DataFrame({
            'date': date_range,
            'value': values
        })
    
    else:
        # Multiple series (for multivariate analysis)
        data = {'date': date_range}
        
        for series_idx in range(n_series):
            values = np.random.randn(n_points).cumsum() * params['noise']
            
            # Add complexity...
            for i in range(params['seasonal_components']):
                period = np.random.choice([24, 168, 24*7])
                amplitude = np.random.uniform(0.5, 2)
                phase = np.random.uniform(0, 2*np.pi)
                
                seasonal = amplitude * np.sin(
                    2 * np.pi * np.arange(n_points) / period + phase
                )
                values += seasonal
            
            data[f'series_{series_idx+1}'] = values
        
        return pd.DataFrame(data)


def validate_time_series_result(result: Dict[str, Any], expected_type: str) -> bool:
    """
    Validate time series analysis results.
    
    Parameters:
    -----------
    result : Dict[str, Any]
        Result dictionary from time series analysis
    expected_type : str
        Expected analysis type ('basic', 'forecast', 'anomaly', 'changepoint')
        
    Returns:
    --------
    bool
        Whether the result is valid
    """
    
    if 'error' in result:
        return False
    
    # Basic validation
    if expected_type == 'basic':
        required_keys = ['analysis_type', 'trend_analysis', 'seasonality_analysis', 
                        'stationarity_tests', 'descriptive_stats']
        return all(key in result for key in required_keys)
    
    elif expected_type == 'forecast':
        required_keys = ['model_type', 'forecast_values', 'forecast_index', 'model_metrics']
        return all(key in result for key in required_keys)
    
    elif expected_type == 'anomaly':
        required_keys = ['detection_method', 'anomaly_indices', 'anomaly_scores']
        return all(key in result for key in required_keys)
    
    elif expected_type == 'changepoint':
        required_keys = ['detection_method', 'change_points', 'segments_analysis']
        return all(key in result for key in required_keys)
    
    return False


def measure_performance(func, *args, **kwargs) -> Tuple[Any, float]:
    """
    Measure execution time of a function.
    
    Parameters:
    -----------
    func : callable
        Function to measure
    *args, **kwargs
        Arguments to pass to the function
        
    Returns:
    --------
    Tuple[Any, float]
        Function result and execution time in seconds
    """
    start_time = time.time()
    result = func(*args, **kwargs)
    execution_time = time.time() - start_time
    
    return result, execution_time


def create_test_database(data: pd.DataFrame, db_path: str, table_name: str = 'data') -> str:
    """
    Create a test SQLite database from DataFrame.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Data to store in database
    db_path : str
        Path for the database file
    table_name : str
        Name of the table to create
        
    Returns:
    --------
    str
        Path to the created database
    """
    
    with sqlite3.connect(db_path) as conn:
        data.to_sql(table_name, conn, index=False, if_exists='replace')
    
    return db_path


def generate_performance_report(performance_metrics: Dict[str, Any]) -> str:
    """
    Generate a formatted performance report.
    
    Parameters:
    -----------
    performance_metrics : Dict[str, Any]
        Dictionary of performance metrics
        
    Returns:
    --------
    str
        Formatted performance report
    """
    
    report = "📊 TIME SERIES PERFORMANCE REPORT\n"
    report += "=" * 50 + "\n\n"
    
    for category, metrics in performance_metrics.items():
        report += f"🔹 {category.upper().replace('_', ' ')}\n"
        report += "-" * 30 + "\n"
        
        if isinstance(metrics, dict):
            for metric_name, value in metrics.items():
                if isinstance(value, float):
                    if 'time' in metric_name.lower():
                        report += f"  {metric_name}: {value:.3f}s\n"
                    elif 'memory' in metric_name.lower():
                        report += f"  {metric_name}: {value:.1f}MB\n"
                    else:
                        report += f"  {metric_name}: {value:.4f}\n"
                else:
                    report += f"  {metric_name}: {value}\n"
        else:
            report += f"  Value: {metrics}\n"
        
        report += "\n"
    
    return report


def create_edge_case_data(case_type: str) -> pd.DataFrame:
    """
    Create edge case datasets for testing.
    
    Parameters:
    -----------
    case_type : str
        Type of edge case ('short', 'missing', 'constant', 'extreme_outliers')
        
    Returns:
    --------
    pd.DataFrame
        Edge case dataset
    """
    
    if case_type == 'short':
        # Very short time series
        return pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=5, freq='D'),
            'value': [1, 2, 1.5, 2.5, 2]
        })
    
    elif case_type == 'missing':
        # Time series with missing values
        dates = pd.date_range('2023-01-01', periods=50, freq='D')
        values = np.random.randn(50).cumsum()
        
        # Introduce missing values
        missing_indices = np.random.choice(50, size=10, replace=False)
        values[missing_indices] = np.nan
        
        return pd.DataFrame({
            'date': dates,
            'value': values
        })
    
    elif case_type == 'constant':
        # Constant time series
        return pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=100, freq='D'),
            'value': np.full(100, 42.0)
        })
    
    elif case_type == 'extreme_outliers':
        # Time series with extreme outliers
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        values = np.random.randn(100)
        
        # Add extreme outliers
        outlier_indices = [10, 30, 70, 90]
        values[outlier_indices] = [1000, -1000, 2000, -1500]
        
        return pd.DataFrame({
            'date': dates,
            'value': values
        })
    
    else:
        raise ValueError(f"Unknown edge case type: {case_type}")


def validate_forecasting_quality(
    actual: np.ndarray, 
    forecast: np.ndarray,
    tolerance: float = 0.3
) -> Dict[str, Any]:
    """
    Validate forecasting quality with basic metrics.
    
    Parameters:
    -----------
    actual : np.ndarray
        Actual values (for validation period)
    forecast : np.ndarray
        Forecasted values
    tolerance : float
        Tolerance level for MAPE (Mean Absolute Percentage Error)
        
    Returns:
    --------
    Dict[str, Any]
        Quality metrics
    """
    
    if len(actual) != len(forecast):
        return {'error': 'Length mismatch between actual and forecast'}
    
    # Calculate basic metrics
    mae = np.mean(np.abs(actual - forecast))
    rmse = np.sqrt(np.mean((actual - forecast) ** 2))
    
    # Handle division by zero for MAPE
    mape = np.mean(np.abs((actual - forecast) / np.where(actual != 0, actual, 1e-10))) * 100
    
    # Directional accuracy (for trend prediction)
    actual_direction = np.diff(actual) > 0
    forecast_direction = np.diff(forecast) > 0
    directional_accuracy = np.mean(actual_direction == forecast_direction) * 100 if len(actual) > 1 else 0
    
    quality_metrics = {
        'mae': float(mae),
        'rmse': float(rmse),
        'mape': float(mape),
        'directional_accuracy': float(directional_accuracy),
        'is_acceptable': mape < (tolerance * 100),
        'forecast_range': {
            'min': float(np.min(forecast)),
            'max': float(np.max(forecast)),
            'mean': float(np.mean(forecast))
        }
    }
    
    return quality_metrics


class TimeSeriesTestFixture:
    """Fixture class for comprehensive time series testing."""
    
    def __init__(self, tmp_dir: Path):
        self.tmp_dir = Path(tmp_dir)
        self.databases = {}
        self.datasets = {}
        
    def create_standard_datasets(self):
        """Create standard test datasets."""
        
        # Daily time series with trend and seasonality
        self.datasets['daily'] = create_synthetic_time_series(
            start_date='2020-01-01',
            end_date='2023-12-31', 
            freq='D',
            trend=0.1,
            seasonality=True,
            seasonal_periods=365,
            noise_level=0.1,
            anomalies=0.02
        )
        
        # Hourly high-frequency data
        self.datasets['hourly'] = generate_high_frequency_data(
            start_date='2023-01-01',
            end_date='2023-06-30',
            freq='H',
            n_series=1,
            complexity='high'
        )
        
        # Multivariate time series
        self.datasets['multivariate'] = create_multivariate_time_series(
            start_date='2022-01-01',
            end_date='2023-12-31',
            freq='D',
            n_variables=4,
            cross_correlations=True,
            causality_patterns=True
        )
        
        # Create databases
        for name, dataset in self.datasets.items():
            db_path = self.tmp_dir / f"{name}_data.db"
            create_test_database(dataset, str(db_path), f"{name}_table")
            self.databases[name] = str(db_path)
    
    def get_database_path(self, dataset_name: str) -> str:
        """Get database path for a dataset."""
        return self.databases.get(dataset_name)
    
    def get_dataset(self, dataset_name: str) -> pd.DataFrame:
        """Get a dataset by name."""
        return self.datasets.get(dataset_name)
    
    def cleanup(self):
        """Clean up temporary files."""
        import os
        for db_path in self.databases.values():
            if os.path.exists(db_path):
                os.unlink(db_path)


def assert_performance_within_limits(
    metrics: Dict[str, float],
    limits: Dict[str, float]
) -> None:
    """
    Assert that performance metrics are within specified limits.
    
    Parameters:
    -----------
    metrics : Dict[str, float]
        Performance metrics
    limits : Dict[str, float]
        Performance limits
        
    Raises:
    -------
    AssertionError
        If any metric exceeds its limit
    """
    
    for metric_name, limit in limits.items():
        if metric_name in metrics:
            actual_value = metrics[metric_name]
            assert actual_value <= limit, (
                f"Performance limit exceeded for {metric_name}: "
                f"{actual_value:.3f} > {limit:.3f}"
            )


def generate_test_summary(test_results: Dict[str, Any]) -> str:
    """
    Generate a comprehensive test summary report.
    
    Parameters:
    -----------
    test_results : Dict[str, Any]
        Test results dictionary
        
    Returns:
    --------
    str
        Formatted test summary
    """
    
    summary = "🧪 TIME SERIES INTEGRATION TEST SUMMARY\n"
    summary += "=" * 60 + "\n\n"
    
    total_tests = sum(1 for v in test_results.values() if isinstance(v, dict) and 'status' in v)
    passed_tests = sum(1 for v in test_results.values() 
                      if isinstance(v, dict) and v.get('status') == 'passed')
    
    summary += f"📊 Overall Results: {passed_tests}/{total_tests} tests passed\n"
    summary += f"✅ Success Rate: {(passed_tests/total_tests)*100:.1f}%\n\n"
    
    # Detailed results by category
    categories = {}
    for test_name, result in test_results.items():
        category = test_name.split('_')[0] if '_' in test_name else 'general'
        if category not in categories:
            categories[category] = []
        categories[category].append((test_name, result))
    
    for category, tests in categories.items():
        summary += f"🔹 {category.upper()}\n"
        summary += "-" * 30 + "\n"
        
        for test_name, result in tests:
            if isinstance(result, dict):
                status = result.get('status', 'unknown')
                duration = result.get('duration', 'N/A')
                
                status_icon = "✅" if status == 'passed' else "❌"
                summary += f"  {status_icon} {test_name}: {status}"
                
                if isinstance(duration, (int, float)):
                    summary += f" ({duration:.2f}s)"
                summary += "\n"
            else:
                summary += f"  ℹ️  {test_name}: {result}\n"
        
        summary += "\n"
    
    return summary