"""
Tests for Time Series Advanced Forecasting Methods.

This module tests the advanced forecasting capabilities including Prophet,
Exponential Smoothing, Ensemble methods, and forecast evaluation metrics.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
import warnings

from localdata_mcp.domains.time_series_analysis import (
    AdvancedForecastingTransformer,
    ProphetForecaster,
    ExponentialSmoothingForecaster,
    EnsembleForecaster,
    ForecastEvaluator,
    TimeSeriesValidationError
)


@pytest.fixture
def sample_time_series():
    """Create a sample time series with trend and seasonality."""
    dates = pd.date_range('2020-01-01', periods=200, freq='D')
    
    # Create data with trend and weekly seasonality
    trend = np.linspace(100, 150, 200)
    seasonal = 10 * np.sin(2 * np.pi * np.arange(200) / 7)
    noise = np.random.RandomState(42).normal(0, 5, 200)
    
    values = trend + seasonal + noise
    
    return pd.DataFrame({'value': values}, index=dates)


@pytest.fixture
def sample_forecasting_data():
    """Create sample data for forecasting tests with train/test split."""
    dates = pd.date_range('2020-01-01', periods=150, freq='D')
    
    # Simple trend with noise
    trend = np.linspace(100, 120, 150)
    noise = np.random.RandomState(42).normal(0, 3, 150)
    values = trend + noise
    
    train_data = pd.DataFrame({'value': values[:120]}, index=dates[:120])
    test_data = pd.Series(values[120:], index=dates[120:])
    
    return train_data, test_data


@pytest.fixture
def sample_evaluation_data():
    """Create sample data for evaluation tests."""
    np.random.seed(42)
    n_points = 50
    
    actual = np.random.normal(100, 10, n_points)
    predicted = actual + np.random.normal(0, 5, n_points)  # Add some forecast error
    historical = np.random.normal(95, 12, 100)  # For MASE calculation
    
    return actual, predicted, historical


class TestAdvancedForecastingTransformer:
    """Test the main AdvancedForecastingTransformer class."""
    
    def test_initialization(self):
        """Test transformer initialization with different parameters."""
        # Default initialization
        transformer = AdvancedForecastingTransformer()
        assert transformer.method == 'auto'
        assert transformer.forecast_steps == 10
        assert transformer.confidence_level == 0.95
        
        # Custom initialization
        transformer = AdvancedForecastingTransformer(
            method='exponential_smoothing',
            forecast_steps=20,
            confidence_level=0.90
        )
        assert transformer.method == 'exponential_smoothing'
        assert transformer.forecast_steps == 20
        assert transformer.confidence_level == 0.90
        
    def test_method_selection(self, sample_time_series):
        """Test automatic method selection based on data characteristics."""
        transformer = AdvancedForecastingTransformer(method='auto')
        
        # Test with small dataset - should select exponential_smoothing
        small_data = sample_time_series.iloc[:30]
        selected = transformer._select_method(small_data.iloc[:, 0])
        assert selected == 'exponential_smoothing'
        
        # Test with larger dataset - may select ensemble or other methods
        selected = transformer._select_method(sample_time_series.iloc[:, 0])
        assert selected in ['exponential_smoothing', 'prophet', 'ensemble']
        
    def test_seasonality_detection(self, sample_time_series):
        """Test seasonality detection in data."""
        transformer = AdvancedForecastingTransformer()
        
        # Test with seasonal data
        has_seasonality = transformer._detect_seasonality(sample_time_series.iloc[:, 0])
        assert isinstance(has_seasonality, bool)
        
        # Test with non-seasonal data
        random_data = pd.Series(np.random.random(100), 
                               index=pd.date_range('2020-01-01', periods=100))
        has_seasonality = transformer._detect_seasonality(random_data)
        assert isinstance(has_seasonality, bool)
        
    def test_fit_exponential_smoothing_method(self, sample_forecasting_data):
        """Test fitting with exponential smoothing method."""
        train_data, _ = sample_forecasting_data
        
        transformer = AdvancedForecastingTransformer(
            method='exponential_smoothing',
            forecast_steps=10
        )
        
        # Should fit without errors
        fitted_transformer = transformer.fit(train_data)
        assert fitted_transformer is transformer
        assert transformer.selected_method_ == 'exponential_smoothing'
        assert transformer.exponential_smoothing_forecaster_ is not None
        
    def test_fit_ensemble_method(self, sample_forecasting_data):
        """Test fitting with ensemble method."""
        train_data, _ = sample_forecasting_data
        
        transformer = AdvancedForecastingTransformer(
            method='ensemble',
            forecast_steps=10
        )
        
        # Should fit without errors
        fitted_transformer = transformer.fit(train_data)
        assert fitted_transformer is transformer
        assert transformer.selected_method_ == 'ensemble'
        assert transformer.ensemble_forecaster_ is not None
        
    def test_transform_generates_forecasts(self, sample_forecasting_data):
        """Test that transform generates forecasts correctly."""
        train_data, _ = sample_forecasting_data
        
        transformer = AdvancedForecastingTransformer(
            method='exponential_smoothing',
            forecast_steps=5
        )
        
        transformer.fit(train_data)
        result = transformer.transform(train_data)
        
        # Check result structure
        assert hasattr(result, 'data')
        assert 'forecast_values' in result.data
        assert 'selected_method' in result.data
        assert result.data['selected_method'] == 'exponential_smoothing'
        
        # Check forecast length
        forecast_values = result.data['forecast_values']
        assert len(forecast_values) == 5
        
    def test_validation_error_handling(self):
        """Test error handling for invalid inputs."""
        transformer = AdvancedForecastingTransformer()
        
        # Test with invalid data type
        with pytest.raises(TimeSeriesValidationError):
            transformer.fit("invalid_data")
            
        # Test with multivariate data
        multi_data = pd.DataFrame({
            'col1': np.random.random(50),
            'col2': np.random.random(50)
        }, index=pd.date_range('2020-01-01', periods=50))
        
        with pytest.raises(ValueError):
            transformer.fit(multi_data)


class TestProphetForecaster:
    """Test the ProphetForecaster class."""
    
    def test_initialization(self):
        """Test Prophet forecaster initialization."""
        forecaster = ProphetForecaster()
        assert forecaster.forecast_steps == 10
        assert forecaster.confidence_level == 0.95
        assert forecaster.growth == 'linear'
        assert forecaster.seasonality_mode == 'additive'
        
    def test_prophet_availability_check(self):
        """Test Prophet availability checking."""
        forecaster = ProphetForecaster()
        
        # The availability check should return a boolean
        availability = forecaster._check_prophet_availability()
        assert isinstance(availability, bool)
        
    def test_data_preparation(self, sample_time_series):
        """Test Prophet data format preparation."""
        forecaster = ProphetForecaster()
        data = sample_time_series.iloc[:, 0]
        
        prophet_data = forecaster._prepare_prophet_data(data)
        
        # Check required columns
        assert 'ds' in prophet_data.columns
        assert 'y' in prophet_data.columns
        assert len(prophet_data) == len(data)
        
    @patch('localdata_mcp.domains.time_series_analysis.Prophet')
    def test_fit_with_mock_prophet(self, mock_prophet, sample_forecasting_data):
        """Test fitting with mocked Prophet to avoid dependency issues."""
        train_data, _ = sample_forecasting_data
        
        # Mock Prophet class and instance
        mock_prophet_instance = MagicMock()
        mock_prophet.return_value = mock_prophet_instance
        
        forecaster = ProphetForecaster(forecast_steps=10)
        forecaster._prophet_available = True  # Force availability
        
        # Should fit without errors
        with patch('localdata_mcp.domains.time_series_analysis.logger'):
            fitted_forecaster = forecaster.fit(train_data)
            
        assert fitted_forecaster is forecaster
        assert forecaster.model_ is not None
        mock_prophet_instance.fit.assert_called_once()
        
    def test_prophet_not_available_error(self, sample_forecasting_data):
        """Test error when Prophet is not available."""
        train_data, _ = sample_forecasting_data
        
        forecaster = ProphetForecaster()
        forecaster._prophet_available = False  # Force unavailability
        
        with pytest.raises(ImportError, match="Prophet package is required"):
            forecaster.fit(train_data)
            
    def test_default_holidays_creation(self, sample_time_series):
        """Test default holidays creation."""
        forecaster = ProphetForecaster()
        data = sample_time_series.iloc[:, 0]
        
        holidays = forecaster._create_default_holidays(data)
        # Currently returns None, but test the method exists
        assert holidays is None
        
    def test_invalid_data_handling(self):
        """Test handling of invalid data formats."""
        forecaster = ProphetForecaster()
        
        # Test with multivariate data
        multi_data = pd.DataFrame({
            'col1': np.random.random(50),
            'col2': np.random.random(50)
        }, index=pd.date_range('2020-01-01', periods=50))
        
        with pytest.raises(ValueError, match="Prophet forecasting requires univariate"):
            forecaster.fit(multi_data)


class TestExponentialSmoothingForecaster:
    """Test the ExponentialSmoothingForecaster class."""
    
    def test_initialization(self):
        """Test Exponential Smoothing forecaster initialization."""
        forecaster = ExponentialSmoothingForecaster()
        assert forecaster.forecast_steps == 10
        assert forecaster.confidence_level == 0.95
        assert forecaster.trend == 'auto'
        assert forecaster.seasonal == 'auto'
        
    def test_seasonality_period_detection(self, sample_time_series):
        """Test detection of seasonality periods."""
        forecaster = ExponentialSmoothingForecaster()
        data = sample_time_series.iloc[:, 0]
        
        periods = forecaster._detect_seasonality_periods(data)
        assert isinstance(periods, int)
        assert periods > 0
        
    def test_model_component_selection(self, sample_time_series):
        """Test automatic selection of model components."""
        forecaster = ExponentialSmoothingForecaster()
        data = sample_time_series.iloc[:, 0]
        
        trend, seasonal, periods = forecaster._select_model_components(data)
        
        assert trend in [None, 'add', 'mul']
        assert seasonal in [None, 'add', 'mul']
        assert isinstance(periods, (int, type(None)))
        
    def test_fit_and_transform(self, sample_forecasting_data):
        """Test fitting and forecasting with Exponential Smoothing."""
        train_data, _ = sample_forecasting_data
        
        forecaster = ExponentialSmoothingForecaster(
            trend='add',
            seasonal=None,  # No seasonality for simple test
            forecast_steps=10
        )
        
        # Fit the model
        fitted_forecaster = forecaster.fit(train_data)
        assert fitted_forecaster is forecaster
        assert forecaster.fitted_model_ is not None
        
        # Generate forecasts
        result = forecaster.transform(train_data)
        
        # Check result structure
        assert hasattr(result, 'data')
        assert 'forecast_values' in result.data
        assert 'confidence_intervals' in result.data
        assert 'model_parameters' in result.data
        
        # Check forecast length
        forecast_values = result.data['forecast_values']
        assert len(forecast_values) == 10
        
    def test_auto_component_detection(self, sample_forecasting_data):
        """Test automatic component detection and fitting."""
        train_data, _ = sample_forecasting_data
        
        forecaster = ExponentialSmoothingForecaster(
            trend='auto',
            seasonal='auto',
            forecast_steps=5
        )
        
        # Should fit without errors even with auto detection
        fitted_forecaster = forecaster.fit(train_data)
        assert fitted_forecaster is forecaster
        
        # Check that components were selected
        assert forecaster.selected_trend_ in [None, 'add', 'mul']
        assert forecaster.selected_seasonal_ in [None, 'add', 'mul']
        
    def test_interpretation_generation(self, sample_forecasting_data):
        """Test interpretation text generation."""
        train_data, _ = sample_forecasting_data
        
        forecaster = ExponentialSmoothingForecaster(forecast_steps=5)
        forecaster.fit(train_data)
        result = forecaster.transform(train_data)
        
        interpretation = result.data['interpretation']
        assert isinstance(interpretation, str)
        assert len(interpretation) > 0
        assert 'forecast' in interpretation.lower()
        
    def test_recommendations_generation(self, sample_forecasting_data):
        """Test recommendations generation."""
        train_data, _ = sample_forecasting_data
        
        forecaster = ExponentialSmoothingForecaster(forecast_steps=5)
        forecaster.fit(train_data)
        result = forecaster.transform(train_data)
        
        recommendations = result.data['recommendations']
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        assert all(isinstance(rec, str) for rec in recommendations)


class TestEnsembleForecaster:
    """Test the EnsembleForecaster class."""
    
    def test_initialization(self):
        """Test Ensemble forecaster initialization."""
        forecaster = EnsembleForecaster()
        assert forecaster.forecast_steps == 10
        assert forecaster.confidence_level == 0.95
        assert 'exponential_smoothing' in forecaster.methods
        assert 'arima' in forecaster.methods
        assert forecaster.combination_method == 'weighted_average'
        
    def test_data_splitting(self, sample_forecasting_data):
        """Test train/validation data splitting."""
        train_data, _ = sample_forecasting_data
        forecaster = EnsembleForecaster(validation_split=0.2)
        
        data_series = train_data.iloc[:, 0]
        train_split, val_split = forecaster._split_data(data_series)
        
        # Check splits
        assert len(train_split) + len(val_split) == len(data_series)
        assert len(val_split) == int(len(data_series) * 0.2)
        
    def test_model_initialization(self, sample_forecasting_data):
        """Test initialization of individual models in ensemble."""
        train_data, _ = sample_forecasting_data
        
        forecaster = EnsembleForecaster(
            methods=['exponential_smoothing', 'arima']
        )
        forecaster.validation_data_ = train_data.iloc[-10:, 0]  # Mock validation data
        
        models = forecaster._initialize_models()
        
        assert 'exponential_smoothing' in models
        assert 'arima' in models
        assert len(models) == 2
        
    def test_weights_computation(self):
        """Test ensemble weights computation."""
        forecaster = EnsembleForecaster()
        
        # Test with provided weights
        forecaster.weights = {'method1': 0.6, 'method2': 0.4}
        weights = forecaster._compute_weights()
        assert abs(sum(weights.values()) - 1.0) < 1e-10  # Should sum to 1
        
        # Test with performance-based weights
        forecaster.weights = {}
        forecaster.model_performance_ = {
            'method1': {'score': 0.8},
            'method2': {'score': 0.6}
        }
        weights = forecaster._compute_weights()
        assert abs(sum(weights.values()) - 1.0) < 1e-10
        assert weights['method1'] > weights['method2']  # Better score = higher weight
        
    def test_fit_with_limited_data(self, sample_forecasting_data):
        """Test fitting with limited data (no validation split)."""
        train_data, _ = sample_forecasting_data
        small_data = train_data.iloc[:30]  # Small dataset
        
        forecaster = EnsembleForecaster(
            methods=['exponential_smoothing'],  # Only one method for simplicity
            forecast_steps=5
        )
        
        fitted_forecaster = forecaster.fit(small_data)
        assert fitted_forecaster is forecaster
        assert len(forecaster.fitted_models_) > 0
        
    def test_forecast_combination_methods(self, sample_evaluation_data):
        """Test different forecast combination methods."""
        forecaster = EnsembleForecaster()
        
        # Create sample forecasts
        forecasts = {
            'method1': pd.Series([100, 105, 110]),
            'method2': pd.Series([98, 103, 108]),
            'method3': pd.Series([102, 107, 112])
        }
        
        forecaster.computed_weights_ = {'method1': 0.5, 'method2': 0.3, 'method3': 0.2}
        
        # Test weighted average
        weighted_avg = forecaster._weighted_average_combination(forecasts)
        assert len(weighted_avg) == 3
        assert isinstance(weighted_avg, pd.Series)
        
        # Test median combination
        median_combo = forecaster._median_combination(forecasts)
        assert len(median_combo) == 3
        assert isinstance(median_combo, pd.Series)
        
        # Test best performer combination
        forecaster.model_performance_ = {
            'method1': {'score': 0.8},
            'method2': {'score': 0.6},
            'method3': {'score': 0.9}
        }
        best_combo = forecaster._best_performer_combination(forecasts)
        assert len(best_combo) == 3
        assert isinstance(best_combo, pd.Series)
        
    def test_confidence_intervals_combination(self):
        """Test combination of confidence intervals from multiple models."""
        forecaster = EnsembleForecaster()
        
        # Create sample intervals
        intervals = {
            'method1': pd.DataFrame({'lower': [95, 100, 105], 'upper': [105, 110, 115]}),
            'method2': pd.DataFrame({'lower': [93, 98, 103], 'upper': [107, 112, 117]})
        }
        
        combined = forecaster._combine_confidence_intervals(intervals)
        
        assert combined is not None
        assert 'lower' in combined.columns
        assert 'upper' in combined.columns
        assert len(combined) == 3


class TestForecastEvaluator:
    """Test the ForecastEvaluator class."""
    
    def test_initialization(self):
        """Test evaluator initialization with different metrics."""
        # Default initialization
        evaluator = ForecastEvaluator()
        assert 'mae' in evaluator.metrics
        assert 'mape' in evaluator.metrics
        assert 'rmse' in evaluator.metrics
        
        # Custom metrics
        evaluator = ForecastEvaluator(metrics=['mae', 'rmse'])
        assert evaluator.metrics == ['mae', 'rmse']
        
    def test_basic_metrics_calculation(self, sample_evaluation_data):
        """Test calculation of basic evaluation metrics."""
        actual, predicted, historical = sample_evaluation_data
        evaluator = ForecastEvaluator()
        
        metrics = evaluator.evaluate_forecast(actual, predicted, historical)
        
        # Check that all requested metrics are present
        assert 'mae' in metrics
        assert 'mape' in metrics
        assert 'rmse' in metrics
        assert 'mase' in metrics
        
        # Check metric values are reasonable
        assert metrics['mae'] >= 0
        assert metrics['rmse'] >= 0
        assert metrics['mape'] >= 0
        
    def test_mae_calculation(self):
        """Test Mean Absolute Error calculation."""
        evaluator = ForecastEvaluator()
        
        actual = np.array([100, 110, 120])
        predicted = np.array([105, 108, 125])
        
        mae = evaluator._calculate_mae(actual, predicted)
        expected_mae = np.mean([5, 2, 5])  # |100-105|, |110-108|, |120-125|
        assert abs(mae - expected_mae) < 1e-10
        
    def test_mape_calculation(self):
        """Test Mean Absolute Percentage Error calculation."""
        evaluator = ForecastEvaluator()
        
        actual = np.array([100, 200, 50])
        predicted = np.array([110, 180, 55])
        
        mape = evaluator._calculate_mape(actual, predicted)
        expected_mape = np.mean([10, 10, 10])  # 10%, 10%, 10%
        assert abs(mape - expected_mape) < 1e-10
        
    def test_mape_with_zero_values(self):
        """Test MAPE calculation with zero actual values."""
        evaluator = ForecastEvaluator()
        
        actual = np.array([0, 100, 200])
        predicted = np.array([5, 110, 180])
        
        mape = evaluator._calculate_mape(actual, predicted)
        # Should handle zero values and calculate MAPE for non-zero values only
        expected_mape = np.mean([10, 10])  # Only for non-zero actuals
        assert abs(mape - expected_mape) < 1e-10
        
    def test_rmse_calculation(self):
        """Test Root Mean Square Error calculation."""
        evaluator = ForecastEvaluator()
        
        actual = np.array([100, 110, 120])
        predicted = np.array([103, 107, 123])
        
        rmse = evaluator._calculate_rmse(actual, predicted)
        expected_rmse = np.sqrt(np.mean([9, 9, 9]))  # sqrt(mean([3^2, 3^2, 3^2]))
        assert abs(rmse - expected_rmse) < 1e-10
        
    def test_mase_calculation(self):
        """Test Mean Absolute Scaled Error calculation."""
        evaluator = ForecastEvaluator(seasonal_period=1)
        
        actual = np.array([100, 110, 120])
        predicted = np.array([105, 108, 125])
        historical = np.array([90, 95, 100, 105])  # For naive forecast baseline
        
        mase = evaluator._calculate_mase(actual, predicted, historical)
        assert isinstance(mase, float)
        assert mase >= 0  # MASE should be non-negative
        
    def test_directional_accuracy_calculation(self):
        """Test directional accuracy calculation."""
        evaluator = ForecastEvaluator()
        
        # Perfect directional accuracy
        actual = np.array([100, 110, 120, 115])
        predicted = np.array([105, 115, 125, 120])  # Same direction changes
        
        dir_acc = evaluator._calculate_directional_accuracy(actual, predicted)
        assert dir_acc == 100.0  # Perfect directional accuracy
        
        # Poor directional accuracy
        actual = np.array([100, 110, 120, 115])
        predicted = np.array([105, 100, 95, 120])  # Opposite directions
        
        dir_acc = evaluator._calculate_directional_accuracy(actual, predicted)
        assert dir_acc < 100.0
        
    def test_evaluation_report_creation(self, sample_evaluation_data):
        """Test comprehensive evaluation report creation."""
        actual, predicted, historical = sample_evaluation_data
        evaluator = ForecastEvaluator()
        
        report = evaluator.create_evaluation_report(
            actual, predicted, historical, model_name="Test Model"
        )
        
        # Check report structure
        assert 'model_name' in report
        assert 'evaluation_metrics' in report
        assert 'performance_category' in report
        assert 'interpretation' in report
        assert 'recommendations' in report
        assert 'timestamp' in report
        
        assert report['model_name'] == "Test Model"
        assert isinstance(report['recommendations'], list)
        
    def test_performance_categorization(self):
        """Test performance category assignment."""
        evaluator = ForecastEvaluator()
        
        # Excellent performance
        metrics = {'mape': 3.0}
        category = evaluator._categorize_performance(metrics)
        assert category == 'excellent'
        
        # Good performance
        metrics = {'mape': 12.0}
        category = evaluator._categorize_performance(metrics)
        assert category == 'good'
        
        # Moderate performance
        metrics = {'mape': 20.0}
        category = evaluator._categorize_performance(metrics)
        assert category == 'moderate'
        
        # Poor performance
        metrics = {'mape': 40.0}
        category = evaluator._categorize_performance(metrics)
        assert category == 'poor'
        
    def test_edge_cases(self):
        """Test edge cases in evaluation."""
        evaluator = ForecastEvaluator()
        
        # Test with identical values
        actual = np.array([100, 100, 100])
        predicted = np.array([100, 100, 100])
        
        metrics = evaluator.evaluate_forecast(actual, predicted)
        assert metrics['mae'] == 0.0
        assert metrics['mape'] == 0.0
        assert metrics['rmse'] == 0.0
        
        # Test with single data point
        actual = np.array([100])
        predicted = np.array([105])
        
        metrics = evaluator.evaluate_forecast(actual, predicted)
        assert metrics['mae'] == 5.0
        assert metrics['n_observations'] == 1
        
    def test_empty_data_handling(self):
        """Test handling of empty data."""
        evaluator = ForecastEvaluator()
        
        with pytest.raises(ValueError, match="No data points to evaluate"):
            evaluator.evaluate_forecast([], [])


class TestIntegration:
    """Integration tests for advanced forecasting components."""
    
    def test_advanced_forecasting_with_exponential_smoothing(self, sample_forecasting_data):
        """Test end-to-end advanced forecasting with exponential smoothing."""
        train_data, test_data = sample_forecasting_data
        
        # Create and fit forecaster
        forecaster = AdvancedForecastingTransformer(
            method='exponential_smoothing',
            forecast_steps=len(test_data),
            confidence_level=0.95
        )
        
        forecaster.fit(train_data)
        result = forecaster.transform(train_data)
        
        # Evaluate forecasts
        forecast_values = pd.Series(result.data['forecast_values'])
        evaluator = ForecastEvaluator()
        
        metrics = evaluator.evaluate_forecast(test_data.values, forecast_values.values)
        
        # Basic checks
        assert metrics['mae'] >= 0
        assert metrics['rmse'] >= 0
        assert len(forecast_values) == len(test_data)
        
    def test_ensemble_forecasting_integration(self, sample_forecasting_data):
        """Test end-to-end ensemble forecasting."""
        train_data, test_data = sample_forecasting_data
        
        # Create ensemble with limited methods to ensure compatibility
        forecaster = EnsembleForecaster(
            methods=['exponential_smoothing'],  # Single method for reliable test
            forecast_steps=10,
            validation_split=0.1
        )
        
        forecaster.fit(train_data)
        result = forecaster.transform(train_data)
        
        # Check result structure
        assert 'ensemble_details' in result.data
        assert 'methods' in result.data['ensemble_details']
        assert 'weights' in result.data['ensemble_details']
        assert len(result.data['forecast_values']) == 10
        
    def test_evaluation_with_real_forecasts(self, sample_forecasting_data):
        """Test evaluation with actual forecasting results."""
        train_data, test_data = sample_forecasting_data
        
        # Generate forecasts
        forecaster = ExponentialSmoothingForecaster(
            forecast_steps=len(test_data),
            trend='add',
            seasonal=None
        )
        
        forecaster.fit(train_data)
        result = forecaster.transform(train_data)
        
        # Evaluate
        forecast_values = pd.Series(result.data['forecast_values'])
        evaluator = ForecastEvaluator()
        
        report = evaluator.create_evaluation_report(
            test_data.values,
            forecast_values.values,
            train_data.values.flatten(),
            model_name="Exponential Smoothing"
        )
        
        # Verify report completeness
        assert report['model_name'] == "Exponential Smoothing"
        assert 'mae' in report['evaluation_metrics']
        assert len(report['recommendations']) > 0
        assert report['performance_category'] in ['excellent', 'good', 'moderate', 'poor']


if __name__ == "__main__":
    pytest.main([__file__])