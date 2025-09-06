"""
Unit tests for Pre-built Domain Shims in LocalData MCP v2.0 Integration Framework.

This test module validates the functionality of domain-specific shims including
StatisticalShim, RegressionShim, TimeSeriesShim, and PatternRecognitionShim.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
from dataclasses import dataclass
from typing import Any, Dict, List

from src.localdata_mcp.pipeline.integration.domain_shims import (
    BaseDomainShim, StatisticalShim, RegressionShim, TimeSeriesShim, 
    PatternRecognitionShim, DomainShimType, DomainMapping, SemanticContext,
    create_statistical_shim, create_regression_shim, create_time_series_shim,
    create_pattern_recognition_shim, create_all_domain_shims,
    get_compatible_domain_shims, validate_domain_shim_configuration
)
from src.localdata_mcp.pipeline.integration.interfaces import (
    DataFormat, ConversionRequest, ConversionResult, ConversionContext,
    ConversionError
)
from src.localdata_mcp.pipeline.integration.shim_registry import AdapterConfig


class TestBaseDomainShim:
    """Test the base domain shim functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = AdapterConfig(adapter_id="test_shim")
        
        # Create a concrete implementation for testing
        class TestDomainShim(BaseDomainShim):
            def _initialize_domain_knowledge(self):
                self._domain_schemas = {"test": {"required": ["field"]}}
            
            def _load_domain_mappings(self):
                self.supported_mappings.append(DomainMapping(
                    source_domain='test_source',
                    target_domain='test_target',
                    parameter_mappings={'param1': 'param2'},
                    quality_preservation=0.95
                ))
            
            def _perform_domain_conversion(self, request, mapping, context):
                return {"converted": True, "data": request.source_data}
            
            def _normalize_results(self, data, mapping, context):
                data['normalized'] = True
                return data
        
        self.test_shim = TestDomainShim(
            adapter_id="test_shim",
            domain_type=DomainShimType.STATISTICAL,
            config=self.config
        )
    
    def test_initialization(self):
        """Test base domain shim initialization."""
        assert self.test_shim.adapter_id == "test_shim"
        assert self.test_shim.domain_type == DomainShimType.STATISTICAL
        assert self.test_shim.config == self.config
        assert hasattr(self.test_shim, '_pandas_converter')
        assert hasattr(self.test_shim, '_numpy_converter')
    
    def test_extract_domain_from_format(self):
        """Test domain extraction from data formats."""
        # Test statistical formats
        assert self.test_shim._extract_domain_from_format(
            DataFormat.STATISTICAL_RESULT) == 'statistical'
        
        # Test regression formats
        assert self.test_shim._extract_domain_from_format(
            DataFormat.REGRESSION_MODEL) == 'regression'
        
        # Test time series formats
        assert self.test_shim._extract_domain_from_format(
            DataFormat.TIME_SERIES) == 'time_series'
        assert self.test_shim._extract_domain_from_format(
            DataFormat.FORECAST_RESULT) == 'time_series'
        
        # Test pattern recognition formats
        assert self.test_shim._extract_domain_from_format(
            DataFormat.CLUSTERING_RESULT) == 'pattern_recognition'
        assert self.test_shim._extract_domain_from_format(
            DataFormat.PATTERN_RECOGNITION_RESULT) == 'pattern_recognition'
        
        # Test generic formats
        assert self.test_shim._extract_domain_from_format(
            DataFormat.PANDAS_DATAFRAME) == 'generic'
    
    def test_can_convert(self):
        """Test conversion capability evaluation."""
        # Create request with domain-relevant formats
        request = ConversionRequest(
            source_data={"test": "data"},
            source_format=DataFormat.STATISTICAL_RESULT,
            target_format=DataFormat.PANDAS_DATAFRAME,
            context=ConversionContext(
                source_domain='statistical',
                target_domain='generic',
                user_intention='data_transformation'
            )
        )
        
        confidence = self.test_shim.can_convert(request)
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0
        assert confidence > 0.5  # Should have reasonable confidence for domain match
    
    def test_analyze_data_characteristics_dataframe(self):
        """Test data characteristics analysis for DataFrame."""
        df = pd.DataFrame({
            'numeric_col': [1, 2, 3, 4, 5],
            'string_col': ['a', 'b', 'c', 'd', 'e'],
            'missing_col': [1, 2, None, 4, 5]
        })
        
        characteristics = self.test_shim._analyze_data_characteristics(df)
        
        assert characteristics['data_type'] == 'dataframe'
        assert characteristics['shape'] == (5, 3)
        assert 'numeric_col' in characteristics['columns']
        assert characteristics['numeric_columns'] == 2
        assert characteristics['categorical_columns'] == 1
        assert characteristics['missing_values']['missing_col'] == 1
    
    def test_analyze_data_characteristics_numpy(self):
        """Test data characteristics analysis for NumPy array."""
        array = np.random.rand(10, 5)
        
        characteristics = self.test_shim._analyze_data_characteristics(array)
        
        assert characteristics['data_type'] == 'numpy_array'
        assert characteristics['shape'] == (10, 5)
        assert characteristics['dimensions'] == 2
        assert characteristics['size'] == 50
        assert 'float' in characteristics['dtype']
    
    def test_analyze_data_characteristics_dict(self):
        """Test data characteristics analysis for dictionary."""
        data_dict = {
            'key1': [1, 2, 3],
            'key2': 'string_value',
            'key3': np.array([4, 5, 6])
        }
        
        characteristics = self.test_shim._analyze_data_characteristics(data_dict)
        
        assert characteristics['data_type'] == 'dictionary'
        assert set(characteristics['keys']) == {'key1', 'key2', 'key3'}
        assert len(characteristics['values_types']) == 3
    
    def test_extract_semantic_context(self):
        """Test semantic context extraction."""
        request = ConversionRequest(
            source_data={"test": "data"},
            source_format=DataFormat.STATISTICAL_RESULT,
            target_format=DataFormat.REGRESSION_MODEL,
            context=ConversionContext(
                source_domain='statistical',
                target_domain='regression',
                user_intention='model_building'
            )
        )
        
        semantic_context = self.test_shim._extract_semantic_context(request)
        
        assert isinstance(semantic_context, SemanticContext)
        assert semantic_context.analytical_goal == 'model_building'
        assert semantic_context.domain_context == 'statistical'
        assert semantic_context.target_use_case == 'regression'
    
    def test_estimate_cost(self):
        """Test cost estimation."""
        request = ConversionRequest(
            source_data=np.random.rand(1000, 10),
            source_format=DataFormat.NUMPY_ARRAY,
            target_format=DataFormat.PANDAS_DATAFRAME
        )
        
        cost = self.test_shim.estimate_cost(request)
        
        assert hasattr(cost, 'computational_cost')
        assert hasattr(cost, 'memory_cost_mb')
        assert hasattr(cost, 'time_estimate_seconds')
        assert 0.0 <= cost.computational_cost <= 1.0
        assert cost.memory_cost_mb > 0
        assert cost.time_estimate_seconds > 0
    
    def test_get_supported_conversions(self):
        """Test getting supported conversions."""
        conversions = self.test_shim.get_supported_conversions()
        
        assert isinstance(conversions, list)
        assert len(conversions) > 0
        
        for source_format, target_format in conversions:
            assert isinstance(source_format, DataFormat)
            assert isinstance(target_format, DataFormat)


class TestStatisticalShim:
    """Test the Statistical domain shim."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.shim = create_statistical_shim()
        
        # Initialize the shim
        assert self.shim.initialize()
        assert self.shim.activate()
    
    def test_initialization(self):
        """Test statistical shim initialization."""
        assert self.shim.adapter_id == "statistical_shim"
        assert self.shim.domain_type == DomainShimType.STATISTICAL
        assert len(self.shim.supported_mappings) > 0
        
        # Check domain mappings
        mapping_targets = [m.target_domain for m in self.shim.supported_mappings]
        assert 'regression' in mapping_targets
        assert 'time_series' in mapping_targets
        assert 'pattern_recognition' in mapping_targets
    
    def test_convert_correlation_matrix_to_regression(self):
        """Test conversion of correlation matrix to regression format."""
        # Create test correlation matrix
        corr_matrix = pd.DataFrame({
            'A': [1.0, 0.5, -0.3],
            'B': [0.5, 1.0, 0.2],
            'C': [-0.3, 0.2, 1.0]
        }, index=['A', 'B', 'C'])
        
        source_data = {
            'correlation_matrix': corr_matrix,
            'p_values': [[0.0, 0.01, 0.05], [0.01, 0.0, 0.1], [0.05, 0.1, 0.0]]
        }
        
        request = ConversionRequest(
            source_data=source_data,
            source_format=DataFormat.STATISTICAL_RESULT,
            target_format=DataFormat.REGRESSION_MODEL
        )
        
        result = self.shim.convert(request)
        
        assert result.success
        assert 'feature_correlation_matrix' in result.converted_data
        assert 'feature_names' in result.converted_data
        assert result.converted_data['feature_names'] == ['A', 'B', 'C']
    
    def test_convert_hypothesis_test_to_regression(self):
        """Test conversion of hypothesis test results to regression format."""
        source_data = {
            'test_statistic': 2.5,
            'p_value': 0.01,
            'degrees_of_freedom': 25,
            'effect_size': 0.3
        }
        
        request = ConversionRequest(
            source_data=source_data,
            source_format=DataFormat.STATISTICAL_RESULT,
            target_format=DataFormat.REGRESSION_MODEL
        )
        
        result = self.shim.convert(request)
        
        assert result.success
        assert 'model_diagnostics' in result.converted_data
        diagnostics = result.converted_data['model_diagnostics']
        assert diagnostics['test_statistic'] == 2.5
        assert diagnostics['significance'] == 0.01
    
    def test_convert_stationarity_test_to_time_series(self):
        """Test conversion of stationarity test to time series format."""
        source_data = {
            'stationarity_test': True,
            'test_statistic': -3.2,
            'p_value': 0.02
        }
        
        request = ConversionRequest(
            source_data=source_data,
            source_format=DataFormat.STATISTICAL_RESULT,
            target_format=DataFormat.TIME_SERIES
        )
        
        result = self.shim.convert(request)
        
        assert result.success
        assert 'stationarity_info' in result.converted_data
        stationarity_info = result.converted_data['stationarity_info']
        assert stationarity_info['is_stationary'] == True
        assert stationarity_info['differencing_suggested'] == False
    
    def test_convert_pca_to_pattern_recognition(self):
        """Test conversion of PCA results to pattern recognition format."""
        source_data = {
            'principal_components': np.random.rand(5, 3),
            'explained_variance_ratio': [0.6, 0.25, 0.1],
            'n_components': 3
        }
        
        request = ConversionRequest(
            source_data=source_data,
            source_format=DataFormat.STATISTICAL_RESULT,
            target_format=DataFormat.PATTERN_RECOGNITION_RESULT
        )
        
        result = self.shim.convert(request)
        
        assert result.success
        assert 'dimensionality_reduction' in result.converted_data
        dr_info = result.converted_data['dimensionality_reduction']
        assert dr_info['n_components'] == 3
        assert len(dr_info['explained_variance_ratio']) == 3
    
    def test_normalize_regression_results(self):
        """Test result normalization for regression target domain."""
        data = {
            'feature_correlation_matrix': np.eye(3),
            'feature_names': ['A', 'B', 'C']
        }
        
        mapping = DomainMapping(
            source_domain='statistical',
            target_domain='regression',
            quality_preservation=0.95
        )
        
        semantic_context = SemanticContext(
            analytical_goal='feature_selection',
            domain_context='statistical',
            target_use_case='regression'
        )
        
        normalized = self.shim._normalize_results(data, mapping, semantic_context)
        
        assert 'domain_conversion' in normalized
        assert normalized['domain_conversion']['source'] == 'statistical'
        assert normalized['domain_conversion']['target'] == 'regression'
        assert 'feature_selection_hints' in normalized


class TestRegressionShim:
    """Test the Regression domain shim."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.shim = create_regression_shim()
        assert self.shim.initialize()
        assert self.shim.activate()
    
    def test_initialization(self):
        """Test regression shim initialization."""
        assert self.shim.adapter_id == "regression_shim"
        assert self.shim.domain_type == DomainShimType.REGRESSION
        assert len(self.shim.supported_mappings) > 0
        
        # Check domain mappings
        mapping_targets = [m.target_domain for m in self.shim.supported_mappings]
        assert 'time_series' in mapping_targets
        assert 'pattern_recognition' in mapping_targets
        assert 'statistical' in mapping_targets
    
    def test_convert_regression_model_to_time_series(self):
        """Test conversion of regression model to time series format."""
        source_data = {
            'coefficients': [0.5, -0.2, 1.1],
            'fitted_values': np.random.rand(50),
            'residuals': np.random.randn(50) * 0.1,
            'r2_score': 0.85
        }
        
        request = ConversionRequest(
            source_data=source_data,
            source_format=DataFormat.REGRESSION_MODEL,
            target_format=DataFormat.TIME_SERIES
        )
        
        result = self.shim.convert(request)
        
        assert result.success
        assert 'trend_model' in result.converted_data
        trend_model = result.converted_data['trend_model']
        assert 'trend_parameters' in trend_model
        assert 'fitted_trend' in trend_model
        assert 'forecast_info' in result.converted_data
    
    def test_convert_predictions_to_time_series(self):
        """Test conversion of predictions to time series format."""
        predictions = np.random.rand(20)
        confidence_intervals = np.column_stack([predictions - 0.1, predictions + 0.1])
        
        source_data = {
            'predictions': predictions,
            'confidence_intervals': confidence_intervals
        }
        
        request = ConversionRequest(
            source_data=source_data,
            source_format=DataFormat.REGRESSION_MODEL,
            target_format=DataFormat.FORECAST_RESULT
        )
        
        result = self.shim.convert(request)
        
        assert result.success
        assert 'forecasted_values' in result.converted_data
        assert 'forecast_intervals' in result.converted_data
        assert result.converted_data['forecast_method'] == 'regression_based'
    
    def test_convert_feature_importance_to_pattern_recognition(self):
        """Test conversion of feature importance to pattern recognition format."""
        source_data = {
            'feature_importance': [0.4, 0.3, 0.2, 0.1],
            'feature_names': ['feat1', 'feat2', 'feat3', 'feat4']
        }
        
        request = ConversionRequest(
            source_data=source_data,
            source_format=DataFormat.REGRESSION_MODEL,
            target_format=DataFormat.PATTERN_RECOGNITION_RESULT
        )
        
        result = self.shim.convert(request)
        
        assert result.success
        assert 'feature_weights' in result.converted_data
        assert result.converted_data['feature_names'] == ['feat1', 'feat2', 'feat3', 'feat4']
        assert result.converted_data['importance_type'] == 'regression_coefficients'
    
    def test_convert_residuals_to_pattern_recognition(self):
        """Test conversion of residuals for anomaly detection."""
        residuals = np.random.randn(100) * 0.5
        residuals[10] = 3.0  # Add an outlier
        
        source_data = {'residuals': residuals}
        
        request = ConversionRequest(
            source_data=source_data,
            source_format=DataFormat.REGRESSION_MODEL,
            target_format=DataFormat.PATTERN_RECOGNITION_RESULT
        )
        
        result = self.shim.convert(request)
        
        assert result.success
        assert 'anomaly_scores' in result.converted_data
        assert 'residual_statistics' in result.converted_data
        assert result.converted_data['anomaly_method'] == 'residual_based'
    
    def test_convert_model_summary_to_statistical(self):
        """Test conversion of model summary to statistical format."""
        source_data = {
            'coefficients': [1.2, -0.8, 0.5],
            'p_values': [0.01, 0.05, 0.12]
        }
        
        request = ConversionRequest(
            source_data=source_data,
            source_format=DataFormat.REGRESSION_MODEL,
            target_format=DataFormat.STATISTICAL_RESULT
        )
        
        result = self.shim.convert(request)
        
        assert result.success
        assert 'hypothesis_test_results' in result.converted_data
        test_results = result.converted_data['hypothesis_test_results']
        assert len(test_results['significant_features']) == 2  # p < 0.05
    
    def test_normality_test_implementation(self):
        """Test normality test for residuals."""
        # Test with normal residuals
        normal_residuals = np.random.normal(0, 1, 100)
        result = self.shim._test_normality(normal_residuals)
        
        assert 'test_name' in result
        assert 'statistic' in result
        assert 'p_value' in result
        assert 'is_normal' in result
        
        # Test with very non-normal residuals
        non_normal_residuals = np.concatenate([
            np.ones(50), np.zeros(50)
        ])
        result = self.shim._test_normality(non_normal_residuals)
        assert result['is_normal'] == False
    
    def test_homoscedasticity_test_implementation(self):
        """Test homoscedasticity test for residuals."""
        # Test with homoscedastic residuals
        homoscedastic_residuals = np.random.normal(0, 1, 200)
        result = self.shim._test_homoscedasticity(homoscedastic_residuals)
        
        assert 'test_name' in result
        assert 'variance_of_variances' in result
        assert 'is_homoscedastic' in result
    
    def test_autocorrelation_test_implementation(self):
        """Test autocorrelation test for residuals."""
        # Test with independent residuals
        independent_residuals = np.random.normal(0, 1, 100)
        result = self.shim._test_autocorrelation(independent_residuals)
        
        assert 'test_name' in result
        assert 'correlation' in result
        assert 'p_value' in result
        assert 'has_autocorrelation' in result


class TestTimeSeriesShim:
    """Test the Time Series domain shim."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.shim = create_time_series_shim()
        assert self.shim.initialize()
        assert self.shim.activate()
    
    def test_initialization(self):
        """Test time series shim initialization."""
        assert self.shim.adapter_id == "time_series_shim"
        assert self.shim.domain_type == DomainShimType.TIME_SERIES
        assert len(self.shim.supported_mappings) > 0
        
        # Check domain mappings
        mapping_targets = [m.target_domain for m in self.shim.supported_mappings]
        assert 'statistical' in mapping_targets
        assert 'regression' in mapping_targets
        assert 'pattern_recognition' in mapping_targets
    
    def test_convert_time_series_to_statistical(self):
        """Test conversion of time series data to statistical format."""
        # Create time series data
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        values = np.cumsum(np.random.randn(100)) + 100
        ts_data = pd.DataFrame({'value': values}, index=dates)
        
        request = ConversionRequest(
            source_data=ts_data,
            source_format=DataFormat.TIME_SERIES,
            target_format=DataFormat.STATISTICAL_RESULT
        )
        
        result = self.shim.convert(request)
        
        assert result.success
        assert 'time_series_statistics' in result.converted_data
        stats = result.converted_data['time_series_statistics']
        assert 'mean' in stats
        assert 'std' in stats
        assert 'trend_slope' in stats
        assert 'stationarity_score' in stats
    
    def test_convert_time_series_to_regression(self):
        """Test conversion of time series to regression format."""
        values = np.cumsum(np.random.randn(50)) + 10
        ts_data = pd.DataFrame({'value': values})
        
        request = ConversionRequest(
            source_data=ts_data,
            source_format=DataFormat.TIME_SERIES,
            target_format=DataFormat.REGRESSION_MODEL
        )
        
        result = self.shim.convert(request)
        
        assert result.success
        assert 'feature_matrix' in result.converted_data
        assert 'target_variable' in result.converted_data
        assert 'feature_names' in result.converted_data
        
        # Check that lagged features were created
        feature_names = result.converted_data['feature_names']
        assert any('lag_' in name for name in feature_names)
        assert 'trend' in feature_names
    
    def test_convert_forecast_results_to_regression(self):
        """Test conversion of forecast results to regression format."""
        source_data = {
            'forecast_result': {
                'forecasted_values': np.random.rand(10),
                'confidence_intervals': np.random.rand(10, 2),
                'forecast_method': 'arima'
            },
            'actual_values': np.random.rand(10)
        }
        
        request = ConversionRequest(
            source_data=source_data,
            source_format=DataFormat.FORECAST_RESULT,
            target_format=DataFormat.REGRESSION_MODEL
        )
        
        result = self.shim.convert(request)
        
        assert result.success
        assert 'prediction_data' in result.converted_data
        assert 'validation_data' in result.converted_data
        
        prediction_data = result.converted_data['prediction_data']
        assert 'predicted_values' in prediction_data
        assert 'prediction_method' in prediction_data
    
    def test_convert_time_series_to_pattern_recognition(self):
        """Test conversion of time series to pattern recognition format."""
        # Create time series with some pattern
        t = np.linspace(0, 4*np.pi, 100)
        values = np.sin(t) + 0.1 * np.random.randn(100)
        ts_data = pd.DataFrame({'value': values})
        
        request = ConversionRequest(
            source_data=ts_data,
            source_format=DataFormat.TIME_SERIES,
            target_format=DataFormat.PATTERN_RECOGNITION_RESULT
        )
        
        result = self.shim.convert(request)
        
        assert result.success
        assert 'temporal_features' in result.converted_data
        assert 'pattern_detection' in result.converted_data
        assert 'sequence_characteristics' in result.converted_data
    
    def test_trend_slope_calculation(self):
        """Test trend slope calculation."""
        # Test increasing trend
        increasing_values = np.array([1, 2, 3, 4, 5])
        slope = self.shim._calculate_trend_slope(increasing_values)
        assert slope > 0
        
        # Test decreasing trend
        decreasing_values = np.array([5, 4, 3, 2, 1])
        slope = self.shim._calculate_trend_slope(decreasing_values)
        assert slope < 0
        
        # Test flat trend
        flat_values = np.array([3, 3, 3, 3, 3])
        slope = self.shim._calculate_trend_slope(flat_values)
        assert abs(slope) < 0.1
    
    def test_stationarity_assessment(self):
        """Test stationarity assessment."""
        # Test stationary series (white noise)
        stationary_series = np.random.randn(100)
        stationarity_score = self.shim._assess_stationarity(stationary_series)
        assert 0.0 <= stationarity_score <= 1.0
        
        # Test non-stationary series (random walk)
        non_stationary_series = np.cumsum(np.random.randn(100))
        stationarity_score = self.shim._assess_stationarity(non_stationary_series)
        assert 0.0 <= stationarity_score <= 1.0
    
    def test_autocorrelation_calculation(self):
        """Test autocorrelation calculation."""
        # Test with AR(1) process
        ar_process = np.zeros(100)
        ar_process[0] = np.random.randn()
        for i in range(1, 100):
            ar_process[i] = 0.7 * ar_process[i-1] + np.random.randn() * 0.5
        
        acf = self.shim._calculate_autocorrelation(ar_process)
        
        assert acf is not None
        assert len(acf) > 5
        assert acf[0] == 1.0  # Autocorrelation at lag 0 should be 1
        assert abs(acf[1]) > 0.3  # Should have significant autocorrelation at lag 1
    
    def test_lagged_features_creation(self):
        """Test creation of lagged features."""
        values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        n_lags = 3
        
        lagged_features = self.shim._create_lagged_features(values, n_lags)
        
        assert lagged_features.shape == (7, 3)  # 10 - 3 = 7 rows, 3 lag columns
        
        # Check that first row contains lags 3, 2, 1 of the original series
        assert lagged_features[0, 0] == 3  # lag_1 at position 3
        assert lagged_features[0, 1] == 2  # lag_2 at position 3  
        assert lagged_features[0, 2] == 1  # lag_3 at position 3
    
    def test_seasonal_pattern_detection(self):
        """Test seasonal pattern detection."""
        # Create series with seasonal pattern (period 12)
        t = np.arange(144)  # 12 years of monthly data
        seasonal_series = np.sin(2 * np.pi * t / 12) + 0.1 * np.random.randn(144)
        
        has_seasonality = self.shim._detect_seasonality(seasonal_series)
        assert has_seasonality == True
        
        # Test with random series (should not detect seasonality)
        random_series = np.random.randn(144)
        has_seasonality = self.shim._detect_seasonality(random_series)
        assert has_seasonality == False


class TestPatternRecognitionShim:
    """Test the Pattern Recognition domain shim."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.shim = create_pattern_recognition_shim()
        assert self.shim.initialize()
        assert self.shim.activate()
    
    def test_initialization(self):
        """Test pattern recognition shim initialization."""
        assert self.shim.adapter_id == "pattern_recognition_shim"
        assert self.shim.domain_type == DomainShimType.PATTERN_RECOGNITION
        assert len(self.shim.supported_mappings) > 0
        
        # Check domain mappings
        mapping_targets = [m.target_domain for m in self.shim.supported_mappings]
        assert 'statistical' in mapping_targets
        assert 'regression' in mapping_targets
        assert 'time_series' in mapping_targets
    
    def test_convert_clustering_results_to_statistical(self):
        """Test conversion of clustering results to statistical format."""
        source_data = {
            'cluster_labels': np.array([0, 0, 1, 1, 1, 2, 2]),
            'centroids': np.array([[1, 2], [3, 4], [5, 6]]),
            'silhouette_score': 0.8,
            'inertia': 15.2
        }
        
        request = ConversionRequest(
            source_data=source_data,
            source_format=DataFormat.CLUSTERING_RESULT,
            target_format=DataFormat.STATISTICAL_RESULT
        )
        
        result = self.shim.convert(request)
        
        assert result.success
        assert 'group_statistics' in result.converted_data
        assert 'clustering_validity' in result.converted_data
        
        validity = result.converted_data['clustering_validity']
        assert validity['n_clusters'] == 3
        assert validity['silhouette_score'] == 0.8
    
    def test_convert_classification_results_to_statistical(self):
        """Test conversion of classification results to statistical format."""
        predictions = np.array([0, 1, 0, 1, 2, 2, 1])
        probabilities = np.array([
            [0.9, 0.1, 0.0],
            [0.2, 0.8, 0.0],
            [0.8, 0.2, 0.0],
            [0.1, 0.9, 0.0],
            [0.0, 0.1, 0.9],
            [0.0, 0.0, 1.0],
            [0.3, 0.7, 0.0]
        ])
        
        source_data = {
            'predictions': predictions,
            'probabilities': probabilities
        }
        
        request = ConversionRequest(
            source_data=source_data,
            source_format=DataFormat.PATTERN_RECOGNITION_RESULT,
            target_format=DataFormat.STATISTICAL_RESULT
        )
        
        result = self.shim.convert(request)
        
        assert result.success
        assert 'classification_statistics' in result.converted_data
        assert 'prediction_confidence' in result.converted_data
        
        class_stats = result.converted_data['classification_statistics']
        assert 'class_0' in class_stats
        assert 'class_1' in class_stats
        assert 'class_2' in class_stats
    
    def test_convert_feature_importance_to_regression(self):
        """Test conversion of feature importance to regression format."""
        source_data = {
            'feature_importance': [0.5, 0.3, 0.15, 0.05],
            'feature_names': ['feature1', 'feature2', 'feature3', 'feature4']
        }
        
        request = ConversionRequest(
            source_data=source_data,
            source_format=DataFormat.PATTERN_RECOGNITION_RESULT,
            target_format=DataFormat.REGRESSION_MODEL
        )
        
        result = self.shim.convert(request)
        
        assert result.success
        assert 'feature_selection' in result.converted_data
        
        feature_selection = result.converted_data['feature_selection']
        assert feature_selection['selection_method'] == 'pattern_recognition_based'
        assert len(feature_selection['feature_importance']) == 4
    
    def test_convert_cluster_labels_to_regression(self):
        """Test conversion of cluster labels to regression format."""
        source_data = {
            'cluster_labels': np.array([0, 1, 0, 2, 1, 2, 0])
        }
        
        request = ConversionRequest(
            source_data=source_data,
            source_format=DataFormat.CLUSTERING_RESULT,
            target_format=DataFormat.REGRESSION_MODEL
        )
        
        result = self.shim.convert(request)
        
        assert result.success
        assert 'categorical_predictors' in result.converted_data
        
        cat_predictors = result.converted_data['categorical_predictors']
        assert 'cluster_dummies' in cat_predictors
        assert 'n_clusters' in cat_predictors
        assert cat_predictors['n_clusters'] == 3
        
        # Check dummy matrix shape and values
        dummy_matrix = cat_predictors['cluster_dummies']
        assert dummy_matrix.shape == (7, 3)  # 7 samples, 3 clusters
        assert np.all(np.sum(dummy_matrix, axis=1) == 1)  # One-hot encoding
    
    def test_convert_sequential_patterns_to_time_series(self):
        """Test conversion of sequential patterns to time series format."""
        source_data = {
            'sequential_patterns': [
                {'pattern': [1, 2, 3], 'frequency': 5},
                {'pattern': [2, 3, 1], 'frequency': 3}
            ]
        }
        
        request = ConversionRequest(
            source_data=source_data,
            source_format=DataFormat.PATTERN_RECOGNITION_RESULT,
            target_format=DataFormat.TIME_SERIES
        )
        
        result = self.shim.convert(request)
        
        assert result.success
        assert 'pattern_analysis' in result.converted_data
        
        pattern_analysis = result.converted_data['pattern_analysis']
        assert 'detected_patterns' in pattern_analysis
        assert 'pattern_frequency' in pattern_analysis
        assert pattern_analysis['temporal_structure'] == 'sequential'
    
    def test_convert_change_points_to_time_series(self):
        """Test conversion of change points to time series format."""
        source_data = {
            'change_points': [10, 25, 40],
            'anomaly_scores': np.random.rand(50)
        }
        
        request = ConversionRequest(
            source_data=source_data,
            source_format=DataFormat.PATTERN_RECOGNITION_RESULT,
            target_format=DataFormat.TIME_SERIES
        )
        
        result = self.shim.convert(request)
        
        assert result.success
        assert 'structural_break_analysis' in result.converted_data
        
        break_analysis = result.converted_data['structural_break_analysis']
        assert break_analysis['n_change_points'] == 3
        assert break_analysis['break_detection_method'] == 'pattern_recognition_based'
    
    def test_convert_temporal_clustering_to_time_series(self):
        """Test conversion of temporal clustering to time series format."""
        source_data = {
            'cluster_labels': [0, 0, 1, 1, 2, 2, 0, 0],
            'temporal_info': {
                'timestamps': ['2023-01-01', '2023-01-02', '2023-01-03', 
                              '2023-01-04', '2023-01-05', '2023-01-06',
                              '2023-01-07', '2023-01-08']
            }
        }
        
        request = ConversionRequest(
            source_data=source_data,
            source_format=DataFormat.CLUSTERING_RESULT,
            target_format=DataFormat.TIME_SERIES
        )
        
        result = self.shim.convert(request)
        
        assert result.success
        assert 'regime_identification' in result.converted_data
        
        regime_id = result.converted_data['regime_identification']
        assert regime_id['n_regimes'] == 3
        assert len(regime_id['regime_changes']) > 0  # Should detect regime changes


class TestDomainShimUtilities:
    """Test utility functions for domain shim management."""
    
    def test_create_all_domain_shims(self):
        """Test creation of all domain shims."""
        shims = create_all_domain_shims(auto_register=False)
        
        assert len(shims) == 4
        assert 'statistical' in shims
        assert 'regression' in shims
        assert 'time_series' in shims
        assert 'pattern_recognition' in shims
        
        # Check types
        assert isinstance(shims['statistical'], StatisticalShim)
        assert isinstance(shims['regression'], RegressionShim)
        assert isinstance(shims['time_series'], TimeSeriesShim)
        assert isinstance(shims['pattern_recognition'], PatternRecognitionShim)
    
    def test_get_compatible_domain_shims(self):
        """Test finding compatible domain shims."""
        available_shims = create_all_domain_shims(auto_register=False)
        
        # Test statistical to regression compatibility
        compatible = get_compatible_domain_shims(
            'statistical', 'regression', available_shims
        )
        assert len(compatible) > 0
        assert any(isinstance(shim, StatisticalShim) for shim in compatible)
        
        # Test time_series to statistical compatibility
        compatible = get_compatible_domain_shims(
            'time_series', 'statistical', available_shims
        )
        assert len(compatible) > 0
        assert any(isinstance(shim, TimeSeriesShim) for shim in compatible)
        
        # Test non-existent compatibility
        compatible = get_compatible_domain_shims(
            'non_existent', 'also_non_existent', available_shims
        )
        assert len(compatible) == 0
    
    def test_validate_domain_shim_configuration(self):
        """Test validation of domain shim configuration."""
        # Test valid configuration
        valid_shims = create_all_domain_shims(auto_register=False)
        result = validate_domain_shim_configuration(valid_shims)
        
        assert result.is_valid
        assert result.score > 0.8
        assert len(result.errors) == 0
        
        # Test configuration with missing shims
        incomplete_shims = {
            'statistical': valid_shims['statistical'],
            'regression': valid_shims['regression']
        }
        result = validate_domain_shim_configuration(incomplete_shims)
        
        assert len(result.warnings) > 0
        assert 'Missing domain shims' in result.warnings[0]
        assert result.details['missing_shims'] == ['pattern_recognition', 'time_series']
    
    def test_factory_functions(self):
        """Test individual factory functions."""
        # Test statistical shim factory
        stat_shim = create_statistical_shim("custom_stat_id")
        assert stat_shim.adapter_id == "custom_stat_id"
        assert isinstance(stat_shim, StatisticalShim)
        
        # Test regression shim factory
        reg_shim = create_regression_shim("custom_reg_id")
        assert reg_shim.adapter_id == "custom_reg_id"
        assert isinstance(reg_shim, RegressionShim)
        
        # Test time series shim factory
        ts_shim = create_time_series_shim("custom_ts_id")
        assert ts_shim.adapter_id == "custom_ts_id"
        assert isinstance(ts_shim, TimeSeriesShim)
        
        # Test pattern recognition shim factory
        pr_shim = create_pattern_recognition_shim("custom_pr_id")
        assert pr_shim.adapter_id == "custom_pr_id"
        assert isinstance(pr_shim, PatternRecognitionShim)


class TestDomainShimIntegration:
    """Test integration scenarios between domain shims."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.shims = create_all_domain_shims(auto_register=False)
        
        # Initialize all shims
        for shim in self.shims.values():
            assert shim.initialize()
            assert shim.activate()
    
    def test_statistical_to_regression_to_time_series_workflow(self):
        """Test a complete workflow: statistical → regression → time series."""
        # Step 1: Statistical analysis results
        correlation_matrix = pd.DataFrame({
            'A': [1.0, 0.7, -0.3],
            'B': [0.7, 1.0, 0.2],
            'C': [-0.3, 0.2, 1.0]
        }, index=['A', 'B', 'C'])
        
        stat_data = {
            'correlation_matrix': correlation_matrix,
            'p_values': [[0.0, 0.01, 0.05], [0.01, 0.0, 0.1], [0.05, 0.1, 0.0]]
        }
        
        # Step 2: Convert statistical → regression
        stat_to_reg_request = ConversionRequest(
            source_data=stat_data,
            source_format=DataFormat.STATISTICAL_RESULT,
            target_format=DataFormat.REGRESSION_MODEL
        )
        
        reg_result = self.shims['statistical'].convert(stat_to_reg_request)
        assert reg_result.success
        
        # Step 3: Convert regression → time series (using fitted values)
        reg_data = reg_result.converted_data
        reg_data['fitted_values'] = np.cumsum(np.random.randn(50)) * 0.1 + 10
        reg_data['residuals'] = np.random.randn(50) * 0.5
        reg_data['coefficients'] = [0.8, -0.3, 0.2]
        
        reg_to_ts_request = ConversionRequest(
            source_data=reg_data,
            source_format=DataFormat.REGRESSION_MODEL,
            target_format=DataFormat.TIME_SERIES
        )
        
        ts_result = self.shims['regression'].convert(reg_to_ts_request)
        assert ts_result.success
        assert 'trend_model' in ts_result.converted_data
    
    def test_time_series_to_pattern_recognition_to_statistical_workflow(self):
        """Test workflow: time series → pattern recognition → statistical."""
        # Step 1: Time series data with patterns
        t = np.linspace(0, 4*np.pi, 100)
        values = np.sin(t) + np.cos(0.5*t) + 0.1 * np.random.randn(100)
        ts_data = pd.DataFrame({'value': values})
        
        # Step 2: Convert time series → pattern recognition
        ts_to_pr_request = ConversionRequest(
            source_data=ts_data,
            source_format=DataFormat.TIME_SERIES,
            target_format=DataFormat.PATTERN_RECOGNITION_RESULT
        )
        
        pr_result = self.shims['time_series'].convert(ts_to_pr_request)
        assert pr_result.success
        
        # Step 3: Convert pattern recognition → statistical
        pr_data = pr_result.converted_data
        # Simulate clustering results
        pr_data['cluster_labels'] = np.random.choice([0, 1, 2], 100)
        pr_data['centroids'] = np.random.rand(3, 2)
        pr_data['silhouette_score'] = 0.7
        
        pr_to_stat_request = ConversionRequest(
            source_data=pr_data,
            source_format=DataFormat.CLUSTERING_RESULT,
            target_format=DataFormat.STATISTICAL_RESULT
        )
        
        stat_result = self.shims['pattern_recognition'].convert(pr_to_stat_request)
        assert stat_result.success
        assert 'group_statistics' in stat_result.converted_data
    
    def test_cross_domain_quality_preservation(self):
        """Test that quality is preserved across domain conversions."""
        # Create high-quality source data
        source_data = {
            'correlation_matrix': pd.DataFrame(np.eye(5)),  # Perfect correlation matrix
            'p_values': np.zeros((5, 5))  # All significant
        }
        
        request = ConversionRequest(
            source_data=source_data,
            source_format=DataFormat.STATISTICAL_RESULT,
            target_format=DataFormat.REGRESSION_MODEL
        )
        
        result = self.shims['statistical'].convert(request)
        
        assert result.success
        assert result.quality_score > 0.8  # High quality should be preserved
        
        # Check metadata preservation
        assert 'domain_conversion' in result.metadata
        assert result.metadata['domain_conversion']['source'] == 'statistical'
        assert result.metadata['domain_conversion']['target'] == 'regression'
    
    def test_error_handling_in_domain_conversion(self):
        """Test error handling in domain conversion scenarios."""
        # Test with invalid source data
        invalid_request = ConversionRequest(
            source_data=None,  # Invalid data
            source_format=DataFormat.STATISTICAL_RESULT,
            target_format=DataFormat.REGRESSION_MODEL
        )
        
        result = self.shims['statistical'].convert(invalid_request)
        
        # Should handle gracefully
        assert not result.success
        assert len(result.errors) > 0
        assert result.quality_score == 0.0
    
    def test_semantic_context_preservation(self):
        """Test that semantic context is preserved across conversions."""
        source_data = {'test': 'data'}
        
        context = ConversionContext(
            source_domain='statistical',
            target_domain='regression',
            user_intention='feature_selection_for_modeling'
        )
        
        request = ConversionRequest(
            source_data=source_data,
            source_format=DataFormat.STATISTICAL_RESULT,
            target_format=DataFormat.REGRESSION_MODEL,
            context=context
        )
        
        result = self.shims['statistical'].convert(request)
        
        # Check that semantic context is preserved in metadata
        assert 'domain_shim' in result.metadata
        shim_metadata = result.metadata['domain_shim']
        assert 'semantic_context' in shim_metadata
        semantic_context = shim_metadata['semantic_context']
        assert semantic_context['analytical_goal'] == 'feature_selection_for_modeling'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])