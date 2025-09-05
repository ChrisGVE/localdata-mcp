"""
Tests for Multivariate Time Series Analysis.

This module tests the multivariate time series analysis capabilities including
VAR modeling, cointegration analysis, Granger causality testing, and impulse
response analysis.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
import warnings

from localdata_mcp.domains.time_series_analysis import (
    MultivariateTimeSeriesTransformer,
    VARModelForecaster,
    CointegrationAnalyzer,
    GrangerCausalityAnalyzer,
    ImpulseResponseAnalyzer,
    TimeSeriesValidationError
)


@pytest.fixture
def sample_multivariate_data():
    """Create sample multivariate time series data."""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=200, freq='D')
    
    # Create three correlated time series
    n_obs = len(dates)
    
    # Generate common factor
    common_factor = np.cumsum(np.random.randn(n_obs)) * 0.1
    
    # Series with different relationships to common factor
    series1 = common_factor + np.cumsum(np.random.randn(n_obs) * 0.05)
    series2 = 2 * common_factor + np.cumsum(np.random.randn(n_obs) * 0.05)
    series3 = -common_factor + np.cumsum(np.random.randn(n_obs) * 0.05)
    
    return pd.DataFrame({
        'series1': series1,
        'series2': series2,
        'series3': series3
    }, index=dates)


@pytest.fixture
def sample_causal_data():
    """Create sample data with known causal relationships."""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=200, freq='D')
    n_obs = len(dates)
    
    # Initialize series
    cause = np.zeros(n_obs)
    effect = np.zeros(n_obs)
    independent = np.zeros(n_obs)
    
    # Generate causal relationship: cause -> effect with lag 1
    cause[0] = np.random.randn()
    effect[0] = np.random.randn()
    independent[0] = np.random.randn()
    
    for i in range(1, n_obs):
        cause[i] = 0.3 * cause[i-1] + np.random.randn() * 0.5
        effect[i] = 0.2 * effect[i-1] + 0.4 * cause[i-1] + np.random.randn() * 0.3
        independent[i] = 0.1 * independent[i-1] + np.random.randn() * 0.4
    
    return pd.DataFrame({
        'cause': cause,
        'effect': effect,
        'independent': independent
    }, index=dates)


@pytest.fixture
def sample_cointegrated_data():
    """Create sample cointegrated time series data."""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=200, freq='D')
    n_obs = len(dates)
    
    # Generate I(1) series with cointegrating relationship
    # Common stochastic trend
    common_trend = np.cumsum(np.random.randn(n_obs))
    
    # Cointegrated series: series1 - 2*series2 + series3 ~ I(0)
    series1 = common_trend + np.random.randn(n_obs) * 0.1
    series2 = 0.5 * common_trend + np.random.randn(n_obs) * 0.1
    series3 = -common_trend + np.random.randn(n_obs) * 0.1
    
    return pd.DataFrame({
        'series1': series1,
        'series2': series2,
        'series3': series3
    }, index=dates)


class TestMultivariateTimeSeriesTransformer:
    """Test cases for MultivariateTimeSeriesTransformer base class."""
    
    def test_init(self):
        """Test initialization of multivariate transformer."""
        transformer = MultivariateTimeSeriesTransformer()
        assert transformer.min_series == 2
        assert transformer.max_series is None
        assert transformer.require_stationarity is False
        
        # Test with custom parameters
        transformer = MultivariateTimeSeriesTransformer(
            min_series=3, 
            max_series=5, 
            require_stationarity=True
        )
        assert transformer.min_series == 3
        assert transformer.max_series == 5
        assert transformer.require_stationarity is True
    
    def test_validate_multivariate_data_success(self, sample_multivariate_data):
        """Test successful multivariate data validation."""
        transformer = MultivariateTimeSeriesTransformer()
        validated_data = transformer._validate_multivariate_data(sample_multivariate_data)
        
        assert isinstance(validated_data, pd.DataFrame)
        assert validated_data.shape == sample_multivariate_data.shape
        assert list(validated_data.columns) == list(sample_multivariate_data.columns)
    
    def test_validate_multivariate_data_insufficient_series(self, sample_multivariate_data):
        """Test validation failure with insufficient series."""
        transformer = MultivariateTimeSeriesTransformer(min_series=5)
        
        with pytest.raises(TimeSeriesValidationError, match="Insufficient number of time series"):
            transformer._validate_multivariate_data(sample_multivariate_data)
    
    def test_validate_multivariate_data_too_many_series(self, sample_multivariate_data):
        """Test validation failure with too many series."""
        transformer = MultivariateTimeSeriesTransformer(max_series=2)
        
        with pytest.raises(TimeSeriesValidationError, match="Too many time series"):
            transformer._validate_multivariate_data(sample_multivariate_data)
    
    def test_check_multicollinearity_warning(self):
        """Test multicollinearity detection warning."""
        # Create highly correlated data
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        series1 = np.cumsum(np.random.randn(100))
        series2 = series1 + np.random.randn(100) * 0.01  # Almost identical
        
        data = pd.DataFrame({
            'series1': series1,
            'series2': series2
        }, index=dates)
        
        transformer = MultivariateTimeSeriesTransformer()
        
        with patch('localdata_mcp.domains.time_series_analysis.logger') as mock_logger:
            transformer._check_multicollinearity(data)
            mock_logger.warning.assert_called_once()


class TestVARModelForecaster:
    """Test cases for VARModelForecaster."""
    
    def test_init(self):
        """Test initialization of VAR forecaster."""
        forecaster = VARModelForecaster()
        assert forecaster.max_lags == 10
        assert forecaster.ic == 'aic'
        assert forecaster.forecast_horizon == 10
        assert forecaster.confidence_level == 0.95
        assert forecaster.trend == 'c'
    
    def test_init_invalid_ic(self):
        """Test initialization with invalid information criterion."""
        with pytest.raises(ValueError, match="Information criterion must be one of"):
            VARModelForecaster(ic='invalid')
    
    def test_init_invalid_trend(self):
        """Test initialization with invalid trend specification."""
        with pytest.raises(ValueError, match="Trend specification must be one of"):
            VARModelForecaster(trend='invalid')
    
    def test_analysis_logic_success(self, sample_multivariate_data):
        """Test successful VAR analysis."""
        forecaster = VARModelForecaster(
            max_lags=3, 
            forecast_horizon=5,
            confidence_level=0.90
        )
        
        result = forecaster._analysis_logic(sample_multivariate_data)
        
        assert result.analysis_type == "VAR_forecasting"
        assert result.forecast_values is not None
        assert result.forecast_confidence_intervals is not None
        assert result.forecast_horizon == 5
        assert result.confidence_level == 0.90
        
        # Check forecast dimensions
        assert result.forecast_values.shape[0] == 5
        assert result.forecast_values.shape[1] == 3
        
        # Check model parameters
        assert 'optimal_lags' in result.model_parameters
        assert 'coefficients' in result.model_parameters
        assert 'ic_used' in result.model_parameters
        
        # Check diagnostics
        assert 'aic' in result.model_diagnostics
        assert 'bic' in result.model_diagnostics
        assert 'n_observations' in result.model_diagnostics
    
    def test_analysis_logic_with_small_dataset(self):
        """Test VAR analysis with insufficient data."""
        # Create very small dataset
        dates = pd.date_range('2020-01-01', periods=15, freq='D')
        small_data = pd.DataFrame({
            'series1': np.random.randn(15),
            'series2': np.random.randn(15)
        }, index=dates)
        
        forecaster = VARModelForecaster()
        result = forecaster._analysis_logic(small_data)
        
        # Should return error result
        assert result.analysis_type == "VAR_forecasting"
        assert "failed" in result.interpretation.lower()
    
    def test_generate_var_interpretation(self, sample_multivariate_data):
        """Test VAR interpretation generation."""
        forecaster = VARModelForecaster()
        
        # Mock a VAR result
        mock_var_fitted = MagicMock()
        mock_var_fitted.neqs = 3
        mock_var_fitted.nobs = 150
        
        interpretation = forecaster._generate_var_interpretation(
            mock_var_fitted, 
            optimal_lags=2,
            diagnostics={'aic': 100.5},
            fit_stats={'rsquared_avg': 0.75}
        )
        
        assert "VAR(2)" in interpretation
        assert "3 time series" in interpretation
        assert "150 observations" in interpretation
        assert "0.750" in interpretation
        assert "good explanatory power" in interpretation


class TestCointegrationAnalyzer:
    """Test cases for CointegrationAnalyzer."""
    
    def test_init(self):
        """Test initialization of cointegration analyzer."""
        analyzer = CointegrationAnalyzer()
        assert analyzer.det_order == -1
        assert analyzer.k_ar_diff == 1
        assert analyzer.significance_level == 0.05
    
    def test_init_invalid_det_order(self):
        """Test initialization with invalid deterministic order."""
        with pytest.raises(ValueError, match="det_order must be -1"):
            CointegrationAnalyzer(det_order=2)
    
    def test_init_invalid_significance_level(self):
        """Test initialization with invalid significance level."""
        with pytest.raises(ValueError, match="significance_level must be between 0 and 1"):
            CointegrationAnalyzer(significance_level=1.5)
    
    def test_analysis_logic_success(self, sample_cointegrated_data):
        """Test successful cointegration analysis."""
        analyzer = CointegrationAnalyzer()
        
        result = analyzer._analysis_logic(sample_cointegrated_data)
        
        assert result.analysis_type == "cointegration_analysis"
        assert result.model_parameters is not None
        assert 'n_coint' in result.model_parameters
        assert 'trace_stat' in result.model_parameters
        assert 'eigenvalues' in result.model_parameters
        
        # Check diagnostics
        assert 'n_cointegrating_relationships_final' in result.model_diagnostics
        assert 'trace_statistics' in result.model_diagnostics
        assert 'eigenvalues' in result.model_diagnostics
    
    def test_generate_cointegration_interpretation(self):
        """Test cointegration interpretation generation."""
        analyzer = CointegrationAnalyzer()
        
        # Mock cointegration result
        mock_coint_result = MagicMock()
        mock_coint_result.lr1 = [15.2, 8.1, 2.3]
        
        interpretation = analyzer._generate_cointegration_interpretation(
            n_coint=1, 
            n_series=3, 
            coint_result=mock_coint_result,
            series_names=['A', 'B', 'C']
        )
        
        assert "3 time series" in interpretation
        assert "1 cointegrating" in interpretation
        assert "single long-term equilibrium" in interpretation


class TestGrangerCausalityAnalyzer:
    """Test cases for GrangerCausalityAnalyzer."""
    
    def test_init(self):
        """Test initialization of Granger causality analyzer."""
        analyzer = GrangerCausalityAnalyzer()
        assert analyzer.max_lags == 4
        assert analyzer.significance_level == 0.05
        assert analyzer.test_all_pairs is True
    
    def test_init_invalid_max_lags(self):
        """Test initialization with invalid max_lags."""
        with pytest.raises(ValueError, match="max_lags must be at least 1"):
            GrangerCausalityAnalyzer(max_lags=0)
    
    def test_analysis_logic_success(self, sample_causal_data):
        """Test successful Granger causality analysis."""
        analyzer = GrangerCausalityAnalyzer(max_lags=2)
        
        result = analyzer._analysis_logic(sample_causal_data)
        
        assert result.analysis_type == "granger_causality"
        assert result.model_parameters is not None
        assert 'causality_results' in result.model_parameters
        assert 'significant_relationships' in result.model_parameters
        assert 'causality_matrix' in result.model_parameters
        
        # Check if causal relationship was detected
        causality_results = result.model_parameters['causality_results']
        assert len(causality_results) > 0
        
        # Should detect cause -> effect relationship
        cause_to_effect_found = any(
            'cause → effect' in key for key in causality_results.keys()
        )
        assert cause_to_effect_found
    
    def test_create_causality_matrix(self, sample_causal_data):
        """Test causality matrix creation."""
        analyzer = GrangerCausalityAnalyzer()
        
        # Mock causality results
        causality_results = {
            'cause → effect': {
                'cause': 'cause',
                'effect': 'effect',
                'min_p_value': 0.01
            },
            'effect → cause': {
                'cause': 'effect',
                'effect': 'cause',
                'min_p_value': 0.8
            }
        }
        
        series_names = ['cause', 'effect', 'independent']
        matrix = analyzer._create_causality_matrix(causality_results, series_names)
        
        assert isinstance(matrix, pd.DataFrame)
        assert matrix.shape == (3, 3)
        assert matrix.loc['cause', 'effect'] == 0.01
        assert matrix.loc['effect', 'cause'] == 0.8
    
    def test_generate_granger_interpretation(self):
        """Test Granger causality interpretation generation."""
        analyzer = GrangerCausalityAnalyzer()
        
        significant_relationships = [
            {
                'relationship': 'X → Y',
                'p_value': 0.001,
                'lag': 2
            }
        ]
        
        interpretation = analyzer._generate_granger_interpretation(
            significant_relationships, total_tests=6, series_names=['X', 'Y', 'Z']
        )
        
        assert "6 directional relationships" in interpretation
        assert "3 time series" in interpretation
        assert "1 significant" in interpretation
        assert "X → Y" in interpretation


class TestImpulseResponseAnalyzer:
    """Test cases for ImpulseResponseAnalyzer."""
    
    def test_init(self):
        """Test initialization of impulse response analyzer."""
        analyzer = ImpulseResponseAnalyzer()
        assert analyzer.periods == 10
        assert analyzer.orthogonalized is True
        assert analyzer.confidence_level == 0.95
        assert analyzer.bootstrap_reps == 1000
        assert analyzer.cumulative is False
    
    def test_init_invalid_periods(self):
        """Test initialization with invalid periods."""
        with pytest.raises(ValueError, match="periods must be at least 1"):
            ImpulseResponseAnalyzer(periods=0)
    
    def test_init_invalid_bootstrap_reps(self):
        """Test initialization with invalid bootstrap reps."""
        with pytest.raises(ValueError, match="bootstrap_reps must be at least 100"):
            ImpulseResponseAnalyzer(bootstrap_reps=50)
    
    def test_analysis_logic_success(self, sample_multivariate_data):
        """Test successful impulse response analysis."""
        analyzer = ImpulseResponseAnalyzer(periods=5, cumulative=True)
        
        result = analyzer._analysis_logic(sample_multivariate_data)
        
        assert result.analysis_type == "impulse_response_analysis"
        assert result.model_parameters is not None
        assert 'impulse_responses' in result.model_parameters
        assert 'impulse_response_df' in result.model_parameters
        assert 'significant_responses' in result.model_parameters
        
        # Check impulse response structure
        irf_df = result.model_parameters['impulse_response_df']
        assert isinstance(irf_df, pd.DataFrame)
        assert irf_df.shape[0] == 5  # periods
        
        # Should have responses for all variable pairs
        expected_responses = []
        series_names = list(sample_multivariate_data.columns)
        for shock_var in series_names:
            for response_var in series_names:
                expected_responses.append(f"{shock_var} → {response_var}")
        
        for response in expected_responses:
            assert response in irf_df.columns
        
        # Check cumulative responses
        assert 'cumulative_response_df' in result.model_parameters
        cumulative_df = result.model_parameters['cumulative_response_df']
        assert cumulative_df is not None
    
    def test_generate_irf_interpretation(self):
        """Test impulse response interpretation generation."""
        analyzer = ImpulseResponseAnalyzer()
        
        significant_responses = [
            {
                'relationship': 'X → Y',
                'max_response': 0.5,
                'max_period': 2,
                'total_cumulative_effect': 1.2
            },
            {
                'relationship': 'Y → X',
                'max_response': 0.3,
                'max_period': 1,
                'total_cumulative_effect': 0.8
            }
        ]
        
        interpretation = analyzer._generate_irf_interpretation(
            significant_responses, n_vars=2, periods=10
        )
        
        assert "over 10 periods" in interpretation
        assert "2 shock transmission pathways" in interpretation
        assert "X → Y" in interpretation
        assert "0.500" in interpretation
        assert "period 2" in interpretation
        assert "persistent effects" in interpretation


class TestIntegrationScenarios:
    """Test integration scenarios combining multiple multivariate methods."""
    
    def test_var_to_granger_causality_workflow(self, sample_causal_data):
        """Test workflow from VAR modeling to Granger causality analysis."""
        # First, fit VAR model
        var_forecaster = VARModelForecaster(max_lags=3, forecast_horizon=5)
        var_result = var_forecaster.fit_transform(sample_causal_data)
        
        assert var_result.analysis_type == "VAR_forecasting"
        
        # Then, perform Granger causality analysis
        granger_analyzer = GrangerCausalityAnalyzer(max_lags=3)
        granger_result = granger_analyzer.fit_transform(sample_causal_data)
        
        assert granger_result.analysis_type == "granger_causality"
        
        # Both should work on the same data
        assert var_result.processing_time > 0
        assert granger_result.processing_time > 0
    
    def test_cointegration_to_impulse_response_workflow(self, sample_cointegrated_data):
        """Test workflow from cointegration analysis to impulse response."""
        # First, test for cointegration
        coint_analyzer = CointegrationAnalyzer()
        coint_result = coint_analyzer.fit_transform(sample_cointegrated_data)
        
        assert coint_result.analysis_type == "cointegration_analysis"
        
        # Then, perform impulse response analysis
        irf_analyzer = ImpulseResponseAnalyzer(periods=8)
        irf_result = irf_analyzer.fit_transform(sample_cointegrated_data)
        
        assert irf_result.analysis_type == "impulse_response_analysis"
        
        # Both should provide complementary information
        assert coint_result.interpretation is not None
        assert irf_result.interpretation is not None
    
    def test_multivariate_analysis_pipeline(self, sample_multivariate_data):
        """Test complete multivariate analysis pipeline."""
        analyses = [
            VARModelForecaster(max_lags=2, forecast_horizon=3),
            CointegrationAnalyzer(),
            GrangerCausalityAnalyzer(max_lags=2),
            ImpulseResponseAnalyzer(periods=5)
        ]
        
        results = []
        for analyzer in analyses:
            result = analyzer.fit_transform(sample_multivariate_data)
            results.append(result)
            assert result.processing_time > 0
            assert result.interpretation is not None
            assert len(result.recommendations) > 0
        
        # Check that all analyses completed successfully
        expected_types = [
            "VAR_forecasting",
            "cointegration_analysis", 
            "granger_causality",
            "impulse_response_analysis"
        ]
        
        actual_types = [r.analysis_type for r in results]
        assert actual_types == expected_types


class TestErrorHandling:
    """Test error handling in multivariate analysis."""
    
    def test_insufficient_data_handling(self):
        """Test handling of insufficient data scenarios."""
        # Create minimal dataset
        dates = pd.date_range('2020-01-01', periods=10, freq='D')
        insufficient_data = pd.DataFrame({
            'series1': np.random.randn(10),
            'series2': np.random.randn(10)
        }, index=dates)
        
        analyzers = [
            VARModelForecaster(),
            CointegrationAnalyzer(),
            GrangerCausalityAnalyzer(),
            ImpulseResponseAnalyzer()
        ]
        
        for analyzer in analyzers:
            result = analyzer.fit_transform(insufficient_data)
            # Should handle gracefully with error message
            assert result is not None
            assert result.analysis_type is not None
    
    def test_invalid_data_handling(self):
        """Test handling of invalid data scenarios."""
        # Create data with NaN values
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        invalid_data = pd.DataFrame({
            'series1': np.random.randn(100),
            'series2': np.random.randn(100)
        }, index=dates)
        
        # Introduce NaN values
        invalid_data.iloc[50:60] = np.nan
        
        analyzers = [
            VARModelForecaster(),
            CointegrationAnalyzer(),
            GrangerCausalityAnalyzer(),
            ImpulseResponseAnalyzer()
        ]
        
        for analyzer in analyzers:
            result = analyzer.fit_transform(invalid_data)
            # Should handle gracefully
            assert result is not None
    
    def test_single_series_validation_error(self):
        """Test error when single series provided to multivariate analyzers."""
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        single_series = pd.DataFrame({
            'series1': np.random.randn(100)
        }, index=dates)
        
        analyzers = [
            VARModelForecaster(),
            CointegrationAnalyzer(),
            GrangerCausalityAnalyzer(),
            ImpulseResponseAnalyzer()
        ]
        
        for analyzer in analyzers:
            with pytest.raises(TimeSeriesValidationError, match="Insufficient number of time series"):
                analyzer._validate_multivariate_data(single_series)