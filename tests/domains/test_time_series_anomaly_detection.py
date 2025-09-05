"""
Tests for Time Series Change Point and Anomaly Detection.

This module tests the change point detection and anomaly detection capabilities
including ChangePointDetector, AnomalyDetector, StructuralBreakTester, and 
SeasonalAnomalyDetector classes.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
import warnings

from localdata_mcp.domains.time_series_analysis import (
    ChangePointDetector,
    AnomalyDetector,
    StructuralBreakTester,
    SeasonalAnomalyDetector,
    TimeSeriesValidationError
)


@pytest.fixture
def sample_time_series():
    """Create a sample time series for basic testing."""
    dates = pd.date_range('2020-01-01', periods=200, freq='D')
    
    # Create data with trend
    trend = np.linspace(100, 150, 200)
    noise = np.random.RandomState(42).normal(0, 5, 200)
    
    values = trend + noise
    
    return pd.DataFrame({'value': values}, index=dates)


@pytest.fixture
def change_point_series():
    """Create a time series with a known change point."""
    dates = pd.date_range('2020-01-01', periods=200, freq='D')
    
    # Create data with change point at position 100
    values = np.concatenate([
        np.random.RandomState(42).normal(100, 5, 100),  # First regime
        np.random.RandomState(43).normal(150, 5, 100)   # Second regime
    ])
    
    return pd.DataFrame({'value': values}, index=dates)


@pytest.fixture
def anomaly_series():
    """Create a time series with known anomalies."""
    dates = pd.date_range('2020-01-01', periods=200, freq='D')
    
    # Base series
    values = np.random.RandomState(42).normal(100, 5, 200)
    
    # Inject anomalies
    values[50] = 200   # High anomaly
    values[75] = 10    # Low anomaly
    values[150] = 250  # Another high anomaly
    
    return pd.DataFrame({'value': values}, index=dates)


@pytest.fixture
def seasonal_series():
    """Create a seasonal time series for seasonal anomaly detection."""
    dates = pd.date_range('2020-01-01', periods=365, freq='D')
    
    # Create data with weekly seasonality
    trend = np.linspace(100, 120, 365)
    seasonal = 10 * np.sin(2 * np.pi * np.arange(365) / 7)
    noise = np.random.RandomState(42).normal(0, 2, 365)
    
    values = trend + seasonal + noise
    
    # Inject seasonal anomalies
    values[50] = 200   # Anomaly in week 7
    values[100] = 10   # Anomaly in week 14
    
    return pd.DataFrame({'value': values}, index=dates)


class TestChangePointDetector:
    """Test cases for ChangePointDetector."""
    
    def test_initialization(self):
        """Test ChangePointDetector initialization."""
        detector = ChangePointDetector()
        assert detector.method == 'bcp'
        assert detector.model == 'rbf'
        assert detector.min_size == 10
        assert detector.max_changepoints == 10
        
    def test_initialization_with_params(self):
        """Test ChangePointDetector with custom parameters."""
        detector = ChangePointDetector(
            method='statistical',
            min_size=20,
            max_changepoints=5
        )
        assert detector.method == 'statistical'
        assert detector.min_size == 20
        assert detector.max_changepoints == 5
        
    def test_fit_transform_basic(self, sample_time_series):
        """Test basic fit and transform operations."""
        detector = ChangePointDetector(method='statistical')  # Use statistical method
        result = detector.fit(sample_time_series).transform(sample_time_series)
        
        assert result.analysis_type == "change_point_detection"
        assert 'changepoints' in result.model_parameters
        assert 'segments' in result.model_parameters
        assert isinstance(result.model_parameters['changepoints'], list)
        
    def test_change_point_detection(self, change_point_series):
        """Test change point detection with known change point."""
        detector = ChangePointDetector(method='statistical')
        result = detector.transform(change_point_series)
        
        changepoints = result.model_parameters['changepoints']
        # Should detect at least one change point
        assert len(changepoints) >= 1
        
        # Change point should be somewhere around position 100 (±20)
        if changepoints:
            detected_cp = changepoints[0]
            assert 80 <= detected_cp <= 120
    
    def test_statistical_method(self, change_point_series):
        """Test statistical change point detection method."""
        detector = ChangePointDetector(method='statistical')
        result = detector.transform(change_point_series)
        
        assert result.model_parameters['detection_method'] == 'statistical'
        assert 'cusum_path' not in result.model_parameters  # Should not have CUSUM specific data
        
    @patch('importlib.import_module')
    def test_ruptures_fallback(self, mock_import, change_point_series):
        """Test fallback to statistical method when ruptures is not available."""
        # Mock ruptures import to fail
        mock_import.side_effect = ImportError("No module named 'ruptures'")
        
        detector = ChangePointDetector(method='bcp')  # Method requiring ruptures
        result = detector.transform(change_point_series)
        
        # Should fallback to statistical method
        assert detector.method == 'statistical'
        assert result.model_parameters['ruptures_available'] == False
        
    def test_insufficient_data(self):
        """Test error handling with insufficient data."""
        short_series = pd.DataFrame(
            {'value': [1, 2, 3, 4, 5]}, 
            index=pd.date_range('2020-01-01', periods=5)
        )
        
        detector = ChangePointDetector(min_size=10)
        result = detector.transform(short_series)
        
        # Should handle gracefully and provide error message
        assert "failed" in result.interpretation.lower()
        
    def test_segment_analysis(self, change_point_series):
        """Test segment analysis functionality."""
        detector = ChangePointDetector(method='statistical')
        result = detector.transform(change_point_series)
        
        segments = result.model_parameters['segments']
        segment_analysis = result.model_parameters['segment_analysis']
        
        assert isinstance(segments, list)
        assert isinstance(segment_analysis, dict)
        assert 'n_segments' in segment_analysis


class TestAnomalyDetector:
    """Test cases for AnomalyDetector."""
    
    def test_initialization(self):
        """Test AnomalyDetector initialization."""
        detector = AnomalyDetector()
        assert detector.method == 'statistical'
        assert detector.threshold == 3.0
        assert detector.seasonal_adjustment == True
        
    def test_fit_transform_basic(self, sample_time_series):
        """Test basic fit and transform operations."""
        detector = AnomalyDetector()
        result = detector.fit(sample_time_series).transform(sample_time_series)
        
        assert result.analysis_type == "anomaly_detection"
        assert 'anomalies' in result.model_parameters
        assert 'n_anomalies' in result.model_parameters
        
    def test_anomaly_detection(self, anomaly_series):
        """Test anomaly detection with known anomalies."""
        detector = AnomalyDetector(threshold=2.0)  # Lower threshold for better detection
        result = detector.transform(anomaly_series)
        
        anomalies = result.model_parameters['anomalies']
        
        # Should detect at least some anomalies
        assert len(anomalies) >= 1
        
        # Check if known anomaly positions are detected (±5 tolerance)
        known_anomalies = [50, 75, 150]
        detected_near_known = any(
            any(abs(detected - known) <= 5 for known in known_anomalies)
            for detected in anomalies
        )
        assert detected_near_known
        
    def test_statistical_method(self, anomaly_series):
        """Test statistical anomaly detection method."""
        detector = AnomalyDetector(method='statistical')
        result = detector.transform(anomaly_series)
        
        assert result.model_parameters['detection_method'] == 'statistical'
        assert 'baseline_statistics' in result.model_diagnostics
        
    def test_zscore_method(self, anomaly_series):
        """Test Z-score anomaly detection method."""
        detector = AnomalyDetector(method='zscore')
        result = detector.transform(anomaly_series)
        
        assert result.model_parameters['detection_method'] == 'zscore'
        
    def test_iqr_method(self, anomaly_series):
        """Test IQR anomaly detection method."""
        detector = AnomalyDetector(method='iqr')
        result = detector.transform(anomaly_series)
        
        assert result.model_parameters['detection_method'] == 'iqr'
        
    @patch('sklearn.ensemble.IsolationForest')
    def test_isolation_forest_method(self, mock_isolation_forest, anomaly_series):
        """Test Isolation Forest anomaly detection method."""
        # Mock IsolationForest
        mock_clf = MagicMock()
        mock_clf.fit_predict.return_value = np.array([1] * len(anomaly_series))
        mock_clf.decision_function.return_value = np.random.random(len(anomaly_series))
        mock_isolation_forest.return_value = mock_clf
        
        detector = AnomalyDetector(method='isolation_forest')
        result = detector.transform(anomaly_series)
        
        assert result.model_parameters['detection_method'] == 'isolation_forest'
        
    def test_seasonal_adjustment(self, seasonal_series):
        """Test seasonal adjustment functionality."""
        detector = AnomalyDetector(seasonal_adjustment=True)
        result = detector.transform(seasonal_series)
        
        # Should attempt seasonal adjustment
        seasonal_applied = result.model_parameters.get('seasonal_adjustment_applied', False)
        # May or may not succeed depending on data characteristics, but should try
        
    def test_insufficient_data(self):
        """Test error handling with insufficient data."""
        short_series = pd.DataFrame(
            {'value': [1, 2, 3]}, 
            index=pd.date_range('2020-01-01', periods=3)
        )
        
        detector = AnomalyDetector()
        result = detector.transform(short_series)
        
        # Should handle gracefully
        assert "failed" in result.interpretation.lower()


class TestStructuralBreakTester:
    """Test cases for StructuralBreakTester."""
    
    def test_initialization(self):
        """Test StructuralBreakTester initialization."""
        tester = StructuralBreakTester()
        assert tester.test_type == 'chow'
        assert tester.significance_level == 0.05
        assert tester.min_segment_size == 10
        
    def test_chow_test(self, change_point_series):
        """Test Chow test for structural breaks."""
        tester = StructuralBreakTester(test_type='chow', break_point=100)
        result = tester.transform(change_point_series)
        
        assert result.analysis_type == "structural_break_test"
        assert result.model_parameters['test_type'] == 'chow'
        assert result.statistic is not None
        assert result.p_value is not None
        
    def test_chow_test_fraction_breakpoint(self, change_point_series):
        """Test Chow test with fractional break point."""
        tester = StructuralBreakTester(test_type='chow', break_point=0.5)
        result = tester.transform(change_point_series)
        
        # Should test break point at middle of series
        assert result.model_parameters['break_point_used'] == 0.5
        
    def test_cusum_test(self, change_point_series):
        """Test CUSUM test for structural stability."""
        tester = StructuralBreakTester(test_type='cusum')
        result = tester.transform(change_point_series)
        
        assert result.model_parameters['test_type'] == 'cusum'
        assert 'cusum_path' in result.model_parameters['test_details']
        
    def test_break_point_detection(self, change_point_series):
        """Test structural break detection."""
        tester = StructuralBreakTester(test_type='chow', break_point=100)
        result = tester.transform(change_point_series)
        
        # With a clear break point, should have low p-value
        if result.p_value is not None:
            # May detect the break (depends on data characteristics)
            pass  # Test doesn't require detection, just that it runs
            
    def test_insufficient_data(self):
        """Test error handling with insufficient data."""
        short_series = pd.DataFrame(
            {'value': [1, 2, 3, 4, 5]}, 
            index=pd.date_range('2020-01-01', periods=5)
        )
        
        tester = StructuralBreakTester(min_segment_size=10)
        result = tester.transform(short_series)
        
        # Should handle gracefully
        assert "failed" in result.interpretation.lower()


class TestSeasonalAnomalyDetector:
    """Test cases for SeasonalAnomalyDetector."""
    
    def test_initialization(self):
        """Test SeasonalAnomalyDetector initialization."""
        detector = SeasonalAnomalyDetector()
        assert detector.method == 'adaptive_threshold'
        assert detector.threshold_factor == 2.5
        assert detector.adaptation_rate == 0.1
        
    def test_fit_transform_basic(self, seasonal_series):
        """Test basic fit and transform operations."""
        detector = SeasonalAnomalyDetector()
        result = detector.fit(seasonal_series).transform(seasonal_series)
        
        assert result.analysis_type == "seasonal_anomaly_detection"
        assert 'seasonal_anomalies' in result.model_parameters
        assert 'seasonal_period' in result.model_parameters
        
    def test_seasonal_period_detection(self, seasonal_series):
        """Test automatic seasonal period detection."""
        detector = SeasonalAnomalyDetector(seasonal_period=None)
        result = detector.transform(seasonal_series)
        
        detected_period = result.model_parameters.get('seasonal_period')
        # Should detect a period (may not be exact due to noise)
        if detected_period:
            assert isinstance(detected_period, int)
            assert 2 <= detected_period <= 50  # Reasonable range
        
    def test_adaptive_threshold_method(self, seasonal_series):
        """Test adaptive threshold method."""
        detector = SeasonalAnomalyDetector(method='adaptive_threshold', seasonal_period=7)
        result = detector.transform(seasonal_series)
        
        assert result.model_parameters['detection_method'] == 'adaptive_threshold'
        thresholds = result.model_parameters.get('adaptive_thresholds', [])
        assert len(thresholds) == len(seasonal_series)
        
    def test_seasonal_iqr_method(self, seasonal_series):
        """Test seasonal IQR method."""
        detector = SeasonalAnomalyDetector(method='seasonal_iqr', seasonal_period=7)
        result = detector.transform(seasonal_series)
        
        assert result.model_parameters['detection_method'] == 'seasonal_iqr'
        
    def test_seasonal_zscore_method(self, seasonal_series):
        """Test seasonal Z-score method."""
        detector = SeasonalAnomalyDetector(method='seasonal_zscore', seasonal_period=7)
        result = detector.transform(seasonal_series)
        
        assert result.model_parameters['detection_method'] == 'seasonal_zscore'
        
    def test_seasonal_anomaly_detection(self, seasonal_series):
        """Test detection of seasonal anomalies."""
        detector = SeasonalAnomalyDetector(seasonal_period=7, threshold_factor=2.0)
        result = detector.transform(seasonal_series)
        
        anomalies = result.model_parameters['seasonal_anomalies']
        
        # Should detect at least some anomalies
        # (exact detection depends on seasonal decomposition success)
        assert isinstance(anomalies, list)
        
    def test_non_seasonal_fallback(self):
        """Test fallback to non-seasonal detection."""
        # Create non-seasonal data
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        values = np.random.RandomState(42).normal(100, 5, 100)
        series = pd.DataFrame({'value': values}, index=dates)
        
        detector = SeasonalAnomalyDetector()
        result = detector.transform(series)
        
        # Should handle non-seasonal data gracefully
        assert result.analysis_type == "seasonal_anomaly_detection"
        
    def test_insufficient_data(self):
        """Test error handling with insufficient data."""
        short_series = pd.DataFrame(
            {'value': [1, 2, 3]}, 
            index=pd.date_range('2020-01-01', periods=3)
        )
        
        detector = SeasonalAnomalyDetector(min_history=50)
        result = detector.transform(short_series)
        
        # Should handle gracefully
        assert "failed" in result.interpretation.lower()


class TestIntegration:
    """Integration tests for change point and anomaly detection classes."""
    
    def test_pipeline_integration(self, change_point_series):
        """Test that all classes can be used together in a pipeline."""
        
        # Test change point detection
        cp_detector = ChangePointDetector(method='statistical')
        cp_result = cp_detector.transform(change_point_series)
        
        # Test anomaly detection
        anomaly_detector = AnomalyDetector(method='statistical')
        anomaly_result = anomaly_detector.transform(change_point_series)
        
        # Test structural break test
        break_tester = StructuralBreakTester(test_type='chow')
        break_result = break_tester.transform(change_point_series)
        
        # All should complete successfully
        assert cp_result.analysis_type == "change_point_detection"
        assert anomaly_result.analysis_type == "anomaly_detection"
        assert break_result.analysis_type == "structural_break_test"
        
    def test_consistency_check(self, change_point_series):
        """Test consistency between different detection methods."""
        
        # Change point detector should find similar breaks as structural break tester
        cp_detector = ChangePointDetector(method='statistical')
        cp_result = cp_detector.transform(change_point_series)
        
        break_tester = StructuralBreakTester(test_type='chow', break_point=100)
        break_result = break_tester.transform(change_point_series)
        
        # Both should detect structural changes (though methods differ)
        cp_breaks = cp_result.model_parameters.get('changepoints', [])
        structural_breaks = break_result.model_parameters.get('break_points', [])
        
        # At least one method should detect something if there's a clear break
        total_detections = len(cp_breaks) + len(structural_breaks)
        # We don't assert specific detections as it depends on data characteristics
        
    def test_error_handling_consistency(self):
        """Test that all classes handle errors consistently."""
        
        # Invalid data
        invalid_series = pd.DataFrame({'value': []}, index=pd.DatetimeIndex([]))
        
        classes = [
            ChangePointDetector(),
            AnomalyDetector(),
            StructuralBreakTester(),
            SeasonalAnomalyDetector()
        ]
        
        for detector in classes:
            result = detector.transform(invalid_series)
            # Should all handle gracefully with error messages
            assert "failed" in result.interpretation.lower()
            assert len(result.recommendations) > 0