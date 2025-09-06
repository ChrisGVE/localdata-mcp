"""
Time Series Integration Tests - Comprehensive integration testing for the complete Time Series Analysis Domain.

This test suite validates:
1. All time series tools working together in complex workflows
2. Cross-component compatibility and result consistency  
3. End-to-end pipeline testing from data ingestion to analysis
4. Real-world scenario testing with synthetic but realistic data
5. Cross-domain integration with pattern recognition and statistical analysis
6. Streaming architecture performance with high-frequency data
7. Memory management and error handling under load

Test Categories:
- Individual Tool Integration Tests
- Cross-Component Workflow Tests  
- Pipeline Integration Tests
- Performance Benchmarking Tests
- Cross-Domain Integration Tests
- Error Handling and Edge Case Tests
"""

import pytest
import numpy as np
import pandas as pd
import json
import time
import tempfile
import os
from pathlib import Path
from typing import Dict, Any, List
import sqlite3

# Import test utilities and fixtures
from ..test_utils import (
    create_synthetic_time_series, 
    create_multivariate_time_series,
    validate_time_series_result,
    measure_performance,
    generate_high_frequency_data,
    create_test_database
)


class TestTimeSeriesIntegration:
    """Comprehensive integration tests for Time Series Analysis Domain."""
    
    @pytest.fixture(autouse=True)
    def setup_test_environment(self, tmp_path):
        """Set up test environment with synthetic data and database connections."""
        self.test_dir = tmp_path
        self.db_connections = {}
        self.performance_results = {}
        
        # Create various test datasets
        self.create_test_datasets()
        
        yield
        
        # Cleanup
        self.cleanup_test_environment()
    
    def create_test_datasets(self):
        """Create comprehensive test datasets for different scenarios."""
        
        # Dataset 1: Basic daily time series with trend and seasonality
        self.daily_ts = create_synthetic_time_series(
            start_date='2020-01-01',
            end_date='2023-12-31',
            freq='D',
            trend=0.1,
            seasonality=True,
            seasonal_periods=365,
            noise_level=0.1,
            anomalies=0.02
        )
        
        # Dataset 2: High-frequency hourly data for performance testing
        self.hourly_ts = generate_high_frequency_data(
            start_date='2023-01-01',
            end_date='2023-12-31', 
            freq='H',
            n_series=1,
            complexity='high'
        )
        
        # Dataset 3: Multivariate time series for VAR/Granger testing
        self.multivariate_ts = create_multivariate_time_series(
            start_date='2022-01-01',
            end_date='2023-12-31',
            freq='D',
            n_variables=4,
            cross_correlations=True,
            causality_patterns=True
        )
        
        # Dataset 4: Time series with known change points
        self.changepoint_ts = create_synthetic_time_series(
            start_date='2020-01-01',
            end_date='2023-12-31',
            freq='D',
            change_points=[('2021-01-01', 'level'), ('2022-06-01', 'trend')],
            trend=0.05,
            seasonality=True,
            noise_level=0.05
        )
        
        # Dataset 5: Short series for edge case testing
        self.short_ts = create_synthetic_time_series(
            start_date='2023-01-01',
            end_date='2023-02-01',
            freq='D',
            trend=0.0,
            seasonality=False,
            noise_level=0.1
        )
        
        # Create test databases
        self.create_test_databases()
    
    def create_test_databases(self):
        """Create test databases with time series data."""
        
        # SQLite database with daily time series
        daily_db_path = self.test_dir / "daily_timeseries.db"
        with sqlite3.connect(daily_db_path) as conn:
            self.daily_ts.to_sql('daily_data', conn, index=True, if_exists='replace')
        self.db_connections['daily'] = str(daily_db_path)
        
        # SQLite database with hourly high-frequency data  
        hourly_db_path = self.test_dir / "hourly_timeseries.db"
        with sqlite3.connect(hourly_db_path) as conn:
            self.hourly_ts.to_sql('hourly_data', conn, index=True, if_exists='replace')
        self.db_connections['hourly'] = str(hourly_db_path)
        
        # SQLite database with multivariate data
        multi_db_path = self.test_dir / "multivariate_timeseries.db"
        with sqlite3.connect(multi_db_path) as conn:
            self.multivariate_ts.to_sql('multi_data', conn, index=True, if_exists='replace')
        self.db_connections['multivariate'] = str(multi_db_path)
        
        # SQLite database with change point data
        cp_db_path = self.test_dir / "changepoint_timeseries.db"
        with sqlite3.connect(cp_db_path) as conn:
            self.changepoint_ts.to_sql('changepoint_data', conn, index=True, if_exists='replace')
        self.db_connections['changepoint'] = str(cp_db_path)
    
    def cleanup_test_environment(self):
        """Clean up test environment and resources."""
        # Close database connections and remove temporary files
        for db_path in self.db_connections.values():
            if os.path.exists(db_path):
                os.unlink(db_path)


class TestIndividualToolIntegration(TestTimeSeriesIntegration):
    """Test individual time series tools integration with LocalData MCP."""
    
    def test_basic_time_series_analysis_integration(self):
        """Test basic time series analysis tool integration."""
        from src.localdata_mcp.localdata_mcp import _db_manager
        
        # Connect to test database
        conn_result = _db_manager.connect_database(
            name="test_daily",
            db_type="sqlite",
            conn_string=self.db_connections['daily']
        )
        assert json.loads(conn_result)["success"] is True
        
        # Perform basic analysis
        start_time = time.time()
        result = _db_manager.analyze_time_series_basic(
            name="test_daily",
            table_name="daily_data",
            date_column="date",
            value_column="value"
        )
        execution_time = time.time() - start_time
        
        # Validate result
        result_data = json.loads(result)
        assert 'analysis_type' in result_data
        assert result_data['analysis_type'] == 'basic'
        assert 'trend_analysis' in result_data
        assert 'seasonality_analysis' in result_data
        assert 'stationarity_tests' in result_data
        assert 'autocorrelation_analysis' in result_data
        
        # Performance validation
        assert execution_time < 30.0  # Should complete within 30 seconds
        self.performance_results['basic_analysis'] = execution_time
        
        print(f"✅ Basic time series analysis completed in {execution_time:.2f}s")
    
    def test_arima_forecasting_integration(self):
        """Test ARIMA forecasting tool integration."""
        from src.localdata_mcp.localdata_mcp import _db_manager
        
        # Connect to test database
        conn_result = _db_manager.connect_database(
            name="test_daily",
            db_type="sqlite", 
            conn_string=self.db_connections['daily']
        )
        assert json.loads(conn_result)["success"] is True
        
        # Perform ARIMA forecasting
        start_time = time.time()
        result = _db_manager.forecast_arima(
            name="test_daily",
            table_name="daily_data", 
            date_column="date",
            value_column="value",
            forecast_steps=30,
            auto_arima=True
        )
        execution_time = time.time() - start_time
        
        # Validate result
        result_data = json.loads(result)
        assert 'model_type' in result_data
        assert result_data['model_type'] == 'ARIMA'
        assert 'forecast_values' in result_data
        assert 'forecast_index' in result_data
        assert 'confidence_intervals' in result_data
        assert 'model_metrics' in result_data
        assert len(result_data['forecast_values']) == 30
        
        # Validate forecast quality
        assert 'aic' in result_data['model_metrics']
        assert 'rmse' in result_data['model_metrics']
        assert result_data['model_metrics']['rmse'] > 0
        
        # Performance validation
        assert execution_time < 60.0  # Should complete within 60 seconds
        self.performance_results['arima_forecasting'] = execution_time
        
        print(f"✅ ARIMA forecasting completed in {execution_time:.2f}s")
    
    def test_exponential_smoothing_integration(self):
        """Test exponential smoothing forecasting tool integration."""
        from src.localdata_mcp.localdata_mcp import _db_manager
        
        # Connect to test database
        conn_result = _db_manager.connect_database(
            name="test_daily",
            db_type="sqlite",
            conn_string=self.db_connections['daily']
        )
        assert json.loads(conn_result)["success"] is True
        
        # Test different smoothing methods
        methods = ['auto', 'simple', 'double', 'triple']
        
        for method in methods:
            start_time = time.time()
            result = _db_manager.forecast_exponential_smoothing(
                name="test_daily",
                table_name="daily_data",
                date_column="date", 
                value_column="value",
                method=method,
                forecast_steps=30
            )
            execution_time = time.time() - start_time
            
            # Validate result
            result_data = json.loads(result)
            assert 'model_type' in result_data
            assert result_data['model_type'] == 'ExponentialSmoothing'
            assert 'forecast_values' in result_data
            assert len(result_data['forecast_values']) == 30
            
            # Performance validation
            assert execution_time < 30.0
            
            print(f"✅ {method} exponential smoothing completed in {execution_time:.2f}s")
    
    def test_anomaly_detection_integration(self):
        """Test time series anomaly detection tool integration."""
        from src.localdata_mcp.localdata_mcp import _db_manager
        
        # Connect to test database
        conn_result = _db_manager.connect_database(
            name="test_daily",
            db_type="sqlite",
            conn_string=self.db_connections['daily']
        )
        assert json.loads(conn_result)["success"] is True
        
        # Test different anomaly detection methods
        methods = ['statistical', 'isolation_forest']
        
        for method in methods:
            start_time = time.time()
            result = _db_manager.detect_time_series_anomalies(
                name="test_daily",
                table_name="daily_data",
                date_column="date",
                value_column="value", 
                method=method,
                contamination=0.05
            )
            execution_time = time.time() - start_time
            
            # Validate result
            result_data = json.loads(result)
            assert 'detection_method' in result_data
            assert result_data['detection_method'] == method
            assert 'anomaly_indices' in result_data
            assert 'anomaly_scores' in result_data
            assert 'anomaly_periods' in result_data
            
            # Validate detection quality
            anomaly_count = len(result_data['anomaly_indices'])
            total_points = result_data['series_info']['length']
            anomaly_rate = anomaly_count / total_points
            
            # Should detect reasonable number of anomalies (1-10%)
            assert 0.01 <= anomaly_rate <= 0.10
            
            # Performance validation
            assert execution_time < 30.0
            
            print(f"✅ {method} anomaly detection completed in {execution_time:.2f}s, found {anomaly_count} anomalies")
    
    def test_change_point_detection_integration(self):
        """Test change point detection tool integration."""
        from src.localdata_mcp.localdata_mcp import _db_manager
        
        # Connect to test database with known change points
        conn_result = _db_manager.connect_database(
            name="test_changepoint",
            db_type="sqlite",
            conn_string=self.db_connections['changepoint']
        )
        assert json.loads(conn_result)["success"] is True
        
        # Test different change point detection methods
        methods = ['cusum']  # Add 'pelt' if ruptures is available
        
        for method in methods:
            start_time = time.time()
            result = _db_manager.detect_change_points(
                name="test_changepoint",
                table_name="changepoint_data",
                date_column="date",
                value_column="value",
                method=method,
                min_size=10
            )
            execution_time = time.time() - start_time
            
            # Validate result
            result_data = json.loads(result)
            assert 'detection_method' in result_data
            assert result_data['detection_method'] == method
            assert 'change_points' in result_data
            assert 'change_point_dates' in result_data
            assert 'segments_analysis' in result_data
            
            # Should detect some change points (we injected 2)
            change_point_count = len(result_data['change_points'])
            assert change_point_count > 0
            
            # Performance validation  
            assert execution_time < 30.0
            
            print(f"✅ {method} change point detection completed in {execution_time:.2f}s, found {change_point_count} change points")
    
    def test_multivariate_analysis_integration(self):
        """Test multivariate time series analysis tool integration."""
        from src.localdata_mcp.localdata_mcp import _db_manager
        
        # Connect to multivariate test database
        conn_result = _db_manager.connect_database(
            name="test_multivariate",
            db_type="sqlite",
            conn_string=self.db_connections['multivariate']
        )
        assert json.loads(conn_result)["success"] is True
        
        # Test VAR analysis
        start_time = time.time()
        var_result = _db_manager.analyze_multivariate_time_series(
            name="test_multivariate",
            table_name="multi_data",
            analysis_type="var",
            max_lags=5
        )
        var_execution_time = time.time() - start_time
        
        # Validate VAR result
        var_data = json.loads(var_result)
        assert 'model_type' in var_data
        assert var_data['model_type'] == 'VAR'
        assert 'optimal_lags' in var_data
        assert 'aic' in var_data
        assert 'forecast' in var_data
        assert 'variables' in var_data
        
        # Test Granger causality analysis
        start_time = time.time()
        granger_result = _db_manager.analyze_multivariate_time_series(
            name="test_multivariate",
            table_name="multi_data",
            analysis_type="granger",
            max_lags=3
        )
        granger_execution_time = time.time() - start_time
        
        # Validate Granger result
        granger_data = json.loads(granger_result)
        assert 'causality_results' in granger_data
        assert 'variables' in granger_data
        
        # Should have causality tests for variable pairs
        causality_tests = granger_data['causality_results']
        assert len(causality_tests) > 0
        
        # Performance validation
        assert var_execution_time < 60.0
        assert granger_execution_time < 60.0
        
        print(f"✅ VAR analysis completed in {var_execution_time:.2f}s")
        print(f"✅ Granger causality analysis completed in {granger_execution_time:.2f}s")


class TestCrossComponentWorkflows(TestTimeSeriesIntegration):
    """Test complex workflows combining multiple time series tools."""
    
    def test_comprehensive_time_series_pipeline(self):
        """Test complete time series analysis pipeline."""
        from src.localdata_mcp.localdata_mcp import _db_manager
        
        # Connect to test database
        conn_result = _db_manager.connect_database(
            name="test_pipeline",
            db_type="sqlite",
            conn_string=self.db_connections['daily']
        )
        assert json.loads(conn_result)["success"] is True
        
        pipeline_results = {}
        total_start_time = time.time()
        
        # Step 1: Basic Analysis
        print("📊 Step 1: Basic Time Series Analysis")
        basic_result = _db_manager.analyze_time_series_basic(
            name="test_pipeline",
            table_name="daily_data", 
            date_column="date",
            value_column="value"
        )
        pipeline_results['basic'] = json.loads(basic_result)
        
        # Step 2: Anomaly Detection  
        print("🔍 Step 2: Anomaly Detection")
        anomaly_result = _db_manager.detect_time_series_anomalies(
            name="test_pipeline",
            table_name="daily_data",
            date_column="date",
            value_column="value",
            method="statistical"
        )
        pipeline_results['anomalies'] = json.loads(anomaly_result)
        
        # Step 3: Change Point Detection
        print("📈 Step 3: Change Point Detection") 
        changepoint_result = _db_manager.detect_change_points(
            name="test_pipeline",
            table_name="daily_data",
            date_column="date",
            value_column="value",
            method="cusum"
        )
        pipeline_results['changepoints'] = json.loads(changepoint_result)
        
        # Step 4: ARIMA Forecasting
        print("🔮 Step 4: ARIMA Forecasting")
        arima_result = _db_manager.forecast_arima(
            name="test_pipeline", 
            table_name="daily_data",
            date_column="date",
            value_column="value",
            forecast_steps=30,
            auto_arima=True
        )
        pipeline_results['arima_forecast'] = json.loads(arima_result)
        
        # Step 5: Exponential Smoothing Forecasting
        print("📉 Step 5: Exponential Smoothing Forecasting")
        smoothing_result = _db_manager.forecast_exponential_smoothing(
            name="test_pipeline",
            table_name="daily_data", 
            date_column="date",
            value_column="value",
            method="auto",
            forecast_steps=30
        )
        pipeline_results['smoothing_forecast'] = json.loads(smoothing_result)
        
        total_execution_time = time.time() - total_start_time
        
        # Validate pipeline results
        assert all(result for result in pipeline_results.values())
        
        # Cross-validate results
        self._validate_pipeline_consistency(pipeline_results)
        
        # Performance validation
        assert total_execution_time < 180.0  # Complete pipeline within 3 minutes
        
        print(f"✅ Complete time series pipeline completed in {total_execution_time:.2f}s")
        
        return pipeline_results
    
    def _validate_pipeline_consistency(self, results):
        """Validate consistency across pipeline results."""
        
        # Check that all analyses used the same data length
        data_lengths = []
        for key, result in results.items():
            if 'series_info' in result and 'length' in result['series_info']:
                data_lengths.append(result['series_info']['length'])
        
        assert len(set(data_lengths)) == 1, "Inconsistent data lengths across analyses"
        
        # Check forecast consistency
        if 'arima_forecast' in results and 'smoothing_forecast' in results:
            arima_forecast = results['arima_forecast']['forecast_values']
            smoothing_forecast = results['smoothing_forecast']['forecast_values']
            
            # Forecasts should be reasonable in scale relative to each other
            arima_mean = np.mean(arima_forecast)
            smoothing_mean = np.mean(smoothing_forecast)
            
            ratio = arima_mean / smoothing_mean if smoothing_mean != 0 else float('inf')
            assert 0.1 < ratio < 10.0, "Forecasts are inconsistent in scale"
        
        print("✅ Pipeline consistency validation passed")
    
    def test_multivariate_analysis_workflow(self):
        """Test comprehensive multivariate time series workflow."""
        from src.localdata_mcp.localdata_mcp import _db_manager
        
        # Connect to multivariate test database
        conn_result = _db_manager.connect_database(
            name="test_multi_workflow",
            db_type="sqlite", 
            conn_string=self.db_connections['multivariate']
        )
        assert json.loads(conn_result)["success"] is True
        
        workflow_results = {}
        start_time = time.time()
        
        # Step 1: VAR Model Analysis
        print("📊 Multivariate Step 1: VAR Model Analysis")
        var_result = _db_manager.analyze_multivariate_time_series(
            name="test_multi_workflow",
            table_name="multi_data",
            analysis_type="var",
            max_lags=5
        )
        workflow_results['var'] = json.loads(var_result)
        
        # Step 2: Granger Causality Testing
        print("🔗 Multivariate Step 2: Granger Causality Testing") 
        granger_result = _db_manager.analyze_multivariate_time_series(
            name="test_multi_workflow",
            table_name="multi_data",
            analysis_type="granger",
            max_lags=3
        )
        workflow_results['granger'] = json.loads(granger_result)
        
        # Step 3: Individual series analysis for each variable
        print("📈 Multivariate Step 3: Individual Series Analysis")
        columns = workflow_results['var']['variables']
        individual_analyses = {}
        
        for column in columns[:2]:  # Test first 2 variables to save time
            query = f"SELECT date, {column} as value FROM multi_data ORDER BY date"
            
            basic_result = _db_manager.analyze_time_series_basic(
                name="test_multi_workflow",
                query=query,
                date_column="date",
                value_column="value"
            )
            individual_analyses[column] = json.loads(basic_result)
        
        workflow_results['individual'] = individual_analyses
        
        total_execution_time = time.time() - start_time
        
        # Validate workflow results
        assert workflow_results['var']['model_type'] == 'VAR'
        assert len(workflow_results['granger']['causality_results']) > 0
        assert len(workflow_results['individual']) == 2
        
        # Performance validation
        assert total_execution_time < 120.0  # Complete within 2 minutes
        
        print(f"✅ Multivariate workflow completed in {total_execution_time:.2f}s")


class TestPerformanceBenchmarking(TestTimeSeriesIntegration):
    """Performance benchmarking tests for high-frequency data scenarios."""
    
    def test_high_frequency_data_performance(self):
        """Test performance with high-frequency time series data."""
        from src.localdata_mcp.localdata_mcp import _db_manager
        
        # Connect to high-frequency test database
        conn_result = _db_manager.connect_database(
            name="test_high_freq",
            db_type="sqlite",
            conn_string=self.db_connections['hourly']
        )
        assert json.loads(conn_result)["success"] is True
        
        performance_metrics = {}
        
        # Test 1: Basic Analysis on High-Frequency Data
        print("⚡ Performance Test 1: Basic Analysis on High-Frequency Data")
        start_time = time.time()
        basic_result = _db_manager.analyze_time_series_basic(
            name="test_high_freq",
            table_name="hourly_data",
            date_column="date", 
            value_column="value"
        )
        basic_time = time.time() - start_time
        performance_metrics['basic_hf'] = basic_time
        
        # Validate result and performance
        result_data = json.loads(basic_result)
        assert 'trend_analysis' in result_data
        assert basic_time < 45.0  # Should complete within 45 seconds for high-frequency data
        
        # Test 2: Anomaly Detection on High-Frequency Data
        print("⚡ Performance Test 2: Anomaly Detection on High-Frequency Data")
        start_time = time.time()
        anomaly_result = _db_manager.detect_time_series_anomalies(
            name="test_high_freq",
            table_name="hourly_data",
            date_column="date",
            value_column="value",
            method="statistical"
        )
        anomaly_time = time.time() - start_time
        performance_metrics['anomaly_hf'] = anomaly_time
        
        # Validate result and performance
        result_data = json.loads(anomaly_result)
        assert 'anomaly_indices' in result_data
        assert anomaly_time < 60.0  # Should complete within 60 seconds
        
        # Test 3: ARIMA Forecasting on Subset of High-Frequency Data
        print("⚡ Performance Test 3: ARIMA Forecasting on High-Frequency Subset")
        start_time = time.time()
        forecast_result = _db_manager.forecast_arima(
            name="test_high_freq",
            table_name="hourly_data",
            date_column="date",
            value_column="value",
            forecast_steps=24,  # 24 hour forecast
            auto_arima=True,
            sample_size=2000  # Limit sample size for performance
        )
        forecast_time = time.time() - start_time  
        performance_metrics['forecast_hf'] = forecast_time
        
        # Validate result and performance
        result_data = json.loads(forecast_result)
        assert 'forecast_values' in result_data
        assert len(result_data['forecast_values']) == 24
        assert forecast_time < 90.0  # Should complete within 90 seconds
        
        # Performance Summary
        total_time = sum(performance_metrics.values())
        print(f"\n📊 High-Frequency Performance Summary:")
        print(f"   Basic Analysis: {basic_time:.2f}s")
        print(f"   Anomaly Detection: {anomaly_time:.2f}s")
        print(f"   ARIMA Forecasting: {forecast_time:.2f}s")
        print(f"   Total Time: {total_time:.2f}s")
        
        # Overall performance validation
        assert total_time < 180.0  # Complete all high-frequency tests within 3 minutes
        
        print("✅ High-frequency performance benchmarking completed")
        
        return performance_metrics
    
    def test_memory_usage_validation(self):
        """Test memory usage under different data sizes.""" 
        import psutil
        import os
        
        from src.localdata_mcp.localdata_mcp import _db_manager
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / (1024 * 1024)  # MB
        
        memory_metrics = {
            'initial_memory_mb': initial_memory,
            'peak_memory_mb': initial_memory,
            'final_memory_mb': initial_memory
        }
        
        # Connect to high-frequency database
        conn_result = _db_manager.connect_database(
            name="test_memory",
            db_type="sqlite",
            conn_string=self.db_connections['hourly']
        )
        assert json.loads(conn_result)["success"] is True
        
        # Perform memory-intensive operations
        operations = [
            ("Basic Analysis", lambda: _db_manager.analyze_time_series_basic(
                name="test_memory", table_name="hourly_data", 
                date_column="date", value_column="value")),
            ("Anomaly Detection", lambda: _db_manager.detect_time_series_anomalies(
                name="test_memory", table_name="hourly_data",
                date_column="date", value_column="value", method="isolation_forest")),
            ("Change Point Detection", lambda: _db_manager.detect_change_points(
                name="test_memory", table_name="hourly_data", 
                date_column="date", value_column="value"))
        ]
        
        for operation_name, operation_func in operations:
            print(f"🔍 Memory Test: {operation_name}")
            
            # Measure memory before operation
            before_memory = process.memory_info().rss / (1024 * 1024)
            
            # Perform operation
            result = operation_func()
            
            # Measure memory after operation  
            after_memory = process.memory_info().rss / (1024 * 1024)
            
            # Update peak memory
            memory_metrics['peak_memory_mb'] = max(memory_metrics['peak_memory_mb'], after_memory)
            
            # Validate operation completed successfully
            result_data = json.loads(result)
            assert 'analysis_type' in result_data or 'detection_method' in result_data or 'model_type' in result_data
            
            print(f"   Memory: {before_memory:.1f}MB → {after_memory:.1f}MB (Δ{after_memory-before_memory:+.1f}MB)")
        
        # Final memory measurement
        final_memory = process.memory_info().rss / (1024 * 1024)
        memory_metrics['final_memory_mb'] = final_memory
        
        # Memory usage validation
        memory_increase = memory_metrics['peak_memory_mb'] - memory_metrics['initial_memory_mb']
        assert memory_increase < 1000, f"Memory usage increased by {memory_increase:.1f}MB (limit: 1000MB)"
        
        print(f"\n📊 Memory Usage Summary:")
        print(f"   Initial: {initial_memory:.1f}MB")
        print(f"   Peak: {memory_metrics['peak_memory_mb']:.1f}MB")
        print(f"   Final: {final_memory:.1f}MB")
        print(f"   Max Increase: {memory_increase:.1f}MB")
        
        print("✅ Memory usage validation completed")
        
        return memory_metrics
    
    def test_streaming_architecture_validation(self):
        """Test streaming architecture effectiveness."""
        from src.localdata_mcp.localdata_mcp import _db_manager
        
        # Connect to test database
        conn_result = _db_manager.connect_database(
            name="test_streaming",
            db_type="sqlite", 
            conn_string=self.db_connections['hourly']
        )
        assert json.loads(conn_result)["success"] is True
        
        streaming_metrics = {}
        
        # Test streaming status monitoring
        print("📡 Testing Streaming Status Monitoring")
        start_time = time.time()
        status_result = _db_manager.get_streaming_status()
        status_time = time.time() - start_time
        
        status_data = json.loads(status_result)
        assert 'memory_status' in status_data
        assert 'streaming_buffers' in status_data
        assert 'performance_config' in status_data
        
        # Tool discovery should be sub-100ms
        assert status_time < 0.1, f"Tool discovery took {status_time:.3f}s (limit: 0.1s)"
        streaming_metrics['tool_discovery_time'] = status_time
        
        # Test memory management
        print("🧹 Testing Memory Management")
        start_time = time.time()
        memory_result = _db_manager.manage_memory_bounds()
        memory_time = time.time() - start_time
        
        memory_data = json.loads(memory_result)
        assert 'memory_status' in memory_data
        assert 'actions_taken' in memory_data
        
        streaming_metrics['memory_management_time'] = memory_time
        
        # Test with actual data processing
        print("⚡ Testing Streaming with Data Processing")
        start_time = time.time()
        
        # Perform analysis that should use streaming
        basic_result = _db_manager.analyze_time_series_basic(
            name="test_streaming",
            table_name="hourly_data",
            date_column="date",
            value_column="value"
        )
        
        processing_time = time.time() - start_time
        streaming_metrics['streaming_processing_time'] = processing_time
        
        # Validate result
        result_data = json.loads(basic_result)
        assert 'trend_analysis' in result_data
        
        # Streaming should provide reasonable performance
        assert processing_time < 30.0, f"Streaming processing took {processing_time:.2f}s (limit: 30s)"
        
        print(f"\n📊 Streaming Architecture Metrics:")
        print(f"   Tool Discovery: {status_time*1000:.1f}ms")
        print(f"   Memory Management: {memory_time:.3f}s")
        print(f"   Streaming Processing: {processing_time:.2f}s")
        
        print("✅ Streaming architecture validation completed")
        
        return streaming_metrics


class TestCrossDomainIntegration(TestTimeSeriesIntegration):
    """Test integration between time series and other analysis domains."""
    
    def test_time_series_with_pattern_recognition(self):
        """Test time series analysis combined with pattern recognition."""
        from src.localdata_mcp.localdata_mcp import _db_manager
        
        # Connect to test database
        conn_result = _db_manager.connect_database(
            name="test_cross_domain",
            db_type="sqlite",
            conn_string=self.db_connections['multivariate']
        )
        assert json.loads(conn_result)["success"] is True
        
        cross_domain_results = {}
        
        # Step 1: Time Series Analysis on each variable
        print("📊 Cross-Domain Step 1: Time Series Analysis")
        var_basic_result = _db_manager.analyze_multivariate_time_series(
            name="test_cross_domain", 
            table_name="multi_data",
            analysis_type="var",
            max_lags=3
        )
        cross_domain_results['time_series'] = json.loads(var_basic_result)
        
        # Step 2: Pattern Recognition on the same data
        print("🔍 Cross-Domain Step 2: Clustering Analysis") 
        clustering_result = _db_manager.perform_clustering(
            name="test_cross_domain",
            table_name="multi_data",
            algorithm="kmeans",
            n_clusters=3,
            sample_size=1000
        )
        cross_domain_results['clustering'] = json.loads(clustering_result)
        
        # Step 3: Anomaly Detection (Pattern Recognition domain)
        print("⚠️  Cross-Domain Step 3: Anomaly Detection")
        anomaly_result = _db_manager.detect_anomalies(
            name="test_cross_domain",
            table_name="multi_data", 
            algorithm="isolation_forest",
            contamination=0.05,
            sample_size=1000
        )
        cross_domain_results['anomaly_detection'] = json.loads(anomaly_result)
        
        # Step 4: Time Series Anomaly Detection for comparison
        print("📈 Cross-Domain Step 4: Time Series Anomaly Detection")
        # Test on first numeric column
        first_var_query = f"SELECT date, {cross_domain_results['time_series']['variables'][0]} as value FROM multi_data ORDER BY date"
        
        ts_anomaly_result = _db_manager.detect_time_series_anomalies(
            name="test_cross_domain",
            query=first_var_query,
            date_column="date",
            value_column="value",
            method="statistical"
        )
        cross_domain_results['ts_anomaly_detection'] = json.loads(ts_anomaly_result)
        
        # Validate cross-domain results
        assert cross_domain_results['time_series']['model_type'] == 'VAR'
        assert 'labels' in cross_domain_results['clustering']
        assert 'anomaly_labels' in cross_domain_results['anomaly_detection']
        assert 'anomaly_indices' in cross_domain_results['ts_anomaly_detection']
        
        # Compare anomaly detection approaches
        general_anomalies = len(cross_domain_results['anomaly_detection']['anomaly_labels'])
        ts_anomalies = len(cross_domain_results['ts_anomaly_detection']['anomaly_indices'])
        
        print(f"📋 Cross-Domain Results Comparison:")
        print(f"   VAR Model Variables: {len(cross_domain_results['time_series']['variables'])}")
        print(f"   Clustering Labels: {len(set(cross_domain_results['clustering']['labels']))}")
        print(f"   General Anomalies: {general_anomalies}")
        print(f"   Time Series Anomalies: {ts_anomalies}")
        
        print("✅ Cross-domain integration validation completed")
        
        return cross_domain_results
    
    def test_time_series_with_statistical_analysis(self):
        """Test time series analysis combined with statistical analysis."""
        from src.localdata_mcp.localdata_mcp import _db_manager
        
        # Connect to test database
        conn_result = _db_manager.connect_database(
            name="test_stats_integration",
            db_type="sqlite", 
            conn_string=self.db_connections['daily']
        )
        assert json.loads(conn_result)["success"] is True
        
        stats_integration_results = {}
        
        # Step 1: Basic Time Series Analysis
        print("📊 Stats Integration Step 1: Time Series Analysis")
        ts_basic_result = _db_manager.analyze_time_series_basic(
            name="test_stats_integration",
            table_name="daily_data",
            date_column="date",
            value_column="value"
        )
        stats_integration_results['time_series'] = json.loads(ts_basic_result)
        
        # Step 2: Statistical Data Profiling
        print("📈 Stats Integration Step 2: Statistical Profiling")
        profile_result = _db_manager.profile_table(
            name="test_stats_integration",
            table_name="daily_data",
            sample_size=0,  # Use all data
            include_distributions=True
        )
        stats_integration_results['profiling'] = json.loads(profile_result)
        
        # Step 3: Distribution Analysis
        print("📉 Stats Integration Step 3: Distribution Analysis")
        dist_result = _db_manager.analyze_distributions(
            name="test_stats_integration", 
            table_name="daily_data",
            columns="value",
            sample_size=0,
            bins=50,
            percentiles="5,25,50,75,95"
        )
        stats_integration_results['distributions'] = json.loads(dist_result)
        
        # Validate integration results
        assert 'trend_analysis' in stats_integration_results['time_series']
        assert 'column_profiles' in stats_integration_results['profiling']
        assert 'column_distributions' in stats_integration_results['distributions']
        
        # Cross-validate statistical consistency
        ts_stats = stats_integration_results['time_series']['descriptive_stats']
        profile_stats = None
        
        # Find value column in profiling results
        for col_name, col_profile in stats_integration_results['profiling']['column_profiles'].items():
            if 'value' in col_name.lower():
                profile_stats = col_profile['statistics']
                break
        
        if profile_stats and ts_stats:
            # Compare mean values (should be similar)
            ts_mean = ts_stats['mean']
            profile_mean = profile_stats['mean']
            
            # Allow 1% difference due to potential sampling differences
            mean_diff = abs(ts_mean - profile_mean) / max(abs(ts_mean), abs(profile_mean), 1e-10)
            assert mean_diff < 0.01, f"Mean values inconsistent: TS={ts_mean}, Profile={profile_mean}"
        
        print("✅ Statistical analysis integration validation completed")
        
        return stats_integration_results


class TestErrorHandlingAndEdgeCases(TestTimeSeriesIntegration):
    """Test error handling and edge cases in time series analysis."""
    
    def test_insufficient_data_handling(self):
        """Test handling of insufficient data scenarios."""
        from src.localdata_mcp.localdata_mcp import _db_manager
        
        # Connect to short time series database
        conn_result = _db_manager.connect_database(
            name="test_short_data",
            db_type="sqlite",
            conn_string=self.db_connections['changepoint']  # Use any existing DB for setup
        )
        assert json.loads(conn_result)["success"] is True
        
        # Create a very short time series
        short_data = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=5, freq='D'),
            'value': [1, 2, 1.5, 2.5, 2]
        })
        
        # Create temporary database with short data
        short_db_path = self.test_dir / "short_data.db"
        with sqlite3.connect(short_db_path) as conn:
            short_data.to_sql('short_data', conn, index=False, if_exists='replace')
        
        conn_result = _db_manager.connect_database(
            name="test_very_short",
            db_type="sqlite",
            conn_string=str(short_db_path)
        )
        assert json.loads(conn_result)["success"] is True
        
        print("🔍 Testing insufficient data scenarios")
        
        # Test 1: Basic analysis with very short data
        basic_result = _db_manager.analyze_time_series_basic(
            name="test_very_short",
            table_name="short_data",
            date_column="date", 
            value_column="value"
        )
        basic_data = json.loads(basic_result)
        
        # Should handle short data gracefully
        assert 'trend_analysis' in basic_data
        assert 'seasonality_analysis' in basic_data
        
        # Test 2: ARIMA with insufficient data
        arima_result = _db_manager.forecast_arima(
            name="test_very_short",
            table_name="short_data",
            date_column="date",
            value_column="value",
            forecast_steps=3
        )
        
        # Should either provide result or graceful error
        arima_data = json.loads(arima_result)
        if 'error' not in arima_data:
            assert 'forecast_values' in arima_data
        else:
            assert 'insufficient' in arima_data['error'].lower() or 'data' in arima_data['error'].lower()
        
        # Test 3: Change point detection with insufficient data
        cp_result = _db_manager.detect_change_points(
            name="test_very_short",
            table_name="short_data",
            date_column="date",
            value_column="value",
            min_size=2
        )
        
        cp_data = json.loads(cp_result)
        # Should handle gracefully
        assert 'change_points' in cp_data or 'error' in cp_data
        
        print("✅ Insufficient data handling validation completed")
    
    def test_missing_data_handling(self):
        """Test handling of missing data in time series."""
        from src.localdata_mcp.localdata_mcp import _db_manager
        
        # Create time series with missing values
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        values = np.random.randn(100).cumsum()
        
        # Introduce missing values
        missing_indices = np.random.choice(100, size=10, replace=False)
        values[missing_indices] = np.nan
        
        missing_data = pd.DataFrame({
            'date': dates,
            'value': values
        })
        
        # Create database with missing data
        missing_db_path = self.test_dir / "missing_data.db"
        with sqlite3.connect(missing_db_path) as conn:
            missing_data.to_sql('missing_data', conn, index=False, if_exists='replace')
        
        conn_result = _db_manager.connect_database(
            name="test_missing",
            db_type="sqlite", 
            conn_string=str(missing_db_path)
        )
        assert json.loads(conn_result)["success"] is True
        
        print("🕳️  Testing missing data scenarios")
        
        # Test basic analysis with missing data
        basic_result = _db_manager.analyze_time_series_basic(
            name="test_missing",
            table_name="missing_data",
            date_column="date",
            value_column="value"
        )
        
        basic_data = json.loads(basic_result)
        assert 'trend_analysis' in basic_data or 'error' in basic_data
        
        # Test anomaly detection with missing data
        anomaly_result = _db_manager.detect_time_series_anomalies(
            name="test_missing",
            table_name="missing_data",
            date_column="date",
            value_column="value",
            method="statistical"
        )
        
        anomaly_data = json.loads(anomaly_result)
        assert 'anomaly_indices' in anomaly_data or 'error' in anomaly_data
        
        print("✅ Missing data handling validation completed")
    
    def test_invalid_parameters_handling(self):
        """Test handling of invalid parameters."""
        from src.localdata_mcp.localdata_mcp import _db_manager
        
        # Connect to valid test database
        conn_result = _db_manager.connect_database(
            name="test_invalid_params",
            db_type="sqlite",
            conn_string=self.db_connections['daily']
        )
        assert json.loads(conn_result)["success"] is True
        
        print("❌ Testing invalid parameter scenarios")
        
        # Test 1: Invalid ARIMA order
        arima_result = _db_manager.forecast_arima(
            name="test_invalid_params",
            table_name="daily_data",
            date_column="date",
            value_column="value", 
            order="invalid_order",
            forecast_steps=10
        )
        
        arima_data = json.loads(arima_result)
        assert 'error' in arima_data
        assert 'order' in arima_data['error'].lower()
        
        # Test 2: Invalid seasonal order
        seasonal_result = _db_manager.forecast_arima(
            name="test_invalid_params",
            table_name="daily_data",
            date_column="date",
            value_column="value",
            seasonal_order="1,2,3",  # Missing 4th component
            forecast_steps=10
        )
        
        seasonal_data = json.loads(seasonal_result)
        assert 'error' in seasonal_data
        assert 'seasonal' in seasonal_data['error'].lower()
        
        # Test 3: Non-existent table
        nonexistent_result = _db_manager.analyze_time_series_basic(
            name="test_invalid_params",
            table_name="nonexistent_table",
            date_column="date",
            value_column="value"
        )
        
        nonexistent_data = json.loads(nonexistent_result)
        assert 'error' in nonexistent_data
        
        # Test 4: Non-existent columns
        invalid_col_result = _db_manager.analyze_time_series_basic(
            name="test_invalid_params",
            table_name="daily_data",
            date_column="nonexistent_date",
            value_column="nonexistent_value"
        )
        
        invalid_col_data = json.loads(invalid_col_result)
        # Should either handle gracefully or return meaningful error
        assert 'error' in invalid_col_data or 'analysis_type' in invalid_col_data
        
        print("✅ Invalid parameter handling validation completed")


def run_integration_test_suite():
    """Run the complete time series integration test suite."""
    
    print("🚀 Starting Time Series Integration Test Suite")
    print("=" * 60)
    
    # Test categories to run
    test_classes = [
        TestIndividualToolIntegration,
        TestCrossComponentWorkflows, 
        TestPerformanceBenchmarking,
        TestCrossDomainIntegration,
        TestErrorHandlingAndEdgeCases
    ]
    
    total_start_time = time.time()
    results_summary = {
        'individual_tools': {},
        'workflows': {},
        'performance': {}, 
        'cross_domain': {},
        'error_handling': {}
    }
    
    try:
        # Run each test category
        for test_class in test_classes:
            print(f"\n📋 Running {test_class.__name__}")
            print("-" * 50)
            
            # Create temporary directory for this test class
            import tempfile
            with tempfile.TemporaryDirectory() as tmp_dir:
                test_instance = test_class()
                test_instance.setup_test_environment(Path(tmp_dir))
                
                # Run test methods
                test_methods = [method for method in dir(test_instance) 
                               if method.startswith('test_') and callable(getattr(test_instance, method))]
                
                for test_method in test_methods:
                    try:
                        print(f"  🧪 {test_method}")
                        getattr(test_instance, test_method)()
                        print(f"    ✅ PASSED")
                    except Exception as e:
                        print(f"    ❌ FAILED: {e}")
                        raise
                
                # Cleanup
                test_instance.cleanup_test_environment()
        
        total_execution_time = time.time() - total_start_time
        
        print(f"\n🎉 TIME SERIES INTEGRATION TEST SUITE COMPLETED")
        print("=" * 60)
        print(f"Total Execution Time: {total_execution_time:.2f} seconds")
        print("\n📊 Test Summary:")
        print("  ✅ Individual Tool Integration: PASSED")
        print("  ✅ Cross-Component Workflows: PASSED")
        print("  ✅ Performance Benchmarking: PASSED")
        print("  ✅ Cross-Domain Integration: PASSED")
        print("  ✅ Error Handling & Edge Cases: PASSED")
        
        print(f"\n🏆 ALL INTEGRATION TESTS PASSED!")
        print(f"Time Series Analysis Domain is ready for production use.")
        
        return True
        
    except Exception as e:
        print(f"\n💥 INTEGRATION TEST SUITE FAILED: {e}")
        return False


if __name__ == "__main__":
    success = run_integration_test_suite()
    exit(0 if success else 1)