#!/usr/bin/env python3
"""
Complete Time Series Analysis Example

This comprehensive example demonstrates all time series analysis capabilities
of LocalData MCP v2.0 using realistic synthetic datasets.

Features Demonstrated:
1. Data setup and database connection
2. Basic time series analysis (trend, seasonality, stationarity)
3. ARIMA and exponential smoothing forecasting
4. Anomaly detection using multiple methods
5. Change point detection
6. Multivariate analysis (VAR, Granger causality)
7. Performance monitoring and optimization
8. Cross-domain integration examples
9. Error handling and best practices
10. Results interpretation and visualization

Usage:
    python time_series_complete_example.py [--output-dir OUTPUT_DIR]
"""

import argparse
import json
import sqlite3
import tempfile
import time
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd

# Import LocalData MCP time series tools
try:
    from src.localdata_mcp.localdata_mcp import (
        connect_database,
        analyze_time_series_basic,
        forecast_arima,
        forecast_exponential_smoothing,
        detect_time_series_anomalies,
        detect_change_points,
        analyze_multivariate_time_series,
        get_streaming_status,
        manage_memory_bounds,
        # Cross-domain tools
        perform_clustering,
        detect_anomalies,
        profile_table
    )
except ImportError as e:
    print(f"Error importing LocalData MCP: {e}")
    print("Please ensure LocalData MCP is properly installed and accessible")
    exit(1)


class TimeSeriesExampleRunner:
    """Comprehensive time series analysis example runner."""
    
    def __init__(self, output_dir: str = "time_series_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.databases = {}
        self.results = {}
        
        print("🚀 Time Series Analysis Complete Example")
        print("=" * 50)
    
    def create_example_datasets(self):
        """Create realistic synthetic datasets for demonstration."""
        print("\n📊 Creating example datasets...")
        
        # Dataset 1: E-commerce sales with seasonality and trend
        print("  Creating e-commerce sales dataset...")
        dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
        
        # Base trend (growing business)
        trend = np.arange(len(dates)) * 0.02
        
        # Seasonal patterns (weekly and yearly)
        yearly_season = 10 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25)
        weekly_season = 5 * np.sin(2 * np.pi * np.arange(len(dates)) / 7)
        
        # Random noise
        noise = np.random.normal(0, 2, len(dates))
        
        # Special events (Black Friday, Christmas spike)
        special_events = np.zeros(len(dates))
        for date_idx, date in enumerate(dates):
            if date.month == 11 and date.day >= 25:  # Black Friday week
                special_events[date_idx] += 20
            elif date.month == 12 and date.day >= 20:  # Christmas week  
                special_events[date_idx] += 15
        
        # Combine all components
        sales = 100 + trend + yearly_season + weekly_season + noise + special_events
        
        # Add some anomalies (system outages, viral events)
        anomaly_indices = np.random.choice(len(dates), size=20, replace=False)
        for idx in anomaly_indices:
            if np.random.random() > 0.5:
                sales[idx] *= 0.1  # System outage (low sales)
            else:
                sales[idx] *= 3.0  # Viral event (high sales)
        
        ecommerce_data = pd.DataFrame({
            'date': dates,
            'sales': sales,
            'orders': sales * np.random.uniform(0.8, 1.2, len(dates)),  # Correlated metric
            'website_visits': sales * np.random.uniform(10, 15, len(dates))  # Leading indicator
        })
        
        # Save to database
        ecommerce_db = self.output_dir / "ecommerce_sales.db"
        with sqlite3.connect(ecommerce_db) as conn:
            ecommerce_data.to_sql('daily_sales', conn, index=False, if_exists='replace')
        self.databases['ecommerce'] = str(ecommerce_db)
        
        # Dataset 2: Stock market data with change points
        print("  Creating stock market dataset...")
        dates = pd.date_range('2018-01-01', '2023-12-31', freq='D')
        
        # Create different market regimes (change points)
        regime_changes = [
            ('2020-03-15', 'crash', -30),      # COVID crash
            ('2020-06-01', 'recovery', 25),    # Recovery
            ('2022-01-01', 'correction', -15), # Tech correction
            ('2023-01-01', 'stabilization', 5) # Stabilization
        ]
        
        price = 100  # Starting price
        prices = []
        
        for date in dates:
            # Apply regime changes
            for change_date, regime, impact in regime_changes:
                if date == pd.to_datetime(change_date):
                    price *= (1 + impact/100)
            
            # Daily random walk with volatility
            daily_return = np.random.normal(0.0005, 0.02)  # 0.05% daily return, 2% volatility
            price *= (1 + daily_return)
            prices.append(price)
        
        stock_data = pd.DataFrame({
            'date': dates,
            'price': prices,
            'volume': np.random.lognormal(15, 0.5, len(dates)),  # Trading volume
            'volatility': np.abs(np.diff(np.log(prices), prepend=np.log(prices[0]))) * 100
        })
        
        # Save to database
        stock_db = self.output_dir / "stock_market.db"
        with sqlite3.connect(stock_db) as conn:
            stock_data.to_sql('stock_prices', conn, index=False, if_exists='replace')
        self.databases['stock'] = str(stock_db)
        
        # Dataset 3: IoT sensor data with anomalies
        print("  Creating IoT sensor dataset...")
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='H')  # Hourly data
        
        # Temperature sensor with daily cycles
        hours = np.arange(len(dates)) % 24
        daily_temp_cycle = 5 * np.sin(2 * np.pi * hours / 24)
        seasonal_temp = 10 * np.sin(2 * np.pi * np.arange(len(dates)) / (24*365))
        noise = np.random.normal(0, 1, len(dates))
        
        temperature = 20 + seasonal_temp + daily_temp_cycle + noise
        
        # Add sensor malfunctions and extreme weather events
        anomaly_indices = np.random.choice(len(dates), size=50, replace=False)
        for idx in anomaly_indices:
            if np.random.random() > 0.7:
                temperature[idx] = np.random.uniform(-10, 50)  # Sensor malfunction
            else:
                temperature[idx] += np.random.uniform(-15, 15)  # Extreme weather
        
        iot_data = pd.DataFrame({
            'timestamp': dates,
            'temperature': temperature,
            'humidity': 50 + np.random.normal(0, 10, len(dates)),
            'pressure': 1013.25 + np.random.normal(0, 5, len(dates))
        })
        
        # Save to database
        iot_db = self.output_dir / "iot_sensors.db"
        with sqlite3.connect(iot_db) as conn:
            iot_data.to_sql('sensor_readings', conn, index=False, if_exists='replace')
        self.databases['iot'] = str(iot_db)
        
        print(f"  ✅ Created 3 example datasets in {self.output_dir}")
    
    def setup_database_connections(self):
        """Set up database connections for all datasets."""
        print("\n🔌 Setting up database connections...")
        
        for name, db_path in self.databases.items():
            print(f"  Connecting to {name} database...")
            
            result = connect_database(
                name=f"{name}_db",
                db_type="sqlite",
                conn_string=db_path
            )
            
            result_data = json.loads(result)
            if result_data.get("success"):
                print(f"    ✅ Connected to {name}_db")
            else:
                print(f"    ❌ Failed to connect to {name}_db: {result_data.get('error')}")
    
    def demonstrate_basic_analysis(self):
        """Demonstrate basic time series analysis."""
        print("\n📊 BASIC TIME SERIES ANALYSIS")
        print("-" * 40)
        
        # Analyze e-commerce sales data
        print("Analyzing e-commerce daily sales...")
        
        start_time = time.time()
        basic_result = analyze_time_series_basic(
            name="ecommerce_db",
            table_name="daily_sales",
            date_column="date",
            value_column="sales",
            freq="D"
        )
        execution_time = time.time() - start_time
        
        # Parse and interpret results
        basic_data = json.loads(basic_result)
        self.results['basic_analysis'] = basic_data
        
        if 'error' not in basic_data:
            print(f"  ✅ Analysis completed in {execution_time:.2f}s")
            
            # Trend analysis
            trend = basic_data['trend_analysis']['linear_trend']
            if trend['significant']:
                direction = "increasing" if trend['slope'] > 0 else "decreasing"
                print(f"  📈 Significant {direction} trend: {trend['slope']:.4f} sales/day")
                print(f"      R-squared: {trend['r_squared']:.3f}")
            
            # Seasonality
            if basic_data['seasonality_analysis'].get('decomposition_available'):
                seasonal_strength = basic_data['seasonality_analysis']['seasonal_strength']
                print(f"  🔄 Seasonal strength: {seasonal_strength:.3f}")
                
                pattern = basic_data['seasonality_analysis']['dominant_pattern']
                print(f"      Dominant pattern: {pattern}")
            
            # Stationarity
            adf_test = basic_data['stationarity_tests']['augmented_dickey_fuller']
            stationarity = "stationary" if adf_test['is_stationary'] else "non-stationary"
            print(f"  📊 Series is {stationarity} (p-value: {adf_test['p_value']:.4f})")
            
            # Descriptive statistics
            stats = basic_data['descriptive_stats']
            print(f"  📋 Summary: mean={stats['mean']:.2f}, std={stats['std']:.2f}")
            
        else:
            print(f"  ❌ Analysis failed: {basic_data['error']}")
        
        # Also analyze stock price data
        print("\nAnalyzing stock price data...")
        
        stock_result = analyze_time_series_basic(
            name="stock_db",
            table_name="stock_prices",
            date_column="date",
            value_column="price",
            freq="D"
        )
        
        stock_data = json.loads(stock_result)
        if 'error' not in stock_data:
            trend = stock_data['trend_analysis']['linear_trend']
            direction = "bullish" if trend['slope'] > 0 else "bearish"
            print(f"  📈 Market trend: {direction} (slope: {trend['slope']:.6f})")
        
        print("  ✅ Basic analysis demonstration completed")
    
    def demonstrate_forecasting(self):
        """Demonstrate forecasting capabilities."""
        print("\n🔮 FORECASTING DEMONSTRATION")
        print("-" * 40)
        
        # ARIMA forecasting on sales data
        print("ARIMA forecasting for e-commerce sales...")
        
        start_time = time.time()
        arima_result = forecast_arima(
            name="ecommerce_db",
            table_name="daily_sales",
            date_column="date",
            value_column="sales",
            forecast_steps=30,
            auto_arima=True
        )
        arima_time = time.time() - start_time
        
        arima_data = json.loads(arima_result)
        if 'error' not in arima_data:
            print(f"  ✅ ARIMA forecast completed in {arima_time:.2f}s")
            
            # Model diagnostics
            metrics = arima_data['model_metrics']
            print(f"  📊 Model metrics:")
            print(f"      AIC: {metrics['aic']:.2f}")
            print(f"      RMSE: {metrics['rmse']:.2f}")
            print(f"      MAPE: {metrics['mape']:.2f}%")
            
            # Forecast summary
            forecasts = arima_data['forecast_values']
            print(f"  🔮 30-day forecast:")
            print(f"      Mean: {np.mean(forecasts):.2f}")
            print(f"      Range: {min(forecasts):.2f} - {max(forecasts):.2f}")
            
            self.results['arima_forecast'] = arima_data
        else:
            print(f"  ❌ ARIMA failed: {arima_data['error']}")
        
        # Exponential smoothing for comparison
        print("\nExponential smoothing forecasting...")
        
        smoothing_result = forecast_exponential_smoothing(
            name="ecommerce_db",
            table_name="daily_sales",
            date_column="date",
            value_column="sales",
            method="auto",
            forecast_steps=30
        )
        
        smoothing_data = json.loads(smoothing_result)
        if 'error' not in smoothing_data:
            print(f"  ✅ Exponential smoothing completed")
            
            # Compare forecasts
            if 'arima_forecast' in self.results:
                arima_mean = np.mean(self.results['arima_forecast']['forecast_values'])
                smoothing_mean = np.mean(smoothing_data['forecast_values'])
                
                print(f"  📈 Forecast comparison:")
                print(f"      ARIMA mean: {arima_mean:.2f}")
                print(f"      Exponential smoothing mean: {smoothing_mean:.2f}")
                print(f"      Difference: {abs(arima_mean - smoothing_mean):.2f}")
            
            self.results['smoothing_forecast'] = smoothing_data
        
        print("  ✅ Forecasting demonstration completed")
    
    def demonstrate_anomaly_detection(self):
        """Demonstrate anomaly detection capabilities."""
        print("\n🔍 ANOMALY DETECTION DEMONSTRATION")
        print("-" * 40)
        
        # Statistical anomaly detection on sales data
        print("Statistical anomaly detection on sales...")
        
        statistical_result = detect_time_series_anomalies(
            name="ecommerce_db",
            table_name="daily_sales",
            date_column="date",
            value_column="sales",
            method="statistical",
            contamination=0.05
        )
        
        stat_data = json.loads(statistical_result)
        if 'error' not in stat_data:
            anomaly_count = len(stat_data['anomaly_indices'])
            total_points = stat_data['series_info']['length']
            anomaly_rate = (anomaly_count / total_points) * 100
            
            print(f"  ✅ Statistical detection completed")
            print(f"  🚨 Found {anomaly_count} anomalies ({anomaly_rate:.2f}% of data)")
            
            # Show some anomalies
            anomalies = stat_data['anomaly_periods'][:5]  # First 5 anomalies
            print(f"  📋 Sample anomalies:")
            for anomaly in anomalies:
                print(f"      {anomaly['date']}: value={anomaly['value']:.2f}, score={anomaly['score']:.3f}")
            
            self.results['statistical_anomalies'] = stat_data
        
        # Machine learning anomaly detection on IoT sensor data
        print("\nML anomaly detection on IoT temperature data...")
        
        ml_result = detect_time_series_anomalies(
            name="iot_db",
            table_name="sensor_readings",
            date_column="timestamp",
            value_column="temperature",
            method="isolation_forest",
            contamination=0.02  # Expect 2% anomalies
        )
        
        ml_data = json.loads(ml_result)
        if 'error' not in ml_data:
            anomaly_count = len(ml_data['anomaly_indices'])
            print(f"  ✅ ML detection completed")
            print(f"  🔥 Found {anomaly_count} temperature anomalies")
            
            # Analyze anomaly temperatures
            anomaly_temps = [a['value'] for a in ml_data['anomaly_periods']]
            if anomaly_temps:
                print(f"  🌡️  Anomaly temperature range: {min(anomaly_temps):.1f}°C - {max(anomaly_temps):.1f}°C")
            
            self.results['ml_anomalies'] = ml_data
        
        print("  ✅ Anomaly detection demonstration completed")
    
    def demonstrate_change_point_detection(self):
        """Demonstrate change point detection."""
        print("\n📈 CHANGE POINT DETECTION DEMONSTRATION")
        print("-" * 40)
        
        # Change point detection on stock prices (we know there are regime changes)
        print("Change point detection on stock prices...")
        
        cp_result = detect_change_points(
            name="stock_db",
            table_name="stock_prices",
            date_column="date",
            value_column="price",
            method="cusum",
            min_size=30  # Minimum 30-day segments
        )
        
        cp_data = json.loads(cp_result)
        if 'error' not in cp_data:
            change_points = cp_data['change_points']
            change_dates = cp_data['change_point_dates']
            
            print(f"  ✅ Change point detection completed")
            print(f"  📊 Found {len(change_points)} change points")
            
            # Show detected change points
            print(f"  📅 Change point dates:")
            for date in change_dates[:5]:  # Show first 5
                print(f"      {date}")
            
            # Analyze segments
            segments = cp_data['segments_analysis']
            print(f"  📋 Market segments analysis:")
            for i, segment in enumerate(segments[:3]):  # Show first 3 segments
                trend_direction = "bullish" if segment['trend'] > 0 else "bearish"
                print(f"      Segment {i+1}: {segment['length']} days, mean=${segment['mean']:.2f}, {trend_direction}")
            
            self.results['change_points'] = cp_data
        else:
            print(f"  ❌ Change point detection failed: {cp_data['error']}")
        
        print("  ✅ Change point detection demonstration completed")
    
    def demonstrate_multivariate_analysis(self):
        """Demonstrate multivariate time series analysis."""
        print("\n🔗 MULTIVARIATE ANALYSIS DEMONSTRATION")
        print("-" * 40)
        
        # VAR model on e-commerce metrics (sales, orders, website visits)
        print("VAR model analysis on e-commerce metrics...")
        
        var_result = analyze_multivariate_time_series(
            name="ecommerce_db",
            table_name="daily_sales",
            analysis_type="var",
            max_lags=5
        )
        
        var_data = json.loads(var_result)
        if 'error' not in var_data:
            print(f"  ✅ VAR model fitted")
            print(f"  📊 Variables analyzed: {var_data['variables']}")
            print(f"      Optimal lags: {var_data['optimal_lags']}")
            print(f"      Model AIC: {var_data['aic']:.2f}")
            
            self.results['var_analysis'] = var_data
        
        # Granger causality testing
        print("\nGranger causality testing...")
        
        granger_result = analyze_multivariate_time_series(
            name="ecommerce_db",
            table_name="daily_sales",
            analysis_type="granger",
            max_lags=3
        )
        
        granger_data = json.loads(granger_result)
        if 'error' not in granger_data:
            print(f"  ✅ Granger causality tests completed")
            
            # Analyze causality relationships
            causality_results = granger_data['causality_results']
            significant_relationships = []
            
            for relationship, result in causality_results.items():
                if result['causality_detected']:
                    significant_relationships.append((relationship, result['min_p_value']))
            
            print(f"  🔗 Significant causality relationships:")
            for relationship, p_value in significant_relationships:
                print(f"      {relationship} (p-value: {p_value:.4f})")
            
            if not significant_relationships:
                print("      No significant causality relationships detected")
            
            self.results['granger_causality'] = granger_data
        
        print("  ✅ Multivariate analysis demonstration completed")
    
    def demonstrate_performance_monitoring(self):
        """Demonstrate performance monitoring capabilities."""
        print("\n⚡ PERFORMANCE MONITORING DEMONSTRATION")
        print("-" * 40)
        
        # Check streaming status
        print("Checking streaming status...")
        status_result = get_streaming_status()
        status_data = json.loads(status_result)
        
        memory_status = status_data['memory_status']
        print(f"  💾 Memory status:")
        print(f"      Total: {memory_status['total_gb']:.1f}GB")
        print(f"      Available: {memory_status['available_gb']:.1f}GB")
        print(f"      Used: {memory_status['used_gb']:.1f}GB ({memory_status['percent_used']:.1f}%)")
        
        active_buffers = len(status_data.get('streaming_buffers', {}))
        print(f"  📊 Active streaming buffers: {active_buffers}")
        
        # Memory management
        print("\nExecuting memory management...")
        memory_result = manage_memory_bounds()
        memory_data = json.loads(memory_result)
        
        actions_taken = memory_data.get('actions_taken', [])
        print(f"  🧹 Memory management actions: {len(actions_taken)}")
        for action in actions_taken:
            print(f"      {action}")
        
        print("  ✅ Performance monitoring demonstration completed")
    
    def demonstrate_cross_domain_integration(self):
        """Demonstrate integration with other analysis domains."""
        print("\n🔄 CROSS-DOMAIN INTEGRATION DEMONSTRATION")
        print("-" * 40)
        
        # Statistical profiling + time series analysis
        print("Statistical profiling of e-commerce data...")
        
        profile_result = profile_table(
            name="ecommerce_db",
            table_name="daily_sales",
            sample_size=1000,  # Use sample for speed
            include_distributions=True
        )
        
        profile_data = json.loads(profile_result)
        if 'error' not in profile_data:
            print(f"  ✅ Statistical profiling completed")
            
            # Compare with time series statistics
            if 'basic_analysis' in self.results:
                ts_mean = self.results['basic_analysis']['descriptive_stats']['mean']
                
                # Find sales column in profile
                sales_profile = None
                for col_name, col_data in profile_data['column_profiles'].items():
                    if 'sales' in col_name.lower():
                        sales_profile = col_data['statistics']
                        break
                
                if sales_profile:
                    profile_mean = sales_profile['mean']
                    print(f"  📊 Mean comparison:")
                    print(f"      Time series analysis: {ts_mean:.2f}")
                    print(f"      Statistical profile: {profile_mean:.2f}")
                    print(f"      Difference: {abs(ts_mean - profile_mean):.4f}")
        
        # Pattern recognition on multivariate data
        print("\nClustering analysis on multivariate data...")
        
        clustering_result = perform_clustering(
            name="ecommerce_db",
            table_name="daily_sales",
            algorithm="kmeans",
            n_clusters=4,
            sample_size=1000
        )
        
        clustering_data = json.loads(clustering_result)
        if 'error' not in clustering_data:
            print(f"  ✅ Clustering completed")
            
            # Analyze cluster characteristics
            n_clusters = len(set(clustering_data['labels']))
            cluster_sizes = {}
            for label in clustering_data['labels']:
                cluster_sizes[label] = cluster_sizes.get(label, 0) + 1
            
            print(f"  🎯 Found {n_clusters} clusters:")
            for cluster_id, size in cluster_sizes.items():
                print(f"      Cluster {cluster_id}: {size} points")
        
        # Cross-domain anomaly detection comparison
        print("\nComparing time series vs general anomaly detection...")
        
        if 'statistical_anomalies' in self.results:
            # General anomaly detection
            general_anomaly_result = detect_anomalies(
                name="ecommerce_db",
                table_name="daily_sales",
                algorithm="isolation_forest",
                contamination=0.05,
                sample_size=1000
            )
            
            general_anomaly_data = json.loads(general_anomaly_result)
            if 'error' not in general_anomaly_data:
                ts_anomalies = len(self.results['statistical_anomalies']['anomaly_indices'])
                general_anomalies = len([l for l in general_anomaly_data['anomaly_labels'] if l == -1])
                
                print(f"  🔍 Anomaly detection comparison:")
                print(f"      Time series method: {ts_anomalies} anomalies")
                print(f"      General pattern method: {general_anomalies} anomalies")
        
        print("  ✅ Cross-domain integration demonstration completed")
    
    def demonstrate_error_handling(self):
        """Demonstrate error handling and edge cases."""
        print("\n⚠️ ERROR HANDLING DEMONSTRATION")
        print("-" * 40)
        
        # Test with non-existent table
        print("Testing error handling with non-existent table...")
        
        error_result = analyze_time_series_basic(
            name="ecommerce_db",
            table_name="non_existent_table",
            date_column="date",
            value_column="value"
        )
        
        error_data = json.loads(error_result)
        if 'error' in error_data:
            print(f"  ✅ Graceful error handling: {error_data['error']}")
        else:
            print(f"  ❌ Expected error but got success")
        
        # Test with invalid ARIMA parameters
        print("Testing with invalid ARIMA parameters...")
        
        invalid_arima_result = forecast_arima(
            name="ecommerce_db",
            table_name="daily_sales",
            date_column="date",
            value_column="sales",
            order="invalid_order",
            forecast_steps=10
        )
        
        invalid_data = json.loads(invalid_arima_result)
        if 'error' in invalid_data:
            print(f"  ✅ Parameter validation working: {invalid_data['error']}")
        else:
            print(f"  ❌ Expected parameter error")
        
        # Test with very small dataset
        print("Testing with minimal dataset...")
        
        # Create tiny dataset
        tiny_data = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=5, freq='D'),
            'value': [1, 2, 3, 2, 1]
        })
        
        tiny_db = self.output_dir / "tiny_data.db"
        with sqlite3.connect(tiny_db) as conn:
            tiny_data.to_sql('tiny_series', conn, index=False, if_exists='replace')
        
        # Connect and test
        connect_database(name="tiny_db", db_type="sqlite", conn_string=str(tiny_db))
        
        tiny_result = analyze_time_series_basic(
            name="tiny_db",
            table_name="tiny_series",
            date_column="date",
            value_column="value"
        )
        
        tiny_data_result = json.loads(tiny_result)
        if 'error' not in tiny_data_result:
            print(f"  ✅ Handled small dataset gracefully")
        else:
            print(f"  ⚠️ Small dataset limitation: {tiny_data_result['error']}")
        
        print("  ✅ Error handling demonstration completed")
    
    def save_results(self):
        """Save all results to files."""
        print("\n💾 SAVING RESULTS")
        print("-" * 40)
        
        # Save individual results
        for result_name, result_data in self.results.items():
            output_file = self.output_dir / f"{result_name}.json"
            with open(output_file, 'w') as f:
                json.dump(result_data, f, indent=2, default=str)
            print(f"  Saved {result_name} to {output_file}")
        
        # Save comprehensive summary
        summary = {
            'example_run_timestamp': time.time(),
            'datasets_created': list(self.databases.keys()),
            'analyses_performed': list(self.results.keys()),
            'total_results': len(self.results),
            'output_directory': str(self.output_dir)
        }
        
        summary_file = self.output_dir / "example_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"  ✅ All results saved to {self.output_dir}")
    
    def generate_report(self):
        """Generate comprehensive report of the example run."""
        print("\n📋 EXAMPLE EXECUTION REPORT")
        print("=" * 50)
        
        # Summary statistics
        total_analyses = len(self.results)
        successful_analyses = sum(1 for r in self.results.values() if 'error' not in r)
        
        print(f"Datasets created: {len(self.databases)}")
        print(f"Analyses performed: {total_analyses}")
        print(f"Successful analyses: {successful_analyses}/{total_analyses}")
        print(f"Success rate: {(successful_analyses/total_analyses)*100:.1f}%")
        
        # Feature demonstration summary
        features_demonstrated = [
            "✅ Basic time series analysis (trend, seasonality, stationarity)",
            "✅ ARIMA and exponential smoothing forecasting",
            "✅ Statistical and ML-based anomaly detection", 
            "✅ Change point detection",
            "✅ Multivariate analysis (VAR, Granger causality)",
            "✅ Performance monitoring and memory management",
            "✅ Cross-domain integration (clustering, profiling)",
            "✅ Error handling and edge cases"
        ]
        
        print(f"\nFeatures demonstrated:")
        for feature in features_demonstrated:
            print(f"  {feature}")
        
        # Performance insights
        if 'basic_analysis' in self.results:
            series_length = self.results['basic_analysis']['series_info']['length']
            print(f"\nLargest dataset analyzed: {series_length:,} data points")
        
        print(f"\nResults saved to: {self.output_dir}")
        print(f"Review individual JSON files for detailed analysis results.")
        
        print(f"\n🎉 TIME SERIES ANALYSIS EXAMPLE COMPLETED SUCCESSFULLY!")
    
    def run_complete_example(self):
        """Run the complete time series analysis example."""
        try:
            # Setup
            self.create_example_datasets()
            self.setup_database_connections()
            
            # Core demonstrations
            self.demonstrate_basic_analysis()
            self.demonstrate_forecasting()
            self.demonstrate_anomaly_detection()
            self.demonstrate_change_point_detection()
            self.demonstrate_multivariate_analysis()
            
            # Advanced demonstrations
            self.demonstrate_performance_monitoring()
            self.demonstrate_cross_domain_integration()
            self.demonstrate_error_handling()
            
            # Wrap up
            self.save_results()
            self.generate_report()
            
            return True
            
        except Exception as e:
            print(f"\n💥 EXAMPLE EXECUTION FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Main function to run the complete example."""
    parser = argparse.ArgumentParser(
        description="Complete Time Series Analysis Example for LocalData MCP v2.0"
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='time_series_results',
        help='Directory to save results (default: time_series_results)'
    )
    
    args = parser.parse_args()
    
    # Run example
    runner = TimeSeriesExampleRunner(args.output_dir)
    success = runner.run_complete_example()
    
    # Exit with appropriate code
    exit(0 if success else 1)


if __name__ == "__main__":
    main()