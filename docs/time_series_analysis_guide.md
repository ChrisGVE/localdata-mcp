# Time Series Analysis Guide - LocalData MCP v2.0

## Overview

The Time Series Analysis Domain provides comprehensive time series analysis and forecasting capabilities optimized for LLM-driven workflows. This guide covers all available tools, usage patterns, performance considerations, and integration examples.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Available Tools](#available-tools)  
3. [Basic Usage Examples](#basic-usage-examples)
4. [Advanced Workflows](#advanced-workflows)
5. [Performance Optimization](#performance-optimization)
6. [Cross-Domain Integration](#cross-domain-integration)
7. [Best Practices](#best-practices)
8. [Troubleshooting](#troubleshooting)

## Quick Start

### Prerequisites

Ensure you have LocalData MCP v2.0 installed with time series dependencies:

```bash
pip install localdata-mcp[timeseries]
```

Required dependencies:
- `statsmodels` - Statistical time series models
- `scipy` - Statistical computations
- `scikit-learn` - Machine learning algorithms
- `pandas` - Data manipulation
- `numpy` - Numerical computing

### Basic Setup

```python
from localdata_mcp import connect_database, analyze_time_series_basic

# Connect to your time series data
result = connect_database(
    name="my_timeseries", 
    db_type="sqlite",
    conn_string="path/to/timeseries.db"
)

# Perform basic analysis
analysis = analyze_time_series_basic(
    name="my_timeseries",
    table_name="sales_data", 
    date_column="date",
    value_column="sales"
)
```

## Available Tools

### 1. Basic Time Series Analysis

**Tool:** `analyze_time_series_basic`

Performs comprehensive basic analysis including trend detection, seasonality analysis, stationarity testing, and autocorrelation analysis.

**Parameters:**
- `name` (str): Database connection name
- `table_name` (str): Table containing time series data
- `query` (str): Alternative to table_name - custom SQL query
- `date_column` (str, optional): Date column name (auto-detected if None)
- `value_column` (str, optional): Value column name (auto-detected if None)  
- `freq` (str, optional): Time series frequency ('D', 'H', 'M', etc.)
- `seasonal_periods` (int, optional): Periods in a season
- `sample_size` (int): Number of rows to sample (0 = all rows)

**Returns:** JSON with trend analysis, seasonality metrics, stationarity tests, and descriptive statistics.

### 2. ARIMA Forecasting

**Tool:** `forecast_arima`

Advanced ARIMA (AutoRegressive Integrated Moving Average) forecasting with automatic parameter selection.

**Parameters:**
- `name` (str): Database connection name
- `table_name` / `query` (str): Data source
- `date_column` (str, optional): Date column name
- `value_column` (str, optional): Value column name
- `order` (str, optional): ARIMA order as 'p,d,q' (e.g., '1,1,1')
- `seasonal_order` (str, optional): Seasonal order as 'P,D,Q,s' (e.g., '1,1,1,12')
- `forecast_steps` (int): Number of periods to forecast (default: 12)
- `auto_arima` (bool): Auto-select optimal parameters (default: True)
- `sample_size` (int): Sample size for modeling

**Returns:** JSON with forecasts, confidence intervals, model metrics, and diagnostics.

### 3. Exponential Smoothing

**Tool:** `forecast_exponential_smoothing`

Multiple exponential smoothing methods including Simple, Double (Holt's), and Triple (Holt-Winters).

**Parameters:**
- `name` (str): Database connection name
- `table_name` / `query` (str): Data source
- `date_column` (str, optional): Date column name
- `value_column` (str, optional): Value column name
- `method` (str): Smoothing method ('simple', 'double', 'triple', 'auto')
- `seasonal` (str, optional): Seasonal type ('add', 'mul', None)
- `seasonal_periods` (int, optional): Seasonal periods
- `forecast_steps` (int): Forecast horizon (default: 12)
- `sample_size` (int): Sample size for modeling

**Returns:** JSON with forecasts and model parameters.

### 4. Time Series Anomaly Detection

**Tool:** `detect_time_series_anomalies`

Advanced anomaly detection using statistical and machine learning methods.

**Parameters:**
- `name` (str): Database connection name
- `table_name` / `query` (str): Data source
- `date_column` (str, optional): Date column name
- `value_column` (str, optional): Value column name
- `method` (str): Detection method ('statistical', 'isolation_forest')
- `contamination` (float): Expected anomaly proportion (default: 0.05)
- `window_size` (int, optional): Window size for methods
- `sample_size` (int): Sample size

**Returns:** JSON with anomaly indices, scores, and detailed periods.

### 5. Change Point Detection

**Tool:** `detect_change_points`

Detects structural breaks and regime changes in time series.

**Parameters:**
- `name` (str): Database connection name  
- `table_name` / `query` (str): Data source
- `date_column` (str, optional): Date column name
- `value_column` (str, optional): Value column name
- `method` (str): Detection method ('cusum', 'pelt')
- `min_size` (int): Minimum segment size (default: 10)
- `sample_size` (int): Sample size

**Returns:** JSON with change points, dates, and segment analysis.

### 6. Multivariate Time Series Analysis

**Tool:** `analyze_multivariate_time_series`

Vector Autoregression (VAR) models and Granger causality testing.

**Parameters:**
- `name` (str): Database connection name
- `table_name` / `query` (str): Data source (must have multiple numeric columns)
- `analysis_type` (str): Analysis type ('var', 'granger')
- `max_lags` (int): Maximum lags to consider (default: 5)
- `sample_size` (int): Sample size

**Returns:** JSON with model parameters, forecasts, or causality results.

## Basic Usage Examples

### Example 1: Sales Data Analysis

```python
# Connect to sales database
connect_database(
    name="sales_db",
    db_type="postgresql", 
    conn_string="postgresql://user:pass@localhost/sales"
)

# Basic analysis of daily sales
basic_result = analyze_time_series_basic(
    name="sales_db",
    table_name="daily_sales",
    date_column="sale_date", 
    value_column="total_sales",
    freq="D"
)

print("Trend Analysis:", basic_result['trend_analysis'])
print("Seasonality:", basic_result['seasonality_analysis'])
```

### Example 2: Stock Price Forecasting

```python
# ARIMA forecasting for stock prices
arima_forecast = forecast_arima(
    name="market_db",
    table_name="stock_prices",
    date_column="trading_date",
    value_column="close_price", 
    forecast_steps=30,
    auto_arima=True  # Automatically select best parameters
)

# Extract forecast values and confidence intervals
forecasts = arima_forecast['forecast_values']
lower_bound = arima_forecast['confidence_intervals']['lower']
upper_bound = arima_forecast['confidence_intervals']['upper']

print(f"30-day forecast: {forecasts}")
print(f"Model AIC: {arima_forecast['model_metrics']['aic']}")
```

### Example 3: IoT Sensor Anomaly Detection

```python
# Detect anomalies in sensor readings
anomalies = detect_time_series_anomalies(
    name="iot_db",
    table_name="temperature_sensors",
    date_column="timestamp",
    value_column="temperature",
    method="isolation_forest",
    contamination=0.02  # Expect 2% anomalies
)

# Review detected anomalies
for anomaly in anomalies['anomaly_periods']:
    print(f"Anomaly on {anomaly['date']}: {anomaly['value']:.2f}")
```

### Example 4: Website Traffic Change Points

```python  
# Detect change points in website traffic
change_points = detect_change_points(
    name="analytics_db",
    table_name="daily_visitors",
    date_column="date",
    value_column="unique_visitors",
    method="cusum",
    min_size=7  # Minimum 7-day segments
)

# Analyze segments between change points
for segment in change_points['segments_analysis']:
    print(f"Segment: days {segment['start_idx']}-{segment['end_idx']}")
    print(f"  Mean: {segment['mean']:.0f} visitors")
    print(f"  Trend: {segment['trend']:.2f} visitors/day")
```

## Advanced Workflows

### Comprehensive Time Series Pipeline

This example demonstrates a complete analysis pipeline:

```python
def comprehensive_time_series_analysis(db_name, table_name, date_col, value_col):
    """Complete time series analysis pipeline."""
    
    results = {}
    
    # Step 1: Basic Analysis
    print("📊 Performing basic analysis...")
    results['basic'] = analyze_time_series_basic(
        name=db_name,
        table_name=table_name,
        date_column=date_col,
        value_column=value_col
    )
    
    # Step 2: Anomaly Detection
    print("🔍 Detecting anomalies...")
    results['anomalies'] = detect_time_series_anomalies(
        name=db_name,
        table_name=table_name,
        date_column=date_col,
        value_column=value_col,
        method="statistical"
    )
    
    # Step 3: Change Point Detection
    print("📈 Detecting change points...")
    results['change_points'] = detect_change_points(
        name=db_name,
        table_name=table_name,
        date_column=date_col,
        value_column=value_col
    )
    
    # Step 4: Forecasting
    print("🔮 Generating forecasts...")
    results['arima'] = forecast_arima(
        name=db_name,
        table_name=table_name,
        date_column=date_col,
        value_column=value_col,
        forecast_steps=30
    )
    
    results['smoothing'] = forecast_exponential_smoothing(
        name=db_name,
        table_name=table_name,
        date_column=date_col,
        value_column=value_col,
        method="auto",
        forecast_steps=30
    )
    
    return results

# Usage
analysis = comprehensive_time_series_analysis(
    "sales_db", "monthly_revenue", "month", "revenue"
)
```

### Multivariate Analysis Workflow

```python
def multivariate_analysis_workflow(db_name, table_name):
    """Complete multivariate time series analysis."""
    
    # VAR Model Analysis
    print("📊 Fitting VAR model...")
    var_result = analyze_multivariate_time_series(
        name=db_name,
        table_name=table_name,
        analysis_type="var",
        max_lags=5
    )
    
    print(f"Optimal lags: {var_result['optimal_lags']}")
    print(f"Variables: {var_result['variables']}")
    
    # Granger Causality Testing  
    print("🔗 Testing Granger causality...")
    granger_result = analyze_multivariate_time_series(
        name=db_name,
        table_name=table_name,
        analysis_type="granger",
        max_lags=3
    )
    
    # Analyze causality relationships
    for relationship, test_result in granger_result['causality_results'].items():
        if test_result['causality_detected']:
            print(f"✅ {relationship}: p-value = {test_result['min_p_value']:.4f}")
    
    return var_result, granger_result

# Usage for economic indicators
connect_database("econ_db", "postgresql", "postgresql://localhost/economics")

var_results, granger_results = multivariate_analysis_workflow(
    "econ_db", "economic_indicators"  # Table with GDP, inflation, unemployment, etc.
)
```

### High-Frequency Data Processing

```python
def process_high_frequency_data(db_name, table_name, sample_size=10000):
    """Process high-frequency time series efficiently."""
    
    # Use sampling for initial analysis
    print(f"🚀 Processing high-frequency data (sample: {sample_size:,})...")
    
    # Basic analysis with sampling
    basic_result = analyze_time_series_basic(
        name=db_name,
        table_name=table_name,
        date_column="timestamp",
        value_column="price",
        sample_size=sample_size
    )
    
    # Anomaly detection with streaming approach
    anomalies = detect_time_series_anomalies(
        name=db_name,
        table_name=table_name,
        date_column="timestamp", 
        value_column="price",
        method="isolation_forest",
        sample_size=sample_size
    )
    
    # Light-weight change point detection
    change_points = detect_change_points(
        name=db_name,
        table_name=table_name,
        date_column="timestamp",
        value_column="price",
        method="cusum",
        sample_size=sample_size
    )
    
    return {
        'basic_analysis': basic_result,
        'anomalies': anomalies,
        'change_points': change_points,
        'sample_size_used': sample_size
    }

# Process minute-level trading data
connect_database("trading_db", "postgresql", "postgresql://localhost/trading")
hf_results = process_high_frequency_data("trading_db", "minute_prices", 50000)
```

## Performance Optimization

### Data Size Recommendations

| Data Points | Recommended Tools | Performance Notes |
|------------|------------------|-------------------|
| < 1,000 | All tools | Fast execution |
| 1,000 - 10,000 | All tools | Good performance |
| 10,000 - 50,000 | Use sampling for complex forecasting | Monitor memory |
| > 50,000 | Use sample_size parameter | Streaming recommended |

### Streaming Configuration

```python
# Check streaming status
from localdata_mcp import get_streaming_status, manage_memory_bounds

# Monitor streaming performance
status = get_streaming_status()
print("Memory status:", status['memory_status'])
print("Active buffers:", len(status['streaming_buffers']))

# Manage memory usage
memory_result = manage_memory_bounds()
print("Memory management:", memory_result['actions_taken'])
```

### Optimization Tips

1. **Use Sampling**: For large datasets, use `sample_size` parameter:
   ```python
   # Sample 10,000 points for initial analysis
   result = analyze_time_series_basic(
       name="large_db",
       table_name="huge_table",
       sample_size=10000
   )
   ```

2. **Batch Processing**: Process data in chunks:
   ```python
   # Process data by year
   for year in range(2020, 2024):
       query = f"SELECT * FROM sales WHERE YEAR(date) = {year}"
       yearly_result = analyze_time_series_basic(
           name="sales_db",
           query=query
       )
   ```

3. **Optimize Forecasting**: Use appropriate forecast horizons:
   ```python
   # Short-term forecast (faster)
   short_forecast = forecast_arima(
       name="db", table_name="data",
       forecast_steps=12  # Instead of 365
   )
   ```

## Cross-Domain Integration

### Time Series + Pattern Recognition

```python
# Combine time series analysis with clustering
def time_series_clustering_analysis(db_name, table_name):
    # Time series analysis
    ts_result = analyze_time_series_basic(
        name=db_name,
        table_name=table_name,
        date_column="date",
        value_column="value"
    )
    
    # Cluster similar time series patterns
    clustering_result = perform_clustering(
        name=db_name,
        table_name=table_name,
        algorithm="kmeans",
        n_clusters=5
    )
    
    # Compare anomaly detection approaches
    ts_anomalies = detect_time_series_anomalies(
        name=db_name,
        table_name=table_name,
        method="statistical"
    )
    
    pattern_anomalies = detect_anomalies(
        name=db_name,
        table_name=table_name,
        algorithm="isolation_forest"
    )
    
    return {
        'time_series': ts_result,
        'clustering': clustering_result,
        'ts_anomalies': ts_anomalies,
        'pattern_anomalies': pattern_anomalies
    }
```

### Time Series + Statistical Analysis

```python
# Combine with statistical profiling
def comprehensive_data_analysis(db_name, table_name):
    # Statistical profiling
    profile = profile_table(
        name=db_name,
        table_name=table_name,
        include_distributions=True
    )
    
    # Time series analysis
    ts_analysis = analyze_time_series_basic(
        name=db_name,
        table_name=table_name
    )
    
    # Distribution analysis
    distributions = analyze_distributions(
        name=db_name,
        table_name=table_name,
        columns="value",
        bins=50
    )
    
    return {
        'profile': profile,
        'time_series': ts_analysis,
        'distributions': distributions
    }
```

## Best Practices

### 1. Data Preparation

```python
# Ensure proper data types and formatting
CREATE TABLE clean_timeseries AS
SELECT 
    CAST(date_column AS DATE) as date,
    CAST(value_column AS FLOAT) as value
FROM raw_data 
WHERE date_column IS NOT NULL 
  AND value_column IS NOT NULL
ORDER BY date_column;
```

### 2. Parameter Selection

**ARIMA Parameters:**
- Start with `auto_arima=True` for automatic selection
- For manual tuning, use ACF/PACF plots from basic analysis
- Common orders: (1,1,1), (2,1,2) for non-seasonal data

**Seasonal Parameters:**
- Daily data: `seasonal_periods=365` (yearly seasonality)
- Hourly data: `seasonal_periods=24` (daily seasonality)
- Monthly data: `seasonal_periods=12` (yearly seasonality)

**Anomaly Detection:**
- Start with `contamination=0.05` (5% anomalies expected)
- Use `method="statistical"` for interpretable results
- Use `method="isolation_forest"` for complex patterns

### 3. Result Interpretation

**Trend Analysis:**
```python
trend = result['trend_analysis']['linear_trend']
if trend['significant'] and trend['slope'] > 0:
    print(f"Significant upward trend: {trend['slope']:.4f} per period")
```

**Stationarity Tests:**
```python
adf_test = result['stationarity_tests']['augmented_dickey_fuller']
if adf_test['is_stationary']:
    print("Series is stationary - good for ARIMA modeling")
else:
    print("Series needs differencing - consider higher 'd' parameter")
```

**Forecast Quality:**
```python
metrics = forecast_result['model_metrics']
if metrics['rmse'] < metrics['mae']:
    print("Model handles large errors well")
else:
    print("Model may have outlier sensitivity")
```

### 4. Error Handling

```python
def safe_time_series_analysis(db_name, table_name, **kwargs):
    """Robust time series analysis with error handling."""
    
    try:
        result = analyze_time_series_basic(
            name=db_name,
            table_name=table_name,
            **kwargs
        )
        
        # Parse result
        if isinstance(result, str):
            import json
            result = json.loads(result)
        
        # Check for errors
        if 'error' in result:
            print(f"Analysis error: {result['error']}")
            return None
            
        return result
        
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

# Usage
result = safe_time_series_analysis("db", "table", date_column="date")
if result:
    print("Analysis successful")
else:
    print("Analysis failed - check data format")
```

## Troubleshooting

### Common Issues

#### 1. "No numeric columns found"
**Problem:** Time series tools require numeric value columns.

**Solution:**
```sql
-- Check column types
SELECT column_name, data_type 
FROM information_schema.columns 
WHERE table_name = 'your_table';

-- Convert text to numeric
ALTER TABLE your_table 
ALTER COLUMN value_column TYPE FLOAT 
USING value_column::FLOAT;
```

#### 2. "Insufficient data for analysis"
**Problem:** Time series too short for analysis.

**Solution:**
- Ensure minimum 10-20 points for basic analysis
- For forecasting, need at least 2 seasonal periods
- For change point detection, need at least 50-100 points

#### 3. "ARIMA model failed to converge"
**Problem:** Data not suitable for ARIMA or poor parameter selection.

**Solution:**
```python
# Try simpler exponential smoothing first
smoothing_result = forecast_exponential_smoothing(
    name=db_name,
    table_name=table_name,
    method="simple"  # Start with simple method
)

# Or check stationarity first
basic_result = analyze_time_series_basic(name=db_name, table_name=table_name)
if not basic_result['stationarity_tests']['augmented_dickey_fuller']['is_stationary']:
    print("Data needs differencing - try manual ARIMA order")
```

#### 4. "Memory usage too high"
**Problem:** Large dataset causing memory issues.

**Solution:**
```python
# Use sampling
result = analyze_time_series_basic(
    name=db_name,
    table_name=table_name, 
    sample_size=10000  # Limit to 10k points
)

# Monitor memory
status = get_streaming_status()
if status['memory_status']['is_low_memory']:
    manage_memory_bounds()  # Clean up buffers
```

### Performance Issues

#### Slow Forecasting
- Reduce `forecast_steps` 
- Use `sample_size` for large datasets
- Try exponential smoothing instead of ARIMA for speed

#### High Memory Usage
- Use `sample_size` parameter
- Call `manage_memory_bounds()` regularly
- Process data in smaller time windows

#### Tool Discovery Slow
- Check database connection status
- Verify MCP server is running properly
- Consider connection pooling for multiple analyses

### Data Quality Issues

#### Missing Values
```python
# Check for missing values first
profile = profile_table(name=db_name, table_name=table_name)
missing_rate = profile['column_profiles']['value_column']['missing_percentage']

if missing_rate > 10:
    print(f"Warning: {missing_rate}% missing values")
    # Consider data cleaning or interpolation
```

#### Irregular Time Intervals
```python
# Check for irregular intervals
basic_result = analyze_time_series_basic(name=db_name, table_name=table_name)
if basic_result['series_info']['frequency'] is None:
    print("Warning: Irregular time intervals detected")
    # May need to resample or interpolate data
```

## Conclusion

The Time Series Analysis Domain provides powerful, LLM-optimized tools for comprehensive time series analysis. By following this guide and best practices, you can effectively analyze trends, generate forecasts, detect anomalies, and identify change points in your time series data.

For additional help or advanced use cases, refer to the API documentation or contact the development team.

---

**Version:** LocalData MCP v2.0  
**Last Updated:** January 2025  
**Documentation Status:** Complete