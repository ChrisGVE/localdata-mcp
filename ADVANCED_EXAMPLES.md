# LocalData MCP - Advanced Usage Examples

## Table of Contents

1. [Multi-Source Data Pipeline](#multi-source-data-pipeline)
2. [Large Dataset Processing](#large-dataset-processing)
3. [Development to Production Migration](#development-to-production-migration)
4. [Real-Time Analytics Setup](#real-time-analytics-setup)
5. [Configuration-Driven Workflows](#configuration-driven-workflows)
6. [Error Handling and Recovery](#error-handling-and-recovery)
7. [Performance Optimization Patterns](#performance-optimization-patterns)
8. [Security-First Data Access](#security-first-data-access)
9. [Concurrent Multi-Database Operations](#concurrent-multi-database-operations)
10. [Data Warehouse Integration](#data-warehouse-integration)

## Multi-Source Data Pipeline

This example demonstrates integrating data from multiple sources for comprehensive analysis.

```python
# Example: E-commerce Analytics Pipeline
# Combines sales database, user behavior logs, and configuration settings

# 1. Connect diverse data sources
connect_database("sales_db", "postgresql", "postgresql://localhost/ecommerce_sales")
connect_database("behavior_logs", "json", "./user_behavior_logs.json")
connect_database("product_config", "yaml", "./product_categories.yaml")
connect_database("external_data", "csv", "./market_trends.csv")

# 2. Sales performance analysis
sales_data = execute_query("sales_db", """
    SELECT 
        product_id,
        product_name,
        SUM(quantity) as total_quantity,
        SUM(price * quantity) as total_revenue,
        COUNT(DISTINCT customer_id) as unique_customers,
        DATE_TRUNC('month', order_date) as month
    FROM orders o
    JOIN order_items oi ON o.order_id = oi.order_id
    WHERE order_date >= '2024-01-01'
    GROUP BY product_id, product_name, month
    ORDER BY total_revenue DESC
""")

# 3. User behavior analysis from JSON logs
behavior_analysis = execute_query("behavior_logs", """
    SELECT 
        event_type,
        product_id,
        COUNT(*) as event_count,
        COUNT(DISTINCT user_id) as unique_users,
        AVG(session_duration) as avg_session_duration
    FROM events
    WHERE event_date >= '2024-01-01'
    GROUP BY event_type, product_id
    ORDER BY event_count DESC
""")

# 4. Product categorization from YAML config
product_categories = read_text_file("./product_categories.yaml", "yaml")

# 5. Market trends from CSV
market_trends = execute_query("external_data", """
    SELECT 
        category,
        trend_score,
        competition_level,
        market_growth_rate
    FROM trends
    WHERE quarter = '2024-Q1'
""")

# 6. Cross-reference analysis
# AI can now correlate sales performance with user behavior, 
# product categories, and market trends for comprehensive insights
```

## Large Dataset Processing

Demonstrates intelligent handling of massive datasets with automatic buffering and chunked processing.

```python
# Example: Processing 10GB+ Customer Dataset
# LocalData MCP automatically handles large files and query results

# 1. Connect to large CSV file (>100MB triggers automatic SQLite conversion)
connect_database("big_customers", "csv", "./customer_data_10gb.csv")
# LocalData MCP automatically converts to temporary SQLite for efficiency

# 2. Initial exploration - small queries work normally
schema_info = describe_database("big_customers")
sample_data = get_table_sample("big_customers", "customers", 20)

# 3. Large query automatically triggers buffering system
large_result = execute_query_json("big_customers", """
    SELECT 
        customer_id,
        customer_segment,
        lifetime_value,
        last_purchase_date,
        geographic_region,
        risk_score
    FROM customers
    WHERE lifetime_value > 1000
        AND last_purchase_date >= '2023-01-01'
    ORDER BY lifetime_value DESC
""")

# Result structure for large queries:
# {
#     "metadata": {
#         "total_rows": 850000,
#         "columns": ["customer_id", "customer_segment", ...],
#         "query_execution_time": "12.3s"
#     },
#     "first_10_rows": [...],
#     "buffering_info": {
#         "query_id": "big_customers_1640995200_a1b2",
#         "buffer_size": 850000,
#         "expiry_time": "2024-01-15T15:45:00Z"
#     }
# }

query_id = large_result["buffering_info"]["query_id"]

# 4. Process results in manageable chunks
chunk_size = 10000
current_position = 11  # Start after first 10 rows

while current_position < large_result["metadata"]["total_rows"]:
    # Get next chunk
    chunk = get_query_chunk(query_id, current_position, str(chunk_size))
    
    # Process chunk (AI analysis, transformation, etc.)
    process_customer_chunk(chunk)
    
    current_position += chunk_size
    
    # Optional: Check buffer status
    buffer_info = get_buffered_query_info(query_id)
    if buffer_info["expired"]:
        # Buffer expired, need to re-run query
        break

# 5. Segment analysis with automatic chunking
segment_analysis = execute_query_json("big_customers", """
    SELECT 
        customer_segment,
        geographic_region,
        COUNT(*) as customer_count,
        AVG(lifetime_value) as avg_ltv,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY lifetime_value) as median_ltv,
        MAX(last_purchase_date) as most_recent_purchase
    FROM customers
    GROUP BY customer_segment, geographic_region
    HAVING COUNT(*) > 1000
    ORDER BY avg_ltv DESC
""")

# 6. Memory-efficient aggregation processing
def process_large_aggregation():
    """Process large dataset aggregations efficiently"""
    
    # This query processes the entire 10GB dataset but returns manageable results
    regional_metrics = execute_query("big_customers", """
        WITH customer_metrics AS (
            SELECT 
                geographic_region,
                customer_segment,
                SUM(lifetime_value) as total_revenue,
                COUNT(*) as customer_count,
                AVG(risk_score) as avg_risk
            FROM customers
            GROUP BY geographic_region, customer_segment
        )
        SELECT 
            geographic_region,
            SUM(total_revenue) as region_revenue,
            SUM(customer_count) as total_customers,
            AVG(avg_risk) as region_risk_score,
            COUNT(DISTINCT customer_segment) as segments_present
        FROM customer_metrics
        GROUP BY geographic_region
        ORDER BY region_revenue DESC
    """)
    
    return regional_metrics
```

## Development to Production Migration

Shows how to seamlessly transition between environments while maintaining identical workflows.

```python
# Example: ML Model Training Pipeline - Dev to Prod
# Same code works across all environments

import os

# Environment-aware connection setup
def setup_environment_connections():
    environment = os.getenv('ENVIRONMENT', 'development')
    
    if environment == 'development':
        # Development: Local SQLite databases
        connect_database("user_data", "sqlite", "./dev_users.db")
        connect_database("features", "csv", "./dev_features.csv")
        connect_database("config", "yaml", "./dev_config.yaml")
        
    elif environment == 'staging':
        # Staging: MySQL with test data
        connect_database("user_data", "mysql", os.getenv('STAGING_DB_URL'))
        connect_database("features", "csv", "./staging_features.csv")
        connect_database("config", "yaml", "./staging_config.yaml")
        
    elif environment == 'production':
        # Production: PostgreSQL with full dataset
        connect_database("user_data", "postgresql", os.getenv('PROD_DB_URL'))
        connect_database("features", "csv", "./prod_features.csv")  # Large file, auto-SQLite
        connect_database("config", "yaml", "./prod_config.yaml")

# Setup connections based on environment
setup_environment_connections()

# Identical queries work across all environments
def extract_training_features():
    """Extract ML training features - same logic across environments"""
    
    # User demographic features
    user_features = execute_query("user_data", """
        SELECT 
            user_id,
            age_group,
            geographic_region,
            account_tenure_months,
            subscription_tier,
            CASE 
                WHEN last_login_date >= CURRENT_DATE - INTERVAL '7 days' 
                THEN 'active' 
                ELSE 'inactive' 
            END as activity_status
        FROM users
        WHERE created_date <= CURRENT_DATE - INTERVAL '30 days'
    """)
    
    # Behavioral features from CSV
    behavioral_features = execute_query("features", """
        SELECT 
            user_id,
            avg_session_duration,
            total_page_views,
            conversion_rate,
            churn_probability_score,
            engagement_score
        FROM user_behavior_features
    """)
    
    # Model configuration
    config = read_text_file("./config.yaml", "yaml")
    
    return {
        'user_features': user_features,
        'behavioral_features': behavioral_features,
        'config': config
    }

# Feature extraction works identically across environments
training_data = extract_training_features()

# Environment-specific processing
def process_for_environment():
    environment = os.getenv('ENVIRONMENT', 'development')
    
    if environment == 'development':
        # Development: Small sample for quick testing
        sample_users = execute_query("user_data", "SELECT * FROM users LIMIT 1000")
        
    elif environment in ['staging', 'production']:
        # Staging/Production: Full dataset with intelligent buffering
        all_users_result = execute_query_json("user_data", "SELECT * FROM users")
        
        if 'buffering_info' in all_users_result:
            # Large result set - process in chunks
            query_id = all_users_result['buffering_info']['query_id']
            process_users_in_chunks(query_id)
        else:
            # Small enough to process directly
            process_users_directly(all_users_result)
```

## Real-Time Analytics Setup

Demonstrates setting up near real-time analytics with multiple data sources.

```python
# Example: Real-Time Dashboard Data Pipeline
# Combines live transaction database with cached metrics and configuration

# 1. Setup connections for real-time pipeline
connect_database("live_transactions", "postgresql", "postgresql://prod-db/transactions")
connect_database("cached_metrics", "sqlite", "./cache/hourly_metrics.db")
connect_database("dashboard_config", "json", "./config/dashboard_settings.json")

# 2. Real-time transaction monitoring
def get_real_time_metrics():
    """Get current hour's transaction metrics"""
    
    current_transactions = execute_query("live_transactions", """
        SELECT 
            DATE_TRUNC('minute', transaction_time) as minute,
            COUNT(*) as transaction_count,
            SUM(amount) as total_amount,
            AVG(amount) as avg_amount,
            COUNT(DISTINCT customer_id) as unique_customers,
            COUNT(CASE WHEN transaction_type = 'refund' THEN 1 END) as refund_count
        FROM transactions
        WHERE transaction_time >= DATE_TRUNC('hour', NOW())
        GROUP BY DATE_TRUNC('minute', transaction_time)
        ORDER BY minute DESC
    """)
    
    return current_transactions

# 3. Historical comparison from cache
def get_historical_comparison():
    """Get historical metrics for comparison"""
    
    historical_metrics = execute_query("cached_metrics", """
        SELECT 
            hour_of_day,
            day_of_week,
            AVG(transaction_count) as avg_hourly_transactions,
            AVG(total_amount) as avg_hourly_amount,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY transaction_count) as median_transactions
        FROM hourly_metrics
        WHERE date >= DATE('now', '-30 days')
        GROUP BY hour_of_day, day_of_week
    """)
    
    return historical_metrics

# 4. Dashboard configuration
dashboard_config = read_text_file("./config/dashboard_settings.json", "json")

# 5. Anomaly detection query
def detect_anomalies():
    """Detect unusual patterns in real-time data"""
    
    # Complex query that benefits from LocalData MCP's optimization
    anomaly_analysis = execute_query("live_transactions", """
        WITH current_hour AS (
            SELECT 
                COUNT(*) as current_count,
                SUM(amount) as current_amount
            FROM transactions
            WHERE transaction_time >= DATE_TRUNC('hour', NOW())
        ),
        historical_avg AS (
            SELECT 
                AVG(hourly_count) as avg_count,
                STDDEV(hourly_count) as stddev_count,
                AVG(hourly_amount) as avg_amount,
                STDDEV(hourly_amount) as stddev_amount
            FROM (
                SELECT 
                    DATE_TRUNC('hour', transaction_time) as hour,
                    COUNT(*) as hourly_count,
                    SUM(amount) as hourly_amount
                FROM transactions
                WHERE transaction_time >= NOW() - INTERVAL '30 days'
                    AND DATE_TRUNC('hour', transaction_time) < DATE_TRUNC('hour', NOW())
                GROUP BY DATE_TRUNC('hour', transaction_time)
            ) hourly_stats
        )
        SELECT 
            c.current_count,
            c.current_amount,
            h.avg_count,
            h.avg_amount,
            CASE 
                WHEN ABS(c.current_count - h.avg_count) > 2 * h.stddev_count 
                THEN 'ANOMALY_COUNT'
                ELSE 'NORMAL_COUNT'
            END as count_status,
            CASE 
                WHEN ABS(c.current_amount - h.avg_amount) > 2 * h.stddev_amount 
                THEN 'ANOMALY_AMOUNT'
                ELSE 'NORMAL_AMOUNT'
            END as amount_status
        FROM current_hour c, historical_avg h
    """)
    
    return anomaly_analysis

# 6. Automated cache update
def update_metrics_cache():
    """Update cached metrics for performance"""
    
    # Calculate hourly metrics and store in cache
    hourly_update = execute_query("live_transactions", """
        INSERT INTO cached_metrics.hourly_metrics 
        SELECT 
            DATE_TRUNC('hour', transaction_time) as hour,
            EXTRACT(hour FROM transaction_time) as hour_of_day,
            EXTRACT(dow FROM transaction_time) as day_of_week,
            COUNT(*) as transaction_count,
            SUM(amount) as total_amount,
            AVG(amount) as avg_amount,
            COUNT(DISTINCT customer_id) as unique_customers,
            CURRENT_DATE as date
        FROM transactions
        WHERE transaction_time >= DATE_TRUNC('hour', NOW()) - INTERVAL '1 hour'
            AND transaction_time < DATE_TRUNC('hour', NOW())
        GROUP BY DATE_TRUNC('hour', transaction_time),
                 EXTRACT(hour FROM transaction_time),
                 EXTRACT(dow FROM transaction_time)
    """)

# 7. Real-time dashboard data assembly
def assemble_dashboard_data():
    """Assemble complete real-time dashboard data"""
    
    return {
        'current_metrics': get_real_time_metrics(),
        'historical_comparison': get_historical_comparison(),
        'anomaly_status': detect_anomalies(),
        'config': dashboard_config,
        'timestamp': execute_query("live_transactions", "SELECT NOW() as current_time")
    }
```

## Configuration-Driven Workflows

Shows how to build flexible, configuration-driven data workflows.

```python
# Example: Multi-Client Data Processing Pipeline
# Configuration determines data sources, transformations, and outputs

# 1. Load workflow configuration
connect_database("workflow_config", "yaml", "./workflows/client_data_processing.yaml")
workflow_config = read_text_file("./workflows/client_data_processing.yaml", "yaml")

# 2. Dynamic connection setup based on configuration
def setup_dynamic_connections(config):
    """Setup database connections based on configuration"""
    
    for client_name, client_config in config['clients'].items():
        for db_name, db_config in client_config['databases'].items():
            connection_name = f"{client_name}_{db_name}"
            
            connect_database(
                connection_name,
                db_config['type'],
                db_config['connection_string']
            )

setup_dynamic_connections(workflow_config)

# 3. Configuration-driven query execution
def execute_configured_queries(client_name, config):
    """Execute queries based on configuration"""
    
    client_config = config['clients'][client_name]
    results = {}
    
    for query_name, query_config in client_config['queries'].items():
        db_name = f"{client_name}_{query_config['database']}"
        query_sql = query_config['sql']
        
        # Execute query with parameters from config
        if 'parameters' in query_config:
            # Parameter substitution for dynamic queries
            for param, value in query_config['parameters'].items():
                query_sql = query_sql.replace(f"${{{param}}}", str(value))
        
        result = execute_query(db_name, query_sql)
        results[query_name] = result
    
    return results

# 4. Multi-client processing
def process_all_clients(config):
    """Process data for all configured clients"""
    
    all_results = {}
    
    for client_name in config['clients'].keys():
        print(f"Processing client: {client_name}")
        
        client_results = execute_configured_queries(client_name, config)
        all_results[client_name] = client_results
        
        # Apply client-specific transformations
        if 'transformations' in config['clients'][client_name]:
            transformed_results = apply_transformations(
                client_results, 
                config['clients'][client_name]['transformations']
            )
            all_results[client_name] = transformed_results
    
    return all_results

# 5. Example workflow configuration (YAML):
"""
clients:
  client_a:
    databases:
      primary:
        type: postgresql
        connection_string: "postgresql://localhost/client_a_db"
      analytics:
        type: csv
        connection_string: "./data/client_a_analytics.csv"
    queries:
      revenue_analysis:
        database: primary
        sql: >
          SELECT 
            product_category,
            SUM(revenue) as total_revenue,
            COUNT(*) as transaction_count
          FROM sales
          WHERE date >= '${start_date}'
          GROUP BY product_category
        parameters:
          start_date: "2024-01-01"
      customer_segments:
        database: analytics
        sql: >
          SELECT 
            segment,
            COUNT(*) as customer_count,
            AVG(lifetime_value) as avg_ltv
          FROM customer_data
          GROUP BY segment
    transformations:
      - type: currency_conversion
        from: USD
        to: EUR
        columns: [total_revenue, avg_ltv]

  client_b:
    databases:
      warehouse:
        type: mysql
        connection_string: "mysql://localhost/client_b_warehouse"
    queries:
      inventory_report:
        database: warehouse
        sql: >
          SELECT 
            product_id,
            product_name,
            current_stock,
            reorder_point,
            CASE WHEN current_stock <= reorder_point THEN 'REORDER' ELSE 'OK' END as status
          FROM inventory
          WHERE category = '${category}'
        parameters:
          category: "electronics"
"""

# 6. Advanced configuration features
def process_with_advanced_config():
    """Demonstrate advanced configuration features"""
    
    # Load advanced configuration with includes and variables
    connect_database("advanced_config", "yaml", "./config/advanced_workflow.yaml")
    advanced_config = read_text_file("./config/advanced_workflow.yaml", "yaml")
    
    # Configuration-driven large dataset handling
    for dataset_config in advanced_config['large_datasets']:
        dataset_name = dataset_config['name']
        
        # Connect with size awareness
        connect_database(dataset_name, dataset_config['type'], dataset_config['path'])
        
        # Query with automatic buffering for large results
        if dataset_config['expected_size'] == 'large':
            result = execute_query_json(dataset_name, dataset_config['query'])
            
            if 'buffering_info' in result:
                # Process in configured chunk sizes
                chunk_size = dataset_config.get('chunk_size', 10000)
                process_large_dataset_chunks(result['buffering_info']['query_id'], chunk_size)
        else:
            result = execute_query(dataset_name, dataset_config['query'])
            process_small_dataset(result)
```

## Error Handling and Recovery

Demonstrates robust error handling and recovery patterns.

```python
# Example: Production-Grade Error Handling and Recovery

import logging
import time
from contextlib import contextmanager

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProcessingError(Exception):
    """Custom exception for data processing errors"""
    pass

@contextmanager
def database_connection_manager(name, db_type, connection_string):
    """Context manager for safe database connections"""
    try:
        connect_database(name, db_type, connection_string)
        logger.info(f"Connected to database: {name}")
        yield name
    except Exception as e:
        logger.error(f"Failed to connect to {name}: {e}")
        raise
    finally:
        try:
            disconnect_database(name)
            logger.info(f"Disconnected from database: {name}")
        except Exception as e:
            logger.warning(f"Error disconnecting from {name}: {e}")

def execute_query_with_retry(db_name, query, max_retries=3, retry_delay=1.0):
    """Execute query with exponential backoff retry logic"""
    
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Executing query on {db_name} (attempt {attempt + 1}/{max_retries})")
            result = execute_query(db_name, query)
            logger.info(f"Query executed successfully on attempt {attempt + 1}")
            return result
            
        except Exception as e:
            last_exception = e
            logger.warning(f"Query failed on attempt {attempt + 1}: {e}")
            
            if attempt < max_retries - 1:
                delay = retry_delay * (2 ** attempt)  # Exponential backoff
                logger.info(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                logger.error(f"Query failed after {max_retries} attempts")
                raise DataProcessingError(f"Query failed after {max_retries} attempts: {last_exception}")

def safe_large_query_processing(db_name, query):
    """Safely process large queries with buffer management"""
    
    try:
        # Execute large query
        result = execute_query_json(db_name, query)
        
        if 'buffering_info' not in result:
            # Small result, return directly
            return result
        
        # Large result, process with buffer management
        query_id = result['buffering_info']['query_id']
        all_data = result['first_10_rows']
        current_position = 11
        
        while True:
            try:
                # Check if buffer is still valid
                buffer_info = get_buffered_query_info(query_id)
                
                if buffer_info.get('expired', False):
                    logger.warning("Query buffer expired, re-executing query")
                    return safe_large_query_processing(db_name, query)
                
                # Get next chunk
                chunk = get_query_chunk(query_id, current_position, "1000")
                
                if not chunk or len(chunk) == 0:
                    break  # No more data
                
                all_data.extend(chunk)
                current_position += len(chunk)
                
                logger.info(f"Processed {len(all_data)} rows so far")
                
            except Exception as e:
                logger.error(f"Error processing chunk at position {current_position}: {e}")
                
                # Try to recover by clearing buffer and re-executing
                try:
                    clear_query_buffer(query_id)
                except:
                    pass  # Buffer might already be expired
                
                raise DataProcessingError(f"Failed to process large query: {e}")
        
        # Cleanup buffer
        clear_query_buffer(query_id)
        logger.info(f"Successfully processed {len(all_data)} total rows")
        
        return {
            'data': all_data,
            'total_rows': len(all_data),
            'processing_status': 'completed'
        }
        
    except Exception as e:
        logger.error(f"Error in large query processing: {e}")
        raise DataProcessingError(f"Large query processing failed: {e}")

def robust_multi_database_workflow():
    """Demonstrate robust multi-database workflow with comprehensive error handling"""
    
    results = {}
    errors = {}
    
    # Database configurations with fallbacks
    database_configs = [
        {
            'name': 'primary_db',
            'type': 'postgresql', 
            'connection': 'postgresql://primary-server/db',
            'fallback': {'type': 'sqlite', 'connection': './fallback_primary.db'}
        },
        {
            'name': 'analytics_db',
            'type': 'mysql',
            'connection': 'mysql://analytics-server/db',
            'fallback': {'type': 'csv', 'connection': './fallback_analytics.csv'}
        },
        {
            'name': 'config_data',
            'type': 'yaml',
            'connection': './config/production.yaml',
            'fallback': {'type': 'yaml', 'connection': './config/default.yaml'}
        }
    ]
    
    # Process each database with fallback logic
    for db_config in database_configs:
        db_name = db_config['name']
        
        try:
            # Try primary connection
            with database_connection_manager(
                db_name, 
                db_config['type'], 
                db_config['connection']
            ) as conn_name:
                
                # Execute critical queries with retry logic
                if db_name == 'primary_db':
                    query = """
                        SELECT 
                            customer_id,
                            order_total,
                            order_date,
                            status
                        FROM orders
                        WHERE order_date >= CURRENT_DATE - INTERVAL '7 days'
                        ORDER BY order_date DESC
                    """
                    results[db_name] = execute_query_with_retry(conn_name, query)
                
                elif db_name == 'analytics_db':
                    query = """
                        SELECT 
                            product_category,
                            SUM(views) as total_views,
                            SUM(conversions) as total_conversions,
                            AVG(conversion_rate) as avg_conversion_rate
                        FROM product_analytics
                        WHERE date >= DATE_SUB(CURDATE(), INTERVAL 30 DAY)
                        GROUP BY product_category
                    """
                    results[db_name] = safe_large_query_processing(conn_name, query)
                
                elif db_name == 'config_data':
                    results[db_name] = read_text_file(db_config['connection'], 'yaml')
                
                logger.info(f"Successfully processed {db_name}")
                
        except Exception as e:
            logger.error(f"Primary connection failed for {db_name}: {e}")
            errors[f"{db_name}_primary"] = str(e)
            
            # Try fallback connection
            if 'fallback' in db_config:
                try:
                    fallback_config = db_config['fallback']
                    fallback_name = f"{db_name}_fallback"
                    
                    logger.info(f"Attempting fallback connection for {db_name}")
                    
                    with database_connection_manager(
                        fallback_name,
                        fallback_config['type'],
                        fallback_config['connection']
                    ) as fallback_conn:
                        
                        # Simplified queries for fallback data sources
                        if db_name == 'primary_db':
                            fallback_query = "SELECT * FROM orders LIMIT 1000"
                            results[db_name] = execute_query_with_retry(fallback_conn, fallback_query)
                            
                        elif db_name == 'analytics_db':
                            fallback_query = "SELECT * FROM analytics_summary"
                            results[db_name] = execute_query_with_retry(fallback_conn, fallback_query)
                            
                        elif db_name == 'config_data':
                            results[db_name] = read_text_file(fallback_config['connection'], 'yaml')
                        
                        logger.info(f"Successfully used fallback for {db_name}")
                        
                except Exception as fallback_error:
                    logger.error(f"Fallback also failed for {db_name}: {fallback_error}")
                    errors[f"{db_name}_fallback"] = str(fallback_error)
                    
                    # Set minimal default data to prevent complete failure
                    results[db_name] = {
                        'status': 'fallback_failed',
                        'error': str(fallback_error),
                        'data': []
                    }
            else:
                # No fallback available
                results[db_name] = {
                    'status': 'failed',
                    'error': str(e),
                    'data': []
                }
    
    # Validate results and provide summary
    successful_connections = sum(1 for result in results.values() 
                               if not isinstance(result, dict) or result.get('status') != 'failed')
    
    total_connections = len(database_configs)
    
    logger.info(f"Workflow completed: {successful_connections}/{total_connections} connections successful")
    
    if errors:
        logger.warning(f"Errors encountered: {errors}")
    
    return {
        'results': results,
        'errors': errors,
        'summary': {
            'successful_connections': successful_connections,
            'total_connections': total_connections,
            'success_rate': successful_connections / total_connections
        }
    }

# Usage example with comprehensive error handling
if __name__ == "__main__":
    try:
        workflow_results = robust_multi_database_workflow()
        
        # Process results with error awareness
        for db_name, result in workflow_results['results'].items():
            if isinstance(result, dict) and result.get('status') in ['failed', 'fallback_failed']:
                logger.warning(f"Data from {db_name} is incomplete or unavailable")
                # Implement business logic for handling missing data
            else:
                logger.info(f"Processing data from {db_name}: {len(result)} records")
                # Process normal data
        
    except Exception as e:
        logger.critical(f"Workflow failed completely: {e}")
        # Implement emergency fallback procedures
```

## Performance Optimization Patterns

Advanced patterns for optimizing LocalData MCP performance.

```python
# Example: Performance-Optimized Data Processing Patterns

import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# 1. Connection Pool Management for Optimal Performance
class OptimizedConnectionManager:
    def __init__(self):
        self.active_connections = {}
        self.connection_usage = {}
        self.max_connections = 10  # LocalData MCP limit
    
    def get_optimal_connection_strategy(self, databases):
        """Determine optimal connection strategy based on usage patterns"""
        
        # Analyze database usage patterns
        usage_analysis = {}
        for db_name, db_config in databases.items():
            estimated_queries = db_config.get('estimated_queries', 1)
            data_size = db_config.get('expected_data_size', 'medium')
            
            usage_analysis[db_name] = {
                'priority': self._calculate_priority(estimated_queries, data_size),
                'connection_duration': self._estimate_duration(estimated_queries, data_size)
            }
        
        # Sort by priority for optimal connection order
        return sorted(usage_analysis.items(), key=lambda x: x[1]['priority'], reverse=True)
    
    def _calculate_priority(self, queries, size):
        """Calculate connection priority based on usage"""
        base_priority = queries
        if size == 'large':
            base_priority *= 2
        return base_priority
    
    def _estimate_duration(self, queries, size):
        """Estimate how long connection will be needed"""
        base_duration = queries * 2  # 2 seconds per query estimate
        if size == 'large':
            base_duration *= 3
        return base_duration

# 2. Intelligent Query Batching
def optimized_query_execution(database_queries):
    """Execute multiple queries with intelligent batching"""
    
    # Group queries by database for efficiency
    queries_by_db = {}
    for query_info in database_queries:
        db_name = query_info['database']
        if db_name not in queries_by_db:
            queries_by_db[db_name] = []
        queries_by_db[db_name].append(query_info)
    
    results = {}
    
    for db_name, queries in queries_by_db.items():
        start_time = time.time()
        
        # Sort queries by complexity (simple first, complex last)
        sorted_queries = sorted(queries, key=lambda q: q.get('complexity', 'medium'))
        
        db_results = []
        for query_info in sorted_queries:
            query = query_info['sql']
            expected_size = query_info.get('expected_size', 'small')
            
            if expected_size == 'large':
                # Use JSON execution for large results (automatic buffering)
                result = execute_query_json(db_name, query)
                
                if 'buffering_info' in result:
                    # Process large results immediately to free buffer space
                    processed_result = process_buffered_result_efficiently(result)
                    db_results.append(processed_result)
                else:
                    db_results.append(result)
            else:
                # Use regular execution for small results
                result = execute_query(db_name, query)
                db_results.append(result)
        
        execution_time = time.time() - start_time
        results[db_name] = {
            'data': db_results,
            'execution_time': execution_time,
            'query_count': len(queries)
        }
        
        print(f"Executed {len(queries)} queries on {db_name} in {execution_time:.2f} seconds")
    
    return results

def process_buffered_result_efficiently(buffered_result):
    """Efficiently process large buffered results"""
    
    if 'buffering_info' not in buffered_result:
        return buffered_result
    
    query_id = buffered_result['buffering_info']['query_id']
    total_rows = buffered_result['metadata']['total_rows']
    
    # Process in optimal chunk sizes based on data characteristics
    optimal_chunk_size = min(10000, max(1000, total_rows // 100))
    
    processed_data = buffered_result['first_10_rows']
    current_position = 11
    
    while current_position < total_rows:
        chunk = get_query_chunk(query_id, current_position, str(optimal_chunk_size))
        if not chunk:
            break
            
        # Process chunk immediately to save memory
        processed_chunk = process_data_chunk(chunk)
        processed_data.extend(processed_chunk)
        
        current_position += len(chunk)
        
        # Optional: Provide progress feedback
        if current_position % (optimal_chunk_size * 10) == 0:
            progress = (current_position / total_rows) * 100
            print(f"Processing progress: {progress:.1f}% ({current_position}/{total_rows} rows)")
    
    # Clean up buffer immediately
    clear_query_buffer(query_id)
    
    return {
        'processed_data': processed_data,
        'total_rows': len(processed_data),
        'processing_method': 'chunked_buffered'
    }

def process_data_chunk(chunk):
    """Process individual data chunks efficiently"""
    # Implement your data processing logic here
    # Example: data transformation, aggregation, filtering
    return chunk  # Placeholder - replace with actual processing

# 3. Concurrent Database Operations (within connection limits)
def concurrent_database_operations(operation_configs):
    """Execute database operations concurrently while respecting connection limits"""
    
    # LocalData MCP allows max 10 concurrent connections
    max_concurrent = min(len(operation_configs), 10)
    
    results = {}
    errors = {}
    
    with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        # Submit all operations
        future_to_operation = {}
        
        for operation_config in operation_configs:
            future = executor.submit(execute_single_operation, operation_config)
            future_to_operation[future] = operation_config['name']
        
        # Process completed operations
        for future in as_completed(future_to_operation):
            operation_name = future_to_operation[future]
            
            try:
                result = future.result()
                results[operation_name] = result
                print(f"Completed operation: {operation_name}")
                
            except Exception as e:
                errors[operation_name] = str(e)
                print(f"Failed operation: {operation_name} - {e}")
    
    return {'results': results, 'errors': errors}

def execute_single_operation(operation_config):
    """Execute a single database operation"""
    
    operation_name = operation_config['name']
    db_config = operation_config['database']
    queries = operation_config['queries']
    
    try:
        # Connect to database
        connect_database(operation_name, db_config['type'], db_config['connection'])
        
        # Execute all queries for this operation
        operation_results = []
        for query in queries:
            start_time = time.time()
            
            if query.get('large_result_expected', False):
                result = execute_query_json(operation_name, query['sql'])
                if 'buffering_info' in result:
                    # Process large result efficiently
                    processed = process_buffered_result_efficiently(result)
                    operation_results.append(processed)
                else:
                    operation_results.append(result)
            else:
                result = execute_query(operation_name, query['sql'])
                operation_results.append(result)
            
            query_time = time.time() - start_time
            print(f"Query in {operation_name} took {query_time:.2f} seconds")
        
        return {
            'operation': operation_name,
            'results': operation_results,
            'status': 'success'
        }
        
    except Exception as e:
        return {
            'operation': operation_name,
            'error': str(e),
            'status': 'failed'
        }
        
    finally:
        # Clean up connection
        try:
            disconnect_database(operation_name)
        except:
            pass

# 4. Memory-Efficient Large File Processing
def process_multiple_large_files(file_configs):
    """Process multiple large files with memory optimization"""
    
    results = {}
    
    for file_config in file_configs:
        file_name = file_config['name']
        file_path = file_config['path']
        file_type = file_config['type']
        
        print(f"Processing large file: {file_name}")
        
        # Connect to file (LocalData MCP will auto-detect size and use SQLite if >100MB)
        connect_database(file_name, file_type, file_path)
        
        # Get file information
        file_info = describe_database(file_name)
        
        # Estimate processing strategy based on file size
        table_name = list(file_info['tables'].keys())[0]  # Get first table name
        
        # Count total rows to plan processing
        row_count_result = execute_query(file_name, f"SELECT COUNT(*) as total FROM {table_name}")
        total_rows = row_count_result[0]['total']
        
        print(f"File {file_name} has {total_rows:,} rows")
        
        if total_rows > 100000:  # Large file processing
            # Process in chunks to avoid memory issues
            chunk_size = 50000
            processed_rows = 0
            file_results = []
            
            while processed_rows < total_rows:
                chunk_query = f"""
                    SELECT * FROM {table_name} 
                    LIMIT {chunk_size} OFFSET {processed_rows}
                """
                
                chunk_data = execute_query(file_name, chunk_query)
                processed_chunk = process_data_chunk(chunk_data)
                file_results.extend(processed_chunk)
                
                processed_rows += len(chunk_data)
                progress = (processed_rows / total_rows) * 100
                print(f"Progress on {file_name}: {progress:.1f}% ({processed_rows:,}/{total_rows:,} rows)")
            
            results[file_name] = {
                'total_rows_processed': len(file_results),
                'processing_method': 'chunked',
                'data': file_results
            }
            
        else:  # Small file - process all at once
            all_data = execute_query(file_name, f"SELECT * FROM {table_name}")
            processed_data = process_data_chunk(all_data)
            
            results[file_name] = {
                'total_rows_processed': len(processed_data),
                'processing_method': 'full_load',
                'data': processed_data
            }
        
        # Clean up connection to free resources for next file
        disconnect_database(file_name)
        print(f"Completed processing: {file_name}")
    
    return results

# 5. Usage Examples
if __name__ == "__main__":
    
    # Example 1: Optimized multi-database workflow
    database_configs = {
        'sales_db': {
            'type': 'postgresql',
            'connection': 'postgresql://localhost/sales',
            'estimated_queries': 5,
            'expected_data_size': 'large'
        },
        'inventory': {
            'type': 'mysql', 
            'connection': 'mysql://localhost/inventory',
            'estimated_queries': 3,
            'expected_data_size': 'medium'
        },
        'config': {
            'type': 'yaml',
            'connection': './config.yaml',
            'estimated_queries': 1,
            'expected_data_size': 'small'
        }
    }
    
    # Setup optimal connections
    connection_manager = OptimizedConnectionManager()
    optimal_strategy = connection_manager.get_optimal_connection_strategy(database_configs)
    print(f"Optimal connection strategy: {optimal_strategy}")
    
    # Example 2: Concurrent operations
    concurrent_operations = [
        {
            'name': 'sales_analysis',
            'database': {'type': 'postgresql', 'connection': 'postgresql://localhost/sales'},
            'queries': [
                {'sql': 'SELECT * FROM daily_sales WHERE date >= CURRENT_DATE - INTERVAL \'7 days\''},
                {'sql': 'SELECT product_id, SUM(quantity) FROM sales GROUP BY product_id', 'large_result_expected': True}
            ]
        },
        {
            'name': 'inventory_check',
            'database': {'type': 'mysql', 'connection': 'mysql://localhost/inventory'},
            'queries': [
                {'sql': 'SELECT * FROM low_stock_items'},
                {'sql': 'SELECT category, COUNT(*) FROM products GROUP BY category'}
            ]
        }
    ]
    
    concurrent_results = concurrent_database_operations(concurrent_operations)
    print(f"Concurrent operations completed: {len(concurrent_results['results'])} successful, {len(concurrent_results['errors'])} failed")
```

This advanced examples document showcases the sophisticated capabilities of LocalData MCP and provides practical patterns for real-world usage. Each example demonstrates production-ready approaches to common data integration challenges.

## v1.3.1 Memory-Safe Architecture Examples

### Intelligent Query Analysis and Buffering

LocalData MCP v1.3.1 introduces intelligent pre-query analysis that automatically determines the optimal execution strategy based on query complexity and expected data size.

```python
# Example: Leveraging v1.3.1 Intelligent Query Analysis
# The system automatically analyzes queries before execution

def demonstrate_intelligent_analysis():
    """Show v1.3.1's intelligent query analysis in action"""
    
    connect_database("analytics", "postgresql", "postgresql://analytics-db/warehouse")
    
    # Small query - executed directly with no buffering
    small_result = execute_query_json("analytics", """
        SELECT customer_segment, COUNT(*) as count
        FROM customers 
        GROUP BY customer_segment
    """)
    
    print("Small Query Response:")
    print(f"- Execution strategy: Direct")
    print(f"- Memory usage: {small_result['metadata']['memory_usage_mb']}MB")
    print(f"- Query complexity: {small_result['metadata']['query_complexity']}")
    print(f"- Estimated tokens: {small_result['metadata'].get('estimated_tokens', 'N/A')}")
    
    # Medium query - streaming with first chunk returned immediately
    medium_result = execute_query_json("analytics", """
        SELECT 
            o.customer_id,
            c.customer_name,
            c.segment,
            o.order_date,
            o.total_amount,
            p.product_category
        FROM orders o
        JOIN customers c ON o.customer_id = c.customer_id  
        JOIN products p ON o.product_id = p.product_id
        WHERE o.order_date >= '2024-01-01'
        ORDER BY o.order_date DESC
    """)
    
    if 'buffering_info' in medium_result:
        print("\nMedium Query Response:")
        print(f"- Execution strategy: Buffered")
        print(f"- Total rows: {medium_result['metadata']['total_rows']:,}")
        print(f"- Buffering activated: Query ID {medium_result['buffering_info']['query_id']}")
        print(f"- Estimated processing time: {medium_result['metadata']['query_execution_time']}")
        
        # Access additional chunks
        query_id = medium_result['buffering_info']['query_id']
        
        # Get buffer status
        buffer_info = get_buffered_query_info(query_id)
        print(f"- Buffer expiry: {buffer_info['buffer_details']['expiry_time']}")
        print(f"- Chunks available: {buffer_info['buffer_details']['total_chunks']}")
        
        # Access specific chunks efficiently
        chunk = get_query_chunk(query_id, 11, "1000")
        print(f"- Retrieved chunk with {len(chunk['chunk_data'])} rows")
        
        # Cleanup when done
        clear_query_buffer(query_id)
        print("- Buffer cleaned up successfully")

# Advanced buffering patterns for large datasets
def advanced_buffering_patterns():
    """Demonstrate advanced patterns with v1.3.1 buffering system"""
    
    connect_database("warehouse", "postgresql", "postgresql://warehouse-db/data")
    
    # Large analytical query with automatic buffering
    analysis_result = execute_query_json("warehouse", """
        WITH customer_metrics AS (
            SELECT 
                customer_id,
                DATE_TRUNC('month', order_date) as month,
                SUM(order_total) as monthly_revenue,
                COUNT(*) as order_count,
                AVG(order_total) as avg_order_value,
                MAX(order_date) as last_order
            FROM transactions 
            WHERE order_date >= '2023-01-01'
            GROUP BY customer_id, DATE_TRUNC('month', order_date)
        ),
        customer_ltv AS (
            SELECT 
                customer_id,
                SUM(monthly_revenue) as lifetime_value,
                COUNT(DISTINCT month) as active_months,
                MAX(last_order) as most_recent_order,
                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY monthly_revenue) as median_monthly_revenue
            FROM customer_metrics
            GROUP BY customer_id
        )
        SELECT 
            c.customer_id,
            c.customer_name,
            c.acquisition_date,
            c.customer_segment,
            ltv.lifetime_value,
            ltv.active_months,
            ltv.most_recent_order,
            ltv.median_monthly_revenue,
            CASE 
                WHEN ltv.most_recent_order >= CURRENT_DATE - INTERVAL '30 days' THEN 'Active'
                WHEN ltv.most_recent_order >= CURRENT_DATE - INTERVAL '90 days' THEN 'At Risk'
                ELSE 'Churned'
            END as customer_status
        FROM customers c
        JOIN customer_ltv ltv ON c.customer_id = ltv.customer_id
        WHERE ltv.lifetime_value > 500
        ORDER BY ltv.lifetime_value DESC
    """)
    
    if 'buffering_info' in analysis_result:
        query_id = analysis_result['buffering_info']['query_id']
        total_chunks = analysis_result['buffering_info']['chunks_available']
        
        print(f"Complex Analysis Query:")
        print(f"- Query complexity: {analysis_result['metadata']['query_complexity']}")
        print(f"- Total customers analyzed: {analysis_result['metadata']['total_rows']:,}")
        print(f"- Execution time: {analysis_result['metadata']['query_execution_time']}")
        print(f"- Memory efficiency: Streaming {total_chunks} chunks")
        
        # Process high-value customers first (already sorted by lifetime_value DESC)
        high_value_chunk = get_query_chunk(query_id, 1, "500")  # Top 500 customers
        
        print(f"- Top 500 customers by LTV retrieved")
        print(f"- Highest LTV: ${high_value_chunk['chunk_data'][0]['lifetime_value']:,.2f}")
        
        # Stream remaining data in optimal chunks
        chunk_size = 2000  # Larger chunks for batch processing
        processed_customers = 500  # Already processed top 500
        
        while processed_customers < analysis_result['metadata']['total_rows']:
            chunk = get_query_chunk(query_id, processed_customers + 1, str(chunk_size))
            
            if not chunk['chunk_data']:
                break
                
            # Process chunk (example: customer segmentation analysis)
            segment_analysis = analyze_customer_segment(chunk['chunk_data'])
            processed_customers += len(chunk['chunk_data'])
            
            # Progress tracking
            progress = (processed_customers / analysis_result['metadata']['total_rows']) * 100
            print(f"- Processed {processed_customers:,} customers ({progress:.1f}%)")
        
        clear_query_buffer(query_id)
    
def analyze_customer_segment(customers):
    """Example processing function for customer chunks"""
    segments = {}
    for customer in customers:
        segment = customer['customer_segment']
        if segment not in segments:
            segments[segment] = {'count': 0, 'total_ltv': 0}
        segments[segment]['count'] += 1
        segments[segment]['total_ltv'] += customer['lifetime_value']
    
    return segments
```

### YAML Configuration Management Examples

LocalData MCP v1.3.1 introduces powerful YAML configuration capabilities for complex multi-database environments.

```python
# Example: Advanced YAML Configuration Patterns
# Create sophisticated configuration files for different environments

def setup_multi_environment_config():
    """Demonstrate v1.3.1 YAML configuration capabilities"""
    
    # Example production configuration
    production_config = """
# production.yaml - Advanced multi-database configuration
databases:
  # Primary OLTP database
  transactions:
    type: postgresql
    host: ${PROD_DB_HOST}
    port: ${PROD_DB_PORT:-5432}
    user: ${PROD_DB_USER}
    password: ${PROD_DB_PASSWORD}
    database: ${PROD_DB_NAME}
    
    # Performance optimization
    timeout: 30
    max_memory_mb: 1024
    connection_pool_size: 15
    server_side_cursors: true
    
    # Security settings
    ssl_mode: require
    ssl_cert_path: /etc/ssl/client.crt
    ssl_key_path: /etc/ssl/client.key
    
  # Analytics replica with different settings
  analytics_replica:
    type: postgresql
    host: ${ANALYTICS_DB_HOST}
    port: ${ANALYTICS_DB_PORT:-5432}
    user: ${ANALYTICS_DB_USER}
    password: ${ANALYTICS_DB_PASSWORD}
    database: ${ANALYTICS_DB_NAME}
    
    # Optimized for long-running analytical queries
    timeout: 1800  # 30 minutes for complex analytics
    max_memory_mb: 4096  # 4GB memory limit
    connection_pool_size: 5  # Lower concurrency, higher per-query resources
    server_side_cursors: true
    
  # Document store for unstructured data
  documents:
    type: mongodb
    host: ${MONGO_HOST}
    port: ${MONGO_PORT:-27017}
    user: ${MONGO_USER}
    password: ${MONGO_PASSWORD}
    database: ${MONGO_DATABASE}
    auth_source: admin
    replica_set: rs0
    timeout: 60
    max_memory_mb: 512
    
  # High-speed cache layer
  cache:
    type: redis
    host: ${REDIS_HOST}
    port: ${REDIS_PORT:-6379}
    password: ${REDIS_PASSWORD}
    database: 0
    timeout: 5
    max_memory_mb: 256

# Logging configuration for production
logging:
  level: WARNING
  format: json
  console: false
  file: /var/log/localdata/production.log
  max_size_mb: 100
  backup_count: 10
  syslog: true
  
  # Structured logging with correlation IDs
  include_timestamp: true
  include_process_id: true
  include_correlation_id: true

# Performance tuning for production workloads
performance:
  global_memory_limit_mb: 8192  # 8GB total limit
  operation_memory_limit_mb: 2048  # 2GB per operation
  default_chunk_size: 5000  # Larger chunks for better throughput
  max_tokens_direct: 8000  # Higher token limit for complex responses
  buffer_timeout_seconds: 3600  # 1 hour buffer retention
  enable_query_cache: true
  cache_ttl_seconds: 900  # 15 minute cache TTL

# Security configuration
security:
  enable_sql_validation: true
  allowed_sql_statements:
    - SELECT
    - WITH
    - EXPLAIN
  require_ssl: true
  enable_authentication: true
  auth_token: ${API_AUTH_TOKEN}
  enable_rate_limiting: true
  rate_limit_requests: 1000  # Per minute
  
  path_security:
    allow_parent_access: false
    allowed_paths:
      - /data/exports
      - /data/imports
      - /tmp/localdata

# Monitoring and alerting
monitoring:
  enable_metrics: true
  metrics_port: 9090
  enable_health_checks: true
  health_check_port: 8080
  alert_on_connection_failure: true
  alert_on_memory_limit: true
  alert_webhook_url: ${ALERT_WEBHOOK_URL}

# Environment-specific overrides
environments:
  production:
    logging:
      level: ERROR  # Even more restrictive in production
    performance:
      max_connections: 25  # Higher limits for production load
"""
    
    # Write configuration to file
    with open('./production-config.yaml', 'w') as f:
        f.write(production_config)
    
    print("✅ Production configuration created")
    
    # Demonstrate configuration loading and usage
    import os
    os.environ['LOCALDATA_CONFIG_FILE'] = './production-config.yaml'
    
    # Set required environment variables
    os.environ.update({
        'PROD_DB_HOST': 'prod-db.company.com',
        'PROD_DB_USER': 'localdata_prod',
        'PROD_DB_PASSWORD': 'secure_password',
        'PROD_DB_NAME': 'production',
        'ANALYTICS_DB_HOST': 'analytics-db.company.com',
        'ANALYTICS_DB_USER': 'analytics_user',
        'ANALYTICS_DB_PASSWORD': 'analytics_password',
        'ANALYTICS_DB_NAME': 'warehouse',
        'API_AUTH_TOKEN': 'secure_api_token_here',
        'ALERT_WEBHOOK_URL': 'https://alerts.company.com/webhook'
    })
    
    print("✅ Environment variables configured")
    print("✅ Ready for production deployment with advanced configuration")

def demonstrate_hot_configuration_reload():
    """Show v1.3.1 hot configuration reload capabilities"""
    
    # Initial configuration
    initial_config = """
logging:
  level: INFO
  format: plain

performance:
  default_chunk_size: 1000
  max_tokens_direct: 4000
"""
    
    with open('./dynamic-config.yaml', 'w') as f:
        f.write(initial_config)
    
    print("Initial configuration written")
    
    # Updated configuration for hot reload
    updated_config = """
logging:
  level: DEBUG
  format: json
  file: ./debug.log

performance:
  default_chunk_size: 2000
  max_tokens_direct: 8000
  enable_query_cache: true
"""
    
    # Simulate configuration change
    import time
    time.sleep(2)  # Wait for initial startup
    
    with open('./dynamic-config.yaml', 'w') as f:
        f.write(updated_config)
    
    print("Configuration updated - changes will be automatically reloaded")
    print("Hot reload allows configuration changes without service restart")
```

### Enhanced Error Handling and Recovery

```python
# Example: v1.3.1 Enhanced Error Handling Patterns
# Leverage improved error messages and recovery mechanisms

def demonstrate_enhanced_error_handling():
    """Show v1.3.1's enhanced error handling and recovery"""
    
    def robust_connection_with_fallback():
        """Demonstrate connection with fallback strategy"""
        
        primary_configs = [
            {
                'name': 'primary_db',
                'type': 'postgresql',
                'connection': 'postgresql://primary-db.company.com:5432/prod',
                'priority': 1
            },
            {
                'name': 'replica_db', 
                'type': 'postgresql',
                'connection': 'postgresql://replica-db.company.com:5432/prod',
                'priority': 2
            },
            {
                'name': 'cache_fallback',
                'type': 'sqlite',
                'connection': './fallback_cache.db',
                'priority': 3
            }
        ]
        
        connected_db = None
        
        for config in primary_configs:
            try:
                connect_database(config['name'], config['type'], config['connection'])
                connected_db = config['name']
                print(f"✅ Connected to {config['name']} (priority {config['priority']})")
                break
                
            except Exception as e:
                print(f"❌ Failed to connect to {config['name']}: {e}")
                
                # v1.3.1 provides detailed error information
                if hasattr(e, 'error_details'):
                    print(f"   Error details: {e.error_details}")
                    print(f"   Suggestions: {e.suggestions}")
                    print(f"   Documentation: {e.documentation}")
                
                continue
        
        if not connected_db:
            raise Exception("All database connections failed")
            
        return connected_db
    
    def intelligent_query_retry():
        """Demonstrate intelligent query retry with v1.3.1 metadata"""
        
        db_name = robust_connection_with_fallback()
        
        complex_query = """
            SELECT 
                c.customer_id,
                c.customer_name,
                COUNT(o.order_id) as order_count,
                SUM(o.total_amount) as total_spent,
                AVG(o.total_amount) as avg_order_value,
                MAX(o.order_date) as last_order_date,
                EXTRACT(epoch FROM (MAX(o.order_date) - MIN(o.order_date))) / 86400 as customer_lifetime_days
            FROM customers c
            LEFT JOIN orders o ON c.customer_id = o.customer_id
            WHERE c.registration_date >= '2023-01-01'
            GROUP BY c.customer_id, c.customer_name
            HAVING COUNT(o.order_id) > 0
            ORDER BY total_spent DESC
        """
        
        retry_strategies = [
            {'description': 'Full query', 'query': complex_query},
            {'description': 'Limited query', 'query': complex_query + ' LIMIT 10000'},
            {'description': 'Simplified query', 'query': """
                SELECT customer_id, customer_name, COUNT(*) as order_count
                FROM customers c
                LEFT JOIN orders o USING(customer_id)
                WHERE c.registration_date >= '2023-01-01'
                GROUP BY customer_id, customer_name
                LIMIT 5000
            """},
            {'description': 'Basic fallback', 'query': """
                SELECT customer_id, customer_name FROM customers 
                WHERE registration_date >= '2023-01-01' 
                LIMIT 1000
            """}
        ]
        
        for i, strategy in enumerate(retry_strategies):
            try:
                print(f"Attempting {strategy['description']}...")
                
                result = execute_query_json(db_name, strategy['query'])
                
                # Analyze result metadata for optimization hints
                metadata = result.get('metadata', {})
                print(f"✅ Success with {strategy['description']}")
                print(f"   Execution time: {metadata.get('query_execution_time', 'N/A')}")
                print(f"   Memory usage: {metadata.get('memory_usage_mb', 'N/A')}MB")
                print(f"   Query complexity: {metadata.get('query_complexity', 'N/A')}")
                
                # Check for performance hints
                if 'performance_hints' in result:
                    print(f"   Performance hints: {result['performance_hints']}")
                
                # Handle buffering if needed
                if 'buffering_info' in result:
                    query_id = result['buffering_info']['query_id']
                    print(f"   Large result buffered: {query_id}")
                    print(f"   Total rows: {metadata.get('total_rows', 'N/A'):,}")
                    
                    # Access first meaningful chunk
                    chunk = get_query_chunk(query_id, 1, "100")
                    print(f"   Sample data accessed: {len(chunk['chunk_data'])} rows")
                    
                    # Cleanup
                    clear_query_buffer(query_id)
                
                return result
                
            except Exception as e:
                print(f"❌ Failed {strategy['description']}: {e}")
                
                # v1.3.1 enhanced error analysis
                if hasattr(e, 'error_code'):
                    if e.error_code == 'memory_limit_exceeded':
                        print("   → Memory limit hit, trying simpler query")
                        continue
                    elif e.error_code == 'query_timeout':
                        print("   → Timeout occurred, trying limited query")
                        continue
                    elif e.error_code == 'query_too_complex':
                        print("   → Query too complex, simplifying")
                        continue
                
                if i == len(retry_strategies) - 1:
                    print("All retry strategies failed")
                    raise
        
        return None
    
    # Execute the robust query with fallback
    try:
        result = intelligent_query_retry()
        print("✅ Successfully executed query with intelligent retry strategy")
    except Exception as e:
        print(f"❌ All strategies failed: {e}")
```

This enhanced examples document demonstrates the powerful new capabilities in LocalData MCP v1.3.1, including intelligent query analysis, sophisticated YAML configuration management, and enhanced error handling with detailed diagnostic information. These patterns enable robust, production-ready data processing workflows that can handle any scale of data while maintaining optimal performance and reliability.

<function_calls>
<invoke name="TodoWrite">
<parameter name="todos">[{"id": "blog_post", "status": "completed", "content": "Create comprehensive technical blog post about localdata-mcp launch"}, {"id": "social_twitter", "status": "completed", "content": "Create Twitter/X announcement thread for technical communities"}, {"id": "social_linkedin", "status": "completed", "content": "Create LinkedIn professional announcement"}, {"id": "social_reddit", "status": "completed", "content": "Create Reddit posts for r/Python, r/MachineLearning, r/LocalLLaMA communities"}, {"id": "social_discord", "status": "completed", "content": "Create Discord/Slack community announcements"}, {"id": "forum_claude", "status": "completed", "content": "Create Claude/Anthropic community forum announcement"}, {"id": "forum_mcp", "status": "completed", "content": "Create MCP developer community announcements"}, {"id": "forum_python", "status": "completed", "content": "Create Python packaging community announcements"}, {"id": "docs_faq", "status": "completed", "content": "Add FAQ section for common questions"}, {"id": "docs_troubleshooting", "status": "completed", "content": "Create troubleshooting guide"}, {"id": "docs_advanced", "status": "completed", "content": "Add advanced usage examples"}, {"id": "docs_polish", "status": "in_progress", "content": "Final documentation polish and review"}]