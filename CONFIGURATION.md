# LocalData MCP Configuration Guide

## Overview

LocalData MCP v1.3.1 introduces a powerful dual configuration system that supports both simple environment variables for basic setups and comprehensive YAML configuration files for complex multi-database environments. This guide covers all configuration options, best practices, and advanced scenarios.

## Table of Contents

1. [Configuration Hierarchy](#configuration-hierarchy)
2. [Environment Variables](#environment-variables)
3. [YAML Configuration](#yaml-configuration)
4. [Configuration Discovery](#configuration-discovery)
5. [Database-Specific Configuration](#database-specific-configuration)
6. [Logging Configuration](#logging-configuration)
7. [Performance Tuning](#performance-tuning)
8. [Security Configuration](#security-configuration)
9. [Hot Configuration Reload](#hot-configuration-reload)
10. [Configuration Examples](#configuration-examples)
11. [Validation and Troubleshooting](#validation-and-troubleshooting)

## Configuration Hierarchy

LocalData MCP uses a hierarchical configuration system with the following precedence (highest to lowest):

```
1. Command Line Arguments (--config-file, --log-level, etc.)
   ↓
2. Environment Variables (LOCALDATA_*, DATABASE_*)
   ↓  
3. YAML Configuration Files (localdata-config.yaml)
   ↓
4. Built-in Defaults
```

This allows for flexible configuration management across different environments while maintaining clear override behavior.

## Environment Variables

### Core Configuration

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `LOCALDATA_CONFIG_FILE` | string | auto-discover | Path to YAML configuration file |
| `LOCALDATA_LOG_LEVEL` | string | `WARNING` | Logging level: DEBUG, INFO, WARNING, ERROR |
| `LOCALDATA_LOG_FORMAT` | string | `json` | Log format: plain, json |
| `LOCALDATA_LOG_FILE` | string | - | Log file path (optional) |
| `LOCALDATA_MAX_MEMORY_MB` | integer | `1024` | Global memory limit in MB |
| `LOCALDATA_DEFAULT_CHUNK_SIZE` | integer | `1000` | Default chunk size for streaming |
| `LOCALDATA_BUFFER_TIMEOUT` | integer | `600` | Buffer expiration time in seconds |

### Database Connection Variables

#### PostgreSQL
```bash
# Simple connection string (legacy)
POSTGRES_URL=postgresql://user:password@host:port/database

# Granular configuration (v1.3.1+)
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=username
POSTGRES_PASSWORD=password
POSTGRES_DATABASE=dbname
POSTGRES_TIMEOUT=30
POSTGRES_MAX_MEMORY_MB=512
POSTGRES_POOL_SIZE=5
```

#### MySQL
```bash
# Simple connection string (legacy)
MYSQL_URL=mysql://user:password@host:port/database

# Granular configuration (v1.3.1+)
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=username
MYSQL_PASSWORD=password
MYSQL_DATABASE=dbname
MYSQL_TIMEOUT=60
MYSQL_MAX_MEMORY_MB=256
MYSQL_CHARSET=utf8mb4
```

#### MongoDB
```bash
# Simple connection string (legacy)
MONGODB_URL=mongodb://user:password@host:port/database

# Granular configuration (v1.3.1+)
MONGODB_HOST=localhost
MONGODB_PORT=27017
MONGODB_USER=username
MONGODB_PASSWORD=password
MONGODB_DATABASE=dbname
MONGODB_AUTH_SOURCE=admin
MONGODB_TIMEOUT=30
```

#### Redis
```bash
# Simple connection string (legacy)
REDIS_URL=redis://password@host:port/database

# Granular configuration (v1.3.1+)
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=password
REDIS_DATABASE=0
REDIS_TIMEOUT=5
```

### Environment Variable Substitution

YAML configuration files support environment variable substitution:

```yaml
databases:
  production:
    host: ${DB_HOST}                    # Required variable
    port: ${DB_PORT:-5432}              # Optional with default
    password: ${DB_PASSWORD}            # Required variable
    connection_string: ${DB_URL}        # Full connection string override
```

## YAML Configuration

### Basic YAML Configuration

```yaml
# localdata-config.yaml
databases:
  main:
    type: postgresql
    host: localhost
    port: 5432
    user: localdata
    password: secret123
    database: production
    timeout: 30
    max_memory_mb: 512

logging:
  level: INFO
  format: json
  file: ./logs/localdata.log

performance:
  default_chunk_size: 1000
  max_tokens_direct: 4000
  buffer_timeout_seconds: 600
```

### Complete YAML Schema

```yaml
# Complete configuration with all available options
databases:
  <database_name>:
    # Connection settings
    type: string                      # postgresql, mysql, sqlite, mongodb, redis, etc.
    host: string                      # Database host
    port: integer                     # Database port
    user: string                      # Username
    password: string                  # Password (supports ${ENV_VAR})
    database: string                  # Database/schema name
    connection_string: string         # Override individual settings with full connection string
    
    # Performance settings
    timeout: integer                  # Query timeout in seconds (default: 60)
    max_memory_mb: integer            # Memory limit for this database (default: 256)
    connection_pool_size: integer     # Connection pool size (default: 3)
    max_overflow: integer             # Pool overflow connections (default: 2)
    pool_recycle_seconds: integer     # Connection recycle time (default: 3600)
    
    # Database-specific options
    charset: string                   # MySQL: charset (default: utf8mb4)
    auth_source: string               # MongoDB: authentication database
    replica_set: string               # MongoDB: replica set name
    ssl_mode: string                  # PostgreSQL: SSL mode (disable/require/verify-ca/verify-full)
    connect_timeout: integer          # Connection establishment timeout
    
    # Query optimization
    server_side_cursors: boolean      # PostgreSQL: enable server-side cursors
    autocommit: boolean               # Enable autocommit mode
    isolation_level: string           # Transaction isolation level

logging:
  level: string                       # DEBUG, INFO, WARNING, ERROR, CRITICAL
  format: string                      # plain, json, structured
  file: string                        # Log file path (optional)
  max_size_mb: integer               # Log file size limit (default: 10)
  backup_count: integer              # Number of backup files (default: 5)
  rotation: string                   # Rotation policy: size, time, both
  
  # Structured logging options
  include_timestamp: boolean         # Include timestamp in logs (default: true)
  include_level: boolean             # Include log level (default: true)
  include_logger: boolean            # Include logger name (default: true)
  
  # Additional log outputs
  console: boolean                   # Log to console (default: true)
  syslog: boolean                    # Log to syslog (default: false)
  
performance:
  # Memory management
  global_memory_limit_mb: integer    # Global memory limit (default: 1024)
  operation_memory_limit_mb: integer # Per-operation memory limit (default: 256)
  memory_check_interval: integer     # Memory monitoring interval in seconds (default: 10)
  
  # Query execution
  default_chunk_size: integer        # Default streaming chunk size (default: 1000)
  max_chunk_size: integer           # Maximum chunk size (default: 10000)
  min_chunk_size: integer           # Minimum chunk size (default: 100)
  
  # Response handling
  max_tokens_direct: integer         # Max tokens for direct response (default: 4000)
  max_tokens_streaming: integer      # Max tokens for streaming response (default: 8000)
  buffer_timeout_seconds: integer    # Buffer expiration time (default: 600)
  
  # Connection management
  max_connections: integer           # Global connection limit (default: 10)
  connection_retry_attempts: integer # Connection retry attempts (default: 3)
  connection_retry_delay: integer    # Delay between retries in seconds (default: 1)
  
  # Query optimization
  enable_query_cache: boolean        # Enable query result caching (default: false)
  cache_ttl_seconds: integer         # Cache TTL in seconds (default: 300)
  enable_query_optimization: boolean # Enable automatic query optimization (default: true)

security:
  # SQL security
  enable_sql_validation: boolean     # Enable SQL query validation (default: true)
  allowed_sql_statements:           # Allowed SQL statement types
    - SELECT
    - WITH
    - EXPLAIN
  
  # File access security
  allowed_file_extensions:          # Allowed file extensions
    - .csv
    - .json
    - .yaml
    - .xlsx
    - .parquet
  
  path_security:
    allow_parent_access: boolean     # Allow ../ in file paths (default: false)
    allowed_paths:                  # Explicitly allowed file paths
      - ./data
      - ./config
      - /tmp/uploads
    
  # Connection security
  require_ssl: boolean              # Require SSL for database connections (default: false)
  ssl_cert_path: string            # Path to SSL certificate
  ssl_key_path: string             # Path to SSL private key
  ssl_ca_path: string              # Path to SSL CA certificate
  
  # Authentication
  enable_authentication: boolean    # Enable authentication (default: false)
  auth_token: string               # API authentication token
  auth_header: string              # Authentication header name (default: Authorization)

monitoring:
  # Metrics collection
  enable_metrics: boolean           # Enable metrics collection (default: false)
  metrics_port: integer            # Metrics HTTP server port (default: 9090)
  metrics_path: string             # Metrics endpoint path (default: /metrics)
  
  # Health checks
  enable_health_checks: boolean     # Enable health check endpoint (default: true)
  health_check_port: integer       # Health check port (default: 8080)
  health_check_path: string        # Health check path (default: /health)
  health_check_interval: integer   # Health check interval in seconds (default: 30)
  
  # Alerting
  alert_on_connection_failure: boolean # Alert on database connection failures
  alert_on_memory_limit: boolean    # Alert when approaching memory limits
  alert_webhook_url: string         # Webhook URL for alerts

# Environment-specific overrides
environments:
  development:
    logging:
      level: DEBUG
      format: plain
    performance:
      max_connections: 5
      
  production:
    logging:
      level: WARNING
      format: json
    security:
      require_ssl: true
      enable_authentication: true
```

## Configuration Discovery

LocalData MCP automatically discovers configuration files in the following order:

1. **Explicit file**: `LOCALDATA_CONFIG_FILE` environment variable
2. **Current directory**: `./localdata-config.yaml`  
3. **User directory**: `~/.localdata/config.yaml`
4. **System directory**: `/etc/localdata/config.yaml`

### Configuration File Examples

#### Development Configuration
```yaml
# ~/.localdata/development.yaml
databases:
  dev_db:
    type: sqlite
    database: ./dev_database.db
    timeout: 10
    
logging:
  level: DEBUG
  format: plain
  console: true
  
performance:
  max_connections: 3
  default_chunk_size: 500
```

#### Production Configuration
```yaml
# /etc/localdata/production.yaml
databases:
  primary:
    type: postgresql
    host: ${DB_PRIMARY_HOST}
    port: ${DB_PRIMARY_PORT:-5432}
    user: ${DB_PRIMARY_USER}
    password: ${DB_PRIMARY_PASSWORD}
    database: ${DB_PRIMARY_NAME}
    timeout: 30
    max_memory_mb: 1024
    connection_pool_size: 10
    ssl_mode: require
    
  replica:
    type: postgresql
    host: ${DB_REPLICA_HOST}
    port: ${DB_REPLICA_PORT:-5432}
    user: ${DB_REPLICA_USER}
    password: ${DB_REPLICA_PASSWORD}
    database: ${DB_REPLICA_NAME}
    timeout: 120
    max_memory_mb: 2048
    connection_pool_size: 5
    ssl_mode: require
    
logging:
  level: WARNING
  format: json
  file: /var/log/localdata/production.log
  max_size_mb: 50
  backup_count: 10
  
security:
  enable_sql_validation: true
  require_ssl: true
  ssl_ca_path: /etc/ssl/certs/ca-certificates.crt
  
monitoring:
  enable_metrics: true
  enable_health_checks: true
  alert_on_connection_failure: true
  alert_webhook_url: https://alerts.company.com/webhook
```

## Database-Specific Configuration

### PostgreSQL Advanced Configuration

```yaml
databases:
  analytics_pg:
    type: postgresql
    host: analytics-db.company.com
    port: 5432
    user: ${ANALYTICS_USER}
    password: ${ANALYTICS_PASSWORD}
    database: analytics
    
    # Performance tuning
    timeout: 300                      # Long-running analytics queries
    max_memory_mb: 2048              # Large result sets
    connection_pool_size: 8          # High concurrency
    server_side_cursors: true        # Memory efficiency
    
    # SSL configuration
    ssl_mode: verify-full
    ssl_cert_path: /etc/ssl/client.crt
    ssl_key_path: /etc/ssl/client.key
    ssl_ca_path: /etc/ssl/ca.crt
    
    # Connection optimization
    connect_timeout: 10
    application_name: localdata-mcp-analytics
    
    # Query optimization
    isolation_level: read_committed
    autocommit: true
```

### MongoDB Configuration

```yaml
databases:
  document_store:
    type: mongodb
    host: mongo.company.com
    port: 27017
    user: ${MONGO_USER}
    password: ${MONGO_PASSWORD}
    database: documents
    auth_source: admin
    
    # Replica set configuration
    replica_set: rs0
    read_preference: secondaryPreferred
    
    # Connection settings
    timeout: 60
    connect_timeout: 10
    max_memory_mb: 512
    connection_pool_size: 5
    
    # MongoDB-specific options
    journal: true
    write_concern: majority
    read_concern: majority
```

### Redis Configuration

```yaml
databases:
  cache:
    type: redis
    host: cache.company.com
    port: 6379
    password: ${REDIS_PASSWORD}
    database: 0
    
    # Performance settings
    timeout: 5
    max_memory_mb: 128
    connection_pool_size: 3
    
    # Redis-specific options
    decode_responses: true
    retry_on_timeout: true
    socket_keepalive: true
    socket_keepalive_options:
      TCP_KEEPIDLE: 1
      TCP_KEEPINTVL: 3  
      TCP_KEEPCNT: 5
```

### Multi-Database Environment

```yaml
databases:
  # Primary transactional database
  transactions:
    type: postgresql
    host: ${TX_DB_HOST}
    port: 5432
    user: ${TX_DB_USER}
    password: ${TX_DB_PASSWORD}
    database: transactions
    timeout: 30
    max_memory_mb: 512
    connection_pool_size: 8
    
  # Analytics database (read-only)
  analytics:
    type: postgresql
    host: ${ANALYTICS_DB_HOST}
    port: 5432
    user: ${ANALYTICS_DB_USER}
    password: ${ANALYTICS_DB_PASSWORD}
    database: analytics
    timeout: 300                     # Long-running queries allowed
    max_memory_mb: 2048             # Large result sets
    connection_pool_size: 4
    server_side_cursors: true
    
  # Document store
  documents:
    type: mongodb
    host: ${MONGO_HOST}
    port: 27017
    user: ${MONGO_USER}
    password: ${MONGO_PASSWORD}
    database: documents
    auth_source: admin
    timeout: 60
    max_memory_mb: 256
    
  # Cache layer
  cache:
    type: redis
    host: ${REDIS_HOST}
    port: 6379
    password: ${REDIS_PASSWORD}
    database: 0
    timeout: 5
    max_memory_mb: 128
    
  # Time-series data
  metrics:
    type: influxdb
    host: ${INFLUX_HOST}
    port: 8086
    user: ${INFLUX_USER}
    password: ${INFLUX_PASSWORD}
    database: metrics
    timeout: 60
    max_memory_mb: 256
```

## Logging Configuration

### Basic Logging Setup

```yaml
logging:
  level: INFO
  format: json
  file: ./logs/localdata.log
  console: true
```

### Advanced Logging Configuration

```yaml
logging:
  # Log levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
  level: INFO
  
  # Log formats: plain, json, structured
  format: json
  
  # Output destinations
  console: true                       # Log to stdout
  file: /var/log/localdata/app.log   # Log file path
  syslog: false                      # Log to syslog
  
  # File rotation
  max_size_mb: 10                    # Rotate when file reaches size
  backup_count: 5                    # Keep N backup files
  rotation: size                     # Rotation trigger: size, time, both
  
  # Structured logging options
  include_timestamp: true            # Include ISO timestamp
  include_level: true               # Include log level
  include_logger: true              # Include logger name
  include_process_id: true          # Include process ID
  
  # Log filtering
  exclude_loggers:                  # Exclude specific loggers
    - urllib3.connectionpool
    - requests.packages.urllib3
    
  # Environment-specific overrides
  environments:
    development:
      level: DEBUG
      format: plain
      console: true
      file: null                    # No file logging in dev
    
    production:
      level: WARNING
      format: json
      console: false                # No console in production
      syslog: true                 # Use syslog in production
```

### Log Format Examples

#### Plain Format (Development)
```
2024-01-15 14:30:25,123 INFO     [localdata.database] Connected to database 'primary'
2024-01-15 14:30:25,234 DEBUG    [localdata.query] Executing query: SELECT COUNT(*) FROM users
2024-01-15 14:30:25,456 WARNING  [localdata.memory] Memory usage: 512MB (50% of limit)
```

#### JSON Format (Production)
```json
{
  "timestamp": "2024-01-15T14:30:25.123Z",
  "level": "INFO",
  "logger": "localdata.database",
  "message": "Connected to database 'primary'",
  "process_id": 1234,
  "thread_id": 5678,
  "extra": {
    "database_name": "primary",
    "connection_pool_size": 8
  }
}
```

## Performance Tuning

### Memory Management

```yaml
performance:
  # Global memory limits
  global_memory_limit_mb: 2048      # Total memory limit
  operation_memory_limit_mb: 512    # Per-operation limit
  
  # Memory monitoring
  memory_check_interval: 10         # Check every 10 seconds
  memory_warning_threshold: 0.8     # Warn at 80% usage
  memory_critical_threshold: 0.95   # Critical at 95% usage
  
  # Garbage collection
  auto_gc_enabled: true            # Enable automatic GC
  gc_threshold_mb: 256             # Trigger GC when this much is allocated
```

### Query Performance

```yaml
performance:
  # Chunk size optimization
  default_chunk_size: 1000         # Default streaming chunk size
  min_chunk_size: 100              # Minimum allowed chunk size
  max_chunk_size: 10000            # Maximum allowed chunk size
  adaptive_chunk_sizing: true      # Automatically adjust chunk sizes
  
  # Response optimization
  max_tokens_direct: 4000          # Direct response token limit
  max_tokens_streaming: 8000       # Streaming response token limit
  token_estimation_accuracy: high  # Token estimation mode: fast, balanced, high
  
  # Query caching
  enable_query_cache: true         # Enable result caching
  cache_ttl_seconds: 300          # Cache time-to-live
  cache_size_mb: 256              # Maximum cache size
  cache_key_include_params: true   # Include parameters in cache key
```

### Connection Pool Tuning

```yaml
databases:
  high_traffic_db:
    # Connection pool sizing
    connection_pool_size: 10        # Base pool size
    max_overflow: 5                # Additional connections allowed
    pool_recycle_seconds: 3600     # Recycle connections after 1 hour
    
    # Connection health
    pool_pre_ping: true            # Test connections before use
    pool_reset_on_return: commit   # Reset behavior: commit, rollback, none
    
    # Connection timeouts
    connect_timeout: 10            # Connection establishment timeout
    pool_timeout: 30              # Timeout waiting for connection from pool
```

## Security Configuration

### SQL Security

```yaml
security:
  enable_sql_validation: true
  
  # Allowed SQL statement types
  allowed_sql_statements:
    - SELECT
    - WITH
    - EXPLAIN
    - ANALYZE
    
  # Blocked SQL patterns (regex)
  blocked_sql_patterns:
    - "DROP\\s+TABLE"
    - "DELETE\\s+FROM"
    - "UPDATE\\s+SET"
    - "INSERT\\s+INTO"
    - "CREATE\\s+TABLE"
    - "ALTER\\s+TABLE"
    
  # Query complexity limits
  max_query_length: 10000          # Maximum query length in characters
  max_subqueries: 5               # Maximum nested subqueries
  max_joins: 10                   # Maximum JOIN clauses
```

### File Access Security

```yaml
security:
  path_security:
    allow_parent_access: false      # Disable ../ access
    
    # Explicitly allowed paths
    allowed_paths:
      - ./data
      - ./config
      - ./uploads
      - /tmp/localdata
      
    # Allowed file extensions
    allowed_file_extensions:
      - .csv
      - .json
      - .yaml
      - .yml
      - .xlsx
      - .xls
      - .parquet
      - .feather
      - .tsv
      
  # File size limits
  max_file_size_mb: 1024           # Maximum file size to process
  temp_file_cleanup: true          # Automatically clean temp files
  temp_file_ttl_minutes: 60       # Temp file TTL
```

### Connection Security

```yaml
security:
  # SSL/TLS configuration
  require_ssl: true                # Require SSL for all database connections
  ssl_verify_certificates: true   # Verify SSL certificates
  ssl_ca_bundle: /etc/ssl/certs/ca-certificates.crt
  
  # Authentication
  enable_authentication: false    # Enable API authentication
  auth_token: ${API_AUTH_TOKEN}   # API authentication token
  auth_header: Authorization      # Authentication header name
  
  # Rate limiting
  enable_rate_limiting: true      # Enable request rate limiting
  rate_limit_requests: 100        # Requests per minute
  rate_limit_window: 60           # Rate limit window in seconds
```

## Hot Configuration Reload

LocalData MCP v1.3.1 supports hot configuration reload for most settings (database connections require restart).

### Enabling Hot Reload

```yaml
# Enable file watching
configuration:
  enable_hot_reload: true          # Enable hot configuration reload
  watch_config_file: true         # Watch config file for changes
  reload_delay_seconds: 5         # Delay before applying changes
  
  # Reloadable settings
  hot_reload_sections:
    - logging                     # Log level, format changes
    - performance                 # Memory limits, chunk sizes
    - security                    # SQL validation rules
    - monitoring                  # Metrics, health check settings
```

### Configuration Change Detection

```python
# Example: Programmatically reload configuration
from localdata_mcp import ConfigManager

config_manager = ConfigManager()

# Register callback for configuration changes
def on_config_change(changes):
    print(f"Configuration changed: {changes}")
    
    if 'logging' in changes:
        print("Log level updated")
    
    if 'performance' in changes:
        print("Performance settings updated")

config_manager.register_reload_callback(on_config_change)
```

### Manual Reload

```bash
# Send SIGHUP to reload configuration
kill -HUP $(pgrep localdata-mcp)

# Or use the management API
curl -X POST http://localhost:8080/admin/reload-config
```

## Configuration Examples

### Microservices Architecture

```yaml
# Configuration for microservices deployment
databases:
  user_service:
    type: postgresql
    host: user-db.internal
    port: 5432
    user: ${USER_DB_USER}
    password: ${USER_DB_PASSWORD}
    database: users
    timeout: 30
    max_memory_mb: 256
    
  order_service:
    type: postgresql  
    host: order-db.internal
    port: 5432
    user: ${ORDER_DB_USER}
    password: ${ORDER_DB_PASSWORD}
    database: orders
    timeout: 60
    max_memory_mb: 512
    
  session_store:
    type: redis
    host: redis.internal
    port: 6379
    password: ${REDIS_PASSWORD}
    database: 0
    timeout: 5
    
logging:
  level: INFO
  format: json
  console: false
  syslog: true
  
monitoring:
  enable_metrics: true
  metrics_port: 9090
  enable_health_checks: true
  health_check_port: 8080
```

### Data Science Environment

```yaml
# Configuration for data science/analytics workloads
databases:
  warehouse:
    type: postgresql
    host: warehouse.company.com
    port: 5432
    user: ${WAREHOUSE_USER}
    password: ${WAREHOUSE_PASSWORD}
    database: data_warehouse
    timeout: 1800                   # 30 minute timeout for long queries
    max_memory_mb: 4096            # 4GB memory limit
    connection_pool_size: 4        # Lower concurrency, higher memory
    server_side_cursors: true      # Essential for large result sets
    
  feature_store:
    type: parquet
    base_path: /data/features      # Directory of Parquet files
    max_memory_mb: 2048
    
logging:
  level: DEBUG                     # Verbose logging for analysis
  format: json
  file: ./logs/data_science.log
  
performance:
  default_chunk_size: 5000        # Larger chunks for analytics
  max_tokens_streaming: 16000     # Allow larger responses
  buffer_timeout_seconds: 3600    # 1 hour buffer timeout
  
security:
  allowed_file_extensions:
    - .parquet
    - .feather  
    - .csv
    - .json
    - .h5
  allowed_paths:
    - /data
    - /shared/datasets
```

### Multi-Tenant SaaS

```yaml
# Configuration for multi-tenant SaaS application
databases:
  # Tenant database routing
  tenant_1:
    type: postgresql
    host: ${TENANT_1_DB_HOST}
    port: 5432
    user: ${TENANT_1_DB_USER}
    password: ${TENANT_1_DB_PASSWORD}
    database: tenant_1
    timeout: 60
    max_memory_mb: 512
    
  tenant_2:
    type: postgresql
    host: ${TENANT_2_DB_HOST}
    port: 5432
    user: ${TENANT_2_DB_USER}
    password: ${TENANT_2_DB_PASSWORD}
    database: tenant_2
    timeout: 60
    max_memory_mb: 512
    
  # Shared services
  shared_analytics:
    type: postgresql
    host: analytics.internal
    port: 5432
    user: ${ANALYTICS_USER}
    password: ${ANALYTICS_PASSWORD}
    database: analytics
    timeout: 300
    max_memory_mb: 1024
    
security:
  enable_authentication: true
  auth_token: ${API_TOKEN}
  enable_rate_limiting: true
  rate_limit_requests: 1000
  
  # Tenant isolation
  path_security:
    allowed_paths:
      - ./tenant_data/${TENANT_ID}
      
monitoring:
  enable_metrics: true
  alert_on_connection_failure: true
  alert_webhook_url: ${ALERT_WEBHOOK}
```

## Validation and Troubleshooting

### Configuration Validation

```bash
# Validate configuration file syntax
localdata-mcp --validate-config

# Check configuration with verbose output
localdata-mcp --config-file ./config.yaml --dry-run --verbose

# Test database connections
localdata-mcp --test-connections
```

### Common Configuration Issues

#### Invalid YAML Syntax
```bash
# Error: Invalid YAML syntax
yaml.scanner.ScannerError: mapping values are not allowed here

# Solution: Check YAML indentation and syntax
yamllint localdata-config.yaml
```

#### Environment Variable Not Found
```bash
# Error: Environment variable DB_PASSWORD not set
ConfigurationError: Required environment variable 'DB_PASSWORD' not found

# Solution: Set missing environment variables
export DB_PASSWORD=your_password_here
```

#### Database Connection Failed
```bash
# Error: Connection to database failed
ConnectionError: Could not connect to postgresql://localhost:5432/nonexistent

# Solution: Verify database settings
psql -h localhost -p 5432 -U username -d database_name -c "SELECT 1"
```

### Configuration Debugging

#### Enable Debug Logging
```yaml
logging:
  level: DEBUG
  format: plain
  console: true
```

#### Configuration Dump
```python
# Dump current configuration
from localdata_mcp import ConfigManager

config = ConfigManager()
print(config.dump_config())
```

#### Test Configuration
```python
# Test configuration programmatically
import yaml
from localdata_mcp import ConfigValidator

with open('localdata-config.yaml') as f:
    config = yaml.safe_load(f)

validator = ConfigValidator()
result = validator.validate(config)

if result.valid:
    print("✅ Configuration is valid")
else:
    print(f"❌ Configuration errors: {result.errors}")
```

### Performance Troubleshooting

#### Memory Usage Issues
```yaml
# Monitor memory usage
logging:
  level: DEBUG
  
performance:
  memory_check_interval: 5        # Check every 5 seconds
  
# Check logs for memory warnings
tail -f localdata.log | grep "memory"
```

#### Connection Pool Issues
```yaml
# Debug connection pool
logging:
  level: DEBUG
  
databases:
  problematic_db:
    connection_pool_size: 5
    max_overflow: 2
    pool_pre_ping: true          # Test connections
    
# Monitor connection pool metrics
curl http://localhost:9090/metrics | grep connection_pool
```

## Environment-Specific Examples

### Development Environment
```yaml
# development.yaml - Fast iteration, detailed logging
databases:
  dev_db:
    type: sqlite
    database: ./dev.db
    timeout: 10
    
logging:
  level: DEBUG
  format: plain
  console: true
  
performance:
  max_connections: 3
  default_chunk_size: 500
  
security:
  enable_sql_validation: false    # Allow experimental queries
```

### Staging Environment  
```yaml
# staging.yaml - Production-like with debugging
databases:
  staging_db:
    type: postgresql
    host: staging-db.company.com
    port: 5432
    user: ${STAGING_DB_USER}
    password: ${STAGING_DB_PASSWORD}
    database: staging
    timeout: 60
    max_memory_mb: 512
    
logging:
  level: INFO
  format: json
  file: ./logs/staging.log
  
performance:
  max_connections: 8
  
security:
  enable_sql_validation: true
  require_ssl: false              # Internal network
  
monitoring:
  enable_metrics: true
  enable_health_checks: true
```

### Production Environment
```yaml
# production.yaml - Optimized for stability and security
databases:
  primary:
    type: postgresql
    host: ${PROD_DB_HOST}
    port: 5432
    user: ${PROD_DB_USER}
    password: ${PROD_DB_PASSWORD}
    database: production
    timeout: 30
    max_memory_mb: 1024
    connection_pool_size: 10
    ssl_mode: require
    
logging:
  level: WARNING
  format: json
  console: false
  file: /var/log/localdata/production.log
  syslog: true
  max_size_mb: 50
  backup_count: 10
  
performance:
  global_memory_limit_mb: 4096
  max_connections: 15
  enable_query_cache: true
  
security:
  enable_sql_validation: true
  require_ssl: true
  enable_authentication: true
  auth_token: ${PROD_API_TOKEN}
  enable_rate_limiting: true
  
monitoring:
  enable_metrics: true
  enable_health_checks: true
  alert_on_connection_failure: true
  alert_on_memory_limit: true
  alert_webhook_url: ${ALERT_WEBHOOK}
```

This comprehensive configuration guide covers all aspects of LocalData MCP v1.3.1 configuration. Use it as a reference for setting up environments from simple development setups to complex multi-tenant production deployments.

For additional help:
- [Migration Guide](MIGRATION_GUIDE.md) - Upgrading from v1.3.0
- [Architecture Guide](ARCHITECTURE.md) - Understanding the system design
- [Troubleshooting Guide](TROUBLESHOOTING.md) - Solving common issues
- [API Reference](API_REFERENCE.md) - Complete tool documentation