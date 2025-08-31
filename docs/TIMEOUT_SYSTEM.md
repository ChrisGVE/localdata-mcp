# Advanced Query Timeout System

The LocalData MCP v1.3.1 includes a comprehensive timeout management system that prevents runaway queries from consuming resources indefinitely while providing database-specific timeout configurations and graceful cancellation.

## Overview

The timeout system provides:

- **Per-database timeout configuration** - Different databases can have different timeout settings
- **Database-specific cancellation support** - Network databases support full cancellation, file formats have limited support
- **Graceful resource cleanup** - Proper cleanup of connections, buffers, and memory on timeout
- **Integration with streaming pipeline** - Timeout checks during chunk processing
- **Comprehensive error handling** - Clear timeout messages with execution context
- **Real-time monitoring** - Track active operations and remaining time

## Configuration

Configure timeouts in your `localdata.yaml` configuration file:

```yaml
databases:
  # Local database - short timeout expected
  local_sqlite:
    type: sqlite
    connection_string: "sqlite:///local.db"
    query_timeout: 30          # 30 seconds
    connection_timeout: 10     # Connection timeout
    
  # Remote database - longer timeout for network latency  
  remote_postgres:
    type: postgresql
    connection_string: "postgresql://user:pass@host:5432/db"
    query_timeout: 300         # 5 minutes
    connection_timeout: 60     # 1 minute connection timeout
    
  # File format - medium timeout for I/O operations
  data_csv:
    type: csv
    connection_string: "/path/to/data.csv"
    query_timeout: 120         # 2 minutes
    connection_timeout: 15
```

## Timeout Configuration by Database Type

### Network Databases (Full Cancellation Support)
- **PostgreSQL**: 180-300 seconds (network latency + query complexity)
- **MySQL**: 120-300 seconds (depends on query complexity)
- **Redis**: 5-15 seconds (cache operations should be fast)
- **MongoDB**: 60-180 seconds (document query complexity varies)
- **Elasticsearch**: 30-120 seconds (search operations)

### Local Databases (Limited Cancellation Support)
- **SQLite**: 15-60 seconds (local file operations)
- **DuckDB**: 60-600 seconds (analytical workloads can be long)

### File Formats (No Cancellation Support)
- **CSV/TSV**: 60-300 seconds (depends on file size)
- **Excel/ODS**: 30-120 seconds (format parsing overhead)
- **JSON/YAML**: 30-180 seconds (parsing complexity)
- **Parquet**: 120-600 seconds (big data processing)

## Database-Specific Cancellation

The timeout system provides different levels of cancellation support:

### Full Cancellation Support
Network-based databases support immediate query cancellation:
- PostgreSQL, MySQL, Redis, MongoDB, Elasticsearch, InfluxDB, Neo4j, CouchDB
- Cancellation terminates the query on the database server
- Automatic transaction rollback and resource cleanup

### Limited Cancellation Support  
Local databases have limited cancellation capabilities:
- SQLite, DuckDB
- Process interruption may leave locks or incomplete transactions
- Manual cleanup may be required

### No Cancellation Support
File-based operations cannot be cancelled mid-process:
- CSV, JSON, YAML, Excel, Parquet, XML, etc.
- Must wait for operation to complete naturally
- Only resource cleanup is performed on timeout

## Usage Examples

### Basic Query Execution with Timeout

```python
# Query will use database-specific timeout from configuration
result = mcp_server.execute_query("remote_postgres", "SELECT * FROM large_table")

# If query exceeds timeout limit, returns error message:
# "Query Timeout Error: Query timed out after 305.2s (database: remote_postgres) 
# (execution time: 305.2s, reason: user_timeout)"
```

### Monitoring Active Operations

```python
from localdata_mcp.timeout_manager import get_timeout_manager

timeout_manager = get_timeout_manager()

# Get information about currently running queries
active_ops = timeout_manager.get_active_operations()
print(json.dumps(active_ops, indent=2))

# Example output:
{
  "streaming_001": {
    "database_name": "remote_postgres",
    "database_type": "postgresql", 
    "running_time": 45.2,
    "timeout_limit": 300,
    "time_remaining": 254.8,
    "cancelled": false,
    "supports_cancellation": true
  }
}
```

### Manual Cancellation

```python
# Cancel a long-running operation
success = timeout_manager.cancel_operation("streaming_001", 
                                          reason=TimeoutReason.MANUAL_CANCEL)
```

## Integration with Streaming Pipeline

The timeout system integrates seamlessly with the streaming execution pipeline:

1. **Timeout Context**: Each streaming query runs within a timeout context
2. **Chunk-Level Checks**: Timeout is checked before and after each chunk processing
3. **Graceful Termination**: If timeout occurs, current chunk completes and resources are cleaned up
4. **Metadata Reporting**: Timeout information is included in query results

```json
{
  "metadata": {
    "timeout_info": {
      "timeout_configured": true,
      "timeout_limit_seconds": 300,
      "time_remaining_seconds": 254.8,
      "database_name": "remote_postgres"
    },
    "streaming": true,
    "chunks_processed": 3
  }
}
```

## Error Messages and Debugging

### Timeout Error Messages

The system provides detailed timeout error messages:

```
Query Timeout Error: Query timed out after 305.2s (database: remote_postgres) 
(execution time: 305.2s, reason: user_timeout)
```

```  
Query Timeout Error: Query cancelled due to memory pressure after 45.6s 
(database: local_sqlite) (execution time: 45.6s, reason: memory_pressure)
```

### Timeout Reasons

- `user_timeout`: Query exceeded configured timeout limit
- `database_timeout`: Database-specific timeout exceeded  
- `memory_pressure`: Cancelled due to high memory usage
- `manual_cancel`: Manually cancelled by user
- `global_timeout`: System-wide timeout exceeded

## Best Practices

### Timeout Configuration

1. **Local databases**: Use short timeouts (15-60s) - should be fast
2. **Remote databases**: Allow for network latency (180-300s)
3. **File operations**: Consider file size and format complexity (60-300s)
4. **Analytical workloads**: Allow longer timeouts for complex queries (300-600s)
5. **Cache operations**: Use very short timeouts (5-15s)

### Query Optimization

1. **Monitor timeout patterns**: Look for frequently timing out queries
2. **Use query analysis**: Enable `enable_query_analysis: true` for timeout predictions
3. **Optimize slow queries**: Add indexes, rewrite complex queries
4. **Use appropriate chunk sizes**: Smaller chunks allow more responsive timeout checking

### Resource Management

1. **Set memory limits**: Configure `memory_limit_mb` to prevent memory exhaustion
2. **Enable auto-cleanup**: Use `auto_cleanup_buffers: true` for automatic resource management
3. **Monitor active operations**: Regularly check `get_active_operations()` for stuck queries
4. **Manual intervention**: Use `cancel_operation()` for manual query termination

## Configuration Reference

### Database-Level Timeout Settings

```yaml
databases:
  database_name:
    type: database_type
    connection_string: "connection_string"
    query_timeout: 300         # Query execution timeout in seconds
    connection_timeout: 30     # Connection establishment timeout in seconds
    max_connections: 10        # Maximum concurrent connections
    enabled: true              # Enable/disable database
    tags: ["tag1", "tag2"]     # Database tags for organization
```

### Performance Settings

```yaml
performance:
  memory_limit_mb: 2048              # Memory limit for query execution
  query_buffer_timeout: 600          # Buffer retention timeout
  max_concurrent_connections: 10     # Global connection limit
  enable_query_analysis: true        # Enable timeout prediction
  auto_cleanup_buffers: true         # Auto-cleanup on timeout
  memory_warning_threshold: 0.85     # Memory warning threshold
```

## API Reference

### QueryTimeoutManager

```python
from localdata_mcp.timeout_manager import get_timeout_manager

timeout_manager = get_timeout_manager()

# Get timeout configuration for database
config = timeout_manager.get_timeout_config("database_name")

# Check if operation is cancelled  
is_cancelled = timeout_manager.is_cancelled("operation_id")

# Cancel operation manually
success = timeout_manager.cancel_operation("operation_id")

# Get active operations
active_ops = timeout_manager.get_active_operations()
```

### TimeoutConfig

```python
from localdata_mcp.timeout_manager import TimeoutConfig

config = TimeoutConfig(
    query_timeout=300,              # Query timeout in seconds
    connection_timeout=30,          # Connection timeout in seconds
    database_name="my_database",    # Database identifier
    database_type=DatabaseType.POSTGRESQL,  # Database type
    allow_cancellation=True,        # Whether cancellation is supported
    cleanup_on_timeout=True         # Whether to cleanup on timeout
)
```

### QueryTimeoutError

```python
from localdata_mcp.timeout_manager import QueryTimeoutError, TimeoutReason

try:
    # Execute query that might timeout
    result = execute_query_with_timeout()
except QueryTimeoutError as e:
    print(f"Query timed out: {e.message}")
    print(f"Execution time: {e.execution_time}s") 
    print(f"Database: {e.database_name}")
    print(f"Reason: {e.timeout_reason.value}")
```

## Troubleshooting

### Common Issues

1. **Queries timing out frequently**
   - Increase timeout limits for the database type
   - Optimize queries with indexes or query rewriting
   - Check network connectivity for remote databases

2. **Memory pressure cancellations**
   - Increase `memory_limit_mb` setting
   - Reduce `chunk_size` to use less memory per operation
   - Enable `auto_cleanup_buffers` for automatic memory management

3. **Cannot cancel operations**  
   - File-based operations (CSV, Excel) cannot be cancelled mid-process
   - Local databases (SQLite) have limited cancellation support
   - Only network databases support full cancellation

4. **Timeout configuration not working**
   - Verify database configuration in YAML file
   - Check that database name matches exactly
   - Ensure `enable_query_analysis: true` for timeout predictions

### Debug Logging

Enable debug logging to troubleshoot timeout issues:

```yaml
logging:
  level: debug
  console_output: true
  file_path: "/var/log/localdata-timeout-debug.log"
```

This will show detailed timeout monitoring, cancellation attempts, and cleanup operations.