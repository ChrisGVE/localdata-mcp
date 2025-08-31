# LocalData MCP API Reference

## Overview

LocalData MCP v1.3.1 provides 14 comprehensive MCP tools for database operations, file processing, and data management. This reference documents all tools, their parameters, response formats, and usage examples.

## Table of Contents

1. [Connection Management](#connection-management)
2. [Query Execution](#query-execution)
3. [Database Exploration](#database-exploration)
4. [Data Sampling](#data-sampling)
5. [File Operations](#file-operations)
6. [Buffer Management](#buffer-management)
7. [Response Formats](#response-formats)
8. [Error Handling](#error-handling)
9. [Performance Considerations](#performance-considerations)
10. [Advanced Usage Patterns](#advanced-usage-patterns)

## Connection Management

### connect_database

Establishes a connection to a database or structured file.

**Parameters:**
- `database_name` (string, required): Unique identifier for the connection
- `database_type` (string, required): Type of database or file format
- `connection_string` (string, required): Connection details or file path

**Supported Database Types:**
- `postgresql`, `mysql`, `sqlite`, `duckdb`
- `mongodb`, `redis`, `elasticsearch`, `influxdb`, `neo4j`, `couchdb`  
- `csv`, `tsv`, `json`, `yaml`, `toml`, `xml`, `ini`
- `xlsx`, `xls`, `ods`, `numbers`
- `parquet`, `feather`, `arrow`, `hdf5`

**Examples:**

```python
# PostgreSQL database
connect_database("prod_db", "postgresql", "postgresql://user:pass@localhost:5432/database")

# Local SQLite file
connect_database("local_data", "sqlite", "./data/local.db")

# CSV file (large files automatically use SQLite storage)
connect_database("sales_data", "csv", "./data/sales_2024.csv")

# Excel spreadsheet with all sheets
connect_database("workbook", "xlsx", "./reports/quarterly_report.xlsx")

# Excel with specific sheet
connect_database("q1_data", "xlsx", "./reports/quarterly_report.xlsx?sheet=Q1_Sales")

# JSON structured data
connect_database("config", "json", "./config/application.json")

# MongoDB collection
connect_database("documents", "mongodb", "mongodb://localhost:27017/app_db")

# Redis cache
connect_database("cache", "redis", "redis://localhost:6379/0")
```

**Response:**
```json
{
  "success": true,
  "message": "Successfully connected to database 'prod_db'",
  "connection_info": {
    "database_name": "prod_db",
    "database_type": "postgresql", 
    "host": "localhost",
    "port": 5432,
    "database": "database",
    "connection_pool_size": 5,
    "timeout": 30,
    "ssl_mode": "prefer"
  },
  "metadata": {
    "connection_time": "2024-01-15T14:30:25.123Z",
    "server_version": "PostgreSQL 15.2",
    "available_schemas": ["public", "analytics"],
    "estimated_size": "2.5GB"
  }
}
```

**v1.3.1 Enhancements:**
- Automatic large file detection and SQLite conversion (>100MB files)
- Connection pooling with configurable pool sizes
- Enhanced metadata including server version and schema information
- Multi-sheet Excel support with automatic table naming

---

### disconnect_database

Closes a database connection and releases resources.

**Parameters:**
- `database_name` (string, required): Name of the database connection to close

**Example:**
```python
disconnect_database("prod_db")
```

**Response:**
```json
{
  "success": true,
  "message": "Successfully disconnected from database 'prod_db'",
  "cleanup_info": {
    "connections_closed": 1,
    "buffers_cleared": 2,
    "temp_files_cleaned": 1,
    "memory_freed_mb": 45
  }
}
```

---

### list_databases

Lists all active database connections with their status.

**Parameters:** None

**Example:**
```python
list_databases()
```

**Response:**
```json
{
  "active_connections": [
    {
      "database_name": "prod_db",
      "database_type": "postgresql",
      "status": "connected",
      "connection_time": "2024-01-15T14:30:25.123Z",
      "last_query_time": "2024-01-15T14:45:12.456Z",
      "query_count": 15,
      "connection_pool_usage": "3/5",
      "memory_usage_mb": 23
    },
    {
      "database_name": "sales_data",
      "database_type": "csv",
      "status": "connected",
      "file_size_mb": 156,
      "temp_storage": "sqlite",
      "temp_file": "/tmp/localdata_sales_data_abc123.db",
      "memory_usage_mb": 12
    }
  ],
  "summary": {
    "total_connections": 2,
    "connection_limit": 10,
    "total_memory_usage_mb": 35,
    "memory_limit_mb": 1024
  }
}
```

## Query Execution

### execute_query

Executes SQL queries and returns results in markdown table format.

**Parameters:**
- `database_name` (string, required): Name of the database connection
- `query` (string, required): SQL query to execute

**Example:**
```python
execute_query("prod_db", "SELECT product_name, sales, region FROM sales_summary ORDER BY sales DESC LIMIT 10")
```

**Response (Small Result Set):**
```markdown
| product_name | sales | region |
|--------------|-------|---------|
| Widget Pro   | 15000 | North   |
| Gadget Ultra | 12500 | South   |
| Tool Master  | 11000 | East    |
```

**Response (Large Result Set - Auto-Buffering):**
```markdown
| product_name | sales | region |
|--------------|-------|---------|
| Widget Pro   | 15000 | North   |
| Gadget Ultra | 12500 | South   |
| Tool Master  | 11000 | East    |
... (showing first 10 rows of 25,000 total)

**Query Results:** 25,000 rows total, showing first 10 rows. 
Large result set has been buffered with ID: `prod_db_1640995200_a1b2c3`
Access more data using: `get_query_chunk("prod_db_1640995200_a1b2c3", 11, "1000")`
Buffer expires: 2024-01-15T15:45:00Z
```

---

### execute_query_json

Executes SQL queries and returns results in JSON format with enhanced metadata.

**Parameters:**
- `database_name` (string, required): Name of the database connection
- `query` (string, required): SQL query to execute

**Example:**
```python
execute_query_json("prod_db", """
    SELECT 
        customer_segment,
        COUNT(*) as customer_count,
        AVG(lifetime_value) as avg_ltv,
        SUM(total_orders) as total_orders
    FROM customer_analytics 
    WHERE last_activity >= '2024-01-01'
    GROUP BY customer_segment
    ORDER BY avg_ltv DESC
""")
```

**Response (Direct Result):**
```json
{
  "data": [
    {
      "customer_segment": "Premium",
      "customer_count": 1250,
      "avg_ltv": 4500.75,
      "total_orders": 18750
    },
    {
      "customer_segment": "Standard", 
      "customer_count": 3400,
      "avg_ltv": 1200.50,
      "total_orders": 22100
    }
  ],
  "metadata": {
    "total_rows": 2,
    "columns": ["customer_segment", "customer_count", "avg_ltv", "total_orders"],
    "column_types": {
      "customer_segment": "string",
      "customer_count": "integer", 
      "avg_ltv": "float",
      "total_orders": "integer"
    },
    "query_execution_time": "0.245s",
    "memory_usage_mb": 0.8,
    "query_complexity": "medium",
    "estimated_cost": "low"
  }
}
```

**Response (Buffered Large Result):**
```json
{
  "first_10_rows": [
    {"customer_id": 1, "name": "Alice Johnson", "segment": "Premium"},
    {"customer_id": 2, "name": "Bob Smith", "segment": "Standard"}
  ],
  "metadata": {
    "total_rows": 85000,
    "columns": ["customer_id", "name", "segment", "lifetime_value", "last_order"],
    "column_types": {
      "customer_id": "integer",
      "name": "string",
      "segment": "string",
      "lifetime_value": "float",
      "last_order": "datetime"
    },
    "query_execution_time": "12.3s",
    "memory_usage_mb": 156.7,
    "query_complexity": "high",
    "estimated_tokens": 2500000,
    "row_sample_analysis": {
      "avg_row_size_bytes": 320,
      "text_heavy_columns": ["name"],
      "numeric_columns": ["customer_id", "lifetime_value"]
    }
  },
  "buffering_info": {
    "query_id": "prod_db_1640995200_a1b2c3",
    "buffer_size": 85000,
    "chunk_size": 1000,
    "chunks_available": 85,
    "expiry_time": "2024-01-15T15:45:00Z",
    "buffer_storage": "temporary_sqlite",
    "estimated_chunks_needed": 85
  },
  "performance_hints": [
    "Consider adding LIMIT clause for faster response",
    "Query involves full table scan - consider adding index on last_order column",
    "Large result set - use get_query_chunk() for pagination"
  ]
}
```

## Database Exploration

### describe_database

Returns comprehensive information about database structure and metadata.

**Parameters:**
- `database_name` (string, required): Name of the database connection

**Example:**
```python
describe_database("prod_db")
```

**Response:**
```json
{
  "database_info": {
    "database_name": "prod_db",
    "database_type": "postgresql",
    "server_version": "PostgreSQL 15.2",
    "connection_status": "active",
    "uptime": "25 days, 14:32:18"
  },
  "schemas": {
    "public": {
      "table_count": 12,
      "view_count": 3,
      "index_count": 45,
      "estimated_size": "1.2GB"
    },
    "analytics": {
      "table_count": 8,
      "view_count": 5,
      "index_count": 23,
      "estimated_size": "800MB"
    }
  },
  "tables": {
    "users": {
      "schema": "public",
      "row_count": 25000,
      "column_count": 12,
      "size": "45MB",
      "last_modified": "2024-01-15T12:30:00Z",
      "indexes": ["users_pkey", "users_email_idx", "users_created_at_idx"]
    },
    "orders": {
      "schema": "public", 
      "row_count": 150000,
      "column_count": 8,
      "size": "89MB",
      "last_modified": "2024-01-15T14:25:00Z",
      "indexes": ["orders_pkey", "orders_user_id_idx", "orders_created_at_idx"]
    }
  },
  "performance_stats": {
    "total_queries_today": 1250,
    "avg_query_time": "0.145s",
    "cache_hit_ratio": "94.2%",
    "active_connections": "3/100"
  }
}
```

---

### describe_table

Returns detailed information about a specific table structure and statistics.

**Parameters:**
- `database_name` (string, required): Name of the database connection
- `table_name` (string, required): Name of the table to describe

**Example:**
```python
describe_table("prod_db", "customer_analytics")
```

**Response:**
```json
{
  "table_info": {
    "table_name": "customer_analytics",
    "schema": "public",
    "table_type": "BASE TABLE",
    "row_count": 85000,
    "column_count": 15,
    "size": "156MB",
    "created": "2023-06-15T10:00:00Z",
    "last_modified": "2024-01-15T14:25:00Z",
    "last_analyzed": "2024-01-14T02:00:00Z"
  },
  "columns": [
    {
      "column_name": "customer_id",
      "data_type": "integer",
      "is_nullable": false,
      "is_primary_key": true,
      "has_index": true,
      "default_value": null,
      "max_length": null,
      "numeric_precision": 32,
      "numeric_scale": 0
    },
    {
      "column_name": "customer_segment",
      "data_type": "varchar",
      "is_nullable": true,
      "is_primary_key": false,
      "has_index": true,
      "default_value": "Standard",
      "max_length": 50,
      "unique_values": 4,
      "most_common_values": ["Standard", "Premium", "Enterprise", "Trial"]
    },
    {
      "column_name": "lifetime_value",
      "data_type": "numeric",
      "is_nullable": true,
      "is_primary_key": false,
      "has_index": false,
      "default_value": 0.0,
      "max_length": null,
      "numeric_precision": 10,
      "numeric_scale": 2,
      "min_value": 0.0,
      "max_value": 25000.0,
      "avg_value": 1250.75
    }
  ],
  "indexes": [
    {
      "index_name": "customer_analytics_pkey",
      "columns": ["customer_id"],
      "is_unique": true,
      "is_primary": true,
      "size": "2MB"
    },
    {
      "index_name": "customer_segment_idx",
      "columns": ["customer_segment"],
      "is_unique": false,
      "is_primary": false,
      "size": "1.5MB"
    }
  ],
  "relationships": {
    "foreign_keys": [
      {
        "column": "customer_id",
        "references_table": "customers",
        "references_column": "id",
        "constraint_name": "fk_customer_analytics_customer"
      }
    ],
    "referenced_by": [
      {
        "table": "customer_orders",
        "column": "customer_id",
        "constraint_name": "fk_orders_customer_analytics"
      }
    ]
  },
  "statistics": {
    "table_scans_today": 25,
    "index_scans_today": 450,
    "inserts_today": 1200,
    "updates_today": 340,
    "deletes_today": 12
  },
  "performance_recommendations": [
    "Consider partitioning by customer_segment for better query performance",
    "Add index on (last_activity, customer_segment) for common query patterns",
    "Table statistics are recent - no ANALYZE needed"
  ]
}
```

---

### find_table

Searches for tables matching a name pattern across all schemas.

**Parameters:**
- `database_name` (string, required): Name of the database connection
- `table_pattern` (string, required): Pattern to search for (supports wildcards)

**Example:**
```python
find_table("prod_db", "customer*")
```

**Response:**
```json
{
  "matches": [
    {
      "table_name": "customers",
      "schema": "public",
      "table_type": "BASE TABLE",
      "row_count": 25000,
      "size": "45MB",
      "match_type": "exact_prefix"
    },
    {
      "table_name": "customer_analytics",
      "schema": "public", 
      "table_type": "BASE TABLE",
      "row_count": 85000,
      "size": "156MB",
      "match_type": "exact_prefix"
    },
    {
      "table_name": "customer_segments",
      "schema": "analytics",
      "table_type": "VIEW",
      "row_count": null,
      "size": null,
      "match_type": "exact_prefix"
    }
  ],
  "search_info": {
    "pattern": "customer*",
    "total_matches": 3,
    "search_time": "0.025s",
    "schemas_searched": ["public", "analytics", "reporting"]
  },
  "suggestions": [
    "Use describe_table() to get detailed information about specific tables",
    "Consider get_table_sample() to preview data from these tables"
  ]
}
```

## Data Sampling

### get_table_sample

Returns a sample of data from a table for quick data exploration.

**Parameters:**
- `database_name` (string, required): Name of the database connection
- `table_name` (string, required): Name of the table to sample
- `limit` (integer, optional): Number of rows to return (default: 10, max: 100)

**Example:**
```python
get_table_sample("prod_db", "customer_analytics", 5)
```

**Response (Markdown Format):**
```markdown
# Sample data from customer_analytics (5 of 85,000 rows)

| customer_id | customer_segment | lifetime_value | total_orders | last_activity | 
|-------------|------------------|----------------|--------------|---------------|
| 1001        | Premium          | 4500.75        | 45           | 2024-01-14    |
| 1002        | Standard         | 1200.50        | 12           | 2024-01-13    |
| 1003        | Enterprise       | 12000.00       | 125          | 2024-01-15    |
| 1004        | Standard         | 875.25         | 8            | 2024-01-10    |
| 1005        | Premium          | 3200.80        | 32           | 2024-01-14    |

**Table Statistics:**
- Total rows: 85,000
- Sample represents: 0.006% of data
- Column types: integer, string, float, integer, date
- Last updated: 2024-01-15T14:25:00Z
```

---

### get_table_sample_json

Returns a sample of data from a table in JSON format with enhanced metadata.

**Parameters:**
- `database_name` (string, required): Name of the database connection
- `table_name` (string, required): Name of the table to sample
- `limit` (integer, optional): Number of rows to return (default: 10, max: 100)

**Example:**
```python
get_table_sample_json("prod_db", "customer_analytics", 5)
```

**Response:**
```json
{
  "sample_data": [
    {
      "customer_id": 1001,
      "customer_segment": "Premium",
      "lifetime_value": 4500.75,
      "total_orders": 45,
      "last_activity": "2024-01-14"
    },
    {
      "customer_id": 1002,
      "customer_segment": "Standard", 
      "lifetime_value": 1200.50,
      "total_orders": 12,
      "last_activity": "2024-01-13"
    }
  ],
  "sample_info": {
    "sample_size": 5,
    "total_rows": 85000,
    "sample_percentage": 0.006,
    "sampling_method": "random",
    "sample_time": "2024-01-15T14:45:30.123Z"
  },
  "column_analysis": {
    "customer_id": {
      "data_type": "integer",
      "min_value": 1001,
      "max_value": 1005,
      "unique_values": 5,
      "null_count": 0
    },
    "customer_segment": {
      "data_type": "string",
      "unique_values": 3,
      "value_distribution": {
        "Premium": 2,
        "Standard": 2, 
        "Enterprise": 1
      },
      "null_count": 0
    },
    "lifetime_value": {
      "data_type": "float",
      "min_value": 875.25,
      "max_value": 12000.0,
      "avg_value": 4355.66,
      "null_count": 0
    }
  },
  "data_quality": {
    "completeness": 100.0,
    "consistency_score": 95.2,
    "data_freshness": "2 hours ago",
    "potential_issues": []
  },
  "recommendations": [
    "Sample shows good data quality with no null values",
    "Consider stratified sampling by customer_segment for better representation",
    "Use execute_query() with WHERE clause for targeted data exploration"
  ]
}
```

## File Operations

### read_text_file

Reads and parses structured text files (JSON, YAML, XML, etc.).

**Parameters:**
- `file_path` (string, required): Path to the file to read
- `file_format` (string, required): File format for parsing

**Supported Formats:**
- `json`, `yaml`, `yml`, `toml`, `xml`, `ini`, `csv`, `tsv`

**Example:**
```python
read_text_file("./config/application.yaml", "yaml")
```

**Response:**
```json
{
  "file_info": {
    "file_path": "./config/application.yaml",
    "file_format": "yaml",
    "file_size": "2.3KB",
    "last_modified": "2024-01-14T16:30:00Z",
    "encoding": "utf-8"
  },
  "parsed_content": {
    "database": {
      "host": "localhost",
      "port": 5432,
      "name": "production"
    },
    "logging": {
      "level": "INFO",
      "format": "json"
    },
    "features": {
      "caching": true,
      "metrics": true
    }
  },
  "structure_analysis": {
    "total_keys": 7,
    "nested_levels": 2,
    "data_types": {
      "string": 4,
      "integer": 1,
      "boolean": 2
    }
  }
}
```

## Buffer Management

### get_query_chunk

Retrieves a specific chunk of data from a buffered query result.

**Parameters:**
- `query_id` (string, required): Buffer ID from previous query execution
- `start_row` (integer, required): Starting row number (1-indexed)
- `limit` (string, required): Number of rows to retrieve

**Example:**
```python
get_query_chunk("prod_db_1640995200_a1b2c3", 1001, "1000")
```

**Response:**
```json
{
  "chunk_data": [
    {
      "customer_id": 2001,
      "name": "Sarah Johnson",
      "segment": "Premium",
      "lifetime_value": 3500.0
    }
  ],
  "chunk_info": {
    "query_id": "prod_db_1640995200_a1b2c3",
    "start_row": 1001,
    "returned_rows": 1000,
    "chunk_number": 2,
    "total_chunks": 85,
    "has_more_data": true,
    "next_start_row": 2001
  },
  "buffer_status": {
    "expiry_time": "2024-01-15T15:45:00Z",
    "time_remaining": "8m 23s",
    "memory_usage_mb": 156.7,
    "access_count": 3,
    "created_time": "2024-01-15T14:35:00Z"
  }
}
```

---

### get_buffered_query_info

Returns information about a buffered query without retrieving data.

**Parameters:**
- `query_id` (string, required): Buffer ID to inspect

**Example:**
```python
get_buffered_query_info("prod_db_1640995200_a1b2c3")
```

**Response:**
```json
{
  "query_info": {
    "query_id": "prod_db_1640995200_a1b2c3",
    "database_name": "prod_db",
    "original_query": "SELECT * FROM customer_analytics WHERE last_activity >= '2024-01-01'",
    "created_time": "2024-01-15T14:35:00Z",
    "expiry_time": "2024-01-15T15:45:00Z",
    "query_execution_time": "12.3s"
  },
  "buffer_details": {
    "total_rows": 85000,
    "chunk_size": 1000,
    "total_chunks": 85,
    "chunks_accessed": 3,
    "buffer_storage": "temporary_sqlite",
    "buffer_size_mb": 156.7,
    "compression_ratio": "85%"
  },
  "access_pattern": {
    "first_access": "2024-01-15T14:35:05Z",
    "last_access": "2024-01-15T14:42:18Z",
    "access_count": 3,
    "unique_chunk_requests": [1, 2, 3],
    "most_requested_chunk": 1
  },
  "status": {
    "expired": false,
    "time_remaining": "8m 12s",
    "health": "healthy",
    "warnings": []
  }
}
```

---

### clear_query_buffer

Manually clears a query buffer and frees associated resources.

**Parameters:**
- `query_id` (string, required): Buffer ID to clear

**Example:**
```python
clear_query_buffer("prod_db_1640995200_a1b2c3")
```

**Response:**
```json
{
  "success": true,
  "message": "Query buffer cleared successfully",
  "cleanup_info": {
    "query_id": "prod_db_1640995200_a1b2c3",
    "rows_cleared": 85000,
    "memory_freed_mb": 156.7,
    "temp_files_removed": 1,
    "cleanup_time": "0.125s"
  }
}
```

## Response Formats

### Standard Response Structure

All LocalData MCP tools follow a consistent response structure:

```json
{
  "success": true,                    // Operation success status
  "data": {},                        // Main response data
  "metadata": {                      // Operation metadata
    "execution_time": "0.145s",
    "memory_usage_mb": 2.3,
    "query_complexity": "medium"
  },
  "warnings": [],                    // Non-critical warnings
  "performance_hints": []            // Optimization suggestions
}
```

### Error Response Format

Error responses include detailed troubleshooting information:

```json
{
  "success": false,
  "error": {
    "code": "connection_failed",
    "message": "Failed to connect to PostgreSQL database",
    "details": {
      "database_name": "prod_db",
      "host": "localhost",
      "port": 5432,
      "error_type": "connection_timeout"
    },
    "suggestions": [
      "Verify database server is running",
      "Check network connectivity",
      "Validate connection credentials"
    ],
    "documentation": "https://github.com/ChrisGVE/localdata-mcp/blob/main/TROUBLESHOOTING.md#connection-issues"
  },
  "timestamp": "2024-01-15T14:45:30.123Z"
}
```

### Metadata Fields

**Common metadata fields across responses:**

| Field | Type | Description |
|-------|------|-------------|
| `execution_time` | string | Time taken to execute the operation |
| `memory_usage_mb` | float | Memory consumed during operation |
| `query_complexity` | string | Query complexity: low, medium, high |
| `estimated_tokens` | integer | Estimated AI tokens in response |
| `cache_hit` | boolean | Whether result came from cache |
| `optimization_applied` | array | Applied query optimizations |

## Error Handling

### Error Categories

**Connection Errors:**
- `connection_failed`: Database connection failure
- `connection_timeout`: Connection establishment timeout
- `authentication_failed`: Invalid credentials
- `connection_limit_exceeded`: Too many concurrent connections

**Query Errors:**
- `sql_syntax_error`: Invalid SQL syntax
- `sql_injection_detected`: Blocked malicious query
- `query_timeout`: Query execution timeout
- `permission_denied`: Insufficient database permissions

**Resource Errors:**
- `memory_limit_exceeded`: Operation exceeds memory limits
- `buffer_expired`: Query buffer has expired
- `file_not_found`: Requested file doesn't exist
- `path_security_violation`: File path outside allowed directory

**Data Errors:**
- `table_not_found`: Specified table doesn't exist
- `column_not_found`: Specified column doesn't exist
- `data_type_error`: Data type conversion failure
- `constraint_violation`: Database constraint violation

### Error Recovery Examples

**Connection Recovery:**
```python
try:
    connect_database("prod_db", "postgresql", "postgresql://localhost:5432/db")
except ConnectionError as e:
    if e.code == "connection_timeout":
        # Retry with longer timeout
        connect_database("prod_db", "postgresql", 
                        "postgresql://localhost:5432/db?connect_timeout=60")
    elif e.code == "authentication_failed":
        # Check credentials and retry
        # Update credentials and reconnect
        pass
```

**Query Recovery:**
```python
try:
    result = execute_query_json("prod_db", "SELECT * FROM huge_table")
except QueryError as e:
    if e.code == "query_timeout":
        # Use pagination instead
        result = execute_query_json("prod_db", "SELECT * FROM huge_table LIMIT 10000")
    elif e.code == "memory_limit_exceeded":
        # Use smaller chunks
        result = execute_query_json("prod_db", "SELECT * FROM huge_table LIMIT 1000")
```

**Buffer Recovery:**
```python
try:
    chunk = get_query_chunk("prod_db_1640995200_a1b2c3", 1001, "1000")
except BufferError as e:
    if e.code == "buffer_expired":
        # Re-execute original query
        result = execute_query_json("prod_db", original_query)
        query_id = result["buffering_info"]["query_id"]
        chunk = get_query_chunk(query_id, 1001, "1000")
```

## Performance Considerations

### Query Optimization Guidelines

**Small Queries (< 1000 rows):**
- Use `execute_query()` for markdown output
- Use `execute_query_json()` for programmatic access
- Results returned immediately, no buffering

**Medium Queries (1000-10000 rows):**
- Automatic streaming with first chunk returned immediately
- Consider using `LIMIT` clauses for faster response
- Monitor memory usage in response metadata

**Large Queries (> 10000 rows):**
- Automatic buffering to temporary storage
- Access via `get_query_chunk()` for pagination
- Use `get_buffered_query_info()` to check status

### Memory Management Best Practices

**Monitor Memory Usage:**
```python
# Check memory usage in response metadata
result = execute_query_json("db", query)
memory_used = result["metadata"]["memory_usage_mb"]

if memory_used > 100:
    print("High memory usage detected, consider optimization")
```

**Optimize for Large Results:**
```python
# Instead of: SELECT * FROM large_table
# Use pagination:
for offset in range(0, 100000, 5000):
    chunk_query = f"SELECT * FROM large_table LIMIT 5000 OFFSET {offset}"
    chunk_result = execute_query("db", chunk_query)
    process_chunk(chunk_result)
```

### Connection Pool Optimization

**Monitor Connection Usage:**
```python
connections = list_databases()
usage = connections["summary"]

if usage["total_connections"] >= usage["connection_limit"] * 0.8:
    # Close unused connections
    disconnect_database("unused_db")
```

**Reuse Connections:**
```python
# Instead of multiple connections:
# connect_database("db1", "postgresql", conn_string)
# connect_database("db2", "postgresql", conn_string)  # Same database!

# Use one connection:
connect_database("shared_db", "postgresql", conn_string)
```

## Advanced Usage Patterns

### Multi-Database Analytics

```python
# Setup connections
connect_database("sales", "postgresql", "postgresql://localhost/sales")
connect_database("marketing", "mysql", "mysql://localhost/marketing")
connect_database("config", "yaml", "./config.yaml")

# Query across databases
sales_data = execute_query_json("sales", """
    SELECT product_id, sales_amount, sale_date 
    FROM transactions 
    WHERE sale_date >= '2024-01-01'
""")

marketing_data = execute_query_json("marketing", """
    SELECT campaign_id, product_id, impressions, clicks, cost
    FROM campaigns 
    WHERE start_date >= '2024-01-01'
""")

# Read configuration
config = read_text_file("./config.yaml", "yaml")
```

### Large Dataset Processing

```python
# Connect to large CSV file
connect_database("bigdata", "csv", "./massive_dataset.csv")

# Get overview
info = describe_database("bigdata")
print(f"Dataset has {info['tables']['data']['row_count']} rows")

# Process in chunks
large_result = execute_query_json("bigdata", "SELECT * FROM data WHERE value > 1000")

if "buffering_info" in large_result:
    query_id = large_result["buffering_info"]["query_id"]
    total_chunks = large_result["buffering_info"]["chunks_available"]
    
    # Process each chunk
    for chunk_num in range(1, total_chunks + 1):
        start_row = (chunk_num - 1) * 1000 + 1
        chunk = get_query_chunk(query_id, start_row, "1000")
        
        # Process chunk data
        process_data_chunk(chunk["chunk_data"])
        
        # Optional: Progress reporting
        if chunk_num % 10 == 0:
            progress = (chunk_num / total_chunks) * 100
            print(f"Processed {progress:.1f}% of data")
    
    # Cleanup
    clear_query_buffer(query_id)
```

### Streaming Data Pipeline

```python
def setup_streaming_pipeline():
    """Setup multi-source streaming data pipeline"""
    
    # Connect to various sources
    connect_database("transactions", "postgresql", "postgresql://prod-db/transactions")
    connect_database("user_events", "mongodb", "mongodb://events-db/events")
    connect_database("cache", "redis", "redis://cache/0")
    
    # Real-time transaction stream
    latest_transactions = execute_query_json("transactions", """
        SELECT transaction_id, user_id, amount, created_at
        FROM transactions 
        WHERE created_at > NOW() - INTERVAL '1 hour'
        ORDER BY created_at DESC
    """)
    
    # User behavior events
    recent_events = execute_query_json("user_events", """
        {
            "find": "events",
            "filter": {
                "timestamp": {"$gte": new Date(Date.now() - 3600000)}
            },
            "sort": {"timestamp": -1}
        }
    """)
    
    # Cache frequently accessed data
    execute_query("cache", "SET hourly_summary '{json.dumps(summary)}'")
    
    return {
        "transactions": latest_transactions,
        "events": recent_events,
        "pipeline_status": "active"
    }
```

### Multi-Format Data Integration

```python
def integrate_multi_format_data():
    """Integrate data from multiple file formats"""
    
    # Excel spreadsheet with multiple sheets
    connect_database("sales_report", "xlsx", "./Q4_2024_Sales.xlsx")
    
    # JSON configuration
    connect_database("config", "json", "./data_config.json")
    
    # Parquet analytics files
    connect_database("analytics", "parquet", "./user_analytics.parquet")
    
    # CSV transaction exports
    connect_database("exports", "csv", "./transaction_export.csv")
    
    # Query each source
    sales_summary = execute_query("sales_report", "SELECT * FROM Q4_Summary")
    config_data = read_text_file("./data_config.json", "json")
    analytics_data = execute_query("analytics", "SELECT user_id, conversion_rate FROM data")
    transactions = execute_query("exports", "SELECT * FROM data WHERE amount > 1000")
    
    return {
        "sales": sales_summary,
        "config": config_data["parsed_content"],
        "analytics": analytics_data,
        "transactions": transactions
    }
```

This comprehensive API reference covers all LocalData MCP v1.3.1 tools and their enhanced capabilities. The new streaming architecture, intelligent buffering, and rich metadata provide powerful capabilities for handling datasets of any size while maintaining optimal performance and user experience.

For additional information:
- [Architecture Guide](ARCHITECTURE.md) - System design and components
- [Configuration Guide](CONFIGURATION.md) - Detailed configuration options  
- [Migration Guide](MIGRATION_GUIDE.md) - Upgrading from previous versions
- [Troubleshooting Guide](TROUBLESHOOTING.md) - Common issues and solutions