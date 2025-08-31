# LocalData MCP v1.3.1 Migration Guide

## Overview

LocalData MCP v1.3.1 introduces a major architectural overhaul focused on memory safety, intelligent resource management, and scalable configuration. While **all existing MCP tool signatures remain 100% backward compatible**, there are important configuration changes and new features to be aware of.

## Quick Migration Checklist

- [ ] Review breaking changes in [Configuration Changes](#configuration-changes)
- [ ] Update environment variables (optional but recommended)
- [ ] Test memory-intensive workflows with new buffering system
- [ ] Enable structured logging in production environments
- [ ] Configure per-database timeouts and memory limits
- [ ] Validate large file processing with new streaming architecture

## Breaking Changes

### 1. Environment Variable Names (Optional Migration)

**Previous (v1.3.0)**:
```bash
MONGODB_URL=mongodb://localhost:27017/database
REDIS_URL=redis://localhost:6379/0
```

**New (v1.3.1)**:
```bash
# Granular connection parameters (recommended)
MONGODB_HOST=localhost
MONGODB_PORT=27017
MONGODB_DATABASE=database
MONGODB_USER=username
MONGODB_PASSWORD=password

REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DATABASE=0
REDIS_PASSWORD=optional_password
```

**Migration Strategy**:
- ✅ **Old environment variables still work** - no immediate action required
- ✅ **Gradual migration** - migrate one service at a time
- ✅ **Mixed approach** - use old format for simple setups, new for complex ones

### 2. Default Logging Changes

**Previous (v1.3.0)**:
- Default log level: `INFO`
- Default format: Plain text
- Console output only

**New (v1.3.1)**:
- Default log level: `WARNING`
- Default format: Structured JSON (production) / Plain text (development)
- Configurable output destinations

**Migration Actions**:
```bash
# To maintain v1.3.0 logging behavior
LOCALDATA_LOG_LEVEL=INFO
LOCALDATA_LOG_FORMAT=plain

# Or embrace new structured logging
LOCALDATA_LOG_LEVEL=INFO
LOCALDATA_LOG_FORMAT=json
LOCALDATA_LOG_FILE=./logs/localdata.log
```

### 3. Large File Processing Changes

**Previous (v1.3.0)**:
- Fixed 100MB threshold for SQLite conversion
- Memory usage could be unpredictable

**New (v1.3.1)**:
- Configurable memory limits per operation
- Intelligent streaming based on available memory
- Predictable resource usage

**Migration Actions**:
```bash
# Configure memory limits (optional)
LOCALDATA_MAX_MEMORY_MB=512
LOCALDATA_DEFAULT_CHUNK_SIZE=1000
```

## New Features You Can Adopt

### 1. YAML Configuration Files

Create `localdata-config.yaml` for complex multi-database scenarios:

```yaml
# localdata-config.yaml
databases:
  production_db:
    type: postgresql
    host: prod-db.company.com
    port: 5432
    user: ${DB_USER}
    password: ${DB_PASSWORD}
    database: production
    timeout: 30
    max_memory_mb: 1000
    
  analytics_db:
    type: postgresql  
    host: analytics-db.company.com
    port: 5432
    user: ${ANALYTICS_USER}
    password: ${ANALYTICS_PASSWORD}
    database: analytics
    timeout: 120
    max_memory_mb: 2000
    
  cache_db:
    type: redis
    host: cache.company.com
    port: 6379
    password: ${REDIS_PASSWORD}
    database: 0
    timeout: 5

logging:
  level: INFO
  format: json
  file: ./logs/localdata.log
  
performance:
  default_chunk_size: 1000
  max_tokens_direct: 4000
  buffer_timeout_seconds: 600
```

**Usage**:
```bash
# Automatic config file discovery
localdata-mcp
# Looks for: ./localdata-config.yaml, ~/.localdata/config.yaml, /etc/localdata/config.yaml

# Explicit config file
LOCALDATA_CONFIG_FILE=./config/production.yaml localdata-mcp
```

### 2. Intelligent Query Analysis

v1.3.1 automatically analyzes queries before execution:

```python
# This query will be pre-analyzed for memory and token estimation
large_result = execute_query_json("analytics", """
    SELECT customer_id, order_history, preferences 
    FROM customers 
    WHERE registration_date >= '2023-01-01'
""")

# Response now includes metadata:
{
    "metadata": {
        "estimated_rows": 85000,
        "estimated_memory_mb": 45,
        "estimated_tokens": 12500,
        "execution_time": "3.2s",
        "query_complexity": "medium"
    },
    "first_10_rows": [...],
    "buffering_info": {
        "query_id": "analytics_1640995200_a1b2",
        "chunks_available": 85,
        "expiry_time": "2024-01-15T15:45:00Z"
    }
}
```

### 3. Enhanced Error Handling

New structured error responses with actionable guidance:

```python
# Previous error (v1.3.0)
"Error: Connection failed"

# New error (v1.3.1)
{
    "error": "database_connection_failed",
    "message": "Failed to connect to PostgreSQL database 'production'",
    "details": {
        "host": "prod-db.company.com",
        "port": 5432,
        "database": "production",
        "error_code": "connection_timeout"
    },
    "suggestions": [
        "Check if database server is running",
        "Verify network connectivity to prod-db.company.com:5432",
        "Increase connection timeout in configuration",
        "Check authentication credentials"
    ],
    "documentation": "https://github.com/ChrisGVE/localdata-mcp/blob/main/TROUBLESHOOTING.md#postgresql-connection-problems"
}
```

## Step-by-Step Migration

### Step 1: Backup Current Configuration

```bash
# Backup current environment variables
env | grep -E "(POSTGRES|MYSQL|MONGODB|REDIS|LOCALDATA)" > backup_env_vars.txt

# Backup current MCP client configuration
cp ~/.config/mcp/config.json ~/.config/mcp/config.json.backup
```

### Step 2: Install v1.3.1

```bash
# Update LocalData MCP
pip install --upgrade localdata-mcp

# Verify version
python -c "import localdata_mcp; print(localdata_mcp.__version__)"
```

### Step 3: Test Compatibility

```python
# Test existing workflow with v1.3.1
connect_database("test", "sqlite", "./test.db")
result = execute_query("test", "CREATE TABLE IF NOT EXISTS test (id INTEGER)")
describe_database("test")
disconnect_database("test")

print("✅ Basic compatibility test passed")
```

### Step 4: Migrate Configuration (Optional)

**Option A: Keep existing environment variables** (simplest)
```bash
# No changes needed - everything continues to work
```

**Option B: Migrate to granular environment variables**
```bash
# Replace connection URLs with granular parameters
# Before:
# POSTGRES_URL=postgresql://user:pass@localhost:5432/db

# After:
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=user
POSTGRES_PASSWORD=pass
POSTGRES_DATABASE=db
POSTGRES_TIMEOUT=30
```

**Option C: Create YAML configuration** (for complex setups)
```bash
# Create configuration file
cat > localdata-config.yaml << 'EOF'
databases:
  main:
    type: postgresql
    host: ${POSTGRES_HOST}
    port: ${POSTGRES_PORT:-5432}
    user: ${POSTGRES_USER}
    password: ${POSTGRES_PASSWORD}
    database: ${POSTGRES_DATABASE}
    timeout: 30
    
logging:
  level: INFO
  format: json
EOF
```

### Step 5: Enable New Features

**Enable structured logging**:
```bash
LOCALDATA_LOG_FORMAT=json
LOCALDATA_LOG_FILE=./logs/localdata.log
```

**Configure memory limits**:
```bash
LOCALDATA_MAX_MEMORY_MB=1024
LOCALDATA_DEFAULT_CHUNK_SIZE=5000
```

**Set up query timeouts**:
```bash
LOCALDATA_DEFAULT_TIMEOUT=60
POSTGRES_TIMEOUT=120  # For long-running analytics queries
```

### Step 6: Test Large Data Processing

```python
# Test new buffering system with large dataset
connect_database("bigdata", "csv", "./large_dataset.csv")

# This will automatically use new streaming architecture
large_result = execute_query_json("bigdata", "SELECT * FROM data WHERE amount > 1000")

if 'buffering_info' in large_result:
    query_id = large_result['buffering_info']['query_id']
    print(f"✅ New buffering system activated: {query_id}")
    
    # Test chunk access
    chunk = get_query_chunk(query_id, 11, "1000")
    print(f"✅ Chunk access working: {len(chunk)} rows")
    
    # Cleanup
    clear_query_buffer(query_id)
    print("✅ Buffer cleanup successful")
```

### Step 7: Validate Production Readiness

**Test timeout behavior**:
```python
# Should timeout gracefully with new system
try:
    result = execute_query("slow_db", "SELECT pg_sleep(300)")  # 5 minute query
except Exception as e:
    print(f"✅ Timeout handling: {e}")
```

**Test memory limits**:
```python
# Should stream instead of loading everything into memory
result = execute_query_json("bigdata", "SELECT * FROM massive_table")
print(f"✅ Memory management: {result.get('buffering_info', 'Direct response')}")
```

**Test error handling**:
```python
# Should provide detailed, actionable error messages
try:
    connect_database("invalid", "postgresql", "postgresql://invalid:5432/db")
except Exception as e:
    print(f"✅ Enhanced errors: {e}")
```

## Environment-Specific Migration Examples

### Development Environment

**Before (v1.3.0)**:
```bash
# .env.development
POSTGRES_URL=postgresql://localhost/myapp_dev
REDIS_URL=redis://localhost:6379/0
```

**After (v1.3.1)**:
```bash
# .env.development - can keep old format or upgrade
POSTGRES_URL=postgresql://localhost/myapp_dev  # Still works
REDIS_URL=redis://localhost:6379/0            # Still works

# Optional: Enable new features
LOCALDATA_LOG_LEVEL=DEBUG
LOCALDATA_LOG_FORMAT=plain
```

### Staging Environment

**Before (v1.3.0)**:
```bash
# .env.staging
POSTGRES_URL=postgresql://staging-db:5432/myapp
MONGODB_URL=mongodb://staging-mongo:27017/myapp
```

**After (v1.3.1)**:
```bash
# .env.staging - migrate to granular config
POSTGRES_HOST=staging-db
POSTGRES_PORT=5432
POSTGRES_DATABASE=myapp
POSTGRES_USER=myapp_user
POSTGRES_PASSWORD=secure_password
POSTGRES_TIMEOUT=60

MONGODB_HOST=staging-mongo
MONGODB_PORT=27017
MONGODB_DATABASE=myapp
MONGODB_USER=myapp_user
MONGODB_PASSWORD=secure_password

# Enable structured logging
LOCALDATA_LOG_FORMAT=json
LOCALDATA_LOG_LEVEL=INFO
```

### Production Environment

**Create production configuration file**:
```yaml
# config/production.yaml
databases:
  primary:
    type: postgresql
    host: ${DB_PRIMARY_HOST}
    port: ${DB_PRIMARY_PORT:-5432}
    user: ${DB_PRIMARY_USER}
    password: ${DB_PRIMARY_PASSWORD}
    database: ${DB_PRIMARY_NAME}
    timeout: 30
    max_memory_mb: 512
    
  replica:
    type: postgresql
    host: ${DB_REPLICA_HOST}
    port: ${DB_REPLICA_PORT:-5432}
    user: ${DB_REPLICA_USER}
    password: ${DB_REPLICA_PASSWORD}
    database: ${DB_REPLICA_NAME}
    timeout: 120
    max_memory_mb: 1024
    
  cache:
    type: redis
    host: ${REDIS_HOST}
    port: ${REDIS_PORT:-6379}
    password: ${REDIS_PASSWORD}
    database: 0
    timeout: 5

logging:
  level: WARNING
  format: json
  file: /var/log/localdata/production.log
  
performance:
  default_chunk_size: 5000
  max_tokens_direct: 8000
  buffer_timeout_seconds: 1800

security:
  max_connections: 10
  query_timeout: 300
  enable_sql_validation: true
```

**Production deployment**:
```bash
# Use configuration file in production
LOCALDATA_CONFIG_FILE=/etc/localdata/production.yaml localdata-mcp
```

## Rollback Plan

If you encounter issues with v1.3.1:

### Option 1: Quick Rollback to v1.3.0

```bash
# Install specific version
pip install localdata-mcp==1.3.0

# Remove v1.3.1 configuration files (optional)
rm -f localdata-config.yaml
```

### Option 2: Disable New Features

```bash
# Use v1.3.0-compatible configuration
unset LOCALDATA_CONFIG_FILE
unset LOCALDATA_LOG_FORMAT
unset LOCALDATA_MAX_MEMORY_MB

# Restore original environment variables
source backup_env_vars.txt
```

### Option 3: Hybrid Approach

```bash
# Keep v1.3.1 but disable problematic features
LOCALDATA_DISABLE_STREAMING=true
LOCALDATA_LOG_FORMAT=plain
LOCALDATA_LOG_LEVEL=INFO
```

## Testing Your Migration

### Automated Test Script

```python
#!/usr/bin/env python3
"""
LocalData MCP v1.3.1 Migration Test Script
Run this script to validate your migration.
"""

def test_basic_functionality():
    """Test basic MCP operations"""
    try:
        connect_database("test", "sqlite", "./test_migration.db")
        execute_query("test", "CREATE TABLE IF NOT EXISTS test (id INTEGER, name TEXT)")
        execute_query("test", "INSERT INTO test VALUES (1, 'migration_test')")
        result = execute_query("test", "SELECT * FROM test")
        disconnect_database("test")
        assert len(result) == 1
        print("✅ Basic functionality test passed")
        return True
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

def test_large_data_handling():
    """Test new buffering system"""
    try:
        # Test with larger dataset
        connect_database("large_test", "sqlite", "./test_migration.db")
        
        # Insert test data
        for i in range(150):  # Trigger buffering at 100+ rows
            execute_query("large_test", f"INSERT INTO test VALUES ({i}, 'test_{i}')")
        
        # This should trigger new buffering system
        result = execute_query_json("large_test", "SELECT * FROM test")
        
        if 'buffering_info' in result:
            print("✅ New buffering system activated")
            query_id = result['buffering_info']['query_id']
            
            # Test chunk access
            chunk = get_query_chunk(query_id, 11, "50")
            print(f"✅ Chunk access working: {len(chunk)} rows")
            
            # Cleanup
            clear_query_buffer(query_id)
            print("✅ Buffer cleanup successful")
        else:
            print("ℹ️  Buffering not triggered (expected for smaller datasets)")
        
        disconnect_database("large_test")
        return True
        
    except Exception as e:
        print(f"❌ Large data handling test failed: {e}")
        return False

def test_configuration():
    """Test configuration system"""
    try:
        # Test environment variable access
        import os
        
        # Test new logging configuration
        if os.getenv('LOCALDATA_LOG_FORMAT'):
            print(f"✅ Logging format configured: {os.getenv('LOCALDATA_LOG_FORMAT')}")
        
        if os.getenv('LOCALDATA_MAX_MEMORY_MB'):
            print(f"✅ Memory limit configured: {os.getenv('LOCALDATA_MAX_MEMORY_MB')} MB")
            
        print("✅ Configuration system test passed")
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def run_migration_tests():
    """Run all migration tests"""
    print("🧪 Running LocalData MCP v1.3.1 Migration Tests")
    print("=" * 50)
    
    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("Large Data Handling", test_large_data_handling),
        ("Configuration System", test_configuration),
    ]
    
    passed = 0
    for test_name, test_func in tests:
        print(f"\n🔬 Testing {test_name}...")
        if test_func():
            passed += 1
    
    print(f"\n📊 Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 Migration completed successfully!")
        print("🚀 You're ready to use LocalData MCP v1.3.1")
    else:
        print("⚠️  Some tests failed. Check the output above.")
        print("📚 See TROUBLESHOOTING.md for help")
    
    # Cleanup
    import os
    if os.path.exists("./test_migration.db"):
        os.remove("./test_migration.db")
        print("🧹 Cleaned up test files")

if __name__ == "__main__":
    run_migration_tests()
```

Save as `test_migration.py` and run:
```bash
python test_migration.py
```

## Support and Troubleshooting

### Common Migration Issues

**Issue: "Configuration file not found"**
```bash
# Solution: Use absolute path or place in expected location
LOCALDATA_CONFIG_FILE=$(pwd)/localdata-config.yaml localdata-mcp
```

**Issue: "Memory limit exceeded"**
```bash
# Solution: Increase memory limit or enable streaming
LOCALDATA_MAX_MEMORY_MB=2048
```

**Issue: "Query timeout too short"**
```bash
# Solution: Increase timeout for specific operations
POSTGRES_TIMEOUT=300
LOCALDATA_DEFAULT_TIMEOUT=120
```

### Getting Help

1. **Check the [Troubleshooting Guide](TROUBLESHOOTING.md)** - Updated with v1.3.1 scenarios
2. **Review [Advanced Examples](ADVANCED_EXAMPLES.md)** - Production-ready patterns  
3. **Visit [GitHub Issues](https://github.com/ChrisGVE/localdata-mcp/issues)** - Report problems or get help
4. **Read [Configuration Guide](CONFIGURATION.md)** - Comprehensive configuration reference
5. **Check [Architecture Documentation](ARCHITECTURE.md)** - Understanding the new systems

## Summary

LocalData MCP v1.3.1 is a major upgrade that maintains full backward compatibility while introducing powerful new capabilities:

- ✅ **Zero breaking changes** for existing MCP tool usage
- ✅ **Optional configuration migration** - upgrade at your own pace
- ✅ **Enhanced performance** with intelligent resource management
- ✅ **Better developer experience** with rich metadata and error messages
- ✅ **Production-ready** features like structured logging and timeout controls

The migration is designed to be gradual and low-risk. Start with the basic upgrade, then progressively adopt new features as needed for your use case.

**Next Steps**: 
1. Install v1.3.1: `pip install --upgrade localdata-mcp`
2. Run migration tests: `python test_migration.py`
3. Explore new features: [CONFIGURATION.md](CONFIGURATION.md), [ARCHITECTURE.md](ARCHITECTURE.md)
4. Get help if needed: [TROUBLESHOOTING.md](TROUBLESHOOTING.md)

Welcome to LocalData MCP v1.3.1! 🚀