# LocalData MCP v1.3.1 - Backward Compatibility Guide

## Overview

LocalData MCP v1.3.1 maintains full backward compatibility with v1.3.0 while introducing new features and improvements. This guide explains how existing configurations and API usage continue to work, what deprecation warnings you might see, and how to migrate to new patterns.

## 🔄 What's Maintained

### API Compatibility
- **execute_query function**: All existing calls continue working exactly as before
- **Response formats**: JSON responses maintain the same structure for existing integrations
- **Error handling**: Error messages and formats remain consistent

### Configuration Compatibility
- **Environment variables**: All legacy environment variables (POSTGRES_HOST, MYSQL_HOST, etc.) continue working
- **Connection strings**: Existing database connection patterns work unchanged
- **File paths**: SQLite and DuckDB path configurations work as before

## 🆕 New Features (Optional)

### Enhanced execute_query Parameters
```python
# Old usage (still works):
execute_query(database_name, sql_query)

# New usage (recommended):
execute_query(database_name, sql_query, chunk_size=1000, enable_analysis=True)
```

### YAML Configuration Support
```yaml
# New localdata.yaml configuration (recommended):
databases:
  my_postgres:
    type: postgresql
    host: localhost
    port: 5432
    database: mydb
    user: myuser
    password: mypass

logging:
  level: INFO
  
performance:
  memory_limit_mb: 512
```

## 🔍 Checking Your Compatibility Status

Use the new `check_compatibility` MCP tool to assess your current setup:

```json
{
  "name": "check_compatibility",
  "arguments": {
    "generate_migration_script": true
  }
}
```

This will return:
- Current compatibility status
- Detected legacy patterns
- Migration recommendations
- Optional migration script

## ⚠️ Deprecation Warnings

You may see these informational warnings in logs:

### 1. Legacy Environment Variables
```
DEPRECATED: Single Database Environment Variables (POSTGRES_HOST, etc.)
Replacement: Multi-database YAML configuration
Migration Guide: Use databases section in localdata.yaml with named database configurations.
```

**Action**: Consider migrating to YAML configuration for better organization.

### 2. Environment Variable Only Configuration  
```
DEPRECATED: Environment Variable Only Configuration
Replacement: YAML Configuration Files
Migration Guide: Create localdata.yaml config file. See documentation for examples.
```

**Action**: Create a `localdata.yaml` file alongside your environment variables.

### 3. Execute Query Without Analysis
```
DEPRECATED: execute_query without enable_analysis parameter
Replacement: execute_query with enable_analysis=True
Migration Guide: Add enable_analysis=True to execute_query calls for optimal performance.
```

**Action**: Add `enable_analysis=True` to your execute_query calls.

## 📋 Migration Scenarios

### Scenario 1: Basic PostgreSQL Setup
**Current setup** (environment variables):
```bash
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=myuser
POSTGRES_PASSWORD=mypass
POSTGRES_DB=mydb
```

**Migration to YAML**:
```yaml
databases:
  primary:
    type: postgresql
    host: localhost
    port: 5432
    database: mydb
    user: myuser
    password: mypass
```

### Scenario 2: Multi-Database Setup
**Current setup**:
```bash
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=pguser
POSTGRES_DB=pgdb

MYSQL_HOST=mysql-server
MYSQL_PORT=3306
MYSQL_USER=mysqluser
MYSQL_DB=mysqldb

SQLITE_PATH=/data/analytics.db
```

**Migration to YAML**:
```yaml
databases:
  warehouse:
    type: postgresql
    host: localhost
    port: 5432
    database: pgdb
    user: pguser
    password: ${POSTGRES_PASSWORD}  # Environment variable substitution
    
  application:
    type: mysql
    host: mysql-server
    port: 3306
    database: mysqldb
    user: mysqluser
    password: ${MYSQL_PASSWORD}
    
  analytics:
    type: sqlite
    path: /data/analytics.db
```

## 🛠️ Migration Tools

### Automatic Migration Script
The compatibility checker can generate a migration script:

```python
#!/usr/bin/env python3
# Generated migration script
import os
import yaml
from pathlib import Path

def migrate_configuration():
    # Automatically detects your environment variables
    # and creates equivalent YAML configuration
    config = {...}  # Generated based on your setup
    
    with open('./localdata.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    print("Configuration migrated to localdata.yaml")

if __name__ == '__main__':
    migrate_configuration()
```

### Manual Migration Steps
1. **Run compatibility check**: Use `check_compatibility` tool
2. **Review recommendations**: Check what legacy patterns were detected
3. **Create YAML config**: Start with generated script or create manually
4. **Test alongside existing**: Both environment variables and YAML can coexist
5. **Gradually migrate**: Move databases one at a time
6. **Remove environment variables**: After confirming YAML works

## 🚫 What's NOT Breaking

- **No API signature changes**: All function calls work exactly as before
- **No response format changes**: JSON structures remain the same
- **No configuration requirements**: Environment variables continue working
- **No immediate action needed**: All deprecation warnings are informational

## ⏰ Timeline

- **v1.3.1 (Current)**: Legacy patterns deprecated but fully supported
- **v1.4.0 (Future)**: Legacy patterns will be removed
- **Recommended timeline**: Migrate within 6 months for best experience

## 🔧 Troubleshooting

### I see deprecation warnings but everything still works
✅ **Normal**: Warnings are informational only. Your setup continues working.

### My execute_query calls seem slower
💡 **Solution**: Add `enable_analysis=True` parameter for performance optimization.

### I want to try YAML but keep environment variables as backup
✅ **Supported**: Both can coexist. YAML takes precedence when both exist.

### The migration script doesn't work perfectly
💡 **Expected**: Generated scripts are starting points. You may need to customize them for complex setups.

### I have a custom database configuration
📖 **Guidance**: Legacy patterns work unchanged. Consider YAML for better organization of complex configs.

## 📞 Support

If you encounter any breaking changes or unexpected behavior:

1. **Check compatibility status**: Run `check_compatibility` tool
2. **Review logs**: Look for specific error messages beyond deprecation warnings  
3. **Test incrementally**: Try new features alongside existing setup
4. **Report issues**: If something truly breaks, this indicates a bug in our compatibility layer

## 🎯 Best Practices

### Immediate (No Rush)
- ✅ Continue using current setup
- ✅ Review deprecation warnings when convenient
- ✅ Test `enable_analysis=True` parameter

### Short Term (1-3 months)
- 🔄 Create `localdata.yaml` configuration file
- 🔄 Test new configuration alongside existing environment variables
- 🔄 Migrate one database at a time

### Long Term (3-6 months)  
- 🎯 Remove environment variables in favor of YAML
- 🎯 Use named database configurations for better organization
- 🎯 Take advantage of advanced YAML features (environment variable substitution, etc.)

---

**Remember**: v1.3.1 maintains 100% backward compatibility. All deprecation warnings are guidance for future improvements, not immediate requirements.