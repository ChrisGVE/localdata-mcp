# Database Connections Guide - LocalData MCP Server

This guide provides comprehensive information for connecting to all database types supported by LocalData MCP Server.

## Table of Contents

- [Environment Variables Overview](#environment-variables-overview)
- [SQL Databases](#sql-databases)
- [Modern Databases](#modern-databases)
- [Security Best Practices](#security-best-practices)
- [Connection Examples](#connection-examples)
- [Troubleshooting](#troubleshooting)

## Environment Variables Overview

LocalData MCP Server supports optional pre-configuration through environment variables. These are completely optional - databases can be connected dynamically using the `connect_database` tool.

### Core Application

```bash
# Required for proper logging
PYTHONUNBUFFERED=1

# Optional Python runtime optimizations
PYTHONDONTWRITEBYTECODE=1      # Prevent .pyc file generation
PIP_NO_CACHE_DIR=1             # Disable pip caching
PIP_DISABLE_PIP_VERSION_CHECK=1 # Skip pip version checks
```

### Performance & Resource Management

```bash
LOCALDATA_MAX_MEMORY=1G           # Maximum memory usage (default: 1G)
LOCALDATA_CHUNK_SIZE=10000        # Rows per chunk for large datasets (default: 10000)
LOCALDATA_BUFFER_TIMEOUT=600      # Buffer expiry in seconds (default: 600)
LOCALDATA_MAX_CONNECTIONS=10      # Maximum database connections (default: 10)
LOCALDATA_MAX_FILE_SIZE=104857600 # Maximum file size in bytes (default: 100MB)
```

### Security

```bash
LOCALDATA_ALLOWED_PATHS=/app/data # Comma-separated allowed file paths (default: /app/data)
```

### Logging

```bash
LOG_LEVEL=INFO      # Logging level: DEBUG, INFO, WARNING, ERROR (default: INFO)
LOG_FORMAT=json     # Log format: json, text (default: json)
```

## SQL Databases

### PostgreSQL

**Connection String Format:**
```
postgresql://[username]:[password]@[host]:[port]/[database]
```

**Environment Variable (Optional):**
```bash
POSTGRES_URL=postgresql://user:password@localhost:5432/mydb
```

**Connection Examples:**
```python
# Local PostgreSQL
connect_database("postgres_local", "postgresql", "postgresql://user:pass@localhost:5432/mydb")

# Remote PostgreSQL with SSL
connect_database("postgres_prod", "postgresql", "postgresql://user:pass@prod-server:5432/mydb?sslmode=require")

# Docker container
connect_database("postgres_docker", "postgresql", "postgresql://testuser:testpass@postgres:5432/testdb")
```

**Required Dependencies:**
- `psycopg2-binary` or `psycopg2`

### MySQL/MariaDB

**Connection String Format:**
```
mysql+pymysql://[username]:[password]@[host]:[port]/[database]
```

**Environment Variable (Optional):**
```bash
MYSQL_URL=mysql+pymysql://user:password@localhost:3306/mydb
```

**Connection Examples:**
```python
# Local MySQL
connect_database("mysql_local", "mysql", "mysql+pymysql://user:pass@localhost:3306/mydb")

# MariaDB with charset
connect_database("mariadb", "mysql", "mysql+pymysql://user:pass@localhost:3306/mydb?charset=utf8mb4")

# Docker container
connect_database("mysql_docker", "mysql", "mysql+pymysql://testuser:testpass@mysql:3306/testdb")
```

**Required Dependencies:**
- `PyMySQL` or `mysqlclient`

### SQLite

**Connection String Format:**
```
./path/to/database.db
```

**Connection Examples:**
```python
# Local SQLite file
connect_database("local_db", "sqlite", "./data/mydb.sqlite")

# In-memory database
connect_database("memory_db", "sqlite", ":memory:")

# Relative path
connect_database("relative_db", "sqlite", "data/relative.db")
```

**Required Dependencies:**
- `sqlite3` (built into Python)

### DuckDB

**Connection String Format:**
```
./path/to/database.duckdb
```

**Connection Examples:**
```python
# Local DuckDB file
connect_database("analytics", "duckdb", "./data/analytics.duckdb")

# In-memory DuckDB
connect_database("temp_analytics", "duckdb", ":memory:")
```

**Required Dependencies:**
- `duckdb`

## Modern Databases

### MongoDB

**Connection String Format:**
```
mongodb://[username]:[password]@[host]:[port]/[database]
```

**Environment Variable (Optional):**
```bash
MONGODB_URL=mongodb://user:password@localhost:27017/mydb
```

**Connection Examples:**
```python
# Local MongoDB
connect_database("mongo_local", "mongodb", "mongodb://user:pass@localhost:27017/mydb")

# MongoDB Atlas
connect_database("mongo_atlas", "mongodb", "mongodb+srv://user:pass@cluster.mongodb.net/mydb")

# Docker container
connect_database("mongo_docker", "mongodb", "mongodb://testuser:testpass@mongodb:27017/testdb")
```

**Required Dependencies:**
- `pymongo`

### Redis

**Connection String Format:**
```
redis://:[password]@[host]:[port]/[db_number]
```

**Environment Variable (Optional):**
```bash
REDIS_URL=redis://:password@localhost:6379/0
```

**Connection Examples:**
```python
# Local Redis
connect_database("redis_local", "redis", "redis://:mypassword@localhost:6379/0")

# Redis without password
connect_database("redis_simple", "redis", "redis://localhost:6379/0")

# Docker container
connect_database("redis_docker", "redis", "redis://:testpass@redis:6379/0")
```

**Required Dependencies:**
- `redis`

### Elasticsearch

**Connection String Format:**
```
http://[host]:[port]
https://[username]:[password]@[host]:[port]
```

**Environment Variable (Optional):**
```bash
ELASTICSEARCH_URL=http://localhost:9200
```

**Connection Examples:**
```python
# Local Elasticsearch
connect_database("elastic_local", "elasticsearch", "http://localhost:9200")

# Elasticsearch with authentication
connect_database("elastic_auth", "elasticsearch", "https://user:pass@elastic.example.com:9200")

# Docker container
connect_database("elastic_docker", "elasticsearch", "http://elasticsearch:9200")
```

**Required Dependencies:**
- `elasticsearch`

### InfluxDB

**Connection String Format:**
```
http://[host]:[port]
```

**Environment Variable (Optional):**
```bash
INFLUXDB_URL=http://localhost:8086
INFLUXDB_TOKEN=your_token_here
INFLUXDB_ORG=your_org
INFLUXDB_BUCKET=your_bucket
```

**Connection Examples:**
```python
# InfluxDB 2.x
connect_database("influx", "influxdb", "http://localhost:8086")

# Docker container
connect_database("influx_docker", "influxdb", "http://influxdb:8086")
```

**Required Dependencies:**
- `influxdb-client`

### Neo4j

**Connection String Format:**
```
bolt://[host]:[port]
neo4j://[host]:[port]
```

**Environment Variable (Optional):**
```bash
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
```

**Connection Examples:**
```python
# Local Neo4j
connect_database("graph", "neo4j", "bolt://localhost:7687")

# Neo4j with authentication
connect_database("graph_auth", "neo4j", "bolt://neo4j:password@localhost:7687")

# Docker container
connect_database("graph_docker", "neo4j", "bolt://neo4j:7687")
```

**Required Dependencies:**
- `neo4j`

### CouchDB

**Connection String Format:**
```
http://[username]:[password]@[host]:[port]
```

**Environment Variable (Optional):**
```bash
COUCHDB_URL=http://admin:password@localhost:5984
```

**Connection Examples:**
```python
# Local CouchDB
connect_database("couch", "couchdb", "http://admin:password@localhost:5984")

# Docker container
connect_database("couch_docker", "couchdb", "http://admin:password@couchdb:5984")
```

**Required Dependencies:**
- `couchdb`

## Security Best Practices

### Credential Management

1. **Never hard-code credentials** in your connection strings
2. **Use environment variables** for sensitive information
3. **Use Docker secrets** for production deployments
4. **Rotate passwords regularly**

### Environment Files

Create a `.env` file for local development:

```bash
# .env
POSTGRES_URL=postgresql://user:secure_password@localhost:5432/mydb
MONGODB_URL=mongodb://user:secure_password@localhost:27017/mydb
REDIS_URL=redis://:secure_password@localhost:6379/0
```

Add to `.gitignore`:
```
.env
.env.local
.env.production
```

### Docker Secrets

For production Docker deployments:

```yaml
# docker-compose.yml
version: '3.8'

services:
  localdata-mcp:
    image: localdata-mcp
    secrets:
      - postgres_password
      - mongodb_password
    environment:
      - POSTGRES_URL=postgresql://user:$${POSTGRES_PASSWORD}@postgres:5432/mydb
      - MONGODB_URL=mongodb://user:$${MONGODB_PASSWORD}@mongodb:27017/mydb

secrets:
  postgres_password:
    external: true
  mongodb_password:
    external: true
```

### Network Security

1. **Use encrypted connections** (SSL/TLS) when possible
2. **Restrict network access** using Docker networks
3. **Use non-standard ports** for additional security
4. **Implement firewall rules** on the host

## Connection Examples

### Multi-Database Setup

```python
# Connect to multiple databases
connect_database("users_db", "postgresql", "postgresql://user:pass@localhost:5432/users")
connect_database("sessions", "redis", "redis://:pass@localhost:6379/0")
connect_database("analytics", "mongodb", "mongodb://user:pass@localhost:27017/analytics")
connect_database("search", "elasticsearch", "http://localhost:9200")

# Query across different databases
user_data = execute_query("users_db", "SELECT * FROM users WHERE active = true")
session_data = execute_query("sessions", "GET user:123:session")
```

### Development vs Production

```python
# Development (local)
if environment == "development":
    connect_database("main", "sqlite", "./dev.db")
    connect_database("cache", "redis", "redis://localhost:6379/0")

# Production (containerized)
else:
    connect_database("main", "postgresql", os.getenv("POSTGRES_URL"))
    connect_database("cache", "redis", os.getenv("REDIS_URL"))
```

## Troubleshooting

### Common Connection Issues

1. **Connection Refused**
   - Check if database service is running
   - Verify host and port are correct
   - Check network connectivity

2. **Authentication Failed**
   - Verify username and password
   - Check user permissions
   - Ensure user exists in database

3. **SSL/TLS Errors**
   - Check SSL certificate validity
   - Try without SSL first
   - Verify SSL mode requirements

4. **Network Timeouts**
   - Increase connection timeout
   - Check firewall settings
   - Verify network routing

### Debug Connection

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Test connection
try:
    result = connect_database("test", "postgresql", "postgresql://user:pass@localhost:5432/test")
    print(f"Connection successful: {result}")
except Exception as e:
    print(f"Connection failed: {e}")
```

### Container Networking

When using Docker, ensure containers are on the same network:

```yaml
version: '3.8'

services:
  localdata-mcp:
    networks:
      - app-network
  
  postgres:
    networks:
      - app-network

networks:
  app-network:
    driver: bridge
```

### Port Mapping Issues

Check if ports are properly exposed:

```bash
# Check if port is accessible
telnet localhost 5432

# Check Docker port mapping
docker port <container_name>
```

---

This guide covers all supported database types. For specific use cases or additional help, refer to the main documentation or create an issue in the repository.