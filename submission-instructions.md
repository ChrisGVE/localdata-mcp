# Docker MCP Registry Submission Instructions

## Submission Process

### Step 1: Fork and Clone the Registry
```bash
# 1. Go to https://github.com/docker/mcp-registry and click Fork
# 2. Clone your fork
git clone https://github.com/YOUR-USERNAME/mcp-registry.git
cd mcp-registry
```

### Step 2: Use the Interactive Wizard
```bash
# Run the interactive wizard to generate configuration
task wizard

# When prompted, provide:
# - Repository URL: https://github.com/ChrisGVE/localdata-mcp
# - Category: Database
# - Name: localdata
# - Title: LocalData MCP Server
# - Description: A comprehensive MCP server for databases, spreadsheets, and structured data files
```

### Step 3: Alternative - Manual Creation
If wizard doesn't work, manually create `servers/localdata.yaml`:

```yaml
name: localdata
type: server
meta:
  category: Database
  tags:
    - database
    - postgresql
    - mysql
    - sqlite
    - mongodb
    - redis
    - elasticsearch
    - spreadsheet
    - excel
    - csv
    - json
    - yaml
    - xml
    - data-analysis
    - file-processing
about:
  title: LocalData MCP Server
  description: A comprehensive MCP server for databases, spreadsheets, and structured data files with security features, performance optimization, and extensive format support.
  icon: https://raw.githubusercontent.com/ChrisGVE/localdata-mcp/main/assets/logo.png
source:
  project: https://github.com/ChrisGVE/localdata-mcp
config:
  description: |
    LocalData MCP Server provides comprehensive data connectivity with optional configuration.
    
    ## Environment Variables
    
    ### Core Application (Required)
    - PYTHONUNBUFFERED: Enable real-time logging output (default: 1)
    
    ### Python Runtime (Optional)
    - PYTHONDONTWRITEBYTECODE: Prevent .pyc file generation (default: 1)
    - PIP_NO_CACHE_DIR: Disable pip caching (default: 1)
    - PIP_DISABLE_PIP_VERSION_CHECK: Skip pip version checks (default: 1)
    
    ### Database Connections (Optional - for pre-configured connections)
    - POSTGRES_URL: PostgreSQL connection string
    - MYSQL_URL: MySQL connection string  
    - MONGODB_URL: MongoDB connection string
    - REDIS_URL: Redis connection string
    - ELASTICSEARCH_URL: Elasticsearch URL
    - INFLUXDB_URL: InfluxDB connection URL
    - NEO4J_URI: Neo4j connection URI
    - COUCHDB_URL: CouchDB connection URL
    
    ### Performance & Resource Management
    - LOCALDATA_MAX_MEMORY: Maximum memory usage (default: 1G)
    - LOCALDATA_CHUNK_SIZE: Rows per chunk for large datasets (default: 10000)
    - LOCALDATA_BUFFER_TIMEOUT: Buffer expiry in seconds (default: 600)
    - LOCALDATA_MAX_CONNECTIONS: Maximum database connections (default: 10)
    - LOCALDATA_MAX_FILE_SIZE: Maximum file size in bytes (default: 104857600)
    
    ### Security
    - LOCALDATA_ALLOWED_PATHS: Comma-separated allowed file paths (default: /app/data)
    
    ### Logging
    - LOG_LEVEL: Logging level - DEBUG, INFO, WARNING, ERROR (default: INFO)
    - LOG_FORMAT: Log format - json, text (default: json)
    
    ## Volume Mappings
    
    ### Required Volumes
    ```bash
    # Data directory (required)
    -v /host/data:/app/data
    
    # Log directory (recommended)
    -v /host/logs:/app/logs
    
    # Temporary files (recommended for large files)
    -v /host/temp:/app/temp
    ```
    
    ### Optional Volumes  
    ```bash
    # Configuration files (read-only)
    -v /host/config:/app/config:ro
    
    # Examples directory
    -v /host/examples:/app/examples:ro
    ```
    
    ## Features
    - **Multi-Database Support**: PostgreSQL, MySQL, SQLite, DuckDB, MongoDB, Redis, Elasticsearch, InfluxDB, Neo4j, CouchDB
    - **Spreadsheet Formats**: Excel (.xlsx/.xls), LibreOffice Calc (.ods), Apple Numbers with multi-sheet support
    - **Structured Files**: CSV, TSV, JSON, YAML, TOML, XML, INI with automatic conversion
    - **Analytical Formats**: Parquet, Feather, Arrow, HDF5 for data science workflows
    - **Security**: Path restrictions, SQL injection prevention, connection limits
    - **Performance**: Large file buffering (100MB+), query result buffering (100+ rows), chunk retrieval
    
    ## Usage Notes
    - Database connections are established dynamically using connect_database tool
    - Environment variables provide optional pre-configuration
    - All file paths restricted to mounted data directory for security
    - Large files (100MB+) automatically use temporary SQLite storage
    - Query results with 100+ rows automatically use buffering system
```

### Step 4: Test and Build
```bash
# Test the configuration
task build

# Generate catalog to verify
task catalog
```

### Step 5: Submit Pull Request
```bash
# Commit your changes
git add servers/localdata.yaml
git commit -m "Add LocalData MCP Server

A comprehensive MCP server supporting:
- SQL & NoSQL databases (PostgreSQL, MySQL, SQLite, MongoDB, Redis, etc.)
- Spreadsheets (Excel, LibreOffice Calc, Apple Numbers) 
- Structured files (CSV, JSON, YAML, XML, etc.)
- Analytical formats (Parquet, Feather, Arrow, HDF5)
- Advanced security and performance features"

# Push to your fork
git push origin main

# Create pull request at https://github.com/docker/mcp-registry
```

## Enhanced Docker Usage Examples

### Basic Usage with Logging
```bash
# Create directories
mkdir -p data logs temp

# Run with proper volume mappings
docker run -it --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/temp:/app/temp \
  -e PYTHONUNBUFFERED=1 \
  -e LOG_LEVEL=INFO \
  -e LOG_FORMAT=json \
  localdata-mcp
```

### Production Docker Compose
```yaml
version: '3.8'

services:
  localdata-mcp:
    image: localdata-mcp
    restart: unless-stopped
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
      - ./temp:/app/temp
    environment:
      - PYTHONUNBUFFERED=1
      - LOG_LEVEL=INFO
      - LOG_FORMAT=json
      - LOCALDATA_MAX_MEMORY=2G
      - LOCALDATA_CHUNK_SIZE=10000
      - LOCALDATA_BUFFER_TIMEOUT=600
      - LOCALDATA_MAX_CONNECTIONS=10
      - LOCALDATA_MAX_FILE_SIZE=104857600
      - LOCALDATA_ALLOWED_PATHS=/app/data
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '1.0'
```

### MCP Client Configuration with Logging
```json
{
  "mcpServers": {
    "localdata": {
      "command": "docker",
      "args": [
        "run", "-i", "--rm",
        "-v", "/host/data:/app/data",
        "-v", "/host/logs:/app/logs", 
        "-v", "/host/temp:/app/temp",
        "-e", "PYTHONUNBUFFERED=1",
        "-e", "LOG_LEVEL=INFO",
        "-e", "LOG_FORMAT=json",
        "localdata-mcp"
      ]
    }
  }
}
```

## Recommendation: Docker-Built Image

Choose **Docker-Built** option for:
- Cryptographic signatures
- Provenance tracking
- Software Bills of Materials (SBOMs)
- Automatic security updates

## Review Timeline
- Docker team reviews submissions promptly
- Approved servers available within 24 hours
- Available on Docker Desktop's MCP Toolkit and Docker Hub

## Support
- GitHub Issues: https://github.com/docker/mcp-registry/issues
- Documentation: https://docs.docker.com/ai/mcp-catalog-and-toolkit/