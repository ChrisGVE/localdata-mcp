# Docker Usage Guide - LocalData MCP Server

## 🐳 Quick Start

### Option 1: Direct Docker Run

```bash
# Build the image
docker build -t localdata-mcp .

# Run the container with your data directory mounted
docker run -it --rm \
  -v $(pwd)/data:/app/data \
  localdata-mcp
```

### Option 2: Docker Compose (Recommended)

```bash
# Start the full stack with database services
docker-compose up -d

# View logs
docker-compose logs -f localdata-mcp

# Stop all services
docker-compose down
```

## 📁 Directory Structure

```
your-project/
├── data/                    # Your data files (mounted to container)
│   ├── *.csv
│   ├── *.xlsx
│   ├── *.json
│   └── ...
├── examples/               # Example files (optional)
├── docker-compose.yml     # Multi-service setup
└── Dockerfile            # Container definition
```

## 🔧 Configuration Options

### Environment Variables

```bash
# Set in docker-compose.yml or pass to docker run
PYTHONUNBUFFERED=1    # Better logging output
```

### Volume Mounting

```bash
# Mount your data directory
-v /path/to/your/data:/app/data

# Mount additional directories
-v /path/to/config:/app/config:ro  # Read-only config
```

## 🗄️ Using with Databases

### Connecting to Container Databases

When using docker-compose, databases are available at:

```python
# PostgreSQL
connect_database("postgres", "postgresql", "postgresql://testuser:testpass@postgres:5432/testdb")

# MySQL
connect_database("mysql", "mysql", "mysql+pymysql://testuser:testpass@mysql:3306/testdb")

# MongoDB
connect_database("mongo", "mongodb", "mongodb://testuser:testpass@mongodb:27017/testdb")

# Redis
connect_database("redis", "redis", "redis://:testpass@redis:6379/0")

# Elasticsearch
connect_database("elastic", "elasticsearch", "http://elasticsearch:9200")
```

### Connecting to External Databases

```bash
# Add network access to external databases
docker run -it --rm \
  --network host \
  -v $(pwd)/data:/app/data \
  localdata-mcp
```

## 📊 Working with Files

### Supported File Operations

```bash
# Place your files in the data directory
data/
├── sales.csv           # Available as sales.csv
├── config.json         # Available as config.json
├── sheets/
│   └── quarterly.xlsx  # Available as sheets/quarterly.xlsx
└── exports/
    └── data.parquet   # Available as exports/data.parquet
```

### Example MCP Client Configuration

```json
{
  "mcpServers": {
    "localdata": {
      "command": "docker",
      "args": [
        "run", 
        "-i", 
        "--rm",
        "-v", "/path/to/your/data:/app/data",
        "localdata-mcp"
      ]
    }
  }
}
```

## 🚀 Advanced Usage

### Custom Database Configuration

Create a custom docker-compose.override.yml:

```yaml
version: '3.8'

services:
  localdata-mcp:
    environment:
      - CUSTOM_DB_URL=postgresql://user:pass@external-db:5432/mydb
    
  # Add more databases
  influxdb:
    image: influxdb:2.7
    environment:
      - INFLUXDB_DB=mydb
      - INFLUXDB_ADMIN_USER=admin
      - INFLUXDB_ADMIN_PASSWORD=password
    ports:
      - "8086:8086"
```

### Production Deployment

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  localdata-mcp:
    restart: always
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
    resources:
      limits:
        memory: 1G
        cpus: '0.5'
```

### Health Monitoring

```bash
# Check container health
docker-compose ps

# View detailed logs
docker-compose logs -f localdata-mcp

# Execute commands in running container
docker-compose exec localdata-mcp bash
```

## 🔒 Security Considerations

### File Access Security

- The container runs as a non-root user (mcpuser)
- File access is restricted to the mounted `/app/data` directory
- Use read-only mounts for configuration files

### Network Security

```yaml
# Isolate services in custom network
networks:
  localdata-net:
    driver: bridge
    internal: true  # No internet access
```

### Database Security

```yaml
# Use environment files for credentials
env_file:
  - .env.production

# Example .env.production
POSTGRES_PASSWORD=secure_random_password
MYSQL_ROOT_PASSWORD=another_secure_password
```

## 📋 Troubleshooting

### Common Issues

1. **Permission Denied**
   ```bash
   # Fix file permissions
   sudo chown -R 1000:1000 ./data
   ```

2. **Database Connection Failures**
   ```bash
   # Wait for databases to start
   docker-compose up -d postgres mysql
   sleep 10
   docker-compose up localdata-mcp
   ```

3. **Port Conflicts**
   ```yaml
   # Change port mapping in docker-compose.yml
   ports:
     - "5433:5432"  # Use different host port
   ```

4. **Memory Issues with Large Files**
   ```yaml
   # Increase container memory limit
   deploy:
     resources:
       limits:
         memory: 2G
   ```

### Debug Mode

```bash
# Run with debug output
docker-compose run --rm localdata-mcp python -c "
import logging
logging.basicConfig(level=logging.DEBUG)
from localdata_mcp.localdata_mcp import main
main()
"
```

## 🧪 Development and Testing

### Development Setup

```bash
# Mount source code for development
docker run -it --rm \
  -v $(pwd):/app \
  -v $(pwd)/data:/app/data \
  python:3.11-slim-bullseye \
  bash

# Inside container
cd /app
pip install -e .
localdata-mcp
```

### Testing with Sample Data

```bash
# Create sample data
mkdir -p data
echo "id,name,value" > data/sample.csv
echo "1,test,100" >> data/sample.csv

# Start container
docker-compose up -d localdata-mcp

# Test connection
# (Use your MCP client to connect and test)
```

## 📖 Integration Examples

### With Claude Desktop

```json
// claude_desktop_config.json
{
  "mcpServers": {
    "localdata": {
      "command": "docker",
      "args": [
        "run", "-i", "--rm",
        "-v", "/Users/yourname/data:/app/data",
        "localdata-mcp"
      ]
    }
  }
}
```

### With VS Code

```json
// .vscode/settings.json
{
  "mcp.servers": {
    "localdata": {
      "command": ["docker-compose", "run", "--rm", "localdata-mcp"]
    }
  }
}
```

This containerized setup provides a complete, isolated environment for running LocalData MCP Server with optional database services, making it easy to deploy and manage in any Docker-compatible environment.