# Docker Usage Guide

Run LocalData MCP in a container with optional database services for development and production use.

## Quick start

### Direct Docker run

```bash
docker build -t localdata-mcp .

docker run -it --rm \
  -v $(pwd)/data:/app/data \
  localdata-mcp
```

### Docker Compose (recommended)

The included `docker-compose.yml` starts LocalData MCP alongside PostgreSQL, MySQL, MongoDB, Redis, and Elasticsearch:

```bash
docker-compose up -d
docker-compose logs -f localdata-mcp

# Stop all services
docker-compose down
```

## MCP client configuration

### With a pre-built image

```json
{
  "mcpServers": {
    "localdata": {
      "command": "docker",
      "args": [
        "run", "-i", "--rm",
        "-v", "/path/to/your/data:/app/data",
        "localdata-mcp"
      ]
    }
  }
}
```

### With Docker Compose

```json
{
  "mcpServers": {
    "localdata": {
      "command": "docker-compose",
      "args": ["run", "--rm", "localdata-mcp"]
    }
  }
}
```

## Volumes

### Required

```bash
-v /path/to/your/data:/app/data     # Data files accessible to the server
```

### Optional

```bash
-v /path/to/logs:/app/logs           # Server logs
-v /path/to/config:/app/config:ro    # Configuration files (read-only)
```

## Connecting to databases

When using Docker Compose, database services are accessible by container hostname on the internal network:

```python
# PostgreSQL (testuser/testpass on port 5432)
connect_database("pg", "postgresql", "postgresql://testuser:testpass@postgres:5432/testdb")

# MySQL (testuser/testpass on port 3306)
connect_database("my", "mysql", "mysql://testuser:testpass@mysql:3306/testdb")

# MongoDB (testuser/testpass on port 27017)
connect_database("mongo", "mongodb", "mongodb://testuser:testpass@mongodb:27017/testdb?authSource=admin")

# Redis (password: testpass on port 6379)
connect_database("cache", "redis", "redis://:testpass@redis:6379/0")

# Elasticsearch (no auth, port 9200)
connect_database("search", "elasticsearch", "http://elasticsearch:9200")
```

For complete connection string formats and all supported database types, see the [data sources documentation](docs/data-sources/complete-reference.md).

### External databases

To connect to databases outside the Docker network:

```bash
docker run -it --rm \
  --network host \
  -v $(pwd)/data:/app/data \
  localdata-mcp
```

## Environment variables

Pass environment variables to configure the server. For the full reference, see the [configuration documentation](docs/configuration.md).

```yaml
# docker-compose.override.yml
services:
  localdata-mcp:
    environment:
      - LOCALDATA_LOG_LEVEL=debug
      - LOCALDATA_MEMORY_LIMIT_MB=2048
      - LOCALDATA_QUERY_CHUNK_SIZE=500
```

Or via `docker run`:

```bash
docker run -it --rm \
  -e LOCALDATA_LOG_LEVEL=debug \
  -v $(pwd)/data:/app/data \
  localdata-mcp
```

## Production deployment

```yaml
# docker-compose.prod.yml
services:
  localdata-mcp:
    restart: always
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
    environment:
      - LOCALDATA_LOG_LEVEL=warning
      - LOCALDATA_SECURITY_RESTRICT_PATHS=true
```

### Credential management

Store credentials in environment files (never commit them):

```bash
# .env.production (add to .gitignore)
LOCALDATA_DB_PROD_TYPE=postgresql
LOCALDATA_DB_PROD_CONNECTION_STRING=postgresql://user:password@db-host:5432/production
```

```yaml
services:
  localdata-mcp:
    env_file:
      - .env.production
```

## Security

- The container runs as a non-root user (`mcpuser`, UID 1000)
- File access is restricted to the mounted `/app/data` directory
- Use read-only mounts for configuration: `-v /path/to/config:/app/config:ro`
- Isolate services on an internal Docker network:
  ```yaml
  networks:
    localdata-net:
      driver: bridge
      internal: true  # No internet access
  ```

## Troubleshooting

### Permission denied on mounted volumes

```bash
# Ensure the data directory is owned by UID 1000
sudo chown -R 1000:1000 ./data
```

### Database services not ready

Database containers may take a few seconds to initialize. If LocalData MCP starts before they are ready:

```bash
docker-compose up -d postgres mysql mongodb redis elasticsearch
sleep 5
docker-compose up -d localdata-mcp
```

### Port conflicts

If a host port is already in use, override it in `docker-compose.override.yml`:

```yaml
services:
  postgres:
    ports:
      - "5433:5432"
```

### Debug mode

```bash
docker-compose run --rm \
  -e LOCALDATA_LOG_LEVEL=debug \
  localdata-mcp
```

### Memory issues with large files

Increase the container memory limit:

```yaml
deploy:
  resources:
    limits:
      memory: 4G
```

The server's built-in memory management (streaming, chunking) handles large datasets automatically, but the container needs enough headroom for the OS and Python runtime.
