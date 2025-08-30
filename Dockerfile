# LocalData MCP Server Dockerfile
# Provides a containerized environment for running the MCP server

FROM python:3.11-slim-bullseye

# Set metadata
LABEL maintainer="Christian C. Berclaz <christian.berclaz@mac.com>"
LABEL description="LocalData MCP Server - Comprehensive database and file format support"
LABEL version="1.3.0"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies required for various database and format support
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Build essentials for Python packages
    gcc \
    g++ \
    # Database client libraries
    libpq-dev \
    default-libmysqlclient-dev \
    # XML processing libraries
    libxml2-dev \
    libxslt1-dev \
    # HDF5 support
    libhdf5-dev \
    pkg-config \
    # Cleanup
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create a non-root user for security
RUN groupadd --gid 1000 mcpuser && \
    useradd --uid 1000 --gid mcpuser --shell /bin/bash --create-home mcpuser

# Set working directory
WORKDIR /app

# Copy requirements first for better Docker layer caching
COPY pyproject.toml README.md ./
COPY src/ src/

# Install the package and dependencies
RUN pip install --no-cache-dir -e . && \
    # Install optional modern database dependencies
    pip install --no-cache-dir \
    redis>=6.0.0 \
    elasticsearch>=9.0.0 \
    pymongo>=4.0.0 \
    influxdb-client>=1.40.0 \
    neo4j>=5.0.0 \
    couchdb>=1.2.0 \
    duckdb>=0.9.0

# Create directories for data and ensure proper permissions
RUN mkdir -p /app/data /app/temp && \
    chown -R mcpuser:mcpuser /app

# Switch to non-root user
USER mcpuser

# Create a simple entrypoint script
RUN cat > /app/entrypoint.sh << 'EOF'
#!/bin/bash
set -e

# Default to stdio transport if no arguments provided
if [ $# -eq 0 ]; then
    exec localdata-mcp
else
    exec "$@"
fi
EOF

# Make entrypoint executable
USER root
RUN chmod +x /app/entrypoint.sh
USER mcpuser

# Set default working directory to /app/data for file operations
WORKDIR /app/data

# Expose the working directory as a volume
VOLUME ["/app/data"]

# Health check to ensure the server can start
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import localdata_mcp.localdata_mcp; print('OK')" || exit 1

# Set the entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]

# Default command - runs the MCP server with stdio transport
CMD []