# LocalData MCP server, as a container.
#
# The server speaks MCP over stdio, so a client runs it with `docker run -i`
# and mounts the files it should see under /app/data:
#
#   docker run --rm -i -v "$PWD:/app/data" chrisgve/localdata-mcp
#
# Dependencies come from uv.lock, so the image holds exactly the versions the
# test suite ran against, with every format and database extra (`[all]`).
# SQL Server is reached through unixODBC with the FreeTDS driver; Microsoft's
# own ODBC driver needs its licence accepted, which an image built for anyone
# cannot do on their behalf — install it in a derived image if you need it.

FROM python:3.12-slim-bookworm AS build

COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src/ src/
RUN uv sync --frozen --no-dev --extra all --no-editable


FROM python:3.12-slim-bookworm

ARG VERSION=unknown
LABEL org.opencontainers.image.title="LocalData MCP" \
      org.opencontainers.image.description="SQL over local data files and databases, for LLM agents" \
      org.opencontainers.image.source="https://github.com/ChrisGVE/localdata-mcp" \
      org.opencontainers.image.version="${VERSION}" \
      io.modelcontextprotocol.server.name="io.github.chrisgve/localdata-mcp"

RUN apt-get update \
    && apt-get install -y --no-install-recommends unixodbc tdsodbc \
    && rm -rf /var/lib/apt/lists/* \
    && odbcinst -i -d -f /usr/share/tdsodbc/odbcinst.ini

RUN groupadd --gid 1000 mcpuser \
    && useradd --uid 1000 --gid mcpuser --create-home mcpuser \
    && mkdir -p /app/data && chown mcpuser:mcpuser /app/data

COPY --from=build /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

USER mcpuser
WORKDIR /app/data
VOLUME ["/app/data"]

# A stdio server has no port to probe; this proves the installed server and
# its entry point load, which is what a broken image fails first.
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from localdata_mcp.server import main" || exit 1

ENTRYPOINT ["localdata-mcp"]
