# The endpoint test runner: pytest, every extra's driver, and the system
# libraries those drivers need, in a container rather than on the host
# (task 35, #139).
#
# Why a container: the host's libraries decided whether a dialect could be
# tested at all. Microsoft's ODBC driver on macOS loads Homebrew's OpenSSL,
# which moved to 4, and refused it, so SQL Server only ran behind a
# DYLD_LIBRARY_PATH workaround pointing at openssl@3. Here the driver comes from
# Microsoft's own Debian repository together with the OpenSSL it was built
# against, and nothing on the host is consulted.
#
# What is baked in, and what is not:
#
# * Baked in: the Python interpreter, the system libraries, and a virtualenv at
#   /opt/localdata-venv holding every locked dependency of every extra
#   (`uv sync --frozen --all-extras --no-install-project`). That venv sits
#   outside the repository, so the host's own .venv is never read or touched.
# * Not baked in: the project's source and its tests. The repository is
#   mounted at /repo when the container runs, and `PYTHONPATH=/repo/src` makes
#   that mount the only `localdata_mcp` on the path. An image built yesterday
#   therefore tests today's code; only a change to pyproject.toml or uv.lock
#   needs a rebuild, and scripts/endpoint-run.sh rebuilds on every run anyway.
#
# Nothing here writes into the mount on its own account: bytecode is not
# written (PYTHONDONTWRITEBYTECODE) and scripts/endpoint-run.sh passes
# `-p no:cacheprovider`, so a run leaves no __pycache__ or .pytest_cache in the
# working tree.
#
# No HEALTHCHECK: this container is a test process that runs once and exits
# with pytest's status, so "healthy" has no meaning for it.

# uv, from Astral's own image, pinned to the version this project's lock was
# last written with (uv 0.13.6, 2026-11-03).
FROM ghcr.io/astral-sh/uv:0.13.6 AS uv

# The project's floor (requires-python >= 3.11). Bookworm rather than the
# newer default because it is the Debian release Microsoft's repository
# publishes msodbcsql18 for under config/debian/12.
FROM python:3.11-slim-bookworm

LABEL org.opencontainers.image.source="https://github.com/ChrisGVE/localdata-mcp" \
      io.github.chrisgve.localdata-mcp.test-service="runner"

ARG DEBIAN_FRONTEND=noninteractive

# System libraries, one layer, with the apt lists removed in the same layer.
#
# * unixodbc — the driver manager pyodbc's manylinux wheel links against.
# * msodbcsql18 — Microsoft's SQL Server driver, from Microsoft's repository;
#   ACCEPT_EULA is the package's own required acknowledgement.
# * krb5-user — MIT's `kinit`, which tests/endpoints.py needs to fetch a ticket
#   for the Kerberos mode, and the GSSAPI libraries libpq loads for it.
#
# curl and gnupg are needed only to add Microsoft's key and list, and are
# purged again before the layer is committed. curl is bounded so a stalled
# mirror fails the build instead of hanging it.
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends ca-certificates curl gnupg unixodbc krb5-user; \
    curl -fsSL --connect-timeout 20 --max-time 120 \
        https://packages.microsoft.com/keys/microsoft.asc \
        | gpg --dearmor -o /usr/share/keyrings/microsoft-prod.gpg; \
    curl -fsSL --connect-timeout 20 --max-time 120 \
        https://packages.microsoft.com/config/debian/12/prod.list \
        -o /etc/apt/sources.list.d/mssql-release.list; \
    apt-get update; \
    ACCEPT_EULA=Y apt-get install -y --no-install-recommends msodbcsql18; \
    apt-get purge -y curl gnupg; \
    apt-get autoremove -y; \
    rm -rf /var/lib/apt/lists/*

COPY --from=uv /uv /usr/local/bin/uv

# The dependency set, resolved exactly as uv.lock records it (`--frozen`), for
# every extra, without the project itself. No uv cache is kept, so the layer
# holds the venv and nothing else.
ENV UV_PROJECT_ENVIRONMENT=/opt/localdata-venv \
    UV_NO_CACHE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PYTHON=/usr/local/bin/python3.11
WORKDIR /opt/localdata-build
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --all-extras --no-install-project

# An unprivileged default user. scripts/endpoint-run.sh runs the container as
# the invoking host user instead (`--user`), so anything a test does write into
# the mount belongs to whoever ran it; either way it is not root.
RUN useradd --create-home --uid 1000 runner

ENV PATH=/opt/localdata-venv/bin:$PATH \
    PYTHONPATH=/repo/src \
    PYTHONDONTWRITEBYTECODE=1 \
    HOME=/tmp
WORKDIR /repo
USER runner

CMD ["python", "-m", "pytest", "-p", "no:cacheprovider", "-q"]
