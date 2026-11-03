# localdata-test-mssql — SQL Server 2022.
#
# One Dockerfile per service in docker-compose.test.yml (task 35, #139): the
# test run builds this image, runs it, and removes both container and image when
# it is done, so this file is what stays between runs. See
# scripts/endpoint-run.sh.
#
# Nothing is added to the upstream image on purpose. Environment, ports, the
# command and the healthcheck all stay in the compose file, because
# tests/endpoints.py reads ports and credentials out of it and a second copy
# here would be a second source of truth that could disagree.
FROM mcr.microsoft.com/mssql/server:2022-latest

LABEL org.opencontainers.image.source="https://github.com/ChrisGVE/localdata-mcp" \
      io.github.chrisgve.localdata-mcp.test-service="localdata-test-mssql"
