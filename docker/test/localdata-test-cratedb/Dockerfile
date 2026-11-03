# localdata-test-cratedb — CrateDB over its HTTP endpoint.
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
#
# Pinned from `latest` (2026-11-03). 6.5.0 is the server measured live
# (docs/CONSTRAINTS.md, the CrateDB amendments) and the
# `org.opencontainers.image.version` label of the image this machine held as
# `latest`; the registry resolves 6.5.0 to the same digest,
# sha256:f5a2aebcad96…, so the pin changes no bytes.
FROM crate:6.5.0

LABEL org.opencontainers.image.source="https://github.com/ChrisGVE/localdata-mcp" \
      io.github.chrisgve.localdata-mcp.test-service="localdata-test-cratedb"
