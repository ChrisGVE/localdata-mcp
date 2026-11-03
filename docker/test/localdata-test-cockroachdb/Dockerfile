# localdata-test-cockroachdb — CockroachDB, single node, insecure.
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
# Pinned from `latest` (2026-11-03). v26.3.4 is the version measured live
# (docs/CONSTRAINTS.md, the semicolon table) and the `version` label of the
# image this machine held as `latest`; the registry resolves v26.3.4 to the
# same index digest, sha256:e9a8c0a4f23e…, so the pin changes no bytes.
FROM cockroachdb/cockroach:v26.3.4

LABEL org.opencontainers.image.source="https://github.com/ChrisGVE/localdata-mcp" \
      io.github.chrisgve.localdata-mcp.test-service="localdata-test-cockroachdb"
