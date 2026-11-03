# localdata-test-monetdb — MonetDB, the vendor's own image.
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
# Pinned from `latest` (2026-11-03). MonetDB 11.55.7 (Dec2025-SP3) is the
# server measured live (docs/CONSTRAINTS.md, the MonetDB section); the
# registry resolves Dec2025-SP3 to the digest of the image this machine held
# as `latest`, sha256:a71e6e8c8402…, so the pin changes no bytes. MonetDB
# tags by release name, not by number: `11.55.7` does not exist as a tag.
FROM monetdb/monetdb:Dec2025-SP3

LABEL org.opencontainers.image.source="https://github.com/ChrisGVE/localdata-mcp" \
      io.github.chrisgve.localdata-mcp.test-service="localdata-test-monetdb"
