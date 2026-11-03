# localdata-test-kdc — the Kerberos realm the Kerberos PostgreSQL
# authenticates against.
#
# One Dockerfile per service in docker-compose.test.yml (task 35, #139); see
# scripts/endpoint-run.sh for how it is built, run and removed.
#
# The KDC packages are installed when the image is built rather than on every
# start, as the compose `command:` used to do. The script that runs is
# unchanged and still rebuilds the realm on every start; environment, ports,
# volumes and the healthcheck stay in the compose file.
FROM alpine:3

LABEL org.opencontainers.image.source="https://github.com/ChrisGVE/localdata-mcp" \
      io.github.chrisgve.localdata-mcp.test-service="localdata-test-kdc"

RUN apk add --no-cache krb5-server krb5

COPY kdc.sh /usr/local/bin/kdc.sh

CMD ["sh", "/usr/local/bin/kdc.sh"]
