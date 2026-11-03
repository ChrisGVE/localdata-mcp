# localdata-test-tls-ca — the one-shot certificate authority the TLS
# PostgreSQL waits for.
#
# One Dockerfile per service in docker-compose.test.yml (task 35, #139); see
# scripts/endpoint-run.sh for how it is built, run and removed.
#
# `openssl` is installed when the image is built rather than on every start, as
# the compose `command:` used to do. The script that runs is unchanged and
# still makes fresh material on every start; volumes stay in the compose file.
FROM alpine:3

LABEL org.opencontainers.image.source="https://github.com/ChrisGVE/localdata-mcp" \
      io.github.chrisgve.localdata-mcp.test-service="localdata-test-tls-ca"

RUN apk add --no-cache openssl

COPY make-tls.sh /usr/local/bin/make-tls.sh

CMD ["sh", "/usr/local/bin/make-tls.sh"]
