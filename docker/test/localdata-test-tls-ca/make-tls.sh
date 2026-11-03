#!/bin/sh
# make-tls.sh — a certificate authority, one server certificate and one client
# certificate, made fresh on every start of localdata-test-tls-ca.
#
# Moved out of docker-compose.test.yml's `command:` into a file (task 35, #139),
# where it reads as a script rather than as YAML. What it does is unchanged; why
# each step is there is said in the compose entry and beside each step below.
#
# Writes to two mounts the compose entry provides: /tls, a named volume the TLS
# PostgreSQL reads (server key, server certificate, CA, pg_hba.conf, the
# certuser role), and /public, a bind mount of tests/tls where the *client*
# half goes (CA certificate, client certificate and key).
set -e
cd /tls
openssl req -x509 -newkey rsa:2048 -nodes -days 3650 \
	-keyout ca.key -out ca.crt -subj "/CN=localdata-test-ca"
openssl req -newkey rsa:2048 -nodes -keyout server.key -out server.csr \
	-subj "/CN=localhost"
# `IP:127.0.0.1` in the SAN is the part that matters and the part most
# easily left out: `sslmode=verify-full` checks the name the client *asked
# for* against the certificate, and every URL from the host asks for
# 127.0.0.1. A certificate naming only `localhost` verifies under
# `verify-ca` and fails under `verify-full`, which would leave the stricter
# of the two modes — the one worth testing — quietly untested.
#
# The names come from TLS_SERVER_NAMES in the compose entry, because inside the
# compose network the client asks for the service's own name instead
# (task 35, #139). An IPv4 address becomes an `IP:` entry, anything else `DNS:`.
san=""
for name in $TLS_SERVER_NAMES; do
	case "$name" in
	*[!0-9.]*) entry="DNS:$name" ;;
	*) entry="IP:$name" ;;
	esac
	san="${san:+$san, }$entry"
done
echo "subjectAltName = $san" >server.ext
openssl x509 -req -in server.csr -CA ca.crt -CAkey ca.key -CAcreateserial \
	-out server.crt -days 3650 -extfile server.ext
# The client certificate's CN *is* the database user it authenticates
# as. That is what `cert` authentication means: no password is sent and
# the name in the certificate is the name PostgreSQL logs in.
openssl req -newkey rsa:2048 -nodes -keyout client.key -out client.csr \
	-subj "/CN=certuser"
openssl x509 -req -in client.csr -CA ca.crt -CAkey ca.key -CAcreateserial \
	-out client.crt -days 3650
cat >pg_hba.conf <<'HBA'
# The local socket stays trusting so that `initdb` and the healthcheck
# can work while every TCP connection is held to TLS.
local   all all                 trust
# Ordered, and the order is the point: `certuser` is matched first and
# must present a certificate, everyone else may send a password — over
# TLS either way, since there is no plain `host` line at all.
hostssl all certuser 0.0.0.0/0  cert
hostssl all certuser ::/0       cert
hostssl all all      0.0.0.0/0  scram-sha-256
hostssl all all      ::/0       scram-sha-256
HBA
# Authenticating is not the same as being allowed to do anything, and
# this is where the difference shows: a certificate says who the
# connection is, and PostgreSQL 15 and later give a fresh role no rights
# in `public` at all. Without the grant the mode connects perfectly and
# then cannot create a table, which reads like a broken endpoint rather
# than an unprivileged one. The entrypoint runs this against
# `POSTGRES_DB`, so the grant lands where the tests work.
cat >/tls/certuser.sql <<'SQL'
CREATE ROLE certuser LOGIN;
GRANT ALL ON SCHEMA public TO certuser;
SQL
chown 70:70 server.key server.crt ca.crt pg_hba.conf certuser.sql
chmod 600 server.key
chmod 644 server.crt ca.crt pg_hba.conf certuser.sql
cp ca.crt client.crt client.key /public/
chmod 600 /public/client.key
chmod 644 /public/ca.crt /public/client.crt
