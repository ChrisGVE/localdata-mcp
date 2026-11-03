#!/bin/sh
# kdc.sh — build the LOCALDATA.TEST realm and serve it, on every start of
# localdata-test-kdc.
#
# Moved out of docker-compose.test.yml's `command:` into a file (task 35, #139).
# What it does is unchanged. The realm, the user principal and its password come
# from the compose entry's environment, which tests/endpoints.py reads too.
#
# Writes into /krb, a named volume the Kerberos PostgreSQL reads: the service
# keytab, the pg_hba.conf that names the method, and the krbuser role.
set -e
cat >/etc/krb5.conf <<EOF
[libdefaults]
  default_realm = $KRB5_REALM
  dns_lookup_realm = false
  dns_lookup_kdc = false
  rdns = false
[realms]
  $KRB5_REALM = {
    kdc = 127.0.0.1:88
  }
EOF
# Built on every start, not once behind a marker in the volume. The
# marker used to live in /krb while the KDC database it stood for lived
# in the container's own filesystem, so the two had different lifetimes:
# `docker compose down` destroyed the database and kept the marker, and
# the next `up` skipped the build and died with `cannot initialize realm
# LOCALDATA.TEST`. A `stop`/`up` cycle never showed it, because that
# restarts the container rather than recreating it (#63).
#
# Rebuilding unconditionally also keeps the keytab and the database in
# step, which the marker could not: a half-restored state where the
# keytab is the old realm's and the database is new authenticates
# nobody, and says so in a message about neither.
# The directory has to survive: `kdb5_util create` writes into it and
# will not make it, answering `No such file or directory while creating
# database` if it is gone.
rm -rf /var/lib/krb5kdc /krb/.made
mkdir -p /var/lib/krb5kdc
kdb5_util create -s -P "$KRB5_PASSWORD" -r "$KRB5_REALM"
kadmin.local -q "addprinc -pw $KRB5_PASSWORD $KRB5_PRINCIPAL@$KRB5_REALM"
# One service principal per host a client may name, all in the one keytab
# PostgreSQL reads: `postgres/127.0.0.1` for a client on the host, and
# `postgres/<service name>` for one inside the compose network. PostgreSQL
# accepts a ticket for any principal its keytab holds.
rm -f /krb/postgres.keytab
for host in $KRB5_SERVICE_HOSTS; do
	kadmin.local -q "addprinc -randkey postgres/$host@$KRB5_REALM"
	kadmin.local -q "ktadd -k /krb/postgres.keytab postgres/$host@$KRB5_REALM"
done
cat >/krb/pg_hba.conf <<'HBA'
local   all all             trust
# `include_realm=0` is what makes the database user `krbuser` rather than
# `krbuser@LOCALDATA.TEST`, which is not a role and would be refused
# after a successful authentication — the confusing kind of failure.
host    all all 0.0.0.0/0   gss include_realm=0
host    all all ::/0        gss include_realm=0
HBA
cat >/krb/krbuser.sql <<'SQL'
CREATE ROLE krbuser LOGIN;
GRANT ALL ON SCHEMA public TO krbuser;
SQL
chown 70:70 /krb/postgres.keytab /krb/pg_hba.conf /krb/krbuser.sql
chmod 640 /krb/postgres.keytab
chmod 644 /krb/pg_hba.conf /krb/krbuser.sql
exec krb5kdc -n
