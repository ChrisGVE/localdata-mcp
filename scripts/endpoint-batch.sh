#!/usr/bin/env bash
# endpoint-batch.sh — run one batch of endpoint containers end to end, then put the
# disk back the way it was found.
#
# The endpoint catalogue is sixteen database dialects plus a four-service
# authentication axis, and this machine holds about six containers at once before the
# Docker VM starves them (issue #46, docs/CONSTRAINTS.md §13.4). So the suite is run in
# batches, and the batches are written down HERE rather than in prose, because prose
# went stale twice: the working sets had to be reconstructed from a GitHub issue's own
# comment history on 2026-08-01.
#
# The images are large — 33 GB for the sixteen, of which Exasol alone is 12 GB — and
# they are pulled from public registries, so keeping them between sessions costs disk
# to save a download. This script therefore REMOVES them when the batch is done. That
# is the deliberate trade (Chris, 2026-08-03): re-download every time, and leave
# nothing behind.
#
# Usage:
#   ./scripts/endpoint-batch.sh <batch>...   # a, b, c, d, e — or 'all'
#   ./scripts/endpoint-batch.sh a --keep     # leave the containers and images up
#   ./scripts/endpoint-batch.sh all -- -x -q # everything after -- goes to pytest
#
# Exit code is the worst pytest exit code across the batches run. Cleanup happens even
# when the tests fail, because a red suite is exactly when the next batch is wanted.

set -euo pipefail

# mapfile and ${x^^} are bash 4. macOS still ships 3.2 as /bin/bash, so say so plainly
# rather than failing later with a syntax error that reads like a bug in this script.
if ((BASH_VERSINFO[0] < 4)); then
	echo "$0 needs bash 4+; this is ${BASH_VERSION}." >&2
	echo "Run it directly (the shebang finds a newer bash) rather than as 'bash $0'." >&2
	exit 2
fi

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE=(docker compose -f "$REPO/docker-compose.test.yml")
PYTEST=("$REPO/.venv/bin/python" -m pytest)

HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-600}" # Oracle and Exasol are the slow ones.
HEALTH_INTERVAL=5

# --- the batches -----------------------------------------------------------------
#
# Sized to stay under the six-container ceiling rather than at it. PostgreSQL appears
# in every batch because a good deal of the suite needs a reference engine to compare
# against, not because it is cheap.

batch_services() {
	case "$1" in
	a) echo "localdata-test-postgres localdata-test-mysql localdata-test-mariadb localdata-test-mssql localdata-test-oracle localdata-test-firebird" ;;
	b) echo "localdata-test-postgres localdata-test-clickhouse localdata-test-cockroachdb localdata-test-yugabytedb localdata-test-trino localdata-test-monetdb" ;;
	c) echo "localdata-test-postgres localdata-test-cratedb localdata-test-opengauss localdata-test-ydb localdata-test-databend" ;;
	d) echo "localdata-test-postgres localdata-test-exasol" ;;
	e) echo "localdata-test-postgres-trust localdata-test-clickhouse-noauth localdata-test-postgres-tls localdata-test-postgres-krb" ;;
	*) return 1 ;;
	esac
}

# Batch E's four services are ordinary entries in docker-compose.test.yml — the
# file defines no profiles at all, so the flags below are inert and the batch
# works because a service with no `profiles:` key always starts. They are kept
# only so the flag shape is ready if profiles are ever introduced. What does the
# real work is that compose starts what a named service *depends on*, so the CA
# and the KDC come up with them (CONSTRAINTS §27.2).
batch_profiles() {
	case "$1" in
	e) echo "--profile tls --profile krb" ;;
	*) echo "" ;;
	esac
}

batch_dialects() {
	case "$1" in
	a) echo "postgresql, mysql, mariadb, mssql, oracle, firebird" ;;
	b) echo "clickhouse, cockroachdb, yugabytedb, trino, monetdb" ;;
	c) echo "cratedb, opengauss, ydb, databend" ;;
	d) echo "exasol" ;;
	e) echo "the authentication axis — trust, no-auth, TLS, Kerberos" ;;
	esac
}

# --- waiting ---------------------------------------------------------------------
#
# Ask compose which containers it started rather than guessing their names: the
# project prefix is a property of the directory, and a hand-written name went stale
# once already: the earlier scripts/wait-for-databases.sh named three services after
# the compose file had grown past them, and was deleted rather than repaired.
#
# A container with no healthcheck counts as ready once it is running. Every service in
# this compose file has one, so that branch is a safety net rather than a normal path.

wait_for_health() {
	local services=("$@")
	local waited=0

	echo "  waiting up to ${HEALTH_TIMEOUT}s for health..."
	while true; do
		local pending=()
		for svc in "${services[@]}"; do
			local cid
			cid="$("${COMPOSE[@]}" ps -q "$svc" 2>/dev/null || true)"
			if [[ -z "$cid" ]]; then
				pending+=("$svc(no container)")
				continue
			fi
			local state health
			state="$(docker inspect --format '{{.State.Status}}' "$cid" 2>/dev/null || echo missing)"
			health="$(docker inspect --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}' "$cid" 2>/dev/null || echo missing)"
			case "$health" in
			healthy) ;;
			none) [[ "$state" == "running" ]] || pending+=("$svc($state)") ;;
			*) pending+=("$svc($health)") ;;
			esac
		done

		if [[ ${#pending[@]} -eq 0 ]]; then
			echo "  all ${#services[@]} healthy after ${waited}s"
			return 0
		fi
		if [[ $waited -ge $HEALTH_TIMEOUT ]]; then
			echo "  TIMEOUT after ${waited}s — still pending: ${pending[*]}" >&2
			echo "  read 'docker logs' before believing a red suite: a starved container" >&2
			echo "  looks exactly like a broken commit (issue #46)." >&2
			return 1
		fi
		sleep "$HEALTH_INTERVAL"
		waited=$((waited + HEALTH_INTERVAL))
	done
}

# --- teardown --------------------------------------------------------------------
#
# The image list comes from compose, so it cannot drift from the services above; a
# hand-maintained list is the thing that rots. An image another project is still using
# refuses to be removed, and that refusal is fine — hence the per-image loop rather
# than one call that would abort the rest.

teardown() {
	local batch="$1"
	shift
	local services=("$@")
	local profiles
	read -r -a profiles <<<"$(batch_profiles "$batch")"

	echo "  removing containers, volumes and network..."
	"${COMPOSE[@]}" "${profiles[@]}" down -v --remove-orphans >/dev/null 2>&1 || true

	local images
	mapfile -t images < <("${COMPOSE[@]}" config --images "${services[@]}" 2>/dev/null | sort -u)
	echo "  removing ${#images[@]} images..."
	for img in "${images[@]}"; do
		# alpine is a base half the world shares and re-pulls in a second; removing it
		# buys ~8 MB and costs every other project on this machine a pull. Said out
		# loud rather than skipped silently, so the count matches the lines under it.
		if [[ "$img" == alpine:* ]]; then
			echo "    kept    $img (shared base, not worth re-pulling)"
			continue
		fi
		if docker rmi "$img" >/dev/null 2>&1; then
			echo "    removed $img"
		else
			echo "    kept    $img (in use elsewhere, or already gone)"
		fi
	done
}

# --- arguments -------------------------------------------------------------------

KEEP=0
BATCHES=()
PYTEST_ARGS=()
while [[ $# -gt 0 ]]; do
	case "$1" in
	--keep) KEEP=1 ;;
	--)
		shift
		PYTEST_ARGS=("$@")
		break
		;;
	all) BATCHES+=(a b c d e) ;;
	a | b | c | d | e) BATCHES+=("$1") ;;
	*)
		echo "unknown argument: $1" >&2
		echo "usage: $0 <a|b|c|d|e|all>... [--keep] [-- <pytest args>]" >&2
		exit 2
		;;
	esac
	shift
done

if [[ ${#BATCHES[@]} -eq 0 ]]; then
	echo "usage: $0 <a|b|c|d|e|all>... [--keep] [-- <pytest args>]" >&2
	exit 2
fi

# --- run -------------------------------------------------------------------------

worst=0
declare -a SUMMARY=()

for batch in "${BATCHES[@]}"; do
	read -r -a services <<<"$(batch_services "$batch")"
	read -r -a profiles <<<"$(batch_profiles "$batch")"

	echo
	echo "=============================================================="
	echo "batch ${batch^^} — $(batch_dialects "$batch")"
	echo "  ${#services[@]} containers: ${services[*]}"
	echo "=============================================================="

	# Cleanup must survive a failure anywhere below, including a Ctrl-C, or a stopped
	# batch leaves 30 GB behind — the whole thing this script exists to prevent.
	if [[ $KEEP -eq 0 ]]; then
		trap 'echo; echo "interrupted — cleaning up"; teardown "$batch" "${services[@]}"; exit 130' INT TERM
	fi

	echo "  pulling and starting..."
	"${COMPOSE[@]}" "${profiles[@]}" up -d "${services[@]}"

	rc=0
	if wait_for_health "${services[@]}"; then
		echo "  running the suite..."
		"${PYTEST[@]}" "${PYTEST_ARGS[@]}" || rc=$?
	else
		rc=1
	fi

	[[ $rc -gt $worst ]] && worst=$rc
	SUMMARY+=("batch ${batch^^}: $([[ $rc -eq 0 ]] && echo PASS || echo "FAIL (pytest exit $rc)")")

	if [[ $KEEP -eq 1 ]]; then
		echo "  --keep: leaving ${#services[@]} containers and their images in place"
	else
		trap - INT TERM
		teardown "$batch" "${services[@]}"
	fi
done

echo
echo "=============================================================="
printf '%s\n' "${SUMMARY[@]}"
echo
echo "Dialects NOT covered by this run are every one not listed above —"
echo "a green line from one batch is not a green line for the catalogue (issue #46)."
[[ $KEEP -eq 0 ]] && docker system df | head -3
echo "=============================================================="

exit "$worst"
