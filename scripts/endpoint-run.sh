#!/usr/bin/env bash
# endpoint-run.sh — run the endpoint suite one database at a time, each in an
# isolated compose project, with the tests themselves running in a container,
# then put the machine back the way it was found (task 35, #139).
#
# First, under the machine-wide lock, build the runner image (docker/test/runner)
# with a tag of this run's own, so that another invocation's cleanup can never
# remove it (#142). Then, for each service named, serially:
#
#   1. take the machine-wide lock, so only one test database is ever up;
#   2. build its image from docker/test/<service>/Dockerfile;
#   3. `up` it under a compose project of its own, which gives it its own
#      network, so it never meets another stack running on this machine;
#   4. wait for its healthcheck;
#   5. run pytest in the runner container (docker/test/runner) on that network,
#      with LOCALDATA_TEST_IN_NETWORK=1 so tests/endpoints.py addresses the
#      database by service name;
#   6. ALWAYS, through a trap and even on failure or Ctrl-C: `down -v --rmi
#      local`, remove any base image this run pulled, and release the lock.
#
# When the last service is done the runner image goes too, with the build cache
# this run created. The Dockerfiles are what stays.
#
# A second invocation waits for the lock and is otherwise independent: every
# image and build cache record a run removes is one it created itself. That holds
# because this script builds only while it holds the lock, so what the build
# cache gains during one of its builds is that build's.
#
# A service that needs a helper brings it up with it, by its compose
# `depends_on`: the TLS PostgreSQL brings the CA, the Kerberos one the KDC. That
# is one database and its helper, and still one database at a time.
#
# Usage:
#   scripts/endpoint-run.sh <service|all>... [--keep] [--seed] [-- <pytest args>]
#
#   scripts/endpoint-run.sh postgres               # one engine, standard selection
#   scripts/endpoint-run.sh all                    # every engine, one after another
#   scripts/endpoint-run.sh mssql -- -x -k stats   # ad hoc: these pytest args instead
#   scripts/endpoint-run.sh --seed --keep postgres # a seeded database left up
#
# A service is named as in docker-compose.test.yml (`localdata-test-postgres`)
# or without the prefix (`postgres`). `all` is every service that has tests of
# its own, in the compose file's order; the CA and the KDC are helpers and are
# never run on their own.
#
# The standard selection for a service is `-q -rs -m endpoint -k <selection>`, where
# the selection comes from tests/endpoints.py (`selection()`), so the mapping
# from a service to its tests lives in one place. Everything after `--`
# replaces it.
#
# --keep   leave the database (and its helper) up when the run ends, and keep the
#          runner image. Takes exactly one service, and the lock stays held while
#          it is up, because one container at a time is the rule this machine
#          runs under. The script prints how to reach it and how to dispose of it.
# --seed   load the standard assets into the database (scripts/seed_endpoint.py)
#          once it is healthy. pytest then runs only if args follow `--`: a seed
#          is for an ad-hoc session, not for the standard suite.
# --dispose <service>
#          undo a --keep run: everything a normal run removes at its end
#          (containers, volumes, network, images, pulled bases, the runner, the
#          build cache the run created) and the lock it left held. It works
#          from state the kept run saved, so it removes exactly what that run
#          added.
#
# Exit code: the worst pytest exit code across the services run; 1 for a
# service that never became healthy or could not be seeded; 2 for a usage error.
#
# Environment:
#   HEALTH_TIMEOUT         seconds to wait for a healthcheck (default 900)
#   LOCALDATA_DOCKER_LOCK  the lock directory (default /tmp/localdata-docker.lock)
#   LOCALDATA_LOCK_POLL    seconds between attempts at a held lock (default 20)

set -euo pipefail

# mapfile is bash 4. macOS still ships 3.2 as /bin/bash.
if ((BASH_VERSINFO[0] < 4)); then
	echo "$0 needs bash 4+; this is ${BASH_VERSION}." >&2
	echo "Run it directly (the shebang finds a newer bash) rather than as 'bash $0'." >&2
	exit 2
fi

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_FILES=(-f "$REPO/docker-compose.test.yml" -f "$REPO/docker/test/in-network.compose.yml")
RUNNER_IMAGE="" # localdata-test/runner:<this run's id>, set once the run has one
RUNNER_DOCKERFILE="$REPO/docker/test/runner/Dockerfile"
LOCK="${LOCALDATA_DOCKER_LOCK:-/tmp/localdata-docker.lock}"
LOCK_POLL="${LOCALDATA_LOCK_POLL:-20}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-900}" # Exasol builds a cluster before it serves.
HEALTH_POLL=5
PREFIX="localdata-test-"
# Who the runner runs as: the invoking user's uid and gid, so that on a Linux
# host, where a bind mount keeps real ownership, the tests can read what that
# user can (the TLS client key is 0600). That uid has no entry in the image's
# /etc/passwd, so it is given a name through USER and LOGNAME, which
# getpass.getuser() reads before it falls back to the password database.
# Measured: without them pyexasol's handshake died on every attempt with
# `KeyError: 'getpwuid(): uid not found: 501'` and all 40 Exasol tests errored.
RUNNER_AS=(--user "$(id -u):$(id -g)" -e USER=localdata -e LOGNAME=localdata)

# State the cleanup path reads. Set before the trap is, so the trap never sees
# an unset name.
KEEP=0
SEED=0
ADHOC=0
REQUESTED=()
PYTEST_ARGS=()
PULLED=()        # bases pulled for the current service; removed with it
RUNNER_PULLED=() # bases pulled for the runner; removed with the runner
HOLDING_LOCK=0
CURRENT_PROJECT=""
WORK=""

usage() {
	echo "usage: $0 <service|all>... [--keep] [--seed] [-- <pytest args>]" >&2
	echo "       $0 --dispose <service>" >&2
	exit 2
}

# --- images -----------------------------------------------------------------------
#
# A base image that was already on the machine is someone else's and is never
# touched. One this run pulls is pulled as a tagged image rather than left to the
# builder, because a base the builder pulls itself lands in the build cache, where
# `docker rmi` cannot reach it (measured, docs/CONSTRAINTS.md §13.7).

pull_if_absent() {
	local image="$1"
	if ! docker image inspect "$image" >/dev/null 2>&1; then
		echo "  pulling $image (removed again afterwards)"
		docker pull -q "$image" >/dev/null
		PULLED+=("$image")
	fi
}

bases_of() {
	# Every `FROM` image of a Dockerfile.
	awk 'toupper($1) == "FROM" { print $2 }' "$1"
}

remove_pulled() {
	local image
	for image in "${PULLED[@]}"; do
		if docker rmi "$image" >/dev/null 2>&1; then
			echo "  removed pulled base $image"
		else
			echo "  kept    pulled base $image (in use)"
		fi
	done
	PULLED=()
}

# Removing an image does not remove the layers BuildKit cached while building it;
# measured, the runner leaves about 800 MB behind (docs/CONSTRAINTS.md §13.7). So
# the end of a run reclaims the records its own builds created that are no longer
# shared with any image. A record that was here before is never touched, and
# neither is one another invocation created while this run was waiting for the
# lock or testing without it (#142).
#
# In passes, because the records form chains and BuildKit keeps a parent until
# its children are gone: measured, one pass over the runner's records left six of
# eleven behind, and the fourth pass cleared the last.
cache_ids() {
	docker buildx du --format '{{.ID}}' 2>/dev/null | sort || true
}

# Run a build command, and add the build cache records that appeared during it to
# the ones this run owns. Called only while the lock is held, which is what makes
# what appeared this build's and not another invocation's.
owned_build() {
	local before rc=0
	before="$(mktemp "$WORK/cache-before.XXXXXX")"
	cache_ids >"$before"
	"$@" || rc=$?
	cache_ids | comm -13 "$before" - >>"$WORK/cache-mine"
	rm -f "$before"
	return "$rc"
}

reclaim_build_cache() {
	local id candidates present pass
	[[ -s "$WORK/cache-mine" ]] || return 0
	candidates="$WORK/cache-candidates"
	docker buildx du --format '{{.ID}} {{.Shared}}' 2>/dev/null |
		awk '$2 == "false" { print $1 }' | sort | comm -12 - <(sort -u "$WORK/cache-mine") >"$candidates"
	[[ -s "$candidates" ]] || return 0
	for pass in 1 2 3 4 5 6 7 8 9 10; do
		present="$(docker buildx du --format '{{.ID}}' 2>/dev/null | grep -Fx -f "$candidates" || true)"
		if [[ -z "$present" ]]; then
			echo "reclaimed the build cache this run created ($(wc -l <"$candidates" | xargs) records, $((pass - 1)) passes)"
			return 0
		fi
		for id in $present; do
			docker buildx prune -f --filter "id=$id" >/dev/null 2>&1 || true
		done
	done
	echo "build cache: $(wc -w <<<"$present" | xargs) records this run created are still held" >&2
}

dispose_runner() {
	if [[ -n "$RUNNER_IMAGE" ]]; then
		docker rmi "$RUNNER_IMAGE" >/dev/null 2>&1 && echo "removed $RUNNER_IMAGE"
	fi
	PULLED=("${RUNNER_PULLED[@]}")
	remove_pulled
	reclaim_build_cache
}

# Python inside the runner, against the mounted repository, with no network.
endpoints_say() {
	docker run --rm --network none "${RUNNER_AS[@]}" -v "$REPO:/repo:ro" \
		"$RUNNER_IMAGE" python -c \
		"import sys; sys.path.insert(0, 'tests'); import endpoints; $1"
}

# --- the lock ---------------------------------------------------------------------
#
# A directory, because mkdir is atomic: whoever creates it holds the lock. It is
# the same lock every process on this machine that starts a test database takes
# (CONTRIBUTING.md, "Endpoint tests"), so a run here waits for another one's
# container to go rather than starving it.

take_lock() {
	local waited=0
	until mkdir "$LOCK" 2>/dev/null; do
		if ((waited % 120 == 0)); then
			echo "  waiting for $LOCK ($(cat "$LOCK/owner" 2>/dev/null || echo 'no owner file'))"
		fi
		sleep "$LOCK_POLL"
		waited=$((waited + LOCK_POLL))
	done
	# "keep" or "run", so that --dispose can tell a kept run's lock from anyone
	# else's.
	echo "endpoint-run $$ $1 $([[ $KEEP -eq 1 ]] && echo keep || echo run) $(date)" >"$LOCK/owner"
	HOLDING_LOCK=1
}

release_lock() {
	if [[ $HOLDING_LOCK -eq 1 ]]; then
		rm -f "$LOCK/owner"
		rmdir "$LOCK" 2>/dev/null || true
		HOLDING_LOCK=0
	fi
}

# --- one service ------------------------------------------------------------------

compose() {
	docker compose -p "$CURRENT_PROJECT" "${COMPOSE_FILES[@]}" "$@"
}

teardown() {
	[[ -n "$CURRENT_PROJECT" ]] || return 0
	echo "  removing containers, volumes, network and images of $CURRENT_PROJECT..."
	compose down -v --rmi local --remove-orphans >/dev/null 2>&1 || true
	remove_pulled
	CURRENT_PROJECT=""
}

wait_for_health() {
	local service="$1" waited=0 cid state="" health=""
	echo "  waiting up to ${HEALTH_TIMEOUT}s for $service to be healthy..."
	while true; do
		cid="$(compose ps -q "$service" 2>/dev/null || true)"
		if [[ -n "$cid" ]]; then
			state="$(docker inspect --format '{{.State.Status}}' "$cid" 2>/dev/null || echo missing)"
			health="$(docker inspect --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}' "$cid" 2>/dev/null || echo missing)"
			if [[ "$health" == healthy || ("$health" == none && "$state" == running) ]]; then
				echo "  healthy after ${waited}s"
				return 0
			fi
			if [[ "$state" == exited || "$state" == dead ]]; then
				echo "  $service $state before it was healthy; its log ends:" >&2
				docker logs --tail 30 "$cid" >&2 2>&1 || true
				return 1
			fi
		fi
		if ((waited >= HEALTH_TIMEOUT)); then
			echo "  TIMEOUT after ${waited}s (state ${state:-none}, health ${health:-none})." >&2
			echo "  Read 'docker logs' before believing a red run: a starved container" >&2
			echo "  looks exactly like a broken commit (issue #46)." >&2
			return 1
		fi
		sleep "$HEALTH_POLL"
		waited=$((waited + HEALTH_POLL))
	done
}

# A container from the runner image on this project's network, addressing the
# database by service name. The repository is mounted read-only: nothing run
# here has any business writing into the tree, and a mount it cannot write to
# proves that it does not.
in_network() {
	docker run --rm --init \
		--network "$CURRENT_PROJECT-net" \
		"${RUNNER_AS[@]}" \
		-e LOCALDATA_TEST_IN_NETWORK=1 \
		-v "$REPO:/repo:ro" \
		"$RUNNER_IMAGE" "$@"
}

run_tests() {
	local log="$1"
	shift
	in_network python -m pytest -p no:cacheprovider "$@" 2>&1 | tee "$log"
	return "${PIPESTATUS[0]}"
}

# What a kept run leaves for --dispose: its runner's tag, the build cache records
# its builds created, and the bases it pulled, so that disposing later removes
# exactly what the run added and nothing that was here before it.
keep_state_dir() {
	echo "${TMPDIR:-/tmp}/localdata-endpoint-run/$1"
}

save_keep_state() {
	local state
	state="$(keep_state_dir "$CURRENT_PROJECT")"
	mkdir -p "$state"
	touch "$WORK/cache-mine"
	cp "$WORK/cache-mine" "$state/cache-mine"
	echo "$RUNNER_IMAGE" >"$state/runner-image"
	printf '%s\n' "${PULLED[@]}" >"$state/pulled"
	printf '%s\n' "${RUNNER_PULLED[@]}" >"$state/runner-pulled"
}

print_keep_instructions() {
	local service="$1" short="$2" container_port published
	container_port="$(endpoints_say "print([t for t in endpoints.TARGETS if t.service == '$service'][0].container_port)")"
	published="$(compose port "$service" "$container_port" 2>/dev/null || true)"
	echo
	echo "  --keep: $service is up as compose project $CURRENT_PROJECT, and $LOCK is"
	echo "  still held, so no other test database can start until this one is gone."
	echo "  From the host it answers on ${published:-its published port}; from a container"
	echo "  on network $CURRENT_PROJECT-net it is $service:$container_port. The URL and"
	echo "  its credentials are what tests/endpoints.py builds from docker-compose.test.yml."
	echo "  Dispose of it, its images, the runner, the build cache this run made and the"
	echo "  lock with:"
	echo
	echo "    $0 --dispose $short"
}

# --dispose <service>: undo a --keep run, from the state it saved.
dispose_kept() {
	local short="${1#"$PREFIX"}" state
	CURRENT_PROJECT="ldrun-$short"
	state="$(keep_state_dir "$CURRENT_PROJECT")"
	if [[ ! -d "$state" ]]; then
		echo "no kept run of $short to dispose of ($state does not exist)" >&2
		exit 2
	fi
	WORK="$state"
	RUNNER_IMAGE="$(cat "$state/runner-image")"
	mapfile -t PULLED < <(grep -v '^$' "$state/pulled" || true)
	mapfile -t RUNNER_PULLED < <(grep -v '^$' "$state/runner-pulled" || true)
	# The kept run took the lock and left it held for the database it left up.
	# Released only if it is still that run's: a lock whose owner file names
	# anything else belongs to someone who took it since, and is not ours.
	if grep -q "^endpoint-run [0-9]* $PREFIX$short keep " "$LOCK/owner" 2>/dev/null; then
		HOLDING_LOCK=1
	else
		echo "  $LOCK is not held by the kept run of $short; leaving it alone"
	fi
	teardown
	dispose_runner
	release_lock
	rm -rf "$state"
	WORK=""
	echo "disposed of $short"
}

# The one cleanup path, whatever ends the run: a normal exit, a failure under
# `set -e`, or a signal. With --keep it removes only its scratch directory.
# shellcheck disable=SC2329 # invoked by the EXIT trap below
cleanup() {
	local status=$?
	trap - EXIT INT TERM
	if [[ $KEEP -eq 0 ]]; then
		# The runner and its pulled bases go before the lock does, so that what
		# this run removes is never a base another invocation is building on:
		# that invocation builds only while it holds the lock.
		teardown
		dispose_runner
		release_lock
	fi
	[[ -n "$WORK" ]] && rm -rf "$WORK"
	exit "$status"
}

# --- arguments --------------------------------------------------------------------

DISPOSE=""
while [[ $# -gt 0 ]]; do
	case "$1" in
	--keep) KEEP=1 ;;
	--seed) SEED=1 ;;
	--dispose)
		[[ $# -ge 2 ]] || usage
		DISPOSE="$2"
		shift
		;;
	--)
		shift
		PYTEST_ARGS=("$@")
		ADHOC=1
		break
		;;
	-*)
		echo "unknown option: $1" >&2
		usage
		;;
	*) REQUESTED+=("$1") ;;
	esac
	shift
done
if [[ -n "$DISPOSE" ]]; then
	[[ ${#REQUESTED[@]} -eq 0 && $KEEP -eq 0 && $SEED -eq 0 ]] || usage
	dispose_kept "$DISPOSE"
	exit 0
fi
[[ ${#REQUESTED[@]} -gt 0 ]] || usage

# --- run --------------------------------------------------------------------------

WORK="$(mktemp -d "${TMPDIR:-/tmp}/endpoint-run.XXXXXX")"
trap cleanup EXIT
trap 'echo; echo "interrupted — cleaning up"; exit 130' INT TERM

# The runner first: it is what answers which services exist and what each one's
# tests are. Its tag is this run's own, from the scratch directory's unique name,
# and it is built under the lock, so that its build cache is attributable to this
# run and no other invocation's cleanup can remove it (#142): with one shared tag,
# a run that finished removed the runner of a run still waiting for the lock,
# which then failed at `docker run` with exit 125. Every run rebuilds it, which
# takes seconds when pyproject.toml and uv.lock have not changed, and is what
# keeps the image from going stale when they have.
RUNNER_IMAGE="localdata-test/runner:run-${WORK##*.}"
take_lock runner
echo "building the runner image ($RUNNER_IMAGE)..."
while read -r base; do pull_if_absent "$base"; done < <(bases_of "$RUNNER_DOCKERFILE")
RUNNER_PULLED=("${PULLED[@]}")
PULLED=()
owned_build docker build -q -f "$RUNNER_DOCKERFILE" -t "$RUNNER_IMAGE" "$REPO" >/dev/null
release_lock

SERVICES=()
for name in "${REQUESTED[@]}"; do
	if [[ "$name" == all ]]; then
		mapfile -t runnable < <(endpoints_say 'print("\n".join(endpoints.runnable_services()))')
		SERVICES+=("${runnable[@]}")
		continue
	fi
	[[ "$name" == "$PREFIX"* ]] || name="$PREFIX$name"
	SERVICES+=("$name")
done

if [[ $KEEP -eq 1 && ${#SERVICES[@]} -ne 1 ]]; then
	echo "--keep leaves a database up, so it takes exactly one service, not ${#SERVICES[@]}:" >&2
	echo "one container at a time is the rule on this machine." >&2
	exit 2
fi

worst=0
SUMMARY=()

remaining=${#SERVICES[@]}
for service in "${SERVICES[@]}"; do
	remaining=$((remaining - 1))
	short="${service#"$PREFIX"}"
	echo
	echo "=============================================================="
	echo "$service"
	echo "=============================================================="

	selection="$(endpoints_say "print(endpoints.selection('$service'))")"
	if [[ -z "$selection" ]]; then
		echo "  $service is a helper with no tests of its own; it comes up with the" >&2
		echo "  database that depends on it. Skipped." >&2
		SUMMARY+=("$short: not run (helper)")
		continue
	fi

	take_lock "$service"
	CURRENT_PROJECT="ldrun-$short"

	# Bases of this service and of every helper it depends on: `config` on one
	# service includes what it depends on, and each has a build context.
	mapfile -t contexts < <(compose config --format json "$service" |
		grep -o '"context": *"[^"]*"' | cut -d'"' -f4)
	for context in "${contexts[@]}"; do
		while read -r base; do pull_if_absent "$base"; done < <(bases_of "$context/Dockerfile")
	done

	echo "  building..."
	owned_build compose build -q --with-dependencies "$service"
	echo "  starting under project $CURRENT_PROJECT..."
	compose up -d "$service"

	rc=0
	log="$WORK/$short.log"
	if wait_for_health "$service"; then
		if [[ $SEED -eq 1 ]]; then
			echo "  seeding the standard assets..."
			in_network python scripts/seed_endpoint.py "$service" || rc=1
		fi
		if [[ $rc -eq 0 && $ADHOC -eq 1 ]]; then
			run_tests "$log" "${PYTEST_ARGS[@]}" || rc=$?
		elif [[ $rc -eq 0 && $SEED -eq 0 ]]; then
			# -rs lists every skip with its reason: a skipped test reads like a
			# passing one in a summary line, so the reasons are printed with it.
			run_tests "$log" -q -rs -m endpoint -k "$selection" || rc=$?
		fi
	else
		rc=1
	fi

	[[ $rc -gt $worst ]] && worst=$rc
	counts="$(grep -E '^=* *[0-9]+ (passed|failed|skipped|error|deselected)' "$log" 2>/dev/null | tail -1 | tr -d '=' | xargs || true)"
	SUMMARY+=("$short: $([[ $rc -eq 0 ]] && echo PASS || echo "FAIL (exit $rc)")${counts:+ — $counts}")

	if [[ $KEEP -eq 1 ]]; then
		save_keep_state
		print_keep_instructions "$service" "$short"
	else
		teardown
		# After the last service the lock is held on into the cleanup, which
		# disposes of the runner under it.
		((remaining == 0)) || release_lock
	fi
done

echo
echo "=============================================================="
printf '%s\n' "${SUMMARY[@]}"
echo
echo "Engines NOT covered by this run are every one not listed above:"
echo "a green line for one service says nothing about another (issue #46)."
echo "=============================================================="

exit "$worst"
