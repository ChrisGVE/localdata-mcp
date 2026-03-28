#!/usr/bin/env bash
# wait-for-databases.sh — poll docker-compose.test.yml containers until healthy.
# Usage: ./scripts/wait-for-databases.sh [timeout_seconds]
#
# Exit 0 when all three containers report healthy, or exit 1 on timeout.

set -euo pipefail

TIMEOUT="${1:-120}"
INTERVAL=3
ELAPSED=0

SERVICES=("localdata-test-postgres" "localdata-test-mysql" "localdata-test-mssql")

echo "Waiting up to ${TIMEOUT}s for database containers to become healthy..."

while true; do
	all_healthy=true
	for svc in "${SERVICES[@]}"; do
		status=$(docker inspect --format='{{.State.Health.Status}}' "$svc" 2>/dev/null || echo "missing")
		if [ "$status" != "healthy" ]; then
			all_healthy=false
			break
		fi
	done

	if $all_healthy; then
		echo "All database containers are healthy."
		exit 0
	fi

	if [ "$ELAPSED" -ge "$TIMEOUT" ]; then
		echo "Timed out after ${TIMEOUT}s. Container status:"
		for svc in "${SERVICES[@]}"; do
			status=$(docker inspect --format='{{.State.Health.Status}}' "$svc" 2>/dev/null || echo "missing")
			echo "  $svc: $status"
		done
		exit 1
	fi

	sleep "$INTERVAL"
	ELAPSED=$((ELAPSED + INTERVAL))
done
