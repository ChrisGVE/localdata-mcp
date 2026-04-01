#!/usr/bin/env python3
"""Automated integration test runner with Docker lifecycle management.

Handles: start Docker services -> wait for health -> run pytest -> report -> teardown.
Teardown always runs, even on failure or SIGINT/SIGTERM.
"""

from __future__ import annotations

import argparse
import signal
import subprocess
import sys
import time

PROJECT_ROOT = subprocess.check_output(
    ["git", "rev-parse", "--show-toplevel"], text=True
).strip()

COMPOSE_FILE = f"{PROJECT_ROOT}/docker-compose.test.yml"
TESTS_DIR = f"{PROJECT_ROOT}/tests/integration"

# Map pytest marker names to docker-compose service names.
SERVICE_MAP: dict[str, str] = {
    "postgres": "localdata-test-postgres",
    "mysql": "localdata-test-mysql",
    "mssql": "localdata-test-mssql",
    "oracle": "localdata-test-oracle",
    "mongodb": "localdata-test-mongodb",
    "redis": "localdata-test-redis",
    "elasticsearch": "localdata-test-elasticsearch",
    "influxdb": "localdata-test-influxdb",
    "neo4j": "localdata-test-neo4j",
    "couchdb": "localdata-test-couchdb",
}

ALL_SERVICES = list(SERVICE_MAP.keys())

# Global flag so signal handler can request teardown.
_teardown_requested = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run integration tests with automatic Docker lifecycle management.",
    )
    parser.add_argument(
        "--services",
        default=",".join(ALL_SERVICES),
        help="Comma-separated services to start (default: all). "
        f"Options: {','.join(ALL_SERVICES)}",
    )
    parser.add_argument(
        "--markers",
        default=None,
        help="Comma-separated pytest markers to run (e.g. postgres,mongodb). "
        "Defaults to matching --services.",
    )
    parser.add_argument(
        "--keep-up",
        action="store_true",
        help="Don't tear down Docker after tests (for debugging).",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Pass -v to pytest.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Max seconds to wait for health checks (default: 300).",
    )
    return parser.parse_args()


def compose_cmd(*args: str) -> list[str]:
    return ["docker", "compose", "-f", COMPOSE_FILE, *args]


def run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    print(f"  $ {' '.join(cmd)}")
    return subprocess.run(cmd, text=True, capture_output=True, check=check)


def docker_up(services: list[str]) -> None:
    compose_services = [SERVICE_MAP[s] for s in services]
    print(f"\n==> Starting Docker services: {', '.join(services)}")
    result = run(compose_cmd("up", "-d", *compose_services), check=False)
    if result.returncode != 0:
        print(result.stderr)
        sys.exit(1)


def docker_down() -> None:
    print("\n==> Tearing down Docker services")
    run(compose_cmd("down", "--volumes", "--remove-orphans"), check=False)


def wait_for_healthy(services: list[str], timeout: int) -> None:
    """Poll docker compose ps until all requested services report healthy."""
    compose_services = set(SERVICE_MAP[s] for s in services)
    print(f"\n==> Waiting for services to become healthy (timeout: {timeout}s)")

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = run(
            compose_cmd("ps", "--format", "{{.Name}} {{.Status}}"),
            check=False,
        )
        if result.returncode != 0:
            time.sleep(2)
            continue

        healthy_containers: list[str] = []
        for line in result.stdout.strip().splitlines():
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                name, status = parts
                if "(healthy)" in status.lower():
                    healthy_containers.append(name)

        # Match service names against container names (container names
        # include a project prefix and instance suffix).
        pending = {
            svc
            for svc in compose_services
            if not any(svc in cname for cname in healthy_containers)
        }
        if not pending:
            print("  All services healthy.")
            return

        remaining = int(deadline - time.monotonic())
        pending_short = [
            k for k, v in SERVICE_MAP.items() if v in pending and k in services
        ]
        print(
            f"  Waiting for: {', '.join(sorted(pending_short))} "
            f"({remaining}s remaining)"
        )
        time.sleep(5)

    print("\nERROR: Timed out waiting for services to become healthy.")
    # Show current status for debugging.
    run(compose_cmd("ps"), check=False)
    sys.exit(1)


def run_pytest(markers: list[str], verbose: bool) -> int:
    """Run pytest and return the exit code."""
    marker_expr = " or ".join(markers)
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        TESTS_DIR,
        "-m",
        marker_expr,
        "--tb=short",
        "-q",
    ]
    if verbose:
        cmd.append("-v")

    print(f"\n==> Running pytest with markers: {marker_expr}")
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode


def print_summary(exit_code: int) -> None:
    print("\n" + "=" * 60)
    if exit_code == 0:
        print("  RESULT: All integration tests passed.")
    else:
        print(f"  RESULT: Tests failed (pytest exit code {exit_code}).")
        print(
            "  Re-run with --verbose for details, or --keep-up to inspect containers."
        )
    print("=" * 60)


def main() -> None:
    args = parse_args()

    services = [s.strip() for s in args.services.split(",") if s.strip()]
    unknown = set(services) - set(ALL_SERVICES)
    if unknown:
        print(f"ERROR: Unknown services: {', '.join(sorted(unknown))}")
        print(f"Available: {', '.join(ALL_SERVICES)}")
        sys.exit(1)

    markers = (
        [m.strip() for m in args.markers.split(",") if m.strip()]
        if args.markers
        else services
    )

    # Install signal handlers so teardown always runs.
    def _signal_handler(signum: int, _frame: object) -> None:
        global _teardown_requested
        sig_name = signal.Signals(signum).name
        print(f"\n==> Caught {sig_name}, initiating teardown...")
        _teardown_requested = True
        if not args.keep_up:
            docker_down()
        sys.exit(128 + signum)

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    pytest_exit = 1
    try:
        docker_up(services)
        wait_for_healthy(services, args.timeout)
        pytest_exit = run_pytest(markers, args.verbose)
    except SystemExit:
        raise
    except Exception as exc:
        print(f"\nERROR: {exc}")
    finally:
        if not args.keep_up and not _teardown_requested:
            docker_down()
        elif args.keep_up:
            print("\n==> --keep-up: Docker services left running.")
            print(f"  Tear down manually: docker compose -f {COMPOSE_FILE} down -v")

    print_summary(pytest_exit)
    sys.exit(pytest_exit)


if __name__ == "__main__":
    main()
