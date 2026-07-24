#!/usr/bin/env python3
"""Collect-and-build the DB fixtures for the batteries (NFR-504, E14.2).

Companion to build_oracle_datasets.py — the database side of the fixture
provisioning. The relational batteries self-seed their own tables through any
reachable endpoint, so this script's job is to bring the backends up, wait for
them to accept connections (the readiness gate the batteries would otherwise
race), and — for the extras engines that have no battery row of their own —
load-and-verify a probe fixture that proves the connector round-trips data.

Backends (registry in testbench.fixtures.db_fixtures):
  * core (per-PR):  PostgreSQL, MySQL — drivers ship in the base install;
    provisioned in CI as GitHub Actions `services:`, and locally via the
    compose file below.
  * extras (nightly):  MSSQL, Oracle — drivers ship in the `mssql` /
    `enterprise` extras; fixture-tested via --load once brought up.

SQLite / DuckDB are file-based and the kv/graph/tree stores are embedded
SQLite, so they need no container and are not in the registry.

Modes (a backend takes part only when its LOCALDATA_TEST_*_DSN is set):
  --up [--tier core|extras|all]     docker compose up the tier's services
  --wait [--timeout N] [--tier …]   poll every discovered backend until ready
                                    (the default); exit 1 on timeout
  --load [--tier …]                 wait, then load-and-verify the probe
                                    fixture on every discovered backend
  --down                            docker compose down

DSNs are read from the environment (same vars the batteries read), never
invented here — CI or the developer points them at the running containers.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from localdata_mcp.testbench.fixtures.db_fixtures import (  # noqa: E402
    TIER_CORE,
    TIER_EXTRAS,
    Backend,
    discover,
    load_and_verify,
    wait_until_ready,
)

COMPOSE_FILE = _REPO_ROOT / "docker-compose.test.yml"

# Backend name → compose service. Kept here (not in the pure module) because it
# is a fact about this repo's compose file, not about the fixture logic.
_COMPOSE_SERVICE = {
    "postgresql": "localdata-test-postgres",
    "mysql": "localdata-test-mysql",
    "mssql": "localdata-test-mssql",
    "oracle": "localdata-test-oracle",
}

WAIT_TIMEOUT_S = 120.0
WAIT_INTERVAL_S = 3.0


def _engine(dsn: str) -> Any:
    from sqlalchemy import create_engine

    return create_engine(dsn, pool_pre_ping=True)


def _compose(*args: str) -> int:
    return subprocess.run(
        ["docker", "compose", "-f", str(COMPOSE_FILE), *args],
        check=False,
    ).returncode


def _tier_arg(tier: str) -> str | None:
    return None if tier == "all" else tier


def _up(tier: str) -> int:
    services = [
        _COMPOSE_SERVICE[backend.name] for backend in _registry(_tier_arg(tier))
    ]
    print(f"Bringing up: {', '.join(services)}")
    return _compose("up", "-d", *services)


def _down() -> int:
    print("Tearing down compose fixtures")
    return _compose("down")


def _registry(tier: str | None) -> tuple[Backend, ...]:
    from localdata_mcp.testbench.fixtures.db_fixtures import backends_for_tier

    return backends_for_tier(tier)


def _wait(tier: str, timeout_s: float) -> int:
    found = discover(os.environ, _tier_arg(tier))
    if not found:
        print("No fixture DSNs set — nothing to wait for.")
        return 0
    failed = False
    for backend, dsn in found:
        engine = _engine(dsn)
        print(f"Waiting for {backend.name} (up to {timeout_s:.0f}s)...")
        ready = wait_until_ready(
            engine.connect,
            timeout_s=timeout_s,
            interval_s=WAIT_INTERVAL_S,
            now=time.monotonic,
            sleep=time.sleep,
        )
        if ready:
            print(f"  {backend.name}: ready")
        else:
            print(f"  {backend.name}: TIMED OUT")
            failed = True
    return 1 if failed else 0


def _load(tier: str, timeout_s: float) -> int:
    if _wait(tier, timeout_s) != 0:
        return 1
    for backend, dsn in discover(os.environ, _tier_arg(tier)):
        print(f"Loading + verifying probe fixture on {backend.name}...")
        engine = _engine(dsn)
        with engine.connect() as connection:
            load_and_verify(
                lambda sql: connection.exec_driver_sql(sql),
                lambda sql: connection.exec_driver_sql(sql).fetchall(),
            )
            connection.commit()
        print(f"  {backend.name}: fixture round-trips")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--up", action="store_true", help="docker compose up the tier.")
    mode.add_argument("--down", action="store_true", help="docker compose down.")
    mode.add_argument(
        "--wait",
        action="store_true",
        help="Poll every discovered backend until ready (default).",
    )
    mode.add_argument(
        "--load",
        action="store_true",
        help="Wait, then load-and-verify the probe fixture on each backend.",
    )
    parser.add_argument(
        "--tier",
        choices=(TIER_CORE, TIER_EXTRAS, "all"),
        default="all",
        help="Which backends to act on (default: all).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=WAIT_TIMEOUT_S,
        help="Readiness timeout in seconds.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.up:
        sys.exit(_up(args.tier))
    if args.down:
        sys.exit(_down())
    if args.load:
        sys.exit(_load(args.tier, args.timeout))
    sys.exit(_wait(args.tier, args.timeout))


if __name__ == "__main__":
    main()
