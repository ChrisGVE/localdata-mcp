"""tests/v3/testbench/conftest.py — shared fixtures for results-store tests.

Builds canonical stores and per-worker files in pytest tmp dirs. Every
configurable value (busy timeout, retry attempts, retention) is passed
explicitly — the results store deliberately has no module-level defaults;
their configuration home arrives with the config nexus in E1.
"""

from pathlib import Path
from sqlite3 import Connection
from typing import Any, Iterator, List, Optional

import pytest

from localdata_mcp.testbench.results_store import schema, store

# Explicit test values (NOT defaults of the code under test).
BUSY_TIMEOUT_MS = 250
MAX_WRITE_ATTEMPTS = 3


def make_run(
    index: int = 0,
    battery_name: str = "statistical",
    run_mode: str = "deterministic",
    seed: Optional[int] = 42,
) -> store.BatteryRun:
    """A synthetic battery run; `index` orders runs in time and names them."""
    return store.BatteryRun(
        run_id=f"run-{battery_name}-{run_mode}-{index:04d}",
        battery_name=battery_name,
        run_mode=run_mode,
        seed=seed,
        dataset_hash="sha256:fixture-set-aabbcc",
        software_versions={"python": "3.12.9", "localdata-mcp": "3.0.0.dev0"},
        started_at=f"2026-07-22T10:{index // 60:02d}:{index % 60:02d}Z",
        finished_at=f"2026-07-22T11:{index // 60:02d}:{index % 60:02d}Z",
        git_sha="0123abc",
    )


def make_results(
    run_id: str, count: int = 3, numeric_output: Any = None
) -> List[store.BatteryResult]:
    """Synthetic per-test results for one run."""
    return [
        store.BatteryResult(
            run_id=run_id,
            test_id=f"statistical.sleep.t_test.case_{i}",
            passed=(i % 2 == 0),
            numeric_output=numeric_output or {"statistic": 1.5 + i},
            duration_ms=12.5 + i,
        )
        for i in range(count)
    ]


def open_store(db_path: Path) -> Connection:
    """Connect with the explicit test busy-timeout and migrate to current."""
    connection = schema.connect(db_path, busy_timeout_ms=BUSY_TIMEOUT_MS)
    schema.ensure_schema(connection)
    return connection


def make_worker_file(
    db_path: Path, run: store.BatteryRun, results: List[store.BatteryResult]
) -> Path:
    """A per-worker results file: run row (coordinator-minted id) + results."""
    connection = open_store(db_path)
    with store.write_transaction(connection, max_attempts=MAX_WRITE_ATTEMPTS):
        store.write_run(connection, run)
        store.write_results(connection, results)
    connection.close()
    return db_path


@pytest.fixture()
def canonical(tmp_path: Path) -> Iterator[Connection]:
    """A canonical store at the current schema version, closed after the test."""
    connection = open_store(tmp_path / "battery_results.db")
    yield connection
    connection.close()
