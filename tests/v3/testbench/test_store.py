"""tests/v3/testbench/test_store.py — store.py round-trip and write wrapper.

Covers: L2 round-trip of a synthetic run (write via store.py, read back
equal), JSON field fidelity, the parameterized-write path refusing
duplicate results, and commit/rollback behavior of the bounded-retry
write transaction.
"""

import sqlite3
from sqlite3 import Connection

import pytest

from localdata_mcp.testbench.results_store import store

from .conftest import MAX_WRITE_ATTEMPTS, make_results, make_run


class TestRoundTrip:
    def test_run_round_trips(self, canonical: Connection) -> None:
        run = make_run()
        with store.write_transaction(canonical, max_attempts=MAX_WRITE_ATTEMPTS):
            store.write_run(canonical, run)
        assert store.read_run(canonical, run.run_id) == run

    def test_results_round_trip(self, canonical: Connection) -> None:
        run = make_run()
        results = make_results(run.run_id, count=4)
        with store.write_transaction(canonical, max_attempts=MAX_WRITE_ATTEMPTS):
            store.write_run(canonical, run)
            store.write_results(canonical, results)
        read_back = store.read_results(canonical, run.run_id)
        assert sorted(read_back, key=lambda r: r.test_id) == sorted(
            results, key=lambda r: r.test_id
        )

    def test_missing_run_reads_as_none(self, canonical: Connection) -> None:
        assert store.read_run(canonical, "absent") is None

    def test_null_numeric_output_round_trips(self, canonical: Connection) -> None:
        run = make_run()
        result = store.BatteryResult(
            run_id=run.run_id,
            test_id="statistical.sleep.t_test.smoke",
            passed=True,
            numeric_output=None,
            duration_ms=None,
        )
        with store.write_transaction(canonical, max_attempts=MAX_WRITE_ATTEMPTS):
            store.write_run(canonical, run)
            store.write_results(canonical, [result])
        assert store.read_results(canonical, run.run_id) == [result]


class TestWriteConstraints:
    def test_duplicate_result_key_is_refused(self, canonical: Connection) -> None:
        """(run_id, test_id) is unique; plain writes must not silently dedupe."""
        run = make_run()
        results = make_results(run.run_id, count=1)
        with store.write_transaction(canonical, max_attempts=MAX_WRITE_ATTEMPTS):
            store.write_run(canonical, run)
            store.write_results(canonical, results)
        with pytest.raises(sqlite3.IntegrityError):
            with store.write_transaction(canonical, max_attempts=MAX_WRITE_ATTEMPTS):
                store.write_results(canonical, results)


class TestWriteTransaction:
    def test_rolls_back_on_error(self, canonical: Connection) -> None:
        run = make_run()
        with pytest.raises(RuntimeError):
            with store.write_transaction(canonical, max_attempts=MAX_WRITE_ATTEMPTS):
                store.write_run(canonical, run)
                raise RuntimeError("mid-transaction failure")
        assert store.read_run(canonical, run.run_id) is None

    def test_retries_are_bounded(self, canonical: Connection) -> None:
        with pytest.raises(ValueError):
            with store.write_transaction(canonical, max_attempts=0):
                pass  # pragma: no cover - never entered
