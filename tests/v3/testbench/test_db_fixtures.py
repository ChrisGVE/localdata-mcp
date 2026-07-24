"""tests/v3/testbench/test_db_fixtures.py — the pure NFR-504 fixture logic.

Covers the fixture-backend registry, DSN discovery from the environment, the
injected-clock readiness poll, and the load-and-verify probe (exercised on a
real in-memory SQLite engine, so the probe SQL itself is validated without a
container). The docker/process side lives in scripts/build_db_fixtures.py.
"""

from __future__ import annotations

from localdata_mcp.testbench.fixtures import db_fixtures as f


# --- registry + discovery --------------------------------------------------


def test_backends_for_tier_filters() -> None:
    core = {b.name for b in f.backends_for_tier(f.TIER_CORE)}
    extras = {b.name for b in f.backends_for_tier(f.TIER_EXTRAS)}
    assert core == {"postgresql", "mysql"}
    assert extras == {"mssql", "oracle"}
    assert f.backends_for_tier(None) == f.BACKENDS


def test_discover_returns_only_backends_with_a_dsn() -> None:
    environ = {"LOCALDATA_TEST_POSTGRES_DSN": "postgresql+psycopg2://x/db"}
    found = f.discover(environ)
    assert [b.name for b, _ in found] == ["postgresql"]
    assert found[0][1] == "postgresql+psycopg2://x/db"


def test_discover_ignores_blank_dsn_and_honours_tier() -> None:
    environ = {
        "LOCALDATA_TEST_POSTGRES_DSN": "postgresql+psycopg2://x/db",
        "LOCALDATA_TEST_MYSQL_DSN": "   ",
        "LOCALDATA_TEST_MSSQL_DSN": "mssql+pyodbc://x/db",
    }
    core = {b.name for b, _ in f.discover(environ, f.TIER_CORE)}
    extras = {b.name for b, _ in f.discover(environ, f.TIER_EXTRAS)}
    assert core == {"postgresql"}  # blank mysql dropped
    assert extras == {"mssql"}


# --- readiness poll (injected clock/sleep, no real DB) ---------------------


class _FakeConn:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def test_wait_becomes_ready_after_retries_and_closes_handle() -> None:
    attempts = {"n": 0}
    last: dict[str, _FakeConn] = {}

    def connect() -> object:
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise RuntimeError("not up yet")
        last["conn"] = _FakeConn()
        return last["conn"]

    clock = {"t": 0.0}
    ready = f.wait_until_ready(
        connect,
        timeout_s=100.0,
        interval_s=1.0,
        now=lambda: clock["t"],
        sleep=lambda s: clock.__setitem__("t", clock["t"] + s),
    )
    assert ready is True
    assert attempts["n"] == 3
    assert last["conn"].closed is True  # throwaway probe connection closed


def test_wait_times_out_when_never_ready() -> None:
    clock = {"t": 0.0}

    def connect() -> object:
        raise RuntimeError("down")

    ready = f.wait_until_ready(
        connect,
        timeout_s=5.0,
        interval_s=1.0,
        now=lambda: clock["t"],
        sleep=lambda s: clock.__setitem__("t", clock["t"] + s),
    )
    assert ready is False


# --- load-and-verify probe (real in-memory SQLite) -------------------------


def test_load_and_verify_round_trips_on_sqlite() -> None:
    from sqlalchemy import create_engine

    engine = create_engine("sqlite://")
    with engine.connect() as connection:
        f.load_and_verify(
            lambda sql: connection.exec_driver_sql(sql),
            lambda sql: connection.exec_driver_sql(sql).fetchall(),
        )
        # The probe drops its own table — nothing left behind.
        remaining = connection.exec_driver_sql(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        assert f.PROBE_TABLE not in {row[0] for row in remaining}


def test_load_and_verify_raises_on_bad_round_trip() -> None:
    import pytest

    def execute(_sql: str) -> None:
        return None

    def read(_sql: str) -> list[tuple[int, str]]:
        return [(9, "wrong")]

    with pytest.raises(f.FixtureVerificationError):
        f.load_and_verify(execute, read)
