"""Binding tests, written the way `docs/CONSTRAINTS.md` §5.1 requires.

Every assertion here is on **what a query returns** — `sum`, `avg`, `max`,
`ORDER BY`, an equality join. None is on `typeof` alone, and none is merely
"no exception was raised".

That rule is not stylistic. The silent-BLOB class this file guards against
binds cleanly, commits, passes a `typeof` check, and then answers `sum()` with
`0.0`. A test asserting the insert succeeded would pass against completely
corrupt storage.
"""

from __future__ import annotations

import datetime as _dt
import sqlite3
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from localdata_mcp import binding

binding.install()


@pytest.fixture()
def conn():
    connection = sqlite3.connect(":memory:")
    yield connection
    connection.close()


def _roundtrip(conn, values, declared="INTEGER"):
    """Insert values one per row and hand back a cursor over the column."""
    conn.execute(f"CREATE TABLE t (v {declared})")
    conn.executemany("INSERT INTO t (v) VALUES (?)", ((v,) for v in values))
    return conn


# ---------------------------------------------------------------------------
# The silent-BLOB class: binds fine, answers wrong
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "values,expected_sum",
    [
        ([np.int64(1), np.int64(2), np.int64(3)], 6),
        ([np.int32(1), np.int32(2), np.int32(3)], 6),
        ([np.uint32(1), np.uint32(2), np.uint32(3)], 6),
        ([np.bool_(True), np.bool_(True), np.bool_(False)], 2),
    ],
)
def test_numpy_scalars_aggregate_correctly(conn, values, expected_sum):
    """Un-adapted these store as BLOB and `sum()` returns 0.0."""
    _roundtrip(conn, values)
    assert conn.execute("SELECT sum(v) FROM t").fetchone()[0] == expected_sum
    assert (
        conn.execute("SELECT count(*) FROM t WHERE typeof(v)='blob'").fetchone()[0] == 0
    )


def test_extension_dtype_column_aggregates_correctly(conn):
    """Non-null `Int64` yields `np.int64` scalars — the measured blob case."""
    series = pd.array([1, 2, 3], dtype="Int64")
    _roundtrip(conn, list(series))
    assert conn.execute("SELECT sum(v) FROM t").fetchone()[0] == 6
    assert conn.execute("SELECT avg(v) FROM t").fetchone()[0] == 2.0


def test_numpy_datetime64_in_object_column_is_not_a_blob(conn):
    """The fourth silent-blob case: `np.datetime64` inside an object column."""
    values = [np.datetime64("2024-01-01"), np.datetime64("2024-01-02")]
    _roundtrip(conn, values)
    assert (
        conn.execute("SELECT count(*) FROM t WHERE typeof(v)='blob'").fetchone()[0] == 0
    )
    stored = [r[0] for r in conn.execute("SELECT v FROM t ORDER BY v")]
    assert stored[1] - stored[0] == 86_400 * 1_000_000_000


# ---------------------------------------------------------------------------
# Temporals: the aggregates ISO-8601 text gets wrong
# ---------------------------------------------------------------------------


def test_durations_aggregate_and_order_correctly(conn):
    """The exact case ISO-8601 text fails: sum/avg 0.0, max wrong, order wrong."""
    values = [pd.Timedelta(days=2), pd.Timedelta(days=10), pd.Timedelta(hours=1)]
    _roundtrip(conn, values)

    truth_ns = sum(v.value for v in values)
    assert conn.execute("SELECT sum(v) FROM t").fetchone()[0] == truth_ns
    assert conn.execute("SELECT avg(v) FROM t").fetchone()[0] == truth_ns / 3
    # Stored as text this returns 'P2DT0H0M0S' against a truth of 10 days.
    assert (
        conn.execute("SELECT max(v) FROM t").fetchone()[0]
        == pd.Timedelta(days=10).value
    )
    # Stored as text this orders 1 h, 10 d, 2 d.
    ordered = [r[0] for r in conn.execute("SELECT v FROM t ORDER BY v")]
    assert ordered == sorted(v.value for v in values)


def test_same_instant_in_two_offsets_joins(conn):
    """Offset-preserving text joins zero rows here; epoch ticks join one."""
    utc = pd.Timestamp("2024-11-03 06:30", tz="UTC")
    local = utc.tz_convert("America/New_York")

    conn.execute("CREATE TABLE left_t (v INTEGER)")
    conn.execute("CREATE TABLE right_t (v INTEGER)")
    conn.execute("INSERT INTO left_t VALUES (?)", (utc,))
    conn.execute("INSERT INTO right_t VALUES (?)", (local,))

    matched = conn.execute(
        "SELECT count(*) FROM left_t JOIN right_t ON left_t.v = right_t.v"
    ).fetchone()[0]
    assert matched == 1


def test_naive_timestamps_round_trip_exactly(conn):
    values = [pd.Timestamp("2024-01-01 12:00:00.123456789")]
    _roundtrip(conn, values)
    stored = conn.execute("SELECT v FROM t").fetchone()[0]
    assert pd.Timestamp(stored) == values[0]


def test_bare_time_binds_and_sums(conn):
    """Un-adapted, `datetime.time` raises — an ordinary spreadsheet is refused."""
    values = [_dt.time(1, 0, 0), _dt.time(2, 0, 0)]
    _roundtrip(conn, values)
    total = conn.execute("SELECT sum(v) FROM t").fetchone()[0]
    assert total == 3 * 3600 * 1_000_000_000


def test_bare_date_stores_midnight_utc(conn):
    _roundtrip(conn, [_dt.date(1970, 1, 2)])
    assert conn.execute("SELECT v FROM t").fetchone()[0] == 86_400 * 1_000_000_000


# ---------------------------------------------------------------------------
# Missing values
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("missing", [pd.NA, pd.NaT, np.float64("nan")])
def test_missing_markers_become_sql_null(conn, missing):
    """NaN bound through as a float makes `WHERE v IS NULL` miss every gap."""
    _roundtrip(conn, [missing])
    assert conn.execute("SELECT count(*) FROM t WHERE v IS NULL").fetchone()[0] == 1


def test_nulls_are_excluded_from_averages(conn):
    _roundtrip(conn, [np.float64(1.0), np.float64("nan"), np.float64(3.0)])
    assert conn.execute("SELECT avg(v) FROM t").fetchone()[0] == 2.0


# ---------------------------------------------------------------------------
# Values that must be refused rather than stored
# ---------------------------------------------------------------------------


def test_uint64_above_signed_range_is_refused(conn):
    """Measured: this stores a silent blob and `sum()` returns 0.0."""
    conn.execute("CREATE TABLE t (v INTEGER)")
    with pytest.raises(binding.UnrepresentableValue):
        conn.execute("INSERT INTO t VALUES (?)", (np.uint64(2**64 - 1),))


def test_uint64_within_range_still_works(conn):
    _roundtrip(conn, [np.uint64(2**63 - 1)])
    assert conn.execute("SELECT v FROM t").fetchone()[0] == 2**63 - 1


def test_python_int_wider_than_64_bits_raises(conn):
    """Not adaptable — SQLite INTEGER is 64-bit signed, full stop."""
    conn.execute("CREATE TABLE t (v INTEGER)")
    with pytest.raises(OverflowError):
        conn.execute("INSERT INTO t VALUES (?)", (2**70,))


# ---------------------------------------------------------------------------
# Decimal is lossy on purpose
# ---------------------------------------------------------------------------


def test_decimal_stores_as_real_and_sums(conn):
    _roundtrip(conn, [Decimal("1.50"), Decimal("2.25")], declared="REAL")
    assert conn.execute("SELECT sum(v) FROM t").fetchone()[0] == pytest.approx(3.75)


# ---------------------------------------------------------------------------
# The declared contract
# ---------------------------------------------------------------------------


def test_temporal_kind_names_the_semantics():
    assert (
        binding.temporal_kind(pd.Series(pd.to_datetime(["2024-01-01"])).dtype)
        == "timestamp"
    )
    assert binding.temporal_kind(pd.Series([pd.Timedelta(days=1)]).dtype) == "duration"
    assert binding.temporal_kind(pd.Series([1, 2, 3]).dtype) is None


def test_install_is_idempotent():
    binding.install()
    binding.install()
