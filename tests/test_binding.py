"""Binding tests, written the way `docs/CONSTRAINTS.md` §5.1 requires.

Every assertion here is on **what a query returns** — `sum`, `avg`, `max`,
`ORDER BY`, an equality join. None is on `typeof` alone, and none is merely
"no exception was raised".

That rule is not stylistic. The silent-BLOB class this file guards against
binds cleanly, commits, passes a `typeof` check, and then answers `sum()` with
`0.0`. A test asserting the insert succeeded would pass against completely
corrupt storage.

**These run through the real load path**, not against a `sqlite3` connection
with adapters registered on it. That distinction is the point of the change
they were rewritten for: the conversions used to live in a driver-global
registry, so a test could exercise the registry while the pipeline quietly did
something else, and no other backend was protected at all. Going through
:meth:`Workspace.insert_frame` means what is asserted is what a caller gets.
"""

from __future__ import annotations

import datetime as _dt
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from localdata_mcp import binding
from localdata_mcp.loader import LoadError, Workspace

TAG = "bulk"


@pytest.fixture()
def workspace():
    space = Workspace.in_memory()
    space.attach_memory(TAG)
    yield space
    space.close()


def stored(workspace, values, *, dtype=None) -> Workspace:
    """Load a one-column frame the way any file would be loaded."""
    frame = pd.DataFrame({"v": pd.Series(values, dtype=dtype)})
    workspace.insert_frame(frame, "t", source="test", tag=TAG)
    return workspace


def answer(workspace, sql: str):
    _, rows = workspace.query(TAG, sql)
    return rows[0][0]


def column(workspace, sql: str) -> list:
    _, rows = workspace.query(TAG, sql)
    return [row[0] for row in rows]


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
def test_numpy_scalars_aggregate_correctly(workspace, values, expected_sum):
    """Un-adapted these store as BLOB and `sum()` returns 0.0."""
    stored(workspace, values)
    assert answer(workspace, "SELECT sum(v) FROM t") == expected_sum
    assert answer(workspace, "SELECT count(*) FROM t WHERE typeof(v)='blob'") == 0


@pytest.mark.parametrize(
    "values,expected_sum",
    [
        ([np.int64(1), np.int64(2), np.int64(3)], 6),
        ([np.bool_(True), np.bool_(False)], 1),
        ([np.float32(1.5), np.float32(2.5)], 4.0),
    ],
)
def test_numpy_scalars_survive_an_object_column_too(workspace, values, expected_sum):
    """The per-value path, which is what an ``object`` column actually takes.

    A typed column is converted with numpy in one pass; a column pandas could
    not type is converted one value at a time. Both have to arrive at the same
    place, and only this parametrisation says so.
    """
    stored(workspace, values, dtype=object)
    assert answer(workspace, "SELECT sum(v) FROM t") == expected_sum
    assert answer(workspace, "SELECT count(*) FROM t WHERE typeof(v)='blob'") == 0


def test_extension_dtype_column_aggregates_correctly(workspace):
    """Non-null `Int64` yields `np.int64` scalars — the measured blob case."""
    stored(workspace, [1, 2, 3], dtype="Int64")
    assert answer(workspace, "SELECT sum(v) FROM t") == 6
    assert answer(workspace, "SELECT avg(v) FROM t") == 2.0


def test_numpy_datetime64_in_object_column_is_not_a_blob(workspace):
    """The fourth silent-blob case: `np.datetime64` inside an object column."""
    values = [np.datetime64("2024-01-01"), np.datetime64("2024-01-02")]
    stored(workspace, values, dtype=object)
    assert answer(workspace, "SELECT count(*) FROM t WHERE typeof(v)='blob'") == 0
    ordered = column(workspace, "SELECT v FROM t ORDER BY v")
    assert ordered[1] - ordered[0] == 86_400 * 1_000_000_000


# ---------------------------------------------------------------------------
# Temporals: the aggregates ISO-8601 text gets wrong
# ---------------------------------------------------------------------------


def test_durations_aggregate_and_order_correctly(workspace):
    """The exact case ISO-8601 text fails: sum/avg 0.0, max wrong, order wrong."""
    values = [pd.Timedelta(days=2), pd.Timedelta(days=10), pd.Timedelta(hours=1)]
    stored(workspace, values)

    truth_ns = sum(v.value for v in values)
    assert answer(workspace, "SELECT sum(v) FROM t") == truth_ns
    assert answer(workspace, "SELECT avg(v) FROM t") == truth_ns / 3
    # Stored as text this returns 'P2DT0H0M0S' against a truth of 10 days.
    assert answer(workspace, "SELECT max(v) FROM t") == pd.Timedelta(days=10).value
    # Stored as text this orders 1 h, 10 d, 2 d.
    assert column(workspace, "SELECT v FROM t ORDER BY v") == sorted(
        v.value for v in values
    )


def test_same_instant_in_two_offsets_joins(workspace):
    """Offset-preserving text joins zero rows here; epoch ticks join one."""
    utc = pd.Timestamp("2024-11-03 06:30", tz="UTC")
    local = utc.tz_convert("America/New_York")

    workspace.insert_frame(pd.DataFrame({"v": [utc]}), "left_t", source="test", tag=TAG)
    workspace.insert_frame(
        pd.DataFrame({"v": [local]}), "right_t", source="test", tag=TAG
    )

    matched = answer(
        workspace,
        "SELECT count(*) FROM left_t JOIN right_t ON left_t.v = right_t.v",
    )
    assert matched == 1


def test_naive_timestamps_round_trip_exactly(workspace):
    value = pd.Timestamp("2024-01-01 12:00:00.123456789")
    stored(workspace, [value])
    assert pd.Timestamp(answer(workspace, "SELECT v FROM t")) == value


def test_bare_time_binds_and_sums(workspace):
    """Un-adapted, `datetime.time` raises — an ordinary spreadsheet is refused."""
    stored(workspace, [_dt.time(1, 0, 0), _dt.time(2, 0, 0)], dtype=object)
    assert answer(workspace, "SELECT sum(v) FROM t") == 3 * 3600 * 1_000_000_000


def test_bare_date_stores_midnight_utc(workspace):
    stored(workspace, [_dt.date(1970, 1, 2)], dtype=object)
    assert answer(workspace, "SELECT v FROM t") == 86_400 * 1_000_000_000


# ---------------------------------------------------------------------------
# Missing values
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("missing", [pd.NA, pd.NaT, np.float64("nan"), None])
def test_missing_markers_become_sql_null(workspace, missing):
    """NaN bound through as a float makes `WHERE v IS NULL` miss every gap."""
    stored(workspace, [missing], dtype=object)
    assert answer(workspace, "SELECT count(*) FROM t WHERE v IS NULL") == 1


def test_nulls_are_excluded_from_averages(workspace):
    stored(workspace, [1.0, float("nan"), 3.0])
    assert answer(workspace, "SELECT avg(v) FROM t") == 2.0


def test_a_missing_value_in_a_typed_column_is_null_not_a_sentinel(workspace):
    """The typed path converts before it masks, and NaT views as int64's floor.

    Without the mask restoring the null afterwards, a missing timestamp would
    store as -9223372036854775808 and every aggregate over the column would be
    wrong in a way nothing reports.
    """
    stored(workspace, pd.to_datetime(["2024-01-01", None, "2024-01-03"]))
    assert answer(workspace, "SELECT count(*) FROM t WHERE v IS NULL") == 1
    assert answer(workspace, "SELECT min(v) FROM t") == pd.Timestamp("2024-01-01").value


@pytest.mark.parametrize("dtype", ["Int64", "boolean", "Float64"])
def test_nullable_extension_dtypes_keep_their_gaps(workspace, dtype):
    stored(workspace, [1, None, 1], dtype=dtype)
    assert answer(workspace, "SELECT count(*) FROM t WHERE v IS NULL") == 1
    assert answer(workspace, "SELECT sum(v) FROM t") == 2


# ---------------------------------------------------------------------------
# Values that must be refused rather than stored
# ---------------------------------------------------------------------------


def test_uint64_above_signed_range_is_refused(workspace):
    """Measured: this stores a silent blob and `sum()` returns 0.0.

    Surfaces as a ``LoadError`` because the loader catches the refusal and
    re-raises it naming the source file; the refusal itself is still in the
    chain, which is what carries the value and the reason.
    """
    with pytest.raises(LoadError, match="exceeds the signed INTEGER range"):
        stored(workspace, [np.uint64(2**64 - 1)], dtype="uint64")


def test_uint64_within_range_still_works(workspace):
    stored(workspace, [np.uint64(2**63 - 1)], dtype="uint64")
    assert answer(workspace, "SELECT v FROM t") == 2**63 - 1


def test_python_int_wider_than_64_bits_is_refused_by_name(workspace):
    """Not adaptable — a 64-bit signed column is a 64-bit signed column.

    Refused here rather than left to the driver, which raises an OverflowError
    naming neither the column nor the value.
    """
    with pytest.raises(LoadError, match="wider than the 64-bit"):
        stored(workspace, [2**70], dtype=object)


def test_a_refused_value_is_reported_with_its_source(workspace):
    """The refusal has to reach the caller as their value, not as a statement."""
    frame = pd.DataFrame({"v": pd.Series([2**70], dtype=object)})
    with pytest.raises(LoadError, match="prices.csv"):
        workspace.insert_frame(frame, "t", source="prices.csv", tag=TAG)


# ---------------------------------------------------------------------------
# Decimal is lossy on purpose
# ---------------------------------------------------------------------------


def test_decimal_stores_as_real_and_sums(workspace):
    stored(workspace, [Decimal("1.50"), Decimal("2.25")], dtype=object)
    assert answer(workspace, "SELECT sum(v) FROM t") == pytest.approx(3.75)


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


def test_nothing_is_registered_with_a_driver():
    """The property the rewrite exists to hold.

    A process-global adapter registry made a load's correctness depend on
    something no backend but one could see. If this module ever reaches for it
    again, every other backend silently loses the protections above.
    """
    assert not hasattr(binding, "sqlite3")
    assert "sqlite3" not in vars(binding)
