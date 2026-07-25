"""Value binding between pandas/numpy and SQLite.

**Read `docs/CONSTRAINTS.md` §1 before changing anything in this file.** Every
adapter below exists because the un-adapted value was measured doing something
worse than failing: binding cleanly, committing, and then answering `sum()` with
`0.0`.

The short version:

* `numpy` scalars support the buffer protocol, so SQLite stores them as **BLOB**
  rather than raising. `np.int64`, `np.bool_` and `np.datetime64` all land as
  blobs, and a declared `INTEGER` affinity does not rescue them — storage class
  outranks affinity. `np.float64` is the sole accident that works, because it
  subclasses Python `float`.
* Temporals stored as ISO-8601 text bind fine and are wrong on every aggregate:
  `sum` and `avg` return `0.0`, `max` picks the lexically-largest string, and
  `ORDER BY` sorts `1 h, 10 d, 2 d`. We store **INTEGER ticks** instead.
* Two value classes cannot be represented at all and must be refused on their
  *value*, not their type: `np.uint64` above 2**63-1 (which stores a silent blob
  even with adapters registered, because it is genuinely out of range) and
  Python `int` wider than 64 bits.

Registration is **process-global** — `sqlite3.register_adapter` has no per-
connection scope. That is acceptable here because this package owns its process
(an MCP server speaking stdio), but it means a library embedding us would
inherit these adapters. `install()` is therefore explicit and idempotent rather
than an import side effect.
"""

from __future__ import annotations

import datetime as _dt
import sqlite3
from decimal import Decimal
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "EPOCH",
    "TICK_UNIT",
    "UnrepresentableValue",
    "install",
    "temporal_kind",
    "to_timestamp",
    "to_timedelta",
]

# ---------------------------------------------------------------------------
# The temporal contract
# ---------------------------------------------------------------------------

#: Temporal columns are stored as INTEGER ticks counted from this instant.
EPOCH = "1970-01-01T00:00:00Z"

#: The unit of one tick. Nanoseconds matches pandas' internal resolution, so a
#: `datetime64[ns]` column round-trips exactly.
TICK_UNIT = "nanosecond"

_NS_PER_SECOND = 1_000_000_000
_NS_PER_DAY = 86_400 * _NS_PER_SECOND

#: SQLite integers are 64-bit **signed**. This is the hard boundary behind the
#: two value-domain refusals.
_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1


class UnrepresentableValue(ValueError):
    """A value that SQLite cannot store without silently corrupting it.

    Raised from an adapter, so it surfaces mid-insert. The loader catches it and
    re-raises with the column and row number attached — on its own the message
    from a DBAPI layer names neither.
    """

    def __init__(self, value: Any, reason: str) -> None:
        self.value = value
        self.reason = reason
        super().__init__(f"{reason}: {value!r} ({type(value).__name__})")


# ---------------------------------------------------------------------------
# Temporal conversions
# ---------------------------------------------------------------------------


def to_timestamp(value: Any) -> int | None:
    """Convert an instant to integer nanoseconds since :data:`EPOCH`, in UTC.

    A timezone-aware value is converted to UTC first. **The original offset is
    not recoverable from the stored integer** — this is a deliberate, documented
    loss (`CONSTRAINTS.md` §1.4). Storing the offset instead loses the join: the
    same instant written in two offsets compares unequal as text and returns
    zero matching rows.
    """
    ts = pd.Timestamp(value)
    if ts is pd.NaT:
        return None
    if ts.tz is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return _check_range(ts.value, value)


def to_timedelta(value: Any) -> int | None:
    """Convert a duration to integer nanoseconds."""
    td = pd.Timedelta(value)
    if td is pd.NaT:
        return None
    return _check_range(td.value, value)


def _check_range(ticks: int, original: Any) -> int:
    if not _INT64_MIN <= ticks <= _INT64_MAX:
        raise UnrepresentableValue(
            original, "temporal value outside the 64-bit signed tick range"
        )
    return int(ticks)


def temporal_kind(dtype: Any) -> str | None:
    """Name the temporal semantics of a pandas dtype, or ``None`` if it has none.

    The result is recorded alongside the table so a reader can interpret the
    stored integers. Without it the ticks are indistinguishable from an ordinary
    integer column, which is the one real cost of the tick representation.
    """
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return "timestamp"
    if pd.api.types.is_timedelta64_dtype(dtype):
        return "duration"
    return None


# ---------------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------------


def _adapt_int(value: Any) -> int:
    return int(value)


def _adapt_uint64(value: Any) -> int:
    as_int = int(value)
    if as_int > _INT64_MAX:
        # Measured: this stores a silent BLOB and `sum()` returns 0.0. It is
        # genuinely unrepresentable — SQLite has no unsigned 64-bit integer — so
        # refusal is the only answer that does not lose data quietly.
        raise UnrepresentableValue(
            value, "unsigned 64-bit value exceeds SQLite's signed INTEGER range"
        )
    return as_int


def _adapt_float(value: Any) -> float | None:
    as_float = float(value)
    # NaN is pandas' missing marker for float columns; SQL's is NULL. Binding the
    # NaN through would make `WHERE col IS NULL` miss every missing value.
    if as_float != as_float:
        return None
    return as_float


def _adapt_bool(value: Any) -> int:
    return int(bool(value))


def _adapt_missing(_value: Any) -> None:
    return None


def _adapt_datetime64(value: Any) -> int | None:
    return to_timestamp(value)


def _adapt_timedelta64(value: Any) -> int | None:
    return to_timedelta(value)


def _adapt_date(value: _dt.date) -> int:
    """A bare date is stored as midnight UTC on that date.

    Chosen so a date column and a timestamp column compare and join directly.
    The distinction between them lives in the recorded column kind, not in the
    stored value.
    """
    days = value.toordinal() - _dt.date(1970, 1, 1).toordinal()
    return days * _NS_PER_DAY


def _adapt_time(value: _dt.time) -> int:
    """A bare time is stored as nanoseconds since midnight.

    Excel time cells, parquet ``time32``/``time64`` and TOML local times all land
    here. Un-adapted this type *raises*, so an ordinary spreadsheet was being
    refused outright.
    """
    return (
        value.hour * 3600 + value.minute * 60 + value.second
    ) * _NS_PER_SECOND + value.microsecond * 1000


def _adapt_decimal(value: Decimal) -> float:
    """Store a Decimal as REAL.

    SQLite has no exact-decimal type, so this is lossy for values beyond float64's
    53 bits of mantissa. Storing the text instead would be exact but would break
    every aggregate, which is a worse trade for a column that is almost always
    money or a measurement. The load report flags the column as lossy.
    """
    if value.is_nan():
        return float("nan")
    return float(value)


_installed = False


def install() -> None:
    """Register every adapter. Idempotent; safe to call from multiple entry points.

    Deliberately explicit rather than an import side effect — the registry is
    process-global and silently mutating it on import would be a surprise to any
    caller that embeds this package.
    """
    global _installed
    if _installed:
        return

    integer_types = [
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
    ]
    for np_type in integer_types:
        sqlite3.register_adapter(np_type, _adapt_int)

    # Separate: this is the one integer width that can leave the representable
    # range, so it carries a check the others do not need.
    sqlite3.register_adapter(np.uint64, _adapt_uint64)

    for np_type in (np.float16, np.float32, np.float64):
        sqlite3.register_adapter(np_type, _adapt_float)

    sqlite3.register_adapter(np.bool_, _adapt_bool)
    sqlite3.register_adapter(bool, _adapt_bool)

    # pandas' two missing markers. Un-adapted, both raise mid-insert.
    sqlite3.register_adapter(type(pd.NA), _adapt_missing)
    sqlite3.register_adapter(type(pd.NaT), _adapt_missing)

    sqlite3.register_adapter(pd.Timestamp, _adapt_datetime64)
    sqlite3.register_adapter(pd.Timedelta, _adapt_timedelta64)
    sqlite3.register_adapter(np.datetime64, _adapt_datetime64)
    sqlite3.register_adapter(np.timedelta64, _adapt_timedelta64)

    # `datetime` before `date`: datetime subclasses date, but the registry keys
    # on exact type, so both need their own entry.
    sqlite3.register_adapter(_dt.datetime, _adapt_datetime64)
    sqlite3.register_adapter(_dt.date, _adapt_date)
    sqlite3.register_adapter(_dt.time, _adapt_time)

    sqlite3.register_adapter(Decimal, _adapt_decimal)

    _installed = True
