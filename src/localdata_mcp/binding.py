"""Value binding between pandas/numpy and whatever database is underneath.

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

**Nothing here registers anything with a driver.** These conversions used to be
`sqlite3.register_adapter` entries, which made the correctness of a load depend
on a process-global registry belonging to one driver — and left every other
backend unprotected against the very failures listed above, since a PostgreSQL
or DuckDB driver is handed the same numpy scalars. Values are now converted at
the frame boundary, before any driver sees them, so the protection is the same
whatever answers.
"""

from __future__ import annotations

import datetime as _dt
from decimal import Decimal
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "EPOCH",
    "TICK_UNIT",
    "UnrepresentableValue",
    "adapt_column",
    "adapt_value",
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


#: Value types that become ticks. Kept beside the dispatch table above, which is
#: what actually converts them, so the two cannot drift apart.
_INSTANTS = (_dt.date, _dt.datetime, _dt.time, pd.Timestamp, np.datetime64)
_DURATIONS = (pd.Timedelta, np.timedelta64)


def column_temporal_kind(values: pd.Series) -> str | None:
    """Name a *column's* temporal semantics, looking at values when it must.

    A dtype is not always enough, and the gap was a real defect rather than a
    nicety. A ``datetime.date`` or ``np.datetime64`` sitting in an ``object``
    column — which is what an Excel date cell and a CSV pandas declined to parse
    both produce — has dtype ``object``. It was therefore declared ``TEXT``,
    and SQLite's TEXT affinity stored the integer ticks *as a string*: ``max()``
    picked the lexically-largest and ``ORDER BY`` sorted wrong. That is the exact
    failure `CONSTRAINTS` §1.4 says the tick representation exists to prevent,
    reintroduced one layer up by the declaration.

    The scan exits at the first value that is not temporal, so an ordinary text
    column pays for one comparison.
    """
    by_dtype = temporal_kind(values.dtype)
    if by_dtype is not None:
        return by_dtype
    if values.dtype != object:
        return None

    kind: str | None = None
    for value in values:
        if value is None or value is pd.NA or value is pd.NaT:
            continue
        if isinstance(value, _DURATIONS):
            seen = "duration"
        elif isinstance(value, _INSTANTS):
            seen = "timestamp"
        else:
            return None
        # Instants and durations in one column are not a temporal column; they
        # are a mixed one, and mixing their ticks would compare epochs against
        # elapsed time.
        if kind is not None and kind != seen:
            return None
        kind = seen
    return kind


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


#: Exact type to the conversion it needs, keyed the way the driver registry used
#: to be: on ``type(value)`` rather than ``isinstance``. ``bool`` is here in its
#: own right because it subclasses ``int`` and would otherwise be stored as one,
#: and ``datetime`` is here separately from ``date`` for the same reason.
_BY_TYPE: dict[type, Any] = {
    np.int8: _adapt_int,
    np.int16: _adapt_int,
    np.int32: _adapt_int,
    np.int64: _adapt_int,
    np.uint8: _adapt_int,
    np.uint16: _adapt_int,
    np.uint32: _adapt_int,
    # The one integer width that can leave the representable range, so it
    # carries a check the others do not need.
    np.uint64: _adapt_uint64,
    np.float16: _adapt_float,
    np.float32: _adapt_float,
    np.float64: _adapt_float,
    np.bool_: _adapt_bool,
    bool: _adapt_bool,
    # pandas' two missing markers. Passed through, both reach the driver as
    # objects it has never heard of.
    type(pd.NA): _adapt_missing,
    type(pd.NaT): _adapt_missing,
    pd.Timestamp: _adapt_datetime64,
    pd.Timedelta: _adapt_timedelta64,
    np.datetime64: _adapt_datetime64,
    np.timedelta64: _adapt_timedelta64,
    _dt.datetime: _adapt_datetime64,
    _dt.date: _adapt_date,
    _dt.time: _adapt_time,
    Decimal: _adapt_decimal,
}


def adapt_value(value: Any) -> Any:
    """One value, converted to something any driver can bind.

    The fallback path, for a column whose dtype is ``object`` and whose values
    therefore have to be looked at one at a time. A column with a real dtype
    goes through :func:`adapt_column` instead and never arrives here.
    """
    handler = _BY_TYPE.get(type(value))
    if handler is not None:
        return handler(value)
    if value is None:
        return None
    if type(value) is int and not _INT64_MIN <= value <= _INT64_MAX:
        # Refused rather than passed on: it is wider than a 64-bit column can
        # hold, and every backend either truncates it or raises from inside the
        # driver, where the message names neither the column nor the value.
        raise UnrepresentableValue(value, "integer wider than the 64-bit range")
    if isinstance(value, np.generic):
        # A numpy scalar we did not name. ``item()`` is numpy's own answer for
        # "give me the Python equivalent", so this stays correct as numpy grows
        # types rather than silently storing the next one as a blob.
        return value.item()
    return value


def adapt_column(values: pd.Series) -> list[Any]:
    """A whole column, converted, in one pass rather than one value at a time.

    **This is where the driver-independence lives.** Every conversion below used
    to be a ``sqlite3.register_adapter`` entry, which meant two things that were
    both wrong: the correctness of a load depended on a *process-global* registry
    belonging to one driver, and any other backend — PostgreSQL, DuckDB — got no
    conversion at all and would have stored the same silent blobs that
    ``CONSTRAINTS`` §1 measured. Converting here, before a value is ever handed
    to a driver, is the same protection for every backend at once.

    Typed columns are converted with numpy rather than per value, so the common
    case never pays for the dispatch. ``object`` columns fall back to
    :func:`adapt_value`, which is the only case that genuinely needs to look at
    each value.
    """
    dtype = values.dtype
    missing = values.isna().to_numpy()

    if pd.api.types.is_datetime64_any_dtype(dtype):
        instants = values
        if getattr(dtype, "tz", None) is not None:
            # To UTC first: the offset is deliberately not preserved, because
            # the same instant written in two offsets must compare equal.
            instants = instants.dt.tz_convert("UTC").dt.tz_localize(None)
        converted = instants.to_numpy("datetime64[ns]").view("int64").tolist()
    elif pd.api.types.is_timedelta64_dtype(dtype):
        converted = values.to_numpy("timedelta64[ns]").view("int64").tolist()
    elif pd.api.types.is_bool_dtype(dtype):
        converted = values.to_numpy(dtype="int8", na_value=0).tolist()
    elif pd.api.types.is_integer_dtype(dtype):
        converted = _integers(values, dtype)
    elif pd.api.types.is_float_dtype(dtype):
        converted = values.to_numpy(dtype="float64", na_value=np.nan).tolist()
    else:
        return [adapt_value(value) for value in values]

    # One mask for every typed branch. ``isna`` is what recognises NaN, NaT,
    # None and ``pd.NA`` alike, and the null has to be restored *after* the
    # conversion because each of those becomes a sentinel number on the way
    # through — NaT views as the smallest int64 there is.
    return [None if absent else value for value, absent in zip(converted, missing)]


def _integers(values: pd.Series, dtype: Any) -> list[int]:
    """An integer column, refusing any value the 64-bit range cannot hold."""
    if not pd.api.types.is_signed_integer_dtype(dtype):
        present = values.dropna()
        if len(present) and int(present.max()) > _INT64_MAX:
            offending = present[present > _INT64_MAX].iloc[0]
            raise UnrepresentableValue(
                offending, "unsigned 64-bit value exceeds the signed INTEGER range"
            )
    return values.to_numpy(dtype="int64", na_value=0).tolist()
