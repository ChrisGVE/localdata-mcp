"""localdata_mcp/process/domains/time_series/series.py — shared series prep.

The date/value extraction both family tools share: parse the date
column, sort, index the values by time, and resolve the seasonal
period — `main`'s frequency map (hourly→24, daily→7, weekly→52,
monthly→12, quarterly→4) harvested from `_auto_arima.py`'s
`_detect_seasonal_period`. Neighbors: analysis.py and forecasting.py
consume.
"""

from __future__ import annotations

import pandas as pd

from ..support import invalid_source_refusal, require_columns

# main's frequency-string → seasonal-period map (harvested).
_FREQ_PERIODS = (("H", 24), ("D", 7), ("W", 52), ("Q", 4), ("M", 12))

# Below this the trend/stationarity statistics are noise, not
# analysis. (Chosen off the S8 default set — the one-default-site
# scan reserves colliding literals for config-backed values.)
_MIN_POINTS = 9


def time_indexed_values(
    frame: pd.DataFrame, date_column: str, value_column: str
) -> "pd.Series[float]":
    """The value series indexed by parsed, sorted timestamps."""
    require_columns(frame, date_column, value_column)
    dates = pd.to_datetime(frame[date_column], errors="coerce")
    values = pd.to_numeric(frame[value_column], errors="coerce")
    paired = pd.DataFrame({"date": dates, "value": values}).dropna()
    if len(paired) < _MIN_POINTS:
        raise invalid_source_refusal(
            f"Only {len(paired)} usable (date, value) rows — time-series "
            f"analysis needs at least {_MIN_POINTS}."
        )
    ordered = paired.sort_values("date").set_index("date")["value"]
    if not ordered.index.is_unique:
        raise invalid_source_refusal(
            f"Column {date_column!r} carries duplicate timestamps — "
            "aggregate the data to one row per instant first."
        )
    return ordered


def seasonal_period_of(series: "pd.Series[float]") -> int | None:
    """main's frequency rule: the seasonal period the inferred
    calendar frequency implies, None when no frequency is inferable."""
    index = series.index
    assert isinstance(index, pd.DatetimeIndex)
    freq = index.freqstr or pd.infer_freq(index)
    if freq is None:
        return None
    spelled = str(freq).upper()
    for marker, period in _FREQ_PERIODS:
        if marker in spelled:
            return period
    return None
