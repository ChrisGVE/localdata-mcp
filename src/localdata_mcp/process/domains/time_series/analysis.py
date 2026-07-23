"""localdata_mcp/process/domains/time_series/analysis.py — FR-301.

`analyze_time_series`'s computation, re-authored from `main`'s
`BasicTimeSeriesAnalyzer`: the four analysis blocks kept by name —
trend (least-squares slope over the observation index), stationarity
(augmented Dickey-Fuller), autocorrelation (ACF with the 1.96/√n
significance band), and seasonality (decomposition strength when the
calendar frequency implies a period and the data covers two of them).
Neighbors: series.py preps the series; tools.py declares the
ToolSpec; forecasting.py is the sibling.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from .series import seasonal_period_of, time_indexed_values

# ACF depth: the conventional inspection window, capped by n/2.
_MAX_LAGS = 20

# Trend verdicts need a slope meaningfully away from zero relative to
# the series scale; this is a labeling threshold, not a test.
_FLAT_SHARE = 1e-3


def analyze_series(
    frame: pd.DataFrame,
    date_column: str,
    value_column: str,
    frequency: str | None = None,
) -> dict[str, Any]:
    """The four analysis blocks over the addressed series."""
    series = time_indexed_values(frame, date_column, value_column)
    if frequency is not None:
        series = series.asfreq(frequency)
        if series.isna().any():
            series = series.dropna()
    return {
        "n_observations": int(len(series)),
        "start": str(series.index[0]),
        "end": str(series.index[-1]),
        "value_summary": {
            "mean": float(series.mean()),
            "std": float(series.std(ddof=1)),
            "min": float(series.min()),
            "max": float(series.max()),
        },
        "trend": _trend_block(series),
        "stationarity": _stationarity_block(series),
        "autocorrelation": _autocorrelation_block(series),
        "seasonality": _seasonality_block(series),
    }


def _trend_block(series: "pd.Series[float]") -> dict[str, Any]:
    positions = np.arange(len(series), dtype=float)
    slope, intercept = np.polyfit(positions, series.to_numpy(dtype=float), 1)
    scale = float(series.abs().mean()) or 1.0
    if abs(slope) < _FLAT_SHARE * scale:
        direction = "flat"
    else:
        direction = "increasing" if slope > 0 else "decreasing"
    return {
        "slope_per_step": float(slope),
        "direction": direction,
    }


def _stationarity_block(series: "pd.Series[float]") -> dict[str, Any]:
    from statsmodels.tsa.stattools import adfuller

    statistic, p_value, _lags, _nobs, _critical, _icbest = adfuller(
        series.to_numpy(dtype=float)
    )
    return {
        "test": "adf",
        "statistic": float(statistic),
        "p_value": float(p_value),
        "is_stationary": bool(p_value < 0.05),
    }


def _autocorrelation_block(series: "pd.Series[float]") -> dict[str, Any]:
    from statsmodels.tsa.stattools import acf

    lags = min(_MAX_LAGS, len(series) // 2)
    values = acf(series.to_numpy(dtype=float), nlags=lags)
    band = 1.96 / math.sqrt(len(series))
    significant = [
        int(lag) for lag in range(1, lags + 1) if abs(float(values[lag])) > band
    ]
    return {
        "lag_1": float(values[1]),
        "significant_lags": significant,
        "significance_band": float(band),
    }


def _seasonality_block(series: "pd.Series[float]") -> dict[str, Any]:
    period = seasonal_period_of(series)
    if period is None or len(series) < 2 * period:
        return {"detected": False, "period": period}
    from statsmodels.tsa.seasonal import seasonal_decompose

    parts = seasonal_decompose(series, period=period)
    residual_var = float(np.nanvar(parts.resid))
    seasonal_var = float(np.nanvar(parts.seasonal + parts.resid))
    strength = max(0.0, 1.0 - residual_var / seasonal_var) if seasonal_var else 0.0
    return {
        "detected": bool(strength > 0.5),
        "period": period,
        "seasonal_strength": strength,
    }
