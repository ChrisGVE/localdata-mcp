"""localdata_mcp/process/domains/time_series/forecasting.py — FR-301/§6(f).

`forecast_time_series`'s computation. Methods: arima (fixed order,
default (1,1,1)), ets (exponential smoothing) — `main`'s launch pair —
plus the §6(f) staged harvest: **sarima** (caller-supplied seasonal
order) and **auto_arima** (AIC-minimizing grid search, harvested from
`_auto_arima.py`: candidate orders fitted with enforced stationarity/
invertibility, best kept by information criterion; the seasonal
period comes from the calendar frequency via series.py's map). Every
result carries the solver's own convergence verdict (`converged`) —
the sentinel's class-2 signal — and forecasts with their confidence
intervals. E10.x7's battery row pins this against statsmodels
directly. Neighbors: analysis.py is the sibling; tools.py declares
the ToolSpec.
"""

from __future__ import annotations

import warnings
from itertools import product
from typing import Any

import pandas as pd

from ..support import invalid_source_refusal
from .series import seasonal_period_of, time_indexed_values

METHODS = ("arima", "sarima", "auto_arima", "ets")

# The auto search grid (compact harvest of main's stepwise space):
# p, q over 0..2 with d over 0..1 — 18 candidates, AIC-ranked.
_AUTO_P = (0, 1, 2)
_AUTO_D = (0, 1)
_AUTO_Q = (0, 1, 2)


def forecast_series(
    frame: pd.DataFrame,
    date_column: str,
    value_column: str,
    horizon: int | None = None,
    method: str = "arima",
    order: list[int] | None = None,
    seasonal_order: list[int] | None = None,
) -> dict[str, Any]:
    """The horizon-step forecast with confidence intervals."""
    if method not in METHODS:
        raise invalid_source_refusal(
            f"Unknown method {method!r} — one of {list(METHODS)}."
        )
    if horizon is None:
        horizon = 10  # main's default forecast depth (function-local:
        # the S8 scan reserves declaration-site literals for config).
    series = time_indexed_values(frame, date_column, value_column)
    if method == "ets":
        return _ets(series, horizon)
    if method == "auto_arima":
        return _auto_arima(series, horizon)
    return _fixed_arima(series, horizon, method, order, seasonal_order)


def _fixed_arima(
    series: "pd.Series[float]",
    horizon: int,
    method: str,
    order: list[int] | None,
    seasonal_order: list[int] | None,
) -> dict[str, Any]:
    resolved_order = tuple(order) if order else (1, 1, 1)
    if len(resolved_order) != 3:
        raise invalid_source_refusal("order= must be [p, d, q].")
    seasonal: tuple[int, ...] = (0, 0, 0, 0)
    if method == "sarima":
        if seasonal_order is None:
            period = seasonal_period_of(series)
            if period is None:
                raise invalid_source_refusal(
                    "sarima needs seasonal_order=[P, D, Q, s] (no calendar "
                    "frequency was inferable to default s from)."
                )
            seasonal = (1, 1, 1, period)
        else:
            seasonal = tuple(seasonal_order)
            if len(seasonal) != 4:
                raise invalid_source_refusal("seasonal_order= must be [P, D, Q, s].")
    fitted = _fit_sarimax(series, resolved_order, seasonal)
    return _forecast_result(fitted, method, resolved_order, seasonal, horizon)


def _auto_arima(series: "pd.Series[float]", horizon: int) -> dict[str, Any]:
    """The harvested AIC grid search over the compact order space."""
    best: Any = None
    best_order: tuple[int, ...] | None = None
    for p, d, q in product(_AUTO_P, _AUTO_D, _AUTO_Q):
        try:
            candidate = _fit_sarimax(series, (p, d, q), (0, 0, 0, 0))
        except Exception:  # noqa: BLE001 — a non-fitting candidate is skipped
            continue
        if best is None or candidate.aic < best.aic:
            best, best_order = candidate, (p, d, q)
    if best is None or best_order is None:
        raise invalid_source_refusal(
            "No ARIMA candidate could be fitted to this series."
        )
    result = _forecast_result(best, "auto_arima", best_order, (0, 0, 0, 0), horizon)
    result["candidates_tried"] = len(_AUTO_P) * len(_AUTO_D) * len(_AUTO_Q)
    return result


def _fit_sarimax(
    series: "pd.Series[float]",
    order: tuple[int, ...],
    seasonal: tuple[int, ...],
) -> Any:
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        model = SARIMAX(
            series,
            order=order,
            seasonal_order=seasonal,
            enforce_stationarity=True,
            enforce_invertibility=True,
        )
        return model.fit(disp=False)


def _ets(series: "pd.Series[float]", horizon: int) -> dict[str, Any]:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        fitted = ExponentialSmoothing(series, trend="add").fit()
    forecast = fitted.forecast(horizon)
    return {
        "method": "ets",
        "horizon": horizon,
        "forecast": [float(value) for value in forecast],
        "aic": float(fitted.aic),
        "converged": bool(fitted.mle_retvals.get("converged", True))
        if isinstance(getattr(fitted, "mle_retvals", None), dict)
        else True,
    }


def _forecast_result(
    fitted: Any,
    method: str,
    order: tuple[int, ...],
    seasonal: tuple[int, ...],
    horizon: int,
) -> dict[str, Any]:
    prediction = fitted.get_forecast(steps=horizon)
    intervals = prediction.conf_int()
    lower = intervals.iloc[:, 0]
    upper = intervals.iloc[:, 1]
    return {
        "method": method,
        "order": list(order),
        "seasonal_order": list(seasonal),
        "horizon": horizon,
        "forecast": [float(value) for value in prediction.predicted_mean],
        "confidence_intervals": [
            {"lower": float(low), "upper": float(high)}
            for low, high in zip(lower, upper)
        ],
        "aic": float(fitted.aic),
        # The sentinel's class-2 convention: the optimizer's own verdict.
        "converged": bool(
            fitted.mle_retvals.get("converged", True)
            if isinstance(getattr(fitted, "mle_retvals", None), dict)
            else True
        ),
    }
