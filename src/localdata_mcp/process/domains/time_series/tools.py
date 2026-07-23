"""localdata_mcp/process/domains/time_series/tools.py — E10.d ToolSpecs.

The time-series family's two tools, carried by name from `main`
(DR GP2): `analyze_time_series` (trend/stationarity/autocorrelation/
seasonality) and `forecast_time_series` (arima, ets, plus the §6(f)
harvested sarima and auto_arima). Thin over analysis.py /
forecasting.py with the X-2 addressing contract. Neighbors:
series.py preps; spec_modules.py rosters this module.
"""

from __future__ import annotations

from typing import Any

from localdata_mcp.nexus.contract.spec import Param, TypeShape, tool_spec

from ..support import addressed_frame, source_params
from .analysis import analyze_series
from .forecasting import forecast_series

_DATE = Param("date_column", str, "The timestamp column.")
_VALUE = Param("value_column", str, "The numeric value column.")


@tool_spec(
    name="analyze_time_series",
    summary=(
        "Analyze a time series on an addressed tabular source: trend "
        "direction and slope, ADF stationarity, autocorrelation with "
        "significant lags, and calendar seasonality strength."
    ),
    params=(
        *source_params(),
        _DATE,
        _VALUE,
        Param(
            "frequency",
            str,
            "Pandas frequency alias to align the series on (e.g. 'D', 'MS').",
            required=False,
        ),
    ),
    input_shape=TypeShape.TABULAR,
    output_shape=TypeShape.SCALAR,
    domain="time_series",
)
def analyze_time_series(
    date_column: str,
    value_column: str,
    endpoint: str | None = None,
    path: str | None = None,
    table: str | None = None,
    query: str | None = None,
    **knobs: Any,
) -> Any:
    frame, source = addressed_frame(endpoint, path, table, query)
    result = analyze_series(frame, date_column, value_column, **knobs)
    result["source"] = source
    return result


@tool_spec(
    name="forecast_time_series",
    summary=(
        "Forecast a time series on an addressed tabular source: method "
        "arima (default, order= [p,d,q]), sarima (seasonal_order= "
        "[P,D,Q,s], defaulted from the calendar frequency), auto_arima "
        "(AIC grid search), or ets. Returns the forecast with "
        "confidence intervals and the solver's convergence verdict."
    ),
    params=(
        *source_params(),
        _DATE,
        _VALUE,
        Param(
            "horizon",
            int,
            "Steps ahead to forecast (implementation default 10).",
            required=False,
        ),
        Param(
            "method",
            str,
            "arima (default), sarima, auto_arima, or ets.",
            required=False,
        ),
        Param(
            "order", list, "ARIMA order [p, d, q] (default [1,1,1]).", required=False
        ),
        Param(
            "seasonal_order",
            list,
            "Seasonal order [P, D, Q, s] for sarima.",
            required=False,
        ),
    ),
    input_shape=TypeShape.TABULAR,
    output_shape=TypeShape.VECTOR,
    domain="time_series",
)
def forecast_time_series(
    date_column: str,
    value_column: str,
    endpoint: str | None = None,
    path: str | None = None,
    table: str | None = None,
    query: str | None = None,
    **knobs: Any,
) -> Any:
    frame, source = addressed_frame(endpoint, path, table, query)
    result = forecast_series(frame, date_column, value_column, **knobs)
    result["source"] = source
    return result
