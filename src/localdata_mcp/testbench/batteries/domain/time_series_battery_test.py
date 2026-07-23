"""testbench/batteries/domain/time_series_battery_test.py — E10.d slice.

The NFR-502c domain battery's time-series rows: FR-301 L3 coverage,
the FR-304/NFR-505 dual-assertion oracle, and E10.x7's harvested
auto-ARIMA/SARIMA checks against statsmodels directly. Published
fixture: the Box & Jenkins airline passengers series (R's
`AirPassengers`, monthly 1949-1960) — its documented properties are
the published leg: strongly increasing trend, annual seasonality
(period 12), and ADF non-stationarity in levels. Reference legs
refit the SAME statsmodels model on the test side and compare at the
iterative tolerance (S8 row 15b); the auto-ARIMA row re-runs the
identical AIC grid and asserts the same winning order.
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import Any, Iterator

import anyio
import numpy as np
import pandas as pd
import pytest
from fastmcp import Client

import localdata_mcp.ingest.runtime as runtime
from localdata_mcp.nexus.chokepoint.guard import Chokepoint
from localdata_mcp.nexus.config.models import ConfigModel, SecurityConfig
from localdata_mcp.nexus.contract.registry import default_registry
from localdata_mcp.nexus.response.shaping import configure_shaping
from localdata_mcp.server.mcp_app import app

# R `AirPassengers`: monthly international airline passengers
# (thousands), Jan 1949 - Dec 1960 (Box & Jenkins 1976).
_AIR_PASSENGERS = (
    112,
    118,
    132,
    129,
    121,
    135,
    148,
    148,
    136,
    119,
    104,
    118,
    115,
    126,
    141,
    135,
    125,
    149,
    170,
    170,
    158,
    133,
    114,
    140,
    145,
    150,
    178,
    163,
    172,
    178,
    199,
    199,
    184,
    162,
    146,
    166,
    171,
    180,
    193,
    181,
    183,
    218,
    230,
    242,
    209,
    191,
    172,
    194,
    196,
    196,
    236,
    235,
    229,
    243,
    264,
    272,
    237,
    211,
    180,
    201,
    204,
    188,
    235,
    227,
    234,
    264,
    302,
    293,
    259,
    229,
    203,
    229,
    242,
    233,
    267,
    269,
    270,
    315,
    364,
    347,
    312,
    274,
    237,
    278,
    284,
    277,
    317,
    313,
    318,
    374,
    413,
    405,
    355,
    306,
    271,
    306,
    315,
    301,
    356,
    348,
    355,
    422,
    465,
    467,
    404,
    347,
    305,
    336,
    340,
    318,
    362,
    348,
    363,
    435,
    491,
    505,
    404,
    359,
    310,
    337,
    360,
    342,
    406,
    396,
    420,
    472,
    548,
    559,
    463,
    407,
    362,
    405,
    417,
    391,
    419,
    461,
    472,
    535,
    622,
    606,
    508,
    461,
    390,
    432,
)

_SEED = 42


@pytest.fixture()
def bench(tmp_path: Path) -> Iterator[Path]:
    config = ConfigModel(
        security=SecurityConfig(allowed_paths=(str(tmp_path),)),
    )
    guard = Chokepoint.boot(config, environ=dict(os.environ))
    configure_shaping(config, default_registry())
    runtime.configure_ingest(guard)
    yield tmp_path
    runtime._CHOKEPOINT = None
    configure_shaping(ConfigModel(), default_registry())
    guard.shutdown()


def _call(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    async def session() -> dict[str, Any]:
        async with Client(app) as client:
            result = await client.call_tool(name, arguments)
            assert not result.is_error
            if isinstance(result.structured_content, dict) and (
                "inline" in result.structured_content
            ):
                return result.structured_content
            payload = json.loads(result.content[0].text)
            assert isinstance(payload, dict)
            return payload

    return anyio.run(session)


def _data(envelope: dict[str, Any]) -> Any:
    assert envelope["error"] is None, envelope["error"]
    return envelope["data"]


def _air_series() -> "pd.Series[float]":
    index = pd.date_range("1949-01-01", periods=len(_AIR_PASSENGERS), freq="MS")
    return pd.Series([float(v) for v in _AIR_PASSENGERS], index=index)


def _air_csv(tmp_path: Path) -> str:
    series = _air_series()
    target = tmp_path / "air_passengers.csv"
    months = [str(stamp.date()) for stamp in series.index]
    pd.DataFrame({"month": months, "passengers": series.values}).to_csv(
        target, index=False
    )
    return str(target)


def _ar1_csv(tmp_path: Path) -> tuple[str, "pd.Series[float]"]:
    """A seeded AR(1) series for the auto-ARIMA grid row."""
    rng = np.random.default_rng(_SEED)
    values = [0.0]
    for _ in range(59):
        values.append(0.7 * values[-1] + rng.normal())
    index = pd.date_range("2020-01-01", periods=60, freq="D")
    series = pd.Series(values, index=index)
    target = tmp_path / "ar1.csv"
    days = [str(stamp.date()) for stamp in index]
    pd.DataFrame({"day": days, "value": values}).to_csv(target, index=False)
    return str(target), series


def test_analysis_reads_the_published_airline_properties(bench: Path) -> None:
    """analyze_time_series — the published Box-Jenkins facts + an ADF
    reference-leg recompute (FR-304)."""
    from statsmodels.tsa.stattools import adfuller

    data = _data(
        _call(
            "analyze_time_series",
            {
                "path": _air_csv(bench),
                "date_column": "month",
                "value_column": "passengers",
            },
        )
    )
    assert data["n_observations"] == 144
    assert data["trend"]["direction"] == "increasing"
    assert data["seasonality"]["period"] == 12
    assert data["seasonality"]["detected"] is True
    # Published: the raw series is non-stationary in levels.
    assert data["stationarity"]["is_stationary"] is False
    reference_stat, reference_p, *_rest = adfuller(_air_series().to_numpy())
    config = ConfigModel()
    rtol = config.testbench.tol_closed_form_rtol
    assert data["stationarity"]["statistic"] == pytest.approx(reference_stat, rel=rtol)
    assert data["stationarity"]["p_value"] == pytest.approx(reference_p, rel=rtol)
    assert data["autocorrelation"]["lag_1"] > 0.9


def test_sarima_matches_the_statsmodels_reference(bench: Path) -> None:
    """E10.x7 SARIMA leg: the tool's seasonal fit equals the SAME
    statsmodels model refitted test-side (NFR-505, S8 row 15b)."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    data = _data(
        _call(
            "forecast_time_series",
            {
                "path": _air_csv(bench),
                "date_column": "month",
                "value_column": "passengers",
                "method": "sarima",
                "horizon": 12,
            },
        )
    )
    assert data["seasonal_order"] == [1, 1, 1, 12]
    assert data["converged"] is True
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        reference = SARIMAX(
            _air_series(),
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 12),
            enforce_stationarity=True,
            enforce_invertibility=True,
        ).fit(disp=False)
    expected = reference.get_forecast(steps=12).predicted_mean
    config = ConfigModel()
    rtol = config.testbench.tol_iterative_rtol
    assert data["aic"] == pytest.approx(reference.aic, rel=rtol)
    assert data["forecast"] == pytest.approx(list(expected), rel=rtol)
    assert len(data["confidence_intervals"]) == 12


def test_auto_arima_selects_the_grid_aic_winner(bench: Path) -> None:
    """E10.x7 auto-ARIMA leg: the tool's grid pick equals the same
    grid re-run test-side with statsmodels (order + AIC agreement)."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    path, series = _ar1_csv(bench)
    data = _data(
        _call(
            "forecast_time_series",
            {
                "path": path,
                "date_column": "day",
                "value_column": "value",
                "method": "auto_arima",
                "horizon": 5,
            },
        )
    )
    assert data["candidates_tried"] == 18
    best_aic, best_order = np.inf, None
    for p in (0, 1, 2):
        for d in (0, 1):
            for q in (0, 1, 2):
                try:
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore")
                        fitted = SARIMAX(
                            series,
                            order=(p, d, q),
                            enforce_stationarity=True,
                            enforce_invertibility=True,
                        ).fit(disp=False)
                except Exception:  # noqa: BLE001 — mirror the tool's skip rule
                    continue
                if fitted.aic < best_aic:
                    best_aic, best_order = fitted.aic, [p, d, q]
    assert data["order"] == best_order
    config = ConfigModel()
    assert data["aic"] == pytest.approx(
        best_aic, rel=config.testbench.tol_iterative_rtol
    )


def test_ets_forecast_matches_the_statsmodels_reference(bench: Path) -> None:
    """forecast_time_series ets — reference-library leg."""
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    path, series = _ar1_csv(bench)
    data = _data(
        _call(
            "forecast_time_series",
            {
                "path": path,
                "date_column": "day",
                "value_column": "value",
                "method": "ets",
                "horizon": 5,
            },
        )
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        reference = ExponentialSmoothing(series, trend="add").fit()
    config = ConfigModel()
    rtol = config.testbench.tol_iterative_rtol
    assert data["forecast"] == pytest.approx(list(reference.forecast(5)), rel=rtol)
