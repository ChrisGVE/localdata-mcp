"""testbench/batteries/domain/regression_battery_test.py — E10.b slice.

The NFR-502c domain battery's regression rows: FR-301 L3 coverage and
the FR-304/NFR-505 dual-assertion oracle. Published fixture: R's
`mtcars` — the documented `lm(mpg ~ wt)` fit (intercept 37.285, slope
-5.3445, R² 0.7528), compared at its published rounding precision.
Reference legs recompute with sklearn directly (seed-free: OLS and
ridge on this fixture are deterministic closed-form fits). The ridge
row drives `algorithm_params` through the wire — the FR-306 concern
exercised at the L3 seam (E10.x1 adds the clone()-survival unit leg).
"""

from __future__ import annotations

import json
import os
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

# R `mtcars`: weight (1000 lbs) and fuel economy (mpg), 32 cars.
_MTCARS_WT = (
    2.620,
    2.875,
    2.320,
    3.215,
    3.440,
    3.460,
    3.570,
    3.190,
    3.150,
    3.440,
    3.440,
    4.070,
    3.730,
    3.780,
    5.250,
    5.424,
    5.345,
    2.200,
    1.615,
    1.835,
    2.465,
    3.520,
    3.435,
    3.840,
    3.845,
    1.935,
    2.140,
    1.513,
    3.170,
    2.770,
    3.570,
    2.780,
)
_MTCARS_MPG = (
    21.0,
    21.0,
    22.8,
    21.4,
    18.7,
    18.1,
    14.3,
    24.4,
    22.8,
    19.2,
    17.8,
    16.4,
    17.3,
    15.2,
    10.4,
    10.4,
    14.7,
    32.4,
    30.4,
    33.9,
    21.5,
    15.5,
    15.2,
    13.3,
    19.2,
    27.3,
    26.0,
    30.4,
    15.8,
    19.7,
    15.0,
    21.4,
)
# Published verdict (R: summary(lm(mpg ~ wt, data = mtcars))).
_PUBLISHED_INTERCEPT = 37.2851
_PUBLISHED_SLOPE = -5.3445
_PUBLISHED_R2 = 0.7528
_PUBLISHED_RTOL = 1e-3


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


def _mtcars_csv(tmp_path: Path) -> str:
    target = tmp_path / "mtcars.csv"
    pd.DataFrame({"wt": _MTCARS_WT, "mpg": _MTCARS_MPG}).to_csv(target, index=False)
    return str(target)


def test_linear_fit_matches_reference_and_published(bench: Path) -> None:
    """analyze_regression OLS — both oracle legs (FR-304)."""
    from sklearn.linear_model import LinearRegression

    config = ConfigModel()
    data = _data(
        _call(
            "analyze_regression",
            {"path": _mtcars_csv(bench), "target_column": "mpg"},
        )
    )
    reference = LinearRegression().fit(
        np.asarray(_MTCARS_WT).reshape(-1, 1), _MTCARS_MPG
    )
    rtol = config.testbench.tol_closed_form_rtol
    assert data["coefficients"]["wt"] == pytest.approx(reference.coef_[0], rel=rtol)
    assert data["intercept"] == pytest.approx(reference.intercept_, rel=rtol)
    assert data["intercept"] == pytest.approx(_PUBLISHED_INTERCEPT, rel=_PUBLISHED_RTOL)
    assert data["coefficients"]["wt"] == pytest.approx(
        _PUBLISHED_SLOPE, rel=_PUBLISHED_RTOL
    )
    assert data["metrics"]["r2"] == pytest.approx(_PUBLISHED_R2, rel=_PUBLISHED_RTOL)
    # The sentinel's class-4 keys ride along on every fit.
    assert data["rank"] == data["design_columns"] == 2
    assert data["condition_number"] > 1.0


def test_ridge_algorithm_params_reach_the_estimator(bench: Path) -> None:
    """FR-306 at the wire: algorithm_params={'alpha': 0.5} produces the
    Ridge(alpha=0.5) fit, not the default-alpha one."""
    from sklearn.linear_model import Ridge

    config = ConfigModel()
    data = _data(
        _call(
            "analyze_regression",
            {
                "path": _mtcars_csv(bench),
                "target_column": "mpg",
                "model_type": "ridge",
                "algorithm_params": {"alpha": 0.5},
            },
        )
    )
    features = np.asarray(_MTCARS_WT).reshape(-1, 1)
    tuned = Ridge(alpha=0.5).fit(features, _MTCARS_MPG)
    untuned = Ridge().fit(features, _MTCARS_MPG)
    rtol = config.testbench.tol_closed_form_rtol
    assert data["coefficients"]["wt"] == pytest.approx(tuned.coef_[0], rel=rtol)
    assert data["coefficients"]["wt"] != pytest.approx(untuned.coef_[0], rel=rtol)


def test_logistic_fit_reports_convergence(bench: Path) -> None:
    """The class-2 convergence flag rides along on iterative fits."""
    target = bench / "binary.csv"
    rng = np.random.default_rng(42)
    x = rng.normal(size=80)
    y = (x + rng.normal(scale=0.5, size=80) > 0).astype(int)
    pd.DataFrame({"x": x, "label": y}).to_csv(target, index=False)
    data = _data(
        _call(
            "analyze_regression",
            {
                "path": str(target),
                "target_column": "label",
                "model_type": "logistic",
            },
        )
    )
    assert data["converged"] is True
    assert 0.5 < data["metrics"]["accuracy"] <= 1.0


def test_evaluate_model_performance_matches_sklearn(bench: Path) -> None:
    """evaluate_model_performance — reference-library oracle leg."""
    from sklearn.metrics import mean_absolute_error, r2_score

    target = bench / "predictions.csv"
    actual = list(_MTCARS_MPG)
    predicted = [
        _PUBLISHED_INTERCEPT + _PUBLISHED_SLOPE * weight for weight in _MTCARS_WT
    ]
    pd.DataFrame({"actual": actual, "predicted": predicted}).to_csv(target, index=False)
    data = _data(
        _call(
            "evaluate_model_performance",
            {
                "path": str(target),
                "target_column": "actual",
                "prediction_column": "predicted",
            },
        )
    )
    config = ConfigModel()
    rtol = config.testbench.tol_closed_form_rtol
    assert data["metrics"]["r2"] == pytest.approx(r2_score(actual, predicted), rel=rtol)
    assert data["metrics"]["mae"] == pytest.approx(
        mean_absolute_error(actual, predicted), rel=rtol
    )
    assert data["n_samples"] == 32
