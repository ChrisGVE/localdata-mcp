"""testbench/batteries/domain/sampling_battery_test.py — E10.g slice.

The NFR-502c domain battery's sampling/estimation rows: FR-301 L3
coverage and the FR-304/NFR-505 oracle. The Bayesian row is the
domain's published-statistic leg: under the noninformative conjugate
prior the posterior credible interval IS the classic Student-t
interval (Gelman et al.), pinned here against scipy's t distribution
at closed-form precision. The bootstrap row pins the seed and checks
the percentile interval against the SAME resampling algorithm re-run
test-side (identical rng stream) plus the analytic t-interval it
converges to. The S8 row-30/31 defaults are asserted to arrive
through the guard seam (omitted count → configured count reported).
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Iterator

import anyio
import numpy as np
import pandas as pd
import pytest
from fastmcp import Client
from scipy import stats

import localdata_mcp.ingest.runtime as runtime
from localdata_mcp.nexus.chokepoint.guard import Chokepoint
from localdata_mcp.nexus.config.models import ConfigModel, SecurityConfig
from localdata_mcp.nexus.contract.registry import default_registry
from localdata_mcp.nexus.response.shaping import configure_shaping
from localdata_mcp.server.mcp_app import app

# R `sleep` drug 2 (Cushny & Peebles): the fixture column.
_VALUES = (1.9, 0.8, 1.1, 0.1, -0.1, 4.4, 5.5, 1.6, 4.6, 3.4)

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


def _values_csv(tmp_path: Path) -> str:
    target = tmp_path / "sleep_drug2.csv"
    pd.DataFrame({"extra": _VALUES}).to_csv(target, index=False)
    return str(target)


def _people_csv(tmp_path: Path) -> str:
    target = tmp_path / "people.csv"
    frame = pd.DataFrame(
        {
            "region": ["north"] * 40 + ["south"] * 40 + ["west"] * 20,
            "score": list(range(100)),
        }
    )
    frame.to_csv(target, index=False)
    return str(target)


def test_bootstrap_matches_the_identical_resampling_and_the_t_interval(
    bench: Path,
) -> None:
    """bootstrap_statistic — seeded reference re-run + analytic leg."""
    config = ConfigModel()
    data = _data(
        _call(
            "bootstrap_statistic",
            {"path": _values_csv(bench), "column": "extra", "seed": _SEED},
        )
    )
    # Row-30 default arrived through the guard seam.
    assert data["resamples"] == config.process.bootstrap_default_resamples
    values = np.asarray(_VALUES)
    assert data["estimate"] == pytest.approx(
        float(values.mean()), rel=config.testbench.tol_closed_form_rtol
    )
    # Reference leg: the SAME percentile algorithm, same rng stream.
    rng = np.random.default_rng(_SEED)
    draws = np.array(
        [
            float(np.mean(values[rng.integers(0, len(values), size=len(values))]))
            for _ in range(config.process.bootstrap_default_resamples)
        ]
    )
    tail = (1.0 - data["confidence_level"]) / 2.0
    assert data["confidence_interval"]["lower"] == pytest.approx(
        float(np.quantile(draws, tail)), rel=config.testbench.tol_closed_form_rtol
    )
    assert data["confidence_interval"]["upper"] == pytest.approx(
        float(np.quantile(draws, 1.0 - tail)),
        rel=config.testbench.tol_closed_form_rtol,
    )
    # Analytic leg: the t-interval the percentile bootstrap converges to.
    scale = float(values.std(ddof=1) / math.sqrt(len(values)))
    t_lower, t_upper = stats.t.interval(
        data["confidence_level"],
        df=len(values) - 1,
        loc=float(values.mean()),
        scale=scale,
    )
    width = t_upper - t_lower
    assert abs(data["confidence_interval"]["lower"] - t_lower) < 0.2 * width
    assert abs(data["confidence_interval"]["upper"] - t_upper) < 0.2 * width


def test_bayesian_posterior_is_the_published_student_t_result(bench: Path) -> None:
    """bayesian_estimate — the conjugate-normal published closed form."""
    config = ConfigModel()
    data = _data(
        _call(
            "bayesian_estimate",
            {"path": _values_csv(bench), "column": "extra"},
        )
    )
    values = np.asarray(_VALUES)
    n = len(values)
    scale = float(values.std(ddof=1) / math.sqrt(n))
    expected_lower, expected_upper = stats.t.interval(
        data["credible_level"], df=n - 1, loc=float(values.mean()), scale=scale
    )
    rtol = config.testbench.tol_closed_form_rtol
    assert data["posterior_mean"] == pytest.approx(float(values.mean()), rel=rtol)
    assert data["credible_interval"]["lower"] == pytest.approx(expected_lower, rel=rtol)
    assert data["credible_interval"]["upper"] == pytest.approx(expected_upper, rel=rtol)
    assert data["degrees_of_freedom"] == n - 1


def test_monte_carlo_integration_matches_the_normal_cdf(bench: Path) -> None:
    """monte_carlo_simulate integration — oracle: the fitted normal's
    CDF mass over the bounds; row-31 default via the guard seam."""
    config = ConfigModel()
    values = np.asarray(_VALUES)
    location = float(values.mean())
    spread = float(values.std(ddof=1))
    bounds = [location - spread, location + spread]
    data = _data(
        _call(
            "monte_carlo_simulate",
            {
                "path": _values_csv(bench),
                "column": "extra",
                "simulation_type": "integration",
                "bounds": bounds,
                "seed": _SEED,
            },
        )
    )
    assert data["iterations"] == config.process.monte_carlo_default_iterations
    expected = float(
        stats.norm.cdf(bounds[1], location, spread)
        - stats.norm.cdf(bounds[0], location, spread)
    )
    # MC error at the configured iteration count: a few standard errors.
    tolerance = 4.0 / math.sqrt(data["iterations"])
    assert abs(data["probability_mass"] - expected) < tolerance


def test_stratified_sample_preserves_the_strata_shares(bench: Path) -> None:
    """generate_sample stratified — proportional design, seeded."""
    data = _data(
        _call(
            "generate_sample",
            {
                "path": _people_csv(bench),
                "sampling_method": "stratified",
                "sample_size": 0.5,
                "stratify_column": "region",
                "seed": _SEED,
            },
        )
    )
    assert data["sample_rows"] == 50
    regions = [row[0] for row in data["rows"]]
    assert regions.count("north") == 20
    assert regions.count("south") == 20
    assert regions.count("west") == 10


def test_systematic_sample_draws_the_requested_count(bench: Path) -> None:
    """generate_sample systematic — exact count, deterministic step."""
    data = _data(
        _call(
            "generate_sample",
            {
                "path": _people_csv(bench),
                "sampling_method": "systematic",
                "sample_size": 25,
                "seed": _SEED,
            },
        )
    )
    assert data["sample_rows"] == 25
    scores = [row[1] for row in data["rows"]]
    # Every 4th row from the seeded start: constant stride.
    strides = {b - a for a, b in zip(scores, scores[1:])}
    assert strides == {4}
