"""testbench/batteries/domain/statistical_battery_test.py — E10.a slice.

The NFR-502c domain battery's statistical rows: FR-301 L3 coverage
(every family tool called at the fastmcp.Client seam with its real
signature) and the FR-304/NFR-505 dual-assertion oracle — every
numeric verdict checked against BOTH an independently-computed
reference-library call (scipy/statsmodels, recomputed inside the
test) AND a published statistic from a classic dataset. The published
fixtures are R's `sleep` (Student 1908 / Cushny & Peebles: paired
t = -4.0621, p = 0.002833) and `PlantGrowth` (one-way ANOVA
F = 4.8461, p = 0.01591) — values as printed by R's own documented
analyses, compared at the rounding precision they were published at.
Deterministic order, path-addressed CSV fixtures (the X-2 contract's
file leg). FR-310's closure row asserts the effect-size output is
non-empty with the hand-computed Cohen's d.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Iterator

import anyio
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

# R `sleep`: extra hours of sleep under two drugs, 10 patients (paired).
_SLEEP_DRUG1 = (0.7, -1.6, -0.2, -1.2, -0.1, 3.4, 3.7, 0.8, 0.0, 2.0)
_SLEEP_DRUG2 = (1.9, 0.8, 1.1, 0.1, -0.1, 4.4, 5.5, 1.6, 4.6, 3.4)
# Published paired-t verdict (R: t.test(extra ~ group, paired = TRUE)).
_SLEEP_PUBLISHED_T = -4.0621
_SLEEP_PUBLISHED_P = 0.002833

# R `PlantGrowth`: dried plant weight under control and two treatments.
_PLANT_CTRL = (4.17, 5.58, 5.18, 6.11, 4.50, 4.61, 5.17, 4.53, 5.33, 5.14)
_PLANT_TRT1 = (4.81, 4.17, 4.41, 3.59, 5.87, 3.83, 6.03, 4.89, 4.32, 4.69)
_PLANT_TRT2 = (6.31, 5.12, 5.54, 5.50, 5.37, 5.29, 4.92, 6.15, 5.80, 5.26)
# Published one-way ANOVA verdict (R: summary(aov(weight ~ group))).
_PLANT_PUBLISHED_F = 4.8461
_PLANT_PUBLISHED_P = 0.01591

# Published values print at 4-5 significant digits — the comparison
# tolerance is the rounding precision, not a computation tolerance.
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


def _sleep_csv(tmp_path: Path) -> str:
    target = tmp_path / "sleep.csv"
    pd.DataFrame({"drug1": _SLEEP_DRUG1, "drug2": _SLEEP_DRUG2}).to_csv(
        target, index=False
    )
    return str(target)


def _plant_csv(tmp_path: Path) -> str:
    target = tmp_path / "plant_growth.csv"
    frame = pd.DataFrame(
        {
            "weight": _PLANT_CTRL + _PLANT_TRT1 + _PLANT_TRT2,
            "group": ["ctrl"] * 10 + ["trt1"] * 10 + ["trt2"] * 10,
        }
    )
    frame.to_csv(target, index=False)
    return str(target)


def test_hypothesis_paired_t_matches_reference_and_published(bench: Path) -> None:
    """analyze_hypothesis_test ttest_rel — both oracle legs (FR-304)."""
    config = ConfigModel()
    data = _data(
        _call(
            "analyze_hypothesis_test",
            {
                "path": _sleep_csv(bench),
                "test_type": "ttest_rel",
                "column": "drug1",
                "second_column": "drug2",
            },
        )
    )
    reference = stats.ttest_rel(_SLEEP_DRUG1, _SLEEP_DRUG2)
    rtol = config.testbench.tol_closed_form_rtol
    assert data["statistic"] == pytest.approx(reference.statistic, rel=rtol)
    assert data["p_value"] == pytest.approx(reference.pvalue, rel=rtol)
    assert data["statistic"] == pytest.approx(_SLEEP_PUBLISHED_T, rel=_PUBLISHED_RTOL)
    assert data["p_value"] == pytest.approx(_SLEEP_PUBLISHED_P, rel=_PUBLISHED_RTOL)
    assert data["significant"] is True
    assert data["n_pairs"] == 10


def test_hypothesis_auto_selects_from_supplied_columns(bench: Path) -> None:
    """The auto rule answers the question the columns pose."""
    data = _data(
        _call(
            "analyze_hypothesis_test",
            {
                "path": _plant_csv(bench),
                "column": "weight",
            },
        )
    )
    assert data["test_type"] == "normality"
    reference = stats.shapiro(_PLANT_CTRL + _PLANT_TRT1 + _PLANT_TRT2)
    config = ConfigModel()
    assert data["p_value"] == pytest.approx(
        reference.pvalue, rel=config.testbench.tol_closed_form_rtol
    )


def test_anova_matches_reference_and_published(bench: Path) -> None:
    """analyze_anova on PlantGrowth — both oracle legs (FR-304)."""
    config = ConfigModel()
    data = _data(
        _call(
            "analyze_anova",
            {
                "path": _plant_csv(bench),
                "dependent_var": "weight",
                "group_var": "group",
            },
        )
    )
    reference = stats.f_oneway(_PLANT_CTRL, _PLANT_TRT1, _PLANT_TRT2)
    rtol = config.testbench.tol_closed_form_rtol
    assert data["f_statistic"] == pytest.approx(reference.statistic, rel=rtol)
    assert data["p_value"] == pytest.approx(reference.pvalue, rel=rtol)
    assert data["f_statistic"] == pytest.approx(_PLANT_PUBLISHED_F, rel=_PUBLISHED_RTOL)
    assert data["p_value"] == pytest.approx(_PLANT_PUBLISHED_P, rel=_PUBLISHED_RTOL)
    assert data["significant"] is True
    assert set(data["groups"]) == {"ctrl", "trt1", "trt2"}
    # Tukey post-hoc rides along on a significant omnibus verdict.
    assert {row["group_a"] for row in data["post_hoc"]} <= {"ctrl", "trt1", "trt2"}


def test_effect_sizes_non_empty_with_known_cohens_d(bench: Path) -> None:
    """FR-310 closure: non-empty, correct output for documented input."""
    target = bench / "two_groups.csv"
    frame = pd.DataFrame(
        {
            "value": _PLANT_CTRL + _PLANT_TRT2,
            "group": ["a"] * 10 + ["b"] * 10,
        }
    )
    frame.to_csv(target, index=False)
    data = _data(
        _call(
            "analyze_effect_sizes",
            {"path": str(target), "column": "value", "group_column": "group"},
        )
    )
    # Reference leg: Cohen's d recomputed from the raw formula.
    a = pd.Series(_PLANT_CTRL)
    b = pd.Series(_PLANT_TRT2)
    pooled = math.sqrt(
        ((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
        / (len(a) + len(b) - 2)
    )
    expected_d = (a.mean() - b.mean()) / pooled
    config = ConfigModel()
    rtol = config.testbench.tol_closed_form_rtol
    assert data["cohens_d"] == pytest.approx(expected_d, rel=rtol)
    # FR-310: the documented measures are all present and non-empty.
    for key in ("hedges_g", "glass_delta", "cliffs_delta", "interpretation"):
        assert data[key] not in (None, "", [], {})


def test_ab_test_proportion_matches_statsmodels(bench: Path) -> None:
    """analyze_ab_test binary metric — reference-library oracle leg."""
    from statsmodels.stats.proportion import proportions_ztest

    target = bench / "ab.csv"
    converted_a = [1] * 45 + [0] * 155  # 200 users, 22.5% conversion
    converted_b = [1] * 30 + [0] * 170  # 200 users, 15.0% conversion
    frame = pd.DataFrame(
        {
            "converted": converted_a + converted_b,
            "variant": ["a"] * 200 + ["b"] * 200,
        }
    )
    frame.to_csv(target, index=False)
    data = _data(
        _call(
            "analyze_ab_test",
            {
                "path": str(target),
                "metric_column": "converted",
                "variant_column": "variant",
            },
        )
    )
    assert data["test_type"] == "proportion"
    statistic, p_value = proportions_ztest(count=[45, 30], nobs=[200, 200])
    config = ConfigModel()
    rtol = config.testbench.tol_closed_form_rtol
    assert data["statistic"] == pytest.approx(statistic, rel=rtol)
    assert data["p_value"] == pytest.approx(p_value, rel=rtol)
    assert data["variants"]["a"]["mean"] == pytest.approx(0.225, rel=rtol)
    assert data["relative_lift"] == pytest.approx(0.5, rel=rtol)
    assert data["winner"] == ("a" if data["significant"] else None)
