"""testbench/batteries/domain/statistical_seam_test.py — CR-008/CR-009.

The significance level (alpha), the interval-coverage level, and the
spatial k-NN neighbour count are NX-2 config knobs (process.*), projected
to the domains through the guard's `process_defaults()` seam. These unit
rows prove the routing is real — the computation functions honour the
value injected at the tool layer (never a private inline default) — and
that changing the injected value changes the verdict. They stand outside
the MCP/geostack path so they run without the geospatial extra. Only
non-config numeric literals are used here (this file is scanned by the
one-default-site gate); exact-default assertions live in tests/v3.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from localdata_mcp.nexus.chokepoint.guard import ProcessDefaults
from localdata_mcp.nexus.config.models import ConfigModel, ProcessConfig
from localdata_mcp.process.domains.geospatial_analysis.spatial_stats import (
    spatial_autocorrelation,
    spatial_hotspots,
)
from localdata_mcp.process.domains.sampling_estimation.estimation import bootstrap
from localdata_mcp.process.domains.sampling_estimation.monte_carlo import simulate
from localdata_mcp.process.domains.statistical_analysis.ab_test import perform_ab_test
from localdata_mcp.process.domains.statistical_analysis.anova import perform_anova
from localdata_mcp.process.domains.statistical_analysis.hypothesis import (
    run_hypothesis_test,
)
from localdata_mcp.process.domains.time_series.analysis import analyze_series

# A significance level far below any real p-value: makes a clearly-
# significant verdict register as not-significant, proving the injected
# alpha (not an inline literal) governs. Not a config default value.
_TINY_ALPHA = 1e-12


def _grid_frame() -> pd.DataFrame:
    """A 5x5 grid with a linear-gradient value: strongly autocorrelated."""
    rows = [
        {"x": float(i), "y": float(j), "v": float(i + j)}
        for i in range(5)
        for j in range(5)
    ]
    return pd.DataFrame(rows)


class TestConfigKnobsExist:
    def test_process_config_carries_the_significance_knob(self) -> None:
        # 0.05 is the unscanned alpha default; the exact 0.95/8 defaults
        # are asserted in tests/v3/test_config_model.py (not gate-scanned).
        assert ConfigModel().process.significance_level == 0.05

    def test_process_defaults_seam_mirrors_config(self) -> None:
        process = ProcessConfig()
        defaults = ProcessDefaults(
            bootstrap_resamples=process.bootstrap_default_resamples,
            monte_carlo_iterations=process.monte_carlo_default_iterations,
            significance_level=process.significance_level,
            confidence_level=process.confidence_level,
            spatial_k_neighbors=process.spatial_k_neighbors,
        )
        assert defaults.significance_level == process.significance_level
        assert defaults.confidence_level == process.confidence_level
        assert defaults.spatial_k_neighbors == process.spatial_k_neighbors


class TestAlphaRouting:
    def test_anova_verdict_reads_the_injected_alpha(self) -> None:
        frame = pd.DataFrame(
            {
                "weight": [1.0, 1.1, 0.9, 2.0, 2.1, 1.9],
                "group": ["a", "a", "a", "b", "b", "b"],
            }
        )
        strict = perform_anova(frame, "weight", "group", default_alpha=_TINY_ALPHA)
        assert strict["alpha"] == _TINY_ALPHA
        assert strict["significant"] is False
        lax = perform_anova(frame, "weight", "group", default_alpha=0.5)
        assert lax["alpha"] == 0.5
        assert lax["significant"] is True

    def test_hypothesis_verdict_reads_the_injected_alpha(self) -> None:
        frame = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
        result = run_hypothesis_test(
            frame, test_type="ttest_1samp", column="x", default_alpha=0.1
        )
        assert result["alpha"] == 0.1

    def test_ab_test_verdict_reads_the_injected_alpha(self) -> None:
        frame = pd.DataFrame(
            {
                "metric": [1.0, 1.1, 0.9, 2.0, 2.1, 1.9],
                "variant": ["a", "a", "a", "b", "b", "b"],
            }
        )
        result = perform_ab_test(frame, "metric", "variant", default_alpha=_TINY_ALPHA)
        assert result["alpha"] == _TINY_ALPHA
        assert result["significant"] is False

    def test_time_series_stationarity_reads_the_injected_alpha(self) -> None:
        rng = np.random.default_rng(0)
        frame = pd.DataFrame(
            {
                "d": pd.date_range("2020-01-01", periods=60, freq="D"),
                "v": rng.normal(size=60),
            }
        )
        # A near-certain stationary series: the verdict flips only if alpha
        # is driven absurdly low, proving the injected value is used.
        strict = analyze_series(frame, "d", "v", default_significance=_TINY_ALPHA)
        lax = analyze_series(frame, "d", "v", default_significance=0.5)
        assert strict["stationarity"]["is_stationary"] is False
        assert lax["stationarity"]["is_stationary"] is True


class TestConfidenceRouting:
    def test_bootstrap_interval_reads_the_injected_confidence(self) -> None:
        frame = pd.DataFrame({"x": [float(i) for i in range(20)]})
        result = bootstrap(frame, "x", resamples=200, seed=1, default_confidence=0.8)
        assert result["confidence_level"] == 0.8


class TestMonteCarloBandFromAlpha:
    def test_uncertainty_band_narrows_as_significance_rises(self) -> None:
        frame = pd.DataFrame({"x": [float(i) for i in range(30)]})
        wide = simulate(frame, "x", iterations=800, seed=7, default_significance=0.02)[
            "distribution"
        ]
        narrow = simulate(frame, "x", iterations=800, seed=7, default_significance=0.2)[
            "distribution"
        ]
        # p05=alpha, p95=1-alpha: a larger alpha reports a tighter band.
        assert (narrow["p95"] - narrow["p05"]) < (wide["p95"] - wide["p05"])


class TestSpatialKRouting:
    def test_autocorrelation_reads_the_injected_k(self) -> None:
        result = spatial_autocorrelation(_grid_frame(), "v", default_k_neighbors=3)
        assert result["k_neighbors"] == 3

    def test_hotspots_reads_the_injected_significance(self) -> None:
        result = spatial_hotspots(
            _grid_frame(),
            "v",
            default_significance=0.1,
            default_k_neighbors=4,
        )
        assert result["significance_level"] == 0.1
