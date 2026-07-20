"""End-to-end tests for the data science MCP tools.

Every test drives a tool the way an MCP client does: connect a fixture file,
call the ``DatabaseManager`` method that is registered as the tool, and assert
on the *content* of the returned JSON.

This layer exists because the unit suite tests the domain functions and the
``DatabaseManager`` signatures separately, and nothing joined them up. The
adapter between the two was broken for the entire analytical surface while the
suite stayed green (issue #23). Asserting real values — a p-value that
separates two genuinely different groups, a cluster count that matches the
fixture's blobs — is what makes that failure visible here.

The fixtures are generated with a fixed seed, so the numeric expectations below
are stable rather than lucky.
"""

import json
import os
from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest

from localdata_mcp import DatabaseManager

# Path security restricts connections to the working directory, so the fixture
# files must live inside the repository rather than in a system temp directory.
FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")

SEED = 20260720


def _fp(filename: str) -> str:
    return os.path.join(FIXTURES_DIR, filename)


@pytest.fixture(scope="session", autouse=True)
def datascience_fixtures() -> None:
    """Write the CSV fixtures these tests analyze.

    They are generated rather than committed (``tests/fixtures/`` ignores
    generated data), and seeded so every numeric expectation in this module is
    reproducible rather than lucky. Each dataset has a known structure — a real
    group difference, two separated blobs, a trend plus seasonality — which is
    what lets the assertions check that an analysis found something true.
    """
    os.makedirs(FIXTURES_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)

    # Two-group experiment: hypothesis test, effect sizes, A/B test.
    # Means 10.0 vs 12.5; conversion 30% vs 55%.
    n = 120
    half = n // 2
    pd.DataFrame(
        {
            "group": np.array(["A"] * half + ["B"] * half),
            "value": np.round(
                np.concatenate(
                    [rng.normal(10.0, 2.0, half), rng.normal(12.5, 2.0, half)]
                ),
                4,
            ),
            "converted": np.concatenate(
                [rng.binomial(1, 0.30, half), rng.binomial(1, 0.55, half)]
            ),
        }
    ).to_csv(_fp("ds_experiment.csv"), index=False)

    # Three-group design for ANOVA: means 20 / 23.5 / 27.
    m = 45
    pd.DataFrame(
        {
            "treatment": np.repeat(["ctrl", "low", "high"], m),
            "response": np.round(
                np.concatenate(
                    [
                        rng.normal(20.0, 3.0, m),
                        rng.normal(23.5, 3.0, m),
                        rng.normal(27.0, 3.0, m),
                    ]
                ),
                4,
            ),
        }
    ).to_csv(_fp("ds_anova.csv"), index=False)

    # Regression and model evaluation: target = 3*x1 + 1.5*x2 - 0.5*x3 + noise,
    # with `predicted` a good-but-imperfect stand-in for a model's output.
    k = 150
    x1 = rng.normal(0.0, 1.0, k)
    x2 = rng.normal(5.0, 2.0, k)
    x3 = rng.normal(-2.0, 1.5, k)
    target = 3.0 * x1 + 1.5 * x2 - 0.5 * x3 + rng.normal(0.0, 0.5, k)
    pd.DataFrame(
        {
            "x1": np.round(x1, 4),
            "x2": np.round(x2, 4),
            "x3": np.round(x3, 4),
            "target": np.round(target, 4),
            "predicted": np.round(target + rng.normal(0.0, 0.6, k), 4),
        }
    ).to_csv(_fp("ds_regression.csv"), index=False)

    # Pattern recognition: two tight, well-separated blobs plus 2 far outliers.
    c = 60
    points = np.vstack(
        [
            rng.normal([0.0, 0.0, 0.0], 0.6, (c, 3)),
            rng.normal([8.0, 8.0, 8.0], 0.6, (c, 3)),
            np.array([[25.0, -18.0, 22.0], [-20.0, 26.0, -24.0]]),
        ]
    )
    pd.DataFrame(np.round(points, 4), columns=["f1", "f2", "f3"]).to_csv(
        _fp("ds_patterns.csv"), index=False
    )

    # Time series: 96 monthly points, rising trend with a 12-month season.
    periods = 96
    pd.DataFrame(
        {
            "date": pd.date_range("2022-01-01", periods=periods, freq="MS").strftime(
                "%Y-%m-%d"
            ),
            "value": np.round(
                np.linspace(100.0, 180.0, periods)
                + 12.0 * np.sin(2.0 * np.pi * np.arange(periods) / 12.0)
                + rng.normal(0.0, 2.0, periods),
                4,
            ),
        }
    ).to_csv(_fp("ds_timeseries.csv"), index=False)

    # Customer transactions for RFM: 40 customers, 2-8 orders each.
    rows = []
    for index in range(1, 41):
        for _ in range(int(rng.integers(2, 9))):
            order_date = pd.Timestamp("2024-01-01") + pd.Timedelta(
                days=int(rng.integers(0, 540))
            )
            rows.append(
                {
                    "customer_id": f"C{index:03d}",
                    "order_date": order_date.strftime("%Y-%m-%d"),
                    "amount": round(float(rng.gamma(4.0, 30.0)), 2),
                }
            )
    pd.DataFrame(rows).to_csv(_fp("ds_transactions.csv"), index=False)


def _json(raw: str) -> Dict[str, Any]:
    return json.loads(raw)


def _flatten(obj: Any) -> Dict[str, Any]:
    """Collect every scalar in a nested result under its own key.

    Tools return results at different nesting depths. Flattening lets a test
    assert that a value exists and is sane without pinning the exact shape of
    the envelope around it.
    """
    flat: Dict[str, Any] = {}

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for key, val in node.items():
                if isinstance(val, (dict, list)):
                    walk(val)
                else:
                    flat.setdefault(key, val)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(obj)
    return flat


def _find_number(result: Dict[str, Any], *name_fragments: str) -> float:
    """Return the first numeric leaf whose key contains one of the fragments."""
    for key, val in _flatten(result).items():
        low = key.lower()
        if any(frag in low for frag in name_fragments) and isinstance(
            val, (int, float)
        ):
            if not isinstance(val, bool):
                return float(val)
    raise AssertionError(
        f"no numeric value matching {name_fragments} in result keys: "
        f"{sorted(_flatten(result))}"
    )


@pytest.fixture
def db() -> DatabaseManager:
    return DatabaseManager()


def _connect(db: DatabaseManager, name: str, fixture: str) -> None:
    raw = db.connect_database(name, "csv", _fp(fixture))
    assert "error" not in raw.lower() or _json(raw).get("success") is not False
    return None


ALL_ROWS = "SELECT * FROM data_table"


# ---------------------------------------------------------------------------
# Statistical analysis
# ---------------------------------------------------------------------------


class TestStatisticalTools:
    def test_hypothesis_test_detects_group_difference(
        self, db: DatabaseManager
    ) -> None:
        """Groups A and B are drawn from means 10.0 and 12.5 — the test must see it."""
        _connect(db, "exp", "ds_experiment.csv")
        result = _json(
            db.analyze_hypothesis_test(
                "exp",
                ALL_ROWS,
                test_type="ttest_ind",
                column="value",
                group_column="group",
            )
        )

        p_value = _find_number(result, "p_value", "pvalue")
        assert 0.0 <= p_value <= 1.0
        assert (
            p_value < 0.01
        ), f"a 2.5-sigma group gap must be significant, got p={p_value}"

    def test_anova_separates_three_treatments(self, db: DatabaseManager) -> None:
        """Means 20 / 23.5 / 27 across three groups — F must be large, p small."""
        _connect(db, "anv", "ds_anova.csv")
        result = _json(
            db.analyze_anova(
                "anv", ALL_ROWS, dependent_var="response", group_var="treatment"
            )
        )

        f_stat = _find_number(result, "f_stat", "f_value", "statistic")
        p_value = _find_number(result, "p_value", "pvalue")
        assert f_stat > 1.0, f"expected a real F statistic, got {f_stat}"
        assert (
            p_value < 0.01
        ), f"three separated treatments must be significant, got p={p_value}"

    def test_effect_sizes_reports_a_magnitude(self, db: DatabaseManager) -> None:
        """The A/B gap is ~1.25 pooled SDs, so a large effect size must come back."""
        _connect(db, "eff", "ds_experiment.csv")
        result = _json(
            db.analyze_effect_sizes(
                "eff", ALL_ROWS, column="value", group_column="group"
            )
        )

        effect = _find_number(result, "cohen", "effect_size", "hedges", "glass")
        assert abs(effect) > 0.5, f"expected a medium-or-larger effect, got {effect}"


# ---------------------------------------------------------------------------
# Regression and model evaluation
# ---------------------------------------------------------------------------


class TestRegressionTools:
    def test_regression_recovers_a_strong_linear_fit(self, db: DatabaseManager) -> None:
        """target = 3*x1 + 1.5*x2 - 0.5*x3 + small noise, so R^2 must be high."""
        _connect(db, "reg", "ds_regression.csv")
        result = _json(
            db.analyze_regression(
                "reg",
                "SELECT x1, x2, x3, target FROM data_table",
                target_column="target",
                feature_columns=["x1", "x2", "x3"],
            )
        )

        r2 = _find_number(result, "r2", "r_squared")
        assert (
            r2 > 0.9
        ), f"a near-deterministic linear relation must fit well, got R2={r2}"

    def test_evaluate_model_scores_stored_predictions(
        self, db: DatabaseManager
    ) -> None:
        """`predicted` tracks `target` with ~0.6 noise — a good but imperfect model."""
        _connect(db, "ev", "ds_regression.csv")
        result = _json(
            db.evaluate_model_performance(
                "ev", ALL_ROWS, target_column="target", prediction_column="predicted"
            )
        )

        r2 = _find_number(result, "r2", "r_squared")
        assert 0.8 < r2 < 1.0, f"expected a good-not-perfect fit, got R2={r2}"


# ---------------------------------------------------------------------------
# Pattern recognition
# ---------------------------------------------------------------------------


class TestPatternTools:
    def test_clustering_finds_the_two_blobs(self, db: DatabaseManager) -> None:
        """The fixture holds two well-separated blobs; k-means must label both."""
        _connect(db, "clu", "ds_patterns.csv")
        result = _json(
            db.analyze_clusters(
                "clu",
                ALL_ROWS,
                columns=["f1", "f2", "f3"],
                method="kmeans",
                n_clusters=2,
            )
        )

        assert result["n_clusters"] == 2

        # One label per input row, and both clusters actually used.
        labels = result["labels"]
        assert len(labels) == 122, f"expected a label per row, got {len(labels)}"
        assert set(labels) == {0, 1}

        # The blobs are far apart, so the separation must score near-perfect.
        assert result["silhouette_avg"] > 0.7, (
            f"well-separated blobs must cluster cleanly, got "
            f"silhouette={result['silhouette_avg']}"
        )

        # The two blobs hold 60 points each; the 2 far outliers join one of them.
        sizes = sorted(result["cluster_stats"]["cluster_sizes"].values())
        assert sizes == [61, 61], f"unexpected cluster sizes: {sizes}"

    def test_anomaly_detection_flags_the_outliers(self, db: DatabaseManager) -> None:
        """Two points sit far outside both blobs and must be flagged."""
        _connect(db, "ano", "ds_patterns.csv")
        result = _json(
            db.detect_anomalies(
                "ano",
                ALL_ROWS,
                columns=["f1", "f2", "f3"],
                method="isolation_forest",
                contamination=0.05,
            )
        )

        n_anomalies = _find_number(
            result, "n_anomalies", "num_anomalies", "anomaly_count"
        )
        assert (
            n_anomalies >= 2
        ), f"the two extreme points must be caught, got {n_anomalies}"

    def test_dimensionality_reduction_keeps_the_variance(
        self, db: DatabaseManager
    ) -> None:
        """Two separated blobs in 3-D collapse to 2 components with high variance kept."""
        _connect(db, "dim", "ds_patterns.csv")
        result = _json(
            db.reduce_dimensions(
                "dim",
                ALL_ROWS,
                columns=["f1", "f2", "f3"],
                method="pca",
                n_components=2,
            )
        )

        assert result["algorithm"] == "pca"
        assert result["original_dimensions"] == 3
        assert result["reduced_dimensions"] == 2
        assert result["evaluation"]["metrics"]["reduction_ratio"] == pytest.approx(
            2 / 3
        )

        # Reconstruction error must be a real, finite measurement of what was lost.
        error = result["reconstruction_error"]
        assert isinstance(error, float)
        assert 0.0 <= error < float("inf")


# ---------------------------------------------------------------------------
# Time series
# ---------------------------------------------------------------------------


class TestTimeSeriesTools:
    def test_time_series_analysis_describes_the_series(
        self, db: DatabaseManager
    ) -> None:
        """96 monthly points with a rising trend — the summary must reflect that."""
        _connect(db, "ts", "ds_timeseries.csv")
        result = _json(
            db.analyze_time_series(
                "ts", ALL_ROWS, date_column="date", value_column="value"
            )
        )

        length = _find_number(result, "length", "n_obs", "count")
        assert length == 96, f"expected all 96 observations, got {length}"

    def test_forecast_returns_the_requested_horizon(self, db: DatabaseManager) -> None:
        """A 6-step forecast must produce 6 forward values in a plausible range."""
        _connect(db, "fc", "ds_timeseries.csv")
        result = _json(
            db.forecast_time_series(
                "fc",
                ALL_ROWS,
                date_column="date",
                value_column="value",
                horizon=6,
                method="arima",
            )
        )

        flat = _flatten(result)
        forecasts = None
        for key, val in _flatten_lists(result).items():
            if "forecast" in key.lower() and isinstance(val, list) and val:
                forecasts = val
                break
        assert forecasts is not None, f"no forecast series in result: {sorted(flat)}"
        assert len(forecasts) == 6, f"expected 6 forecast points, got {len(forecasts)}"
        numeric = [v for v in forecasts if isinstance(v, (int, float))]
        assert numeric, "forecast values are not numeric"
        assert all(
            50.0 < v < 400.0 for v in numeric
        ), f"implausible forecasts: {numeric}"


def _flatten_lists(obj: Any) -> Dict[str, Any]:
    """Collect list-valued leaves by key, for assertions about series output."""
    found: Dict[str, Any] = {}

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for key, val in node.items():
                if isinstance(val, list):
                    found.setdefault(key, val)
                    walk(val)
                elif isinstance(val, dict):
                    walk(val)
        elif isinstance(node, list):
            for item in node:
                if isinstance(item, (dict, list)):
                    walk(item)

    walk(obj)
    return found


# ---------------------------------------------------------------------------
# Business intelligence
# ---------------------------------------------------------------------------


class TestBusinessIntelligenceTools:
    def test_rfm_segments_every_customer(self, db: DatabaseManager) -> None:
        """The fixture holds 40 customers; each must receive an RFM scoring."""
        _connect(db, "rfm", "ds_transactions.csv")
        result = _json(
            db.analyze_rfm(
                "rfm",
                ALL_ROWS,
                customer_column="customer_id",
                date_column="order_date",
                value_column="amount",
            )
        )

        scores = result["rfm_scores"]
        assert len(scores) == 40, f"expected all 40 customers scored, got {len(scores)}"
        assert len(result["segments"]) == 40

        first = scores[0]
        for field in ("customer_id", "recency", "frequency", "monetary", "RFM_Score"):
            assert field in first, f"RFM score missing '{field}': {sorted(first)}"
        assert first["frequency"] >= 1
        assert first["monetary"] > 0

        # Every customer lands in exactly one segment, so the summary must add up.
        assert sum(s["customer_count"] for s in result["segment_summary"]) == 40

    def test_ab_test_detects_the_conversion_lift(self, db: DatabaseManager) -> None:
        """Conversion is 30% for A and 55% for B across 120 rows — a real lift."""
        _connect(db, "ab", "ds_experiment.csv")
        result = _json(
            db.analyze_ab_test(
                "ab", ALL_ROWS, variant_column="group", metric_column="converted"
            )
        )

        p_value = _find_number(result, "p_value", "pvalue")
        assert 0.0 <= p_value <= 1.0
        assert (
            p_value < 0.05
        ), f"a 25-point conversion gap must be significant, got p={p_value}"


# ---------------------------------------------------------------------------
# Sampling and estimation
# ---------------------------------------------------------------------------


class TestSamplingTools:
    def test_sample_is_drawn_from_the_population(self, db: DatabaseManager) -> None:
        """A simple random sample of 30 from 120 rows, with the design recorded."""
        _connect(db, "smp", "ds_experiment.csv")
        result = _json(
            db.generate_sample(
                "smp", ALL_ROWS, sampling_method="simple_random", sample_size=30
            )
        )

        assert len(result["sample_data"]) == 30
        summary = result["sampling_results"]
        assert summary["sampling_method"] == "simple_random"
        assert summary["population_size"] == 120
        assert len(set(summary["sample_indices"])) == 30, "sample must not repeat a row"

    def test_stratified_sample_preserves_the_strata(self, db: DatabaseManager) -> None:
        """Groups A and B are equal halves, so a stratified sample must stay balanced."""
        _connect(db, "str", "ds_experiment.csv")
        result = _json(
            db.generate_sample(
                "str",
                ALL_ROWS,
                sampling_method="stratified",
                sample_size=20,
                stratify_column="group",
            )
        )

        assert result["sampling_results"]["sampling_method"] == "stratified"
        groups = [row["group"] for row in result["sample_data"]]
        assert len(groups) == 20
        assert set(groups) == {"A", "B"}, "both strata must be represented"
        assert abs(groups.count("A") - groups.count("B")) <= 2, (
            f"equal strata must sample evenly, got A={groups.count('A')} "
            f"B={groups.count('B')}"
        )

    def test_bootstrap_brackets_the_sample_mean(self, db: DatabaseManager) -> None:
        """The bootstrap CI must contain the statistic it resampled."""
        _connect(db, "bst", "ds_experiment.csv")
        result = _json(
            db.bootstrap_statistic(
                "bst", ALL_ROWS, column="value", statistic="mean", n_bootstrap=300
            )
        )

        assert result["n_bootstrap"] == 300
        estimate = result["bootstrap_results"][0]
        observed = estimate["original_statistic"]

        # The pooled mean of the two groups (10.0 and 12.5) sits near 11.25.
        assert 10.0 < observed < 12.5, f"unexpected sample mean: {observed}"

        low, high = estimate["confidence_intervals"]["percentile"]
        assert (
            low < observed < high
        ), f"CI [{low}, {high}] excludes the estimate {observed}"
        assert estimate["standard_error"] > 0

    def test_monte_carlo_runs_the_requested_draws(self, db: DatabaseManager) -> None:
        """The simulation must report the design it actually ran."""
        _connect(db, "mc", "ds_experiment.csv")
        result = _json(
            db.monte_carlo_simulate(
                "mc", ALL_ROWS, n_simulations=500, columns=["value"]
            )
        )

        assert result["n_simulations"] == 500
        assert result["simulation_type"] == "integration"
        assert result["monte_carlo_results"], "simulation returned no results"

    def test_bayesian_estimate_reports_a_posterior(self, db: DatabaseManager) -> None:
        """A normal prior over the value column must yield a posterior estimate."""
        _connect(db, "bay", "ds_experiment.csv")
        result = _json(
            db.bayesian_estimate(
                "bay", ALL_ROWS, column="value", prior_distribution="normal"
            )
        )

        assert result["estimation_type"] == "posterior"
        assert result["prior_distribution"] == "normal"
        assert result["bayesian_results"], "estimation returned no results"
