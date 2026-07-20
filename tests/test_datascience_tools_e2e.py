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
import sqlite3
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

    # A 4-node cycle plus one chord, for network analysis. Node identifiers are
    # numeric because the analyzer builds a numeric array from the edge list.
    pd.DataFrame(
        {
            "source": [0, 1, 2, 3, 0],
            "target": [1, 2, 3, 0, 2],
            "weight": [1.0, 2.0, 1.5, 3.0, 2.5],
        }
    ).to_csv(_fp("ds_network.csv"), index=False)

    # Linear program and assignment problem. The assignment costs are chosen so
    # the optimum is unambiguous: agent i is cheapest on task i.
    pd.DataFrame(
        {
            "cost": [3.0, 5.0, 2.0],
            "cap1": [1.0, 2.0, 1.5],
            "cap2": [2.0, 1.0, 1.5],
            "task1": [1.0, 9.0, 9.0],
            "task2": [9.0, 1.0, 9.0],
            "task3": [9.0, 9.0, 1.0],
        }
    ).to_csv(_fp("ds_optimization.csv"), index=False)


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


# ---------------------------------------------------------------------------
# Optimization
# ---------------------------------------------------------------------------


class TestOptimizationTools:
    """The optimization tools read a table directly rather than taking a query."""

    def test_linear_program_finds_an_optimum(self, db: DatabaseManager) -> None:
        """Minimising non-negative costs under upper-bound constraints."""
        _connect(db, "lp", "ds_optimization.csv")
        result = _json(
            db.solve_linear_program(
                "lp",
                "data_table",
                objective_column="cost",
                constraint_columns=["cap1", "cap2"],
                constraint_values=[10.0, 10.0],
                constraint_types=["<=", "<="],
            )
        )

        assert result["success"] is True
        solution = result["optimal_solution"]
        assert len(solution) == 3, f"one value per variable, got {solution}"
        assert all(v >= -1e-9 for v in solution), f"bounds violated: {solution}"
        assert isinstance(result["optimal_value"], float)

    def test_assignment_picks_the_cheapest_pairing(self, db: DatabaseManager) -> None:
        """Costs are 1 on the diagonal and 9 elsewhere, so the optimum is the diagonal."""
        _connect(db, "asg", "ds_optimization.csv")
        result = _json(
            db.solve_assignment_problem(
                "asg",
                "data_table",
                cost_matrix_columns=["task1", "task2", "task3"],
            )
        )

        assert result["success"] is True
        assert result["total_cost"] == pytest.approx(
            3.0
        ), f"the diagonal assignment costs 3.0, got {result['total_cost']}"
        assert result["is_perfect_matching"] is True
        pairs = result["assignment_pairs"]
        assert [p["task_index"] for p in pairs] == [
            0,
            1,
            2,
        ], f"not the diagonal: {pairs}"
        assert all(p["cost"] == pytest.approx(1.0) for p in pairs)

    def test_network_analysis_describes_the_graph(self, db: DatabaseManager) -> None:
        """A 4-node cycle plus one chord: 4 nodes, 5 edges, connected."""
        _connect(db, "net", "ds_network.csv")
        result = _json(
            db.analyze_network(
                "net",
                "data_table",
                source_column="source",
                target_column="target",
                weight_column="weight",
            )
        )

        assert result["success"] is True
        properties = result["graph_properties"]
        assert properties["num_nodes"] == 4
        assert properties["num_edges"] == 5
        assert properties["is_connected"] is True
        assert properties["is_directed"] is False
        assert "centrality_measures" in result

    def test_unknown_column_is_named_in_the_error(self, db: DatabaseManager) -> None:
        """A bad column must say which one, not fail inside the SQL string."""
        _connect(db, "bad", "ds_optimization.csv")
        with pytest.raises(ValueError, match="no_such_column"):
            db.solve_linear_program(
                "bad", "data_table", objective_column="no_such_column"
            )

    def test_non_numeric_node_ids_are_rejected_clearly(
        self, db: DatabaseManager
    ) -> None:
        """Text node identifiers fail with a readable message, not a float cast error."""
        _connect(db, "txt", "ds_transactions.csv")
        with pytest.raises(ValueError, match="not numeric"):
            db.analyze_network(
                "txt",
                "data_table",
                source_column="customer_id",
                target_column="order_date",
            )


# ---------------------------------------------------------------------------
# Geospatial analysis
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def geo_db_path(datascience_fixtures: None) -> str:
    """Write the multi-table SQLite fixture the geospatial tools analyze.

    Geospatial work needs several related tables at once — nodes with edges,
    points with polygons — and a CSV connection exposes exactly one table, so
    this domain gets a small SQLite file instead.

    The layout is deliberate, so the assertions can check a *true* finding:

    - ``sensors`` holds three well-separated blobs whose values rise from ~10 to
      ~50 to ~90, so the high blob must come back a hot spot, the low blob a
      cold spot, and the middle neither.
    - ``noise`` holds the same coordinates with values shuffled independently of
      position, so autocorrelation must come back *not* significant. Without
      this contrast a test only shows the tool answers, not that it discriminates.
    - ``net_nodes``/``net_edges`` form a 3x3 unit grid, so shortest-path
      distances are known by hand: node 0 to node 8 is 4 hops of weight 1.
    - ``readings``/``zones`` place two points in zone A (values 10 and 30) and
      one in zone B (value 20), with a fourth point outside both.
    """
    path = _fp("geo_spatial.sqlite")
    if os.path.exists(path):
        os.remove(path)

    rng = np.random.default_rng(SEED)
    con = sqlite3.connect(path)

    x = np.concatenate(
        [rng.normal(0, 1, 20), rng.normal(20, 1, 20), rng.normal(40, 1, 20)]
    )
    y = np.concatenate(
        [rng.normal(0, 1, 20), rng.normal(20, 1, 20), rng.normal(40, 1, 20)]
    )
    value = np.concatenate(
        [rng.normal(10, 1, 20), rng.normal(50, 1, 20), rng.normal(90, 1, 20)]
    )
    pd.DataFrame({"x": x, "y": y, "value": value}).to_sql("sensors", con, index=False)
    pd.DataFrame({"x": x, "y": y, "value": rng.permutation(value)}).to_sql(
        "noise", con, index=False
    )

    node_ids = {}
    nodes = []
    for i in range(3):
        for j in range(3):
            node_ids[(i, j)] = len(nodes)
            nodes.append({"id": len(nodes), "x": float(i), "y": float(j)})
    edges = []
    for i in range(3):
        for j in range(3):
            if i < 2:
                edges.append(
                    {
                        "source": node_ids[(i, j)],
                        "target": node_ids[(i + 1, j)],
                        "weight": 1.0,
                    }
                )
            if j < 2:
                edges.append(
                    {
                        "source": node_ids[(i, j)],
                        "target": node_ids[(i, j + 1)],
                        "weight": 1.0,
                    }
                )
    pd.DataFrame(nodes).to_sql("net_nodes", con, index=False)
    pd.DataFrame(edges).to_sql("net_edges", con, index=False)

    pd.DataFrame(
        {
            "zone": ["A", "B"],
            "geometry": [
                "POLYGON((0 0,1 0,1 1,0 1,0 0))",
                "POLYGON((1 1,2 1,2 2,1 2,1 1))",
            ],
        }
    ).to_sql("zones", con, index=False)
    pd.DataFrame(
        {
            "z2": ["X"],
            "geometry": ["POLYGON((0.5 0.5,1.5 0.5,1.5 1.5,0.5 1.5,0.5 0.5))"],
        }
    ).to_sql("overlap_zone", con, index=False)
    pd.DataFrame(
        {
            "pid": [1, 2, 3, 4],
            "val": [10.0, 20.0, 30.0, 40.0],
            "x": [0.5, 1.5, 0.2, 5.0],
            "y": [0.5, 1.5, 0.7, 5.0],
        }
    ).to_sql("readings", con, index=False)

    con.commit()
    con.close()
    return path


@pytest.fixture
def geo(db: DatabaseManager, geo_db_path: str) -> DatabaseManager:
    """A DatabaseManager with the geospatial fixture connected as 'geo'."""
    db.connect_database("geo", "sqlite", geo_db_path)
    return db


class TestGeospatialCapabilities:
    def test_core_backends_are_reported_available(self, db: DatabaseManager) -> None:
        """Capability discovery must name the backends, not just say 'ok'."""
        result = _json(db.check_geospatial_capabilities())

        assert result["has_core_geospatial"] is True
        for library in ("geopandas", "shapely", "pyproj"):
            assert result["available_libraries"][library] is True
            assert result["versions"][library]
        assert isinstance(result["missing_libraries"], list)


class TestSpatialAutocorrelation:
    def test_morans_i_detects_planted_clustering(self, geo: DatabaseManager) -> None:
        """Values that rise blob by blob must register as strongly clustered."""
        result = _json(
            geo.analyze_spatial_autocorrelation(
                "geo", "SELECT * FROM sensors", value_column="value"
            )
        )

        assert result["statistic"] == "morans_i"
        assert result["value"] > 0.9, "near-perfect clustering should approach I=1"
        assert result["value"] > result["expected_value"]
        assert result["p_value"] < 0.01
        assert result["is_significant"] is True
        assert result["z_score"] > 3
        assert "positive spatial autocorrelation" in result["interpretation"]
        assert result["n_observations"] == 60

    def test_gearys_c_agrees_with_morans_i(self, geo: DatabaseManager) -> None:
        """Geary's C must point the same way as Moran's I, from below 1.

        Geary's variance was negative before, so p_value was None on every call
        and the statistic could never report significance at all.
        """
        result = _json(
            geo.analyze_spatial_autocorrelation(
                "geo", "SELECT * FROM sensors", value_column="value", method="geary"
            )
        )

        assert result["statistic"] == "gearys_c"
        assert 0.0 <= result["value"] < 1.0, "clustering drives C below its mean of 1"
        assert result["p_value"] is not None
        assert result["p_value"] < 0.01
        assert result["z_score"] < 0
        assert "positive spatial autocorrelation" in result["interpretation"]

    def test_shuffled_values_are_not_significant(self, geo: DatabaseManager) -> None:
        """The same coordinates with values shuffled must NOT look clustered.

        This is the half that matters: a test that only checks the clustered
        case passes for a tool that calls everything clustered.
        """
        result = _json(
            geo.analyze_spatial_autocorrelation(
                "geo", "SELECT * FROM noise", value_column="value"
            )
        )

        assert abs(result["value"]) < 0.3
        assert result["p_value"] > 0.05
        assert result["is_significant"] is False

    def test_too_few_points_for_the_neighbourhood_is_rejected(
        self, geo: DatabaseManager
    ) -> None:
        with pytest.raises(ValueError, match="more than"):
            geo.analyze_spatial_autocorrelation(
                "geo", "SELECT * FROM sensors LIMIT 5", value_column="value"
            )

    def test_unknown_method_is_named_in_the_error(self, geo: DatabaseManager) -> None:
        with pytest.raises(ValueError, match="dbscan"):
            geo.analyze_spatial_autocorrelation(
                "geo", "SELECT * FROM sensors", value_column="value", method="dbscan"
            )


class TestSpatialHotspots:
    def test_hot_and_cold_blobs_are_found_where_they_were_planted(
        self, geo: DatabaseManager
    ) -> None:
        """The high blob must be hot, the low blob cold, and the middle neither."""
        result = _json(
            geo.find_spatial_hotspots(
                "geo", "SELECT * FROM sensors", value_column="value"
            )
        )

        assert result["n_points"] == 60
        assert result["n_hotspots"] == 20
        assert result["n_coldspots"] == 20

        points = result["points"]
        assert len(points) == 60
        # The fixture writes the blobs in order: low, middle, high.
        low, middle, high = points[:20], points[20:40], points[40:]
        assert all(p["is_coldspot"] for p in low)
        assert all(p["cluster_id"] == -1 for p in low)
        assert not any(p["is_hotspot"] or p["is_coldspot"] for p in middle)
        assert all(p["is_hotspot"] for p in high)
        assert all(p["cluster_id"] == 1 for p in high)
        assert all(p["gi_star_z_score"] > 0 for p in high)
        assert all(p["gi_star_p_value"] < 0.05 for p in high)

    def test_a_stricter_threshold_cannot_find_more_spots(
        self, geo: DatabaseManager
    ) -> None:
        """Tightening the significance level must never add hot spots."""
        loose = _json(
            geo.find_spatial_hotspots(
                "geo", "SELECT * FROM sensors", value_column="value"
            )
        )
        strict = _json(
            geo.find_spatial_hotspots(
                "geo",
                "SELECT * FROM sensors",
                value_column="value",
                significance_level=0.0001,
            )
        )

        assert strict["n_hotspots"] <= loose["n_hotspots"]
        assert strict["significance_level"] == 0.0001


class TestSpatialDistances:
    def test_nearest_neighbour_is_much_closer_than_the_average_point(
        self, geo: DatabaseManager
    ) -> None:
        """With three tight blobs 20 units apart, neighbours are ~1 and means ~26.

        The nearest-neighbour column was masked with ``eye * inf``, which is nan
        off the diagonal, so every point reported column 0 as its neighbour.
        """
        result = _json(geo.calculate_spatial_distances("geo", "SELECT * FROM sensors"))

        assert result["n_points"] == 60
        summary = result["summary"]
        assert summary["min"] > 0, "self-distances must not enter the summary"
        assert summary["max"] > 50, "the outer blobs are ~57 units apart"
        assert summary["mean_nearest_neighbor_distance"] < 2.0
        assert summary["mean_nearest_neighbor_distance"] < summary["mean"] / 10

        neighbours = result["nearest_neighbors"]
        assert len(neighbours) == 60
        # A point's nearest neighbour must be in its own blob of twenty.
        assert all(
            n["point_index"] // 20 == n["nearest_index"] // 20 for n in neighbours
        )
        assert all(n["point_index"] != n["nearest_index"] for n in neighbours)

    def test_haversine_reports_kilometres_not_coordinate_units(
        self, geo: DatabaseManager
    ) -> None:
        """Treating the coordinates as degrees must give a far larger number."""
        plane = _json(geo.calculate_spatial_distances("geo", "SELECT * FROM sensors"))
        globe = _json(
            geo.calculate_spatial_distances(
                "geo", "SELECT * FROM sensors", distance_type="haversine"
            )
        )

        assert plane["unit"] == "coordinate units"
        assert globe["unit"] == "kilometres"
        assert globe["summary"]["mean"] > plane["summary"]["mean"] * 50

    def test_pairs_are_returned_only_when_asked_for(self, geo: DatabaseManager) -> None:
        without = _json(
            geo.calculate_spatial_distances("geo", "SELECT * FROM sensors LIMIT 5")
        )
        with_pairs = _json(
            geo.calculate_spatial_distances(
                "geo", "SELECT * FROM sensors LIMIT 5", include_pairs=True
            )
        )

        assert "pairs" not in without
        assert len(with_pairs["pairs"]) == 25
        assert all(
            p["distance"] == 0
            for p in with_pairs["pairs"]
            if p["point1_index"] == p["point2_index"]
        )

    def test_distances_to_a_reference_set(self, geo: DatabaseManager) -> None:
        """A second query measures one set against another, not against itself."""
        result = _json(
            geo.calculate_spatial_distances(
                "geo",
                "SELECT * FROM sensors LIMIT 10",
                reference_query="SELECT * FROM sensors LIMIT 3",
            )
        )

        assert result["n_points"] == 10
        assert result["n_reference_points"] == 3
        assert result["summary"]["min"] == 0.0, "the sets share their first rows"

    def test_missing_coordinate_column_is_named(self, geo: DatabaseManager) -> None:
        with pytest.raises(ValueError, match="longitude"):
            geo.calculate_spatial_distances(
                "geo", "SELECT * FROM sensors", x_column="longitude"
            )


class TestSpatialNetworks:
    def test_route_visits_every_waypoint_at_the_known_cost(
        self, geo: DatabaseManager
    ) -> None:
        """On a 3x3 unit grid, 0 -> 4 -> 8 is four unit steps and no more."""
        result = _json(
            geo.optimize_route(
                "geo",
                "SELECT * FROM net_nodes",
                "SELECT * FROM net_edges",
                waypoints=[0, 4, 8],
                weight_column="weight",
            )
        )

        assert result["total_distance"] == 4.0
        assert set(result["waypoints"]) == {0, 4, 8}
        assert result["route_path"][0] == 0
        assert result["route_path"][-1] == 8
        assert 4 in result["route_path"]
        assert len(result["route_coordinates"]) == len(result["route_path"])
        assert result["path_geometry"].startswith("LINESTRING")

    def test_waypoints_given_as_strings_still_resolve(
        self, geo: DatabaseManager
    ) -> None:
        """MCP arguments arrive as JSON, so an integer node id may come as text."""
        as_text = _json(
            geo.optimize_route(
                "geo",
                "SELECT * FROM net_nodes",
                "SELECT * FROM net_edges",
                waypoints=["0", "8"],
                weight_column="weight",
            )
        )
        as_int = _json(
            geo.optimize_route(
                "geo",
                "SELECT * FROM net_nodes",
                "SELECT * FROM net_edges",
                waypoints=[0, 8],
                weight_column="weight",
            )
        )

        assert as_text["total_distance"] == as_int["total_distance"] == 4.0

    def test_unknown_waypoint_is_rejected_by_name(self, geo: DatabaseManager) -> None:
        with pytest.raises(ValueError, match="99"):
            geo.optimize_route(
                "geo",
                "SELECT * FROM net_nodes",
                "SELECT * FROM net_edges",
                waypoints=[0, 99],
                weight_column="weight",
            )

    def test_accessibility_ranks_the_nearer_demand_point_higher(
        self, geo: DatabaseManager
    ) -> None:
        """Node 4 is two steps from node 0 and node 8 is four, so 4 must score better."""
        result = _json(
            geo.analyze_accessibility(
                "geo",
                "SELECT * FROM net_nodes",
                "SELECT * FROM net_edges",
                service_locations=[0],
                demand_locations=[4, 8],
                weight_column="weight",
            )
        )

        assert result["travel_times"]["4"] == 2.0
        assert result["travel_times"]["8"] == 4.0
        assert result["accessibility_scores"]["4"] > result["accessibility_scores"]["8"]
        assert result["unreachable_locations"] == []
        assert result["service_coverage"]["coverage_percentage"] == 100.0

    def test_isochrone_bands_grow_with_travel_time(self, geo: DatabaseManager) -> None:
        """Each band must produce a real polygon covering more ground than the last.

        Every band came back None before: the generator asked the *unweighted*
        shortest-path function for a weighted cutoff, and swallowed the TypeError.
        """
        result = _json(
            geo.generate_service_isochrones(
                "geo",
                "SELECT * FROM net_nodes",
                "SELECT * FROM net_edges",
                service_locations=[0],
                time_bands=[1.0, 2.0],
                weight_column="weight",
            )
        )

        bands = result["isochrones"]
        assert [b["time_band"] for b in bands] == [1.0, 2.0]
        assert all(b["geometry"] is not None for b in bands)
        assert all("POLYGON" in b["geometry"] for b in bands)
        assert all(b["area"] > 0 for b in bands)
        assert bands[1]["area"] > bands[0]["area"]

    def test_an_edge_naming_an_absent_node_is_rejected(
        self, geo: DatabaseManager
    ) -> None:
        """A truncated node query must fail by name, not build a broken graph."""
        with pytest.raises(ValueError, match="absent from the nodes query"):
            geo.optimize_route(
                "geo",
                "SELECT * FROM net_nodes LIMIT 3",
                "SELECT * FROM net_edges",
                waypoints=[0, 1],
                weight_column="weight",
            )


class TestSpatialGeometry:
    def test_points_are_joined_to_the_zone_that_contains_them(
        self, geo: DatabaseManager
    ) -> None:
        """Three of four readings fall in a zone; the fourth is outside both."""
        result = _json(
            geo.perform_spatial_join(
                "geo", "SELECT * FROM readings", "SELECT * FROM zones"
            )
        )

        assert result["match_counts"] == {"matches": 3, "no_matches": 1}
        assert result["row_count"] == 3

        by_point = {row["pid_left"]: row["zone_right"] for row in result["rows"]}
        assert by_point == {1: "A", 3: "A", 2: "B"}
        assert 4 not in by_point, "the point at (5, 5) lies outside every zone"

    def test_overlay_intersection_keeps_only_the_shared_area(
        self, geo: DatabaseManager
    ) -> None:
        """Each unit zone overlaps the offset square in exactly a 0.5 x 0.5 corner."""
        from shapely import wkt as shapely_wkt

        result = _json(
            geo.perform_spatial_overlay(
                "geo", "SELECT * FROM zones", "SELECT * FROM overlap_zone"
            )
        )

        assert result["operation"] == "intersection"
        assert result["output_count"] == 2
        areas = [shapely_wkt.loads(row["geometry"]).area for row in result["rows"]]
        assert all(abs(area - 0.25) < 1e-9 for area in areas)

    def test_point_values_are_summarised_per_polygon(
        self, geo: DatabaseManager
    ) -> None:
        """Zone A holds readings 10 and 30; zone B holds 20 alone.

        The aggregation passed its named-aggregation mapping positionally, so
        pandas raised KeyError on every call before this.
        """
        result = _json(
            geo.aggregate_points_in_polygons(
                "geo",
                "SELECT * FROM readings",
                "SELECT * FROM zones",
                value_column="val",
            )
        )

        assert result["n_points"] == 4
        assert result["n_polygons"] == 2

        by_zone = {row["zone"]: row for row in result["polygons"]}
        assert by_zone["A"]["val_count"] == 2
        assert by_zone["A"]["val_sum"] == 40.0
        assert by_zone["A"]["val_mean"] == 20.0
        assert by_zone["B"]["val_count"] == 1
        assert by_zone["B"]["val_sum"] == 20.0

    def test_custom_aggregation_functions_are_honoured(
        self, geo: DatabaseManager
    ) -> None:
        result = _json(
            geo.aggregate_points_in_polygons(
                "geo",
                "SELECT * FROM readings",
                "SELECT * FROM zones",
                value_column="val",
                aggregation_functions=["min", "max"],
            )
        )

        by_zone = {row["zone"]: row for row in result["polygons"]}
        assert by_zone["A"]["val_min"] == 10.0
        assert by_zone["A"]["val_max"] == 30.0

    def test_a_query_with_neither_geometry_nor_coordinates_is_rejected(
        self, geo: DatabaseManager
    ) -> None:
        """The error must name both encodings it looked for."""
        with pytest.raises(ValueError, match="No geometry found"):
            geo.perform_spatial_join(
                "geo",
                "SELECT val FROM readings",
                "SELECT * FROM zones",
            )
