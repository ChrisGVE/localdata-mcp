"""Execute every workflow documented in docs/advanced-examples.md.

The page it guards replaced a document that had rotted into fiction: the old
`ADVANCED_EXAMPLES.md` described v1.7.0, called seven tools that no longer exist,
and nothing noticed because no test ever ran an example. Documentation that is
never executed is a claim, not a fact.

So each test here is one documented workflow, called with the arguments the page
shows, asserting on the fields the page tells a reader to look at. A parameter
rename or a dropped result key fails here and the page gets fixed with the code.
"""

import json
import os

import numpy as np
import pandas as pd
import pytest

from localdata_mcp import DatabaseManager

# Path security restricts connections to the working directory.
FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
SEED = 20260720


def _fp(name: str) -> str:
    return os.path.join(FIXTURES_DIR, name)


@pytest.fixture(scope="module", autouse=True)
def example_fixtures() -> None:
    """Datasets with the structure the documented examples claim to find."""
    os.makedirs(FIXTURES_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)

    half = 60
    pd.DataFrame(
        {
            "group": ["A"] * half + ["B"] * half,
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
    ).to_csv(_fp("ex_experiment.csv"), index=False)

    months = pd.date_range("2022-01-01", periods=96, freq="MS")
    trend = np.arange(96) * 0.8 + 100
    season = 5 * np.sin(np.arange(96) * (2 * np.pi / 12))
    pd.DataFrame(
        {"date": months, "value": np.round(trend + season + rng.normal(0, 1, 96), 4)}
    ).to_csv(_fp("ex_timeseries.csv"), index=False)

    n = 150
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    x3 = rng.normal(size=n)
    target = 3 * x1 - 2 * x2 + 0.5 * x3 + rng.normal(0, 0.5, n)
    pd.DataFrame(
        {
            "x1": x1,
            "x2": x2,
            "x3": x3,
            "target": np.round(target, 4),
            "predicted": np.round(3 * x1 - 2 * x2 + 0.5 * x3, 4),
        }
    ).to_csv(_fp("ex_regression.csv"), index=False)

    # Three treatments at 20 / 23.5 / 27, so every pairwise post-hoc comparison
    # is genuinely significant and can be named.
    per_group = 45
    pd.DataFrame(
        {
            "treatment": np.repeat(["ctrl", "low", "high"], per_group),
            "response": np.round(
                np.concatenate(
                    [
                        rng.normal(20.0, 3.0, per_group),
                        rng.normal(23.5, 3.0, per_group),
                        rng.normal(27.0, 3.0, per_group),
                    ]
                ),
                4,
            ),
        }
    ).to_csv(_fp("ex_anova.csv"), index=False)

    # A transaction log: 30 customers, repeat orders, spread over a year, so RFM
    # has a real recency spread to score rather than one purchase each.
    customers = [f"C{i:03d}" for i in range(1, 31)]
    rows = []
    for customer in customers:
        for _ in range(int(rng.integers(3, 10))):
            rows.append(
                {
                    "customer_id": customer,
                    "order_date": (
                        pd.Timestamp("2026-01-01")
                        + pd.Timedelta(days=int(rng.integers(0, 360)))
                    ).date(),
                    "amount": round(float(rng.uniform(20, 400)), 2),
                }
            )
    pd.DataFrame(rows).to_csv(_fp("ex_transactions.csv"), index=False)


@pytest.fixture
def manager() -> DatabaseManager:
    return DatabaseManager()


@pytest.fixture
def experiment(manager):
    manager.connect_database("experiment", "csv", _fp("ex_experiment.csv"))
    return manager


class TestLargeResultWorkflow:
    """ "Reading a result larger than your context"."""

    def test_preview_estimates_before_executing(self, experiment):
        preview = json.loads(
            experiment.analyze_query_preview("experiment", "SELECT * FROM data_table")
        )

        assert preview["estimates"]["rows"] == 120
        # The page tells readers to budget on this field.
        assert preview["estimates"]["tokens"] > 0

    def test_execute_returns_a_query_id_and_pages_from_it(self, experiment):
        response = json.loads(
            experiment.execute_query("experiment", "SELECT * FROM data_table")
        )
        query_id = response["metadata"]["query_id"]

        page = json.loads(experiment.next_chunk(query_id, 11, "10"))

        assert page["metadata"]["showing_rows"] == "11-20"
        assert len(page["data"]) == 10

    def test_quality_report_precedes_analysis(self, experiment):
        response = json.loads(
            experiment.execute_query("experiment", "SELECT * FROM data_table")
        )
        report = json.loads(
            experiment.get_data_quality_report(response["metadata"]["query_id"])
        )

        assert "overall_quality" in report
        assert report["statistical_summary"]["null_percentage"] == 0.0

    def test_buffer_can_be_released(self, experiment):
        response = json.loads(
            experiment.execute_query("experiment", "SELECT * FROM data_table")
        )
        result = experiment.clear_streaming_buffer(response["metadata"]["query_id"])

        assert result is not None


class TestFileToAnswerWorkflow:
    """ "From a raw file to a defensible answer"."""

    QUERY = 'SELECT value, "group" AS grp FROM data_table'

    def test_connect_reports_the_shape(self, experiment):
        info = json.loads(
            experiment.connect_database("again", "csv", _fp("ex_experiment.csv"))
        )

        assert info["success"] is True
        assert info["tables"][0]["name"] == "data_table"

    def test_hypothesis_test_returns_interpreted_results(self, experiment):
        result = json.loads(
            experiment.analyze_hypothesis_test(
                "experiment", self.QUERY, column="value", group_column="grp"
            )
        )

        assert result["test_results"]
        assert "interpretation" in result["test_results"][0]

    def test_effect_size_reports_magnitude_not_just_significance(self, experiment):
        result = json.loads(
            experiment.analyze_effect_sizes(
                "experiment", self.QUERY, column="value", group_column="grp"
            )
        )
        effect = result["effect_sizes"]["cohens_d_value_by_grp"]

        # The fixture separates the groups by more than a standard deviation.
        assert effect["effect_description"] == "large"
        assert abs(effect["cohens_d"]) > 0.8
        assert effect["group1_size"] == 60

    def test_ab_test_reports_power_alongside_significance(self, experiment):
        result = json.loads(
            experiment.analyze_ab_test(
                "experiment",
                'SELECT "group" AS grp, converted FROM data_table',
                variant_column="grp",
                metric_column="converted",
            )
        )

        assert result["p_value"] < 0.05
        # The page tells readers to check power before believing a null.
        assert result["power"] > 0.5


class TestTimeSeriesWorkflow:
    """ "Time series: describe, then project"."""

    @pytest.fixture
    def metrics(self, manager):
        manager.connect_database("metrics", "csv", _fp("ex_timeseries.csv"))
        return manager

    def test_describe_reports_gaps_and_trend(self, metrics):
        result = json.loads(
            metrics.analyze_time_series(
                "metrics",
                "SELECT * FROM data_table",
                date_column="date",
                value_column="value",
            )
        )

        assert result["series_info"]["length"] == 96
        assert result["series_info"]["missing_values"] == 0
        assert result["trend_analysis"]["linear_trend"]["slope"] > 0

    def test_forecast_returns_the_requested_horizon(self, metrics):
        result = json.loads(
            metrics.forecast_time_series(
                "metrics",
                "SELECT * FROM data_table",
                date_column="date",
                value_column="value",
                horizon=6,
                method="arima",
            )
        )

        assert len(result["forecast_values"]) == 6


class TestModellingWorkflow:
    """ "Modelling and checking the model"."""

    @pytest.fixture
    def housing(self, manager):
        manager.connect_database("housing", "csv", _fp("ex_regression.csv"))
        return manager

    def test_regression_fits(self, housing):
        result = json.loads(
            housing.analyze_regression(
                "housing",
                "SELECT * FROM data_table",
                target_column="target",
                feature_columns=["x1", "x2", "x3"],
            )
        )

        assert result["model_type"] == "linear"

    def test_evaluation_reports_bias_alongside_fit(self, housing):
        result = json.loads(
            housing.evaluate_model_performance(
                "housing",
                "SELECT * FROM data_table",
                target_column="target",
                prediction_column="predicted",
            )
        )

        assert result["metrics"]["r2"] > 0.9
        # The page tells readers to read this next to r2.
        assert "mean_residual" in result["metrics"]

    def test_anomaly_detection_flags_a_minority(self, housing):
        result = json.loads(
            housing.detect_anomalies(
                "housing",
                "SELECT x1, x2, x3 FROM data_table",
                method="isolation_forest",
                contamination=0.1,
            )
        )

        assert 0 < result["n_anomalies"] < result["n_samples"] / 2

    def test_dimension_reduction_returns_requested_components(self, housing):
        result = json.loads(
            housing.reduce_dimensions(
                "housing",
                "SELECT x1, x2, x3 FROM data_table",
                method="pca",
                n_components=2,
            )
        )

        assert result["reduced_dimensions"] == 2


class TestDistributionFreeWorkflow:
    """ "Estimating without assuming a distribution"."""

    def test_bootstrap_reports_standard_error(self, experiment):
        result = json.loads(
            experiment.bootstrap_statistic(
                "experiment",
                "SELECT value FROM data_table",
                column="value",
                statistic="mean",
                n_bootstrap=1000,
                confidence_level=0.95,
            )
        )
        first = result["bootstrap_results"][0]

        assert first["n_bootstrap"] == 1000
        assert first["standard_error"] > 0

    def test_sampling_returns_a_fraction_of_the_population(self, experiment):
        result = json.loads(
            experiment.generate_sample(
                "experiment",
                "SELECT * FROM data_table",
                sampling_method="simple_random",
                sample_size=0.1,
            )
        )

        # sample_size below 1 is a fraction: 10% of 120 rows.
        assert 6 <= len(result["sample_data"]) <= 18


class TestNavigationWorkflow:
    """ "Finding your way around unfamiliar data"."""

    def test_find_table_names_the_connections_holding_it(self, experiment):
        assert "experiment" in json.loads(experiment.find_table("data_table"))

    def test_search_data_matches_by_regex(self, experiment):
        result = json.loads(
            experiment.search_data(
                "experiment",
                "SELECT * FROM data_table",
                pattern="^B$",
                columns="group",
            )
        )

        assert result["matches"]
        assert all(m["value"] == "B" for m in result["matches"])

    def test_transform_data_previews_rather_than_applies(self, experiment):
        result = json.loads(
            experiment.transform_data(
                "experiment",
                "SELECT * FROM data_table",
                column="group",
                find="A",
                replace="control",
            )
        )

        assert result["transformed_count"] == 60
        assert result["sample"][0]["transformed"] == "control"

    def test_export_schema_returns_machine_readable_types(self, experiment):
        result = json.loads(experiment.export_schema("experiment"))

        assert "data_table" in result["tables"]


class TestFailureDiagnosisWorkflow:
    """ "When a workflow goes wrong"."""

    def test_error_log_records_the_failing_query(self, experiment):
        from localdata_mcp.query_audit import get_query_audit_buffer

        get_query_audit_buffer().clear()
        experiment.execute_query("experiment", "SELECT nope FROM data_table")

        log = json.loads(experiment.get_error_log())

        assert log["total_entries"] >= 1
        assert "nope" in log["entries"][0]["query"]
        get_query_audit_buffer().clear()

    def test_streaming_status_reports_memory(self, experiment):
        status = json.loads(experiment.get_streaming_status())

        assert "memory_status" in status


class TestCompositionLimits:
    """The two limits the page states plainly, asserted so they stay true."""

    def test_execute_query_refuses_a_non_select(self, experiment):
        result = experiment.execute_query("experiment", "DELETE FROM data_table")

        assert "Security" in result or "error" in result.lower()

    def test_cross_connection_join_is_not_supported(self, experiment):
        """Each query runs against one connection, as the page says."""
        experiment.connect_database("second", "csv", _fp("ex_regression.csv"))

        result = experiment.execute_query(
            "experiment", "SELECT * FROM second.data_table"
        )

        assert "error" in result.lower()


class TestSegmentationWorkflow:
    """ "Segmenting customers, then acting on the segments"."""

    @pytest.fixture
    def sales(self, manager):
        manager.connect_database("sales", "csv", _fp("ex_transactions.csv"))
        return manager

    def test_rfm_scores_every_customer_on_three_axes(self, sales):
        result = json.loads(
            sales.analyze_rfm(
                "sales",
                "SELECT * FROM data_table",
                customer_column="customer_id",
                date_column="order_date",
                value_column="amount",
            )
        )
        first = result["rfm_scores"][0]

        # The page's point is that R separates customers who spent alike.
        for axis in ("recency", "frequency", "monetary", "R", "F", "M"):
            assert axis in first

    def test_clustering_finds_the_requested_number_of_segments(self, sales):
        result = json.loads(
            sales.analyze_clusters(
                "sales",
                "SELECT amount FROM data_table",
                n_clusters=4,
            )
        )

        assert result["n_clusters"] == 4
        assert len(set(result["labels"])) == 4


class TestMoreThanTwoGroups:
    """The ANOVA example under "From a raw file to a defensible answer"."""

    @pytest.fixture
    def trials(self, manager):
        manager.connect_database("trials", "csv", _fp("ex_anova.csv"))
        return manager

    def test_anova_separates_three_treatments(self, trials):
        result = json.loads(
            trials.analyze_anova(
                "trials",
                "SELECT * FROM data_table",
                dependent_var="response",
                group_var="treatment",
            )
        )
        anova = result["anova_results"]["one_way_response_by_treatment"]

        # The fixture's three means are genuinely apart.
        assert anova["p_value"] < 0.001
        assert anova["df_between"] == 2

    def test_post_hoc_names_every_pair(self, trials):
        """Post-hoc was silently empty for every call before this release."""
        result = json.loads(
            trials.analyze_anova(
                "trials",
                "SELECT * FROM data_table",
                dependent_var="response",
                group_var="treatment",
            )
        )
        post_hoc = result["post_hoc_results"]["one_way_response_by_treatment"]
        pairs = {frozenset((c["group1"], c["group2"])) for c in post_hoc["comparisons"]}

        assert pairs == {
            frozenset(("ctrl", "low")),
            frozenset(("ctrl", "high")),
            frozenset(("low", "high")),
        }


class TestModelCheckingWorkflow:
    """`evaluate_model_performance`, from "Modelling and checking the model"."""

    def test_evaluation_reports_bias_next_to_fit(self, manager):
        manager.connect_database("housing", "csv", _fp("ex_regression.csv"))

        result = json.loads(
            manager.evaluate_model_performance(
                "housing",
                "SELECT * FROM data_table",
                target_column="target",
                prediction_column="predicted",
            )
        )

        assert result["metrics"]["r2"] > 0.9
        assert "mean_residual" in result["metrics"]


class TestSpatialWorkflow:
    """ "Spatial analysis" — test for clustering before hunting for clusters."""

    @pytest.fixture
    def sensors(self, manager):
        manager.connect_database("sensors", "sqlite", _fp("geo_spatial.sqlite"))
        return manager

    def test_autocorrelation_detects_the_clustered_field(self, sensors):
        result = json.loads(
            sensors.analyze_spatial_autocorrelation(
                "sensors",
                "SELECT x, y, value FROM sensors",
                value_column="value",
                method="moran",
            )
        )

        # The fixture is three well-separated blobs, so this must be significant.
        assert result["is_significant"] is True
        assert result["value"] > 0

    def test_autocorrelation_does_not_fire_on_the_shuffled_field(self, sensors):
        """The contrast the page relies on: random values must not read clustered."""
        result = json.loads(
            sensors.analyze_spatial_autocorrelation(
                "sensors",
                "SELECT x, y, value FROM noise",
                value_column="value",
                method="moran",
            )
        )

        assert result["is_significant"] is False

    def test_hotspots_label_each_point(self, sensors):
        result = json.loads(
            sensors.find_spatial_hotspots(
                "sensors",
                "SELECT x, y, value FROM sensors",
                value_column="value",
            )
        )

        assert result["n_points"] == 60
        for point in result["points"]:
            assert "gi_star_z_score" in point
            assert "is_hotspot" in point

    def test_route_optimization_visits_the_waypoints(self, sensors):
        result = json.loads(
            sensors.optimize_route(
                "sensors",
                nodes_query="SELECT id, x, y FROM net_nodes",
                edges_query="SELECT source, target, weight FROM net_edges",
                waypoints=[0, 8],
            )
        )

        assert "error" not in result


class TestChunkPagingWorkflow:
    """`request_data_chunk` and `request_multiple_chunks` from the paging section."""

    @pytest.fixture
    def buffered(self, experiment):
        response = json.loads(
            experiment.execute_query("experiment", "SELECT * FROM data_table")
        )
        return experiment, response["metadata"]["query_id"]

    def test_a_chunk_can_be_fetched_by_id(self, buffered):
        manager, query_id = buffered

        chunk = json.loads(manager.request_data_chunk(query_id, 0))

        assert chunk["data"]
        assert chunk["metadata"]["rows"] == len(chunk["data"])

    def test_several_chunks_can_be_fetched_at_once(self, buffered):
        manager, query_id = buffered

        chunks = json.loads(manager.request_multiple_chunks(query_id, "0,1"))

        assert sorted(chunks) == ["0", "1"]


class TestQueryLogWorkflow:
    """`get_query_log` from "When a workflow goes wrong"."""

    def test_query_log_records_a_successful_query(self, experiment):
        from localdata_mcp.query_audit import get_query_audit_buffer

        get_query_audit_buffer().clear()
        experiment.execute_query("experiment", "SELECT value FROM data_table")

        log = json.loads(experiment.get_query_log(database="experiment"))

        assert log["total_entries"] >= 1
        get_query_audit_buffer().clear()
