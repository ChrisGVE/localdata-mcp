"""tests/v3/test_explore_profile.py — E9.2/E9.3/E9.4: the X-tools.

FR-202's quality profile (hand-computed reference values — counts and
cardinalities exact, floats direct arithmetic), the exactly-one-source
refusal branches (zero, both, bad second slot — each naming the
parameters), FR-203's known-match search fixture with the unsafe-
pattern refusal, and FR-204's exact-set/frequency mapping with the
report-only note.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pytest
from sqlalchemy import text

import localdata_mcp.ingest.runtime as runtime
from localdata_mcp.explore.categorical import map_categories
from localdata_mcp.explore.quality import profile_data
from localdata_mcp.explore.search import search_data
from localdata_mcp.nexus.chokepoint.guard import Chokepoint, GuardedExecutionError
from localdata_mcp.nexus.config.endpoints import EndpointDeclaration
from localdata_mcp.nexus.config.models import ConfigModel, SecurityConfig


@pytest.fixture()
def booted(tmp_path: Path) -> Iterator[Chokepoint]:
    config = ConfigModel(
        security=SecurityConfig(allowed_paths=(str(tmp_path),)),
        endpoints={
            "warehouse": EndpointDeclaration(
                name="warehouse",
                dsn=f"sqlite:///{tmp_path / 'w.db'}",
                posture="read_write",
            ),
        },
    )
    guard = Chokepoint.boot(config, environ={})
    with guard._persistence.connection("warehouse") as connection:
        connection.execute(
            text(
                "CREATE TABLE orders (id INTEGER, amount REAL, status TEXT, note TEXT)"
            )
        )
        rows = [
            (1, 10.0, "paid", "alpha"),
            (2, 20.0, "paid", "beta"),
            (3, 30.0, "open", None),
            (4, None, "open", "gamma"),
        ]
        for row in rows:
            connection.execute(
                text("INSERT INTO orders VALUES (:a, :b, :c, :d)"),
                {"a": row[0], "b": row[1], "c": row[2], "d": row[3]},
            )
        connection.commit()
    runtime.configure_ingest(guard)
    yield guard
    runtime._CHOKEPOINT = None
    guard.shutdown()


class TestProfileData:
    def test_profile_matches_hand_computed_reference(self, booted: Chokepoint) -> None:
        report = profile_data(endpoint="warehouse", table="orders")
        assert report["row_count"] == 4
        assert report["column_count"] == 4
        amount = report["columns"]["amount"]
        assert amount["null_count"] == 1
        assert amount["cardinality"] == 3
        assert amount["min"] == 10.0
        assert amount["max"] == 30.0
        assert amount["mean"] == 20.0  # (10+20+30)/3, direct arithmetic
        status = report["columns"]["status"]
        assert status["null_count"] == 0
        assert status["cardinality"] == 2

    def test_query_slot_profiles_the_projection(self, booted: Chokepoint) -> None:
        report = profile_data(
            endpoint="warehouse",
            query="SELECT amount FROM orders WHERE status = 'paid'",
        )
        assert report["row_count"] == 2
        assert report["columns"]["amount"]["null_count"] == 0

    def test_path_source_profiles_a_file(
        self, booted: Chokepoint, tmp_path: Path
    ) -> None:
        (tmp_path / "d.csv").write_text("a,b\n1,x\n2,\n")
        report = profile_data(path=str(tmp_path / "d.csv"))
        assert report["row_count"] == 2
        assert report["columns"]["b"]["null_count"] == 1

    def test_zero_sources_refused_naming_both(self, booted: Chokepoint) -> None:
        with pytest.raises(GuardedExecutionError) as refusal:
            profile_data()
        structured = refusal.value.structured
        assert "endpoint=" in structured.message
        assert "path=" in structured.message

    def test_both_sources_refused(self, booted: Chokepoint, tmp_path: Path) -> None:
        with pytest.raises(GuardedExecutionError) as refusal:
            profile_data(endpoint="warehouse", path=str(tmp_path / "d.csv"))
        assert "Exactly one" in refusal.value.structured.message

    def test_endpoint_needs_exactly_one_second_slot(self, booted: Chokepoint) -> None:
        with pytest.raises(GuardedExecutionError, match="table=|query="):
            profile_data(endpoint="warehouse")
        with pytest.raises(GuardedExecutionError, match="table=|query="):
            profile_data(endpoint="warehouse", table="orders", query="SELECT 1")

    def test_table_with_path_refused(self, booted: Chokepoint, tmp_path: Path) -> None:
        (tmp_path / "d.csv").write_text("a\n1\n")
        with pytest.raises(GuardedExecutionError, match="catalog"):
            profile_data(path=str(tmp_path / "d.csv"), table="orders")


class TestSearchData:
    def test_known_match_fixture(self, booted: Chokepoint) -> None:
        report = search_data(endpoint="warehouse", target="orders", query="al.?ha")
        assert report["total_matches"] == 1
        (match,) = report["matches"]
        assert match["column"] == "note"
        assert match["match"] == "alpha"
        assert report["truncated"] is False

    def test_sql_target_narrows_the_scan(self, booted: Chokepoint) -> None:
        report = search_data(
            endpoint="warehouse",
            target="SELECT note FROM orders WHERE status = 'open'",
            query="gamma",
        )
        assert report["total_matches"] == 1
        assert report["rows_searched"] == 2

    def test_unsafe_pattern_refused(self, booted: Chokepoint) -> None:
        with pytest.raises(GuardedExecutionError) as refusal:
            search_data(endpoint="warehouse", target="orders", query="(a+)+b")
        assert "refused" in refusal.value.structured.message

    def test_unknown_column_refused_with_profile_hint(self, booted: Chokepoint) -> None:
        with pytest.raises(GuardedExecutionError) as refusal:
            search_data(
                endpoint="warehouse",
                target="orders",
                query="x",
                columns="absent",
            )
        assert "profile_data" in refusal.value.structured.suggestion


class TestMapCategories:
    def test_exact_set_and_frequencies(self, booted: Chokepoint) -> None:
        report = map_categories(endpoint="warehouse", table="orders", column="status")
        assert report["distinct_count"] == 2
        assert {entry["value"]: entry["count"] for entry in report["values"]} == {
            "paid": 2,
            "open": 2,
        }
        assert report["suggested_encoding"] == "one-hot"
        assert report["values_truncated"] is False
        assert "Report only" in report["note"]

    def test_unknown_column_refused(self, booted: Chokepoint) -> None:
        with pytest.raises(GuardedExecutionError):
            map_categories(endpoint="warehouse", table="orders", column="absent")
