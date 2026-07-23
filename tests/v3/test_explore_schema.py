"""tests/v3/test_explore_schema.py — E9.1: X-1 schema discovery.

FR-201's per-connector-type L3 schema match over the real stack:
describe_database answers for every declared kind — the SQL catalog
(columns, primary key, row counts), the kv/tree key-space shape, the
graph node/edge shape, the rdf triple shape; describe_table and
find_table work the SQL catalog and are refused (with the store-tool
pointer) on store kinds; unknown endpoints carry the NFR-114
guidance.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pytest
from sqlalchemy import text

import localdata_mcp.ingest.runtime as runtime
from localdata_mcp.explore.tools import describe_database, describe_table, find_table
from localdata_mcp.nexus.chokepoint.guard import (
    Chokepoint,
    GuardedExecutionError,
    GuardRefusedError,
)
from localdata_mcp.nexus.config.endpoints import EndpointDeclaration
from localdata_mcp.nexus.config.models import ConfigModel

TTL = '@prefix ex: <http://example.org/> .\nex:a ex:knows ex:b .\nex:a ex:name "A" .\n'


@pytest.fixture()
def booted(tmp_path: Path) -> Iterator[Chokepoint]:
    ttl = tmp_path / "kb.ttl"
    ttl.write_text(TTL)
    config = ConfigModel(
        endpoints={
            "warehouse": EndpointDeclaration(
                name="warehouse",
                dsn=f"sqlite:///{tmp_path / 'w.db'}",
                posture="read_write",
            ),
            "lake": EndpointDeclaration(
                name="lake",
                dsn=f"duckdb:///{tmp_path / 'l.duckdb'}",
                posture="read_write",
            ),
            "notes": EndpointDeclaration(
                name="notes",
                dsn=f"kv+sqlite:///{tmp_path / 'kv.db'}",
                posture="read_write",
            ),
            "social": EndpointDeclaration(
                name="social",
                dsn=f"graph+sqlite:///{tmp_path / 'g.db'}",
                posture="read_write",
            ),
            "kb": EndpointDeclaration(
                name="kb", dsn=f"rdf+turtle:///{ttl}", posture="read_only"
            ),
        }
    )
    guard = Chokepoint.boot(config, environ={})
    with guard._persistence.connection("warehouse") as connection:
        connection.execute(
            text(
                "CREATE TABLE sales_2026 (id INTEGER PRIMARY KEY, "
                "amount REAL, note TEXT)"
            )
        )
        connection.execute(text("CREATE TABLE customers (id INTEGER)"))
        connection.execute(text("INSERT INTO sales_2026 VALUES (1, 9.5, 'x')"))
        connection.commit()
    with guard._persistence.connection("lake") as connection:
        connection.execute("CREATE TABLE metrics (id INTEGER, v DOUBLE)")
        connection.execute("INSERT INTO metrics VALUES (1, 2.5), (2, 3.5)")
    runtime.configure_ingest(guard)
    yield guard
    runtime._CHOKEPOINT = None
    guard.shutdown()


class TestDescribeDatabasePerKind:
    def test_sqlite_catalog_with_columns_and_counts(self, booted: Chokepoint) -> None:
        described = describe_database("warehouse")
        assert described["storage"] == "sql"
        assert described["dialect"] == "sqlite"
        by_name = {table["name"]: table for table in described["tables"]}
        assert set(by_name) == {"sales_2026", "customers"}
        sales = by_name["sales_2026"]
        assert sales["row_count"] == 1
        assert sales["primary_key"] == ["id"]
        assert {column["name"] for column in sales["columns"]} == {
            "id",
            "amount",
            "note",
        }

    def test_duckdb_catalog(self, booted: Chokepoint) -> None:
        described = describe_database("lake")
        assert described["dialect"] == "duckdb"
        (metrics,) = described["tables"]
        assert metrics["name"] == "metrics"
        assert metrics["row_count"] == 2
        assert metrics["columns"][0]["name"] == "id"

    def test_kv_store_answers_with_key_space_shape(self, booted: Chokepoint) -> None:
        described = describe_database("notes")
        assert described["storage"] == "tree"
        assert described["total_nodes"] == 0
        assert "get_children" in described["hint"]

    def test_graph_store_answers_with_graph_shape(self, booted: Chokepoint) -> None:
        described = describe_database("social")
        assert described["storage"] == "graph"
        assert described["node_count"] == 0
        assert described["is_directed"] is True

    def test_rdf_store_answers_with_triple_shape(self, booted: Chokepoint) -> None:
        described = describe_database("kb")
        assert described["storage"] == "rdf"
        assert described["triple_count"] == 2
        assert described["subject_count"] == 1
        assert described["predicate_count"] == 2

    def test_unknown_endpoint_carries_nfr114_guidance(self, booted: Chokepoint) -> None:
        with pytest.raises(GuardedExecutionError) as refusal:
            describe_database("nope")
        assert "list_endpoints()" in refusal.value.structured.suggestion


class TestDescribeTable:
    def test_table_schema_matches(self, booted: Chokepoint) -> None:
        described = describe_table("warehouse", "sales_2026")
        assert described["row_count"] == 1
        assert described["primary_key"] == ["id"]

    def test_missing_table_refused_with_discovery_hint(
        self, booted: Chokepoint
    ) -> None:
        with pytest.raises(GuardedExecutionError) as refusal:
            describe_table("warehouse", "absent")
        assert "describe_database" in refusal.value.structured.suggestion

    def test_store_kind_catalog_refused_toward_store_tools(
        self, booted: Chokepoint
    ) -> None:
        with pytest.raises(GuardRefusedError, match="semantic"):
            describe_table("notes", "nodes")


class TestFindTable:
    def test_glob_matches_within_the_endpoint(self, booted: Chokepoint) -> None:
        found = find_table("warehouse", "sales_*")
        assert found["matches"] == ["sales_2026"]
        assert found["searched_tables"] == 2

    def test_no_match_is_an_empty_list_not_an_error(self, booted: Chokepoint) -> None:
        found = find_table("warehouse", "zz*")
        assert found["matches"] == []

    def test_store_kind_refused(self, booted: Chokepoint) -> None:
        with pytest.raises(GuardRefusedError):
            find_table("social", "*")
