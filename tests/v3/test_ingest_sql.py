"""tests/v3/test_ingest_sql.py — E8.1: the SQL family + list_endpoints.

I-1's exit-gate assertions over the real stack (Chokepoint.boot on
tmp sqlite endpoints, tools crossing the installed runtime): query and
write_query thin over NX-6 with posture enforced, the NFR-114 refusal
branches this story owns — an undeclared endpoint name (i-a) and a
mutation against an undeclared endpoint (ii), both with the
suggestion-content assertions naming list_endpoints() and the
operator-declared model — the backend-kind-agnostic DSN-redacted
enumeration, the explicit zero-endpoints statement, and the fail-closed
not-yet-booted runtime refusal. The local-file branches (i-b) land
with E8.2's query_file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import pytest

import localdata_mcp.ingest.runtime as runtime
from localdata_mcp.ingest.connectors.sql.tools import query, write_query
from localdata_mcp.ingest.endpoints import list_endpoints
from localdata_mcp.nexus.chokepoint.guard import (
    Chokepoint,
    GuardedExecutionError,
    GuardRefusedError,
)
from localdata_mcp.nexus.config.endpoints import EndpointDeclaration
from localdata_mcp.nexus.config.models import ConfigModel


def booted_config(tmp_path: Path) -> ConfigModel:
    shared = f"sqlite:///{tmp_path / 'shared.db'}"
    return ConfigModel(
        endpoints={
            "warehouse": EndpointDeclaration(
                name="warehouse", dsn=shared, posture="read_write"
            ),
            "reporting": EndpointDeclaration(
                name="reporting", dsn=shared, posture="read_only"
            ),
        }
    )


@pytest.fixture()
def booted(tmp_path: Path) -> Iterator[Chokepoint]:
    guard = Chokepoint.boot(booted_config(tmp_path), environ={})
    # DDL is deliberately outside every guarded category, so the table
    # is seeded through NX-5's seam (test scaffolding only).
    with guard._persistence.connection("warehouse") as connection:
        from sqlalchemy import text

        connection.execute(text("CREATE TABLE t (id INTEGER, label TEXT)"))
        connection.execute(text("INSERT INTO t VALUES (1, 'a')"))
        connection.commit()
    runtime.configure_ingest(guard)
    yield guard
    runtime._CHOKEPOINT = None
    guard.shutdown()


class TestQueryTools:
    def test_query_returns_rows_through_the_guard(self, booted: Chokepoint) -> None:
        result = query("warehouse", "SELECT id, label FROM t")
        assert result.columns == ("id", "label")
        assert result.rows == ((1, "a"),)

    def test_query_reads_on_read_only_posture(self, booted: Chokepoint) -> None:
        result = query("reporting", "SELECT count(*) FROM t")
        assert result.rows[0][0] == 1

    def test_write_query_mutates_on_read_write(self, booted: Chokepoint) -> None:
        outcome = write_query("warehouse", "INSERT INTO t VALUES (2, 'b')")
        assert outcome.affected_rows == 1
        assert query("warehouse", "SELECT count(*) FROM t").rows[0][0] == 2

    def test_write_query_refused_on_read_only_posture(self, booted: Chokepoint) -> None:
        with pytest.raises(GuardRefusedError):
            write_query("reporting", "INSERT INTO t VALUES (3, 'c')")

    def test_mutation_text_refused_at_query(self, booted: Chokepoint) -> None:
        with pytest.raises(GuardRefusedError):
            query("warehouse", "DELETE FROM t")


class TestNfr114RefusalBranches:
    def test_branch_ia_undeclared_endpoint_refused_with_guidance(
        self, booted: Chokepoint
    ) -> None:
        """(i-a): a caller-supplied name/DSN reaches no backend; the
        suggestion names the discovery path and the declaration model."""
        with pytest.raises(GuardedExecutionError) as refusal:
            query("postgresql://attacker:pw@evil:5432/db", "SELECT 1")
        suggestion = refusal.value.structured.suggestion
        assert "list_endpoints()" in suggestion
        assert "operator-declared" in suggestion
        assert refusal.value.structured.retryable is False

    def test_branch_ii_undeclared_mutation_refused_with_guidance(
        self, booted: Chokepoint
    ) -> None:
        with pytest.raises(GuardedExecutionError) as refusal:
            write_query("never-declared", "INSERT INTO t VALUES (9, 'z')")
        assert "list_endpoints()" in refusal.value.structured.suggestion

    def test_refusal_message_names_nfr114(self, booted: Chokepoint) -> None:
        with pytest.raises(GuardedExecutionError) as refusal:
            query("nope", "SELECT 1")
        assert "NFR-114" in refusal.value.structured.message


class TestListEndpoints:
    def test_enumerates_every_declared_endpoint_with_state(
        self, booted: Chokepoint
    ) -> None:
        result = list_endpoints()
        assert result.columns == (
            "name",
            "backend_kind",
            "posture",
            "healthy",
            "health_detail",
        )
        by_name = {row[0]: row for row in result.rows}
        assert by_name["warehouse"][1] == "sqlite"
        assert by_name["warehouse"][2] == "read_write"
        assert by_name["reporting"][2] == "read_only"
        assert by_name["warehouse"][3] is True  # warm-up health

    def test_enumeration_is_dsn_redacted(self, tmp_path: Path) -> None:
        """NFR-110: no DSN or credential text in any enumerated field."""
        config = ConfigModel(
            endpoints={
                "pg": EndpointDeclaration(
                    name="pg",
                    dsn="postgresql://svc@db.internal:5432/prod",
                    credentials_ref="PG_SECRET",
                    posture="read_only",
                )
            }
        )
        guard = Chokepoint.boot(config, environ={"PG_SECRET": "hunter2"})
        runtime.configure_ingest(guard)
        try:
            result = list_endpoints()
            flattened = " ".join(str(v) for row in result.rows for v in row)
            assert "hunter2" not in flattened
            assert "postgresql://" not in flattened
        finally:
            runtime._CHOKEPOINT = None
            guard.shutdown()

    def test_zero_endpoints_renders_the_explicit_statement(self) -> None:
        guard = Chokepoint.boot(ConfigModel(), environ={})
        runtime.configure_ingest(guard)
        try:
            statement = list_endpoints()
            assert isinstance(statement, str)
            assert "Zero endpoints declared" in statement
            assert "operator configuration" in statement
        finally:
            runtime._CHOKEPOINT = None


class TestRuntimeSeam:
    def test_unbooted_runtime_refuses_structurally(self) -> None:
        assert runtime._CHOKEPOINT is None
        with pytest.raises(GuardedExecutionError) as refusal:
            query("any", "SELECT 1")
        assert refusal.value.structured.error_type.value == "configuration"
        assert "startup" in refusal.value.structured.message


class TestListEndpointsAcrossKinds:
    def test_store_kinds_enumerate_beside_sql(self, tmp_path: Path) -> None:
        """I-1's acceptance: kv/graph/tree/rdf endpoints appear beside
        SQL ones, each under its declared backend_kind (E8.3)."""
        ttl = tmp_path / "g.ttl"
        ttl.write_text("@prefix ex: <http://example.org/> .\n")
        config = ConfigModel(
            endpoints={
                "warehouse": EndpointDeclaration(
                    name="warehouse",
                    dsn=f"sqlite:///{tmp_path / 'w.db'}",
                    posture="read_write",
                ),
                "notes": EndpointDeclaration(
                    name="notes",
                    dsn=f"kv+sqlite:///{tmp_path / 'kv.db'}",
                    posture="read_write",
                ),
                "conf": EndpointDeclaration(
                    name="conf",
                    dsn=f"tree+sqlite:///{tmp_path / 'tree.db'}",
                    posture="read_write",
                ),
                "social": EndpointDeclaration(
                    name="social",
                    dsn=f"graph+sqlite:///{tmp_path / 'graph.db'}",
                    posture="read_write",
                ),
                "kb": EndpointDeclaration(
                    name="kb",
                    dsn=f"rdf+turtle:///{ttl}",
                    posture="read_only",
                ),
            }
        )
        guard = Chokepoint.boot(config, environ={})
        runtime.configure_ingest(guard)
        try:
            result = list_endpoints()
            kinds = {row[0]: row[1] for row in result.rows}
            assert kinds == {
                "warehouse": "sqlite",
                "notes": "kv",
                "conf": "tree",
                "social": "graph",
                "kb": "rdf",
            }
            healthy = {row[0]: row[3] for row in result.rows}
            assert all(healthy.values())
        finally:
            runtime._CHOKEPOINT = None
            guard.shutdown()
