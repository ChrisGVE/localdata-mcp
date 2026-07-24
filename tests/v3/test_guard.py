"""tests/v3/test_guard.py — E6.1, the keystone entrypoints end to end.

Real sqlite endpoints through the real NX-5: entrypoint discipline
(reads via guarded_query on any posture, writes via guarded_mutation on
read_write only — NFR-113), the E6.2/E6.3 screen refusing before any
connection is touched, NFR-108 containment on extracted path literals
(fail-closed on the empty default), the E6.5 dynamic admission live on
the wire, the capability-narrow Result (GP3), backend failures crossing
NX-3's wire with the E4.0 fault signal, and the streaming handoff to
the E6.6 registry with the pinned connection released on close.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
from sqlalchemy import text

from localdata_mcp.nexus.chokepoint.guard import (
    Chokepoint,
    GuardedExecutionError,
    GuardRefusedError,
    QueryRequest,
    Result,
)
from localdata_mcp.nexus.chokepoint.path_contain import PathRefusedError
from localdata_mcp.nexus.chokepoint.resource_bounds import ResourceRefusedError
from localdata_mcp.nexus.chokepoint.sparql_validate import SparqlRefusedError
from localdata_mcp.nexus.chokepoint.sql_validate.walker import SqlRefusedError
from localdata_mcp.nexus.config.endpoints import EndpointDeclaration, Posture
from localdata_mcp.nexus.config.models import (
    ConfigModel,
    QueryConfig,
    ResourcesConfig,
    SecurityConfig,
)
from localdata_mcp.nexus.error.model import StructuredError
from localdata_mcp.nexus.persistence.manager import PersistenceNexus

_SMALL_CEILING = 16_384  # derives max_analysis_rows == 2 (ceiling // 8192)


def _endpoint(name: str, tmp_path: Path, posture: Posture) -> EndpointDeclaration:
    """Both endpoints share one database file, so the read_only one can
    see data seeded through the read_write one."""
    return EndpointDeclaration(
        name=name, dsn=f"sqlite:///{tmp_path / 'shared.db'}", posture=posture
    )


def build_config(
    tmp_path: Path,
    *,
    allowed_paths: tuple[str, ...] = (),
    ceiling: int | None = None,
    chunk_size: int | None = None,
) -> ConfigModel:
    resources = (
        ResourcesConfig(memory_ceiling_bytes=ceiling) if ceiling else ResourcesConfig()
    )
    query = QueryConfig(default_chunk_size=chunk_size) if chunk_size else QueryConfig()
    return ConfigModel(
        resources=resources,
        query=query,
        security=SecurityConfig(allowed_paths=allowed_paths),
        endpoints={
            "rw": _endpoint("rw", tmp_path, "read_write"),
            "ro": _endpoint("ro", tmp_path, "read_only"),
        },
    )


def build_stack(
    tmp_path: Path, config: ConfigModel | None = None
) -> tuple[Chokepoint, PersistenceNexus]:
    config = config or build_config(tmp_path)
    persistence = PersistenceNexus(config, environ={})
    persistence.warm_up()
    _seed(persistence)
    return Chokepoint(config, persistence), persistence


def _seed(persistence: PersistenceNexus) -> None:
    """Three rows via the read_write endpoint (DDL is not a guarded
    category by design — seeding is test scaffolding, through NX-5
    directly); the read_only endpoint shares the file."""
    for name in ("rw",):
        with persistence.connection(name) as connection:
            connection.execute(text("CREATE TABLE t (id INTEGER, label TEXT)"))
            for row_id, label in enumerate(("a", "b", "c")):
                connection.execute(
                    text("INSERT INTO t VALUES (:i, :l)"),
                    {"i": row_id, "l": label},
                )
            connection.commit()


class TestGuardedQuery:
    def test_select_returns_capability_narrow_result(self, tmp_path: Path) -> None:
        guard, _ = build_stack(tmp_path)
        result = guard.guarded_query(
            "rw", QueryRequest(text="SELECT id, label FROM t ORDER BY id")
        )
        assert result.columns == ("id", "label")
        assert result.rows == ((0, "a"), (1, "b"), (2, "c"))
        assert result.category == "query"
        assert result.row_count == 3

    def test_result_is_frozen_data_only(self, tmp_path: Path) -> None:
        """GP3's corollary: no execute, no engine, no cursor, no
        mutation of the returned object."""
        guard, _ = build_stack(tmp_path)
        result = guard.guarded_query("rw", QueryRequest(text="SELECT 1 AS one"))
        assert not hasattr(result, "execute")
        assert not hasattr(result, "engine")
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.category = "other"  # type: ignore[misc]

    def test_reads_are_permitted_on_read_only_posture(self, tmp_path: Path) -> None:
        guard, _ = build_stack(tmp_path)
        result = guard.guarded_query("ro", QueryRequest(text="SELECT count(*) FROM t"))
        assert result.rows[0][0] == 3

    def test_parameters_bind_never_splice(self, tmp_path: Path) -> None:
        guard, _ = build_stack(tmp_path)
        result = guard.guarded_query(
            "rw",
            QueryRequest(
                text="SELECT label FROM t WHERE id = :wanted",
                parameters={"wanted": 1},
            ),
        )
        assert result.rows == (("b",),)

    def test_mutation_statement_refused_at_the_read_entrypoint(
        self, tmp_path: Path
    ) -> None:
        guard, _ = build_stack(tmp_path)
        with pytest.raises(GuardRefusedError) as refusal:
            guard.guarded_query("rw", QueryRequest(text="DELETE FROM t"))
        assert "guarded_mutation" in str(refusal.value)

    def test_screen_refusals_precede_any_execution(self, tmp_path: Path) -> None:
        """CR-023: an allow-list refusal surfaces as the guard's one
        public refusal type (GuardRefusedError), not the walker-internal
        SqlRefusedError leaking past the seam."""
        guard, _ = build_stack(tmp_path)
        with pytest.raises(GuardRefusedError):
            guard.guarded_query("rw", QueryRequest(text="SELECT 1; SELECT 2"))
        with pytest.raises(GuardRefusedError):
            guard.guarded_query("rw", QueryRequest(text="not sql at all ("))

    def test_unknown_endpoint_propagates_nfr114(self, tmp_path: Path) -> None:
        from localdata_mcp.nexus.persistence.manager import UnknownEndpointError

        guard, _ = build_stack(tmp_path)
        with pytest.raises(UnknownEndpointError):
            guard.guarded_query("undeclared", QueryRequest(text="SELECT 1"))

    def test_bounded_query_releases_residency_on_teardown(self, tmp_path: Path) -> None:
        """CR-007: the working set charged for the query's lifetime is
        released once the result is handed off — the ledger returns to
        zero so a later query starts with full headroom."""
        guard, _ = build_stack(tmp_path)
        guard.guarded_query("rw", QueryRequest(text="SELECT * FROM t"))
        assert guard._bounds.live_residency() == 0


class TestGuardedMutation:
    def test_write_on_read_write_endpoint(self, tmp_path: Path) -> None:
        guard, _ = build_stack(tmp_path)
        result = guard.guarded_mutation(
            "rw",
            QueryRequest(
                text="INSERT INTO t VALUES (:i, :l)",
                parameters={"i": 9, "l": "z"},
            ),
        )
        assert result.category == "mutation"
        assert result.affected_rows == 1
        after = guard.guarded_query("rw", QueryRequest(text="SELECT count(*) FROM t"))
        assert after.rows[0][0] == 4

    def test_nfr113_read_only_posture_refuses_before_validation(
        self, tmp_path: Path
    ) -> None:
        guard, _ = build_stack(tmp_path)
        with pytest.raises(GuardRefusedError) as refusal:
            guard.guarded_mutation("ro", QueryRequest(text="anything at all"))
        assert "read_only" in str(refusal.value)

    def test_read_statement_refused_at_the_write_entrypoint(
        self, tmp_path: Path
    ) -> None:
        guard, _ = build_stack(tmp_path)
        with pytest.raises(GuardRefusedError) as refusal:
            guard.guarded_mutation("rw", QueryRequest(text="SELECT * FROM t"))
        assert "guarded_query" in str(refusal.value)

    def test_ddl_is_outside_every_guarded_category(self, tmp_path: Path) -> None:
        guard, _ = build_stack(tmp_path)
        with pytest.raises(GuardRefusedError):
            guard.guarded_mutation("rw", QueryRequest(text="DROP TABLE t"))


class TestPathContainment:
    def test_local_file_write_fails_closed_on_empty_allowed_paths(
        self, tmp_path: Path
    ) -> None:
        """S8 row 19 live at the entrypoint: ATTACH's path literal is
        refused before any connection is touched."""
        guard, _ = build_stack(tmp_path)
        with pytest.raises(PathRefusedError):
            guard.guarded_mutation(
                "rw",
                QueryRequest(text=f"ATTACH DATABASE '{tmp_path}/x.db' AS extra"),
            )

    def test_contained_local_file_write_passes_the_screen(self, tmp_path: Path) -> None:
        config = build_config(tmp_path, allowed_paths=(str(tmp_path),))
        guard, _ = build_stack(tmp_path, config)
        result = guard.guarded_mutation(
            "rw",
            QueryRequest(text=f"ATTACH DATABASE '{tmp_path}/x.db' AS extra"),
        )
        assert result.category == "local_file_write"

    def test_out_of_tree_literal_refused_even_when_paths_configured(
        self, tmp_path: Path
    ) -> None:
        allowed = tmp_path / "allowed"
        allowed.mkdir()
        config = build_config(tmp_path, allowed_paths=(str(allowed),))
        guard, _ = build_stack(tmp_path, config)
        with pytest.raises(PathRefusedError):
            guard.guarded_mutation(
                "rw",
                QueryRequest(text=f"ATTACH DATABASE '{tmp_path}/y.db' AS extra"),
            )

    def test_contain_path_service_standalone(self, tmp_path: Path) -> None:
        """GP3 names NX-6's path-containment service as its own seam
        (NX-8 writes, ephemeral opens)."""
        config = build_config(tmp_path, allowed_paths=(str(tmp_path),))
        guard, _ = build_stack(tmp_path, config)
        inside = guard.contain_path(tmp_path / "export.csv", mode="write")
        assert inside == (tmp_path / "export.csv").resolve()
        with pytest.raises(PathRefusedError):
            guard.contain_path("/etc/passwd", mode="write")


class TestSparqlScreen:
    def test_service_refused_on_the_read_path(self, tmp_path: Path) -> None:
        guard, _ = build_stack(tmp_path)
        with pytest.raises(SparqlRefusedError):
            guard.guarded_query(
                "rw",
                QueryRequest(
                    text=("SELECT ?s WHERE { SERVICE <http://evil> { ?s ?p ?o } }"),
                    language="sparql",
                ),
            )

    def test_update_form_refused_on_the_read_path(self, tmp_path: Path) -> None:
        guard, _ = build_stack(tmp_path)
        with pytest.raises(SparqlRefusedError):
            guard.guarded_query(
                "rw",
                QueryRequest(text="INSERT DATA { <a> <b> <c> }", language="sparql"),
            )

    def test_sparql_update_needs_read_write_posture(self, tmp_path: Path) -> None:
        guard, _ = build_stack(tmp_path)
        with pytest.raises(GuardRefusedError):
            guard.guarded_mutation(
                "ro",
                QueryRequest(text="INSERT DATA { <a> <b> <c> }", language="sparql"),
            )


class TestDynamicAdmissionOnTheWire:
    def test_over_cap_result_refuses_mid_fetch(self, tmp_path: Path) -> None:
        """S8 row 13 on the wire: three seeded rows against a derived
        cap of two — the refusal lands during retention, not after."""
        config = build_config(tmp_path, ceiling=_SMALL_CEILING)
        assert config.query.max_analysis_rows == 2
        guard, _ = build_stack(tmp_path, config)
        with pytest.raises(ResourceRefusedError):
            guard.guarded_query("rw", QueryRequest(text="SELECT * FROM t"))

    def test_within_cap_passes(self, tmp_path: Path) -> None:
        config = build_config(tmp_path, ceiling=_SMALL_CEILING)
        guard, _ = build_stack(tmp_path, config)
        result = guard.guarded_query("rw", QueryRequest(text="SELECT * FROM t LIMIT 2"))
        assert result.row_count == 2


class TestWirePath:
    def test_backend_failure_becomes_the_structured_shape(self, tmp_path: Path) -> None:
        """§4b in the guard: the raised error carries the one redacted
        StructuredError — never a bare driver traceback."""
        guard, _ = build_stack(tmp_path)
        with pytest.raises(GuardedExecutionError) as failure:
            guard.guarded_query("rw", QueryRequest(text="SELECT * FROM missing_table"))
        assert isinstance(failure.value.structured, StructuredError)
        assert failure.value.structured.error_type is not None

    def test_screen_refusals_are_not_wrapped_as_execution_errors(
        self, tmp_path: Path
    ) -> None:
        """A screen refusal is a structured guard refusal, never a
        backend GuardedExecutionError — and it is the guard's public
        GuardRefusedError, never the walker-internal SqlRefusedError
        (CR-023)."""
        guard, _ = build_stack(tmp_path)
        with pytest.raises(GuardRefusedError):
            guard.guarded_query("rw", QueryRequest(text="SELECT 1; SELECT 2"))
        assert not issubclass(SqlRefusedError, GuardRefusedError)


class TestStreamingHandoff:
    def test_stream_serves_chunks_and_close_releases_the_connection(
        self, tmp_path: Path
    ) -> None:
        config = build_config(tmp_path, chunk_size=2)
        guard, _ = build_stack(tmp_path, config)
        stream_id = guard.open_query_stream(
            "rw", QueryRequest(text="SELECT id, label FROM t ORDER BY id")
        )
        first = guard.request_chunk(stream_id, 0)
        assert list(first["id"]) == [0, 1]
        second = guard.request_chunk(stream_id, 1)
        assert list(second["label"]) == ["c"]
        status = guard.stream_status(stream_id)
        assert status.exhausted and status.total_chunks == 2
        guard.close_stream(stream_id)
        # The pinned connection went back to the pool: the endpoint
        # serves a plain query immediately.
        after = guard.guarded_query("rw", QueryRequest(text="SELECT count(*) FROM t"))
        assert after.rows[0][0] == 3

    def test_stream_refuses_mutation_text(self, tmp_path: Path) -> None:
        guard, _ = build_stack(tmp_path)
        with pytest.raises(GuardRefusedError):
            guard.open_query_stream("rw", QueryRequest(text="DELETE FROM t"))

    def test_failed_open_leaks_no_stream_or_connection(self, tmp_path: Path) -> None:
        guard, _ = build_stack(tmp_path)
        with pytest.raises(GuardedExecutionError):
            guard.open_query_stream(
                "rw", QueryRequest(text="SELECT * FROM missing_table")
            )
        # Endpoint unaffected; no stream is registered anywhere.
        result = guard.guarded_query("rw", QueryRequest(text="SELECT 1 AS one"))
        assert result.rows == ((1,),)


def test_result_shape_is_plain_data() -> None:
    result = Result(columns=("a",), rows=((1,),), category="query")
    assert result.row_count == 1
    assert result.affected_rows is None


class TestEndpointSummaryAccessor:
    def test_single_summary_carries_kind_and_posture(self, tmp_path: Path) -> None:
        guard, _ = build_stack(tmp_path)
        summary = guard.endpoint_summary("rw")
        assert summary.backend_kind == "sqlite"
        assert summary.posture == "read_write"

    def test_unknown_name_raises_the_one_resolution_error(self, tmp_path: Path) -> None:
        from localdata_mcp.nexus.chokepoint.guard import UnknownEndpointError

        guard, _ = build_stack(tmp_path)
        with pytest.raises(UnknownEndpointError):
            guard.endpoint_summary("never-declared")

    def test_summaries_are_the_per_name_view_aggregated(self, tmp_path: Path) -> None:
        guard, _ = build_stack(tmp_path)
        assert guard.endpoint_summaries() == tuple(
            guard.endpoint_summary(name) for name in ("rw", "ro")
        )
