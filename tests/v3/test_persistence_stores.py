"""tests/v3/test_persistence_stores.py — E8.3: store + rdf endpoint kinds.

The NX-5 half of I-3's design: family-prefixed DSNs (`kv+sqlite://`,
`tree+sqlite://`, `graph+sqlite://`, `rdf+turtle://`) become engine
handles — SQLite files carrying the store_schemas.py shapes (schema
ensured at read-write creation, FK cascade live, query_only on
read-only), and an rdflib graph behind the cursor-shaped RdfConnection
execution.py drives unchanged. The guard end: statements against store
kinds walk the sqlite policy through STORE_BACKEND_ALIASES; every rdf
endpoint's text crosses the E6.2b SPARQL screen BY BACKEND KIND (the
caller's language hint cannot bypass it), updates land only through
guarded_mutation on read-write posture, and `SERVICE` is refused on
the read path (S7.1's payload row for the declared-endpoint surface).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pytest
from sqlalchemy import text

from localdata_mcp.nexus.chokepoint.guard import (
    Chokepoint,
    GuardRefusedError,
    QueryRequest,
)
from localdata_mcp.nexus.config.endpoints import EndpointDeclaration
from localdata_mcp.nexus.config.models import ConfigModel
from localdata_mcp.nexus.persistence.engines import backend_kind_of, create_handle
from localdata_mcp.nexus.persistence.health import probe
from localdata_mcp.nexus.persistence.limits import limits_from_config
from localdata_mcp.nexus.persistence.rdf import (
    RdfHandle,
    RdfReadOnlyError,
    UnsupportedRdfFormatError,
    rdf_format_of,
)

TTL_FIXTURE = """\
@prefix ex: <http://example.org/> .
ex:alice ex:knows ex:bob .
ex:bob ex:name "Bob" .
"""


def _handle_for(dsn: str, posture: str = "read_write"):
    declaration = EndpointDeclaration(name="store", dsn=dsn, posture=posture)  # type: ignore[arg-type]
    return create_handle(declaration, limits_from_config(ConfigModel()), {})


class TestBackendKindOf:
    def test_family_prefix_is_the_kind(self) -> None:
        assert backend_kind_of("kv+sqlite:///f.db") == "kv"
        assert backend_kind_of("tree+sqlite:///f.db") == "tree"
        assert backend_kind_of("graph+sqlite:///f.db") == "graph"
        assert backend_kind_of("rdf+turtle:///f.ttl") == "rdf"


class TestStoreHandles:
    def test_read_write_store_carries_its_schema(self, tmp_path: Path) -> None:
        handle = _handle_for(f"tree+sqlite:///{tmp_path / 'tree.db'}")
        with handle.connect() as connection:
            names = {
                row[0]
                for row in connection.execute(
                    text("SELECT name FROM sqlite_master WHERE type = 'table'")
                )
            }
        assert {"nodes", "properties"} <= names
        handle.dispose()

    def test_graph_store_carries_the_graph_schema(self, tmp_path: Path) -> None:
        handle = _handle_for(f"graph+sqlite:///{tmp_path / 'g.db'}")
        with handle.connect() as connection:
            names = {
                row[0]
                for row in connection.execute(
                    text("SELECT name FROM sqlite_master WHERE type = 'table'")
                )
            }
        assert {"graph_nodes", "graph_edges", "graph_properties"} <= names
        handle.dispose()

    def test_kv_store_shares_the_tree_schema(self, tmp_path: Path) -> None:
        handle = _handle_for(f"kv+sqlite:///{tmp_path / 'kv.db'}")
        with handle.connect() as connection:
            found = connection.execute(
                text("SELECT name FROM sqlite_master WHERE name = 'nodes'")
            ).fetchone()
        assert found is not None
        handle.dispose()

    def test_read_only_store_creates_no_schema_and_refuses_writes(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "seeded.db"
        seeder = _handle_for(f"tree+sqlite:///{path}")
        seeder.dispose()
        handle = _handle_for(f"tree+sqlite:///{path}", posture="read_only")
        with handle.connect() as connection:
            with pytest.raises(Exception, match="query_only|readonly|attempt"):
                connection.execute(
                    text(
                        "INSERT INTO nodes (name, path, depth, created_at, "
                        "updated_at) VALUES ('a', 'a', 0, 1.0, 1.0)"
                    )
                )
        handle.dispose()

    def test_foreign_key_cascade_is_live(self, tmp_path: Path) -> None:
        """The tree schema's property cascade works — proof the connect
        listener applies `PRAGMA foreign_keys = ON` (SQLite defaults off)."""
        handle = _handle_for(f"tree+sqlite:///{tmp_path / 'fk.db'}")
        with handle.connect() as connection:
            connection.execute(
                text(
                    "INSERT INTO nodes (id, name, path, depth, created_at, "
                    "updated_at) VALUES (1, 'a', 'a', 0, 1.0, 1.0)"
                )
            )
            connection.execute(
                text(
                    "INSERT INTO properties (node_id, key, value, value_type, "
                    "created_at, updated_at) "
                    "VALUES (1, 'k', 'v', 'string', 1.0, 1.0)"
                )
            )
            connection.execute(text("DELETE FROM nodes WHERE id = 1"))
            connection.commit()
            remaining = connection.execute(
                text("SELECT COUNT(*) FROM properties")
            ).fetchone()[0]
        assert remaining == 0
        handle.dispose()


class TestRdfHandle:
    def _seeded(self, tmp_path: Path, read_only: bool = False) -> RdfHandle:
        path = tmp_path / "data.ttl"
        path.write_text(TTL_FIXTURE)
        return RdfHandle(path=str(path), format="turtle", read_only=read_only)

    def test_select_serves_columns_and_rows(self, tmp_path: Path) -> None:
        handle = self._seeded(tmp_path)
        with handle.connect() as connection:
            cursor = connection.execute(
                "SELECT ?name WHERE { ?s <http://example.org/name> ?name }"
            )
            assert [entry[0] for entry in cursor.description] == ["name"]
            assert cursor.fetchmany(16) == [("Bob",)]

    def test_ask_serves_a_boolean_row(self, tmp_path: Path) -> None:
        handle = self._seeded(tmp_path)
        with handle.connect() as connection:
            cursor = connection.execute("ASK {}")
            assert cursor.fetchmany(1) == [(True,)]

    def test_parameters_bind_never_splice(self, tmp_path: Path) -> None:
        handle = self._seeded(tmp_path)
        with handle.connect() as connection:
            cursor = connection.execute(
                "SELECT ?o WHERE { ?s <http://example.org/name> ?o }",
                {"s": "http://example.org/bob"},
            )
            assert cursor.fetchmany(2) == [("Bob",)]

    def test_update_applies_and_persists_to_the_file(self, tmp_path: Path) -> None:
        handle = self._seeded(tmp_path)
        with handle.connect() as connection:
            connection.execute(
                "INSERT DATA { <http://example.org/carol> "
                "<http://example.org/name> 'Carol' }"
            )
        assert "carol" in Path(handle.path).read_text()

    def test_read_only_handle_refuses_updates(self, tmp_path: Path) -> None:
        handle = self._seeded(tmp_path, read_only=True)
        with handle.connect() as connection:
            with pytest.raises(RdfReadOnlyError):
                connection.execute(
                    "INSERT DATA { <http://example.org/x> <http://example.org/y> 'z' }"
                )

    def test_probe_speaks_sparql(self, tmp_path: Path) -> None:
        result = probe(self._seeded(tmp_path), "rdf")
        assert result.healthy is True

    def test_format_map_refuses_unknown_sub_scheme(self) -> None:
        assert rdf_format_of("ttl") == "turtle"
        with pytest.raises(UnsupportedRdfFormatError):
            rdf_format_of("jsonld")


class TestGuardOverStores:
    @pytest.fixture()
    def guard(self, tmp_path: Path) -> Iterator[Chokepoint]:
        ttl = tmp_path / "g.ttl"
        ttl.write_text(TTL_FIXTURE)
        ttl_ro = tmp_path / "ro.ttl"
        ttl_ro.write_text(TTL_FIXTURE)
        config = ConfigModel(
            endpoints={
                "notes": EndpointDeclaration(
                    name="notes",
                    dsn=f"kv+sqlite:///{tmp_path / 'kv.db'}",
                    posture="read_write",
                ),
                "graphdb": EndpointDeclaration(
                    name="graphdb",
                    dsn=f"rdf+turtle:///{ttl}",
                    posture="read_write",
                ),
                "graphdb_ro": EndpointDeclaration(
                    name="graphdb_ro",
                    dsn=f"rdf+turtle:///{ttl_ro}",
                    posture="read_only",
                ),
            }
        )
        booted = Chokepoint.boot(config, environ={})
        yield booted
        booted.shutdown()

    def test_store_kind_statements_walk_the_sqlite_policy(
        self, guard: Chokepoint
    ) -> None:
        result = guard.guarded_query(
            "notes", QueryRequest(text="SELECT COUNT(*) FROM nodes")
        )
        assert result.rows[0][0] == 0

    def test_store_mutation_crosses_the_mutation_entrypoint(
        self, guard: Chokepoint
    ) -> None:
        outcome = guard.guarded_mutation(
            "notes",
            QueryRequest(
                text=(
                    "INSERT INTO nodes (name, path, depth, created_at, "
                    "updated_at) VALUES (:n, :p, :d, :t, :t)"
                ),
                parameters={"n": "a", "p": "a", "d": 0, "t": 1.0},
            ),
        )
        assert outcome.affected_rows == 1

    def test_rdf_read_is_sparql_by_backend_kind(self, guard: Chokepoint) -> None:
        """No language hint — the DECLARED kind routes to the screen."""
        result = guard.guarded_query(
            "graphdb",
            QueryRequest(text="SELECT ?s WHERE { ?s ?p ?o }"),
        )
        assert result.row_count > 0
        assert result.category == "query"

    def test_rdf_service_clause_refused_on_read_path(self, guard: Chokepoint) -> None:
        """S7.1's SERVICE payload row over the declared-endpoint surface."""
        with pytest.raises(Exception, match="SERVICE|refused"):
            guard.guarded_query(
                "graphdb",
                QueryRequest(
                    text=(
                        "SELECT ?s WHERE { SERVICE <http://evil.example/sparql> "
                        "{ ?s ?p ?o } }"
                    )
                ),
            )

    def test_rdf_update_crosses_mutation_on_read_write(self, guard: Chokepoint) -> None:
        outcome = guard.guarded_mutation(
            "graphdb",
            QueryRequest(
                text=(
                    "INSERT DATA { <http://example.org/dave> "
                    "<http://example.org/name> 'Dave' }"
                )
            ),
        )
        assert outcome.category == "mutation"
        follow_up = guard.guarded_query(
            "graphdb",
            QueryRequest(
                text=(
                    "ASK { <http://example.org/dave> <http://example.org/name> 'Dave' }"
                )
            ),
        )
        assert follow_up.rows == ((True,),)

    def test_rdf_update_refused_on_read_only_posture(self, guard: Chokepoint) -> None:
        with pytest.raises(GuardRefusedError, match="read_only"):
            guard.guarded_mutation(
                "graphdb_ro",
                QueryRequest(
                    text=(
                        "INSERT DATA { <http://example.org/x> "
                        "<http://example.org/y> 'z' }"
                    )
                ),
            )

    def test_rdf_update_text_refused_on_the_read_path(self, guard: Chokepoint) -> None:
        """An update form through guarded_query does not parse as a
        read — refused, never a vacuous pass (E6.2b)."""
        with pytest.raises(Exception, match="refused|parse"):
            guard.guarded_query(
                "graphdb",
                QueryRequest(
                    text=(
                        "INSERT DATA { <http://example.org/x> "
                        "<http://example.org/y> 'z' }"
                    )
                ),
            )

    def test_language_hint_cannot_bypass_the_sparql_screen(
        self, guard: Chokepoint
    ) -> None:
        """language='sql' against an rdf endpoint still hits the SPARQL
        screen — the declared kind rules (negative probe)."""
        with pytest.raises(Exception, match="refused|parse"):
            guard.guarded_query(
                "graphdb",
                QueryRequest(text="SELECT 1", language="sql"),
            )
