"""tests/v3/test_persistence_engines.py — E5.4 posture + limits at creation.

Real engines against tmp_path files: a read-only SQLite/DuckDB engine
refuses writes AT THE ENGINE (§8.1 `cef73b00`) — before any chokepoint
sees the statement — while read_write posture permits them. Networked
engines are built lazily (no server needed) to assert credential
injection (NFR-110) and the hard-capped pool sizing from NX-2 limits.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pytest
from sqlalchemy import text

from localdata_mcp.nexus.config.endpoints import EndpointDeclaration
from localdata_mcp.nexus.config.models import ConfigModel
from localdata_mcp.nexus.persistence.engines import (
    _POOL_CHECKOUT_WAIT_SECONDS,
    DuckDbHandle,
    EngineHandle,
    SqlAlchemyHandle,
    UnsupportedBackendError,
    apply_statement_timeout,
    backend_kind_of,
    create_handle,
)
from localdata_mcp.nexus.persistence.limits import ResourceLimits, limits_from_config

LIMITS = ResourceLimits(
    max_connections=3, statement_timeout_seconds=30, max_concurrent_streams=2
)


def sqlite_declaration(path: Path, posture: str = "read_only") -> EndpointDeclaration:
    return EndpointDeclaration(
        name="ep-sqlite",
        dsn=f"sqlite:///{path}",
        posture=posture,  # type: ignore[arg-type]
    )


class TestBackendKind:
    @pytest.mark.parametrize(
        "dsn,kind",
        [
            ("sqlite:///x.db", "sqlite"),
            ("duckdb:///x.duckdb", "duckdb"),
            ("postgresql://u@h/db", "postgresql"),
            ("postgresql+psycopg2://u@h/db", "postgresql"),
            ("MySQL://u@h/db", "mysql"),
        ],
    )
    def test_scheme_maps_to_backend_family(self, dsn: str, kind: str) -> None:
        assert backend_kind_of(dsn) == kind

    def test_unsupported_backend_is_refused(self, tmp_path: Path) -> None:
        declaration = EndpointDeclaration(name="ep", dsn="mongodb://h/db")
        with pytest.raises(UnsupportedBackendError):
            create_handle(declaration, LIMITS, {})


class TestSqlitePosture:
    def test_read_only_engine_refuses_writes(self, tmp_path: Path) -> None:
        db = tmp_path / "data.db"
        handle = create_handle(sqlite_declaration(db, "read_only"), LIMITS, {})
        assert isinstance(handle, EngineHandle)
        with handle.connect() as connection:
            with pytest.raises(Exception, match="(?i)readonly|query_only"):
                connection.execute(text("CREATE TABLE t (x INTEGER)"))
        handle.dispose()

    def test_read_only_engine_serves_reads(self, tmp_path: Path) -> None:
        db = tmp_path / "data.db"
        seed = create_handle(sqlite_declaration(db, "read_write"), LIMITS, {})
        with seed.connect() as connection:
            connection.execute(text("CREATE TABLE t (x INTEGER)"))
            connection.execute(text("INSERT INTO t VALUES (7)"))
            connection.commit()
        seed.dispose()

        handle = create_handle(sqlite_declaration(db, "read_only"), LIMITS, {})
        with handle.connect() as connection:
            assert connection.execute(text("SELECT x FROM t")).scalar() == 7
        handle.dispose()

    def test_read_write_engine_permits_writes(self, tmp_path: Path) -> None:
        db = tmp_path / "data.db"
        handle = create_handle(sqlite_declaration(db, "read_write"), LIMITS, {})
        with handle.connect() as connection:
            connection.execute(text("CREATE TABLE t (x INTEGER)"))
            connection.commit()
        handle.dispose()


class TestDuckDbPosture:
    def seeded(self, tmp_path: Path) -> Path:
        path = tmp_path / "data.duckdb"
        connection = duckdb.connect(str(path))
        connection.execute("CREATE TABLE t (x INTEGER)")
        connection.execute("INSERT INTO t VALUES (7)")
        connection.close()
        return path

    def test_read_only_handle_refuses_writes(self, tmp_path: Path) -> None:
        path = self.seeded(tmp_path)
        declaration = EndpointDeclaration(name="ep", dsn=f"duckdb:///{path}")
        handle = create_handle(declaration, LIMITS, {})
        assert isinstance(handle, DuckDbHandle) and handle.read_only
        with handle.connect() as connection:
            with pytest.raises(Exception, match="(?i)read.?only"):
                connection.execute("INSERT INTO t VALUES (8)")

    def test_read_only_handle_serves_reads(self, tmp_path: Path) -> None:
        path = self.seeded(tmp_path)
        declaration = EndpointDeclaration(name="ep", dsn=f"duckdb:///{path}")
        handle = create_handle(declaration, LIMITS, {})
        with handle.connect() as connection:
            assert connection.execute("SELECT x FROM t").fetchone() == (7,)

    def test_read_write_posture_opens_writable(self, tmp_path: Path) -> None:
        path = self.seeded(tmp_path)
        declaration = EndpointDeclaration(
            name="ep", dsn=f"duckdb:///{path}", posture="read_write"
        )
        handle = create_handle(declaration, LIMITS, {})
        with handle.connect() as connection:
            connection.execute("INSERT INTO t VALUES (8)")

    def test_each_connect_is_fresh_and_dispose_is_trivial(self, tmp_path: Path) -> None:
        path = self.seeded(tmp_path)
        handle = DuckDbHandle(path=str(path), read_only=True)
        with handle.connect() as first, handle.connect() as second:
            assert first is not second
        handle.dispose()  # nothing retained — must be a no-op, not an error


class TestNetworkedEngines:
    DECLARATION = EndpointDeclaration(
        name="ep-pg",
        dsn="postgresql://svc@db.example/warehouse",
        credentials_ref="EP_PG_SECRET",
    )

    def test_credential_injected_from_environment(self) -> None:
        handle = create_handle(self.DECLARATION, LIMITS, {"EP_PG_SECRET": "s3cret"})
        assert isinstance(handle, SqlAlchemyHandle)
        assert handle.engine.url.password == "s3cret"

    def test_pool_is_hard_capped_from_nx2_limits(self) -> None:
        handle = create_handle(self.DECLARATION, LIMITS, {"EP_PG_SECRET": "s3cret"})
        assert isinstance(handle, SqlAlchemyHandle)
        pool = handle.engine.pool
        assert pool.size() == LIMITS.max_connections
        assert pool._max_overflow == 0  # type: ignore[attr-defined]


class TestStatementTimeoutPerDialect:
    """CR-015: a REAL per-statement timeout, built per dialect. These
    assert the timeout mechanism directly (no live server needed) — the
    session SET for pg/mysql, the connection attribute for mssql/oracle."""

    class _FakeCursor:
        def __init__(self, log: list[str]) -> None:
            self._log = log

        def execute(self, statement: str) -> None:
            self._log.append(statement)

        def close(self) -> None:
            pass

    class _SessionConn:
        def __init__(self) -> None:
            self.executed: list[str] = []

        def cursor(self) -> "TestStatementTimeoutPerDialect._FakeCursor":
            return TestStatementTimeoutPerDialect._FakeCursor(self.executed)

    class _AttrConn:
        pass

    def test_postgresql_sets_statement_timeout_in_ms(self) -> None:
        conn = self._SessionConn()
        apply_statement_timeout(conn, "postgresql", 30)
        assert conn.executed == ["SET statement_timeout = 30000"]

    def test_mysql_sets_max_execution_time_in_ms(self) -> None:
        conn = self._SessionConn()
        apply_statement_timeout(conn, "mysql", 30)
        assert conn.executed == ["SET SESSION max_execution_time = 30000"]

    def test_mssql_sets_the_pyodbc_query_timeout_in_seconds(self) -> None:
        conn = self._AttrConn()
        apply_statement_timeout(conn, "mssql", 30)
        assert conn.timeout == 30  # type: ignore[attr-defined]

    def test_oracle_sets_the_call_timeout_in_ms(self) -> None:
        conn = self._AttrConn()
        apply_statement_timeout(conn, "oracle", 30)
        assert conn.call_timeout == 30000  # type: ignore[attr-defined]

    def test_non_positive_timeout_is_a_no_op(self) -> None:
        conn = self._SessionConn()
        apply_statement_timeout(conn, "postgresql", 0)
        assert conn.executed == []

    def test_networked_pool_timeout_is_the_checkout_wait_not_the_statement_value(
        self,
    ) -> None:
        """The two are decoupled (CR-015): a distinct statement timeout of
        7s must NOT become the pool checkout wait."""
        limits = ResourceLimits(
            max_connections=3, statement_timeout_seconds=7, max_concurrent_streams=2
        )
        handle = create_handle(
            TestNetworkedEngines.DECLARATION, limits, {"EP_PG_SECRET": "s3cret"}
        )
        assert isinstance(handle, SqlAlchemyHandle)
        assert handle.engine.pool._timeout == _POOL_CHECKOUT_WAIT_SECONDS  # type: ignore[attr-defined]
        assert _POOL_CHECKOUT_WAIT_SECONDS != 7


class TestLimitsFromConfig:
    def test_limits_read_the_declared_nx2_fields(self) -> None:
        config = ConfigModel()
        limits = limits_from_config(config)
        assert limits.max_connections == config.resources.max_connections_per_endpoint
        assert (
            limits.statement_timeout_seconds == config.resources.query_timeout_seconds
        )
        assert (
            limits.max_concurrent_streams
            == config.query.max_concurrent_streams_per_endpoint
        )
