"""Tests for database-specific error mappers (subtasks 132.5-132.8)."""

import pytest

from localdata_mcp.error_classification import (
    ErrorMapperRegistry,
    GenericDatabaseErrorMapper,
)
from localdata_mcp.error_handler import ErrorCategory
from localdata_mcp.error_mappers import (
    DuckDBErrorMapper,
    MySQLErrorMapper,
    PostgreSQLErrorMapper,
    SQLiteErrorMapper,
)

# ---------------------------------------------------------------------------
# 132.5 -- SQLiteErrorMapper
# ---------------------------------------------------------------------------


class TestSQLiteErrorMapper:
    mapper = SQLiteErrorMapper()

    def test_auth_error(self) -> None:
        resp = self.mapper.map_error(Exception("authorization denied"))
        assert resp.error_type is ErrorCategory.AUTH_ERROR
        assert resp.is_retryable is False

    def test_readonly_auth_error(self) -> None:
        resp = self.mapper.map_error(Exception("attempt to write a readonly database"))
        assert resp.error_type is ErrorCategory.AUTH_ERROR

    def test_locked_error(self) -> None:
        resp = self.mapper.map_error(Exception("database is locked"))
        assert resp.error_type is ErrorCategory.TRANSIENT_ERROR
        assert resp.is_retryable is True

    def test_busy_error(self) -> None:
        resp = self.mapper.map_error(Exception("database is busy"))
        assert resp.error_type is ErrorCategory.TRANSIENT_ERROR
        assert resp.is_retryable is True

    def test_constraint_error(self) -> None:
        resp = self.mapper.map_error(Exception("UNIQUE constraint failed: users.email"))
        assert resp.error_type is ErrorCategory.CONSTRAINT_ERROR

    def test_not_null_constraint(self) -> None:
        resp = self.mapper.map_error(Exception("NOT NULL constraint failed"))
        assert resp.error_type is ErrorCategory.CONSTRAINT_ERROR

    def test_schema_error(self) -> None:
        resp = self.mapper.map_error(Exception("no such table: users"))
        assert resp.error_type is ErrorCategory.SCHEMA_ERROR

    def test_syntax_error(self) -> None:
        resp = self.mapper.map_error(Exception('near "SELCT": syntax error'))
        assert resp.error_type is ErrorCategory.SYNTAX_ERROR

    def test_resource_error(self) -> None:
        resp = self.mapper.map_error(Exception("disk I/O error"))
        assert resp.error_type is ErrorCategory.RESOURCE_ERROR

    def test_unknown_fallback(self) -> None:
        resp = self.mapper.map_error(Exception("something unexpected"))
        assert resp.error_type is ErrorCategory.QUERY_EXECUTION
        assert resp.message == "something unexpected"


# ---------------------------------------------------------------------------
# 132.6 -- PostgreSQLErrorMapper
# ---------------------------------------------------------------------------


class _PgException(Exception):
    """Mock exception with pgcode attribute."""

    def __init__(self, msg: str, pgcode: str) -> None:
        super().__init__(msg)
        self.pgcode = pgcode


class TestPostgreSQLErrorMapper:
    mapper = PostgreSQLErrorMapper()

    def test_auth_sqlstate_28000(self) -> None:
        resp = self.mapper.map_error(_PgException("auth failed", "28000"))
        assert resp.error_type is ErrorCategory.AUTH_ERROR
        assert resp.database_error_code == "28000"

    def test_schema_42P01(self) -> None:
        resp = self.mapper.map_error(_PgException("undefined table", "42P01"))
        assert resp.error_type is ErrorCategory.SCHEMA_ERROR
        assert resp.database_error_code == "42P01"

    def test_syntax_42601(self) -> None:
        resp = self.mapper.map_error(_PgException("syntax error", "42601"))
        assert resp.error_type is ErrorCategory.SYNTAX_ERROR
        assert resp.database_error_code == "42601"

    def test_deadlock_40P01(self) -> None:
        resp = self.mapper.map_error(_PgException("deadlock detected", "40P01"))
        assert resp.error_type is ErrorCategory.TRANSIENT_ERROR
        assert resp.is_retryable is True

    def test_constraint_23505(self) -> None:
        resp = self.mapper.map_error(_PgException("unique violation", "23505"))
        assert resp.error_type is ErrorCategory.CONSTRAINT_ERROR

    def test_connection_08001(self) -> None:
        resp = self.mapper.map_error(_PgException("connection refused", "08001"))
        assert resp.error_type is ErrorCategory.CONNECTION_ERROR
        assert resp.is_retryable is True

    def test_resource_53000(self) -> None:
        resp = self.mapper.map_error(_PgException("out of resources", "53000"))
        assert resp.error_type is ErrorCategory.RESOURCE_ERROR

    def test_no_pgcode_fallback(self) -> None:
        resp = self.mapper.map_error(Exception("something generic"))
        assert resp.error_type is ErrorCategory.QUERY_EXECUTION

    def test_pgcode_from_cause(self) -> None:
        cause = _PgException("cause", "28P01")
        wrapper = Exception("wrapper")
        wrapper.__cause__ = cause
        resp = self.mapper.map_error(wrapper)
        assert resp.error_type is ErrorCategory.AUTH_ERROR


# ---------------------------------------------------------------------------
# 132.7 -- MySQLErrorMapper
# ---------------------------------------------------------------------------


class _MySQLException(Exception):
    """Mock exception with errno attribute."""

    def __init__(self, msg: str, errno: int) -> None:
        super().__init__(msg)
        self.errno = errno


class TestMySQLErrorMapper:
    mapper = MySQLErrorMapper()

    def test_auth_1045(self) -> None:
        resp = self.mapper.map_error(_MySQLException("access denied", 1045))
        assert resp.error_type is ErrorCategory.AUTH_ERROR
        assert resp.database_error_code == "1045"

    def test_schema_1146(self) -> None:
        resp = self.mapper.map_error(_MySQLException("table not found", 1146))
        assert resp.error_type is ErrorCategory.SCHEMA_ERROR

    def test_syntax_1064(self) -> None:
        resp = self.mapper.map_error(_MySQLException("sql syntax error", 1064))
        assert resp.error_type is ErrorCategory.SYNTAX_ERROR

    def test_lock_timeout_1205(self) -> None:
        resp = self.mapper.map_error(_MySQLException("lock wait timeout", 1205))
        assert resp.error_type is ErrorCategory.TRANSIENT_ERROR
        assert resp.is_retryable is True

    def test_constraint_1062(self) -> None:
        resp = self.mapper.map_error(_MySQLException("duplicate entry", 1062))
        assert resp.error_type is ErrorCategory.CONSTRAINT_ERROR

    def test_connection_2003(self) -> None:
        resp = self.mapper.map_error(_MySQLException("can't connect", 2003))
        assert resp.error_type is ErrorCategory.CONNECTION_ERROR
        assert resp.is_retryable is True

    def test_resource_1114(self) -> None:
        resp = self.mapper.map_error(_MySQLException("table is full", 1114))
        assert resp.error_type is ErrorCategory.RESOURCE_ERROR

    def test_errno_from_args(self) -> None:
        exc = Exception(1045, "access denied")
        resp = self.mapper.map_error(exc)
        assert resp.error_type is ErrorCategory.AUTH_ERROR

    def test_no_errno_fallback(self) -> None:
        resp = self.mapper.map_error(Exception("something unknown"))
        assert resp.error_type is ErrorCategory.QUERY_EXECUTION


# ---------------------------------------------------------------------------
# 132.8 -- DuckDBErrorMapper
# ---------------------------------------------------------------------------


def _make_duckdb_exc(cls_name: str, msg: str) -> Exception:
    """Create an exception whose __class__.__name__ is *cls_name*."""
    exc_cls = type(cls_name, (Exception,), {})
    return exc_cls(msg)


class TestDuckDBErrorMapper:
    mapper = DuckDBErrorMapper()

    def test_parser_exception(self) -> None:
        resp = self.mapper.map_error(_make_duckdb_exc("ParserException", "bad sql"))
        assert resp.error_type is ErrorCategory.SYNTAX_ERROR

    def test_catalog_exception(self) -> None:
        resp = self.mapper.map_error(_make_duckdb_exc("CatalogException", "no table"))
        assert resp.error_type is ErrorCategory.SCHEMA_ERROR

    def test_binder_exception(self) -> None:
        resp = self.mapper.map_error(_make_duckdb_exc("BinderException", "no col"))
        assert resp.error_type is ErrorCategory.SCHEMA_ERROR

    def test_constraint_exception(self) -> None:
        resp = self.mapper.map_error(
            _make_duckdb_exc("ConstraintException", "pk violated"),
        )
        assert resp.error_type is ErrorCategory.CONSTRAINT_ERROR

    def test_out_of_memory(self) -> None:
        resp = self.mapper.map_error(
            _make_duckdb_exc("OutOfMemoryException", "oom"),
        )
        assert resp.error_type is ErrorCategory.RESOURCE_ERROR

    def test_ioexception_resource(self) -> None:
        resp = self.mapper.map_error(
            _make_duckdb_exc("IOException", "no space left"),
        )
        assert resp.error_type is ErrorCategory.RESOURCE_ERROR

    def test_ioexception_connection(self) -> None:
        resp = self.mapper.map_error(
            _make_duckdb_exc("IOException", "cannot open file"),
        )
        assert resp.error_type is ErrorCategory.CONNECTION_ERROR
        assert resp.is_retryable is True

    def test_unknown_exception_fallback(self) -> None:
        resp = self.mapper.map_error(
            _make_duckdb_exc("InternalException", "bug"),
        )
        assert resp.error_type is ErrorCategory.QUERY_EXECUTION


# ---------------------------------------------------------------------------
# Mapper registration
# ---------------------------------------------------------------------------


class TestMapperRegistration:
    def setup_method(self) -> None:
        """Re-register all mappers (previous tests may have cleared the registry)."""
        ErrorMapperRegistry.clear()
        ErrorMapperRegistry.register("generic", GenericDatabaseErrorMapper())
        ErrorMapperRegistry.register("sqlite", SQLiteErrorMapper())
        ErrorMapperRegistry.register("postgresql", PostgreSQLErrorMapper())
        ErrorMapperRegistry.register("postgres", PostgreSQLErrorMapper())
        ErrorMapperRegistry.register("mysql", MySQLErrorMapper())
        ErrorMapperRegistry.register("mariadb", MySQLErrorMapper())
        ErrorMapperRegistry.register("duckdb", DuckDBErrorMapper())

    def test_all_mappers_registered(self) -> None:
        for db_type in (
            "sqlite",
            "postgresql",
            "postgres",
            "mysql",
            "mariadb",
            "duckdb",
            "generic",
        ):
            assert (
                ErrorMapperRegistry.get_mapper(db_type) is not None
            ), f"{db_type} not registered"

    def test_get_or_default_returns_generic(self) -> None:
        mapper = ErrorMapperRegistry.get_or_default("oracle")
        assert isinstance(mapper, GenericDatabaseErrorMapper)
