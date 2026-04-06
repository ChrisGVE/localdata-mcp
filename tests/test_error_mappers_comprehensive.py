"""Comprehensive parameterized test suite for database error mappers (132.18).

Tests every error pattern for SQLite, PostgreSQL, MySQL, and DuckDB mappers
plus edge cases and cross-cutting concerns (retryability, suggestion quality).
"""

import pytest

from localdata_mcp.error_classification import (
    ErrorMapperRegistry,
    GenericDatabaseErrorMapper,
    StructuredErrorResponse,
    classify_error,
    get_error_suggestion,
    is_error_retryable,
)
from localdata_mcp.error_handler import ErrorCategory
from localdata_mcp.error_mappers import (
    DuckDBErrorMapper,
    MySQLErrorMapper,
    PostgreSQLErrorMapper,
    SQLiteErrorMapper,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _PgException(Exception):
    """Mock exception with pgcode attribute."""

    def __init__(self, msg: str, pgcode: str) -> None:
        super().__init__(msg)
        self.pgcode = pgcode


class _MySQLException(Exception):
    """Mock exception with errno attribute."""

    def __init__(self, msg: str, errno: int) -> None:
        super().__init__(msg)
        self.errno = errno


def _make_duckdb_exc(cls_name: str, msg: str) -> Exception:
    """Create an exception whose __class__.__name__ is *cls_name*."""
    exc_cls = type(cls_name, (Exception,), {})
    return exc_cls(msg)


# ---------------------------------------------------------------------------
# Ensure mappers are registered for every test
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _register_mappers():
    """Re-register all mappers before each test."""
    ErrorMapperRegistry.clear()
    ErrorMapperRegistry.register("generic", GenericDatabaseErrorMapper())
    ErrorMapperRegistry.register("sqlite", SQLiteErrorMapper())
    ErrorMapperRegistry.register("postgresql", PostgreSQLErrorMapper())
    ErrorMapperRegistry.register("postgres", PostgreSQLErrorMapper())
    ErrorMapperRegistry.register("mysql", MySQLErrorMapper())
    ErrorMapperRegistry.register("mariadb", MySQLErrorMapper())
    ErrorMapperRegistry.register("duckdb", DuckDBErrorMapper())


# ===================================================================
# SQLite mapper — parameterized over 8+ patterns
# ===================================================================


_SQLITE_CASES = [
    ("authorization denied", ErrorCategory.AUTH_ERROR, False),
    ("attempt to write a readonly database", ErrorCategory.AUTH_ERROR, False),
    ("database is locked", ErrorCategory.TRANSIENT_ERROR, True),
    ("database table is busy", ErrorCategory.TRANSIENT_ERROR, True),
    ("disk I/O error", ErrorCategory.RESOURCE_ERROR, False),
    ("database or disk is full", ErrorCategory.RESOURCE_ERROR, False),
    ("no space left on device", ErrorCategory.RESOURCE_ERROR, False),
    ("UNIQUE constraint failed: users.email", ErrorCategory.CONSTRAINT_ERROR, False),
    ("NOT NULL constraint failed: t.col", ErrorCategory.CONSTRAINT_ERROR, False),
    ("FOREIGN KEY constraint failed", ErrorCategory.CONSTRAINT_ERROR, False),
    ("no such table: users", ErrorCategory.SCHEMA_ERROR, False),
    ("no such column: age", ErrorCategory.SCHEMA_ERROR, False),
    ('near "SELCT": syntax error', ErrorCategory.SYNTAX_ERROR, False),
    ("something completely unexpected", ErrorCategory.QUERY_EXECUTION, False),
]


@pytest.mark.parametrize("msg,expected_cat,expected_retry", _SQLITE_CASES)
def test_sqlite_error_classification(msg, expected_cat, expected_retry):
    mapper = SQLiteErrorMapper()
    resp = mapper.map_error(Exception(msg))
    assert resp.error_type is expected_cat
    assert resp.is_retryable is expected_retry
    assert resp.message == msg


# ===================================================================
# PostgreSQL mapper — SQLSTATE classes and subcodes
# ===================================================================


_PG_CASES = [
    ("connection refused", "08001", ErrorCategory.CONNECTION_ERROR, True),
    ("connection failure", "08006", ErrorCategory.CONNECTION_ERROR, True),
    ("integrity constraint", "23000", ErrorCategory.CONSTRAINT_ERROR, False),
    ("unique_violation", "23505", ErrorCategory.CONSTRAINT_ERROR, False),
    ("foreign_key_violation", "23503", ErrorCategory.CONSTRAINT_ERROR, False),
    ("not_null_violation", "23502", ErrorCategory.CONSTRAINT_ERROR, False),
    ("auth failed", "28000", ErrorCategory.AUTH_ERROR, False),
    ("invalid_password", "28P01", ErrorCategory.AUTH_ERROR, False),
    ("deadlock_detected", "40P01", ErrorCategory.TRANSIENT_ERROR, True),
    ("serialization_failure", "40001", ErrorCategory.TRANSIENT_ERROR, True),
    ("syntax_error", "42601", ErrorCategory.SYNTAX_ERROR, False),
    ("undefined_table", "42P01", ErrorCategory.SCHEMA_ERROR, False),
    ("undefined_column", "42703", ErrorCategory.SCHEMA_ERROR, False),
    ("insufficient_privilege", "42501", ErrorCategory.SCHEMA_ERROR, False),
    ("insufficient_resources", "53000", ErrorCategory.RESOURCE_ERROR, False),
    ("disk_full", "53100", ErrorCategory.RESOURCE_ERROR, False),
    ("out_of_memory", "53200", ErrorCategory.RESOURCE_ERROR, False),
]


@pytest.mark.parametrize("msg,pgcode,expected_cat,expected_retry", _PG_CASES)
def test_pg_error_classification(msg, pgcode, expected_cat, expected_retry):
    mapper = PostgreSQLErrorMapper()
    resp = mapper.map_error(_PgException(msg, pgcode))
    assert resp.error_type is expected_cat
    assert resp.is_retryable is expected_retry
    assert resp.database_error_code == pgcode


def test_pg_pgcode_from_cause():
    cause = _PgException("cause", "28P01")
    wrapper = Exception("wrapper")
    wrapper.__cause__ = cause
    resp = PostgreSQLErrorMapper().map_error(wrapper)
    assert resp.error_type is ErrorCategory.AUTH_ERROR
    assert resp.database_error_code == "28P01"


def test_pg_no_pgcode_falls_through():
    resp = PostgreSQLErrorMapper().map_error(Exception("some timeout thing"))
    # Falls to GenericDatabaseErrorMapper keyword matching
    assert resp.error_type is ErrorCategory.TRANSIENT_ERROR


def test_pg_no_pgcode_generic_fallback():
    resp = PostgreSQLErrorMapper().map_error(Exception("totally unknown"))
    assert resp.error_type is ErrorCategory.QUERY_EXECUTION


# ===================================================================
# MySQL mapper — error number ranges
# ===================================================================


_MYSQL_CASES = [
    (1044, ErrorCategory.AUTH_ERROR, False),
    (1045, ErrorCategory.AUTH_ERROR, False),
    (1146, ErrorCategory.SCHEMA_ERROR, False),
    (1054, ErrorCategory.SCHEMA_ERROR, False),
    (1064, ErrorCategory.SYNTAX_ERROR, False),
    (1205, ErrorCategory.TRANSIENT_ERROR, True),
    (1114, ErrorCategory.RESOURCE_ERROR, False),
    (1062, ErrorCategory.CONSTRAINT_ERROR, False),
    (1452, ErrorCategory.CONSTRAINT_ERROR, False),
    (2002, ErrorCategory.CONNECTION_ERROR, True),
    (2003, ErrorCategory.CONNECTION_ERROR, True),
    (2006, ErrorCategory.CONNECTION_ERROR, True),
]


@pytest.mark.parametrize("errno,expected_cat,expected_retry", _MYSQL_CASES)
def test_mysql_error_classification(errno, expected_cat, expected_retry):
    mapper = MySQLErrorMapper()
    resp = mapper.map_error(_MySQLException(f"error {errno}", errno))
    assert resp.error_type is expected_cat
    assert resp.is_retryable is expected_retry
    assert resp.database_error_code == str(errno)


def test_mysql_errno_from_args():
    exc = Exception(1045, "access denied")
    resp = MySQLErrorMapper().map_error(exc)
    assert resp.error_type is ErrorCategory.AUTH_ERROR


def test_mysql_unknown_errno_falls_through():
    resp = MySQLErrorMapper().map_error(_MySQLException("unknown", 9999))
    assert resp.error_type is ErrorCategory.QUERY_EXECUTION


# ===================================================================
# DuckDB mapper — exception type names
# ===================================================================


_DUCKDB_CASES = [
    ("ParserException", "bad sql", ErrorCategory.SYNTAX_ERROR, False),
    ("CatalogException", "no table", ErrorCategory.SCHEMA_ERROR, False),
    ("BinderException", "unresolved col", ErrorCategory.SCHEMA_ERROR, False),
    ("ConstraintException", "pk violated", ErrorCategory.CONSTRAINT_ERROR, False),
    ("OutOfMemoryException", "oom", ErrorCategory.RESOURCE_ERROR, False),
]


@pytest.mark.parametrize("cls_name,msg,expected_cat,expected_retry", _DUCKDB_CASES)
def test_duckdb_error_classification(cls_name, msg, expected_cat, expected_retry):
    mapper = DuckDBErrorMapper()
    resp = mapper.map_error(_make_duckdb_exc(cls_name, msg))
    assert resp.error_type is expected_cat
    assert resp.is_retryable is expected_retry


def test_duckdb_ioexception_resource():
    resp = DuckDBErrorMapper().map_error(
        _make_duckdb_exc("IOException", "no space left"),
    )
    assert resp.error_type is ErrorCategory.RESOURCE_ERROR


def test_duckdb_ioexception_memory():
    resp = DuckDBErrorMapper().map_error(
        _make_duckdb_exc("IOException", "out of memory"),
    )
    assert resp.error_type is ErrorCategory.RESOURCE_ERROR


def test_duckdb_ioexception_connection():
    resp = DuckDBErrorMapper().map_error(
        _make_duckdb_exc("IOException", "cannot open file"),
    )
    assert resp.error_type is ErrorCategory.CONNECTION_ERROR
    assert resp.is_retryable is True


def test_duckdb_unknown_exception():
    resp = DuckDBErrorMapper().map_error(
        _make_duckdb_exc("InternalException", "bug"),
    )
    assert resp.error_type is ErrorCategory.QUERY_EXECUTION


# ===================================================================
# Retryability accuracy across all mappers
# ===================================================================


_RETRYABLE_CASES = [
    ("sqlite", Exception("database is locked"), True),
    ("sqlite", Exception("UNIQUE constraint failed"), False),
    ("sqlite", Exception("disk I/O error"), False),
    ("postgresql", _PgException("deadlock", "40P01"), True),
    ("postgresql", _PgException("conn refused", "08001"), True),
    ("postgresql", _PgException("syntax", "42601"), False),
    ("mysql", _MySQLException("lock wait", 1205), True),
    ("mysql", _MySQLException("gone away", 2006), True),
    ("mysql", _MySQLException("access denied", 1045), False),
    ("duckdb", _make_duckdb_exc("IOException", "cannot open"), True),
    ("duckdb", _make_duckdb_exc("ParserException", "bad"), False),
    ("generic", Exception("timeout exceeded"), True),
    ("generic", Exception("permission denied"), False),
]


@pytest.mark.parametrize("db_type,exc,expected", _RETRYABLE_CASES)
def test_is_retryable_accuracy(db_type, exc, expected):
    assert is_error_retryable(exc, db_type) is expected


# ===================================================================
# Suggestion quality — non-empty and actionable
# ===================================================================


_SUGGESTION_CASES = [
    ("sqlite", Exception("database is locked")),
    ("sqlite", Exception("no such table: t")),
    ("postgresql", _PgException("auth", "28000")),
    ("postgresql", _PgException("syntax", "42601")),
    ("mysql", _MySQLException("access denied", 1045)),
    ("mysql", _MySQLException("table full", 1114)),
    ("duckdb", _make_duckdb_exc("ParserException", "bad")),
    ("duckdb", _make_duckdb_exc("OutOfMemoryException", "oom")),
    ("generic", Exception("some random error")),
]


@pytest.mark.parametrize("db_type,exc", _SUGGESTION_CASES)
def test_suggestion_non_empty(db_type, exc):
    suggestion = get_error_suggestion(exc, db_type)
    assert isinstance(suggestion, str)
    assert len(suggestion) > 0


# ===================================================================
# Edge cases
# ===================================================================


class TestEdgeCases:
    """Edge-case tests for classify_error and individual mappers."""

    def test_none_message_exception(self):
        """Exception whose str() returns 'None'."""
        exc = Exception(None)
        resp = classify_error(exc, "generic")
        assert resp.error is True
        assert isinstance(resp.message, str)

    def test_empty_message_exception(self):
        resp = classify_error(Exception(""), "generic")
        assert resp.error is True
        assert resp.message == ""

    def test_exception_no_attributes(self):
        """Plain Exception with no errno/pgcode/special attrs."""
        resp = classify_error(Exception("plain error"), "postgresql")
        assert resp.error is True

    def test_unknown_db_type_uses_generic(self):
        resp = classify_error(Exception("timeout"), "oracle")
        assert resp.error_type is ErrorCategory.TRANSIENT_ERROR

    def test_to_dict_structure(self):
        resp = classify_error(Exception("test"), "generic")
        d = resp.to_dict()
        assert set(d.keys()) == {
            "error",
            "error_type",
            "is_retryable",
            "message",
            "suggestion",
            "database_error_code",
            "database",
        }

    def test_structured_response_defaults(self):
        resp = StructuredErrorResponse()
        assert resp.error is True
        assert resp.is_retryable is False
        assert resp.message == ""
        assert resp.database_error_code is None
        assert resp.database is None

    def test_classify_error_preserves_original_message(self):
        msg = "UNIQUE constraint failed: users.email"
        resp = classify_error(Exception(msg), "sqlite")
        assert resp.message == msg

    def test_mysql_errno_none_and_no_int_args(self):
        """MySQL mapper with exception that has no errno and non-int args."""
        exc = Exception("no errno here")
        resp = MySQLErrorMapper().map_error(exc)
        assert resp.error is True

    def test_pg_cause_without_pgcode(self):
        """PostgreSQL mapper when __cause__ has no pgcode either."""
        cause = Exception("bare cause")
        wrapper = Exception("wrapper")
        wrapper.__cause__ = cause
        resp = PostgreSQLErrorMapper().map_error(wrapper)
        assert resp.error is True
