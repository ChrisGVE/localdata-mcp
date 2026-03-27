"""Tests for MS SQL Server support: driver detection, engine creation, and error mapping."""

import pytest

from localdata_mcp.config_manager import DatabaseType
from localdata_mcp.mssql_support import (
    PYMSSQL_AVAILABLE,
    PYODBC_AVAILABLE,
    create_mssql_engine,
)
from localdata_mcp.error_classification import (
    ErrorMapperRegistry,
    StructuredErrorResponse,
)
from localdata_mcp.error_handler import ErrorCategory
from localdata_mcp.error_mappers import MSSQLErrorMapper


# ---------------------------------------------------------------------------
# Driver import guards (139.3)
# ---------------------------------------------------------------------------


def test_pymssql_import_guard():
    """PYMSSQL_AVAILABLE is a bool reflecting whether pymssql is importable."""
    assert isinstance(PYMSSQL_AVAILABLE, bool)


def test_pyodbc_import_guard():
    """PYODBC_AVAILABLE is a bool reflecting whether pyodbc is importable."""
    assert isinstance(PYODBC_AVAILABLE, bool)


def test_create_mssql_engine_no_driver(monkeypatch):
    """ImportError raised when neither pymssql nor pyodbc is available."""
    import localdata_mcp.mssql_support as mod

    monkeypatch.setattr(mod, "PYMSSQL_AVAILABLE", False)
    monkeypatch.setattr(mod, "PYODBC_AVAILABLE", False)
    with pytest.raises(ImportError, match="pymssql"):
        create_mssql_engine("mssql://user:pass@host/db")


# ---------------------------------------------------------------------------
# DatabaseType enum (139.2)
# ---------------------------------------------------------------------------


def test_mssql_enum_exists():
    """DatabaseType('mssql') resolves without error."""
    assert DatabaseType("mssql") is DatabaseType.MSSQL


# ---------------------------------------------------------------------------
# MSSQLErrorMapper (139.14-15)
# ---------------------------------------------------------------------------


def _make_exc(*args: object) -> Exception:
    """Create an Exception with specific args tuple."""
    exc = Exception()
    exc.args = args
    return exc


def _make_msg_exc(msg: str) -> Exception:
    """Create an Exception whose string representation contains *msg*."""
    return Exception(msg)


class TestMSSQLErrorMapperCodes:
    """Verify known error-number mappings."""

    mapper = MSSQLErrorMapper()

    def test_mssql_error_mapper_auth(self):
        resp = self.mapper.map_error(_make_exc(18456, "Login failed"))
        assert resp.error_type is ErrorCategory.AUTH_ERROR
        assert not resp.is_retryable

    def test_mssql_error_mapper_schema(self):
        resp = self.mapper.map_error(_make_exc(208, "Invalid object name"))
        assert resp.error_type is ErrorCategory.SCHEMA_ERROR

    def test_mssql_error_mapper_syntax(self):
        resp = self.mapper.map_error(_make_exc(102, "Incorrect syntax"))
        assert resp.error_type is ErrorCategory.SYNTAX_ERROR

    def test_mssql_error_mapper_deadlock(self):
        resp = self.mapper.map_error(_make_exc(1205, "Deadlocked"))
        assert resp.error_type is ErrorCategory.TRANSIENT_ERROR
        assert resp.is_retryable is True

    def test_mssql_error_mapper_constraint(self):
        resp = self.mapper.map_error(_make_exc(2627, "Unique key violation"))
        assert resp.error_type is ErrorCategory.CONSTRAINT_ERROR

    def test_mssql_error_mapper_resource(self):
        resp = self.mapper.map_error(_make_exc(1105, "Could not allocate space"))
        assert resp.error_type is ErrorCategory.RESOURCE_ERROR


class TestMSSQLErrorMapperSeverity:
    """Verify severity-based fallback classification."""

    mapper = MSSQLErrorMapper()

    def test_mssql_error_mapper_severity_fatal(self):
        exc = _make_msg_exc("Msg 99999, Severity 20, something fatal")
        resp = self.mapper.map_error(exc)
        assert resp.error_type is ErrorCategory.CONNECTION_ERROR
        assert resp.is_retryable is True

    def test_mssql_error_mapper_severity_resource(self):
        exc = _make_msg_exc("Msg 99998, Severity 17, resource issue")
        resp = self.mapper.map_error(exc)
        assert resp.error_type is ErrorCategory.RESOURCE_ERROR


class TestMSSQLErrorMapperFallback:
    """Verify generic fallback for unrecognised errors."""

    mapper = MSSQLErrorMapper()

    def test_mssql_error_mapper_unknown(self):
        exc = _make_msg_exc("something completely unexpected")
        resp = self.mapper.map_error(exc)
        # Falls through to GenericDatabaseErrorMapper
        assert isinstance(resp, StructuredErrorResponse)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


class TestMSSQLErrorMapperHelpers:
    mapper = MSSQLErrorMapper()

    def test_extract_errno_from_args(self):
        exc = _make_exc(208, "Invalid object name 'foo'")
        assert self.mapper._extract_errno(exc) == 208

    def test_extract_severity_from_message(self):
        exc = _make_msg_exc("Msg 208, Severity 20, State 1")
        assert self.mapper._extract_severity(exc) == 20


# ---------------------------------------------------------------------------
# Registry (139.15)
# ---------------------------------------------------------------------------


def test_mssql_mapper_registered():
    """ErrorMapperRegistry returns the MSSQL mapper for 'mssql' key."""
    mapper = ErrorMapperRegistry.get_mapper("mssql")
    assert mapper is not None
    assert isinstance(mapper, MSSQLErrorMapper)


def test_sqlserver_alias_registered():
    """ErrorMapperRegistry returns an MSSQL mapper for 'sqlserver' alias."""
    mapper = ErrorMapperRegistry.get_mapper("sqlserver")
    assert mapper is not None
    assert isinstance(mapper, MSSQLErrorMapper)
