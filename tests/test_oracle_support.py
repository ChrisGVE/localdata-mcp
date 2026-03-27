"""Tests for Oracle Database support (oracledb driver, enum, error mapper)."""

import sys
from unittest.mock import patch

import pytest

from localdata_mcp.config_manager import DatabaseType
from localdata_mcp.error_classification import (
    ErrorMapperRegistry,
    StructuredErrorResponse,
)
from localdata_mcp.error_handler import ErrorCategory
from localdata_mcp.error_mappers import OracleErrorMapper


# ---------------------------------------------------------------------------
# 138.1-2: Import guard and DatabaseType enum
# ---------------------------------------------------------------------------


def test_oracledb_import_guard():
    """ORACLEDB_AVAILABLE is a bool regardless of whether oracledb is installed."""
    from localdata_mcp.oracle_support import ORACLEDB_AVAILABLE

    assert isinstance(ORACLEDB_AVAILABLE, bool)


def test_create_oracle_engine_no_driver():
    """create_oracle_engine raises ImportError when oracledb is unavailable."""
    import localdata_mcp.oracle_support as mod

    original = mod.ORACLEDB_AVAILABLE
    mod.ORACLEDB_AVAILABLE = False
    try:
        with pytest.raises(ImportError, match="oracledb"):
            mod.create_oracle_engine("oracle+oracledb://user:pass@host/db")
    finally:
        mod.ORACLEDB_AVAILABLE = original


def test_oracle_enum_exists():
    """DatabaseType('oracle') resolves to ORACLE."""
    assert DatabaseType("oracle") is DatabaseType.ORACLE


# ---------------------------------------------------------------------------
# 138.10-12: OracleErrorMapper
# ---------------------------------------------------------------------------


@pytest.fixture
def mapper():
    return OracleErrorMapper()


def test_oracle_error_mapper_auth(mapper):
    """ORA-01017 maps to AUTH_ERROR."""
    exc = Exception("ORA-01017: invalid username/password; logon denied")
    resp = mapper.map_error(exc)
    assert resp.error_type == ErrorCategory.AUTH_ERROR
    assert resp.database_error_code == "ORA-01017"
    assert not resp.is_retryable


def test_oracle_error_mapper_schema(mapper):
    """ORA-00942 maps to SCHEMA_ERROR."""
    exc = Exception("ORA-00942: table or view does not exist")
    resp = mapper.map_error(exc)
    assert resp.error_type == ErrorCategory.SCHEMA_ERROR
    assert resp.database_error_code == "ORA-00942"


def test_oracle_error_mapper_syntax(mapper):
    """ORA-00900 maps to SYNTAX_ERROR."""
    exc = Exception("ORA-00900: invalid SQL statement")
    resp = mapper.map_error(exc)
    assert resp.error_type == ErrorCategory.SYNTAX_ERROR
    assert resp.database_error_code == "ORA-00900"


def test_oracle_error_mapper_transient(mapper):
    """ORA-00060 maps to TRANSIENT_ERROR and is retryable."""
    exc = Exception("ORA-00060: deadlock detected while waiting for resource")
    resp = mapper.map_error(exc)
    assert resp.error_type == ErrorCategory.TRANSIENT_ERROR
    assert resp.is_retryable is True
    assert resp.database_error_code == "ORA-00060"


def test_oracle_error_mapper_connection(mapper):
    """ORA-03113 maps to CONNECTION_ERROR and is retryable."""
    exc = Exception("ORA-03113: end-of-file on communication channel")
    resp = mapper.map_error(exc)
    assert resp.error_type == ErrorCategory.CONNECTION_ERROR
    assert resp.is_retryable is True
    assert resp.database_error_code == "ORA-03113"


def test_oracle_error_mapper_resource(mapper):
    """ORA-01653 maps to RESOURCE_ERROR."""
    exc = Exception("ORA-01653: unable to extend table USERS")
    resp = mapper.map_error(exc)
    assert resp.error_type == ErrorCategory.RESOURCE_ERROR
    assert resp.database_error_code == "ORA-01653"


def test_oracle_error_mapper_unknown_code(mapper):
    """An unrecognised ORA code falls back to generic mapper."""
    exc = Exception("ORA-99999: some unknown oracle error")
    resp = mapper.map_error(exc)
    # Should still return a valid response (from generic mapper)
    assert isinstance(resp, StructuredErrorResponse)
    assert resp.database_error_code is None  # generic mapper does not set this


def test_oracle_error_mapper_no_ora_code(mapper):
    """A message without any ORA pattern falls back to generic mapper."""
    exc = Exception("Something went wrong with Oracle")
    resp = mapper.map_error(exc)
    assert isinstance(resp, StructuredErrorResponse)
    assert resp.database_error_code is None


def test_oracle_mapper_registered():
    """The Oracle mapper is registered in ErrorMapperRegistry."""
    mapper = ErrorMapperRegistry.get_mapper("oracle")
    assert mapper is not None
    assert isinstance(mapper, OracleErrorMapper)
