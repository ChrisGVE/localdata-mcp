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
    import importlib

    import localdata_mcp.error_mappers as em

    importlib.reload(em)
    mapper = ErrorMapperRegistry.get_mapper("oracle")
    assert mapper is not None
    assert type(mapper).__name__ == "OracleErrorMapper"


# ---------------------------------------------------------------------------
# 138.5: TNS alias validation
# ---------------------------------------------------------------------------


def test_validate_tns_config_with_env():
    """_validate_tns_config returns True when TNS_ADMIN is set."""
    from localdata_mcp.oracle_support import _validate_tns_config

    with patch.dict("os.environ", {"TNS_ADMIN": "/opt/oracle/network/admin"}):
        assert _validate_tns_config() is True


def test_validate_tns_config_without_env():
    """_validate_tns_config returns False when TNS_ADMIN is unset."""
    from localdata_mcp.oracle_support import _validate_tns_config

    with patch.dict("os.environ", {}, clear=True):
        assert _validate_tns_config() is False


# ---------------------------------------------------------------------------
# 138.6: Wallet auth — invalid path
# ---------------------------------------------------------------------------


def test_create_oracle_engine_wallet_invalid_path():
    """create_oracle_engine raises ValueError for nonexistent wallet path."""
    import localdata_mcp.oracle_support as mod

    original = mod.ORACLEDB_AVAILABLE
    mod.ORACLEDB_AVAILABLE = True
    try:
        with patch.dict(
            "sys.modules",
            {"oracledb": sys.modules.get("oracledb", __import__("types"))},
        ):
            with pytest.raises(ValueError, match="Wallet path does not exist"):
                mod.create_oracle_engine(
                    "oracle+oracledb://user:pass@host/db",
                    auth={
                        "method": "wallet",
                        "wallet_path": "/nonexistent/wallet/path",
                    },
                )
    finally:
        mod.ORACLEDB_AVAILABLE = original


# ---------------------------------------------------------------------------
# 138.9: Certificate auth
# ---------------------------------------------------------------------------


def test_create_oracle_engine_certificate_auth():
    """create_oracle_engine passes cert/key paths in connect_args."""
    from unittest.mock import MagicMock

    import localdata_mcp.oracle_support as mod

    original = mod.ORACLEDB_AVAILABLE
    mod.ORACLEDB_AVAILABLE = True

    mock_create_engine = MagicMock()

    try:
        with patch("localdata_mcp.oracle_support.ORACLEDB_AVAILABLE", True):
            with patch.dict("sys.modules", {"oracledb": MagicMock()}):
                with patch("sqlalchemy.create_engine", mock_create_engine):
                    mod.create_oracle_engine(
                        "oracle+oracledb://user:pass@host/db",
                        auth={
                            "method": "certificate",
                            "cert_path": "/path/to/cert.pem",
                            "key_path": "/path/to/key.pem",
                        },
                    )
        call_kwargs = mock_create_engine.call_args
        connect_args = call_kwargs.kwargs.get(
            "connect_args", call_kwargs[1].get("connect_args", {})
        )
        assert connect_args["ssl_client_cert"] == "/path/to/cert.pem"
        assert connect_args["ssl_client_key"] == "/path/to/key.pem"
    finally:
        mod.ORACLEDB_AVAILABLE = original


# ---------------------------------------------------------------------------
# 138.14: Oracle EXPLAIN parser
# ---------------------------------------------------------------------------


def test_parse_explain_oracle_success():
    """parse_explain_oracle returns estimated rows from mocked output."""
    from unittest.mock import MagicMock

    from localdata_mcp._explain_parsers import parse_explain_oracle

    mock_engine = MagicMock()
    mock_conn = MagicMock()
    mock_engine.connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
    mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)

    xplan_rows = [
        ("Plan hash value: 12345",),
        ("--------------------------------------------",),
        ("| Id  | Operation         | Name  | Rows  |",),
        ("--------------------------------------------",),
        ("|   0 | SELECT STATEMENT  |       |   500 |",),
        ("|   1 |  TABLE ACCESS FULL| USERS |   500 |",),
        ("--------------------------------------------",),
    ]
    mock_result = MagicMock()
    mock_result.fetchall.return_value = xplan_rows
    mock_conn.execute.side_effect = [MagicMock(), mock_result]

    result = parse_explain_oracle(mock_engine, "SELECT * FROM users")
    assert result is not None
    assert result["estimated_rows"] == 500
    assert result["confidence"] == 0.7
    assert result["scan_type"] == "oracle_plan"


def test_parse_explain_oracle_failure():
    """parse_explain_oracle returns None on exception."""
    from unittest.mock import MagicMock

    from localdata_mcp._explain_parsers import parse_explain_oracle

    mock_engine = MagicMock()
    mock_conn = MagicMock()
    mock_engine.connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
    mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)
    mock_conn.execute.side_effect = Exception("ORA-00942: table does not exist")

    result = parse_explain_oracle(mock_engine, "SELECT * FROM nonexistent")
    assert result is None


def test_extract_oracle_rows():
    """_extract_oracle_rows parses row counts from DBMS_XPLAN output."""
    from localdata_mcp._explain_parsers import _extract_oracle_rows

    rows = [
        ("| Id  | Operation         | Name  | Rows  |",),
        ("|   0 | SELECT STATEMENT  |       |  1200 |",),
        ("|   1 |  TABLE ACCESS FULL| EMP   |  1200 |",),
    ]
    assert _extract_oracle_rows(rows) == 1200


def test_extract_oracle_rows_no_match():
    """_extract_oracle_rows returns None when no row count is found."""
    from localdata_mcp._explain_parsers import _extract_oracle_rows

    rows = [
        ("Plan hash value: 12345",),
        ("--------------------------------------------",),
    ]
    assert _extract_oracle_rows(rows) is None


# ---------------------------------------------------------------------------
# 138.17-19: Connection manager integration and health check
# ---------------------------------------------------------------------------


def test_connection_manager_oracle_type():
    """_create_enhanced_engine handles Oracle config via create_oracle_engine."""
    from unittest.mock import MagicMock
    from unittest.mock import patch as _patch

    from localdata_mcp.config_manager import DatabaseConfig, DatabaseType
    from localdata_mcp.connection_manager import EnhancedConnectionManager

    config = DatabaseConfig(
        name="ora_test",
        type=DatabaseType.ORACLE,
        connection_string="oracle+oracledb://user:pass@host/db",
        metadata={"auth": {"method": "password"}},
    )

    mock_engine = MagicMock()

    with _patch("localdata_mcp.oracle_support.ORACLEDB_AVAILABLE", True):
        with _patch(
            "localdata_mcp.oracle_support.create_oracle_engine",
            return_value=mock_engine,
        ) as mock_create:
            mgr = EnhancedConnectionManager.__new__(EnhancedConnectionManager)
            # Stub out _setup_engine_events to avoid SQLAlchemy event
            # registration on a MagicMock engine.
            mgr._setup_engine_events = MagicMock()
            engine = mgr._create_enhanced_engine("ora_test", config)
            mock_create.assert_called_once_with(
                "oracle+oracledb://user:pass@host/db",
                auth={"method": "password"},
            )
            assert engine is mock_engine


def test_connection_manager_oracle_no_driver():
    """_create_enhanced_engine raises ValueError when oracledb is missing."""
    from unittest.mock import patch as _patch

    from localdata_mcp.config_manager import DatabaseConfig, DatabaseType
    from localdata_mcp.connection_manager import EnhancedConnectionManager

    config = DatabaseConfig(
        name="ora_test",
        type=DatabaseType.ORACLE,
        connection_string="oracle+oracledb://user:pass@host/db",
    )

    with _patch("localdata_mcp.oracle_support.ORACLEDB_AVAILABLE", False):
        mgr = EnhancedConnectionManager.__new__(EnhancedConnectionManager)
        with pytest.raises(ValueError, match="oracledb"):
            mgr._create_enhanced_engine("ora_test", config)


def test_health_check_uses_dual():
    """_perform_health_check uses SELECT 1 FROM DUAL for Oracle databases."""
    from unittest.mock import MagicMock, call
    from unittest.mock import patch as _patch

    from localdata_mcp.config_manager import DatabaseConfig, DatabaseType
    from localdata_mcp.connection_manager import (
        ConnectionMetrics,
        EnhancedConnectionManager,
    )

    config = DatabaseConfig(
        name="ora_hc",
        type=DatabaseType.ORACLE,
        connection_string="oracle+oracledb://user:pass@host/db",
    )

    mock_engine = MagicMock()
    mock_conn = MagicMock()
    mock_engine.connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
    mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)

    mgr = EnhancedConnectionManager.__new__(EnhancedConnectionManager)
    mgr._engines = {"ora_hc": mock_engine}
    mgr._db_configs = {"ora_hc": config}
    mgr._metrics = {"ora_hc": ConnectionMetrics()}
    mgr._lock = __import__("threading").RLock()

    result = mgr._perform_health_check("ora_hc")

    # Verify the SQL executed was SELECT 1 FROM DUAL
    executed_sql = mock_conn.execute.call_args[0][0]
    assert str(executed_sql) == "SELECT 1 FROM DUAL"
    assert result.is_healthy


# ---------------------------------------------------------------------------
# 138.18: _get_engine Oracle integration
# ---------------------------------------------------------------------------


def test_get_engine_oracle():
    """_get_engine returns engine from create_oracle_engine for oracle type."""
    from unittest.mock import MagicMock
    from unittest.mock import patch as _patch

    mock_engine = MagicMock()

    with _patch("localdata_mcp.localdata_mcp.ORACLEDB_AVAILABLE", True):
        with _patch(
            "localdata_mcp.oracle_support.create_oracle_engine",
            return_value=mock_engine,
        ) as mock_create:
            from localdata_mcp.localdata_mcp import DatabaseManager

            mcp_instance = DatabaseManager.__new__(DatabaseManager)
            engine = mcp_instance._get_engine("oracle", "oracle+oracledb://u:p@h/d")
            mock_create.assert_called_once_with("oracle+oracledb://u:p@h/d", auth=None)
            assert engine is mock_engine


def test_get_engine_oracle_no_driver():
    """_get_engine raises ValueError when oracledb is unavailable."""
    from unittest.mock import patch as _patch

    with _patch("localdata_mcp.localdata_mcp.ORACLEDB_AVAILABLE", False):
        from localdata_mcp.localdata_mcp import DatabaseManager

        mcp_instance = DatabaseManager.__new__(DatabaseManager)
        with pytest.raises(ValueError, match="oracledb"):
            mcp_instance._get_engine("oracle", "oracle+oracledb://u:p@h/d")


def test_oracle_auth_from_sheet_name_json():
    """_get_engine parses JSON auth from sheet_name parameter."""
    import json
    from unittest.mock import MagicMock
    from unittest.mock import patch as _patch

    auth_dict = {"method": "wallet", "wallet_path": "/opt/wallet"}
    mock_engine = MagicMock()

    with _patch("localdata_mcp.localdata_mcp.ORACLEDB_AVAILABLE", True):
        with _patch(
            "localdata_mcp.oracle_support.create_oracle_engine",
            return_value=mock_engine,
        ) as mock_create:
            from localdata_mcp.localdata_mcp import DatabaseManager

            mcp_instance = DatabaseManager.__new__(DatabaseManager)
            engine = mcp_instance._get_engine(
                "oracle",
                "oracle+oracledb://u:p@h/d",
                sheet_name=json.dumps(auth_dict),
            )
            mock_create.assert_called_once_with(
                "oracle+oracledb://u:p@h/d", auth=auth_dict
            )
            assert engine is mock_engine
