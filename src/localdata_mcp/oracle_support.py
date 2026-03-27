"""Oracle Database support for LocalData MCP using oracledb driver."""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

try:
    import oracledb  # noqa: F401

    ORACLEDB_AVAILABLE = True
except ImportError:
    ORACLEDB_AVAILABLE = False


def create_oracle_engine(connection_string: str, auth: Optional[Dict[str, Any]] = None):
    """Create a SQLAlchemy engine for Oracle Database.

    Supports: basic auth, TNS aliases, Oracle Wallet, Kerberos.

    Args:
        connection_string: SQLAlchemy-style connection URL for Oracle.
        auth: Optional dict with authentication method and parameters.
            Supported methods: "password" (default), "wallet", "kerberos".

    Returns:
        A SQLAlchemy Engine instance configured for Oracle.

    Raises:
        ImportError: If oracledb is not installed.
        RuntimeError: If Kerberos auth requires an unavailable Oracle Client.
    """
    if not ORACLEDB_AVAILABLE:
        raise ImportError(
            "Oracle support requires 'oracledb'. "
            "Install with: pip install localdata-mcp[enterprise]"
        )
    import oracledb as _oracledb
    from sqlalchemy import create_engine

    connect_args: Dict[str, Any] = {}

    if auth:
        method = auth.get("method", "password")
        if method == "wallet":
            wallet_path = auth.get("wallet_path", "")
            connect_args["config_dir"] = wallet_path
            connect_args["wallet_location"] = wallet_path
        elif method == "kerberos":
            try:
                _oracledb.init_oracle_client()
            except Exception as e:
                raise RuntimeError(f"Kerberos requires Oracle Client: {e}") from e

    # Ensure oracle+oracledb:// prefix
    if connection_string.startswith("oracle://"):
        connection_string = connection_string.replace(
            "oracle://", "oracle+oracledb://", 1
        )

    return create_engine(
        connection_string,
        connect_args=connect_args,
        pool_size=5,
        max_overflow=10,
        pool_pre_ping=True,
    )
