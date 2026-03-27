"""MS SQL Server support for LocalData MCP using pymssql/pyodbc drivers."""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

try:
    import pymssql  # noqa: F401

    PYMSSQL_AVAILABLE = True
except ImportError:
    PYMSSQL_AVAILABLE = False

try:
    import pyodbc  # noqa: F401

    PYODBC_AVAILABLE = True
except ImportError:
    PYODBC_AVAILABLE = False


def create_mssql_engine(connection_string: str, auth: Optional[Dict[str, Any]] = None):
    """Create a SQLAlchemy engine for MS SQL Server.

    Supports: SQL Auth, Windows Integrated Auth, Azure AD, Kerberos.
    Prefers pymssql, falls back to pyodbc.

    Args:
        connection_string: SQLAlchemy-style connection URL for MS SQL Server.
        auth: Optional dict with authentication method and parameters.
            Supported methods: "password" (default), "trusted", "azure_ad",
            "kerberos", "certificate".

    Returns:
        A SQLAlchemy Engine instance configured for MS SQL Server.

    Raises:
        ImportError: If neither pymssql nor pyodbc is installed.
        ValueError: If Azure AD auth is requested without pyodbc driver.
    """
    if not PYMSSQL_AVAILABLE and not PYODBC_AVAILABLE:
        raise ImportError(
            "MS SQL support requires 'pymssql' or 'pyodbc'. "
            "Install with: pip install localdata-mcp[enterprise]"
        )
    from sqlalchemy import create_engine

    # Determine driver
    if "pymssql" in connection_string:
        driver = "pymssql"
    elif "pyodbc" in connection_string:
        driver = "pyodbc"
    elif PYMSSQL_AVAILABLE:
        driver = "pymssql"
    else:
        driver = "pyodbc"

    # Normalize connection string
    if (
        "mssql://" in connection_string
        and "+pymssql" not in connection_string
        and "+pyodbc" not in connection_string
    ):
        connection_string = connection_string.replace(
            "mssql://", f"mssql+{driver}://", 1
        )

    connect_args: Dict[str, Any] = {}

    if auth:
        method = auth.get("method", "password")
        if method == "trusted":
            if driver == "pymssql":
                # pymssql doesn't use connect_args for trusted
                pass
            else:
                connect_args["Trusted_Connection"] = "yes"
        elif method == "azure_ad":
            if driver != "pyodbc":
                raise ValueError("Azure AD authentication requires pyodbc driver")
            token = auth.get("token")
            if not token:
                raise ValueError("Azure AD auth requires 'token' in auth dict")
            connect_args["attrs_before"] = {1256: token}
        elif method == "certificate":
            connect_args["encrypt"] = "yes"
            connect_args["trustServerCertificate"] = auth.get("trust_server_cert", "no")
        elif method == "kerberos":
            connect_args["Trusted_Connection"] = "yes"

    return create_engine(
        connection_string,
        connect_args=connect_args,
        pool_size=5,
        max_overflow=10,
        pool_pre_ping=True,
    )
