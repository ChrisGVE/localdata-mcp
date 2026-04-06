"""Database-specific error mappers for SQLite, PostgreSQL, MySQL, DuckDB, Oracle, and MSSQL.

Each mapper translates raw database exceptions into StructuredErrorResponse
instances using backend-specific heuristics (message parsing, SQLSTATE codes,
error numbers, or exception class names).
"""

import re
from typing import Dict, List, Optional, Tuple

from .error_classification import (
    DatabaseErrorMapper,
    ErrorMapperRegistry,
    GenericDatabaseErrorMapper,
    StructuredErrorResponse,
)
from .error_handler import ErrorCategory

# ---------------------------------------------------------------------------
# SQLite
# ---------------------------------------------------------------------------


class SQLiteErrorMapper(DatabaseErrorMapper):
    """Maps sqlite3 exceptions using message-based heuristics."""

    def map_error(self, exception: Exception) -> StructuredErrorResponse:
        msg = str(exception).lower()
        if "authorization" in msg or "readonly" in msg:
            return StructuredErrorResponse(
                error_type=ErrorCategory.AUTH_ERROR,
                message=str(exception),
                suggestion="Check database file permissions or WAL mode.",
            )
        if "locked" in msg or "busy" in msg:
            return StructuredErrorResponse(
                error_type=ErrorCategory.TRANSIENT_ERROR,
                is_retryable=True,
                message=str(exception),
                suggestion="Retry after a brief wait; the database is locked.",
            )
        if "disk" in msg or "full" in msg or "no space" in msg:
            return StructuredErrorResponse(
                error_type=ErrorCategory.RESOURCE_ERROR,
                message=str(exception),
                suggestion="Free disk space or reduce write volume.",
            )
        if (
            "constraint" in msg
            or "unique" in msg
            or "foreign key" in msg
            or "not null" in msg
        ):
            return StructuredErrorResponse(
                error_type=ErrorCategory.CONSTRAINT_ERROR,
                message=str(exception),
                suggestion="The operation violates a SQLite constraint.",
            )
        if "no such table" in msg or "no such column" in msg:
            return StructuredErrorResponse(
                error_type=ErrorCategory.SCHEMA_ERROR,
                message=str(exception),
                suggestion="Verify table/column names in the SQLite database.",
            )
        if "syntax" in msg or "near" in msg:
            return StructuredErrorResponse(
                error_type=ErrorCategory.SYNTAX_ERROR,
                message=str(exception),
                suggestion="Review the SQL statement for syntax errors.",
            )
        return StructuredErrorResponse(
            error_type=ErrorCategory.QUERY_EXECUTION,
            message=str(exception),
            suggestion="Check query and database state.",
        )


# ---------------------------------------------------------------------------
# PostgreSQL
# ---------------------------------------------------------------------------

_PG_PREFIX_MAP: List[Tuple[str, ErrorCategory, bool, str]] = [
    (
        "28",
        ErrorCategory.AUTH_ERROR,
        False,
        "Check PostgreSQL user credentials and pg_hba.conf.",
    ),
    (
        "08",
        ErrorCategory.CONNECTION_ERROR,
        True,
        "Verify the PostgreSQL server is reachable.",
    ),
    (
        "23",
        ErrorCategory.CONSTRAINT_ERROR,
        False,
        "The operation violates a PostgreSQL constraint.",
    ),
    (
        "40",
        ErrorCategory.TRANSIENT_ERROR,
        True,
        "Retry the transaction; a serialisation or deadlock conflict occurred.",
    ),
    (
        "53",
        ErrorCategory.RESOURCE_ERROR,
        False,
        "The PostgreSQL server is low on resources.",
    ),
]


class PostgreSQLErrorMapper(DatabaseErrorMapper):
    """Maps PostgreSQL exceptions using SQLSTATE (pgcode) prefixes."""

    def map_error(self, exception: Exception) -> StructuredErrorResponse:
        pgcode = getattr(exception, "pgcode", None) or getattr(
            getattr(exception, "__cause__", None),
            "pgcode",
            None,
        )
        if pgcode:
            if pgcode == "42601":
                return StructuredErrorResponse(
                    error_type=ErrorCategory.SYNTAX_ERROR,
                    message=str(exception),
                    suggestion="Review the SQL syntax.",
                    database_error_code=pgcode,
                )
            if pgcode.startswith("42"):
                return StructuredErrorResponse(
                    error_type=ErrorCategory.SCHEMA_ERROR,
                    message=str(exception),
                    suggestion="Verify table/column names in the PostgreSQL schema.",
                    database_error_code=pgcode,
                )
            for prefix, cat, retryable, suggestion in _PG_PREFIX_MAP:
                if pgcode.startswith(prefix):
                    return StructuredErrorResponse(
                        error_type=cat,
                        is_retryable=retryable,
                        message=str(exception),
                        suggestion=suggestion,
                        database_error_code=pgcode,
                    )
        return GenericDatabaseErrorMapper().map_error(exception)


# ---------------------------------------------------------------------------
# MySQL / MariaDB
# ---------------------------------------------------------------------------

_MYSQL_CODE_MAP: Dict[int, Tuple[ErrorCategory, bool, str]] = {
    1044: (ErrorCategory.AUTH_ERROR, False, "Check MySQL database-level privileges."),
    1045: (ErrorCategory.AUTH_ERROR, False, "Check MySQL user credentials."),
    1146: (ErrorCategory.SCHEMA_ERROR, False, "The referenced table does not exist."),
    1054: (ErrorCategory.SCHEMA_ERROR, False, "The referenced column does not exist."),
    1064: (ErrorCategory.SYNTAX_ERROR, False, "Review the SQL syntax."),
    1205: (
        ErrorCategory.TRANSIENT_ERROR,
        True,
        "Lock wait timeout; retry the transaction.",
    ),
    1114: (
        ErrorCategory.RESOURCE_ERROR,
        False,
        "The table is full; free space or increase limits.",
    ),
    1062: (
        ErrorCategory.CONSTRAINT_ERROR,
        False,
        "Duplicate entry violates a unique constraint.",
    ),
    1452: (ErrorCategory.CONSTRAINT_ERROR, False, "Foreign key constraint violated."),
    2002: (
        ErrorCategory.CONNECTION_ERROR,
        True,
        "Cannot connect to MySQL server via socket.",
    ),
    2003: (
        ErrorCategory.CONNECTION_ERROR,
        True,
        "Cannot connect to MySQL server; check host/port.",
    ),
    2006: (
        ErrorCategory.CONNECTION_ERROR,
        True,
        "MySQL server has gone away; reconnect.",
    ),
}


class MySQLErrorMapper(DatabaseErrorMapper):
    """Maps MySQL/MariaDB exceptions using error numbers (errno)."""

    def map_error(self, exception: Exception) -> StructuredErrorResponse:
        errno = getattr(exception, "errno", None)
        if errno is None and exception.args and isinstance(exception.args[0], int):
            errno = exception.args[0]
        if errno and errno in _MYSQL_CODE_MAP:
            cat, retryable, suggestion = _MYSQL_CODE_MAP[errno]
            return StructuredErrorResponse(
                error_type=cat,
                is_retryable=retryable,
                message=str(exception),
                suggestion=suggestion,
                database_error_code=str(errno),
            )
        return GenericDatabaseErrorMapper().map_error(exception)


# ---------------------------------------------------------------------------
# DuckDB
# ---------------------------------------------------------------------------

_DUCKDB_TYPE_MAP: Dict[str, Tuple[ErrorCategory, bool, str]] = {
    "ParserException": (
        ErrorCategory.SYNTAX_ERROR,
        False,
        "Review the SQL syntax for DuckDB.",
    ),
    "CatalogException": (
        ErrorCategory.SCHEMA_ERROR,
        False,
        "The referenced catalog object does not exist.",
    ),
    "BinderException": (
        ErrorCategory.SCHEMA_ERROR,
        False,
        "A column or table reference could not be resolved.",
    ),
    "ConstraintException": (
        ErrorCategory.CONSTRAINT_ERROR,
        False,
        "A DuckDB constraint was violated.",
    ),
    "OutOfMemoryException": (
        ErrorCategory.RESOURCE_ERROR,
        False,
        "DuckDB ran out of memory; reduce query scope.",
    ),
}


class DuckDBErrorMapper(DatabaseErrorMapper):
    """Maps DuckDB exceptions by exception class name."""

    def map_error(self, exception: Exception) -> StructuredErrorResponse:
        cls_name = type(exception).__name__
        if cls_name in _DUCKDB_TYPE_MAP:
            cat, retryable, suggestion = _DUCKDB_TYPE_MAP[cls_name]
            return StructuredErrorResponse(
                error_type=cat,
                is_retryable=retryable,
                message=str(exception),
                suggestion=suggestion,
            )
        if cls_name == "IOException":
            msg = str(exception).lower()
            if "memory" in msg or "space" in msg:
                return StructuredErrorResponse(
                    error_type=ErrorCategory.RESOURCE_ERROR,
                    message=str(exception),
                    suggestion="Check available disk space or memory.",
                )
            return StructuredErrorResponse(
                error_type=ErrorCategory.CONNECTION_ERROR,
                is_retryable=True,
                message=str(exception),
                suggestion="Check file or network accessibility.",
            )
        return GenericDatabaseErrorMapper().map_error(exception)


# ---------------------------------------------------------------------------
# Register built-in mappers
# ---------------------------------------------------------------------------

ErrorMapperRegistry.register("sqlite", SQLiteErrorMapper())
ErrorMapperRegistry.register("postgresql", PostgreSQLErrorMapper())
ErrorMapperRegistry.register("postgres", PostgreSQLErrorMapper())
ErrorMapperRegistry.register("mysql", MySQLErrorMapper())
ErrorMapperRegistry.register("mariadb", MySQLErrorMapper())
ErrorMapperRegistry.register("duckdb", DuckDBErrorMapper())


# ---------------------------------------------------------------------------
# Oracle
# ---------------------------------------------------------------------------

_ORACLE_CODE_MAP: Dict[str, Tuple[ErrorCategory, bool, str]] = {
    "ORA-01017": (ErrorCategory.AUTH_ERROR, False, "Invalid username/password"),
    "ORA-01031": (ErrorCategory.AUTH_ERROR, False, "Insufficient privileges"),
    "ORA-01045": (
        ErrorCategory.AUTH_ERROR,
        False,
        "User lacks CREATE SESSION privilege",
    ),
    "ORA-00942": (
        ErrorCategory.SCHEMA_ERROR,
        False,
        "Table or view does not exist",
    ),
    "ORA-00904": (ErrorCategory.SCHEMA_ERROR, False, "Invalid identifier"),
    "ORA-00900": (ErrorCategory.SYNTAX_ERROR, False, "Invalid SQL statement"),
    "ORA-00933": (
        ErrorCategory.SYNTAX_ERROR,
        False,
        "SQL command not properly ended",
    ),
    "ORA-01653": (
        ErrorCategory.RESOURCE_ERROR,
        False,
        "Unable to extend table",
    ),
    "ORA-04031": (
        ErrorCategory.RESOURCE_ERROR,
        False,
        "Unable to allocate shared memory",
    ),
    "ORA-00060": (ErrorCategory.TRANSIENT_ERROR, True, "Deadlock detected"),
    "ORA-08177": (
        ErrorCategory.TRANSIENT_ERROR,
        True,
        "Serialization failure",
    ),
    "ORA-03113": (
        ErrorCategory.CONNECTION_ERROR,
        True,
        "End-of-file on communication channel",
    ),
    "ORA-03114": (
        ErrorCategory.CONNECTION_ERROR,
        True,
        "Not connected to Oracle",
    ),
    "ORA-12541": (ErrorCategory.CONNECTION_ERROR, True, "No listener"),
    "ORA-12154": (
        ErrorCategory.CONNECTION_ERROR,
        True,
        "TNS could not resolve connect identifier",
    ),
}

_ORACLE_SUGGESTIONS: Dict[ErrorCategory, str] = {
    ErrorCategory.AUTH_ERROR: "Check Oracle credentials and privileges",
    ErrorCategory.SCHEMA_ERROR: "Verify table/column names in Oracle catalog",
    ErrorCategory.SYNTAX_ERROR: "Check Oracle SQL syntax",
    ErrorCategory.RESOURCE_ERROR: ("Check Oracle tablespace and memory allocation"),
    ErrorCategory.TRANSIENT_ERROR: "Retry the operation",
    ErrorCategory.CONNECTION_ERROR: ("Check Oracle listener and network connectivity"),
}


class OracleErrorMapper(DatabaseErrorMapper):
    """Maps Oracle ORA-XXXXX error codes to structured error responses."""

    def map_error(self, exception: Exception) -> StructuredErrorResponse:
        msg = str(exception)
        match = re.search(r"ORA-\d{5}", msg)
        if match:
            code = match.group()
            if code in _ORACLE_CODE_MAP:
                cat, retryable, desc = _ORACLE_CODE_MAP[code]
                return StructuredErrorResponse(
                    error_type=cat,
                    is_retryable=retryable,
                    message=desc,
                    suggestion=_ORACLE_SUGGESTIONS.get(cat, "Check database state"),
                    database_error_code=code,
                )
        return GenericDatabaseErrorMapper().map_error(exception)


ErrorMapperRegistry.register("oracle", OracleErrorMapper())


# ---------------------------------------------------------------------------
# MS SQL Server
# ---------------------------------------------------------------------------

_MSSQL_CODE_MAP: Dict[int, Tuple[ErrorCategory, str]] = {
    18456: (ErrorCategory.AUTH_ERROR, "Login failed"),
    18452: (ErrorCategory.AUTH_ERROR, "Login from untrusted domain"),
    208: (ErrorCategory.SCHEMA_ERROR, "Invalid object name"),
    207: (ErrorCategory.SCHEMA_ERROR, "Invalid column name"),
    102: (ErrorCategory.SYNTAX_ERROR, "Incorrect syntax"),
    156: (ErrorCategory.SYNTAX_ERROR, "Incorrect syntax near keyword"),
    1205: (ErrorCategory.TRANSIENT_ERROR, "Transaction deadlocked"),
    1222: (ErrorCategory.TRANSIENT_ERROR, "Lock request timeout"),
    547: (ErrorCategory.CONSTRAINT_ERROR, "Constraint violation"),
    2627: (ErrorCategory.CONSTRAINT_ERROR, "Unique key violation"),
    2601: (ErrorCategory.CONSTRAINT_ERROR, "Duplicate key"),
    1105: (ErrorCategory.RESOURCE_ERROR, "Could not allocate space"),
    9002: (ErrorCategory.RESOURCE_ERROR, "Transaction log full"),
}

_MSSQL_SUGGESTIONS: Dict[ErrorCategory, str] = {
    ErrorCategory.AUTH_ERROR: "Check SQL Server credentials and login permissions",
    ErrorCategory.SCHEMA_ERROR: "Verify object and column names in the database",
    ErrorCategory.SYNTAX_ERROR: "Check T-SQL syntax",
    ErrorCategory.RESOURCE_ERROR: "Check SQL Server disk space and log file size",
    ErrorCategory.TRANSIENT_ERROR: "Retry the operation",
    ErrorCategory.CONNECTION_ERROR: "Check SQL Server availability and network",
    ErrorCategory.CONSTRAINT_ERROR: "Check data constraints and unique keys",
}


class MSSQLErrorMapper(DatabaseErrorMapper):
    """Maps MS SQL Server error numbers and severity to structured responses."""

    def map_error(self, exception: Exception) -> StructuredErrorResponse:
        errno = self._extract_errno(exception)
        severity = self._extract_severity(exception)

        if errno is not None and errno in _MSSQL_CODE_MAP:
            cat, desc = _MSSQL_CODE_MAP[errno]
        elif severity is not None and severity >= 20:
            cat, desc = ErrorCategory.CONNECTION_ERROR, "Fatal error"
        elif severity is not None and severity >= 17:
            cat, desc = ErrorCategory.RESOURCE_ERROR, "Resource error"
        else:
            return GenericDatabaseErrorMapper().map_error(exception)

        return StructuredErrorResponse(
            error_type=cat,
            is_retryable=cat
            in (ErrorCategory.TRANSIENT_ERROR, ErrorCategory.CONNECTION_ERROR),
            message=desc,
            suggestion=_MSSQL_SUGGESTIONS.get(cat, "Check database state"),
            database_error_code=str(errno) if errno is not None else None,
        )

    def _extract_errno(self, exc: Exception) -> Optional[int]:
        """Extract error number from pymssql or pyodbc exception."""
        # pymssql: exc.args = (errno, message)
        if hasattr(exc, "args") and exc.args and isinstance(exc.args[0], int):
            return exc.args[0]
        # Try parsing from message string
        m = re.search(r"Msg (\d+)", str(exc))
        return int(m.group(1)) if m else None

    def _extract_severity(self, exc: Exception) -> Optional[int]:
        """Extract severity level from exception message."""
        m = re.search(r"Severity (\d+)", str(exc))
        return int(m.group(1)) if m else None


ErrorMapperRegistry.register("mssql", MSSQLErrorMapper())
ErrorMapperRegistry.register("sqlserver", MSSQLErrorMapper())
