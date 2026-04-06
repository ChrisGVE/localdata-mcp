"""Integration tests for the structured error classification system.

Verifies end-to-end behaviour: real database errors flow through the
classification pipeline and produce structured, JSON-serializable responses
with the correct category, retryability, and suggestion fields.
"""

import json

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.pool import StaticPool

from localdata_mcp.error_classification import (
    StructuredErrorResponse,
    classify_error,
    get_error_suggestion,
    is_error_retryable,
)
from localdata_mcp.error_handler import (
    DatabaseConnectionError,
    ErrorCategory,
    ErrorHandler,
    LocalDataError,
    QueryExecutionError,
    QueryTimeoutError,
    ResourceExhaustionError,
    SecurityViolationError,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sqlite_engine():
    """Return a disposable in-memory SQLite engine."""
    return create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )


def _execute(engine, sql: str):
    """Execute raw SQL on *engine* and return the result rows."""
    with engine.connect() as conn:
        return conn.execute(text(sql))


# ---------------------------------------------------------------------------
# 1. Syntax error via SQLite
# ---------------------------------------------------------------------------


class TestExecuteQuerySyntaxError:
    """A malformed SQL statement should classify as SYNTAX_ERROR."""

    def test_syntax_error_classification(self):
        engine = _sqlite_engine()
        with pytest.raises(Exception) as exc_info:
            _execute(engine, "SELEC * FROM nowhere")

        resp = classify_error(exc_info.value, db_type="sqlite")

        assert resp.error is True
        assert resp.error_type == ErrorCategory.SYNTAX_ERROR
        assert resp.is_retryable is False
        assert resp.message  # non-empty
        assert resp.suggestion  # non-empty


# ---------------------------------------------------------------------------
# 2. Schema error (non-existent table)
# ---------------------------------------------------------------------------


class TestExecuteQuerySchemaError:
    """Querying a table that does not exist should classify as SCHEMA_ERROR."""

    def test_schema_error_classification(self):
        engine = _sqlite_engine()
        with pytest.raises(Exception) as exc_info:
            _execute(engine, "SELECT * FROM nonexistent_table")

        resp = classify_error(exc_info.value, db_type="sqlite")

        assert resp.error is True
        assert resp.error_type == ErrorCategory.SCHEMA_ERROR
        assert resp.is_retryable is False
        assert "table" in resp.suggestion.lower() or "column" in resp.suggestion.lower()


# ---------------------------------------------------------------------------
# 3. Constraint violation (UNIQUE)
# ---------------------------------------------------------------------------


class TestConstraintViolation:
    """Inserting a duplicate into a UNIQUE column should be CONSTRAINT_ERROR."""

    def test_unique_constraint_error(self):
        engine = _sqlite_engine()
        with engine.connect() as conn:
            conn.execute(
                text("CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT UNIQUE)")
            )
            conn.execute(text("INSERT INTO items (id, name) VALUES (1, 'alpha')"))
            conn.commit()

            with pytest.raises(Exception) as exc_info:
                conn.execute(text("INSERT INTO items (id, name) VALUES (2, 'alpha')"))

        resp = classify_error(exc_info.value, db_type="sqlite")

        assert resp.error_type == ErrorCategory.CONSTRAINT_ERROR
        assert resp.is_retryable is False


# ---------------------------------------------------------------------------
# 4. Connection error (invalid SQLite path)
# ---------------------------------------------------------------------------


class TestConnectDatabaseInvalidPath:
    """Connecting to a non-existent SQLite path triggers a connection-style error."""

    def test_invalid_path_error(self):
        # Simulate the error the application layer would surface when the
        # file path is unreachable.  SQLite itself silently creates files,
        # so we construct the exception that the connection manager raises.
        # The generic mapper picks up "connection" in the message.
        exc = Exception("unable to open database file: connection refused")
        resp = classify_error(exc, db_type="generic")

        assert resp.error_type == ErrorCategory.CONNECTION_ERROR
        assert resp.is_retryable is True


# ---------------------------------------------------------------------------
# 5. StructuredErrorResponse is JSON-serializable
# ---------------------------------------------------------------------------


class TestStructuredResponseJsonSerializable:
    """to_dict() output must survive a round-trip through json.dumps/loads."""

    def test_to_dict_serializable(self):
        resp = StructuredErrorResponse(
            error_type=ErrorCategory.SYNTAX_ERROR,
            is_retryable=False,
            message="near 'SELEC': syntax error",
            suggestion="Check your SQL.",
            database_error_code="SQLITE_ERROR",
            database="test.db",
        )
        d = resp.to_dict()
        serialized = json.dumps(d)
        roundtripped = json.loads(serialized)

        assert roundtripped["error"] is True
        assert roundtripped["error_type"] == "syntax_error"
        assert roundtripped["is_retryable"] is False
        assert roundtripped["message"] == resp.message
        assert roundtripped["suggestion"] == resp.suggestion
        assert roundtripped["database_error_code"] == "SQLITE_ERROR"
        assert roundtripped["database"] == "test.db"


# ---------------------------------------------------------------------------
# 6. LocalDataError subclass → StructuredErrorResponse conversion
# ---------------------------------------------------------------------------


class TestLocalDataErrorConversion:
    """Each LocalDataError subclass should map to the right structured category."""

    @pytest.mark.parametrize(
        "error_cls,init_kwargs,expected_type,expected_retryable",
        [
            (
                DatabaseConnectionError,
                {"message": "host unreachable", "database_name": "mydb"},
                ErrorCategory.CONNECTION_ERROR,
                True,
            ),
            (
                QueryExecutionError,
                {"message": "division by zero"},
                ErrorCategory.QUERY_EXECUTION,
                False,
            ),
            (
                SecurityViolationError,
                {"message": "access denied"},
                ErrorCategory.AUTH_ERROR,
                False,
            ),
            (
                QueryTimeoutError,
                {"message": "query exceeded 30s", "timeout_limit": 30.0},
                ErrorCategory.TRANSIENT_ERROR,
                True,
            ),
            (
                ResourceExhaustionError,
                {"message": "out of memory"},
                ErrorCategory.RESOURCE_ERROR,
                False,
            ),
        ],
    )
    def test_to_structured_response(
        self, error_cls, init_kwargs, expected_type, expected_retryable
    ):
        err = error_cls(**init_kwargs)
        resp = err.to_structured_response()

        assert isinstance(resp, StructuredErrorResponse)
        assert resp.error is True
        assert resp.error_type == expected_type
        assert resp.is_retryable is expected_retryable
        assert resp.message == init_kwargs["message"]


# ---------------------------------------------------------------------------
# 7. ErrorHandler.classify_database_error with mock exceptions
# ---------------------------------------------------------------------------


class TestErrorHandlerClassify:
    """ErrorHandler.classify_database_error delegates to the correct mapper."""

    def setup_method(self):
        self.handler = ErrorHandler()

    def test_sqlite_syntax(self):
        exc = Exception("near 'SELEC': syntax error")
        resp = self.handler.classify_database_error(exc, db_type="sqlite")
        assert resp.error_type == ErrorCategory.SYNTAX_ERROR

    def test_generic_constraint(self):
        exc = Exception("UNIQUE constraint failed: users.email")
        resp = self.handler.classify_database_error(exc, db_type="generic")
        assert resp.error_type == ErrorCategory.CONSTRAINT_ERROR

    def test_database_name_propagation(self):
        exc = Exception("disk full")
        resp = self.handler.classify_database_error(
            exc, db_type="sqlite", database_name="production.db"
        )
        assert resp.database == "production.db"

    def test_postgresql_pgcode(self):
        """Simulate a PostgreSQL exception with a pgcode attribute."""
        exc = Exception("relation does not exist")
        exc.pgcode = "42P01"  # undefined_table
        resp = self.handler.classify_database_error(exc, db_type="postgresql")
        assert resp.error_type == ErrorCategory.SCHEMA_ERROR
        assert resp.database_error_code == "42P01"

    def test_mysql_errno(self):
        """Simulate a MySQL exception with an errno attribute."""
        exc = Exception("Duplicate entry '1' for key 'PRIMARY'")
        exc.errno = 1062
        resp = self.handler.classify_database_error(exc, db_type="mysql")
        assert resp.error_type == ErrorCategory.CONSTRAINT_ERROR
        assert resp.database_error_code == "1062"


# ---------------------------------------------------------------------------
# 8. Helper functions return consistent results
# ---------------------------------------------------------------------------


class TestHelperFunctionsConsistent:
    """classify_error, is_error_retryable, and get_error_suggestion must agree."""

    @pytest.mark.parametrize(
        "message,db_type",
        [
            ("near 'SELEC': syntax error", "sqlite"),
            ("no such table: users", "sqlite"),
            ("UNIQUE constraint failed: items.name", "sqlite"),
            ("database is locked", "sqlite"),
            ("connection refused", "generic"),
        ],
    )
    def test_helpers_agree(self, message, db_type):
        exc = Exception(message)

        full = classify_error(exc, db_type=db_type)
        retryable = is_error_retryable(exc, db_type=db_type)
        suggestion = get_error_suggestion(exc, db_type=db_type)

        assert full.is_retryable == retryable
        assert full.suggestion == suggestion
