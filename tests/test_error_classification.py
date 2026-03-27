"""Tests for the structured error classification system."""

import threading

import pytest

from localdata_mcp.error_handler import (
    ErrorCategory,
    ErrorHandler,
    ErrorSeverity,
    LocalDataError,
)
from localdata_mcp.error_classification import (
    DatabaseErrorMapper,
    ErrorMapperRegistry,
    GenericDatabaseErrorMapper,
    StructuredErrorResponse,
    classify_error,
    get_error_suggestion,
    is_error_retryable,
)


# ---------------------------------------------------------------------------
# 132.1 — New ErrorCategory values exist
# ---------------------------------------------------------------------------


class TestErrorCategoryExtensions:
    """Verify that all PRD-specified categories were added."""

    @pytest.mark.parametrize(
        "name,value",
        [
            ("AUTH_ERROR", "auth_error"),
            ("SCHEMA_ERROR", "schema_error"),
            ("SYNTAX_ERROR", "syntax_error"),
            ("RESOURCE_ERROR", "resource_error"),
            ("TRANSIENT_ERROR", "transient_error"),
            ("CONSTRAINT_ERROR", "constraint_error"),
            ("CONNECTION_ERROR", "connection_error"),
        ],
    )
    def test_new_category_exists(self, name: str, value: str) -> None:
        cat = ErrorCategory[name]
        assert cat.value == value

    def test_backward_compatibility_old_values(self) -> None:
        """Old enum members must still resolve correctly."""
        assert ErrorCategory.CONNECTION.value == "connection"
        assert ErrorCategory.QUERY_EXECUTION.value == "query_execution"
        assert ErrorCategory.SECURITY_VIOLATION.value == "security_violation"
        assert ErrorCategory.TIMEOUT.value == "timeout"
        assert ErrorCategory.RESOURCE_EXHAUSTION.value == "resource_exhaustion"
        assert ErrorCategory.CONFIGURATION.value == "configuration"
        assert ErrorCategory.AUTHENTICATION.value == "authentication"
        assert ErrorCategory.PERMISSION.value == "permission"
        assert ErrorCategory.DATA_VALIDATION.value == "data_validation"
        assert ErrorCategory.SYSTEM.value == "system"


# ---------------------------------------------------------------------------
# 132.2 — StructuredErrorResponse
# ---------------------------------------------------------------------------


class TestStructuredErrorResponse:
    def test_defaults(self) -> None:
        resp = StructuredErrorResponse()
        assert resp.error is True
        assert resp.error_type is ErrorCategory.QUERY_EXECUTION
        assert resp.is_retryable is False
        assert resp.message == ""
        assert resp.suggestion == ""
        assert resp.database_error_code is None
        assert resp.database is None

    def test_custom_creation(self) -> None:
        resp = StructuredErrorResponse(
            error_type=ErrorCategory.AUTH_ERROR,
            is_retryable=False,
            message="access denied",
            suggestion="check credentials",
            database_error_code="1045",
            database="mysql",
        )
        assert resp.error_type is ErrorCategory.AUTH_ERROR
        assert resp.database == "mysql"

    def test_to_dict(self) -> None:
        resp = StructuredErrorResponse(
            error_type=ErrorCategory.TRANSIENT_ERROR,
            is_retryable=True,
            message="deadlock detected",
            suggestion="retry",
            database_error_code="40P01",
            database="postgres",
        )
        d = resp.to_dict()
        assert d == {
            "error": True,
            "error_type": "transient_error",
            "is_retryable": True,
            "message": "deadlock detected",
            "suggestion": "retry",
            "database_error_code": "40P01",
            "database": "postgres",
        }

    def test_to_dict_none_fields(self) -> None:
        d = StructuredErrorResponse().to_dict()
        assert d["database_error_code"] is None
        assert d["database"] is None


# ---------------------------------------------------------------------------
# 132.4 — ErrorMapperRegistry
# ---------------------------------------------------------------------------


class TestErrorMapperRegistry:
    def setup_method(self) -> None:
        """Reset registry between tests, then re-register generic."""
        ErrorMapperRegistry.clear()
        ErrorMapperRegistry.register("generic", GenericDatabaseErrorMapper())

    def test_register_and_get(self) -> None:
        mapper = GenericDatabaseErrorMapper()
        ErrorMapperRegistry.register("sqlite", mapper)
        assert ErrorMapperRegistry.get_mapper("sqlite") is mapper

    def test_get_mapper_missing_returns_none(self) -> None:
        assert ErrorMapperRegistry.get_mapper("oracle") is None

    def test_get_or_default_returns_registered(self) -> None:
        mapper = GenericDatabaseErrorMapper()
        ErrorMapperRegistry.register("mysql", mapper)
        assert ErrorMapperRegistry.get_or_default("mysql") is mapper

    def test_get_or_default_falls_back_to_generic(self) -> None:
        result = ErrorMapperRegistry.get_or_default("unknown_db")
        assert isinstance(result, GenericDatabaseErrorMapper)

    def test_thread_safety(self) -> None:
        """Concurrent registrations must not raise."""
        errors: list = []

        def register_mapper(name: str) -> None:
            try:
                ErrorMapperRegistry.register(name, GenericDatabaseErrorMapper())
            except Exception as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=register_mapper, args=(f"db_{i}",))
            for i in range(20)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        for i in range(20):
            assert ErrorMapperRegistry.get_mapper(f"db_{i}") is not None


# ---------------------------------------------------------------------------
# 132.9 — GenericDatabaseErrorMapper
# ---------------------------------------------------------------------------


class TestGenericDatabaseErrorMapper:
    mapper = GenericDatabaseErrorMapper()

    # -- AUTH_ERROR --
    @pytest.mark.parametrize(
        "msg",
        [
            "Access denied for user 'root'",
            "permission denied for relation",
            "Login failed for user",
        ],
    )
    def test_auth_error(self, msg: str) -> None:
        resp = self.mapper.map_error(Exception(msg))
        assert resp.error_type is ErrorCategory.AUTH_ERROR
        assert resp.is_retryable is False

    # -- SCHEMA_ERROR --
    @pytest.mark.parametrize(
        "msg",
        [
            "Table 'users' not found",
            "Column 'age' does not exist",
            "Unknown column 'foo' in field list",
        ],
    )
    def test_schema_error(self, msg: str) -> None:
        resp = self.mapper.map_error(Exception(msg))
        assert resp.error_type is ErrorCategory.SCHEMA_ERROR
        assert resp.is_retryable is False

    # -- SYNTAX_ERROR --
    @pytest.mark.parametrize(
        "msg",
        [
            "You have an error in your SQL syntax",
            "parse error at position 42",
        ],
    )
    def test_syntax_error(self, msg: str) -> None:
        resp = self.mapper.map_error(Exception(msg))
        assert resp.error_type is ErrorCategory.SYNTAX_ERROR
        assert resp.is_retryable is False

    # -- TRANSIENT_ERROR (retryable) --
    @pytest.mark.parametrize(
        "msg",
        [
            "statement timeout",
            "deadlock detected",
            "could not obtain lock",
            "database is busy",
        ],
    )
    def test_transient_error(self, msg: str) -> None:
        resp = self.mapper.map_error(Exception(msg))
        assert resp.error_type is ErrorCategory.TRANSIENT_ERROR
        assert resp.is_retryable is True

    # -- RESOURCE_ERROR --
    @pytest.mark.parametrize(
        "msg",
        [
            "out of memory",
            "disk full",
            "no space left on device",
        ],
    )
    def test_resource_error(self, msg: str) -> None:
        resp = self.mapper.map_error(Exception(msg))
        assert resp.error_type is ErrorCategory.RESOURCE_ERROR
        assert resp.is_retryable is False

    # -- CONSTRAINT_ERROR --
    @pytest.mark.parametrize(
        "msg",
        [
            "constraint violation",
            "unique constraint failed",
            "foreign key constraint fails",
        ],
    )
    def test_constraint_error(self, msg: str) -> None:
        resp = self.mapper.map_error(Exception(msg))
        assert resp.error_type is ErrorCategory.CONSTRAINT_ERROR
        assert resp.is_retryable is False

    # -- CONNECTION_ERROR --
    @pytest.mark.parametrize(
        "msg",
        [
            "connection refused",
            "network is unreachable",
            "could not resolve host",
        ],
    )
    def test_connection_error(self, msg: str) -> None:
        resp = self.mapper.map_error(Exception(msg))
        assert resp.error_type is ErrorCategory.CONNECTION_ERROR
        assert resp.is_retryable is True

    # -- Default fallback --
    def test_unknown_error_defaults_to_query_execution(self) -> None:
        resp = self.mapper.map_error(Exception("something completely unexpected"))
        assert resp.error_type is ErrorCategory.QUERY_EXECUTION
        assert resp.is_retryable is False

    def test_response_contains_original_message(self) -> None:
        original = "unique constraint failed: users.email"
        resp = self.mapper.map_error(Exception(original))
        assert resp.message == original

    def test_suggestion_is_populated(self) -> None:
        resp = self.mapper.map_error(Exception("deadlock detected"))
        assert resp.suggestion != ""


# ---------------------------------------------------------------------------
# 132.11 — ErrorHandler.classify_database_error
# ---------------------------------------------------------------------------


class TestClassifyDatabaseError:
    def setup_method(self) -> None:
        """Ensure database-specific mappers are registered."""
        from localdata_mcp.error_mappers import SQLiteErrorMapper

        if ErrorMapperRegistry.get_mapper("sqlite") is None:
            ErrorMapperRegistry.register("sqlite", SQLiteErrorMapper())
        if ErrorMapperRegistry.get_mapper("generic") is None:
            ErrorMapperRegistry.register("generic", GenericDatabaseErrorMapper())

    def test_classify_with_sqlite_type(self) -> None:
        handler = ErrorHandler()
        exc = Exception("no such table: users")
        resp = handler.classify_database_error(exc, db_type="sqlite")
        assert resp.error_type is ErrorCategory.SCHEMA_ERROR

    def test_classify_with_unknown_type_uses_generic(self) -> None:
        handler = ErrorHandler()
        exc = Exception("syntax error near SELECT")
        resp = handler.classify_database_error(exc, db_type="oracle")
        assert resp.error_type is ErrorCategory.SYNTAX_ERROR

    def test_classify_sets_database_name(self) -> None:
        handler = ErrorHandler()
        exc = Exception("connection refused")
        resp = handler.classify_database_error(
            exc, db_type="generic", database_name="mydb"
        )
        assert resp.database == "mydb"


# ---------------------------------------------------------------------------
# 132.12 — Helper functions
# ---------------------------------------------------------------------------


class TestHelperFunctions:
    def test_classify_error_returns_response(self) -> None:
        resp = classify_error(Exception("syntax error"))
        assert isinstance(resp, StructuredErrorResponse)
        assert resp.error_type is ErrorCategory.SYNTAX_ERROR

    def test_is_error_retryable_true(self) -> None:
        assert is_error_retryable(Exception("deadlock detected")) is True

    def test_is_error_retryable_false(self) -> None:
        assert is_error_retryable(Exception("syntax error")) is False

    def test_get_error_suggestion_not_empty(self) -> None:
        suggestion = get_error_suggestion(Exception("connection refused"))
        assert isinstance(suggestion, str)
        assert len(suggestion) > 0


# ---------------------------------------------------------------------------
# 132.13 — LocalDataError.to_structured_response
# ---------------------------------------------------------------------------


class TestLocalDataErrorConversion:
    def test_connection_error_maps_to_connection_error(self) -> None:
        err = LocalDataError(
            message="conn lost",
            category=ErrorCategory.CONNECTION,
        )
        resp = err.to_structured_response()
        assert resp.error_type is ErrorCategory.CONNECTION_ERROR

    def test_timeout_maps_to_transient(self) -> None:
        err = LocalDataError(
            message="timed out",
            category=ErrorCategory.TIMEOUT,
        )
        resp = err.to_structured_response()
        assert resp.error_type is ErrorCategory.TRANSIENT_ERROR

    def test_resource_maps_to_resource(self) -> None:
        err = LocalDataError(
            message="out of memory",
            category=ErrorCategory.RESOURCE_EXHAUSTION,
        )
        resp = err.to_structured_response()
        assert resp.error_type is ErrorCategory.RESOURCE_ERROR

    def test_auth_maps_to_auth_error(self) -> None:
        err = LocalDataError(
            message="bad creds",
            category=ErrorCategory.AUTHENTICATION,
        )
        resp = err.to_structured_response()
        assert resp.error_type is ErrorCategory.AUTH_ERROR

    def test_is_retryable_for_transient(self) -> None:
        err = LocalDataError(
            message="timed out",
            category=ErrorCategory.TIMEOUT,
        )
        resp = err.to_structured_response()
        assert resp.is_retryable is True

    def test_suggestion_from_recovery_suggestions(self) -> None:
        err = LocalDataError(
            message="conn lost",
            category=ErrorCategory.CONNECTION,
            recovery_suggestions=["Try reconnecting", "Check host"],
        )
        resp = err.to_structured_response()
        assert resp.suggestion == "Try reconnecting"
