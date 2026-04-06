"""Structured error classification for database operations.

Provides a registry-based error mapping system that classifies database
exceptions into structured, actionable error responses. Each database
backend can register its own mapper; a generic keyword-based mapper
serves as the fallback.
"""

import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .error_handler import ErrorCategory

# ---------------------------------------------------------------------------
# Structured error response
# ---------------------------------------------------------------------------


@dataclass
class StructuredErrorResponse:
    """Uniform error response returned by all error mappers."""

    error: bool = True
    error_type: ErrorCategory = ErrorCategory.QUERY_EXECUTION
    is_retryable: bool = False
    message: str = ""
    suggestion: str = ""
    database_error_code: Optional[str] = None
    database: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error": self.error,
            "error_type": self.error_type.value,
            "is_retryable": self.is_retryable,
            "message": self.message,
            "suggestion": self.suggestion,
            "database_error_code": self.database_error_code,
            "database": self.database,
        }


# ---------------------------------------------------------------------------
# Abstract mapper interface
# ---------------------------------------------------------------------------


class DatabaseErrorMapper(ABC):
    """Base class for database-specific error mappers."""

    @abstractmethod
    def map_error(self, exception: Exception) -> StructuredErrorResponse:
        """Translate a raw database exception into a StructuredErrorResponse."""


# ---------------------------------------------------------------------------
# Mapper registry
# ---------------------------------------------------------------------------


class ErrorMapperRegistry:
    """Thread-safe registry that maps database types to their error mappers."""

    _mappers: Dict[str, DatabaseErrorMapper] = {}
    _lock = threading.RLock()

    @classmethod
    def register(cls, db_type: str, mapper: DatabaseErrorMapper) -> None:
        with cls._lock:
            cls._mappers[db_type] = mapper

    @classmethod
    def get_mapper(cls, db_type: str) -> Optional[DatabaseErrorMapper]:
        return cls._mappers.get(db_type)

    @classmethod
    def get_or_default(cls, db_type: str) -> DatabaseErrorMapper:
        return cls._mappers.get(db_type) or cls._mappers.get("generic")

    @classmethod
    def clear(cls) -> None:
        """Remove all registered mappers (useful in tests)."""
        with cls._lock:
            cls._mappers.clear()


# ---------------------------------------------------------------------------
# Generic (keyword-based) mapper
# ---------------------------------------------------------------------------

# Each entry: (keywords, category, is_retryable, suggestion)
_KEYWORD_RULES: List[Tuple[List[str], ErrorCategory, bool, str]] = [
    (
        ["access denied", "permission", "login"],
        ErrorCategory.AUTH_ERROR,
        False,
        "Check database credentials and user permissions.",
    ),
    (
        ["not found", "does not exist", "unknown column"],
        ErrorCategory.SCHEMA_ERROR,
        False,
        "Verify the table/column names exist in the database schema.",
    ),
    (
        ["syntax", "parse error"],
        ErrorCategory.SYNTAX_ERROR,
        False,
        "Review the SQL query for syntax errors.",
    ),
    (
        ["timeout", "deadlock", "lock", "busy"],
        ErrorCategory.TRANSIENT_ERROR,
        True,
        "The operation may succeed if retried after a brief wait.",
    ),
    (
        ["out of memory", "disk full", "no space"],
        ErrorCategory.RESOURCE_ERROR,
        False,
        "Free system resources or reduce query scope.",
    ),
    (
        ["constraint", "unique", "foreign key"],
        ErrorCategory.CONSTRAINT_ERROR,
        False,
        "The operation violates a database constraint.",
    ),
    (
        ["connection", "network", "host"],
        ErrorCategory.CONNECTION_ERROR,
        True,
        "Check network connectivity and database host availability.",
    ),
]


class GenericDatabaseErrorMapper(DatabaseErrorMapper):
    """Fallback mapper that classifies errors by keyword matching."""

    def map_error(self, exception: Exception) -> StructuredErrorResponse:
        msg = str(exception).lower()

        for keywords, category, retryable, suggestion in _KEYWORD_RULES:
            if any(kw in msg for kw in keywords):
                return StructuredErrorResponse(
                    error_type=category,
                    is_retryable=retryable,
                    message=str(exception),
                    suggestion=suggestion,
                )

        return StructuredErrorResponse(
            error_type=ErrorCategory.QUERY_EXECUTION,
            is_retryable=False,
            message=str(exception),
            suggestion="Review the query and database state.",
        )


# Auto-register the generic mapper so get_or_default always has a fallback.
ErrorMapperRegistry.register("generic", GenericDatabaseErrorMapper())


# ---------------------------------------------------------------------------
# Standalone helper functions
# ---------------------------------------------------------------------------


def classify_error(
    exception: Exception, db_type: str = "generic"
) -> StructuredErrorResponse:
    """Classify a database exception (standalone helper)."""
    mapper = ErrorMapperRegistry.get_or_default(db_type)
    return mapper.map_error(exception)


def is_error_retryable(exception: Exception, db_type: str = "generic") -> bool:
    """Check if an error is retryable."""
    return classify_error(exception, db_type).is_retryable


def get_error_suggestion(exception: Exception, db_type: str = "generic") -> str:
    """Get an actionable suggestion for handling an error."""
    return classify_error(exception, db_type).suggestion


# Import database-specific mappers to trigger their registration.
from . import error_mappers as _error_mappers  # noqa: E402, F401
