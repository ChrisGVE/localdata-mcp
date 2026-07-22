"""localdata_mcp/nexus/error/translate.py — exception → StructuredError.

The ONE place an exception becomes NX-3's model (§4b): wraps the KEPT
backend-mapper feeder (error_mappers.py via error_classification's
registry — per §8 NX-3 the feeder stays, unchanged, as declared
mapping knowledge) and converts its legacy category onto the closed
wire taxonomy through one declared table. NFR-302's always-on check
(tests/v3/test_error_translate.py) asserts this module is the v3
tree's ONLY importer of the legacy feeder — a second translation path
is the re-implementation NX-3 exists to forbid. Neighbors: model.py
defines the target shape; wire.py drives translation on the error
path.
"""

from __future__ import annotations

from localdata_mcp.error_classification import classify_error
from localdata_mcp.nexus.error.model import ErrorType, StructuredError

# Legacy ErrorCategory value → wire taxonomy. Declared data, exhaustive
# over every category the feeder can emit; unknown values fall back to
# QUERY_EXECUTION (the feeder's own default class).
_LEGACY_TO_WIRE: dict[str, ErrorType] = {
    "connection": ErrorType.CONNECTION_ERROR,  # deprecated legacy spelling
    "connection_error": ErrorType.CONNECTION_ERROR,
    "authentication": ErrorType.AUTH_ERROR,
    "auth_error": ErrorType.AUTH_ERROR,
    "permission": ErrorType.PERMISSION,
    "schema_error": ErrorType.SCHEMA_ERROR,
    "syntax_error": ErrorType.SYNTAX_ERROR,
    "constraint_error": ErrorType.CONSTRAINT_ERROR,
    "resource_exhaustion": ErrorType.RESOURCE_ERROR,
    "resource_error": ErrorType.RESOURCE_ERROR,
    "transient_error": ErrorType.TRANSIENT_ERROR,
    "timeout": ErrorType.TIMEOUT,
    "security_violation": ErrorType.SECURITY_VIOLATION,
    "data_validation": ErrorType.DATA_VALIDATION,
    "configuration": ErrorType.CONFIGURATION,
    "query_execution": ErrorType.QUERY_EXECUTION,
    "system": ErrorType.QUERY_EXECUTION,
}


def translate(exception: Exception, backend_kind: str) -> StructuredError:
    """Classify `exception` through the kept feeder into the one shape.

    `backend_kind` selects the backend-specific mapper (sqlite,
    postgresql, mysql, duckdb, oracle, mssql — anything else falls to
    the generic keyword mapper, the feeder's own routing).
    """
    legacy = classify_error(exception, backend_kind)
    error_type = _LEGACY_TO_WIRE.get(legacy.error_type.value, ErrorType.QUERY_EXECUTION)
    return StructuredError(
        error_type=error_type,
        message=legacy.message or str(exception),
        suggestion=legacy.suggestion,
        retryable=bool(legacy.is_retryable),
    )
