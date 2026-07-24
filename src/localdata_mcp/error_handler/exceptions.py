"""NX-3 feeder taxonomy: the one enum the kept mapper path consumes.

Post-E15 this module holds ONLY ``ErrorCategory`` — the legacy category
vocabulary ``error_classification.py`` / ``error_mappers.py`` classify
into and ``nexus/error/translate.py`` translates onto the closed wire
taxonomy (PRD S4.3, NFR-302). The former v2 exception hierarchy
(``LocalDataError`` + subclasses) and the retry/circuit machinery
(``RetryStrategy``, ``CircuitState``, ``ErrorSeverity``) were deleted:
they had zero live consumers and ``RetryStrategy``/``CircuitState``
contradicted NX-3's explicit no-retry stance (GP5/GP6 dead surface).
"""

from enum import Enum


class ErrorCategory(Enum):
    """Categories of errors for classification and handling."""

    CONNECTION = "connection"  # deprecated — use CONNECTION_ERROR
    QUERY_EXECUTION = "query_execution"
    SECURITY_VIOLATION = "security_violation"
    TIMEOUT = "timeout"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    CONFIGURATION = "configuration"
    AUTHENTICATION = "authentication"
    PERMISSION = "permission"
    DATA_VALIDATION = "data_validation"
    SYSTEM = "system"
    # Structured error classification categories
    AUTH_ERROR = "auth_error"
    SCHEMA_ERROR = "schema_error"
    SYNTAX_ERROR = "syntax_error"
    RESOURCE_ERROR = "resource_error"
    TRANSIENT_ERROR = "transient_error"
    CONSTRAINT_ERROR = "constraint_error"
    CONNECTION_ERROR = "connection_error"
