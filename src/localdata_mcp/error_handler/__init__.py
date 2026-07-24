"""NX-3 error-taxonomy feeder remnant (post-E15).

The only surviving members of the former v2 error-handling package are the
declared *mapping knowledge* the v3 error nexus keeps unchanged (PRD S4.3,
NFR-302): ``ErrorCategory`` and its sibling enums/exception classes in
``exceptions``, consumed by ``error_mappers.py`` / ``error_classification.py``
and translated onto the closed wire taxonomy in ``nexus/error/translate.py``
(the tree's ONE importer of this feeder). E15 deleted the legacy machinery
(circuit breaker, retry, recovery, handler, error logging); ``exceptions`` is
pure stdlib and imports unconditionally.
"""

from .exceptions import (
    CircuitState,
    ConfigurationError,
    DatabaseConnectionError,
    ErrorCategory,
    ErrorSeverity,
    LocalDataError,
    QueryExecutionError,
    QueryTimeoutError,
    ResourceExhaustionError,
    RetryStrategy,
    SecurityViolationError,
)

__all__ = [
    "CircuitState",
    "ConfigurationError",
    "DatabaseConnectionError",
    "ErrorCategory",
    "ErrorSeverity",
    "LocalDataError",
    "QueryExecutionError",
    "QueryTimeoutError",
    "ResourceExhaustionError",
    "RetryStrategy",
    "SecurityViolationError",
]
