"""localdata_mcp/nexus/error/model.py — NX-3's one model, taxonomy, wire shape.

The single structured form every exception becomes (ARCHITECTURE §4b):
`{error_type, message, suggestion, retryable}` — nothing else crosses
to NX-7, and no tool returns an ad-hoc error dict (the rejected
re-implementation, §8 NX-3). `retryable` is CALLER-ADVISORY only: v3
implements no retry machinery; the flag carries "transient, safe to
re-ask" for the LLM caller. Neighbors: translate.py builds instances
from the kept mapper feeder; redact.py scrubs the two text fields;
wire.py hands the serialized form to NX-7.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass


class ErrorType(enum.Enum):
    """Closed wire taxonomy — the caller-facing classification.

    Values mirror the kept feeder's structured categories
    (error_handler/exceptions.py ErrorCategory) for the subset that can
    reach the wire, so translation is value-preserving; legacy-internal
    categories (circuit state, severity) do not exist here.
    """

    CONNECTION_ERROR = "connection_error"
    AUTH_ERROR = "auth_error"
    PERMISSION = "permission"
    SCHEMA_ERROR = "schema_error"
    SYNTAX_ERROR = "syntax_error"
    CONSTRAINT_ERROR = "constraint_error"
    RESOURCE_ERROR = "resource_error"
    TRANSIENT_ERROR = "transient_error"
    TIMEOUT = "timeout"
    SECURITY_VIOLATION = "security_violation"
    DATA_VALIDATION = "data_validation"
    CONFIGURATION = "configuration"
    QUERY_EXECUTION = "query_execution"

    @property
    def signals_connection_fault(self) -> bool:
        """Whether classification triggers the §4b→§5 fault signal."""
        return self in (ErrorType.CONNECTION_ERROR, ErrorType.TIMEOUT)


@dataclass(frozen=True)
class StructuredError:
    """The one error shape (§4b): four fields, no traceback, no extras."""

    error_type: ErrorType
    message: str
    suggestion: str
    retryable: bool

    def to_wire(self) -> dict[str, object]:
        """The JSON-serializable wire form NX-7 embeds."""
        return {
            "error_type": self.error_type.value,
            "message": self.message,
            "suggestion": self.suggestion,
            "retryable": self.retryable,
        }
