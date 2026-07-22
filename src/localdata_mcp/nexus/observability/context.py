"""localdata_mcp/nexus/observability/context.py — structured log context.

The kept `logging_manager/context.py` shape (ARCHITECTURE.md section 8,
NX-4): a LogContext value object callers fill with whatever request or
operation fields matter, serialized into every event emitted while it
is bound. v3 binds it through structlog's contextvars support (async-
and thread-safe) instead of the legacy manager's thread-local.
Neighbors: manager.py exposes the bind/unbind entry point; config.py
puts `merge_contextvars` in the processor chain.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict


class LogContext:
    """Context information for structured logging.

    Accepts arbitrary keyword arguments so callers can attach any
    context fields without needing to modify this class.
    """

    def __init__(self, **kwargs: Any) -> None:
        self.request_id = kwargs.pop("request_id", str(uuid.uuid4()))
        self.session_id = kwargs.pop("session_id", None)
        self.operation = kwargs.pop("operation", None)
        self.component = kwargs.pop("component", None)
        self.endpoint_name = kwargs.pop("endpoint_name", None)
        self.query_hash = kwargs.pop("query_hash", None)
        # Store any extra context fields verbatim.
        for key, value in kwargs.items():
            setattr(self, key, value)

    def to_dict(self) -> Dict[str, Any]:
        """The non-empty context fields, ready to merge into an event."""
        return {k: v for k, v in self.__dict__.items() if v is not None}
