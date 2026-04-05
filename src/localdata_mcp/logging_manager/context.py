"""Log context for structured logging."""

import uuid
from typing import Any, Dict


class LogContext:
    """Context information for structured logging.

    Accepts arbitrary keyword arguments so callers can attach any
    context fields without needing to modify this class.
    """

    def __init__(self, **kwargs):
        self.request_id = kwargs.pop("request_id", str(uuid.uuid4()))
        self.session_id = kwargs.pop("session_id", None)
        self.user_id = kwargs.pop("user_id", None)
        self.operation = kwargs.pop("operation", None)
        self.component = kwargs.pop("component", None)
        self.database_name = kwargs.pop("database_name", None)
        self.query_hash = kwargs.pop("query_hash", None)
        self.start_time = kwargs.pop("start_time", None)
        # Store any extra context fields
        for key, value in kwargs.items():
            setattr(self, key, value)

    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary for logging."""
        return {k: v for k, v in self.__dict__.items() if v is not None}
