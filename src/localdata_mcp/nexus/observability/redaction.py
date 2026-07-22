"""localdata_mcp/nexus/observability/redaction.py — the outbound edge.

NX-4's structural redaction (NFR-110, ARCHITECTURE.md section 8): a
structlog processor sitting immediately before the renderer, so every
string the chain emits — event text, keyword values, nested
containers, rendered exception text — passes through the one
credential rewrite before it can leave the nexus. The shapes
themselves live in nexus/config/dsn_patterns.py (one home, shared with
the endpoint refusal path). Neighbors: config.py wires this processor
into the chain; nothing logs around it.
"""

from __future__ import annotations

from typing import Any, Mapping

import structlog

from ..config.dsn_patterns import redact_credentials_text


def redact_event(
    logger: object,
    method_name: str,
    event_dict: structlog.typing.EventDict,
) -> structlog.typing.EventDict:
    """Redact every credential-shaped span in the event, recursively."""
    return {key: _redact_value(value) for key, value in event_dict.items()}


def _redact_value(value: Any) -> Any:
    """One value, type-preserving: strings rewritten, containers
    walked, everything else passed through untouched."""
    if isinstance(value, str):
        return redact_credentials_text(value)
    if isinstance(value, Mapping):
        return {key: _redact_value(inner) for key, inner in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_redact_value(inner) for inner in value)
    return value
