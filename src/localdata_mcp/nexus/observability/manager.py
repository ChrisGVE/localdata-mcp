"""localdata_mcp/nexus/observability/manager.py — two-phase bring-up.

The kept `logging_manager/manager.py` shape reduced to what NX-4 owns
in v3 (metrics dropped, ARCHITECTURE.md section 8): module-level phase
state driving the section-4e bring-up — `bootstrap()` puts a minimal
stderr-only setup in place before any config is read (enough to report
a ConfigurationError), `reconfigure()` applies operator settings once
NX-2's merged model exists. Neighbors: config.py performs the actual
(re)configuration; context.py supplies the LogContext bound through
`log_context()`.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator, Literal

import structlog

from .config import configure
from .context import LogContext

Phase = Literal["unconfigured", "bootstrap", "configured"]

# Bootstrap verbosity: fixed at INFO — informative enough to report a
# startup ConfigurationError, quiet enough before the operator has had
# any say. The operator's level arrives via reconfigure().
_BOOTSTRAP_LEVEL = "INFO"

_phase: Phase = "unconfigured"


def bootstrap(*, force: bool = False) -> None:
    """Enter the stderr-only bootstrap phase (section 4e, step one).

    Idempotent: once any phase is active this is a no-op, so the
    import-time activation cannot clobber an operator reconfigure.
    `force` re-enters bootstrap regardless — for tests that must undo a
    reconfigure.
    """
    global _phase
    if _phase != "unconfigured" and not force:
        return
    configure(_BOOTSTRAP_LEVEL)
    _phase = "bootstrap"


def reconfigure(*, level: str = _BOOTSTRAP_LEVEL) -> None:
    """Apply operator logging settings (section 4e, step two).

    Parameter-driven by design: PRD S8 declares no logging rows, so
    NX-2's ConfigModel carries no logging section today — the caller
    (server/mcp_app.py) passes whatever the config mechanism provides
    once such rows exist. stderr-only is the invariant floor: level is
    refinable, the destination is not a parameter at all.
    """
    global _phase
    configure(level)
    _phase = "configured"


def logging_phase() -> Phase:
    """The current bring-up phase, for startup reporting and tests."""
    return _phase


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """NX-4's one logger accessor — every component logs through this
    (nothing calls print() or opens its own handler, section 8)."""
    bootstrap()
    return structlog.get_logger(name)


@contextmanager
def log_context(**fields: Any) -> Iterator[LogContext]:
    """Bind a LogContext's fields to every event emitted inside the
    `with` block (contextvars-backed: async- and thread-safe)."""
    context = LogContext(**fields)
    tokens = structlog.contextvars.bind_contextvars(**context.to_dict())
    try:
        yield context
    finally:
        structlog.contextvars.reset_contextvars(**tokens)
