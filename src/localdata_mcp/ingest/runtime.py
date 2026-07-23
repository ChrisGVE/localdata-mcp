"""localdata_mcp/ingest/runtime.py — the tool layer's one NX-6 handle (E8.1).

Tool functions are flat callables (NX-1's generated wrapper calls
them with declared parameters only), so the process Chokepoint they
must cross reaches them through this one seam: mcp_app builds NX-5,
warms it up (§4e), wraps it in the Chokepoint, and installs it here at
boot. Before that installation every data-touching tool answers with a
structured configuration refusal — fail-closed, never a bare crash.
Only the guard is held: tool modules never see NX-5 or a connection
(FR-802; the import-graph gate enforces the module side, this seam the
object side). Neighbors: endpoints.py and every connector family call
`chokepoint()`; mcp_app calls `configure_ingest`.
"""

from __future__ import annotations

import threading

from localdata_mcp.nexus.chokepoint.guard import Chokepoint, GuardedExecutionError
from localdata_mcp.nexus.error.model import ErrorType, StructuredError

_LOCK = threading.Lock()
_CHOKEPOINT: Chokepoint | None = None


def configure_ingest(chokepoint: Chokepoint) -> None:
    """§4e: install the process chokepoint after warm-up."""
    global _CHOKEPOINT
    with _LOCK:
        _CHOKEPOINT = chokepoint


def chokepoint() -> Chokepoint:
    """The installed guard, or a structured not-booted refusal."""
    with _LOCK:
        installed = _CHOKEPOINT
    if installed is None:
        raise GuardedExecutionError(
            StructuredError(
                error_type=ErrorType.CONFIGURATION,
                message=(
                    "the server has not completed startup — no data access "
                    "is available yet"
                ),
                suggestion=(
                    "Retry once the server reports 'starting stdio "
                    "transport'; if this persists, the operator "
                    "configuration failed validation at boot."
                ),
                retryable=True,
            )
        )
    return installed
