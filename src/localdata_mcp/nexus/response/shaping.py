"""localdata_mcp/nexus/response/shaping.py — wrapper-applied shaping (E7.2).

The one runtime seam NX-1's GENERATED wrapper calls around every tool
implementation — envelope-shaping is applied here for every registered
tool, NEVER opt-in (O-1; `main`'s live wrappers emitting bare dicts is
the exact rejected defect, T12). One call does, in order: run the
implementation; run the shared degenerate-output sentinel (S3.3 — a
tripped signal becomes a structured NX-3 error envelope, never a
silent success); shape the result through NX-7's envelope. An
exception becomes the error envelope — a guard failure already carries
its NX-3 shape (E6.1), anything else crosses `NX3.wrap` here — so no
tool call can leak a bare traceback or an unshaped payload.
`configure_shaping` is the §4e boot hook: mcp_app installs the loaded
NX-2 model after config validation; before that (and under tests) the
shaper runs on the declared S8 defaults, which is exactly what an
unconfigured process should enforce. Neighbors: envelope.py and
sentinel.py do the work; generators/wrapper.py emits the calls.
"""

from __future__ import annotations

import threading
from typing import Any, Callable, Mapping

from localdata_mcp.nexus.chokepoint.guard import GuardedExecutionError
from localdata_mcp.nexus.config.models import ConfigModel
from localdata_mcp.nexus.contract.registry import ToolRegistry, default_registry
from localdata_mcp.nexus.error.wire import wrap

from .envelope import ResponseShaper, error_envelope
from .sentinel import inspect


class _ShapingState:
    """The process-wide shaper pair, swapped atomically at boot."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._config = ConfigModel()
        self._registry = default_registry()
        self._shaper = ResponseShaper(self._config, self._registry)

    def configure(self, config: ConfigModel, registry: ToolRegistry) -> None:
        with self._lock:
            self._config = config
            self._registry = registry
            self._shaper = ResponseShaper(config, registry)

    def current(self) -> "tuple[ResponseShaper, ConfigModel, ToolRegistry]":
        with self._lock:
            return self._shaper, self._config, self._registry


_STATE = _ShapingState()


def configure_shaping(config: ConfigModel, registry: ToolRegistry) -> None:
    """§4e: install the loaded NX-2 model (and the live registry) as
    the process shaper — called by mcp_app after config validation."""
    _STATE.configure(config, registry)


def shaped_call(
    tool_name: str,
    implementation: Callable[..., Any],
    arguments: Mapping[str, Any],
) -> dict[str, Any]:
    """One guarded, sentinel-checked, envelope-shaped tool call — the
    only body a generated wrapper function has."""
    shaper, config, registry = _STATE.current()
    spec = registry.lookup(tool_name)
    try:
        raw = implementation(**arguments)
    except GuardedExecutionError as failure:
        # Already crossed NX-3's wire inside the guard (E6.1).
        return error_envelope(failure.structured).to_wire()
    except Exception as failure:
        return error_envelope(wrap(failure, "generic")).to_wire()
    tripped = inspect(raw, config.process.sentinel_max_condition_number)
    if tripped is not None:
        return error_envelope(tripped).to_wire()
    return shaper.shape_envelope(raw, spec).to_wire()
