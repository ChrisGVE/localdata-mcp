"""localdata_mcp/nexus/error/wire.py — NX3.wrap, the error-path driver.

The §4b ordered sequence in one place: translate the exception through
the kept feeder, signal a connection-class fault into NX-5's lifecycle
SYNCHRONOUSLY through the E4.0 interface, then redact and hand the one
wire shape onward — never a bare traceback (§6.2's `NX3.wrap(exc)`
seam, the only error API tool modules may import). Neighbors:
translate.py, redact.py, fault_signal.py compose here; NX-7 embeds
the returned form.
"""

from __future__ import annotations

from localdata_mcp.nexus.error.fault_signal import (
    DisposeAndReissueEntry,
    FaultSignal,
)
from localdata_mcp.nexus.error.model import StructuredError
from localdata_mcp.nexus.error.redact import redact_structured
from localdata_mcp.nexus.error.translate import translate


def wrap(
    exception: Exception,
    backend_kind: str,
    *,
    fault_sink: FaultSignal | None = None,
    record_id: str | None = None,
) -> StructuredError:
    """`exception` as the one redacted wire shape (§4b, in order).

    When the classification is a connection-fault class AND the caller
    names the connection record it was using, the E4.0 signal fires
    synchronously before the shape is returned — NX-5 disposes and
    reissues per §5; the returned entry is lifecycle bookkeeping and
    never alters the caller-facing shape.
    """
    structured = translate(exception, backend_kind)
    if (
        structured.error_type.signals_connection_fault
        and fault_sink is not None
        and record_id is not None
    ):
        _signal_fault(fault_sink, record_id)
    return redact_structured(structured)


def _signal_fault(fault_sink: FaultSignal, record_id: str) -> DisposeAndReissueEntry:
    """Fire the §4b→§5 transition; the entry is observable via NX-5."""
    return fault_sink.mark_faulted(record_id)
