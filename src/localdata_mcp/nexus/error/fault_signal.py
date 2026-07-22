"""localdata_mcp/nexus/error/fault_signal.py — the NX-3↔NX-5 fault seam.

E4.0's locked calling convention, authored ONCE before NX-3 and NX-5
build their halves (PRD S6 E4.0; ARCHITECTURE §4b→§5 transition set —
one state machine spanning two nexuses):

- NX-3 owns WHEN the signal fires: classifying a connection-class
  error calls `mark_faulted(record)` SYNCHRONOUSLY on the error path,
  before the wire shape leaves the nexus (§4b's ordered sequence).
- NX-5 owns WHAT the signal does: transition the record
  `active → faulted → resetting`, dispose the underlying connection,
  and reissue a replacement — reported back as the returned entry.

Neighbors: wire.py (NX-3's wrap) calls through this protocol; E5's
`ConnectionRecord` lifecycle implements it; the conformant test double
below is what BOTH epics' tests run against until E6's gate checks the
real implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class DisposeAndReissueEntry:
    """NX-5's synchronous answer to a fault signal (§4b/§5).

    `record_id` names the faulted record; `disposed` confirms the dead
    connection left the pool; `reissued` reports whether a replacement
    was created (False when the endpoint is beyond its limits or the
    reissue itself failed — the caller's wire shape is unaffected
    either way, the flag is lifecycle bookkeeping).
    """

    record_id: str
    disposed: bool
    reissued: bool


@runtime_checkable
class FaultSignal(Protocol):
    """What NX-3 may assume about NX-5's fault handling — nothing more."""

    def mark_faulted(self, record_id: str) -> DisposeAndReissueEntry:
        """Transition `record_id` per §5 and dispose-and-reissue.

        Synchronous: returns only after the record has left `active`
        (the §4b ordered sequence forbids a fire-and-forget signal).
        """
        ...


@dataclass
class RecordingFaultSink:
    """E4.0's CONFORMANT test double — a real, observable implementation.

    Records every transition (`active → faulted → resetting → active`)
    per §5's lifecycle so tests assert the state walk, not a mock call
    count. E5's tests build against this same double; E6's gate swaps
    in the real `ConnectionRecord` lifecycle.
    """

    transitions: list[tuple[str, str]] = field(default_factory=list)
    reissue: bool = True

    def mark_faulted(self, record_id: str) -> DisposeAndReissueEntry:
        self.transitions.append((record_id, "faulted"))
        self.transitions.append((record_id, "resetting"))
        if self.reissue:
            self.transitions.append((record_id, "active"))
        return DisposeAndReissueEntry(
            record_id=record_id, disposed=True, reissued=self.reissue
        )
