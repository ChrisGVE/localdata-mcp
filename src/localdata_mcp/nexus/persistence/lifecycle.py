"""localdata_mcp/nexus/persistence/lifecycle.py — the §5 state machine.

The complete, closed transition set for a `ConnectionRecord`
(ARCHITECTURE §5): `healthy → faulted` when NX-3 marks the record on an
operation exception (§4b); `faulted → resetting` when NX-5 begins the
dispose-and-reissue; `resetting → healthy` when the fresh connection is
issued; `resetting → closed` when the reissue itself fails; and any
state `→ closed` on shutdown or endpoint removal (§4e). Anything else
is a programming error, refused loudly — the machine has no silent
edges. Neighbors: record.py holds the state and calls `transition`;
manager.py drives the fault walk through it.
"""

from __future__ import annotations

from enum import Enum


class LifecycleState(Enum):
    """A record's pool-level state (§5 — record granularity is by
    declaration: one state for the endpoint's pool as a whole)."""

    HEALTHY = "healthy"
    FAULTED = "faulted"
    RESETTING = "resetting"
    CLOSED = "closed"


# The §5 transition set, complete and closed. `→ CLOSED` is legal from
# every state (shutdown/removal), so it is listed from each source.
LEGAL_TRANSITIONS: frozenset[tuple[LifecycleState, LifecycleState]] = frozenset(
    {
        (LifecycleState.HEALTHY, LifecycleState.FAULTED),
        (LifecycleState.FAULTED, LifecycleState.RESETTING),
        (LifecycleState.RESETTING, LifecycleState.HEALTHY),
        (LifecycleState.RESETTING, LifecycleState.CLOSED),
        (LifecycleState.HEALTHY, LifecycleState.CLOSED),
        (LifecycleState.FAULTED, LifecycleState.CLOSED),
    }
)


class LifecycleError(RuntimeError):
    """An illegal transition was requested — a caller bug, never a
    runtime condition to swallow (§5's set is complete by design)."""


def checked_transition(
    current: LifecycleState, target: LifecycleState
) -> LifecycleState:
    """`target` if `current → target` is in the §5 set, else refuse.

    `closed` is terminal: even `closed → closed` is refused so a
    double-close surfaces as the bug it is.
    """
    if (current, target) not in LEGAL_TRANSITIONS:
        raise LifecycleError(
            f"illegal lifecycle transition {current.value} → {target.value}"
        )
    return target
