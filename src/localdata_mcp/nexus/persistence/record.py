"""localdata_mcp/nexus/persistence/record.py — the §5 connection record.

One `ConnectionRecord` per operator-declared endpoint (NFR-114 — the
name is declared, never caller-supplied): backend kind, posture,
credential indirection (NFR-110), the engine handle the pool lives
behind, the latest health result, the NX-2-sourced resource limits, and
the explicit lifecycle `state` the §5 machine walks. Granularity is
record-level by declaration: `state` describes the endpoint's pool as a
whole, and the record id NX-3's fault signal names (E4.0) IS the
endpoint name — one record per endpoint, one id vocabulary. Neighbors:
lifecycle.py refuses illegal state walks; engines.py builds the handle;
manager.py owns the record collection and drives dispose-and-reissue.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from localdata_mcp.nexus.config.endpoints import Posture
from localdata_mcp.nexus.persistence.engines import EngineHandle
from localdata_mcp.nexus.persistence.health import HealthCheckResult
from localdata_mcp.nexus.persistence.lifecycle import (
    LifecycleState,
    checked_transition,
)
from localdata_mcp.nexus.persistence.limits import ResourceLimits


@dataclass
class ConnectionRecord:
    """One declared endpoint's pool state (§5)."""

    name: str
    backend_kind: str
    posture: Posture
    credentials_ref: str | None
    pool: EngineHandle
    limits: ResourceLimits
    health: HealthCheckResult | None = None
    state: LifecycleState = field(default=LifecycleState.HEALTHY)

    def transition(self, target: LifecycleState) -> None:
        """Walk one legal §5 edge; anything else raises LifecycleError."""
        self.state = checked_transition(self.state, target)

    @property
    def issuable(self) -> bool:
        """Only a healthy record may issue connections — faulted and
        resetting records are mid-walk, closed ones are terminal."""
        return self.state is LifecycleState.HEALTHY
