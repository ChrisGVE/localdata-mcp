"""localdata_mcp/nexus/persistence — NX-5, the connection nexus.

The revived `connection_manager/` design (E5.1): the sole owner of
every live connection object in v3 (FR-801/802) — `ConnectionRecord`
per declared endpoint with the §5 lifecycle and dispose-and-reissue
(NFR-112), `EphemeralFileConnection` for ad-hoc local-file opens,
engine creation with posture applied at creation (§8.1), and FR-803
pooling + health checks. Reachable by NX-6 exclusively for backend I/O
(§6.2); the import-graph test enforces that reachability.
"""

from .engines import EngineHandle, UnsupportedBackendError, backend_kind_of
from .ephemeral import EphemeralFileConnection, ephemeral_for
from .health import HealthCheckResult, probe
from .lifecycle import LifecycleError, LifecycleState
from .limits import ResourceLimits, limits_from_config
from .manager import (
    EndpointUnavailableError,
    PersistenceNexus,
    UnknownEndpointError,
)
from .record import ConnectionRecord

__all__ = [
    "ConnectionRecord",
    "EndpointUnavailableError",
    "EngineHandle",
    "EphemeralFileConnection",
    "HealthCheckResult",
    "LifecycleError",
    "LifecycleState",
    "PersistenceNexus",
    "ResourceLimits",
    "UnknownEndpointError",
    "UnsupportedBackendError",
    "backend_kind_of",
    "ephemeral_for",
    "limits_from_config",
    "probe",
]
