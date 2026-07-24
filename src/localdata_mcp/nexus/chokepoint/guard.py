"""localdata_mcp/nexus/chokepoint/guard.py — NX-6's public seam (E6.1).

THE chokepoint (§4a, §6.2, §8 NX-6): `Chokepoint` is the one guarded
data-access surface, and this module is the ONE name a tool module may
import to reach it (FR-802 / §6.2's import rule, enforced by the import
gate). The class is composed from four cohesive surface mixins so no
single class exceeds NFR-404's per-class bound while the public import
name and call surface stay identical:

- `_QueryMutationSurface` (surfaces_query.py) — `guarded_query` /
  `guarded_mutation`, the two entrypoints, plus the ephemeral
  local-file seam (`guarded_file_query`).
- `_StreamingSurface` (surfaces_stream.py) — the I-4 inline/stream
  cutover and the E6.6 ChunkRegistry cursor.
- `_SchemaSurface` (surfaces_schema.py) — the I-1 endpoint enumeration
  and the X-1/E9.1 schema-discovery reads.
- `_ConfigSeams` (surfaces_config.py) — the §6.2 config-value seams,
  the §6.3 composition ledger, and FR-305's expression evaluation.

All four inherit `_GuardCore` (core.py), which owns the shared state
built at boot — NX-5 persistence, the E6.3 validation cache, the
NFR-105 resource ledger, the E6.6 registry — and the cross-cutting
seams: the E6.2/E6.2b statement screen, NX-6's standalone NFR-108
path-containment service (`contain_path`, GP3), and the §4b NX-3 wire
that turns a backend fault into the one structured shape (the E4.0
fault signal fires synchronously against the record before the
structured shape is raised). Every value shape and both structured
refusals live in types.py. Ephemeral local-file opens cross
`contain_path` before NX-5; the streaming path pins the NX-5 connection
the registry's `on_close` releases (§5). Neighbors: every sibling
chokepoint module composes here; nexus/persistence resolves;
nexus/error shapes.
"""

from __future__ import annotations

from typing import Mapping

from localdata_mcp.nexus.config.models import ConfigModel
from localdata_mcp.nexus.persistence.ephemeral import EphemeralEngineKind
from localdata_mcp.nexus.persistence.manager import (
    PersistenceNexus,
    UnknownEndpointError,
)

from .chunk_registry import (
    ChunkAlreadyServedError,
    ChunkNotServableError,
    StreamAdmissionRefusedError,
    StreamExpiredError,
)
from .expr_eval import ExpressionRefusedError
from .resource_bounds import ResourceRefusedError
from .surfaces_config import _ConfigSeams
from .surfaces_query import _QueryMutationSurface
from .surfaces_schema import _SchemaSurface
from .surfaces_stream import _StreamingSurface
from .types import (
    CompositionLimits,
    EndpointSummary,
    GuardRefusedError,
    GuardedExecutionError,
    ProcessDefaults,
    QueryRequest,
    Result,
    ServedChunk,
    StreamOpened,
    VisualizeDefaults,
)

# Re-exported at the seam: tool modules catch the NFR-114 name miss,
# name the ephemeral engine kind, and catch the resource-admission
# refusal through the guard — never by importing NX-5 or a chokepoint
# internal (FR-802 / §6.2's import rule, enforced by the import gate).
__all__ = [
    "Chokepoint",
    "QueryRequest",
    "Result",
    "EndpointSummary",
    "GuardRefusedError",
    "GuardedExecutionError",
    "UnknownEndpointError",
    "EphemeralEngineKind",
    "ResourceRefusedError",
    "StreamAdmissionRefusedError",
    "StreamExpiredError",
    "ChunkAlreadyServedError",
    "ChunkNotServableError",
    "StreamOpened",
    "ServedChunk",
    "ProcessDefaults",
    "VisualizeDefaults",
    "CompositionLimits",
    "ExpressionRefusedError",
]


class Chokepoint(
    _QueryMutationSurface,
    _StreamingSurface,
    _SchemaSurface,
    _ConfigSeams,
):
    """The one guarded data-access surface (§6.2's `NX6`) — composed
    from the four surface mixins over the shared `_GuardCore`."""

    @classmethod
    def boot(cls, config: ConfigModel, environ: Mapping[str, str]) -> "Chokepoint":
        """§4e: build NX-5 from the loaded model, warm it up, and wrap
        it — the entrypoint constructs persistence THROUGH the guard,
        so NX-5 stays reachable by NX-6 exclusively even at boot
        (FR-802 covers construction, not only queries)."""
        persistence = PersistenceNexus(config, environ)
        persistence.warm_up()
        return cls(config, persistence)

    def shutdown(self) -> None:
        """§4e teardown: every live stream released (returning its
        pinned connection), then every record closed and pool disposed
        — order matters, a disposed pool cannot take a connection
        back."""
        self._registry.close_all()
        self._persistence.close_all()
