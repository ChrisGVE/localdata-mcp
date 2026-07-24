"""localdata_mcp/nexus/chokepoint/core.py — NX-6's shared guard core.

The state and the cross-cutting seams every guard surface leans on,
factored out so each surface mixin (surfaces_*.py) and THE composed
`Chokepoint` (guard.py) inherit ONE base instead of one 492-LOC class
(NFR-404's per-class bound). `_GuardCore` holds the four collaborators
built at boot — NX-5 persistence, the E6.3 validation cache, the
NFR-105 resource ledger, the E6.6 chunk registry — and owns the three
concerns every entrypoint shares: the E6.2/E6.2b statement screen that
refuses before any connection is touched, NX-6's standalone NFR-108
path-containment service (GP3), and the §4b NX-3 wire that turns a
backend fault into the one structured shape. Chokepoint-internal by
§6.2 — reachable only through `guard`, never imported from outside the
package (the import-graph gate enforces it).
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from localdata_mcp.nexus.config.models import ConfigModel
from localdata_mcp.nexus.error.wire import wrap
from localdata_mcp.nexus.persistence.manager import PersistenceNexus

from .chunk_registry import ChunkRegistry, StreamAdmissionRefusedError
from .path_contain import AccessMode, contain
from .resource_bounds import ResourceBounds, ResourceRefusedError
from .sparql_validate import screen_read, screen_update
from .sql_validate.cache import ValidationCache
from .sql_validate.walker import SqlClassification
from .types import (
    _SPARQL_BACKEND_KINDS,
    GuardRefusedError,
    GuardedExecutionError,
    QueryRequest,
)


class _GuardCore:
    """The guard's shared state and cross-cutting seams (screening,
    containment, the NX-3 wire) — the base every surface mixin and the
    composed `Chokepoint` inherit."""

    def __init__(self, config: ConfigModel, persistence: PersistenceNexus) -> None:
        self._config = config
        self._persistence = persistence
        self._cache = ValidationCache(config.security.validation_cache_entries)
        self._bounds = ResourceBounds(config)
        self._registry = ChunkRegistry(config, self._bounds)

    # -- NX-6's standalone path-containment service (GP3) -------------

    def contain_path(self, candidate: str | Path, *, mode: AccessMode) -> Path:
        """The one NFR-108 check, against the live operator
        `allowed_paths` — NX-8 writes and ephemeral opens cross here."""
        return contain(candidate, self._config.security.allowed_paths, mode=mode)

    # -- screening ----------------------------------------------------

    def _screen_read_side(self, request: QueryRequest, backend_kind: str) -> str:
        """Classify and refuse everything the read entrypoint may not
        carry; returns the category. Every extracted path literal is
        contained read-side (NFR-108)."""
        if request.language == "sparql" or backend_kind in _SPARQL_BACKEND_KINDS:
            screen_read(request.text)
            return "query"
        classification = self._cache.classify(request.text, backend_kind)
        if classification.category in ("mutation", "local_file_write"):
            raise GuardRefusedError(
                f"query refused: the statement classifies as "
                f"{classification.category} — writes cross "
                "guarded_mutation (NFR-113)"
            )
        if classification.contains_mutation_nodes:
            raise GuardRefusedError(
                "query refused: the statement embeds mutation constructs "
                "(a data-modifying CTE or similar) — writes cross "
                "guarded_mutation (NFR-113)"
            )
        self._contain_all(classification, mode="read")
        return classification.category

    def _screen_write_side(self, request: QueryRequest, backend_kind: str) -> str:
        """The mutation entrypoint's screen: only the enumerated write
        categories pass, path literals contained write-side."""
        if request.language == "sparql" or backend_kind in _SPARQL_BACKEND_KINDS:
            screen_update(request.text)
            return "mutation"
        classification = self._cache.classify(request.text, backend_kind)
        if classification.category in ("query", "local_file_read"):
            raise GuardRefusedError(
                f"mutation refused: the statement classifies as "
                f"{classification.category} — reads cross guarded_query "
                "(entrypoint discipline, §6.2)"
            )
        self._contain_all(classification, mode="write")
        return classification.category

    def _contain_all(self, classification: SqlClassification, mode: AccessMode) -> None:
        for literal in classification.path_literals:
            self.contain_path(literal, mode=mode)

    # -- the NX-3 wire (§4b) ------------------------------------------

    @contextmanager
    def _wired(self, endpoint_name: str, backend_kind: str) -> Iterator[None]:
        """Backend failures become the one structured shape, the E4.0
        fault signal firing synchronously for connection-class errors.
        The chokepoint's own refusals pass through untouched — they are
        already structured verdicts, not backend faults."""
        try:
            yield
        except (
            GuardRefusedError,
            ResourceRefusedError,
            StreamAdmissionRefusedError,
        ):
            raise
        except Exception as failure:
            structured = wrap(
                failure,
                backend_kind,
                fault_sink=self._persistence,
                record_id=endpoint_name,
            )
            raise GuardedExecutionError(structured) from failure
