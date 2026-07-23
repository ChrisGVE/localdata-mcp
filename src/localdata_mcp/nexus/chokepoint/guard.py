"""localdata_mcp/nexus/chokepoint/guard.py — NX-6's two entrypoints (E6.1).

THE chokepoint (§4a, §6.2, §8 NX-6): `guarded_query` and
`guarded_mutation` are the only names a tool module may import to touch
a backend, and NX-6's path-containment service (`contain_path`) is the
only other data-touching seam GP3 names. Every call: resolve the
operator-declared endpoint NAME against NX-5 (tools never see a
connection — FR-802), screen the statement (the E6.2 SQL allow-list
through the E6.3 cache, or the E6.2b SPARQL screen), enforce posture
(NFR-113: the mutation category, write-side local-file constructs, and
SPARQL updates all require `read_write`), contain every extracted path
literal (E6.4, NFR-108), admit retention through the E6.5 dynamic gate,
execute over the NX-5 connection, and hand back a capability-narrow
frozen `Result` — no `execute()`, no `Engine`, no live cursor (GP3's
corollary). A backend failure crosses NX-3's wire IN here (§4b): the
E4.0 fault signal fires synchronously against the record before the
structured shape is raised. The streaming path hands off to the E6.6
ChunkRegistry, the guard holding the pinned NX-5 connection the
registry's `on_close` releases (§5). Ephemeral local-file opens will
cross `contain_path` before NX-5 when the Ingest epic lands — the
service is live here now. Neighbors: every sibling chokepoint module
composes here; nexus/persistence resolves; nexus/error shapes.
"""

from __future__ import annotations

import uuid
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Literal, Mapping

import pandas as pd

from localdata_mcp.nexus.config.models import ConfigModel
from localdata_mcp.nexus.error.model import StructuredError
from localdata_mcp.nexus.error.wire import wrap
from localdata_mcp.nexus.persistence.manager import PersistenceNexus

from .execution import execute_mutation, fetch_bounded, iter_frames
from .chunk_registry import (
    ChunkRegistry,
    StreamAdmissionRefusedError,
    StreamStatus,
)
from .path_contain import AccessMode, contain
from .resource_bounds import ResourceBounds, ResourceRefusedError
from .sparql_validate import screen_read, screen_update
from .sql_validate.cache import ValidationCache
from .sql_validate.walker import SqlClassification

Language = Literal["sql", "sparql"]


@dataclass(frozen=True)
class QueryRequest:
    """One statement as a tool module hands it over: text, language,
    and bound parameters (values bind through the driver — they are
    never spliced into the statement text)."""

    text: str
    language: Language = "sql"
    parameters: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class Result:
    """The capability-narrow return (GP3 corollary): columns, rows,
    the statement's classified category, and the driver-reported
    affected count for writes — data only, nothing live."""

    columns: tuple[str, ...]
    rows: tuple[tuple[Any, ...], ...]
    category: str
    affected_rows: int | None = None

    @property
    def row_count(self) -> int:
        return len(self.rows)


class GuardRefusedError(PermissionError):
    """An entrypoint or posture refusal (NFR-113) — structured, named,
    shaped through NX-3 by the tool wrapper."""


class GuardedExecutionError(RuntimeError):
    """A backend failure, already crossed through NX-3's wire (§4b):
    `structured` is the one redacted shape; the E4.0 fault signal has
    already fired for connection-class errors."""

    def __init__(self, structured: StructuredError) -> None:
        super().__init__(structured.message)
        self.structured = structured


class Chokepoint:
    """The one guarded data-access surface (§6.2's `NX6`)."""

    def __init__(self, config: ConfigModel, persistence: PersistenceNexus) -> None:
        self._config = config
        self._persistence = persistence
        self._cache = ValidationCache(config.security.validation_cache_entries)
        self._bounds = ResourceBounds(config)
        self._registry = ChunkRegistry(config, self._bounds)

    # -- the two entrypoints (§6.2) -----------------------------------

    def guarded_query(self, endpoint_name: str, request: QueryRequest) -> Result:
        """The read path — permitted on ANY posture (NFR-113)."""
        record = self._persistence.record(endpoint_name)
        category = self._screen_read_side(request, record.backend_kind)
        with self._wired(endpoint_name, record.backend_kind):
            with self._persistence.connection(endpoint_name) as connection:
                columns, rows = fetch_bounded(
                    connection,
                    request.text,
                    request.parameters,
                    self._bounds,
                    self._config.query.default_chunk_size,
                )
        return Result(columns=columns, rows=rows, category=category)

    def guarded_mutation(self, endpoint_name: str, request: QueryRequest) -> Result:
        """The write path — `read_write` posture only (NFR-113), for
        the enumerated mutation category, write-side local-file
        constructs, and SPARQL update forms."""
        record = self._persistence.record(endpoint_name)
        if record.posture != "read_write":
            raise GuardRefusedError(
                f"mutation refused: endpoint {endpoint_name!r} is declared "
                "read_only (NFR-113) — the operator must declare "
                "read_write posture for writes"
            )
        category = self._screen_write_side(request, record.backend_kind)
        with self._wired(endpoint_name, record.backend_kind):
            with self._persistence.connection(endpoint_name) as connection:
                affected = execute_mutation(
                    connection, request.text, request.parameters
                )
        return Result(columns=(), rows=(), category=category, affected_rows=affected)

    # -- the streaming handoff (§5, E6.6) -----------------------------

    def open_query_stream(self, endpoint_name: str, request: QueryRequest) -> str:
        """Validate as a read, then hand off to the ChunkRegistry: the
        guard pins the NX-5 connection for the stream's life and the
        registry's `on_close` (explicit close or TTL eviction) is what
        returns it to the pool (§5's declared trade-off)."""
        record = self._persistence.record(endpoint_name)
        self._screen_read_side(request, record.backend_kind)
        stack = ExitStack()
        stream_id = f"{endpoint_name}:{uuid.uuid4().hex}"
        try:
            with self._wired(endpoint_name, record.backend_kind):
                connection = stack.enter_context(
                    self._persistence.connection(endpoint_name)
                )
                source = iter_frames(
                    connection,
                    request.text,
                    request.parameters,
                    self._config.query.default_chunk_size,
                )
                self._registry.open_stream(
                    stream_id,
                    endpoint_name,
                    source,
                    "streaming",
                    on_close=stack.close,
                )
        except BaseException:
            stack.close()
            raise
        return stream_id

    def request_chunk(self, stream_id: str, chunk_id: int) -> pd.DataFrame:
        return self._registry.request_chunk(stream_id, chunk_id)

    def close_stream(self, stream_id: str) -> None:
        self._registry.close_stream(stream_id)

    def stream_status(self, stream_id: str) -> StreamStatus:
        return self._registry.stream_status(stream_id)

    def evict_idle_streams(self) -> tuple[str, ...]:
        return self._registry.evict_idle()

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
        if request.language == "sparql":
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
        if request.language == "sparql":
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
