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
from localdata_mcp.nexus.persistence.ephemeral import EphemeralEngineKind
from localdata_mcp.nexus.persistence.manager import (
    PersistenceNexus,
    UnknownEndpointError,
)

from itertools import chain

from .execution import execute_mutation, fetch_bounded, iter_frames
from .chunk_registry import (
    ChunkAlreadyServedError,
    ChunkNotServableError,
    ChunkRegistry,
    StreamAdmissionRefusedError,
    StreamExpiredError,
    StreamStatus,
)
from .path_contain import AccessMode, contain
from .resource_bounds import ResourceBounds, ResourceRefusedError
from .sparql_validate import screen_read, screen_update
from .sql_validate.cache import ValidationCache
from .sql_validate.walker import SqlClassification

# Re-exported at the seam: tool modules catch the NFR-114 name miss,
# name the ephemeral engine kind, and catch the resource-admission
# refusal through the guard — never by importing NX-5 or a chokepoint
# internal (FR-802 / §6.2's import rule, enforced by the import gate).
__all__ = [
    "UnknownEndpointError",
    "EphemeralEngineKind",
    "ResourceRefusedError",
    "StreamAdmissionRefusedError",
    "StreamExpiredError",
    "ChunkAlreadyServedError",
    "ChunkNotServableError",
    "StreamOpened",
    "ServedChunk",
]

Language = Literal["sql", "sparql"]

# Endpoint kinds whose statements are SPARQL regardless of the caller's
# language hint (E8.3): the language belongs to the DECLARED backend,
# so an rdf endpoint's text always crosses the E6.2b screen.
_SPARQL_BACKEND_KINDS = frozenset({"rdf"})


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


@dataclass(frozen=True)
class EndpointSummary:
    """One declared endpoint's caller-visible state (I-1) — name,
    kind, posture, last-known health. Capability-narrow like every
    guard return (GP3): the health detail is the ALREADY-REDACTED
    NX-5 text (E5's at-construction redaction), never a DSN."""

    name: str
    backend_kind: str
    posture: str
    healthy: bool | None
    health_detail: str


@dataclass(frozen=True)
class StreamOpened:
    """The streaming half of the I-4 cutover: a result too large for
    the inline budget is registered with the ChunkRegistry and the
    caller receives this reference — stream id, the column names, and
    the currently-servable count (T10: derived from the live buffer,
    never a promised total)."""

    stream_id: str
    columns: tuple[str, ...]
    advertised_chunks: int


@dataclass(frozen=True)
class ServedChunk:
    """One `fetch_chunk` answer: the chunk's rows plus the stream's
    live state — `total_chunks` is populated only once the source is
    exhausted (§5: the buffer never claims a total it cannot know)."""

    stream_id: str
    chunk_id: int | None
    columns: tuple[str, ...]
    rows: tuple[tuple[Any, ...], ...]
    advertised_chunks: int
    exhausted: bool
    total_chunks: int | None
    closed: bool = False


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

    def query_or_stream(
        self, endpoint_name: str, request: QueryRequest
    ) -> "Result | StreamOpened":
        """I-4's cutover for the genuinely-streaming SQL path: peek the
        result through the same pull source `open_query_stream` uses;
        a result inside the inline budget (S8 rows 23a/23b) comes back
        as a plain `Result` and the connection returns immediately, a
        larger one is registered as a stream with the peeked frames
        re-chained (nothing is re-executed). The byte side is a
        conservative rendered-text estimate — the envelope's exact
        markdown measurement still governs final rendering."""
        record = self._persistence.record(endpoint_name)
        category = self._screen_read_side(request, record.backend_kind)
        max_rows = self._config.response.inline_max_rows
        max_bytes = self._config.response.inline_max_bytes
        stack = ExitStack()
        try:
            with self._wired(endpoint_name, record.backend_kind):
                connection = stack.enter_context(
                    self._persistence.connection(endpoint_name)
                )
                frames = iter_frames(
                    connection,
                    request.text,
                    request.parameters,
                    self._config.query.default_chunk_size,
                )
                peeked: list[pd.DataFrame] = []
                row_count = 0
                byte_estimate = 0
                exhausted = True
                for frame in frames:
                    peeked.append(frame)
                    row_count += len(frame)
                    byte_estimate += _approx_render_bytes(frame)
                    if row_count > max_rows or byte_estimate > max_bytes:
                        exhausted = False
                        break
                if exhausted:
                    stack.close()
                    return _result_from_frames(peeked, category)
                stream_id = f"{endpoint_name}:{uuid.uuid4().hex}"
                self._registry.open_stream(
                    stream_id,
                    endpoint_name,
                    chain(iter(peeked), frames),
                    "streaming",
                    on_close=stack.close,
                )
        except BaseException:
            stack.close()
            raise
        return StreamOpened(
            stream_id=stream_id,
            columns=tuple(str(column) for column in peeked[0].columns),
            advertised_chunks=self._registry.advertised_count(stream_id),
        )

    def serve_result(self, result: Result, source_name: str) -> "Result | StreamOpened":
        """I-2/I-4's cutover for load-then-serve sources (`read_file`,
        `query_file`): the ALREADY-ADMITTED result passes through
        inside the inline budget, beyond it its rows are sliced into
        chunk-registry frames served from the admitted in-memory
        buffer. The stream pins no NX-5 connection, so the row-24
        per-endpoint cap deliberately does not apply (the stream id
        doubles as its own registry endpoint name) — aggregate memory
        admission and the idle TTL are the operative bounds (I-2)."""
        max_rows = self._config.response.inline_max_rows
        max_bytes = self._config.response.inline_max_bytes
        frame = pd.DataFrame(list(result.rows), columns=list(result.columns))
        if len(result.rows) <= max_rows and _approx_render_bytes(frame) <= max_bytes:
            return result
        chunk_size = self._config.query.default_chunk_size
        slices = [
            frame.iloc[start : start + chunk_size]
            for start in range(0, len(frame), chunk_size)
        ]
        stream_id = f"file:{Path(source_name).name}:{uuid.uuid4().hex}"
        self._registry.open_stream(
            stream_id,
            stream_id,  # its own cap bucket — see docstring
            iter(slices),
            "load_then_serve",
            on_close=lambda: None,
        )
        return StreamOpened(
            stream_id=stream_id,
            columns=result.columns,
            advertised_chunks=self._registry.advertised_count(stream_id),
        )

    def fetch_next_chunk(self, stream_id: str) -> ServedChunk:
        """I-4's `fetch_chunk`: serve the next chunk under cursor
        semantics. Once the source is exhausted and drained the answer
        carries the final total as metadata and the stream is CLOSED
        (its pinned connection released ahead of the TTL) — a further
        fetch is the structured expired refusal."""
        served = self._registry.serve_next(stream_id)
        status = self._registry.stream_status(stream_id)
        if served is None:
            self._registry.close_stream(stream_id)
            return ServedChunk(
                stream_id=stream_id,
                chunk_id=None,
                columns=(),
                rows=(),
                advertised_chunks=0,
                exhausted=True,
                total_chunks=status.total_chunks,
                closed=True,
            )
        chunk_id, payload = served
        return ServedChunk(
            stream_id=stream_id,
            chunk_id=chunk_id,
            columns=tuple(str(column) for column in payload.columns),
            rows=tuple(tuple(row) for row in payload.itertuples(index=False)),
            advertised_chunks=status.advertised_chunks,
            exhausted=status.exhausted,
            total_chunks=status.total_chunks,
        )

    def request_chunk(self, stream_id: str, chunk_id: int) -> pd.DataFrame:
        return self._registry.request_chunk(stream_id, chunk_id)

    def close_stream(self, stream_id: str) -> None:
        self._registry.close_stream(stream_id)

    def stream_status(self, stream_id: str) -> StreamStatus:
        return self._registry.stream_status(stream_id)

    def evict_idle_streams(self) -> tuple[str, ...]:
        return self._registry.evict_idle()

    # -- the ephemeral local-file seam (I-2, NFR-114 i-b) -------------

    def guarded_file_query(
        self,
        path: str | Path,
        request: QueryRequest,
        engine_kind: EphemeralEngineKind,
    ) -> Result:
        """Ad-hoc SQL over a local SQLite/DuckDB file: contained FIRST
        (NFR-108 — an out-of-tree path never reaches NX-5), screened by
        the same allow-list, and READ-ONLY by default — a mutation
        needs an operator `ephemeral_write_paths` grant (§5's ephemeral
        posture rule; NFR-114's local-file branch). The connection
        never outlives this call (no NX-5 pool is pinned, so the row-24
        stream cap does not apply — I-2)."""
        real = self.contain_path(path, mode="read")
        classification = self._cache.classify(request.text, engine_kind)
        wants_write = (
            classification.category in ("mutation", "local_file_write")
            or classification.contains_mutation_nodes
        )
        connection_spec = self._persistence.open_ephemeral(str(real), engine_kind)
        if wants_write and connection_spec.posture != "read_write":
            raise GuardRefusedError(
                f"mutation refused: ad-hoc file source {str(real)!r} is "
                "read-only by default (NFR-114) — only an operator "
                "security.ephemeral_write_paths grant makes it writable"
            )
        self._contain_all(classification, mode="write" if wants_write else "read")
        with self._wired(f"file:{real}", engine_kind):
            with connection_spec.open() as live:
                if wants_write:
                    affected = execute_mutation(live, request.text, request.parameters)
                    return Result(
                        columns=(),
                        rows=(),
                        category=classification.category,
                        affected_rows=affected,
                    )
                columns, rows = fetch_bounded(
                    live,
                    request.text,
                    request.parameters,
                    self._bounds,
                    self._config.query.default_chunk_size,
                )
        return Result(columns=columns, rows=rows, category=classification.category)

    # -- the endpoint-enumeration seam (I-1) --------------------------

    def endpoint_summaries(self) -> tuple[EndpointSummary, ...]:
        """Every declared endpoint's summary, backend-kind-agnostic —
        the one discovery surface NFR-114 refusals point callers to
        (list_endpoints reaches NX-5's state through here, §6.2)."""
        return tuple(
            self.endpoint_summary(name) for name in self._persistence.endpoint_names()
        )

    def endpoint_summary(self, name: str) -> EndpointSummary:
        """One named endpoint's summary (E8.3) — the capability-narrow
        kind/posture view store-family tools dispatch on (a kv call
        against a graph endpoint speaks graph tables); raises the same
        `UnknownEndpointError` every guard resolution raises."""
        record = self._persistence.record(name)
        health = record.health
        return EndpointSummary(
            name=record.name,
            backend_kind=record.backend_kind,
            posture=record.posture,
            healthy=None if health is None else health.healthy,
            health_detail="" if health is None else health.detail,
        )

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


def _approx_render_bytes(frame: pd.DataFrame) -> int:
    """A conservative estimate of a frame's rendered-text weight (the
    inline_max_bytes side of the S8 23a/23b cutover): per-cell string
    length plus separator overhead. The envelope's exact markdown
    measurement remains the rendering authority — this estimate only
    decides whether a stream is opened."""
    if frame.empty:
        return 0
    cells = int(frame.map(lambda value: len(str(value))).to_numpy().sum())
    separators = frame.shape[0] * (3 * frame.shape[1] + 1)
    return cells + separators


def _result_from_frames(frames: list[pd.DataFrame], category: str) -> Result:
    """The peeked frames as one capability-narrow inline Result; an
    empty peek is the explicit zero-row result (columns unknowable
    without a cursor description — the envelope renders the zero
    statement)."""
    if not frames:
        return Result(columns=(), rows=(), category=category)
    columns = tuple(str(column) for column in frames[0].columns)
    rows = tuple(
        tuple(row) for frame in frames for row in frame.itertuples(index=False)
    )
    return Result(columns=columns, rows=rows, category=category)
