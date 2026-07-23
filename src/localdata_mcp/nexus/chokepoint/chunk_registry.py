"""localdata_mcp/nexus/chokepoint/chunk_registry.py — the §5 ChunkRegistry (E6.6).

The ONE owner of streaming-buffer state (§4c, §8 NX-6 Owns): every
live stream's chunk buffer, its accounting, and its lifecycle. The §5
rules, mechanically:

- **Cursor semantics** — a chunk leaves the buffer the moment it is
  served; re-requesting a served `chunk_id` is a structured
  already-served refusal (the caller re-issues the query if it needs
  the data again — the declared departure from `main`'s
  retained-buffer semantics).
- **K/B look-ahead with backpressure** — the source is PULLED, never
  pushed: after every serve the buffer tops up to at most
  `chunk_buffer_max_chunks` chunks / `chunk_buffer_max_bytes` bytes
  (S8 rows 8-9, genuinely-streaming registries only), so "the reader
  pauses at the bound and resumes on retrieval" is structural — an
  un-pulled source cannot occupy memory. A load-then-serve source is
  buffered whole at open (its bound is the upfront `admit_load` gate,
  §5); requesting past the look-ahead without retrieving buffered
  chunks first is refused, keeping the bound a true cap.
- **T10 closure** — the advertised count is computed lazily from the
  buffer's live contents at request time (`advertised_count`), never
  cached or pre-declared; once exhausted the final total is reported
  as metadata, never as servable chunks.
- **Idle-TTL eviction under the retrieval lock** — `evict_idle` takes
  the same lock as `request_chunk` (S8 row 10), returning the NX-5
  connection via the stream's `on_close` and its bytes to the budget;
  an evicted or unknown stream answers every later request with a
  structured expired-and-non-resumable refusal.
- **Per-row extrapolated accounting** — the first chunk is measured
  exactly (`memory_usage(deep=True)`, the kept pattern at
  `streaming/sources.py:288`), later chunks are attributed by per-row
  extrapolation, re-measured on schema change — never a per-chunk deep
  traversal on the hot path. Residency is charged to the E6.5
  aggregate ledger, so all live streams jointly respect the one
  ceiling (§5 bound 2): a pull the ledger refuses simply stops the
  top-up (buffered chunks stay servable; consumption frees headroom).
- **Row-24 admission** — opening a stream past
  `max_concurrent_streams_per_endpoint` is a structured refusal naming
  `close_stream` (the cap is what guarantees row 3's interactive
  connection headroom).

Neighbors: guard.py opens streams (holding the NX-5 connection its
`on_close` releases) and shapes every refusal through NX-3;
resource_bounds.py keeps the aggregate ledger.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable, Iterator, Literal

import pandas as pd

from localdata_mcp.nexus.config.models import ConfigModel

from .resource_bounds import ResourceBounds, ResourceRefusedError

SourceKind = Literal["streaming", "load_then_serve"]


class StreamAdmissionRefusedError(RuntimeError):
    """The row-24 per-endpoint stream cap is reached — the refusal
    names `close_stream` per I-4; guard.py shapes it through NX-3."""


class StreamExpiredError(LookupError):
    """The stream is TTL-evicted, closed, or never existed — expired
    and non-resumable (§5); the caller re-issues the query."""


class ChunkAlreadyServedError(LookupError):
    """Cursor semantics (§5): a served chunk left the buffer; the
    caller re-issues the originating query if it needs the data."""


class ChunkNotServableError(LookupError):
    """The requested chunk is beyond the look-ahead bound or beyond
    the exhausted source's end — the bound stays a true cap."""


@dataclass(frozen=True)
class StreamStatus:
    """One stream's lazily-computed public state (T10)."""

    advertised_chunks: int
    exhausted: bool
    total_chunks: int | None  # metadata once exhausted, never servable


@dataclass
class _Stream:
    """Internal per-stream state; every access is under the registry
    lock."""

    endpoint_name: str
    source_kind: SourceKind
    source: Iterator[pd.DataFrame]
    on_close: Callable[[], None]
    buffer: "OrderedDict[int, pd.DataFrame]" = field(default_factory=OrderedDict)
    served: set[int] = field(default_factory=set)
    chunk_bytes: dict[int, int] = field(default_factory=dict)
    next_pull_index: int = 0
    exhausted: bool = False
    per_row_bytes: float | None = None
    schema_signature: tuple[tuple[str, str], ...] | None = None
    resident_bytes: int = 0
    last_access: float = 0.0

    @property
    def total_chunks(self) -> int | None:
        return self.next_pull_index if self.exhausted else None


class ChunkRegistry:
    """The registry of live streams — one instance per process, owned
    by the chokepoint. One re-entrant lock covers retrieval, eviction,
    and admission (§5: eviction cannot race an in-flight request)."""

    def __init__(
        self,
        config: ConfigModel,
        bounds: ResourceBounds,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._max_chunks = config.query.chunk_buffer_max_chunks
        self._max_bytes = config.query.chunk_buffer_max_bytes
        self._idle_ttl = config.query.stream_idle_ttl_seconds
        self._endpoint_cap = config.query.max_concurrent_streams_per_endpoint
        self._bounds = bounds
        self._clock = clock
        self._streams: dict[str, _Stream] = {}
        self._lock = threading.RLock()

    # -- lifecycle ----------------------------------------------------

    def open_stream(
        self,
        stream_id: str,
        endpoint_name: str,
        source: Iterator[pd.DataFrame],
        source_kind: SourceKind,
        on_close: Callable[[], None],
    ) -> None:
        """Admit and register a stream (row 24), then prime its buffer:
        a streaming source fills to the look-ahead bound, a
        load-then-serve source is read whole (its admission was the
        upfront `admit_load` gate — FR-404's documented behavior)."""
        with self._lock:
            live = sum(
                1
                for stream in self._streams.values()
                if stream.endpoint_name == endpoint_name
            )
            if live >= self._endpoint_cap:
                raise StreamAdmissionRefusedError(
                    f"stream admission refused: endpoint {endpoint_name!r} "
                    f"already has {live} live streams, the "
                    f"max_concurrent_streams_per_endpoint cap (S8 row 24) — "
                    "call close_stream on a finished stream first"
                )
            if stream_id in self._streams:
                raise StreamAdmissionRefusedError(
                    f"stream admission refused: stream id {stream_id!r} is already live"
                )
            stream = _Stream(
                endpoint_name=endpoint_name,
                source_kind=source_kind,
                source=source,
                on_close=on_close,
                last_access=self._clock(),
            )
            self._streams[stream_id] = stream
            self._bounds.charge(stream_id, 0)
            try:
                if source_kind == "load_then_serve":
                    while not stream.exhausted:
                        self._pull_one(stream_id, stream)
                else:
                    self._top_up(stream_id, stream)
            except BaseException:
                # A priming failure (a bad statement surfacing on the
                # first pull, a refused whole-load) unregisters the
                # stream; the CALLER still owns the connection during
                # open, so `on_close` is deliberately not called here.
                self._streams.pop(stream_id, None)
                self._bounds.release(stream_id)
                raise

    def close_stream(self, stream_id: str) -> None:
        """Explicit release (I-4's `close_stream`): idempotent — the
        teardown path never fails on bookkeeping."""
        with self._lock:
            stream = self._streams.pop(stream_id, None)
            if stream is None:
                return
            self._release(stream_id, stream)

    def close_all(self) -> None:
        """§4e teardown: release every live stream — each pinned NX-5
        connection returns to its pool BEFORE the pools dispose."""
        with self._lock:
            for stream_id in tuple(self._streams):
                self._release(stream_id, self._streams.pop(stream_id))

    def evict_idle(self) -> tuple[str, ...]:
        """Evict every stream idle past the TTL (S8 row 10), under the
        retrieval lock — returning each one's NX-5 connection (via
        `on_close`) and its bytes to the budget ledger."""
        with self._lock:
            now = self._clock()
            expired = tuple(
                stream_id
                for stream_id, stream in self._streams.items()
                if now - stream.last_access > self._idle_ttl
            )
            for stream_id in expired:
                self._release(stream_id, self._streams.pop(stream_id))
            return expired

    def live_stream_count(self, endpoint_name: str) -> int:
        with self._lock:
            return sum(
                1
                for stream in self._streams.values()
                if stream.endpoint_name == endpoint_name
            )

    # -- retrieval (§5 cursor semantics) ------------------------------

    def request_chunk(self, stream_id: str, chunk_id: int) -> pd.DataFrame:
        """Serve one chunk and evict it from the buffer, then top the
        look-ahead back up (the pull that "resumes the reader")."""
        with self._lock:
            stream = self._checked_stream(stream_id)
            if chunk_id in stream.served:
                raise ChunkAlreadyServedError(
                    f"chunk {chunk_id} of stream {stream_id!r} was already "
                    "served and left the buffer (cursor semantics, §5) — "
                    "re-issue the originating query if the data is needed "
                    "again"
                )
            if chunk_id not in stream.buffer:
                self._pull_until(stream_id, stream, chunk_id)
            payload = stream.buffer.pop(chunk_id)
            stream.served.add(chunk_id)
            stream.resident_bytes -= stream.chunk_bytes.pop(chunk_id, 0)
            self._charge_current(stream_id, stream)
            stream.last_access = self._clock()
            if stream.source_kind == "streaming":
                self._top_up(stream_id, stream)
            return payload

    def serve_next(self, stream_id: str) -> tuple[int, pd.DataFrame] | None:
        """Serve the lowest servable chunk (I-4's `fetch_chunk` shape):
        the cursor semantics of `request_chunk` under the same lock,
        with the NEXT id derived from the live buffer — None once the
        source is exhausted and the buffer is drained (the caller
        reports the final total as metadata, §5/T10)."""
        with self._lock:
            stream = self._checked_stream(stream_id)
            if not stream.buffer and not stream.exhausted:
                self._pull_one(stream_id, stream)
            if not stream.buffer:
                stream.last_access = self._clock()
                return None
            chunk_id = min(stream.buffer)
            return chunk_id, self.request_chunk(stream_id, chunk_id)

    def advertised_count(self, stream_id: str) -> int:
        """T10: the count of currently-servable chunks, derived from
        the live buffer at call time — there is no second number."""
        with self._lock:
            return len(self._checked_stream(stream_id).buffer)

    def stream_status(self, stream_id: str) -> StreamStatus:
        with self._lock:
            stream = self._checked_stream(stream_id)
            return StreamStatus(
                advertised_chunks=len(stream.buffer),
                exhausted=stream.exhausted,
                total_chunks=stream.total_chunks,
            )

    # -- internals (lock held by every caller) ------------------------

    def _checked_stream(self, stream_id: str) -> _Stream:
        """The live stream, with its own lazy TTL check — an idle
        stream expires on the next touch even between sweeps."""
        stream = self._streams.get(stream_id)
        if stream is not None and (self._clock() - stream.last_access > self._idle_ttl):
            self._release(stream_id, self._streams.pop(stream_id))
            stream = None
        if stream is None:
            raise StreamExpiredError(
                f"stream {stream_id!r} is expired and non-resumable (§5) — "
                "re-issue the originating query"
            )
        return stream

    def _pull_until(self, stream_id: str, stream: _Stream, chunk_id: int) -> None:
        """Pull forward until `chunk_id` is buffered, refusing past the
        look-ahead bound or the source's end — the K/B cap holds even
        for an out-of-order caller."""
        while chunk_id >= stream.next_pull_index and not stream.exhausted:
            if len(stream.buffer) >= self._max_chunks or (
                stream.resident_bytes >= self._max_bytes
            ):
                raise ChunkNotServableError(
                    f"chunk {chunk_id} of stream {stream_id!r} is beyond the "
                    "look-ahead bound (S8 rows 8-9) — retrieve the buffered "
                    "chunks first"
                )
            self._pull_one(stream_id, stream)
        if chunk_id not in stream.buffer:
            total = stream.total_chunks
            raise ChunkNotServableError(
                f"chunk {chunk_id} of stream {stream_id!r} does not exist — "
                f"the exhausted source produced {total} chunks"
            )

    def _pull_one(self, stream_id: str, stream: _Stream) -> None:
        """Advance the source one chunk, measure or extrapolate its
        bytes, and charge the aggregate ledger. A ledger refusal on the
        pre-charge stops the pull cleanly (nothing was read); only the
        first-ever chunk is pulled unestimated."""
        projected = self._projected_bytes(stream)
        if projected is not None:
            # Pre-charge the projection so an over-ceiling pull is
            # refused BEFORE the chunk is materialized; roll back to
            # the actual residency immediately after the verdict.
            try:
                self._bounds.charge(stream_id, stream.resident_bytes + projected)
            finally:
                self._charge_current(stream_id, stream)
        try:
            chunk = next(stream.source)
        except StopIteration:
            stream.exhausted = True
            return
        index = stream.next_pull_index
        stream.next_pull_index += 1
        chunk_bytes = self._attributed_bytes(stream, chunk)
        stream.buffer[index] = chunk
        stream.chunk_bytes[index] = chunk_bytes
        stream.resident_bytes += chunk_bytes
        self._bounds.charge(stream_id, stream.resident_bytes)

    def _top_up(self, stream_id: str, stream: _Stream) -> None:
        """Fill the look-ahead to the K/B bound; a budget refusal
        pauses the top-up rather than failing the serve (backpressure
        against the aggregate ceiling, §5 bound 2)."""
        while (
            not stream.exhausted
            and len(stream.buffer) < self._max_chunks
            and stream.resident_bytes < self._max_bytes
        ):
            try:
                self._pull_one(stream_id, stream)
            except ResourceRefusedError:
                return

    def _projected_bytes(self, stream: _Stream) -> int | None:
        """The next pull's size projection from the per-row estimate
        and the last chunk's row count — None before any measurement."""
        if stream.per_row_bytes is None or not stream.chunk_bytes:
            return None
        last_bytes = next(reversed(stream.chunk_bytes.values()))
        return max(last_bytes, 0)

    def _attributed_bytes(self, stream: _Stream, chunk: pd.DataFrame) -> int:
        """First chunk (and any schema change): measured exactly with
        `memory_usage(deep=True)`; afterwards: rows × per-row estimate
        — the kept `streaming/sources.py:288` pattern, off the hot
        path."""
        signature = tuple(
            (str(name), str(dtype)) for name, dtype in chunk.dtypes.items()
        )
        if stream.per_row_bytes is None or signature != stream.schema_signature:
            measured = int(chunk.memory_usage(deep=True).sum())
            stream.schema_signature = signature
            if len(chunk):
                stream.per_row_bytes = measured / len(chunk)
            return measured
        return int(len(chunk) * stream.per_row_bytes)

    def _charge_current(self, stream_id: str, stream: _Stream) -> None:
        """Re-assert the stream's actual residency on the ledger (a
        decreasing charge cannot be refused)."""
        try:
            self._bounds.charge(stream_id, max(stream.resident_bytes, 0))
        except ResourceRefusedError:
            # The actual residency was admitted when it accrued; only a
            # concurrent ledger shift can refuse here — the stream's
            # own books stay correct either way.
            pass

    def _release(self, stream_id: str, stream: _Stream) -> None:
        """Return the stream's bytes to the ledger and its NX-5
        connection to the pool (§5 rule 3)."""
        self._bounds.release(stream_id)
        stream.on_close()
