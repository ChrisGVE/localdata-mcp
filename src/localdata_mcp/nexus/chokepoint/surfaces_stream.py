"""localdata_mcp/nexus/chokepoint/surfaces_stream.py — the streaming
handoff to the E6.6 ChunkRegistry (§5, I-2/I-4).

`_StreamingSurface` carries the cutover between an inline `Result` and a
registered stream. `query_or_stream` peeks the pull source (the same one
`open_query_stream` uses) and returns a plain `Result` when the answer
sits inside the S8 23a/23b inline budget, otherwise registers a stream
with the peeked frames re-chained — nothing is re-executed. For a
declared endpoint the guard pins the NX-5 connection for the stream's
life and the registry's `on_close` returns it to the pool (§5's
declared trade-off); for a load-then-serve source (`serve_result`) the
stream pins no connection, so the row-24 per-endpoint cap deliberately
does not apply. `fetch_next_chunk` serves under cursor semantics and
closes the stream once drained. Chokepoint-internal by §6.2: composed
into `Chokepoint` (guard.py).
"""

from __future__ import annotations

import uuid
from contextlib import ExitStack
from itertools import chain
from pathlib import Path

import pandas as pd

from .chunk_registry import StreamStatus
from .core import _GuardCore
from .execution import iter_frames
from .types import (
    QueryRequest,
    Result,
    ServedChunk,
    StreamOpened,
    approx_render_bytes,
    result_from_frames,
)


class _StreamingSurface(_GuardCore):
    """The streaming half of the I-4 cutover and the chunk-serving
    cursor over the E6.6 ChunkRegistry."""

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
                    byte_estimate += approx_render_bytes(frame)
                    if row_count > max_rows or byte_estimate > max_bytes:
                        exhausted = False
                        break
                if exhausted:
                    stack.close()
                    return result_from_frames(peeked, category)
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
        if len(result.rows) <= max_rows and approx_render_bytes(frame) <= max_bytes:
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
