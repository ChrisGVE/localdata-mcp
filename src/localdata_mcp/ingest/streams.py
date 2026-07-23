"""localdata_mcp/ingest/streams.py — the I-4 chunk-retrieval tools (E8.4).

The tool surface over the E6.6 ChunkRegistry through the guard:
`fetch_chunk` serves the NEXT chunk under §5's cursor semantics
(serving evicts; a served chunk is gone — the caller re-issues the
query if it needs the data again), `close_stream` releases a stream
ahead of the idle TTL. Backend-agnostic by contract (a stream id may
come from a SQL endpoint's genuinely-streaming source or a file
tool's load-then-serve buffer), so homed at the ingest top level like
endpoints.py — never inside a connector family. `main`'s next_chunk /
request_data_chunk / request_multiple_chunks / get_streaming_status /
clear_streaming_buffer surface is retired into this shape (S9.2).
Neighbors: the guard owns the registry; refusals.py shapes the
expiry/cursor/admission misses.
"""

from __future__ import annotations

from typing import Any

from localdata_mcp.nexus.chokepoint.guard import (
    ChunkAlreadyServedError,
    ChunkNotServableError,
    StreamExpiredError,
)
from localdata_mcp.nexus.contract.spec import Param, TypeShape, tool_spec

from .refusals import stream_refusal
from .runtime import chokepoint


@tool_spec(
    name="fetch_chunk",
    summary=(
        "Retrieve the next servable chunk of a streamed result "
        "(cursor semantics: a served chunk leaves the buffer). Once "
        "the source is exhausted the answer reports the final total "
        "and the stream closes."
    ),
    params=(Param("stream_id", str, "The stream reference a large result returned."),),
    input_shape=TypeShape.NONE,
    output_shape=TypeShape.TABULAR,
    streaming_capable=True,
    domain="ingest",
)
def fetch_chunk(stream_id: str) -> Any:
    try:
        return chokepoint().fetch_next_chunk(stream_id)
    except (
        StreamExpiredError,
        ChunkAlreadyServedError,
        ChunkNotServableError,
    ) as failure:
        raise stream_refusal(failure) from failure


@tool_spec(
    name="close_stream",
    summary=(
        "Release a streamed result ahead of the idle TTL, returning "
        "its buffer memory (and any pinned connection) immediately. "
        "Idempotent."
    ),
    params=(Param("stream_id", str, "The stream reference to release."),),
    input_shape=TypeShape.NONE,
    output_shape=TypeShape.SCALAR,
    domain="ingest",
)
def close_stream(stream_id: str) -> Any:
    chokepoint().close_stream(stream_id)
    return {"stream_id": stream_id, "closed": True}
