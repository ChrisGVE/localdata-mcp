"""localdata_mcp/output/tools.py — the export_result ToolSpec (E13.1).

Feature E-1 (FR-901/902/903): one tool that writes any v3 result to a
file. The flow is source → payload → NX-8: resolve the export source
(exactly one of inline `source=`, a `stream_id=` to drain, or the
composition leaf the engine injects), materialize the payload (unwrap a
rendered chart artifact for svg/png, drain a stream to a frame), and
hand it to `export.interface.export_to_file`, which owns the ONE
renderer roster and the three ordered write guards — NX-6 containment
(NFR-108, fail-closed on empty allowed_paths), the NFR-115
explicit-overwrite refusal, and NFR-111 temp-file-rename atomicity.
Consolidates `main`'s export_structured/export_schema/export_graph
(S9.2). input_shape=TABULAR (its composition-leaf contract — the engine
injects a frame); output_shape=NONE — a write is a terminal sink, its
only value the confirmation envelope. Neighbors: nexus/export/ owns the
renderers and guards; explore.addressing owns the pipeline-input seam.
"""

from __future__ import annotations

import base64
from typing import Any, Mapping

from localdata_mcp.explore.addressing import current_pipeline_input
from localdata_mcp.ingest.refusals import (
    export_shape_refusal,
    export_source_refusal,
    stream_refusal,
    unknown_format_refusal,
)
from localdata_mcp.ingest.runtime import chokepoint
from localdata_mcp.nexus.chokepoint.guard import (
    ChunkAlreadyServedError,
    ChunkNotServableError,
    StreamExpiredError,
)
from localdata_mcp.nexus.contract.spec import Param, TypeShape, tool_spec
from localdata_mcp.nexus.export.interface import (
    ExportError,
    export_to_file,
    supported_formats,
)

import pandas as pd


@tool_spec(
    name="export_result",
    summary=(
        "Write a result to a file in any supported format. format is "
        "one of csv, parquet, arrow, json, excel, markdown (tabular "
        "data), schema (a table mapping), graph or tree (a structure "
        "mapping), or svg/png (a rendered chart artifact). path is the "
        "destination file (inside allowed_paths). The source is exactly "
        "one of: source (inline data — a records list, a mapping, or a "
        "rendered chart envelope), stream_id (a buffered result drained "
        "to the file), or — as a composition terminal — the upstream "
        "stage's output, injected automatically when both are omitted. "
        "An existing target is refused unless overwrite=true. Round-trip "
        "fidelity is type-preserving for parquet/arrow, documented-lossy "
        "for markdown."
    ),
    params=(
        Param(
            "format",
            str,
            "Output format: csv, parquet, arrow, json, excel, markdown, "
            "schema, graph, tree, svg, or png.",
        ),
        Param(
            "path",
            str,
            "Destination file path inside allowed_paths.",
        ),
        Param(
            "source",
            object,
            "Inline data to export: a records list, a mapping (schema / "
            "graph / tree / key-value), or a rendered chart artifact "
            "envelope. Omit when exporting a stream_id or a composition "
            "leaf.",
            required=False,
        ),
        Param(
            "stream_id",
            str,
            "A buffered result to drain to the file (cursor semantics — "
            "the stream is consumed). Omit when exporting inline source "
            "or a composition leaf.",
            required=False,
        ),
        Param(
            "overwrite",
            bool,
            "Replace an existing target file (NFR-115: destructive "
            "operations need the explicit disambiguator). Defaults to "
            "refusing an existing target.",
            required=False,
        ),
    ),
    input_shape=TypeShape.TABULAR,
    output_shape=TypeShape.NONE,
    domain="output",
)
def export_result(
    format: str,
    path: str,
    source: Any = None,
    stream_id: str | None = None,
    overwrite: bool = False,
) -> "dict[str, Any]":
    if format not in supported_formats():
        raise unknown_format_refusal(
            f"Unknown export format {format!r} — supported: "
            f"{list(supported_formats())}."
        )
    payload, label = _resolve_source(source, stream_id)
    try:
        real = export_to_file(
            payload,
            format,
            path,
            contain=chokepoint().contain_path,
            overwrite=overwrite,
        )
    except ExportError as failure:
        raise export_shape_refusal(str(failure)) from failure
    return {
        "format": format,
        "path": str(real),
        "source": label,
        "overwrite": overwrite,
        "bytes_written": real.stat().st_size,
    }


def _resolve_source(source: Any, stream_id: str | None) -> "tuple[Any, str]":
    """The export payload and a source label, under the exactly-one
    contract — inline `source`, a `stream_id` to drain, or the injected
    composition leaf. Zero or more than one explicit slot (with no
    upstream stage to fall back to) is the structured refusal."""
    if source is not None and stream_id is not None:
        raise export_source_refusal(
            "Both source= and stream_id= were supplied — pass exactly one "
            f"(got source of type {type(source).__name__} and "
            f"stream_id={stream_id!r})."
        )
    if source is not None:
        return _materialize_inline(source), "inline"
    if stream_id is not None:
        return _drain_stream(stream_id), f"stream:{stream_id}"
    injected = current_pipeline_input()
    if injected is not None:
        frame, upstream_label = injected
        return frame, upstream_label
    raise export_source_refusal(
        "No export source supplied — pass source= (inline data) or "
        "stream_id= (a buffered result), or run export_result as a "
        "composition terminal so the upstream stage's output is injected."
    )


def _materialize_inline(source: Any) -> Any:
    """An inline source as its renderer-ready payload. A rendered chart
    artifact envelope (`{media_type, encoding, content}`, optionally
    wrapped under an `artifact` key by render_chart) is unwrapped to its
    raw bytes/text so the svg/png renderers see the document they
    validate; every other shape passes through, the renderer owning the
    shape check."""
    envelope = source
    if isinstance(source, Mapping) and isinstance(source.get("artifact"), Mapping):
        envelope = source["artifact"]
    if isinstance(envelope, Mapping) and {"encoding", "content"} <= set(envelope):
        content = envelope["content"]
        if envelope.get("encoding") == "base64":
            return base64.b64decode(content)
        return content
    return source


def _drain_stream(stream_id: str) -> pd.DataFrame:
    """A buffered stream drained to a single frame — every servable
    chunk served in order under the cursor semantics (the stream is
    consumed and closed). A stream that is expired, already served, or
    otherwise not servable is the structured stream refusal."""
    guard = chokepoint()
    columns: "tuple[str, ...] | None" = None
    rows: "list[tuple[Any, ...]]" = []
    while True:
        try:
            chunk = guard.fetch_next_chunk(stream_id)
        except (
            StreamExpiredError,
            ChunkAlreadyServedError,
            ChunkNotServableError,
        ) as failure:
            raise stream_refusal(failure) from failure
        if chunk.chunk_id is None:
            break
        if columns is None:
            columns = chunk.columns
        rows.extend(chunk.rows)
    if columns is None:
        raise stream_refusal(
            StreamExpiredError(
                f"stream {stream_id!r} yielded no chunks — it is empty or "
                "already fully drained (cursor semantics)"
            )
        )
    return pd.DataFrame(list(rows), columns=list(columns))
