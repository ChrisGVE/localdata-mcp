"""localdata_mcp/nexus/response/envelope.py — the one envelope (E7.1).

FR-401/403: every tool result is `{inline, data, composition_metadata,
error}` — `inline` the markdown the LLM reads directly, `data` the
typed payload or a `stream_id` reference, `composition_metadata` the
harvested shape carrying provenance and the FR-606-derived next steps,
`error` the NX-3 wire shape EXCLUSIVE with the other three (enforced
at construction, not by convention). The inline/stream cutover is the
S8 rows 23a/23b budget — whichever of the row or byte bound trips
first switches the envelope to a stream reference (O-1/I-4). A
legitimately empty result renders an EXPLICIT zero statement in
`inline`, never an empty table or bare list, so the agent can tell
first-contact/no-data from truncation or defect (O-1's empty-result
semantics). Neighbors: metadata.py supplies the composition shape;
shaping.py applies this through NX-1's generated wrapper (never
opt-in); NX-3's `StructuredError.to_wire()` is the only thing `error`
ever holds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from localdata_mcp.nexus.config.models import ConfigModel
from localdata_mcp.nexus.contract.registry import ToolRegistry
from localdata_mcp.nexus.contract.spec import ToolSpec
from localdata_mcp.nexus.error.model import StructuredError

from .metadata import CompositionMetadata, applicable_next_steps


class EnvelopeContractError(ValueError):
    """The envelope's own FR-403 invariant was violated at
    construction — a caller defect, never a runtime state."""


@dataclass(frozen=True)
class ResponseEnvelope:
    """The four-region response (FR-403): success carries
    inline/data/composition_metadata; error carries error alone."""

    inline: str | None = None
    data: Any = None
    composition_metadata: CompositionMetadata | None = None
    error: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.error is not None:
            if (
                self.inline is not None
                or self.data is not None
                or self.composition_metadata is not None
            ):
                raise EnvelopeContractError(
                    "error is exclusive with the other regions (FR-403)"
                )
        elif self.inline is None:
            raise EnvelopeContractError(
                "a success envelope must carry inline content (FR-401 — "
                "the LLM-readable region is not optional)"
            )

    def to_wire(self) -> dict[str, Any]:
        """The serialized four-key form — every region always named
        (FR-401's named-field assertion), absent regions None."""
        return {
            "inline": self.inline,
            "data": self.data,
            "composition_metadata": (
                self.composition_metadata.to_wire()
                if self.composition_metadata is not None
                else None
            ),
            "error": dict(self.error) if self.error is not None else None,
        }


def error_envelope(structured: StructuredError) -> ResponseEnvelope:
    """The error leg: NX-3's wire shape, nothing else (FR-403)."""
    return ResponseEnvelope(error=structured.to_wire())


class ResponseShaper:
    """NX-7's shaping service: one instance per process, holding the
    NX-2 budgets and the NX-1 registry the derivations read."""

    def __init__(self, config: ConfigModel, registry: ToolRegistry) -> None:
        self._max_rows = config.response.inline_max_rows
        self._max_bytes = config.response.inline_max_bytes
        self._registry = registry

    def shape_envelope(
        self,
        result: Any,
        tool_spec: ToolSpec,
        *,
        stream_id: str | None = None,
    ) -> ResponseEnvelope:
        """`result` as the one envelope (§6.2's `NX7.shape_envelope`).

        Tabular results (a guard `Result`, or any (columns, rows)
        carrier) render an inline markdown table within the budget and
        cut over to a stream reference beyond it; a mapping is
        admission-subject too — it renders `key: value` lines within the
        budget and, over it, a bounded summary with an explicit
        truncation note while the full typed mapping stays in `data` for
        composition (CR-001; a mapping has no row-stream shape to cut
        over to); true scalars render inline directly; empty results
        state their emptiness explicitly.
        """
        metadata = self._metadata_for(tool_spec)
        if hasattr(result, "chunk_id"):
            return self._shape_chunk(result, metadata)
        if hasattr(result, "stream_id") and hasattr(result, "advertised_chunks"):
            return self._stream_reference(result, metadata)
        tabular = _as_tabular(result)
        if tabular is not None:
            return self._shape_tabular(tabular, tool_spec, metadata, stream_id)
        if isinstance(result, (list, tuple)) and len(result) == 0:
            return ResponseEnvelope(
                inline=_zero_statement(tool_spec.name, "items"),
                data=[],
                composition_metadata=metadata,
            )
        if isinstance(result, Mapping):
            return self._shape_mapping(result, metadata)
        return ResponseEnvelope(
            inline=_inline_scalar(result),
            data=result,
            composition_metadata=metadata,
        )

    def _shape_mapping(
        self, result: Mapping[Any, Any], metadata: CompositionMetadata
    ) -> ResponseEnvelope:
        """A mapping under the S8 23a/23b inline budget (CR-001).

        Entries render as `- key: value` lines until either the row
        bound (`inline_max_rows` entries) or the byte bound
        (`inline_max_bytes`) trips, and any single value's rendered text
        is itself capped so one large embedded list cannot blow the
        inline region. Over budget, an explicit note states what was
        omitted; `data` always carries the FULL typed mapping so
        composition stays arithmetically correct (GP4).
        """
        entries = list(result.items())
        if not entries:
            return ResponseEnvelope(
                inline=_inline_scalar(result),
                data=result,
                composition_metadata=metadata,
            )
        lines: list[str] = []
        used = 0
        shown = 0
        for key, value in entries:
            if shown >= self._max_rows:
                break
            line = f"- {key}: {_capped_value(value, self._max_bytes)}"
            weight = len(line.encode("utf-8")) + 1  # + the joining newline
            if shown > 0 and used + weight > self._max_bytes:
                break
            lines.append(line)
            used += weight
            shown += 1
        inline = "\n".join(lines)
        if shown < len(entries):
            omitted = len(entries) - shown
            inline += (
                f"\n… {omitted} more "
                f"{'entry' if omitted == 1 else 'entries'} omitted — the mapping "
                f"exceeds the inline budget (inline_max_rows={self._max_rows}, "
                f"inline_max_bytes={self._max_bytes}); the full typed mapping is "
                "carried in `data` for composition."
            )
        return ResponseEnvelope(
            inline=inline, data=result, composition_metadata=metadata
        )

    def _shape_tabular(
        self,
        tabular: "tuple[tuple[str, ...], tuple[tuple[Any, ...], ...]]",
        tool_spec: ToolSpec,
        metadata: CompositionMetadata,
        stream_id: str | None,
    ) -> ResponseEnvelope:
        columns, rows = tabular
        if not rows:
            return ResponseEnvelope(
                inline=_zero_statement(tool_spec.name, "rows"),
                data={"columns": list(columns), "rows": []},
                composition_metadata=metadata,
            )
        if len(rows) > self._max_rows:
            return self._cutover(len(rows), metadata, stream_id)
        table = _markdown_table(columns, rows)
        if len(table.encode("utf-8")) > self._max_bytes:
            return self._cutover(len(rows), metadata, stream_id)
        return ResponseEnvelope(
            inline=table,
            data={"columns": list(columns), "rows": [list(row) for row in rows]},
            composition_metadata=metadata,
        )

    def _stream_reference(
        self, opened: Any, metadata: CompositionMetadata
    ) -> ResponseEnvelope:
        """A guard `StreamOpened` (I-4): the result exceeded the inline
        budget at the source, so the envelope carries the stream
        reference and the currently-servable count — never a promised
        total (T10)."""
        return ResponseEnvelope(
            inline=(
                "The result exceeds the inline budget — retrieve it chunk "
                f"by chunk: fetch_chunk(stream_id={opened.stream_id!r}); "
                f"{opened.advertised_chunks} chunk(s) currently servable."
            ),
            data={
                "stream_id": opened.stream_id,
                "columns": list(opened.columns),
                "advertised_chunks": opened.advertised_chunks,
            },
            composition_metadata=metadata,
        )

    def _shape_chunk(
        self, served: Any, metadata: CompositionMetadata
    ) -> ResponseEnvelope:
        """A guard `ServedChunk` (I-4/FR-404): the chunk's rows as an
        inline table with the stream's live state appended; once the
        source is exhausted and drained, the final total arrives as
        metadata and the closure of the stream is stated."""
        data = {
            "stream_id": served.stream_id,
            "chunk_id": served.chunk_id,
            "columns": list(served.columns),
            "rows": [list(row) for row in served.rows],
            "advertised_chunks": served.advertised_chunks,
            "exhausted": served.exhausted,
            "total_chunks": served.total_chunks,
        }
        if served.chunk_id is None:
            inline = (
                f"Stream {served.stream_id!r} is exhausted — every chunk "
                f"was served (final total: {served.total_chunks}); the "
                "stream is now closed."
            )
            return ResponseEnvelope(
                inline=inline, data=data, composition_metadata=metadata
            )
        status = (
            f"Chunk {served.chunk_id} of stream {served.stream_id!r}; "
            f"{served.advertised_chunks} more chunk(s) currently servable"
        )
        if served.exhausted and served.total_chunks is not None:
            status += f"; source exhausted — final total {served.total_chunks}"
        table = _markdown_table(served.columns, served.rows)
        return ResponseEnvelope(
            inline=f"{table}\n\n{status}.",
            data=data,
            composition_metadata=metadata,
        )

    def _cutover(
        self,
        row_count: int,
        metadata: CompositionMetadata,
        stream_id: str | None,
    ) -> ResponseEnvelope:
        """The I-4 switch: beyond the inline budget the envelope
        carries a stream reference, never a truncated table."""
        if stream_id is not None:
            return ResponseEnvelope(
                inline=(
                    f"{row_count} rows exceed the inline budget — retrieve "
                    f"them chunk by chunk from stream {stream_id!r}."
                ),
                data={"stream_id": stream_id},
                composition_metadata=metadata,
            )
        return ResponseEnvelope(
            inline=(
                f"{row_count} rows exceed the inline budget and no stream "
                "was opened — re-issue the request through the streaming "
                "path to retrieve them."
            ),
            data={"row_count": row_count},
            composition_metadata=metadata,
        )

    def _metadata_for(self, tool_spec: ToolSpec) -> CompositionMetadata:
        """The composition region, derived — domain and shapes from the
        spec, next steps from the live registry (O-1)."""
        return CompositionMetadata(
            domain=tool_spec.domain or "core",
            analysis_type=tool_spec.name,
            result_type=tool_spec.output_shape.value,
            input_schema={"shape": tool_spec.input_shape.value},
            output_schema={"shape": tool_spec.output_shape.value},
            transformation_summary={"tool": tool_spec.name},
            recommended_next_steps=tuple(
                {"tool": name}
                for name in applicable_next_steps(
                    tool_spec.output_shape, self._registry
                )
            ),
        )


def _as_tabular(
    result: Any,
) -> "tuple[tuple[str, ...], tuple[tuple[Any, ...], ...]] | None":
    """(columns, rows) when `result` carries them — the guard's
    `Result` shape or any object with `columns`/`rows` — else None."""
    columns = getattr(result, "columns", None)
    rows = getattr(result, "rows", None)
    if columns is None or rows is None:
        return None
    return tuple(str(c) for c in columns), tuple(tuple(r) for r in rows)


def _markdown_table(columns: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    """A plain GFM table; pipe characters in cells are escaped so a
    value cannot break the table structure."""

    def cell(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    header = "| " + " | ".join(cell(c) for c in columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = "\n".join("| " + " | ".join(cell(v) for v in row) + " |" for row in rows)
    return f"{header}\n{divider}\n{body}"


def _inline_scalar(result: Any) -> str:
    """The inline region for a true scalar payload, and the explicit
    empty-mapping statement. Non-empty mappings are budget-shaped by
    `_shape_mapping` (CR-001), never rendered here unbounded."""
    if isinstance(result, Mapping):
        if not result:
            return "The result is an empty mapping — no entries (not a failure)."
        return "\n".join(f"- {key}: {value}" for key, value in result.items())
    return str(result)


def _capped_value(value: Any, cap: int) -> str:
    """One mapping value's rendered text, truncated so a single large
    embedded value (a clustering-label list, an LP assignment vector)
    cannot blow the inline budget (CR-001). The full value survives in
    the envelope's `data` region.

    A list/tuple is rendered element by element and stopped as soon as the
    budget is reached, so a million-element vector never materializes its
    full `str()` just to be sliced away (CR-033) — the transient render
    stays bounded by `cap`, not by the value's size."""
    if isinstance(value, (list, tuple)):
        return _capped_sequence(value, cap)
    rendered = str(value)
    if len(rendered) > cap:
        return rendered[:cap] + f"… (value truncated — {len(rendered)} chars total)"
    return rendered


def _capped_sequence(value: "list[Any] | tuple[Any, ...]", cap: int) -> str:
    """Render `[a, b, …]` one element at a time, stopping once the budget
    is reached — the full sequence is never stringified. The element
    count comes from `len()` (O(1) for list/tuple), so the truncation note
    is exact without materializing the tail."""
    parts: list[str] = []
    used = 1  # the opening "["
    for item in value:
        piece = repr(item)
        if used + len(piece) + 2 > cap:  # + ", "
            return (
                "[" + ", ".join(parts) + f", …] ({len(value)} elements, "
                "truncated to fit the inline budget)"
            )
        parts.append(piece)
        used += len(piece) + 2
    return "[" + ", ".join(parts) + "]"


def _zero_statement(tool_name: str, what: str) -> str:
    """O-1's explicit empty statement — never an empty table/list."""
    return (
        f"Zero {what} returned by {tool_name} — the request matched "
        "nothing (a legitimate empty result, not a truncation or failure)."
    )
