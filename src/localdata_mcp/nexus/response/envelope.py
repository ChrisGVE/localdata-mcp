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
        cut over to a stream reference beyond it; scalars and mappings
        render inline directly; empty results state their emptiness
        explicitly.
        """
        metadata = self._metadata_for(tool_spec)
        tabular = _as_tabular(result)
        if tabular is not None:
            return self._shape_tabular(tabular, tool_spec, metadata, stream_id)
        if isinstance(result, (list, tuple)) and len(result) == 0:
            return ResponseEnvelope(
                inline=_zero_statement(tool_spec.name, "items"),
                data=[],
                composition_metadata=metadata,
            )
        return ResponseEnvelope(
            inline=_inline_scalar(result),
            data=result,
            composition_metadata=metadata,
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
    """The inline region for non-tabular payloads: a readable one-liner
    (mappings render as `key: value` lines)."""
    if isinstance(result, Mapping):
        if not result:
            return "The result is an empty mapping — no entries (not a failure)."
        return "\n".join(f"- {key}: {value}" for key, value in result.items())
    return str(result)


def _zero_statement(tool_name: str, what: str) -> str:
    """O-1's explicit empty statement — never an empty table/list."""
    return (
        f"Zero {what} returned by {tool_name} — the request matched "
        "nothing (a legitimate empty result, not a truncation or failure)."
    )
