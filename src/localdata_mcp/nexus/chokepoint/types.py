"""localdata_mcp/nexus/chokepoint/types.py — NX-6's value vocabulary.

The plain, capability-narrow shapes the guard entrypoints hand back
(GP3's corollary: data only, nothing live) and the two structured
refusals the seam raises, split out of guard.py so THE chokepoint
class (guard.py) stays under NFR-404's per-class bound while every
surface mixin (surfaces_*.py) shares one vocabulary. Chokepoint-internal
by §6.2: tool modules see these names only re-exported through
`guard`. Nothing here touches a backend — the values are what crosses
the seam OUTWARD, and the render-weight estimate the streaming surface
consults to decide the S8 23a/23b inline/stream cutover.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping

import pandas as pd

from localdata_mcp.nexus.error.model import StructuredError

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
class ProcessDefaults:
    """The S8 process-domain default counts (rows 30/31) as a plain
    value — what `process_defaults()` hands E10's stochastic tools,
    so the ConfigModel stays behind the seam (section 6.2)."""

    bootstrap_resamples: int
    monte_carlo_iterations: int


@dataclass(frozen=True)
class VisualizeDefaults:
    """The S8 `visualize.*` styling defaults as a plain value — what
    `visualize_defaults()` hands E12's render_chart, so the ConfigModel
    stays behind the seam (section 6.2), mirroring ProcessDefaults. The
    render tool folds these with per-call palette/style overrides into a
    StyleSpec."""

    default_palette: str
    default_sequential_cmap: str
    figure_width_inches: float
    figure_height_inches: float
    figure_dpi: int
    grid: bool
    despine: bool
    fit_line_color: str
    edge_color: str

    @classmethod
    def from_config(cls, visualize: Any) -> "VisualizeDefaults":
        """Project a `VisualizeConfig` section onto the plain seam value —
        the one place NX-2's visualize section becomes the value E12
        reads (used by `visualize_defaults()` and by the testbench render
        callers that stand outside the chokepoint)."""
        return cls(
            default_palette=visualize.default_palette,
            default_sequential_cmap=visualize.default_sequential_cmap,
            figure_width_inches=visualize.figure_width_inches,
            figure_height_inches=visualize.figure_height_inches,
            figure_dpi=visualize.figure_dpi,
            grid=visualize.grid,
            despine=visualize.despine,
            fit_line_color=visualize.fit_line_color,
            edge_color=visualize.edge_color,
        )


@dataclass(frozen=True)
class CompositionLimits:
    """The S8 composition bounds (row 14) as a plain value — what
    `composition_limits()` hands the E11 engine, so the ConfigModel
    stays behind the seam (section 6.2), mirroring ProcessDefaults."""

    max_pipeline_length: int


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


def approx_render_bytes(frame: pd.DataFrame) -> int:
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


def result_from_frames(frames: list[pd.DataFrame], category: str) -> Result:
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
