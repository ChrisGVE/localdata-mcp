"""localdata_mcp/nexus/config/models.py — NX-2's dataclass-per-truth model.

THE one default site (NFR-403): every S8 row whose Home is NX-2 is a
field here (testbench section in models_testbench.py, composed below),
carrying its default, pin-eligibility, and rationale on the declaration
itself (ARCHITECTURE.md section 5). Neighbors: fields.py supplies the
declaration helper; loaders.py/merge.py populate the model from layered
sources; env_derive.py derives `LOCALDATA_<SECTION>_<FIELD>` names from
the field paths declared here.
"""

from __future__ import annotations

from dataclasses import (
    Field,
    dataclass,
    field,
    fields as dataclass_fields,
    is_dataclass,
    replace,
)
from functools import lru_cache
from typing import Any, ClassVar, Iterator, Mapping, get_type_hints

from .endpoints import EndpointDeclaration
from .fields import DERIVED, META_DERIVE, META_PIN, cfg_field
from .models_testbench import TestbenchConfig

GIB = 2**30
MIB = 2**20
KIB = 2**10

# S8 row 13's derivation constant: a conservative 8 KiB/row estimate that
# reproduces main's proven 500k-order literal at the 4 GiB default ceiling
# and scales with an operator-raised one — one formula, one home.
BYTES_PER_ANALYSIS_ROW = 8192


def _derive_max_analysis_rows(model: "ConfigModel") -> int:
    """S8 row 13: `memory_ceiling_bytes // 8192` (524,288 at default)."""
    return model.resources.memory_ceiling_bytes // BYTES_PER_ANALYSIS_ROW


@dataclass(frozen=True)
class ResourcesConfig:
    """`resources.*` — process-wide ceilings (security-classed: resource
    ceilings are pin-eligible per ARCHITECTURE.md section 5 / NFR-105)."""

    SECURITY_CLASSED: ClassVar[bool] = True

    memory_ceiling_bytes: int = cfg_field(4 * GIB, doc="S8 row 1 (LOCKED)")
    query_timeout_seconds: int = cfg_field(300, doc="S8 row 2 (LOCKED)")
    max_connections_per_endpoint: int = cfg_field(8, doc="S8 row 3 (LOCKED)")
    # 2x the memory ceiling: one full over-ceiling working set plus
    # overhead fits; larger invites unbounded-disk regressions (NFR-203).
    max_spill_bytes: int = cfg_field(8 * GIB, doc="S8 row 5")
    # Absolute free-disk floor under any spill/export write (fail-safe).
    min_free_disk_bytes: int = cfg_field(2 * GIB, doc="S8 row 6")


@dataclass(frozen=True)
class QueryConfig:
    """`query.*` — streaming and admission bounds."""

    SECURITY_CLASSED: ClassVar[bool] = False

    chunk_buffer_max_chunks: int = cfg_field(4, doc="S8 row 8 (K)")
    # 1/16 of the memory ceiling (sizing rationale, not a derivation —
    # per-registry fairness; the aggregate gate owns ceiling protection).
    chunk_buffer_max_bytes: int = cfg_field(256 * MIB, doc="S8 row 9 (B)")
    # Carries main's proven buffer_timeout_seconds=600.
    stream_idle_ttl_seconds: int = cfg_field(600, doc="S8 row 10")
    default_chunk_size: int = cfg_field(100, doc="S8 row 12")
    # Derived, never a literal: the upper cap for analytical admission
    # (admission itself is additionally checked against live residency).
    max_analysis_rows: int = cfg_field(
        DERIVED, derive=_derive_max_analysis_rows, doc="S8 row 13 (derived)"
    )
    # Half the row-3 pool: paused streams pin connections, so this cap
    # structurally guarantees interactive headroom (fail-safe, GP3).
    max_concurrent_streams_per_endpoint: int = cfg_field(4, doc="S8 row 24")


@dataclass(frozen=True)
class SecurityConfig:
    """`security.*` — security-classed: every field pin-eligible
    (fail-closed default per ARCHITECTURE.md section 5)."""

    SECURITY_CLASSED: ClassVar[bool] = True

    # Cache stores classification only — a few KB/entry, bounded.
    validation_cache_entries: int = cfg_field(1024, doc="S8 row 11")
    # Empty = fail-closed: no filesystem reads or writes at all (NFR-108).
    # Entries are introducible only at operator-trust layers; a project
    # layer may narrow, never mint (the introduction rule, section 5).
    allowed_paths: tuple[str, ...] = cfg_field(
        (), doc="S8 row 19", introduction_gated=True
    )
    # ARCHITECTURE section 5 (E5.3): the operator rw grant for ephemeral
    # file connections — read-only is the unconditional default, and
    # only an operator-trust layer may key a canonical path or contained
    # prefix read-write (introduction-gated like allowed_paths: a
    # project-layer file may narrow, never mint).
    ephemeral_write_paths: tuple[str, ...] = cfg_field(
        (), doc="ARCHITECTURE.md section 5 / E5.3", introduction_gated=True
    )


@dataclass(frozen=True)
class CompositionConfig:
    """`composition.*` — pipeline bounds."""

    SECURITY_CLASSED: ClassVar[bool] = False

    # Largest length keeping exhaustive AB/BA coverage tractable; an
    # operator raise is an accepted departure from the battery envelope.
    max_pipeline_length: int = cfg_field(4, doc="S8 row 14 (CONFIRMED)")


@dataclass(frozen=True)
class ResponseConfig:
    """`response.*` — the inline/stream cutover (I-4/O-1)."""

    SECURITY_CLASSED: ClassVar[bool] = False

    # Matches row 12's default_chunk_size by sizing rationale only — the
    # field is independently tunable, not derived.
    inline_max_rows: int = cfg_field(100, doc="S8 row 23a")
    # Byte-side guard for wide/text-heavy rows; whichever of the pair
    # trips first switches the envelope to stream_id.
    inline_max_bytes: int = cfg_field(256 * KIB, doc="S8 row 23b")


@dataclass(frozen=True)
class ProcessConfig:
    """`process.*` — statistical-domain defaults."""

    SECURITY_CLASSED: ClassVar[bool] = False

    # Literature-standard floor for percentile bootstrap CIs
    # (Efron & Tibshirani); caller-overridable per call, seed-pinnable.
    bootstrap_default_resamples: int = cfg_field(1000, doc="S8 row 30")
    # ~1% standard error on tail-probability estimates.
    monte_carlo_default_iterations: int = cfg_field(10000, doc="S8 row 31")
    # Standard ill-conditioning heuristic (Golub & Van Loan): 1e10 leaves
    # ~6 trustworthy digits in double precision.
    sentinel_max_condition_number: float = cfg_field(1e10, doc="S8 row 32")


@dataclass(frozen=True)
class VisualizeConfig:
    """`visualize.*` — chart styling defaults (the one home for the
    palette, figure geometry, and the Tufte frame treatment; the FR-503
    styling layer). Not security-classed: a palette or figure size is a
    presentation choice, never a resource or trust boundary. Each field
    is a per-call override target on `render_chart` — config supplies
    the default, the call may override (progressive disclosure)."""

    SECURITY_CLASSED: ClassVar[bool] = False

    # Qualitative (categorical) cycle: seaborn's colorblind palette is the
    # accessible, honest default (First Principle: clarity; Tufte spirit).
    default_palette: str = cfg_field("colorblind", doc="qualitative palette preset")
    # Continuous channels (heatmap, colour-by-value): perceptually uniform,
    # colourblind-safe — the launch heatmap already used it.
    default_sequential_cmap: str = cfg_field("viridis", doc="sequential colormap")
    # matplotlib's own default figure geometry — a neutral 4:3-ish canvas.
    figure_width_inches: float = cfg_field(6.4, doc="figure width (inches)")
    figure_height_inches: float = cfg_field(4.8, doc="figure height (inches)")
    figure_dpi: int = cfg_field(100, doc="raster (PNG) resolution")
    # A light y-grid behind the marks aids reading without competing with
    # data ink; top/right spines are non-data ink and dropped (Tufte).
    grid: bool = cfg_field(True, doc="light y-grid behind the marks")
    despine: bool = cfg_field(True, doc="drop top/right frame spines (Tufte)")
    # Secondary/annotation ink, de-emphasised from the primary palette:
    # the regression fit accent and the network-edge ink.
    fit_line_color: str = cfg_field("#d55e00", doc="regression fit accent")
    edge_color: str = cfg_field("#8c8c8c", doc="network edge ink")


@dataclass(frozen=True)
class ConfigModel:
    """The one config model — every operator-tunable truth, one home."""

    resources: ResourcesConfig = field(default_factory=ResourcesConfig)
    query: QueryConfig = field(default_factory=QueryConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    composition: CompositionConfig = field(default_factory=CompositionConfig)
    response: ResponseConfig = field(default_factory=ResponseConfig)
    process: ProcessConfig = field(default_factory=ProcessConfig)
    visualize: VisualizeConfig = field(default_factory=VisualizeConfig)
    testbench: TestbenchConfig = field(default_factory=TestbenchConfig)
    # Endpoint entities (E1.4) — declarations by name, not a scalar
    # section: no env encoding, introduction gated to operator layers
    # and pinned per name by merge.py (ARCHITECTURE.md section 5).
    endpoints: Mapping[str, EndpointDeclaration] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Resolve DERIVED fields from their declared derivations."""
        for section_name in section_names():
            section = getattr(self, section_name)
            resolved = _resolve_derived(section, self)
            if resolved is not section:
                object.__setattr__(self, section_name, resolved)


def _resolve_derived(section: Any, model: ConfigModel) -> Any:
    """Fill each still-DERIVED field of `section` from its derivation."""
    changes: dict[str, Any] = {}
    for fld in dataclass_fields(section):
        derive = fld.metadata.get(META_DERIVE)
        if derive is not None and getattr(section, fld.name) is DERIVED:
            changes[fld.name] = derive(model)
    return replace(section, **changes) if changes else section


def section_names() -> tuple[str, ...]:
    """The scalar-section order as declared on ConfigModel (the
    `endpoints` entity mapping is not a scalar section)."""
    return tuple(
        f.name for f in dataclass_fields(ConfigModel) if is_dataclass(f.default_factory)
    )


def section_class(section_name: str) -> type:
    """The dataclass declaring `section_name`'s fields."""
    for fld in dataclass_fields(ConfigModel):
        if fld.name == section_name:
            return fld.default_factory  # type: ignore[return-value]
    raise KeyError(section_name)


def iter_config_fields() -> Iterator[tuple[str, Field[Any]]]:
    """Yield every (section_name, field) pair of the one model."""
    for section_name in section_names():
        for fld in dataclass_fields(section_class(section_name)):
            yield section_name, fld


@lru_cache(maxsize=None)
def _section_hints(section_cls: type) -> Mapping[str, Any]:
    return get_type_hints(section_cls)


def field_type(section_name: str, field_name: str) -> Any:
    """The resolved annotation of one model field — the single place
    annotation strings become runtime types (env_derive.py and merge.py
    both coerce through this)."""
    return _section_hints(section_class(section_name))[field_name]


def is_pin_eligible(section_name: str, fld: Field[Any]) -> bool:
    """Pin-eligibility per ARCHITECTURE.md section 5: the field's own
    metadata wins; an absent flag defaults to the section's security
    classification (fail-closed on security-classed sections)."""
    explicit = fld.metadata.get(META_PIN)
    if explicit is not None:
        return bool(explicit)
    return bool(getattr(section_class(section_name), "SECURITY_CLASSED", False))
