"""localdata_mcp/visualize/style/style.py — the resolved StyleSpec (E12.6).

The one styling value the renderer consumes (the FR-503 styling layer):
figure geometry, the Tufte frame treatment (top/right spines dropped, a
light y-grid behind the marks), and the resolved Palette. `resolve_style`
folds the config-backed `VisualizeDefaults` (reached through the NX-6
seam, never NX-2 directly) with a per-call `palette` choice and a
`style` override map — the override wins, an unknown key or an
out-of-range value is an NX-3 refusal (progressive disclosure: simple
by default, every knob explicit when reached for). `rc_overrides()` are
applied by the renderer inside `matplotlib.rc_context` — never seaborn's
global `set_theme` — so no global pyplot state leaks and SVG stays
byte-reproducible (§6c). Neighbors: guard.py supplies VisualizeDefaults;
palette.py resolves the Palette; render/backend.py applies the spec.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from cycler import cycler
from matplotlib.colors import is_color_like, to_hex

from localdata_mcp.ingest.refusals import invalid_source_refusal
from localdata_mcp.nexus.chokepoint.guard import VisualizeDefaults

from .palette import Palette, resolve_palette

# The per-call `style` override keys — one home; an unknown key is
# refused naming this set (a typo must never silently do nothing).
STYLE_OVERRIDE_KEYS: frozenset[str] = frozenset(
    {
        "figure_width_inches",
        "figure_height_inches",
        "dpi",
        "grid",
        "despine",
        "sequential_cmap",
        "fit_color",
        "edge_color",
    }
)


@dataclass(frozen=True)
class StyleSpec:
    """The resolved styling for one render (§6c): figure geometry, the
    Tufte frame flags, the palette, and the secondary-ink colors."""

    figure_width_inches: float
    figure_height_inches: float
    dpi: int
    grid: bool
    despine: bool
    palette: Palette
    fit_color: str
    edge_color: str

    @property
    def figsize(self) -> tuple[float, float]:
        return (self.figure_width_inches, self.figure_height_inches)

    def rc_overrides(self) -> dict[str, Any]:
        """The rcParams the renderer applies in `rc_context`: the palette
        as the axes color cycle, and the grid treatment. Scoped, never
        global — the leak / determinism invariants depend on it (§6c)."""
        return {
            "axes.prop_cycle": cycler(color=list(self.palette.colors)),
            "axes.grid": self.grid,
            "axes.grid.axis": "y",
            "axes.axisbelow": True,
        }


def resolve_style(
    defaults: VisualizeDefaults,
    palette: "str | list[str] | None" = None,
    style: Mapping[str, Any] | None = None,
) -> StyleSpec:
    """A StyleSpec from the config defaults plus per-call overrides. The
    `palette` (a preset name or a color list) and each `style` key
    override the corresponding default; an unknown `style` key, or an
    out-of-range geometry value, is refused."""
    overrides = _validated_overrides(style)
    sequential_cmap = overrides.get("sequential_cmap", defaults.default_sequential_cmap)
    palette_choice = palette if palette is not None else defaults.default_palette
    return StyleSpec(
        figure_width_inches=_positive_dimension(
            overrides.get("figure_width_inches", defaults.figure_width_inches),
            "figure_width_inches",
        ),
        figure_height_inches=_positive_dimension(
            overrides.get("figure_height_inches", defaults.figure_height_inches),
            "figure_height_inches",
        ),
        dpi=_positive_dpi(overrides.get("dpi", defaults.figure_dpi)),
        grid=bool(overrides.get("grid", defaults.grid)),
        despine=bool(overrides.get("despine", defaults.despine)),
        palette=resolve_palette(palette_choice, sequential_cmap),
        fit_color=_validated_color(
            overrides.get("fit_color", defaults.fit_line_color), "fit_color"
        ),
        edge_color=_validated_color(
            overrides.get("edge_color", defaults.edge_color), "edge_color"
        ),
    )


def _validated_overrides(style: Mapping[str, Any] | None) -> dict[str, Any]:
    """The override map with every key checked against the known set —
    an unknown key is a refusal, not a silent no-op."""
    overrides = dict(style or {})
    unknown = sorted(set(overrides) - STYLE_OVERRIDE_KEYS)
    if unknown:
        raise invalid_source_refusal(
            f"Unknown style key(s) {unknown} — supported keys are "
            f"{sorted(STYLE_OVERRIDE_KEYS)}."
        )
    return overrides


def _positive_dimension(value: Any, name: str) -> float:
    """A figure dimension in inches — a positive real, else a refusal."""
    try:
        dimension = float(value)
    except (TypeError, ValueError):
        raise invalid_source_refusal(f"{name} must be a positive number.")
    if not dimension > 0.0:
        raise invalid_source_refusal(f"{name} must be a positive number.")
    return dimension


def _positive_dpi(value: Any) -> int:
    """A raster resolution — a positive integer, else a refusal."""
    try:
        dpi = int(value)
    except (TypeError, ValueError):
        raise invalid_source_refusal("dpi must be a positive integer.")
    if dpi < 1:
        raise invalid_source_refusal("dpi must be a positive integer.")
    return dpi


def _validated_color(value: Any, name: str) -> str:
    """A single color normalised to hex, else a refusal naming the key."""
    if not is_color_like(value):
        raise invalid_source_refusal(
            f"Invalid {name} {value!r} — use a hex string ('#1b9e77') "
            "or a named matplotlib color."
        )
    return to_hex(value)
