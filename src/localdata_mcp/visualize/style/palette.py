"""localdata_mcp/visualize/style/palette.py — palette resolution (E12.6).

Seaborn used as a PURE color-list provider and nothing more (the FR-503
styling layer): `sns.color_palette(name)` returns RGB tuples with no
global theme mutation, so the renderer's no-pyplot-state /
SVG-determinism / leak invariants (§6c) survive intact — we never call
`sns.set_theme`. A Palette is the resolved pair the renderer consumes:
the qualitative color cycle (categorical channels) and the sequential
colormap name (continuous channels — heatmap, colour-by-value). A
preset NAME resolves the cycle through seaborn's six qualitative
palettes; a custom value is an explicit, validated color list. An
unknown preset or colormap is an NX-3 structured refusal naming the
registered set — never a silent default. Neighbors: style.py folds a
Palette into a StyleSpec; render/marks.py reads its colors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import matplotlib
import seaborn as sns
from matplotlib.colors import is_color_like, to_hex

from localdata_mcp.ingest.refusals import invalid_source_refusal

# The qualitative presets exposed to the caller — seaborn's six named
# qualitative palettes. One home: the tool summary, the config default,
# and the refusal message all read this set.
QUALITATIVE_PRESETS: tuple[str, ...] = (
    "deep",
    "muted",
    "pastel",
    "bright",
    "dark",
    "colorblind",
)


@dataclass(frozen=True)
class Palette:
    """A resolved palette (§6c): the categorical color cycle as hex
    strings, and the sequential colormap name for continuous channels."""

    colors: tuple[str, ...]
    sequential_cmap: str


def resolve_palette(palette: "str | Sequence[str]", sequential_cmap: str) -> Palette:
    """The Palette for a qualitative choice and a colormap name. The
    qualitative choice is a preset name (resolved through seaborn's pure
    color_palette) or an explicit color list; an unknown name, an empty
    or malformed custom list, or an unknown colormap are each refused."""
    return Palette(
        colors=_qualitative_colors(palette),
        sequential_cmap=_validated_cmap(sequential_cmap),
    )


def _qualitative_colors(palette: "str | Sequence[str]") -> tuple[str, ...]:
    """The categorical cycle as hex: a preset name via seaborn, or a
    validated custom color list."""
    if isinstance(palette, str):
        if palette not in QUALITATIVE_PRESETS:
            raise invalid_source_refusal(
                f"Unknown palette {palette!r} — presets are "
                f"{list(QUALITATIVE_PRESETS)}, or pass a list of colors."
            )
        return tuple(sns.color_palette(palette).as_hex())
    return _validated_custom(palette)


def _validated_custom(colors: "Sequence[str]") -> tuple[str, ...]:
    """A caller-supplied color list, each entry normalised to hex. A
    non-list, an empty list, or an unrecognised color is refused."""
    if (
        isinstance(colors, (str, bytes))
        or not isinstance(colors, Sequence)
        or not colors
    ):
        raise invalid_source_refusal(
            "A custom palette must be a non-empty list of colors "
            "(hex like '#1b9e77' or a named matplotlib color)."
        )
    resolved = []
    for color in colors:
        if not is_color_like(color):
            raise invalid_source_refusal(
                f"Invalid palette color {color!r} — use a hex string "
                "('#1b9e77') or a named matplotlib color."
            )
        resolved.append(to_hex(color))
    return tuple(resolved)


def _validated_cmap(name: str) -> str:
    """The colormap name if matplotlib registers it, else a refusal
    naming that it is the continuous-channel colormap."""
    if name not in matplotlib.colormaps:
        raise invalid_source_refusal(
            f"Unknown sequential colormap {name!r} — use a matplotlib "
            "colormap name (e.g. viridis, magma, cividis)."
        )
    return name
