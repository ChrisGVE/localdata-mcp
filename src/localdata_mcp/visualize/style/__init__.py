"""localdata_mcp/visualize/style — the chart styling & palette layer (E12.6).

The FR-503 styling surface: `resolve_style` folds the config-backed
`VisualizeDefaults` (via the NX-6 seam) with a per-call palette choice
and style overrides into a `StyleSpec` the renderer consumes; seaborn is
a pure palette provider only (palette.py). Neighbors: render/ applies a
StyleSpec; visualize/tools.py builds one from the seam per call.
"""

from __future__ import annotations

from .palette import QUALITATIVE_PRESETS, Palette, resolve_palette
from .style import STYLE_OVERRIDE_KEYS, StyleSpec, resolve_style

__all__ = [
    "QUALITATIVE_PRESETS",
    "Palette",
    "resolve_palette",
    "STYLE_OVERRIDE_KEYS",
    "StyleSpec",
    "resolve_style",
]
