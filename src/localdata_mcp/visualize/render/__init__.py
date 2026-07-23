"""matplotlib OO-API rendering — a ChartSpec to raw SVG/PNG bytes (E12.2).

Renders bytes only; NX-8 (nexus/export/renderers/) owns SVG
sanitization and all file output (§7).
"""

from __future__ import annotations

from .backend import render_spec

__all__ = ["render_spec"]
