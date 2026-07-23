"""Declarative chart-spec construction (E12.1, FR-503).

The chart layer's public surface: the CHART_SPEC data types, the
kind registry, and the `build_chart_spec` dispatcher. The renderer
(visualize/render/) and the render_chart tool import from here.
"""

from __future__ import annotations

from .base import ChartKind, ChartSpec
from .spec import CHART_KINDS, build_chart_spec

__all__ = [
    "ChartKind",
    "ChartSpec",
    "CHART_KINDS",
    "build_chart_spec",
]
