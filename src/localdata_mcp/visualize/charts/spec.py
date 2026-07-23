"""localdata_mcp/visualize/charts/spec.py — CHART_SPEC construction (E12.1).

The one dispatcher (§6c): the six launch chart kinds registered under
their names (FR-503 — one kind per Process/Explore domain family), and
`build_chart_spec`, which resolves a requested kind, runs its pure
extractor over the addressed frame and encoding, and returns the
declarative `ChartSpec` the renderer consumes. An unknown kind is a
structured refusal naming the registered set — never a silent default.
The registry is the SSOT for "which kinds exist"; the pipeline battery
and the coverage tests read it, never a restated list. Neighbors:
distribution/relational/spatial.py supply the ChartKinds; tools.py's
render_chart calls build_chart_spec then hands the spec to render/.
"""

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from localdata_mcp.ingest.refusals import invalid_source_refusal

from .base import ChartKind, ChartSpec
from .distribution import HEATMAP, HISTOGRAM
from .relational import LINE_TIMESERIES, SCATTER_FIT
from .spatial import GEO_MAP, NETWORK_LAYOUT

# The launch kind registry (FR-503) — one home, keyed by name. Adding a
# kind is a registration here plus its module; the coverage tests and
# the pipeline battery derive the kind set from this mapping.
CHART_KINDS: Mapping[str, ChartKind] = {
    kind.name: kind
    for kind in (
        HISTOGRAM,
        HEATMAP,
        SCATTER_FIT,
        LINE_TIMESERIES,
        GEO_MAP,
        NETWORK_LAYOUT,
    )
}


def build_chart_spec(
    kind: str,
    frame: pd.DataFrame,
    encoding: Mapping[str, Any] | None,
    title: str | None,
) -> ChartSpec:
    """The declarative ChartSpec for a requested kind over addressed
    data — an unknown kind is refused naming the registered set."""
    handler = CHART_KINDS.get(kind)
    if handler is None:
        raise invalid_source_refusal(
            f"Unknown chart kind {kind!r} — registered kinds are {sorted(CHART_KINDS)}."
        )
    resolved = dict(encoding or {})
    data = handler.extract(frame, resolved)
    return ChartSpec(
        kind=handler.name,
        mark=handler.mark,
        title=title,
        encoding=resolved,
        data=data,
    )
