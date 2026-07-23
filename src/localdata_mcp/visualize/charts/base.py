"""localdata_mcp/visualize/charts/base.py — the CHART_SPEC vocabulary (E12.1).

The two data types the chart layer produces and the renderer consumes
(§6c, §7): a `ChartKind` is one registered chart type — its name, the
mark-element the renderer draws (the structural-test anchor, FR-501's
"presence/count per kind"), and the pure function that turns an
addressed frame plus an encoding into extracted `data`. A `ChartSpec`
is the declarative result — "what chart" independent of "how drawn"
(the reversal boundary a renderer swap must not discard) — carrying
kind, title, the resolved encoding, and the extracted data arrays.
The renderer (visualize/render/) consumes a ChartSpec and never reads
the source frame again. Neighbors: distribution/relational/spatial.py
build the ChartKinds; spec.py registers them and dispatches;
render/backend.py draws a ChartSpec.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import pandas as pd

# A kind's extractor: (frame, encoding) -> the `data` payload the
# renderer plots. Pure — no I/O, no matplotlib, deterministic.
Extractor = Callable[[pd.DataFrame, Mapping[str, Any]], "dict[str, Any]"]


@dataclass(frozen=True)
class ChartKind:
    """One registered chart type (§6c). `mark` is the per-kind data-mark
    token: the renderer tags its data artist with SVG `id="localdata-<mark>"`
    (matplotlib draws every data mark as `<path>`/`<use>`, so a stable
    gid — not the element name — is the structural-test anchor for
    FR-501's presence/count per kind)."""

    name: str
    mark: str
    extract: Extractor


@dataclass(frozen=True)
class ChartSpec:
    """The declarative chart specification (§7): what to draw, resolved
    from the source, before any pixels exist."""

    kind: str
    mark: str
    title: str | None
    encoding: Mapping[str, Any]
    data: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """The plain-dict form the renderer and the composition
        metadata carry (FR-504)."""
        return {
            "kind": self.kind,
            "mark": self.mark,
            "title": self.title,
            "encoding": dict(self.encoding),
            "data": dict(self.data),
        }
