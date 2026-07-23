"""localdata_mcp/nexus/export/renderers/ — the per-format roster (E7.3).

One module per FR-902 format, registered here as declared data — the
consolidation of `main`'s five overlapping export modules into NX-8's
single seam. Adding a format = one module + one roster entry.
"""

from __future__ import annotations

from types import ModuleType
from typing import Mapping

from . import (
    arrow,
    csv,
    excel,
    graph,
    json,
    markdown,
    parquet,
    schema,
    tree,
)

_MODULES: tuple[ModuleType, ...] = (
    csv,
    parquet,
    arrow,
    json,
    excel,
    markdown,
    schema,
    graph,
    tree,
)

RENDERERS: Mapping[str, ModuleType] = {module.FORMAT: module for module in _MODULES}
