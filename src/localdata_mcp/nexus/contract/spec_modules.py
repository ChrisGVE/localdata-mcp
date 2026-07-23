"""localdata_mcp/nexus/contract/spec_modules.py — the spec-module roster.

The one declared list of modules whose import registers ToolSpecs into
the default registry (registry.py). Both consumers import through
here — generate.py before rendering artifacts, and the generated
wrapper (server/tools_generated.py) at server startup — so generation
time and serve time see the identical tool population by construction.
Extended as epics land tool modules (E5+ domains replace the E3
walking-skeleton entries).
"""

from __future__ import annotations

import importlib
from typing import Final

SPEC_MODULES: Final[tuple[str, ...]] = (
    "localdata_mcp.server.skeleton_tools",
    "localdata_mcp.ingest.endpoints",
    "localdata_mcp.ingest.connectors.sql.tools",
)


def load_spec_modules() -> None:
    """Import every declared spec module (idempotent — Python caches
    imports, and the registry refuses duplicates if it didn't)."""
    for module_name in SPEC_MODULES:
        importlib.import_module(module_name)
