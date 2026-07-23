"""localdata_mcp/process/domains/geospatial_analysis/capabilities.py — FR-301.

`check_geospatial_capabilities`'s computation and the family's shared
dependency guard. The probe answers UNCONDITIONALLY (its whole point
is to report whether the extra is installed), listing each backend
library's presence and version. `require_geostack` is what every
other geo tool calls first: a structured refusal naming the missing
libraries when the opt-in extra is absent, so a geo call on a base
install fails legibly instead of with an ImportError traceback.
Neighbors: every sibling module guards through here.
"""

from __future__ import annotations

import importlib
from typing import Any

from ..support import invalid_source_refusal

# The backends the geospatial extra provides (pyproject's `geospatial`).
_GEO_LIBRARIES = ("geopandas", "shapely", "pyproj", "rtree", "skgstat")

# The subset without which no spatial operation can run at all.
_CORE_LIBRARIES = ("geopandas", "shapely", "pyproj")


def _probe(library: str) -> tuple[bool, str | None]:
    try:
        module = importlib.import_module(library)
    except ImportError:
        return False, None
    return True, getattr(module, "__version__", None)


def check_capabilities() -> dict[str, Any]:
    """Which geospatial backends are installed, and what they enable."""
    presence = {library: _probe(library) for library in _GEO_LIBRARIES}
    available = {name: found for name, (found, _v) in presence.items()}
    return {
        "available_libraries": available,
        "versions": {name: version for name, (_f, version) in presence.items()},
        "missing_libraries": [name for name, found in available.items() if not found],
        "has_core_geospatial": all(available[name] for name in _CORE_LIBRARIES),
        "extra_install_hint": "pip install 'localdata-mcp[geospatial]'",
    }


def require_geostack() -> None:
    """Refuse (structured) unless the core geospatial stack is present."""
    missing = [name for name in _CORE_LIBRARIES if not _probe(name)[0]]
    if missing:
        raise invalid_source_refusal(
            f"The geospatial extra is not installed (missing: {missing}). "
            "Install it with: pip install 'localdata-mcp[geospatial]'."
        )
