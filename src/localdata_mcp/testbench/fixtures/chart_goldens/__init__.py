"""chart_goldens — pinned PNG goldens for the FR-502 SSIM gate (E12.3).

One deterministic fixture frame and the (kind, encoding) case per launch
chart kind, shared by the regeneration script and the SSIM test so both
render the identical data. The goldens (`<kind>.png`) and the
render-stack fingerprint (`fingerprint.txt`) live beside this module;
the SSIM test asserts a rendered PNG against its golden ONLY when the
current fingerprint matches the recorded one (image_similarity.py) —
otherwise the antialiasing stack differs and the goldens must be
regenerated in that environment via regenerate.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pandas as pd

GOLDENS_DIR = Path(__file__).resolve().parent
FINGERPRINT_FILE = GOLDENS_DIR / "fingerprint.txt"

# One case per launch kind — the same six the registry declares. Kept
# small and deterministic so the golden is stable and cheap.
GOLDEN_CASES: Mapping[str, Mapping[str, Any]] = {
    "histogram": {"value": "x"},
    "heatmap": {"columns": ["x", "y", "z"]},
    "scatter_fit": {"x": "x", "y": "y"},
    "line_timeseries": {"x": "x", "y": "y"},
    "geo_map": {"lon": "lon", "lat": "lat", "value": "z"},
    "network_layout": {"source": "src", "target": "dst"},
}


def golden_frame() -> pd.DataFrame:
    """The one fixture every golden renders — fixed values, no
    randomness, so regeneration is byte-reproducible up to the render
    stack."""
    return pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            "y": [2.1, 3.9, 6.2, 7.8, 10.1, 12.0, 13.9, 16.2],
            "z": [5.0, 3.0, 1.0, 4.0, 2.0, 6.0, 3.5, 4.5],
            "lon": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0],
            "lat": [40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0],
            "src": ["a", "b", "c", "a", "b", "c", "d", "a"],
            "dst": ["b", "c", "a", "c", "d", "b", "a", "d"],
        }
    )


def golden_path(kind: str) -> Path:
    """The committed golden PNG for a kind."""
    return GOLDENS_DIR / f"{kind}.png"


def recorded_fingerprint() -> str | None:
    """The render-stack the goldens were made under, or None if the
    goldens have not been generated yet."""
    if not FINGERPRINT_FILE.exists():
        return None
    return FINGERPRINT_FILE.read_text(encoding="utf-8").strip()
