"""renderers/png.py — PNG artifact validation for file output (E7.4).

The raster half of the visualization output path (FR-501/902): the
Visualize renderer (E13, matplotlib Agg) produces the bytes; NX-8 owns
file output, so this renderer validates that what reaches the write
path IS a PNG — magic-signature check, refusal otherwise — and passes
the payload through unmodified (a raster format has no active-content
surface to sanitize; correctness gates on the SSIM battery, S8 15e).
"""

from __future__ import annotations

from typing import Any

from ..interface import ExportError

FORMAT = "png"

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def render(payload: Any) -> bytes:
    if not isinstance(payload, (bytes, bytearray)):
        raise ExportError(
            f"png payload must be rendered bytes, got {type(payload).__name__}"
        )
    data = bytes(payload)
    if not data.startswith(_PNG_SIGNATURE):
        raise ExportError("payload does not carry the PNG signature — refused")
    return data
