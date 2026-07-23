"""chart_goldens/regenerate.py — (re)generate the PNG goldens (E12.3).

Run in the environment whose renders the SSIM gate must hold — the
pinned CI stack (S8 15e):

    python -m localdata_mcp.testbench.fixtures.chart_goldens.regenerate

Renders one PNG per launch kind from the shared golden frame and writes
`<kind>.png` plus `fingerprint.txt` (the matplotlib + freetype stack the
goldens are valid under). The SSIM test asserts against these only when
the running fingerprint matches; a mismatch prints this command.
"""

from __future__ import annotations

from localdata_mcp.nexus.chokepoint.guard import VisualizeDefaults
from localdata_mcp.nexus.config.models import ConfigModel
from localdata_mcp.testbench.image_similarity import render_stack_fingerprint
from localdata_mcp.visualize.charts import build_chart_spec
from localdata_mcp.visualize.render import render_spec
from localdata_mcp.visualize.style import resolve_style

from . import FINGERPRINT_FILE, GOLDEN_CASES, golden_frame, golden_path


def regenerate() -> None:
    frame = golden_frame()
    style = resolve_style(VisualizeDefaults.from_config(ConfigModel().visualize))
    for kind, encoding in GOLDEN_CASES.items():
        spec = build_chart_spec(kind, frame, dict(encoding), None)
        golden_path(kind).write_bytes(render_spec(spec, "png", style))
    FINGERPRINT_FILE.write_text(render_stack_fingerprint() + "\n", encoding="utf-8")


if __name__ == "__main__":
    regenerate()
