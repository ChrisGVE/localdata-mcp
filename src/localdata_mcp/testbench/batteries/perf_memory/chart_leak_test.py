"""testbench/batteries/perf_memory/chart_leak_test.py — NFR-204 (E12.4).

The unclosed-Figure leak guard (S8 row 22): rendering a chart
`testbench.leak_loop_iterations` times must not drift resident memory by
more than `testbench.leak_loop_max_drift_mib` — the regression assertion
for the class the OO renderer defends against by disposing every Figure
(backend.py, never pyplot global state). A warm-up absorbs the one-time
matplotlib import, font-cache build, and allocator arena growth so the
measured drift is the per-render residue, not start-up cost. Both output
formats are exercised (SVG and PNG take different canvas paths). The
loop count and bound are the LOCKED S8 values, read from the ConfigModel
(NFR-403), never restated here. Neighbors: discovery_latency_test.py is
the other perf-tier assertion; backend.py is the code under guard.
"""

from __future__ import annotations

import gc

import psutil

from localdata_mcp.nexus.chokepoint.guard import VisualizeDefaults
from localdata_mcp.nexus.config.models import ConfigModel
from localdata_mcp.testbench.fixtures.chart_goldens import golden_frame
from localdata_mcp.visualize.charts import build_chart_spec
from localdata_mcp.visualize.render import render_spec
from localdata_mcp.visualize.style import resolve_style

_WARMUP = 20  # untimed renders that absorb import/font-cache/arena cost
_BYTES_PER_MIB = 1024 * 1024


def _resident_mib() -> float:
    return psutil.Process().memory_info().rss / _BYTES_PER_MIB


def test_repeated_render_does_not_leak_resident_memory() -> None:
    config = ConfigModel().testbench
    iterations = config.leak_loop_iterations
    max_drift = config.leak_loop_max_drift_mib

    frame = golden_frame()
    style = resolve_style(VisualizeDefaults.from_config(ConfigModel().visualize))
    # Two specs, two canvas paths — a scatter (Line2D + PathCollection)
    # and a heatmap (QuadMesh), rendered SVG and PNG in turn.
    specs = [
        build_chart_spec("scatter_fit", frame, {"x": "x", "y": "y"}, "leak"),
        build_chart_spec("heatmap", frame, {"columns": ["x", "y", "z"]}, "leak"),
    ]
    formats = ("svg", "png")

    def render_once(index: int) -> None:
        spec = specs[index % len(specs)]
        image_format = formats[index % len(formats)]
        render_spec(spec, image_format, style)

    for index in range(_WARMUP):
        render_once(index)
    gc.collect()
    baseline = _resident_mib()

    for index in range(iterations):
        render_once(index)
    gc.collect()
    drift = _resident_mib() - baseline

    assert drift < max_drift, (
        f"resident memory drifted {drift:.1f} MiB over {iterations} renders "
        f"(bound {max_drift} MiB) — a Figure is leaking (NFR-204)"
    )
