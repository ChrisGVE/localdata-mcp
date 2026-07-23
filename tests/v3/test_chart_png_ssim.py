"""tests/v3/test_chart_png_ssim.py — PNG SSIM gate (E12.3, FR-502).

The supplementary raster check (S8 15e — the SVG structural tests carry
the exactness burden). Two layers:

- **Always-on (cross-platform):** re-rendering the same ChartSpec is
  byte-deterministic, so SSIM against itself is ~1.0 — this proves the
  SSIM machinery and the renderer's determinism everywhere.
- **Golden-gated:** each kind's PNG must score ≥ testbench.png_ssim_threshold
  against its committed golden — but ONLY when the render-stack
  fingerprint matches the golden's. A mismatched stack (different
  matplotlib/freetype antialiasing) is SKIPPED with the regeneration
  command, never a false failure: the golden is valid only under the
  stack it was rendered in (the "pinned CI container", S8 15e).
"""

from __future__ import annotations

import pytest

from localdata_mcp.nexus.config.models_testbench import (
    TestbenchConfig as _TestbenchConfig,  # aliased: pytest must not collect it
)
from localdata_mcp.testbench.fixtures.chart_goldens import (
    GOLDEN_CASES,
    golden_frame,
    golden_path,
    recorded_fingerprint,
)
from localdata_mcp.testbench.image_similarity import (
    render_stack_fingerprint,
    ssim,
)
from localdata_mcp.visualize.charts import build_chart_spec
from localdata_mcp.visualize.render import render_spec

_THRESHOLD = _TestbenchConfig().png_ssim_threshold
_REGEN = (
    "python -m localdata_mcp.testbench.fixtures.chart_goldens.regenerate "
    "(run in the pinned CI stack)"
)


def _rendered_png(kind: str) -> bytes:
    spec = build_chart_spec(kind, golden_frame(), dict(GOLDEN_CASES[kind]), None)
    return render_spec(spec, "png")


class TestSsimMachinery:
    """Cross-platform: identity and determinism, no golden needed."""

    def test_identical_render_scores_one(self) -> None:
        png = _rendered_png("scatter_fit")
        assert ssim(png, png) == pytest.approx(1.0, abs=1e-6)

    def test_re_render_is_deterministic(self) -> None:
        assert _rendered_png("histogram") == _rendered_png("histogram")

    def test_threshold_is_the_configured_default(self) -> None:
        assert _THRESHOLD == 0.95


class TestPngGoldens:
    """FR-502: each kind matches its golden, gated on the render stack."""

    @pytest.mark.parametrize("kind", sorted(GOLDEN_CASES))
    def test_kind_matches_golden(self, kind: str) -> None:
        recorded = recorded_fingerprint()
        current = render_stack_fingerprint()
        if recorded is None:
            pytest.skip(f"chart goldens not generated — {_REGEN}")
        if recorded != current:
            pytest.skip(
                f"render stack {current!r} != golden stack {recorded!r}; "
                f"regenerate: {_REGEN}"
            )
        golden = golden_path(kind).read_bytes()
        score = ssim(golden, _rendered_png(kind))
        assert score >= _THRESHOLD, f"{kind} SSIM {score:.4f} < {_THRESHOLD}"
