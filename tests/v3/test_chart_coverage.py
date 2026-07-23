"""tests/v3/test_chart_coverage.py — render_chart through the seam (E12.3).

FR-503 per-domain chart coverage and FR-501 inertness, at L3 — the real
tool through the real MCP seam (fastmcp.Client), one kind per FR-503
domain family. Two checks carry the correctness burden here (S8 15e —
the SVG structural checks are the exact gate; PNG SSIM is supplementary,
test_chart_png_ssim.py):

- **Structural:** the sanitized SVG carries the kind's gid-tagged data
  marks (presence per kind) and the title text.
- **Inertness (FR-501):** a chart whose title and data-derived labels
  carry hostile markup (`<script>`, `onload=`, an external `href`)
  emits an SVG with zero deny-set elements/attributes — the payload is
  escaped or dropped, never live.
"""

from __future__ import annotations

import json
import os
import xml.etree.ElementTree as ElementTree
from pathlib import Path
from typing import Any, Iterator

import anyio
import pandas as pd
import pytest
from fastmcp import Client

import localdata_mcp.ingest.runtime as runtime
from localdata_mcp.nexus.chokepoint.guard import Chokepoint
from localdata_mcp.nexus.config.models import ConfigModel, SecurityConfig
from localdata_mcp.nexus.contract.registry import default_registry
from localdata_mcp.nexus.response.shaping import configure_shaping
from localdata_mcp.server.mcp_app import app

_KIND_GIDS = {
    "histogram": ["localdata-bars"],
    "heatmap": ["localdata-cells"],
    "scatter_fit": ["localdata-points", "localdata-fit"],
    "line_timeseries": ["localdata-line"],
    "geo_map": ["localdata-points"],
    "network_layout": ["localdata-edges", "localdata-points"],
}
_KIND_ENCODINGS = {
    "histogram": {"value": "x"},
    "heatmap": {"columns": ["x", "y", "z"]},
    "scatter_fit": {"x": "x", "y": "y"},
    "line_timeseries": {"x": "x", "y": "y"},
    "geo_map": {"lon": "lon", "lat": "lat"},
    "network_layout": {"source": "src", "target": "dst"},
}


@pytest.fixture()
def bench(tmp_path: Path) -> Iterator[Path]:
    config = ConfigModel(security=SecurityConfig(allowed_paths=(str(tmp_path),)))
    guard = Chokepoint.boot(config, environ=dict(os.environ))
    configure_shaping(config, default_registry())
    runtime.configure_ingest(guard)
    yield tmp_path
    runtime._CHOKEPOINT = None
    configure_shaping(ConfigModel(), default_registry())
    guard.shutdown()


def _fixture_csv(tmp_path: Path) -> str:
    target = tmp_path / "chart.csv"
    pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
            "y": [2.1, 3.9, 6.2, 7.8, 10.1],
            "z": [5.0, 3.0, 1.0, 4.0, 2.0],
            "lon": [10.0, 11.0, 12.0, 13.0, 14.0],
            "lat": [40.0, 41.0, 42.0, 43.0, 44.0],
            "src": ["a", "b", "c", "a", "b"],
            "dst": ["b", "c", "a", "c", "a"],
        }
    ).to_csv(target, index=False)
    return str(target)


def _render(arguments: dict[str, Any]) -> dict[str, Any]:
    async def session() -> dict[str, Any]:
        async with Client(app) as client:
            result = await client.call_tool("render_chart", arguments)
            assert not result.is_error, result
            if isinstance(result.structured_content, dict) and (
                "inline" in result.structured_content
            ):
                return result.structured_content
            payload = json.loads(result.content[0].text)
            assert isinstance(payload, dict)
            return payload

    return anyio.run(session)


def _artifact_svg(envelope: dict[str, Any]) -> str:
    assert envelope["error"] is None, envelope["error"]
    artifact = envelope["data"]["artifact"]
    assert artifact["media_type"] == "image/svg+xml"
    return artifact["content"]


class TestPerDomainCoverage:
    """FR-503: each FR-503 chart family renders through the seam."""

    @pytest.mark.parametrize("kind", sorted(_KIND_ENCODINGS))
    def test_kind_renders_svg_with_marks(self, bench: Path, kind: str) -> None:
        path = _fixture_csv(bench)
        envelope = _render(
            {
                "kind": kind,
                "path": path,
                "encoding": _KIND_ENCODINGS[kind],
                "title": f"chart-{kind}",
            }
        )
        assert envelope["data"]["kind"] == kind
        svg = _artifact_svg(envelope)
        for gid in _KIND_GIDS[kind]:
            assert f'id="{gid}"' in svg, f"{kind} missing {gid}"
        assert f"chart-{kind}" in svg  # title text present

    @pytest.mark.parametrize("kind", sorted(_KIND_ENCODINGS))
    def test_kind_renders_png(self, bench: Path, kind: str) -> None:
        path = _fixture_csv(bench)
        envelope = _render(
            {
                "kind": kind,
                "path": path,
                "encoding": _KIND_ENCODINGS[kind],
                "format": "png",
            }
        )
        artifact = envelope["data"]["artifact"]
        assert artifact["media_type"] == "image/png"
        assert artifact["encoding"] == "base64"
        assert len(artifact["content"]) > 100


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _inertness_violations(svg: str) -> list[str]:
    """The FR-501 deny-set, checked STRUCTURALLY over the parsed DOM (a
    substring scan cannot tell a live `onload=` attribute from the same
    letters sitting harmlessly inside escaped `<text>` content): any
    `<script>`/`<image>`/`<foreignObject>` element, any `on*`
    event-handler attribute, or any non-local href reference."""
    root = ElementTree.fromstring(svg)
    violations: list[str] = []
    for element in root.iter():
        name = _local(element.tag)
        if name in ("script", "image", "foreignObject"):
            violations.append(f"element <{name}>")
        for attr, value in element.attrib.items():
            attr_name = _local(attr).lower()
            if attr_name.startswith("on"):
                violations.append(f"attribute {attr_name}")
            if attr_name in ("href", "xlink:href") and not value.startswith("#"):
                violations.append(f"non-local href {value!r}")
    return violations


class TestInertness:
    """FR-501: hostile title/label markup never survives as live SVG —
    verified over the parsed DOM, not by substring."""

    def _hostile_csv(self, tmp_path: Path) -> str:
        target = tmp_path / "hostile.csv"
        pd.DataFrame(
            {
                "<script>alert(1)</script>": [1.0, 2.0, 3.0, 4.0, 5.0],
                "y": [2.0, 4.0, 6.0, 8.0, 10.0],
            }
        ).to_csv(target, index=False)
        return str(target)

    def test_hostile_title_is_neutralized(self, bench: Path) -> None:
        path = _fixture_csv(bench)
        envelope = _render(
            {
                "kind": "histogram",
                "path": path,
                "encoding": {"value": "x"},
                "title": "<script>alert('xss')</script> onload=evil()",
            }
        )
        svg = _artifact_svg(envelope)
        assert _inertness_violations(svg) == []
        # the payload survives, but only as inert escaped text content
        root = ElementTree.fromstring(svg)
        texts = "".join(
            node.text or "" for node in root.iter() if _local(node.tag) == "text"
        )
        assert "<script>" in texts  # present as TEXT, parsed from &lt;script&gt;

    def test_hostile_data_label_is_neutralized(self, bench: Path) -> None:
        path = self._hostile_csv(bench)
        envelope = _render(
            {
                "kind": "histogram",
                "path": path,
                "encoding": {"value": "<script>alert(1)</script>"},
            }
        )
        svg = _artifact_svg(envelope)
        assert _inertness_violations(svg) == []
