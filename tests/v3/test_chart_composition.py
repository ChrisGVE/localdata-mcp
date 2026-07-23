"""tests/v3/test_chart_composition.py — charts fed by a pipeline (E12.5).

FR-504's pipe leg at L3: render_chart is a terminal chart-sink stage
inside compose_pipeline, fed the upstream stage's output through the
stage-handoff channel (explore.addressing.pipeline_input), the real
tools through the real MCP seam. This is the composition-sourced half
of E12.5 (the cluster_then_chart wrapper is the other half). A chart
stage declares output_shape=NONE, so it is a legal terminal leaf and
its artifact rides in the leaf envelope.
"""

from __future__ import annotations

import json
import os
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
    target = tmp_path / "raw.csv"
    pd.DataFrame(
        {
            "age": [34.0, None, 51.0, 29.0, 43.0, 38.0],
            "score": [88.0, 75.0, 62.0, 91.0, 84.0, 70.0],
        }
    ).to_csv(target, index=False)
    return str(target)


def _call(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    async def session() -> dict[str, Any]:
        async with Client(app) as client:
            result = await client.call_tool(name, arguments)
            assert not result.is_error, result
            if isinstance(result.structured_content, dict) and (
                "inline" in result.structured_content
            ):
                return result.structured_content
            payload = json.loads(result.content[0].text)
            assert isinstance(payload, dict)
            return payload

    return anyio.run(session)


class TestChartFedByPipeline:
    """FR-504: a preprocessing stage feeds a terminal render_chart stage."""

    def test_clean_then_chart_runs(self, bench: Path) -> None:
        path = _fixture_csv(bench)
        envelope = _call(
            "compose_pipeline",
            {
                "dag_spec": [
                    {
                        "stage": "clean",
                        "tool": "prepare_missing_values",
                        "params": {"path": path, "missing_strategy": "mean"},
                    },
                    {
                        "stage": "chart",
                        "tool": "render_chart",
                        "params": {"kind": "histogram", "encoding": {"value": "age"}},
                        "depends_on": ["clean"],
                    },
                ]
            },
        )
        assert envelope["error"] is None, envelope["error"]
        results = envelope["data"]["results"]
        assert set(results) == {"chart"}
        leaf = results["chart"]
        # the terminal chart leaf carries the rendered artifact
        chart = leaf["data"] if "data" in leaf else leaf["inline"]
        assert chart["kind"] == "histogram"
        # fed by the upstream stage — the engine labels the injected frame
        assert chart["source"] == "pipeline:clean"
        assert chart["artifact"]["media_type"] == "image/svg+xml"
        assert "<svg" in chart["artifact"]["content"]

    def test_chart_stage_is_a_terminal_leaf(self, bench: Path) -> None:
        """render_chart's output_shape=NONE makes it a legal sink — a
        stage depending ON a chart stage is rejected pre-execution."""
        path = _fixture_csv(bench)
        envelope = _call(
            "compose_pipeline",
            {
                "dag_spec": [
                    {
                        "stage": "chart",
                        "tool": "render_chart",
                        "params": {
                            "kind": "histogram",
                            "path": path,
                            "encoding": {"value": "age"},
                        },
                    },
                    {
                        "stage": "after",
                        "tool": "profile_data",
                        "depends_on": ["chart"],
                    },
                ]
            },
        )
        # a NONE-output stage feeding a downstream stage is a structured
        # rejection naming the offending edge, no partial run
        assert envelope["error"] is not None
