"""tests/v3/test_composition_tool.py — E11.2 compose_pipeline at L3 (FR-601/602/606).

The real tool through the real MCP seam (fastmcp.Client) over the real
registered tool surface: a working prep -> profile chain (FR-601's
acceptance — one MCP tool whose documented purpose is composing two or
more Process/Explore tools, exercised at L3), the FR-606 stage-naming
structured rejection with no partial run, the S8 row-14 length guard,
fan-out's multi-leaf map, and the DYNAMIC nesting bar."""

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
    config = ConfigModel(
        security=SecurityConfig(allowed_paths=(str(tmp_path),)),
    )
    guard = Chokepoint.boot(config, environ=dict(os.environ))
    configure_shaping(config, default_registry())
    runtime.configure_ingest(guard)
    yield tmp_path
    runtime._CHOKEPOINT = None
    configure_shaping(ConfigModel(), default_registry())
    guard.shutdown()


def _call(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    async def session() -> dict[str, Any]:
        async with Client(app) as client:
            result = await client.call_tool(name, arguments)
            assert not result.is_error
            if isinstance(result.structured_content, dict) and (
                "inline" in result.structured_content
            ):
                return result.structured_content
            payload = json.loads(result.content[0].text)
            assert isinstance(payload, dict)
            return payload

    return anyio.run(session)


def _fixture_csv(tmp_path: Path) -> str:
    target = tmp_path / "people.csv"
    pd.DataFrame(
        {
            "age": [34.0, None, 51.0, 29.0, 43.0],
            "score": [88.0, 75.0, None, 91.0, 84.0],
            "group": ["a", "b", "a", "b", "a"],
        }
    ).to_csv(target, index=False)
    return str(target)


class TestComposedChainRuns:
    def test_prep_then_profile_end_to_end(self, bench: Path) -> None:
        """FR-601 acceptance: prepare_missing_values -> profile_data,
        composed through the one MCP tool, L3."""
        path = _fixture_csv(bench)
        envelope = _call(
            "compose_pipeline",
            {
                "dag_spec": [
                    {
                        "stage": "clean",
                        "tool": "prepare_missing_values",
                        "params": {"path": path},
                    },
                    {
                        "stage": "profile",
                        "tool": "profile_data",
                        "depends_on": ["clean"],
                    },
                ]
            },
        )
        assert envelope["error"] is None, envelope["error"]
        data = envelope["data"]
        assert set(data["results"]) == {"profile"}
        leaf = data["results"]["profile"]
        assert leaf["error"] is None
        # drop is the default strategy: the two NaN-carrying rows left.
        assert leaf["data"]["row_count"] == 3
        chain = data["provenance"]
        assert chain["execution_order"] == ["clean", "profile"]
        assert [s["tool"] for s in chain["stages"]] == [
            "prepare_missing_values",
            "profile_data",
        ]

    def test_three_stage_chain_with_params(self, bench: Path) -> None:
        """FR-602 acceptance at L3: >= 3 stages executing in
        topological order (prep -> convert -> profile)."""
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
                        "stage": "cast",
                        "tool": "convert_types",
                        "params": {"conversions": {"age": "numeric"}},
                        "depends_on": ["clean"],
                    },
                    {
                        "stage": "profile",
                        "tool": "profile_data",
                        "depends_on": ["cast"],
                    },
                ]
            },
        )
        assert envelope["error"] is None, envelope["error"]
        data = envelope["data"]
        assert data["provenance"]["execution_order"] == ["clean", "cast", "profile"]
        assert data["results"]["profile"]["data"]["row_count"] == 5

    def test_fan_out_multi_leaf_map(self, bench: Path) -> None:
        path = _fixture_csv(bench)
        envelope = _call(
            "compose_pipeline",
            {
                "dag_spec": [
                    {
                        "stage": "clean",
                        "tool": "prepare_missing_values",
                        "params": {"path": path},
                    },
                    {
                        "stage": "profile",
                        "tool": "profile_data",
                        "depends_on": ["clean"],
                    },
                    {
                        "stage": "search",
                        "tool": "search_data",
                        "params": {"query": "a"},
                        "depends_on": ["clean"],
                    },
                ]
            },
        )
        assert envelope["error"] is None, envelope["error"]
        results = envelope["data"]["results"]
        assert set(results) == {"profile", "search"}
        for leaf in results.values():
            assert leaf["error"] is None
        assert envelope["data"]["provenance"]["terminal_stages"] == [
            "profile",
            "search",
        ]


class TestStructuredRejections:
    def test_incompatible_chain_names_the_stage(self, bench: Path) -> None:
        """FR-606 acceptance: profile_data emits SCALAR; feeding it to
        a TABULAR consumer is rejected pre-execution, the offending
        edge named, nothing run."""
        path = _fixture_csv(bench)
        envelope = _call(
            "compose_pipeline",
            {
                "dag_spec": [
                    {
                        "stage": "profile",
                        "tool": "profile_data",
                        "params": {"path": path},
                    },
                    {
                        "stage": "clean",
                        "tool": "prepare_missing_values",
                        "depends_on": ["profile"],
                    },
                ]
            },
        )
        assert envelope["error"] is not None
        message = envelope["error"]["message"]
        assert "'profile' -> 'clean'" in message
        assert "scalar" in message
        assert envelope["inline"] is None  # error exclusive (FR-403)

    def test_length_guard_at_the_configured_bound(self, bench: Path) -> None:
        path = _fixture_csv(bench)
        bound = ConfigModel().composition.max_pipeline_length
        chain = [
            {
                "stage": "clean0",
                "tool": "prepare_missing_values",
                "params": {"path": path},
            }
        ]
        chain += [
            {
                "stage": f"clean{i}",
                "tool": "prepare_missing_values",
                "depends_on": [f"clean{i - 1}"],
            }
            for i in range(1, bound + 1)
        ]
        envelope = _call("compose_pipeline", {"dag_spec": chain})
        assert envelope["error"] is not None
        assert "max_pipeline_length" in envelope["error"]["message"]

    def test_compose_pipeline_cannot_nest(self, bench: Path) -> None:
        envelope = _call(
            "compose_pipeline",
            {
                "dag_spec": [
                    {"stage": "inner", "tool": "compose_pipeline"},
                ]
            },
        )
        assert envelope["error"] is not None
        assert "DYNAMIC" in envelope["error"]["message"]

    def test_unknown_tool_names_stage_and_tool(self, bench: Path) -> None:
        envelope = _call(
            "compose_pipeline",
            {"dag_spec": [{"stage": "ghost", "tool": "not_a_tool"}]},
        )
        assert envelope["error"] is not None
        assert "ghost" in envelope["error"]["message"]
        assert "not_a_tool" in envelope["error"]["message"]
