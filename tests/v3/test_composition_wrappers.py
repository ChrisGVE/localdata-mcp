"""tests/v3/test_composition_wrappers.py — E11.3 prep wrappers (S3.6 C-2).

The acceptance: each wrapper's L3 result equals the equivalent
explicit compose_pipeline call bit-for-bit — the wrapper is a fixed
dag_spec into the same engine, so equality is by construction and the
test pins it. Plus the wrappers' whole point: clean_then_profile is
callable with a source alone (P-2's drop default governs)."""

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
    target = tmp_path / "data.csv"
    pd.DataFrame(
        {
            "x": [1.0, 2.0, None, 4.0, 5.0, 6.0],
            "y": [2.1, 3.9, 6.2, None, 10.1, 12.0],
            "label": ["a", "b", "a", "b", "a", "b"],
        }
    ).to_csv(target, index=False)
    return str(target)


def _drop_volatile(envelope: dict[str, Any]) -> dict[str, Any]:
    """The composition_metadata's recommended_next_steps is registry-
    order-derived and identical across both calls; the provenance stage
    names differ only where the wrapper fixes them, which we assert
    separately. Compare the substantive result payload."""
    return {
        "error": envelope["error"],
        "results": envelope["data"]["results"] if envelope["data"] else None,
    }


class TestCleanThenProfile:
    def test_equals_explicit_compose(self, bench: Path) -> None:
        path = _fixture_csv(bench)
        wrapper = _call("clean_then_profile", {"path": path})
        explicit = _call(
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
        assert _drop_volatile(wrapper) == _drop_volatile(explicit)

    def test_callable_with_source_alone(self, bench: Path) -> None:
        path = _fixture_csv(bench)
        envelope = _call("clean_then_profile", {"path": path})
        assert envelope["error"] is None, envelope["error"]
        # drop default: the two NaN-bearing rows removed, four left.
        assert envelope["data"]["results"]["profile"]["data"]["row_count"] == 4

    def test_strategy_passthrough_changes_the_result(self, bench: Path) -> None:
        path = _fixture_csv(bench)
        envelope = _call(
            "clean_then_profile", {"path": path, "missing_strategy": "mean"}
        )
        assert envelope["error"] is None, envelope["error"]
        # mean fills rather than drops: all six rows survive.
        assert envelope["data"]["results"]["profile"]["data"]["row_count"] == 6


class TestCleanThenRegress:
    def test_equals_explicit_compose(self, bench: Path) -> None:
        path = _fixture_csv(bench)
        wrapper = _call("clean_then_regress", {"path": path, "target": "y"})
        explicit = _call(
            "compose_pipeline",
            {
                "dag_spec": [
                    {
                        "stage": "clean",
                        "tool": "prepare_missing_values",
                        "params": {"path": path},
                    },
                    {
                        "stage": "regress",
                        "tool": "analyze_regression",
                        "params": {"target_column": "y"},
                        "depends_on": ["clean"],
                    },
                ]
            },
        )
        assert _drop_volatile(wrapper) == _drop_volatile(explicit)

    def test_runs_a_regression(self, bench: Path) -> None:
        path = _fixture_csv(bench)
        envelope = _call("clean_then_regress", {"path": path, "target": "y"})
        assert envelope["error"] is None, envelope["error"]
        assert set(envelope["data"]["results"]) == {"regress"}
        assert envelope["data"]["results"]["regress"]["error"] is None
