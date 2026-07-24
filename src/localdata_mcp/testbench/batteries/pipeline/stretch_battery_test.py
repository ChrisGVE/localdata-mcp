"""testbench/batteries/pipeline/stretch_battery_test.py — E14.7 stretch probe.

The non-gating stretch battery (S8 row 27). It reaches PAST the
exhaustively-executed length-2..4 envelope by SAMPLING alternating
chains up to `testbench.stretch_max_length` — the one place sampling is
allowed (REQUIREMENTS §6(k)) — and running each through the real
compose_pipeline seam. `testbench.stretch_sample_chains` fixes the
sample size and a fixed seed makes the draw reproducible. It is
NON-GATING: a chain that engine-rejects or fails at the domain level is
RECORDED, not asserted against, so a stretch regression surfaces in the
results store (NFR-508) without blocking merge. The battery asserts only
that the configured sample actually executed. Marked `nightly`.
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
from localdata_mcp.testbench.batteries.pipeline.enumeration import (
    ordered_links,
    sampled_stretch_chains,
)

pytestmark = pytest.mark.nightly

_STAGE_FAILURE_PREFIX = "pipeline stage "
# A fixed sampling seed makes the nightly draw reproducible (not an S8
# ConfigModel default — the stretch sample size and length are, and are
# read from config below).
_STRETCH_SEED = 42


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


def _fixture(tmp_path: Path) -> str:
    target = tmp_path / "generic.csv"
    pd.DataFrame(
        {
            "value": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "other": [2.0, 4.0, 5.0, 4.0, 6.0, 7.0],
            "group": ["a", "b", "a", "b", "a", "b"],
        }
    ).to_csv(target, index=False)
    return str(target)


class TestStretchProbe:
    def test_sampled_chains_execute_and_are_recorded(
        self, bench: Path, record_property: Any
    ) -> None:
        config = ConfigModel().testbench
        path = _fixture(bench)
        chains = sampled_stretch_chains(
            ordered_links(default_registry()),
            max_length=config.stretch_max_length,
            sample_count=config.stretch_sample_chains,
            seed=_STRETCH_SEED,
            source_path=path,
        )

        executed = 0
        engine_rejections = 0
        for chain in chains:
            envelope = _call("compose_pipeline", {"dag_spec": chain.dag_spec})
            error = envelope["error"]
            if error is not None and not error["message"].startswith(
                _STAGE_FAILURE_PREFIX
            ):
                engine_rejections += 1
            executed += 1

        # Non-gating: outcomes are recorded, never asserted against.
        record_property(
            "numeric_output",
            {"executed": executed, "engine_rejections": engine_rejections},
        )
        assert executed == config.stretch_sample_chains
