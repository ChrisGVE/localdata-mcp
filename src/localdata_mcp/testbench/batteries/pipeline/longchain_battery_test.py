"""testbench/batteries/pipeline/longchain_battery_test.py — NFR-502d nightly.

The nightly execution tier: every derived-valid alternating chain at
lengths 3 and 4 run end-to-end through the real compose_pipeline seam
(S7.4). The length-2 battery proves the FR-606 classification and DERIVES
the longer-chain totals from the length-2 link facts; this battery
executes those derived-valid chains for real — a valid A-B-A(-B) chain
must not draw an ENGINE-level rejection (a domain-level 'meaningless but
correct' stage failure is expected and accepted, FR-302). Marked
`nightly` so the per-PR tier stays inside its 30-minute budget (the
exhaustive lengths-3-4 run is the 4-hour nightly's job). Each per-chain
outcome is recorded for the results store (NFR-508).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterator, List

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
    DomainLink,
    alternating_dag_spec,
    ordered_links,
)

pytestmark = pytest.mark.nightly

# A pre-execution FR-606 refusal never carries the per-stage failure
# prefix; its absence marks an acceptable domain-level failure.
_STAGE_FAILURE_PREFIX = "pipeline stage "
_NIGHTLY_LENGTHS = (3, 4)


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


def _valid_alternating_links(links: "tuple[DomainLink, ...]") -> List[DomainLink]:
    """Ordered pairs whose BOTH directed links are legal — the chains an
    alternating A-B-A(-B) extension is valid over (NFR-502d)."""
    index = {(link.source_domain, link.target_domain): link.legal for link in links}
    return [
        link
        for link in links
        if index[(link.source_domain, link.target_domain)]
        and index[(link.target_domain, link.source_domain)]
    ]


def _engine_rejected(link: DomainLink, length: int, path: str) -> "tuple[bool, str]":
    spec = alternating_dag_spec(link, length, path)
    envelope = _call("compose_pipeline", {"dag_spec": spec})
    error = envelope["error"]
    if error is None:
        return False, "ran (no engine rejection)"
    message = error["message"]
    return not message.startswith(_STAGE_FAILURE_PREFIX), message


class TestNightlyLongChains:
    @pytest.mark.parametrize("length", _NIGHTLY_LENGTHS)
    def test_derived_valid_chains_execute_without_engine_rejection(
        self, bench: Path, length: int, record_property: Any
    ) -> None:
        path = _fixture(bench)
        valid = _valid_alternating_links(ordered_links(default_registry()))
        assert valid, "no bidirectional domain pairs — the launch set went inert"

        executed = 0
        for link in valid:
            rejected, message = _engine_rejected(link, length, path)
            assert not rejected, (
                f"derived-valid chain {link.source_domain}<->{link.target_domain} "
                f"at length {length} was engine-rejected: {message}"
            )
            executed += 1
        record_property("numeric_output", {"length": length, "executed": executed})
        assert executed == len(valid)
