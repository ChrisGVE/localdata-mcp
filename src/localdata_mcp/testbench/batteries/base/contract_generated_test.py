"""MACHINE-WRITTEN by localdata_mcp.nexus.contract.generators.test_stub — DO NOT EDIT.

Parametrized L3 contract stubs (ARCHITECTURE.md 6.1 artifact 4):
every registered tool answers a well-formed fastmcp.Client call, and
the coverage check fails if any registered ToolSpec lacks an entry
here. Regenerate via `python -m localdata_mcp.nexus.contract.generate`;
hand edits fail CI through nexus/contract/check_drift.py.
"""

from __future__ import annotations

from typing import Any

import anyio
import pytest
from fastmcp import Client

from localdata_mcp.nexus.contract.registry import default_registry
from localdata_mcp.nexus.contract.spec_modules import load_spec_modules
from localdata_mcp.server.mcp_app import app

GENERATED_TOOL_CALLS: "tuple[tuple[str, dict[str, Any]], ...]" = (
    ("ping", {}),
    ("probe_table", {"rows": 3}),
    ("probe_vector", {"length": 3}),
    ("probe_matrix", {"size": 3}),
    ("probe_model", {"points": 3}),
    ("probe_graph", {"nodes": 3}),
    ("probe_geo", {"points": 3}),
    ("probe_chart", {"points": 3}),
    ("probe_sink", {"text": "probe"}),
)


@pytest.mark.parametrize(
    ("name", "arguments"),
    GENERATED_TOOL_CALLS,
    ids=[name for name, _ in GENERATED_TOOL_CALLS],
)
def test_tool_answers_well_formed(name: str, arguments: "dict[str, Any]") -> None:
    async def session() -> None:
        async with Client(app) as client:
            result = await client.call_tool(name, arguments)
            assert not result.is_error

    anyio.run(session)


def test_every_registered_spec_has_a_generated_entry() -> None:
    load_spec_modules()
    registered = {spec.name for spec in default_registry()}
    generated = {name for name, _ in GENERATED_TOOL_CALLS}
    missing = registered - generated
    assert not missing, (
        f"registered ToolSpecs lacking generated L3 entries: {sorted(missing)}"
    )
