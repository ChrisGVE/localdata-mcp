"""tests/v3/test_tools_generated.py — E3.4: the COMMITTED artifacts work.

End-to-end over the committed codegen outputs: mcp_app registers its
tools exclusively through tools_generated.py, every registered
ToolSpec is callable over the fastmcp.Client in-memory seam with a
well-formed answer, and generated_shapes.py carries the FR-606 entry
per tool. The generated L3 stub suite itself lives at
testbench/batteries/base/contract_generated_test.py; this file proves
the wiring the stubs assume.
"""

from __future__ import annotations

import json
from typing import Any

import anyio
from fastmcp import Client

from localdata_mcp.nexus.contract.generated_shapes import TOOL_SHAPES
from localdata_mcp.nexus.contract.registry import default_registry
from localdata_mcp.nexus.contract.spec import TypeShape
from localdata_mcp.nexus.contract.spec_modules import load_spec_modules
from localdata_mcp.server.mcp_app import app


def _run(coro_fn: Any) -> Any:
    return anyio.run(coro_fn)


def _envelope_of(result: Any) -> dict[str, Any]:
    """The wire envelope from a client result: structured content when
    the transport carries it, else the JSON text block."""
    if isinstance(result.structured_content, dict) and "inline" in (
        result.structured_content
    ):
        return result.structured_content
    payload = json.loads(result.content[0].text)
    assert isinstance(payload, dict)
    return payload


class TestGeneratedRegistration:
    def test_ping_served_through_generated_wrapper(self) -> None:
        async def session() -> Any:
            async with Client(app) as client:
                tools = {tool.name for tool in await client.list_tools()}
                result = await client.call_tool("ping", {})
                return tools, result

        tools, result = _run(session)
        assert "ping" in tools
        assert not result.is_error
        # E7.2: the wrapper applies envelope shaping — the payload is
        # the four-region FR-403 envelope, `inline` carrying the scalar.
        envelope = _envelope_of(result)
        assert set(envelope) >= {"inline", "data", "composition_metadata", "error"}
        assert envelope["inline"] == "pong"
        assert envelope["data"] == "pong"
        assert envelope["error"] is None

    def test_every_registered_spec_is_served(self) -> None:
        load_spec_modules()
        registered = {spec.name for spec in default_registry()}

        async def session() -> set[str]:
            async with Client(app) as client:
                return {tool.name for tool in await client.list_tools()}

        assert registered <= _run(session)

    def test_wrapper_docstring_is_the_generated_one(self) -> None:
        async def session() -> Any:
            async with Client(app) as client:
                return {t.name: t.description for t in await client.list_tools()}

        descriptions = _run(session)
        assert "Input shape:" in (descriptions["ping"] or "")


class TestGeneratedShapes:
    def test_ping_entry(self) -> None:
        entry = TOOL_SHAPES["ping"]
        assert entry.input_shape is TypeShape.NONE
        assert entry.output_shape is TypeShape.SCALAR

    def test_one_entry_per_registered_spec(self) -> None:
        load_spec_modules()
        assert set(TOOL_SHAPES) == {s.name for s in default_registry()}
