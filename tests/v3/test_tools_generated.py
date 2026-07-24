"""tests/v3/test_tools_generated.py — E3.4: the COMMITTED artifacts work.

End-to-end over the committed codegen outputs: mcp_app registers its
served tools exclusively through tools_generated.py's register_tools,
every production ToolSpec is callable over the fastmcp.Client in-memory
seam with a well-formed answer, and generated_shapes.py carries the
FR-606 entry per tool. The test_only walking-skeleton probes stay OFF
the served surface (CR-012) — register_skeleton_tools wires them onto a
battery-local app, proven here. The generated L3 stub suite itself
lives at testbench/batteries/base/contract_generated_test.py; this file
proves the wiring the stubs assume.
"""

from __future__ import annotations

import json
from typing import Any

import anyio
from fastmcp import Client, FastMCP

from localdata_mcp.nexus.contract.generated_shapes import TOOL_SHAPES
from localdata_mcp.nexus.contract.registry import default_registry
from localdata_mcp.nexus.contract.spec import TypeShape
from localdata_mcp.nexus.contract.spec_modules import load_spec_modules
from localdata_mcp.server.mcp_app import app
from localdata_mcp.server.tools_generated import register_skeleton_tools


def _new_skeleton_app() -> FastMCP:
    """A throwaway app carrying only the test_only probes — the seam
    that keeps them L3-proven without touching the served surface."""
    probe_app = FastMCP("localdata-skeleton-probes-test")
    register_skeleton_tools(probe_app)
    return probe_app


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


def _served_tool_names() -> set[str]:
    async def session() -> set[str]:
        async with Client(app) as client:
            return {tool.name for tool in await client.list_tools()}

    return _run(session)


class TestGeneratedRegistration:
    def test_skeleton_probes_are_off_the_served_surface(self) -> None:
        """CR-012: no walking-skeleton probe reaches a served MCP client
        — register_tools carries the production surface exclusively."""
        load_spec_modules()
        served = _served_tool_names()
        test_only = {spec.name for spec in default_registry() if spec.test_only}
        assert test_only, "expected the walking-skeleton probes to be registered"
        assert "ping" not in served
        assert not any(name.startswith("probe_") for name in served)
        assert not (test_only & served)

    def test_ping_served_through_the_skeleton_app(self) -> None:
        """The probes stay L3-proven off the served surface: register_
        skeleton_tools wires them onto a battery-local app carrying the
        same generated envelope shaping."""

        async def session() -> Any:
            async with Client(_new_skeleton_app()) as client:
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

    def test_every_production_spec_is_served(self) -> None:
        load_spec_modules()
        production = {spec.name for spec in default_registry() if not spec.test_only}
        assert production <= _served_tool_names()

    def test_wrapper_docstring_is_the_generated_one(self) -> None:
        async def session() -> Any:
            async with Client(app) as client:
                return {t.name: t.description for t in await client.list_tools()}

        descriptions = _run(session)
        assert "Input shape:" in (descriptions["list_endpoints"] or "")


class TestGeneratedShapes:
    def test_ping_entry(self) -> None:
        entry = TOOL_SHAPES["ping"]
        assert entry.input_shape is TypeShape.NONE
        assert entry.output_shape is TypeShape.SCALAR

    def test_one_entry_per_registered_spec(self) -> None:
        load_spec_modules()
        assert set(TOOL_SHAPES) == {s.name for s in default_registry()}
