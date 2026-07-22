"""localdata_mcp/server/skeleton_tools.py — walking-skeleton ToolSpecs.

Trivial but REAL pure tools (each computes its declared output shape,
no stubs) proving the NX-1 one-declaration pipeline end to end: spec
here -> generated wrapper/docs/test-stub/shape-registry artifacts ->
FastMCP registration in mcp_app.py. E3.6 keeps one tool per
non-DYNAMIC TypeShape so FR-701/704 acceptance runs against a
populated registry; E5+ domain tools replace this set. Registered via
nexus/contract/spec_modules.py's roster — never imported directly by
the server.
"""

from __future__ import annotations

from localdata_mcp.nexus.contract.spec import Param, TypeShape, tool_spec


@tool_spec(
    name="ping",
    summary="Report server liveness with a constant probe response.",
    params=[],
    input_shape=TypeShape.NONE,
    output_shape=TypeShape.SCALAR,
)
def ping() -> str:
    return "pong"
