"""localdata_mcp/process/composition/tools.py — the compose_pipeline ToolSpec (E11.2).

FR-601: composition as a first-class MCP tool surface — one thin
`@tool_spec` declaration over the stage_runner engine. Its contract is
§6.1's one sanctioned DYNAMIC entry: `input_shape=output_shape=DYNAMIC`
(excluded from adjacency checks — the engine validates the submitted
dag_spec's concrete stage shapes instead) and barred from appearing as
a stage inside a dag_spec, so it cannot nest. E3.6's deferral closes
here: registration and the L3 contract test land in the same
change-set. Neighbors: stage_runner/sequence.py is the engine;
wrappers.py declares the C-2 convenience chains over the same engine;
spec_modules.py rosters this module.
"""

from __future__ import annotations

from typing import Any

from localdata_mcp.nexus.contract.spec import Param, TypeShape, tool_spec

from .stage_runner.sequence import run_pipeline


@tool_spec(
    name="compose_pipeline",
    summary=(
        "Compose registered tools into one validated pipeline: dag_spec "
        "is an ordered list of {stage, tool, params, depends_on} entries "
        "(linear chains with fan-out; a chain-initial stage addresses its "
        "own source, a dependent stage consumes its upstream stage's "
        "output). The whole chain is validated against the tool contracts "
        "and the type-shape adjacency table BEFORE anything runs — an "
        "incompatible chain is rejected naming the offending stage, with "
        "no partial run. Returns one envelope per terminal stage under a "
        "single provenance chain."
    ),
    params=(
        Param(
            "dag_spec",
            list,
            "Ordered stage entries: {stage: name, tool: registered tool, "
            "params: tool arguments, depends_on: [upstream stage]}.",
        ),
    ),
    input_shape=TypeShape.DYNAMIC,
    output_shape=TypeShape.DYNAMIC,
    domain="composition",
)
def compose_pipeline_tool(dag_spec: "list[Any]") -> dict[str, Any]:
    return run_pipeline(dag_spec)
