"""localdata_mcp/nexus/contract/generators/test_stub.py — artifact 4.

Renders testbench/batteries/base/contract_generated_test.py: one
parametrized L3 contract stub per registered tool through the
fastmcp.Client in-memory seam (FR-702/NFR-501). Served (production)
tools are exercised on the app mcp_app.py serves; test_only walking-
skeleton probes are exercised on a battery-local app so GP5's L3 seam
still covers them without polluting the served MCP surface (CR-012).
The coverage check FAILS when a registered ToolSpec lacks its generated
entry in either list — the always-on L3-presence gate. Sample argument
literals derive from each Param's annotation; an annotation outside the
supported set fails GENERATION (honest refusal, never a silently
skipped tool). Neighbors: generate.py writes the output; purity_runner.py
is the sibling OS-level seam the batteries wrap.
"""

from __future__ import annotations

from localdata_mcp.nexus.contract.errors import ToolContractError
from localdata_mcp.nexus.contract.registry import ToolRegistry
from localdata_mcp.nexus.contract.spec import ToolSpec

GENERATOR_NAME = "localdata_mcp.nexus.contract.generators.test_stub"

# Annotation -> sample literal. Ints stay <= 3 and floats in {0.0, 1.0}
# so generated files never trip the NFR-403 default-site scan.
_SAMPLE_LITERALS: dict[type, str] = {
    str: '"probe"',
    int: "3",
    float: "1.0",
    bool: "True",
    list: '["probe"]',
    dict: "{}",
}

_HEADER = f'''"""MACHINE-WRITTEN by {GENERATOR_NAME} — DO NOT EDIT.

Parametrized L3 contract stubs (ARCHITECTURE.md 6.1 artifact 4): every
served tool answers a well-formed fastmcp.Client call on the served
app, every test_only walking-skeleton probe answers on a battery-local
app kept OFF the served surface (CR-012), and the coverage check fails
if any registered ToolSpec lacks an entry in either list. Regenerate
via `python -m localdata_mcp.nexus.contract.generate`; hand edits fail
CI through nexus/contract/check_drift.py.
"""

from __future__ import annotations

import json
from typing import Any

import anyio
import pytest
from fastmcp import Client, FastMCP

from localdata_mcp.nexus.contract.registry import default_registry
from localdata_mcp.nexus.contract.spec_modules import load_spec_modules
from localdata_mcp.server.mcp_app import app
from localdata_mcp.server.tools_generated import register_skeleton_tools

# The served product surface: exercised against the app mcp_app.py boots.
SERVED_TOOL_CALLS: "tuple[tuple[str, dict[str, Any]], ...]" = (
'''

_MIDDLE = """)

# The test_only walking-skeleton probes: exercised against a battery-
# local app built here, so they never reach the served surface (CR-012)
# yet stay L3-proven at the seam (GP5).
SKELETON_TOOL_CALLS: "tuple[tuple[str, dict[str, Any]], ...]" = (
"""

_FOOTER = """)

_skeleton_app = FastMCP("localdata-skeleton-probes")
register_skeleton_tools(_skeleton_app)

_ENVELOPE_REGIONS = {"inline", "data", "composition_metadata", "error"}


def _envelope_of(result: Any) -> "dict[str, Any]":
    \"\"\"The wire envelope from a client result: structured content when
    the transport carries it, else the JSON text block.\"\"\"
    if isinstance(result.structured_content, dict) and (
        set(result.structured_content) >= _ENVELOPE_REGIONS
    ):
        return result.structured_content
    payload = json.loads(result.content[0].text)
    assert isinstance(payload, dict)
    return payload


def _assert_well_formed(target: Any, name: str, arguments: "dict[str, Any]") -> None:
    \"\"\"Call `name` on `target` and assert the FR-403 four-region
    envelope, with error exclusive of the other regions.\"\"\"

    async def session() -> None:
        async with Client(target) as client:
            result = await client.call_tool(name, arguments)
            assert not result.is_error
            envelope = _envelope_of(result)
            assert set(envelope) >= _ENVELOPE_REGIONS
            if envelope["error"] is None:
                assert envelope["inline"] is not None
            else:
                assert envelope["inline"] is None
                assert envelope["data"] is None
                assert envelope["composition_metadata"] is None

    anyio.run(session)


@pytest.mark.parametrize(
    ("name", "arguments"),
    SERVED_TOOL_CALLS,
    ids=[name for name, _ in SERVED_TOOL_CALLS],
)
def test_served_tool_answers_well_formed(
    name: str, arguments: "dict[str, Any]"
) -> None:
    _assert_well_formed(app, name, arguments)


@pytest.mark.parametrize(
    ("name", "arguments"),
    SKELETON_TOOL_CALLS,
    ids=[name for name, _ in SKELETON_TOOL_CALLS],
)
def test_skeleton_probe_answers_well_formed(
    name: str, arguments: "dict[str, Any]"
) -> None:
    _assert_well_formed(_skeleton_app, name, arguments)


def test_every_registered_spec_has_a_generated_entry() -> None:
    load_spec_modules()
    registered = {spec.name for spec in default_registry()}
    generated = {name for name, _ in SERVED_TOOL_CALLS} | {
        name for name, _ in SKELETON_TOOL_CALLS
    }
    missing = registered - generated
    assert not missing, (
        f"registered ToolSpecs lacking generated L3 entries: {sorted(missing)}"
    )


def test_skeleton_probes_are_absent_from_the_served_app() -> None:
    \"\"\"CR-036: a direct negative assertion that the served app exposes
    NONE of the walking-skeleton probe names. CR-012's OFF-the-surface
    guarantee is otherwise carried only by the generator partition
    (not spec.test_only) plus the drift gate; this proves it at the live
    fastmcp.Client wire, so a regression that wrongly registered a probe
    into production fails here, not only on regeneration.\"\"\"
    skeleton_names = {name for name, _ in SKELETON_TOOL_CALLS}

    async def session() -> None:
        async with Client(app) as client:
            served = {tool.name for tool in await client.list_tools()}
        leaked = skeleton_names & served
        assert not leaked, (
            f"walking-skeleton probes leaked onto the served surface: {sorted(leaked)}"
        )

    anyio.run(session)
"""


def _sample_arguments(spec: ToolSpec) -> str:
    # Required params only: the stub also PROVES every optional param
    # is genuinely caller-omittable at the wire (E8.5).
    pairs = []
    for param in spec.params:
        if not param.required:
            continue
        literal = _SAMPLE_LITERALS.get(param.annotation)
        if literal is None:
            raise ToolContractError(
                f"no sample literal for annotation {param.annotation!r}",
                tool_name=spec.name,
            )
        pairs.append(f'"{param.name}": {literal}')
    return "{" + ", ".join(pairs) + "}"


def _entries(specs: list[ToolSpec]) -> str:
    return "".join(
        f'    ("{spec.name}", {_sample_arguments(spec)}),\n' for spec in specs
    )


def render_test_module(registry: ToolRegistry) -> str:
    """The complete contract_generated_test.py text for `registry`."""
    served = [spec for spec in registry if not spec.test_only]
    skeleton = [spec for spec in registry if spec.test_only]
    return _HEADER + _entries(served) + _MIDDLE + _entries(skeleton) + _FOOTER
