"""localdata_mcp/nexus/contract/generators/test_stub.py — artifact 4.

Renders testbench/batteries/base/contract_generated_test.py: one
parametrized L3 contract stub per registered tool through the
fastmcp.Client in-memory seam (FR-702/NFR-501), plus the coverage
check that FAILS when a registered ToolSpec lacks its generated entry
— the always-on L3-presence gate. Sample argument literals derive
from each Param's annotation; an annotation outside the supported set
fails GENERATION (honest refusal, never a silently skipped tool).
Neighbors: generate.py writes the output; purity_runner.py is the
sibling OS-level seam the batteries wrap.
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
}

_HEADER = f'''"""MACHINE-WRITTEN by {GENERATOR_NAME} — DO NOT EDIT.

Parametrized L3 contract stubs (ARCHITECTURE.md 6.1 artifact 4):
every registered tool answers a well-formed fastmcp.Client call, and
the coverage check fails if any registered ToolSpec lacks an entry
here. Regenerate via `python -m localdata_mcp.nexus.contract.generate`;
hand edits fail CI through nexus/contract/check_drift.py.
"""

from __future__ import annotations

import json
from typing import Any

import anyio
import pytest
from fastmcp import Client

from localdata_mcp.nexus.contract.registry import default_registry
from localdata_mcp.nexus.contract.spec_modules import load_spec_modules
from localdata_mcp.server.mcp_app import app

GENERATED_TOOL_CALLS: "tuple[tuple[str, dict[str, Any]], ...]" = (
'''

_FOOTER = """)

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
            envelope = _envelope_of(result)
            # FR-403: the four-region schema on every tool, and error
            # exclusive with the other regions.
            assert set(envelope) >= _ENVELOPE_REGIONS
            if envelope["error"] is None:
                assert envelope["inline"] is not None
            else:
                assert envelope["inline"] is None
                assert envelope["data"] is None
                assert envelope["composition_metadata"] is None

    anyio.run(session)


def test_every_registered_spec_has_a_generated_entry() -> None:
    load_spec_modules()
    registered = {spec.name for spec in default_registry()}
    generated = {name for name, _ in GENERATED_TOOL_CALLS}
    missing = registered - generated
    assert not missing, (
        f"registered ToolSpecs lacking generated L3 entries: {sorted(missing)}"
    )
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


def render_test_module(registry: ToolRegistry) -> str:
    """The complete contract_generated_test.py text for `registry`."""
    entries = "".join(
        f'    ("{spec.name}", {_sample_arguments(spec)}),\n' for spec in registry
    )
    return _HEADER + entries + _FOOTER
