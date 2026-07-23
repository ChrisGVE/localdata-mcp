"""localdata_mcp/nexus/contract/generators/wrapper.py — artifact 1's renderer.

Renders server/tools_generated.py: the FastMCP registration wrapper
module mcp_app.py imports at startup. Per spec, one wrapper function
with the declared signature and the GENERATED docstring (docstring.py)
delegating to the registered implementation — startup only imports,
never generates (section 6.1). Neighbors: generate.py writes the
output; spec_modules.py gives the generated module the same tool
population the generator saw.
"""

from __future__ import annotations

from localdata_mcp.nexus.contract.generators.docstring import render_docstring
from localdata_mcp.nexus.contract.registry import ToolRegistry
from localdata_mcp.nexus.contract.spec import ToolSpec

GENERATOR_NAME = "localdata_mcp.nexus.contract.generators.wrapper"

_HEADER = f'''"""MACHINE-WRITTEN by {GENERATOR_NAME} — DO NOT EDIT.

FastMCP registration wrappers for every registered ToolSpec
(ARCHITECTURE.md 6.1 artifacts 1+2). Regenerate via
`python -m localdata_mcp.nexus.contract.generate`; hand edits fail CI
through nexus/contract/check_drift.py.
"""

from __future__ import annotations

from typing import Any

from fastmcp import FastMCP

from localdata_mcp.nexus.contract.registry import default_registry
from localdata_mcp.nexus.contract.spec_modules import load_spec_modules
from localdata_mcp.nexus.response.shaping import shaped_call

'''


def _indent_docstring(text: str) -> str:
    """`text` as a triple-quoted docstring block at wrapper depth."""
    body = "\n".join(f"        {line}" if line else "" for line in text.splitlines())
    quotes = '        """'
    return quotes + "\n" + body + "\n" + quotes


def _render_one(spec: ToolSpec) -> str:
    # Required params first (Python signature rule), optional params as
    # `type | None = None` — the MCP schema itself carries optionality
    # (E8.5). An omitted optional is NOT forwarded, so the registered
    # implementation's own default governs (one default site, NFR-403).
    ordered = sorted(spec.params, key=lambda param: not param.required)
    signature = ", ".join(
        f"{param.name}: {param.annotation.__name__}"
        if param.required
        else f"{param.name}: {param.annotation.__name__} | None = None"
        for param in ordered
    )
    required_pairs = ", ".join(
        f'"{param.name}": {param.name}' for param in spec.params if param.required
    )
    optional_forwards = "".join(
        f"        if {param.name} is not None:\n"
        f'            arguments["{param.name}"] = {param.name}\n'
        for param in spec.params
        if not param.required
    )
    return (
        f'    _impl_{spec.name} = registry.lookup("{spec.name}").func\n'
        f"\n"
        f"    def {spec.name}({signature}) -> Any:\n"
        f"{_indent_docstring(render_docstring(spec))}\n"
        f"        arguments: dict[str, Any] = {{{required_pairs}}}\n"
        f"{optional_forwards}"
        # Envelope-shaping is wrapper-applied, never opt-in (E7.2/O-1):
        # every call routes through the one shaping seam.
        f'        return shaped_call("{spec.name}", _impl_{spec.name}, arguments)\n'
        f"\n"
        f"    app.tool({spec.name})\n"
    )


def render_wrapper_module(registry: ToolRegistry) -> str:
    """The complete tools_generated.py text for `registry`."""
    blocks = "\n".join(_render_one(spec) for spec in registry)
    return (
        _HEADER
        + "\ndef register_tools(app: FastMCP) -> None:\n"
        + '    """Register every generated tool wrapper on `app`."""\n'
        + "    load_spec_modules()\n"
        + "    registry = default_registry()\n"
        + "\n"
        + blocks
    )
