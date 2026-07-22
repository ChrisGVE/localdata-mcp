"""MACHINE-WRITTEN by localdata_mcp.nexus.contract.generators.wrapper — DO NOT EDIT.

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


def register_tools(app: FastMCP) -> None:
    """Register every generated tool wrapper on `app`."""
    load_spec_modules()
    registry = default_registry()

    _impl_ping = registry.lookup("ping").func

    def ping() -> Any:
        """
        Report server liveness with a constant probe response.

        Input shape: NONE (chain endpoint — composes with nothing).
        Output shape: SCALAR.
        Streaming-capable: no.
        """
        return _impl_ping()

    app.tool(ping)
