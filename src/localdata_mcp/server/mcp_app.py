"""localdata_mcp/server/mcp_app.py — the v3 process entrypoint.

Boots per ARCHITECTURE.md section 4e: guard fd 1 (fd_guard.py), bring
logging up in stderr-only bootstrap mode, load NX-2's layered config,
reconfigure logging from it, then serve the stdio transport over the
guarded descriptor — driving FastMCP's low-level `_mcp_server.run`
inside `mcp.server.stdio.stdio_server(stdout=...)`, bypassing the
argless `run_stdio_async` (PRD S5.3). WALKING SKELETON: the single
`ping` tool below is transport-proving wiring only — NX-1's generated
tool registry replaces it in E3. Neighbors: fd_guard.py supplies the
guard; nexus/observability and nexus/config are the booted nexuses.
"""

from __future__ import annotations

import sys

import anyio
from fastmcp import FastMCP
from mcp.server.lowlevel.server import NotificationOptions
from mcp.server.stdio import stdio_server

from ..nexus.config import ConfigLoadResult, ConfigurationError, load_config
from ..nexus.observability import get_logger, log_startup_report, reconfigure
from .fd_guard import StdoutGuard, install_stdout_guard

app = FastMCP("localdata")


@app.tool
def ping() -> str:
    """Walking-skeleton connectivity probe (replaced by NX-1 in E3)."""
    return "pong"


async def serve(guard: StdoutGuard) -> None:
    """Run the stdio session with frames bound to the guarded
    descriptor — the S5.3 injectable seam, not the argless default."""
    async with stdio_server(stdout=guard.protocol_out) as (read_stream, write_stream):
        await app._mcp_server.run(
            read_stream,
            write_stream,
            app._mcp_server.create_initialization_options(
                notification_options=NotificationOptions(tools_changed=True),
            ),
        )


def _load_validated_config() -> ConfigLoadResult | None:
    """Section 4e step two: the merged model, or None after logging the
    fatal ConfigurationError (bootstrap logging is enough to report it)."""
    try:
        return load_config()
    except ConfigurationError as error:
        get_logger(__name__).error(
            "configuration invalid; refusing to start",
            error=str(error),
            source=error.source,
            field_path=error.field_path,
        )
        return None


def main() -> int:
    """The section-4e boot order: guard, bootstrap log, config,
    reconfigure, serve. Importing this module already activated
    bootstrap logging (nexus/observability import side effect)."""
    guard = install_stdout_guard()
    load_result = _load_validated_config()
    if load_result is None:
        return 1
    # Parameter-driven by design: PRD S8 declares no logging rows yet,
    # so reconfigure() applies its defaults until such rows exist.
    reconfigure()
    log_startup_report(load_result)
    get_logger(__name__).info(
        "starting stdio transport",
        endpoints=len(load_result.model.endpoints),
    )
    anyio.run(serve, guard)
    return 0


if __name__ == "__main__":
    sys.exit(main())
