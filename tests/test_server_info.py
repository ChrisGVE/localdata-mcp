"""What the server says about itself in the MCP handshake.

An MCP client displays and logs ``serverInfo`` from ``initialize``. The version
there has to be this project's, not the version of the framework it is built
on: a client operator reading the framework's number against this repository
has no way to tell it is a different package's (#83).

fastmcp 4 answers two handshakes: the classic ``initialize`` and the sessionless
2026-07-28 protocol, which has no ``initialize`` at all and reports the server
through a discovery result instead. A client of either era must read this
project's version, so on fastmcp 4 both are checked; fastmcp 3 has only the
classic one.
"""

import asyncio
import inspect

from fastmcp import Client

import localdata_mcp
from localdata_mcp import server as server_module


def _handshakes() -> list[dict]:
    """Client options selecting each handshake the installed fastmcp offers."""
    if "mode" in inspect.signature(Client).parameters:
        return [{"mode": "legacy"}, {"mode": "auto"}]
    return [{}]


def _server_info(**client_options):
    async def _run():
        async with Client(server_module.mcp, **client_options) as client:
            # fastmcp 4 reads the server's identity era-neutrally; fastmcp 3
            # has only the initialize result.
            info = getattr(client, "server_info", None)
            return info if info is not None else client.initialize_result.serverInfo

    return asyncio.run(_run())


def test_the_handshake_reports_this_projects_version():
    handshakes = _handshakes()
    assert handshakes, "no handshake to check"

    for client_options in handshakes:
        info = _server_info(**client_options)

        assert info.name == "localdata", client_options
        assert info.version == localdata_mcp.__version__, client_options
