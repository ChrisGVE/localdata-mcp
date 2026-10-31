"""What the server says about itself in the MCP handshake.

An MCP client displays and logs ``serverInfo`` from ``initialize``. The version
there has to be this project's, not the version of the framework it is built
on: a client operator reading the framework's number against this repository
has no way to tell it is a different package's (#83).
"""

import asyncio

from fastmcp import Client

import localdata_mcp
from localdata_mcp import server as server_module


def _server_info():
    async def _run():
        async with Client(server_module.mcp) as client:
            return client.initialize_result.serverInfo

    return asyncio.run(_run())


def test_the_handshake_reports_this_projects_version():
    info = _server_info()

    assert info.name == "localdata"
    assert info.version == localdata_mcp.__version__
