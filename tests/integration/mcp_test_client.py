"""FastMCP in-memory test client for LocalData MCP integration tests.

Keeps a single persistent Client session open so that in-memory SQLite
databases created by connect_database survive across subsequent tool calls.
"""

import asyncio
import json
from typing import Any, Dict

from fastmcp import Client

# Module-level flag to ensure DatabaseManager is only instantiated once.
_tools_registered = False

# Persistent event loop and client — avoids creating/tearing down
# transports that close in-memory SQLite connections.
_loop = asyncio.new_event_loop()
_client = None
_client_ctx = None


def _ensure_tools_registered():
    """Ensure DatabaseManager is instantiated so tools are registered on the server."""
    global _tools_registered
    from localdata_mcp.localdata_mcp import mcp as server

    if not _tools_registered:
        from localdata_mcp.localdata_mcp import DatabaseManager

        DatabaseManager()
        _tools_registered = True
    return server


async def _get_client():
    """Get or create a persistent client session."""
    global _client, _client_ctx
    if _client is None:
        server = _ensure_tools_registered()
        _client_ctx = Client(server)
        _client = await _client_ctx.__aenter__()
    return _client


def _parse_result(result) -> Any:
    """Extract parsed data from a FastMCP CallToolResult."""
    # FastMCP v3 CallToolResult has a .data attribute with parsed content
    if hasattr(result, "data") and result.data is not None:
        data = result.data
        if isinstance(data, str):
            try:
                return json.loads(data)
            except (json.JSONDecodeError, TypeError):
                return data
        return data

    # Fall back to iterating content blocks
    if hasattr(result, "content") and result.content:
        for block in result.content:
            if hasattr(block, "text"):
                try:
                    return json.loads(block.text)
                except (json.JSONDecodeError, TypeError):
                    return block.text

    # Raw iterable fallback
    if result and hasattr(result, "__iter__"):
        for block in result:
            if hasattr(block, "text"):
                try:
                    return json.loads(block.text)
                except (json.JSONDecodeError, TypeError):
                    return block.text

    return result


async def call_tool_async(tool_name: str, arguments: Dict[str, Any]) -> Any:
    """Call an MCP tool and return parsed result."""
    client = await _get_client()
    result = await client.call_tool(tool_name, arguments)
    return _parse_result(result)


def call_tool(tool_name: str, arguments: Dict[str, Any]) -> Any:
    """Synchronous wrapper for MCP tool calls.

    Uses a persistent event loop and client session so in-memory
    SQLite state survives across calls.
    """
    return _loop.run_until_complete(call_tool_async(tool_name, arguments))
