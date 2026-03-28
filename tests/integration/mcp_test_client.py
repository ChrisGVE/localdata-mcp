"""FastMCP in-memory test client for LocalData MCP integration tests."""

import asyncio
import json
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastmcp import Client

# Module-level flag to ensure DatabaseManager is only instantiated once.
_tools_registered = False


def _ensure_tools_registered():
    """Ensure DatabaseManager is instantiated so tools are registered on the server."""
    global _tools_registered
    from localdata_mcp.localdata_mcp import mcp as server

    if not _tools_registered:
        from localdata_mcp.localdata_mcp import DatabaseManager

        DatabaseManager()
        _tools_registered = True
    return server


@asynccontextmanager
async def mcp_client():
    """Create an in-memory MCP client connected to LocalData server."""
    server = _ensure_tools_registered()
    async with Client(server) as client:
        yield client


async def call_tool_async(tool_name: str, arguments: Dict[str, Any]) -> Any:
    """Call an MCP tool and return parsed result.

    Handles FastMCP v3 CallToolResult which may have .data, .content, or
    text content blocks.
    """
    async with mcp_client() as client:
        result = await client.call_tool(tool_name, arguments)

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


def call_tool(tool_name: str, arguments: Dict[str, Any]) -> Any:
    """Synchronous wrapper for MCP tool calls."""
    return asyncio.run(call_tool_async(tool_name, arguments))
