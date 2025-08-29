"""LocalData MCP - A dynamic MCP server for local databases and text files."""

__version__ = "1.0.0"
__author__ = "Christian C. Berclaz"
__email__ = "christian@berclaz.org"

from .localdata_mcp import DatabaseManager, main

__all__ = ["DatabaseManager", "main"]
