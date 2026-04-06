"""LocalData MCP - A dynamic MCP server for local databases and text files."""

__version__ = "2.0.0"
__author__ = "Christian C. Berclaz"
__email__ = "christian.berclaz@mac.com"

from .localdata_mcp import DatabaseManager, main

__all__ = ["DatabaseManager", "main"]
