"""LocalData MCP - A dynamic MCP server for local databases and text files."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _installed_version

try:
    # Read the version from package metadata rather than restating it. The
    # literal that used to live here said 2.0.0 while pyproject.toml, server.json
    # and the plugin manifest all said 2.1.0 -- a second source of truth that had
    # already drifted a whole minor release.
    __version__ = _installed_version("localdata-mcp")
except PackageNotFoundError:  # running from a source tree with nothing installed
    __version__ = "unknown"

__author__ = "Christian C. Berclaz"
__email__ = "christian.berclaz@mac.com"

from .localdata_mcp import DatabaseManager, main

__all__ = ["DatabaseManager", "main"]
