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

# The legacy v2 server is scaffolded beside the v3 tree until E15 deletes it.
# Its import chain reaches dependencies the v3 manifest removed, so a hard
# import here would break every v3 subpackage (importing localdata_mcp.nexus
# first executes this file). Legacy stays importable where its dependencies
# happen to exist, and is skipped otherwise.
try:
    from .localdata_mcp import DatabaseManager, main

    __all__ = ["DatabaseManager", "main"]
except ImportError:  # legacy dependencies absent under the v3 manifest
    __all__ = []
