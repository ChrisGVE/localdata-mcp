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

# The v3 process entrypoint lives at localdata_mcp.server.mcp_app:main (the
# pyproject console script points there directly). This package __init__ stays
# deliberately import-light: importing the entrypoint here would pull FastMCP
# and the whole tool layer into every `import localdata_mcp.<subpackage>`, so
# the public `main` is exposed lazily instead.
__all__ = ["main"]


def __getattr__(name: str):  # PEP 562 lazy attribute
    if name == "main":
        from .server.mcp_app import main as _main

        return _main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
