"""LocalData MCP server sub-package.

The implementation is split across:
  - ``database_manager`` -- QueryBuffer dataclass and DatabaseManager class
  - ``cli``              -- CLI argument parsing and ``main()`` entry point

Module-level state (MCP instance, feature flags, logging singletons) remains
in ``localdata_mcp.localdata_mcp`` so that existing ``unittest.mock.patch``
targets continue to work.
"""

from .database_manager import DatabaseManager, QueryBuffer  # noqa: F401
from .cli import main, _get_version, _parse_cli_args  # noqa: F401

__all__ = [
    "DatabaseManager",
    "QueryBuffer",
    "main",
    "_get_version",
    "_parse_cli_args",
]
