"""LocalData MCP server sub-package.

v3 members (the new tree, gated by v3-ci):
  - ``fd_guard`` -- the NFR-303 fd-1 guard (PRD S5.3)
  - ``mcp_app``  -- the v3 process entrypoint (ARCHITECTURE.md section 4e)

Legacy v2 members, kept beside the new tree until E15 deletes them:
  - ``database_manager`` -- QueryBuffer dataclass and DatabaseManager class
  - ``query_execution``  -- Memory-aware query execution decision logic
  - ``cli``              -- CLI argument parsing and legacy ``main()``

The legacy import chain reaches dependencies the v3 manifest removed
(the same situation the top-level ``localdata_mcp/__init__.py``
guards): a hard import here would make ``python -m
localdata_mcp.server.mcp_app`` unrunnable. Legacy stays importable
where its dependencies happen to exist, and is skipped otherwise.
Legacy module-level state (MCP instance, feature flags, logging
singletons) remains in ``localdata_mcp.localdata_mcp`` so existing
``unittest.mock.patch`` targets continue to work.
"""

try:
    from .cli import _get_version, _parse_cli_args, main  # noqa: F401
    from .database_manager import DatabaseManager, QueryBuffer  # noqa: F401

    __all__ = [
        "DatabaseManager",
        "QueryBuffer",
        "main",
        "_get_version",
        "_parse_cli_args",
    ]
except ImportError:  # legacy dependencies absent under the v3 manifest
    __all__ = []
