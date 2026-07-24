"""LocalData MCP server sub-package (v3).

Members (the gated v3 tree, per nexus/gated_tree.py):
  - ``fd_guard``        -- the NFR-303 fd-1 guard (PRD S5.3)
  - ``mcp_app``         -- the v3 process entrypoint (ARCHITECTURE.md section 4e)
  - ``skeleton_tools``  -- the walking-skeleton ToolSpec declarations
  - ``tools_generated`` -- the NX-1 generated tool wrappers (artifact 1)

Import-light on purpose: ``mcp_app`` pulls FastMCP and the whole tool
layer, so it is imported by callers explicitly, never at package import.
"""
