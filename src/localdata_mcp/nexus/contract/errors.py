"""localdata_mcp/nexus/contract/errors.py — NX-1's typed refusal taxonomy.

One class per refusal path, mirroring nexus/config/errors.py's pattern.
Neighbors: spec.py raises SpecValidationError on a malformed ToolSpec;
registry.py raises DuplicateToolNameError / UnknownToolError; the
compatibility table (compatibility.py) and drift check (check_drift.py)
report through return values, not exceptions — refusals here are
authoring-time defects, never runtime data errors. NX-3 (E4) will
absorb these into the one error model, matching the config precedent.
"""

from __future__ import annotations


class ToolContractError(Exception):
    """Base of every NX-1 refusal; carries the offending tool name."""

    def __init__(self, message: str, *, tool_name: str | None = None) -> None:
        super().__init__(message)
        self.tool_name = tool_name


class SpecValidationError(ToolContractError):
    """A ToolSpec declaration is malformed (empty name/summary, a
    params entry that is not a Param)."""


class DuplicateToolNameError(ToolContractError):
    """A second ToolSpec tried to claim an already-registered name —
    the one-declaration model admits exactly one spec per tool."""


class UnknownToolError(ToolContractError):
    """A lookup named a tool no ToolSpec has declared."""
