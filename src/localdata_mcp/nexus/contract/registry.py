"""localdata_mcp/nexus/contract/registry.py — the ToolSpec registry.

Holds every registered ToolSpec (spec.py) in registration order and is
the one source the generators (generators/), the drift check
(check_drift.py), and the composition engine's shape registry consult.
Duplicate names are refused — the one-declaration model admits exactly
one spec per tool (ARCHITECTURE.md section 6.1). A process-wide default
registry backs @tool_spec's registration; tests construct their own.
"""

from __future__ import annotations

from collections.abc import Iterator

from localdata_mcp.nexus.contract.errors import (
    DuplicateToolNameError,
    UnknownToolError,
)
from localdata_mcp.nexus.contract.spec import ToolSpec


class ToolRegistry:
    """Ordered name -> ToolSpec mapping with duplicate refusal."""

    def __init__(self) -> None:
        self._specs: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._specs:
            raise DuplicateToolNameError(
                f"tool {spec.name!r} is already registered", tool_name=spec.name
            )
        self._specs[spec.name] = spec

    def lookup(self, name: str) -> ToolSpec:
        try:
            return self._specs[name]
        except KeyError:
            raise UnknownToolError(
                f"no registered tool named {name!r}", tool_name=name
            ) from None

    def __iter__(self) -> Iterator[ToolSpec]:
        return iter(self._specs.values())

    def __len__(self) -> int:
        return len(self._specs)


_DEFAULT_REGISTRY = ToolRegistry()


def default_registry() -> ToolRegistry:
    """The process-wide registry @tool_spec registers into by default."""
    return _DEFAULT_REGISTRY
