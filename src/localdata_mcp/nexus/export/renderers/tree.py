"""renderers/tree.py — a nested mapping as a Markdown outline (FR-902).

The harvested `tree_export`/`markdown_export` tree design over plain
data: a nested mapping renders as an indented bullet outline — branch
keys as bullets, leaves as `key: value` lines, list members as items.
v3's tree tools (E8.3) reconstruct their nested mapping and hand it
here; NX-8 never reads tree storage itself.
"""

from __future__ import annotations

from typing import Any, Mapping

from ..interface import ExportError

FORMAT = "tree"


def render(payload: Any) -> bytes:
    if not isinstance(payload, Mapping):
        raise ExportError(
            f"tree payload must be a nested mapping, got {type(payload).__name__}"
        )
    lines: list[str] = []
    _emit(payload, 0, lines)
    return ("\n".join(lines) + "\n").encode("utf-8")


def _emit(node: Mapping[str, Any], depth: int, lines: list[str]) -> None:
    indent = "  " * depth
    for key, value in node.items():
        if isinstance(value, Mapping):
            lines.append(f"{indent}- **{key}**")
            _emit(value, depth + 1, lines)
        elif isinstance(value, (list, tuple)):
            lines.append(f"{indent}- **{key}**")
            for index, member in enumerate(value):
                if isinstance(member, Mapping):
                    lines.append(f"{indent}  - [{index}]")
                    _emit(member, depth + 2, lines)
                else:
                    lines.append(f"{indent}  - {member}")
        else:
            lines.append(f"{indent}- {key}: {value}")
