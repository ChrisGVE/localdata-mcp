"""localdata_mcp/ingest/connectors/treepaths.py — dot-path escaping (E8.3).

The harvested successor of `tree_storage/paths.py`: the escaped
dot-path grammar tree-store node addresses use (`a.b\\.c` — a literal
dot in a key escapes as `\\.`). One home at the connectors level: the
kv family auto-creates ancestor nodes on set_value and the graph/tree
family navigates by these paths — both must parse and build the SAME
grammar or a node written by one is unreachable by the other.
Neighbors: kv/tools.py and graph_tree/ consume it; the `nodes.path`
column (store_schemas.py) stores what `build_path` produces.
"""

from __future__ import annotations

from typing import List


def escape_path_segment(name: str) -> str:
    r"""Escape a single path segment so dots and backslashes are literal.

    ``\`` -> ``\\``  then  ``.`` -> ``\.``
    """
    return name.replace("\\", "\\\\").replace(".", "\\.")


def unescape_path_segment(segment: str) -> str:
    r"""Reverse :func:`escape_path_segment`.

    Raises ``ValueError`` on invalid escape sequences.
    """
    result: list[str] = []
    i = 0
    while i < len(segment):
        ch = segment[i]
        if ch == "\\":
            if i + 1 >= len(segment):
                raise ValueError("Trailing backslash in path segment")
            nxt = segment[i + 1]
            if nxt == ".":
                result.append(".")
            elif nxt == "\\":
                result.append("\\")
            else:
                raise ValueError(f"Invalid escape sequence '\\{nxt}' at position {i}")
            i += 2
        else:
            result.append(ch)
            i += 1
    return "".join(result)


def build_path(segments: List[str]) -> str:
    """Join raw key names into an escaped dot-path."""
    if not segments:
        return ""
    return ".".join(escape_path_segment(s) for s in segments)


def parse_path(path: str) -> List[str]:
    """Split an escaped dot-path into raw key names."""
    if not path:
        return []
    segments: list[str] = []
    current: list[str] = []
    i = 0
    while i < len(path):
        ch = path[i]
        if ch == "\\":
            if i + 1 >= len(path):
                raise ValueError("Trailing backslash in path")
            nxt = path[i + 1]
            if nxt == ".":
                current.append(".")
            elif nxt == "\\":
                current.append("\\")
            else:
                raise ValueError(f"Invalid escape sequence '\\{nxt}' at position {i}")
            i += 2
        elif ch == ".":
            segments.append("".join(current))
            current = []
            i += 1
        else:
            current.append(ch)
            i += 1
    segments.append("".join(current))
    return segments
