"""Shared utilities for test file generators."""

from __future__ import annotations

import os
from typing import TypeAlias

RowCounts: TypeAlias = dict[str, int]


def ensure_dir(path: str) -> str:
    """Create directory if it does not exist and return the path."""
    os.makedirs(path, exist_ok=True)
    return path


def sub_dir(output_dir: str, name: str) -> str:
    """Return an ensured subdirectory under output_dir."""
    return ensure_dir(os.path.join(output_dir, name))


def write_text(path: str, content: str) -> str:
    """Write text content to a file and return the path."""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(content)
    return path
