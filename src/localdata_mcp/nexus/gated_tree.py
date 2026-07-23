"""localdata_mcp/nexus/gated_tree.py — the gated v3 tree, one home.

The set of packages the v3 gates cover — the same set v3-ci.yml
gates — declared once so every static check sweeps the same tree
(NFR-402: no second SSOT). Neighbors: config/default_site_check.py
(NFR-403) and observability/purity_check.py (the T2 no-print /
no-second-handler teeth) both scan through this module.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

# The v3 packages under src/localdata_mcp/ that the gates cover.
V3_PACKAGES: tuple[str, ...] = (
    "nexus",
    "ingest",
    "explore",
    "process",
    "visualize",
    "output",
    "testbench",
)

# New-tree modules living beside legacy files in a non-gated package
# (server/ keeps its legacy members until E15 deletes them); each is
# swept individually. Extended as epics add guarded entry points.
GUARDED_EXTRA_FILES: tuple[str, ...] = (
    "server/fd_guard.py",
    "server/mcp_app.py",
    "server/skeleton_tools.py",
    "server/tools_generated.py",
)

# src/localdata_mcp — the root the package names above resolve against.
SRC_ROOT = Path(__file__).resolve().parents[1]


def iter_v3_sources(root: Path | None = None) -> Iterator[Path]:
    """Every gated source file: the v3 packages' modules in stable
    order, then the guarded extra files."""
    base = SRC_ROOT if root is None else root
    for package in V3_PACKAGES:
        yield from sorted((base / package).rglob("*.py"))
    for extra in GUARDED_EXTRA_FILES:
        yield base / extra
