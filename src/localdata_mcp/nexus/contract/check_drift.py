"""localdata_mcp/nexus/contract/check_drift.py — the CI-only drift gate.

Regenerates every NX-1 artifact in memory from the live ToolSpec
registry and byte-compares it against the committed file at its
section-9 home (ARCHITECTURE.md section 6.1): any difference — a hand
edit, a stale regeneration, a deleted file — exits non-zero and fails
the build (FR-704's acceptance). Runs in CI exclusively, never on the
startup or request path. Neighbors: generate.py supplies the artifact
map; v3-ci.yml's drift-check job invokes `python -m
localdata_mcp.nexus.contract.check_drift`.
"""

from __future__ import annotations

import sys
from pathlib import Path, PurePosixPath

from localdata_mcp.nexus.contract.generate import (
    REPO_ROOT,
    generate_artifacts,
    loaded_default_registry,
)


def drifted_artifacts(root: Path) -> list[tuple[PurePosixPath, str]]:
    """Every artifact whose committed bytes differ from regeneration.

    Returns `(repo-relative path, reason)` pairs; empty means clean.
    """
    problems: list[tuple[PurePosixPath, str]] = []
    for relpath, expected in generate_artifacts(loaded_default_registry()).items():
        committed = root / relpath
        if not committed.is_file():
            problems.append((relpath, "missing from the tree"))
        elif committed.read_text(encoding="utf-8") != expected:
            problems.append((relpath, "differs from regeneration"))
    return problems


def main() -> int:
    """Exit 0 on a clean tree, 1 with one line per drifted artifact."""
    problems = drifted_artifacts(REPO_ROOT)
    for relpath, reason in problems:
        sys.stderr.write(
            f"DRIFT: {relpath} {reason} - regenerate via "
            "localdata_mcp.nexus.contract.generate, never hand-edit\n"
        )
    if not problems:
        sys.stderr.write("drift check clean: every artifact matches regeneration\n")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
