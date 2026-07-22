"""tests/v3/test_check_drift.py — E3.5 drift-gate behavior.

The gate is exercised three ways: the real tree is clean (the epic's
own regenerate-and-commit discipline), a tampered copy is caught (the
FR-704 hand-edit demonstration), and a missing artifact is caught. The
tamper cases run against a scratch copy of the artifact set — the real
tree is never modified.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from localdata_mcp.nexus.contract.check_drift import (
    REPO_ROOT,
    drifted_artifacts,
)
from localdata_mcp.nexus.contract.generate import (
    generate_artifacts,
    loaded_default_registry,
)


def _scratch_tree(tmp_path: Path) -> Path:
    """Copy every committed artifact into a scratch root."""
    for relpath in generate_artifacts(loaded_default_registry()):
        source = REPO_ROOT / relpath
        target = tmp_path / relpath
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    return tmp_path


class TestCleanTree:
    def test_the_real_tree_has_no_drift(self) -> None:
        assert drifted_artifacts(REPO_ROOT) == []


class TestTamperDetection:
    def test_a_hand_edit_is_caught(self, tmp_path: Path) -> None:
        root = _scratch_tree(tmp_path)
        wrapper = root / "src/localdata_mcp/server/tools_generated.py"
        wrapper.write_text(
            wrapper.read_text(encoding="utf-8") + "\n# hand edit\n",
            encoding="utf-8",
        )
        problems = drifted_artifacts(root)
        assert [(str(p), reason) for p, reason in problems] == [
            (
                "src/localdata_mcp/server/tools_generated.py",
                "differs from regeneration",
            )
        ]

    def test_a_deleted_artifact_is_caught(self, tmp_path: Path) -> None:
        root = _scratch_tree(tmp_path)
        (root / "src/localdata_mcp/nexus/contract/generated_shapes.py").unlink()
        problems = drifted_artifacts(root)
        assert ("missing from the tree") in [reason for _, reason in problems]
