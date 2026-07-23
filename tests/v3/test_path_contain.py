"""tests/v3/test_path_contain.py — E6.4/NFR-108 acceptance.

Fail-closed empty default, canonical-real-path comparison (symlink
escape refused, symlinked allow-root honored), read and write sides,
and the not-yet-existing write target contained through its resolved
parent chain.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from localdata_mcp.nexus.chokepoint.path_contain import (
    PathRefusedError,
    contain,
)


class TestFailClosedDefault:
    def test_empty_allow_list_refuses_reads_and_writes(self, tmp_path: Path) -> None:
        target = tmp_path / "data.csv"
        target.touch()
        for mode in ("read", "write"):
            with pytest.raises(PathRefusedError, match="fail-closed"):
                contain(target, (), mode=mode)  # type: ignore[arg-type]


class TestContainment:
    def test_contained_path_returns_its_real_path(self, tmp_path: Path) -> None:
        target = tmp_path / "data.csv"
        target.touch()
        assert contain(target, (str(tmp_path),), mode="read") == target.resolve()

    def test_the_root_itself_is_contained(self, tmp_path: Path) -> None:
        assert contain(tmp_path, (str(tmp_path),), mode="read") == tmp_path.resolve()

    def test_outside_path_is_refused_with_mode(self, tmp_path: Path) -> None:
        inside = tmp_path / "inside"
        inside.mkdir()
        outside = tmp_path / "outside.csv"
        outside.touch()
        with pytest.raises(PathRefusedError) as refusal:
            contain(outside, (str(inside),), mode="write")
        assert refusal.value.mode == "write"

    def test_dotdot_traversal_is_canonicalized_before_comparison(
        self, tmp_path: Path
    ) -> None:
        inside = tmp_path / "inside"
        inside.mkdir()
        (tmp_path / "secret.txt").touch()
        sneaky = inside / ".." / "secret.txt"
        with pytest.raises(PathRefusedError):
            contain(sneaky, (str(inside),), mode="read")


class TestSymlinks:
    def test_symlink_escaping_the_tree_is_refused(self, tmp_path: Path) -> None:
        inside = tmp_path / "inside"
        inside.mkdir()
        secret = tmp_path / "secret.txt"
        secret.touch()
        link = inside / "innocent.txt"
        link.symlink_to(secret)
        with pytest.raises(PathRefusedError):
            contain(link, (str(inside),), mode="read")

    def test_symlinked_allow_root_is_canonicalized_too(self, tmp_path: Path) -> None:
        real_root = tmp_path / "real"
        real_root.mkdir()
        target = real_root / "data.csv"
        target.touch()
        alias = tmp_path / "alias"
        alias.symlink_to(real_root)
        assert contain(target, (str(alias),), mode="read") == target.resolve()


class TestWriteSideNonexistentTarget:
    def test_new_file_under_allowed_root_is_contained(self, tmp_path: Path) -> None:
        target = tmp_path / "out" / "export.parquet"  # does not exist yet
        (tmp_path / "out").mkdir()
        assert contain(target, (str(tmp_path),), mode="write") == target.resolve()

    def test_new_file_through_escaping_symlink_dir_is_refused(
        self, tmp_path: Path
    ) -> None:
        inside = tmp_path / "inside"
        inside.mkdir()
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        linked_dir = inside / "sub"
        linked_dir.symlink_to(elsewhere)
        with pytest.raises(PathRefusedError):
            contain(linked_dir / "new.csv", (str(inside),), mode="write")
