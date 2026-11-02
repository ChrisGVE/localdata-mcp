"""Refuse a release whose tag and package version disagree (#76).

Both publish workflows fire on a ``v*.*.*`` tag, and both build whatever
version ``pyproject.toml`` declares. Nothing tied the two together, so tagging
``v3.0.0`` on a tree still declaring ``3.0.0.dev0`` would upload a pre-release
that pip and uv never resolve by default, and push a Docker image labelled
with the tag while holding something else. This check runs before either
upload and fails the job instead.

Usage: python scripts/check_release_version.py <git-ref>
A ref that is not a tag (a branch, a manual run) passes: no release is claimed.
Versions are compared under PEP 440, so ``v3.0.0`` matches ``3.0.0``.
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

from packaging.version import Version

TAG_PREFIX = "refs/tags/v"


def main(ref: str) -> int:
    if not ref.startswith(TAG_PREFIX):
        print(f"{ref} is not a release tag; nothing to compare")
        return 0
    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    declared = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"][
        "version"
    ]
    tagged = ref.removeprefix(TAG_PREFIX)
    if Version(tagged) != Version(declared):
        print(
            f"tag v{tagged} does not match pyproject.toml version {declared}; "
            "set project.version (and server.json) to the release before tagging"
        )
        return 1
    print(f"tag v{tagged} matches pyproject.toml version {declared}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
