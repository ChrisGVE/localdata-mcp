#!/usr/bin/env python3
"""scripts/check_pin_drift.py — CI gate asserting exact pins match the lock.

The v3 manifest (ARCHITECTURE.md 7.1) pins the security-critical libraries
sqlglot and asteval with exact `==` versions: their allow-list semantics are
only as stable as the parser they are defined against. This gate fails the
build when either entry in pyproject.toml is not an exact pin, or when
uv.lock resolved a different version than the pin says.

Run: python3 scripts/check_pin_drift.py [--pyproject PATH --lock PATH]
Requires Python >= 3.11 (tomllib).
"""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
from pathlib import Path

# The exact-pin set per ARCHITECTURE.md 7.1 — extend here if a future
# decision pins another parser-stability-critical library.
EXACT_PINNED_PACKAGES = ("sqlglot", "asteval")

_EXACT_PIN_PATTERN = re.compile(
    r"^(?P<name>[A-Za-z0-9._-]+)==(?P<version>[^=<>!~,;\s]+)$"
)


class PinDriftError(Exception):
    """A pin is missing, not exact, or disagrees with the lock."""


def find_requirement(dependencies: list[str], package: str) -> str:
    """Return the requirement string declaring `package`, or refuse."""
    for requirement in dependencies:
        name = re.split(r"[=<>!~\[;\s]", requirement.strip(), maxsplit=1)[0]
        if name.lower() == package.lower():
            return requirement.strip()
    raise PinDriftError(f"{package}: no entry in [project.dependencies]")


def parse_exact_pin(requirement: str, package: str) -> str:
    """Return the version of an exact `name==version` pin, or refuse."""
    match = _EXACT_PIN_PATTERN.match(requirement)
    if match is None or match.group("name").lower() != package.lower():
        raise PinDriftError(
            f"{package}: requirement {requirement!r} is not an exact == pin"
        )
    return match.group("version")


def locked_version(lock_document: dict, package: str) -> str:
    """Return the resolved version of `package` in a parsed uv.lock, or refuse."""
    for entry in lock_document.get("package", []):
        if entry.get("name", "").lower() == package.lower():
            version = entry.get("version")
            if not version:
                raise PinDriftError(f"{package}: uv.lock entry has no version")
            return str(version)
    raise PinDriftError(f"{package}: not present in uv.lock")


def assert_pin_matches_lock(
    dependencies: list[str], lock_document: dict, package: str
) -> None:
    """Refuse unless `package` is exact-pinned and the lock agrees."""
    pinned = parse_exact_pin(find_requirement(dependencies, package), package)
    locked = locked_version(lock_document, package)
    if pinned != locked:
        raise PinDriftError(
            f"{package}: pyproject pins =={pinned} but uv.lock resolved {locked}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pyproject", type=Path, default=Path("pyproject.toml"))
    parser.add_argument("--lock", type=Path, default=Path("uv.lock"))
    args = parser.parse_args(argv)

    pyproject = tomllib.loads(args.pyproject.read_text(encoding="utf-8"))
    lock_document = tomllib.loads(args.lock.read_text(encoding="utf-8"))
    dependencies = pyproject["project"]["dependencies"]

    for package in EXACT_PINNED_PACKAGES:
        try:
            assert_pin_matches_lock(dependencies, lock_document, package)
        except PinDriftError as error:
            print(f"PIN DRIFT: {error}", file=sys.stderr)
            return 1
        print(f"ok: {package} exact pin agrees with uv.lock")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
