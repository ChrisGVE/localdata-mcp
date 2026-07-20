"""Hold every file that states the package version to the same number.

`src/localdata_mcp/__init__.py` carried its own `__version__ = "2.0.0"` while
`pyproject.toml`, `server.json` and `.claude-plugin/plugin.json` all said 2.1.0.
Nothing compared them, so the literal drifted an entire minor release and would
have shipped that way.

The version now comes from package metadata, so it cannot restate anything. What
these tests guard is the set of files that must state it literally: a manifest
cannot read `importlib.metadata`, so each one is checked against `pyproject.toml`
rather than against an installed distribution, whose metadata goes stale in an
editable checkout.
"""

import json
import os
import re

import pytest

ROOT = os.path.dirname(os.path.dirname(__file__))


def _read(*parts: str) -> str:
    with open(os.path.join(ROOT, *parts), encoding="utf-8") as handle:
        return handle.read()


@pytest.fixture(scope="module")
def declared_version() -> str:
    """The version in pyproject.toml — the one the build publishes."""
    match = re.search(
        r'^version\s*=\s*"([^"]+)"', _read("pyproject.toml"), re.MULTILINE
    )
    assert match, "pyproject.toml states no version"
    return match.group(1)


def test_server_manifest_matches(declared_version):
    manifest = json.loads(_read("server.json"))

    assert manifest["version"] == declared_version


def test_server_manifest_package_entries_match(declared_version):
    """The registry manifest repeats the version per package entry."""
    manifest = json.loads(_read("server.json"))

    for package in manifest.get("packages", []):
        assert package.get("version") == declared_version


def test_plugin_manifest_matches(declared_version):
    manifest = json.loads(_read(".claude-plugin", "plugin.json"))

    assert manifest["version"] == declared_version


def test_changelog_documents_the_version_being_shipped(declared_version):
    """A release whose version has no changelog entry ships undocumented."""
    entries = re.findall(r"^## \[([^\]]+)\]", _read("CHANGELOG.md"), re.MULTILINE)

    assert entries, "CHANGELOG.md has no version entries"
    assert entries[0] == declared_version, (
        f"newest CHANGELOG entry is {entries[0]}, but pyproject.toml "
        f"declares {declared_version}"
    )


def test_package_does_not_restate_the_version(declared_version):
    """`__version__` must be derived, not typed — that is how it drifted before."""
    source = _read("src", "localdata_mcp", "__init__.py")

    assert not re.search(r'^__version__\s*=\s*"[0-9]', source, re.MULTILINE), (
        "__init__.py hardcodes a version literal again; derive it from "
        "importlib.metadata so it cannot disagree with pyproject.toml"
    )
