"""The plugin manifests agree with each other and with the package they launch.

The repository is a plugin for two harnesses at once, and each reads its own
manifest: Claude Code reads ``.claude-plugin/plugin.json``, Codex reads
``.codex-plugin/plugin.json``. They share one marketplace file,
``.claude-plugin/marketplace.json``, which Codex reads as its legacy-compatible
location. Nothing at runtime compares them, so a version bumped in
one manifest and not the other, a skill moved one directory too deep, or a
launch command naming a script the package does not install would each ship
silently and fail only on somebody else's machine. These tests are that
comparison.

They read files and start nothing, so they run without any harness installed.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest
from packaging.version import Version

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised only on 3.10
    import tomli as tomllib

ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
PROJECT = PYPROJECT["project"]

# One manifest per harness. Both are read the same way below, which is the point.
MANIFESTS = {
    "claude": ROOT / ".claude-plugin" / "plugin.json",
    "codex": ROOT / ".codex-plugin" / "plugin.json",
}
CLAUDE_MARKETPLACE = ROOT / ".claude-plugin" / "marketplace.json"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("harness", sorted(MANIFESTS))
def test_each_manifest_names_the_package(harness):
    assert load(MANIFESTS[harness])["name"] == PROJECT["name"]


@pytest.mark.parametrize("harness", sorted(MANIFESTS))
def test_each_manifest_carries_the_package_version(harness):
    """The manifests spell it the semver way, pyproject the PEP 440 way.

    ``3.0.0-dev`` and ``3.0.0.dev0`` are the same version to PEP 440, which is
    the comparison made here, so neither file has to adopt the other's spelling
    and a bump that misses one of them still fails.
    """
    declared = load(MANIFESTS[harness])["version"]
    assert Version(declared) == Version(PROJECT["version"]), (
        f"{MANIFESTS[harness].relative_to(ROOT)} says {declared}, "
        f"pyproject.toml says {PROJECT['version']}"
    )


def test_both_harnesses_launch_the_same_server_the_same_way():
    servers = {harness: load(path)["mcpServers"] for harness, path in MANIFESTS.items()}
    assert servers["claude"] == servers["codex"], servers


def test_the_launch_command_runs_the_console_script_the_package_installs():
    """``uvx localdata-mcp`` finds a script only because it shares the package name.

    uvx resolves its argument as a package and runs the executable of the same
    name, so the script declared in pyproject and the package it belongs to must
    both be called what the manifest says.
    """
    (server,) = load(MANIFESTS["claude"])["mcpServers"].values()
    assert server["command"] == "uvx"
    assert server["args"] == [PROJECT["name"]]
    assert PROJECT["name"] in PROJECT["scripts"]


def test_there_is_no_portable_mcp_json_at_the_root():
    """The Agent Plugins ``mcp.json`` would start the server where it cannot work.

    That standard starts a stdio server in the plugin's own directory and offers
    no placeholder for the user's project. This server confines every path to
    its working directory, so under Codex, which honours the standard, it came
    up inside the plugin cache and refused every file the user pointed at. The
    harness-specific manifests start it in the session's directory instead. A
    root ``plugin.json`` matters as much: Codex prefers it to ``.codex-plugin/``
    and then reads servers only from ``mcp.json``, so either file arriving
    brings the failure back rather than adding an option.
    """
    assert not (ROOT / "mcp.json").exists()
    assert not (ROOT / "plugin.json").exists()


def test_the_marketplace_lists_this_plugin_at_the_repository_root():
    """The marketplace file sits in the repository that is also the plugin."""
    (entry,) = load(CLAUDE_MARKETPLACE)["plugins"]
    assert entry["name"] == PROJECT["name"]

    assert entry["source"] == "./"


def test_every_skill_sits_one_level_under_skills_and_is_named_for_its_directory():
    """Both harnesses look exactly one directory down, and match name to directory.

    A ``SKILL.md`` any deeper is not an error anywhere — it is simply never
    found, which is what would have happened to the skill at its old,
    deeper path.
    """
    skills = ROOT / "skills"
    found = sorted(skills.rglob("SKILL.md"))
    assert found, "no skill ships"

    for path in found:
        assert (
            path.parent.parent == skills
        ), f"{path.relative_to(ROOT)} is nested too deep"
        frontmatter = re.match(
            r"^---\n(.*?)\n---\n", path.read_text(encoding="utf-8"), re.S
        )
        assert frontmatter is not None, f"{path.relative_to(ROOT)} has no frontmatter"
        name = re.search(r"^name:\s*(\S+)\s*$", frontmatter.group(1), re.M)
        assert name is not None and name.group(1) == path.parent.name, path


def test_the_codex_manifest_points_at_the_skills_directory():
    assert load(MANIFESTS["codex"])["skills"] == "./skills/"
