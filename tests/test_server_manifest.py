"""The MCP registry manifest, ``server.json``, is valid and names this package (#78).

A registry client reads ``server.json`` to find out where to fetch the server:
``packages[0]`` names a PyPI ``identifier`` and ``version``, and the client
installs exactly that. So the version there has a job a reader's version does
not — it must name the artifact a release of this tree publishes — and the
only version this tree publishes is the one ``pyproject.toml`` declares.
Before #78 the field said ``2.1.0``, which was never released (PyPI answers
404), while the package said ``3.0.0.dev0``.

The registry also rejects a manifest that fails its schema (the description is
capped at 100 characters; it once held 172). The schema is vendored under
``tests/assets/server.schema.json`` at the URL ``server.json`` declares, so
this check runs offline and changes only when the declared schema does.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import jsonschema
from packaging.version import Version

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised on 3.10 only
    import tomli as tomllib

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = json.loads((ROOT / "server.json").read_text(encoding="utf-8"))
SCHEMA = json.loads(
    (ROOT / "tests" / "assets" / "server.schema.json").read_text(encoding="utf-8")
)
PROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
    "project"
]


def test_the_vendored_schema_is_the_one_the_manifest_declares():
    assert SCHEMA["$id"] == MANIFEST["$schema"]


def test_the_manifest_satisfies_the_registry_schema():
    errors = [
        f"{'/'.join(map(str, e.absolute_path)) or '<root>'}: {e.message}"
        for e in jsonschema.Draft7Validator(SCHEMA).iter_errors(MANIFEST)
    ]
    assert errors == []


def test_the_manifest_names_the_version_pyproject_declares():
    declared = Version(PROJECT["version"])
    assert Version(MANIFEST["version"]) == declared
    assert [Version(p["version"]) for p in MANIFEST["packages"]] == [declared]


def test_the_manifest_names_this_pypi_package():
    assert [(p["registryType"], p["identifier"]) for p in MANIFEST["packages"]] == [
        ("pypi", PROJECT["name"])
    ]
