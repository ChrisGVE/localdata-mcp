"""tests/v3/test_persistence_import_graph.py — FR-105/801/802 checks.

The E5 exit gate's static leg, over the whole gated v3 tree
(gated_tree.py — one home for what the gates sweep):

- FR-105/801 sole-owner: engine-creating capability (`create_engine`,
  `duckdb`) appears ONLY inside nexus/persistence/ — the god-class's
  bare connection dict cannot recur if no other module can mint a
  connection.
- FR-802/§6.2 reachability: `localdata_mcp.nexus.persistence` is
  importable by NX-6 (nexus/chokepoint/) exclusively — plus NX-5
  itself. No tool module can obtain a live connection by import.
- Revival, not reuse: the legacy `connection_manager/` stack is
  imported by NO v3 module — E5.1 harvested its design, not its code.
"""

from __future__ import annotations

import ast
from pathlib import Path

from localdata_mcp.nexus.gated_tree import SRC_ROOT, iter_v3_sources

_PERSISTENCE_DIR = SRC_ROOT / "nexus" / "persistence"
_CHOKEPOINT_DIR = SRC_ROOT / "nexus" / "chokepoint"

_PERSISTENCE_MODULE = "localdata_mcp.nexus.persistence"
_LEGACY_STACK = "localdata_mcp.connection_manager"


def _imports_of(path: Path) -> list[str]:
    """Every imported module name in `path`, absolute form."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return names


def _imported_symbols_of(path: Path) -> list[tuple[str, str]]:
    """(module, symbol) pairs of every from-import in `path`."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    pairs: list[tuple[str, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            pairs.extend((node.module, alias.name) for alias in node.names)
    return pairs


class TestSoleConnectionOwner:
    """FR-105/801: only NX-5 can mint an engine."""

    def test_create_engine_is_imported_only_inside_persistence(self) -> None:
        offenders = [
            f"{path}: from {module} import {symbol}"
            for path in iter_v3_sources()
            if not path.is_relative_to(_PERSISTENCE_DIR)
            for module, symbol in _imported_symbols_of(path)
            if module.startswith("sqlalchemy") and symbol == "create_engine"
        ]
        assert offenders == [], offenders

    def test_duckdb_is_imported_only_inside_persistence(self) -> None:
        offenders = [
            f"{path}: {name}"
            for path in iter_v3_sources()
            if not path.is_relative_to(_PERSISTENCE_DIR)
            for name in _imports_of(path)
            if name == "duckdb" or name.startswith("duckdb.")
        ]
        assert offenders == [], offenders


class TestReachability:
    """FR-802/§6.2: NX-5 is reachable by NX-6 exclusively."""

    def test_persistence_is_imported_by_chokepoint_and_itself_only(self) -> None:
        offenders = [
            f"{path}: {name}"
            for path in iter_v3_sources()
            if not (
                path.is_relative_to(_PERSISTENCE_DIR)
                or path.is_relative_to(_CHOKEPOINT_DIR)
            )
            for name in _imports_of(path)
            if name.startswith(_PERSISTENCE_MODULE)
        ]
        assert offenders == [], offenders


class TestRevivalNotReuse:
    def test_no_v3_module_imports_the_legacy_connection_stack(self) -> None:
        offenders = [
            f"{path}: {name}"
            for path in iter_v3_sources()
            for name in _imports_of(path)
            if name.startswith(_LEGACY_STACK)
        ]
        assert offenders == [], offenders
