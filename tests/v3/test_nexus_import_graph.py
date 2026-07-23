"""tests/v3/test_nexus_import_graph.py — the eight-nexus reachability gate (E6.8).

FR-105/NFR-103's static leg generalized to ALL EIGHT nexuses (§6.2,
§8): every data-touching module reaches a backend ONLY through NX-6's
guard, and each nexus's internals stay internal. The per-nexus policy
is declared here as data over the gated tree (gated_tree.py — the same
sweep every static gate uses):

- **NX-1 Contract** — tool packages import only the declaration seam
  (`contract.spec`, the `@tool_spec` decorator the walking skeleton
  established); the registry/generator pipeline is server/NX-internal.
- **NX-2 Config** — free: the one config surface every nexus reads
  (its own teeth are default_site_check.py / merge gating) — except in
  tool packages, whose operator-tunable numbers arrive through the
  nexus seams, never by reading NX-2 directly.
- **NX-3 Error** — `wire` (NX3.wrap), `model` (the wire shape), and
  `fault_signal` (the E4.0 protocol NX-5 implements) are seams;
  translate/redact are error-internal.
- **NX-4 Observability** — the `manager` logging seam is free (§4b);
  purity enforcement is its own gate (purity_check).
- **NX-5 Persistence** — reachable by NX-6 exclusively; engine-minting
  capability confined to it. Asserted in
  test_persistence_import_graph.py (the E5 gate this file
  generalizes) — not restated here (one SSOT per rule, NFR-402).
- **NX-6 Chokepoint** — `guard` is THE seam; sql_validate, the
  registry, bounds, containment, expr_eval are chokepoint-internal
  (importing them directly would bypass the entrypoint screens).
- **NX-7 Response / NX-8 Export** — seam packages by §6.2
  (`shape_envelope`, `render`): importable as declared.

The dynamic leg (the bypass battery static analysis cannot prove) is
E8+ security-battery work; the E4.0 contract against the REAL
ConnectionRecord is already green in test_persistence_manager.py.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterator

from localdata_mcp.nexus.gated_tree import SRC_ROOT, iter_v3_sources

_NEXUS_PREFIX = "localdata_mcp.nexus"

# The data-touching capability packages §6.2 constrains: the four
# domain packages plus the E13 Output capability (export_result reaches
# the NX-8 export seam, never a renderer or a backend directly).
_TOOL_PACKAGES = ("ingest", "explore", "process", "visualize", "output")

# What a tool module may import from the nexus tree — §6.2's declared
# seam set plus the two established cross-cutting seams (NX-1 spec
# declaration, NX-4 logging).
_TOOL_PERMITTED = (
    "localdata_mcp.nexus.chokepoint.guard",
    "localdata_mcp.nexus.contract.spec",
    "localdata_mcp.nexus.error.model",
    "localdata_mcp.nexus.error.wire",
    "localdata_mcp.nexus.export",
    "localdata_mcp.nexus.observability.manager",
    "localdata_mcp.nexus.response",
)

# The composition engine's scoped widening (§6.3): NX-1 owns BOTH
# halves of the FR-606 compatibility mechanism — the ToolSpec registry
# and the declared adjacency table — and the composition engine is
# their one declared consumer among tool packages. Only modules under
# process/composition/ may import these two seams; contract.errors
# rides along (the registry's typed refusals cross its lookups).
_COMPOSITION_HOME = SRC_ROOT / "process" / "composition"
_COMPOSITION_PERMITTED = _TOOL_PERMITTED + (
    "localdata_mcp.nexus.contract.compatibility",
    "localdata_mcp.nexus.contract.errors",
    "localdata_mcp.nexus.contract.registry",
)

# Nexus-internal module sets: importing these from OUTSIDE the owning
# package is a reachability violation (the seam is the package's other
# modules or a named entry above).
_CHOKEPOINT_INTERNAL_HOME = SRC_ROOT / "nexus" / "chokepoint"
_CHOKEPOINT_SEAM = "localdata_mcp.nexus.chokepoint.guard"
_ERROR_INTERNAL_HOME = SRC_ROOT / "nexus" / "error"
_ERROR_SEAMS = frozenset({"wire", "model", "fault_signal"})


def _module_imports(path: Path) -> Iterator[tuple[str, str | None]]:
    """(module, symbol) for every import in `path` — plain imports
    yield (name, None); from-imports yield one pair per symbol."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name, None
        elif isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                yield node.module, alias.name


def _tool_sources() -> Iterator[Path]:
    for path in iter_v3_sources():
        if any(path.is_relative_to(SRC_ROOT / package) for package in _TOOL_PACKAGES):
            yield path


def _resolved(module: str, symbol: str | None) -> str:
    """The effective imported module path: `from pkg import name`
    reaches `pkg.name` when `name` is a submodule — resolve the pair to
    its most specific form for prefix matching."""
    return f"{module}.{symbol}" if symbol else module


class TestToolPackagesImportOnlyTheDeclaredSeams:
    """§6.2: the set domain/Ingest/Explore/Visualize modules may import."""

    def test_nexus_imports_are_on_the_seam_list(self) -> None:
        offenders = [
            f"{path}: {module}" + (f" ({symbol})" if symbol else "")
            for path in _tool_sources()
            for module, symbol in _module_imports(path)
            if module.startswith(_NEXUS_PREFIX)
            and not any(
                _resolved(module, symbol).startswith(permitted)
                or module.startswith(permitted)
                for permitted in (
                    _COMPOSITION_PERMITTED
                    if path.is_relative_to(_COMPOSITION_HOME)
                    else _TOOL_PERMITTED
                )
            )
        ]
        assert offenders == [], offenders

    def test_no_sqlalchemy_execution_api_in_tool_packages(self) -> None:
        """§8 NX-6's rejected re-implementation, made static: no tool
        module imports SQLAlchemy at all — data access is the guard's."""
        offenders = [
            f"{path}: {module}"
            for path in _tool_sources()
            for module, _symbol in _module_imports(path)
            if module == "sqlalchemy" or module.startswith("sqlalchemy.")
        ]
        assert offenders == [], offenders


class TestChokepointInternalsStayInternal:
    """NX-6: guard.py is the seam; a direct import of the validators,
    registry, or bounds from outside would bypass the entrypoint
    screens the seam exists to force."""

    def test_only_guard_is_imported_from_outside(self) -> None:
        offenders = [
            f"{path}: {module}" + (f" ({symbol})" if symbol else "")
            for path in iter_v3_sources()
            if not path.is_relative_to(_CHOKEPOINT_INTERNAL_HOME)
            for module, symbol in _module_imports(path)
            if module.startswith("localdata_mcp.nexus.chokepoint")
            and not _resolved(module, symbol).startswith(_CHOKEPOINT_SEAM)
        ]
        assert offenders == [], offenders


class TestErrorInternalsStayInternal:
    """NX-3: wrap/model/fault_signal are seams (§6.2, E4.0); the
    translate feeder and redactor are wire-internal steps."""

    def test_only_the_seam_modules_are_imported_from_outside(self) -> None:
        offenders = []
        for path in iter_v3_sources():
            if path.is_relative_to(_ERROR_INTERNAL_HOME):
                continue
            for module, symbol in _module_imports(path):
                if not module.startswith("localdata_mcp.nexus.error"):
                    continue
                leaf = _resolved(module, symbol).removeprefix(
                    "localdata_mcp.nexus.error"
                )
                first = leaf.lstrip(".").split(".", 1)[0] if leaf else ""
                if first and first not in _ERROR_SEAMS:
                    offenders.append(f"{path}: {module} ({symbol})")
        assert offenders == [], offenders


class TestPolicyCoversAllEightNexuses:
    """The declarative completeness check: every nexus package under
    nexus/ is either constrained by a rule in this file, covered by its
    own dedicated gate, or declared free — no ninth package appears
    unclassified."""

    _CLASSIFIED = frozenset(
        {
            "contract",  # seam-listed for tools (NX-1)
            "config",  # free by declaration (NX-2)
            "error",  # internal-set rule here (NX-3)
            "observability",  # logging seam free; purity gate (NX-4)
            "persistence",  # test_persistence_import_graph.py (NX-5)
            "chokepoint",  # internal-set rule here (NX-6)
            "response",  # seam package (NX-7)
            "export",  # seam package (NX-8)
        }
    )

    def test_every_nexus_package_is_classified(self) -> None:
        packages = {
            entry.name
            for entry in (SRC_ROOT / "nexus").iterdir()
            if entry.is_dir() and (entry / "__init__.py").exists()
        }
        assert packages == self._CLASSIFIED, (
            "nexus packages changed — classify the delta in this gate's "
            f"policy: {sorted(packages.symmetric_difference(self._CLASSIFIED))}"
        )
