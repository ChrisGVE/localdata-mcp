"""localdata_mcp/nexus/config/default_site_check.py — NFR-403's teeth.

An importable static check (run by pytest, hence by CI) asserting that
no S8 default value is restated in the v3 tree outside the ConfigModel
declarations. The expected values are read from the model itself —
keeping a literal list here would be exactly the second declaration
this check exists to prevent.

Scan rules, mechanical by construction:
- floats (except 0.0/1.0) and ints >= 2048 are distinctive — any
  literal occurrence anywhere is flagged;
- other ints >= 4 are common — flagged only in restatement-prone
  contexts (module/class-level constant assignments and function
  parameter defaults), where a "default" would be re-declared;
- ints 0-3 are structurally unscannable (schema versions, indices) and
  exempt; their one-home discipline rests on review.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from .models import ConfigModel, iter_config_fields

# The v3 packages the gate covers — the same set v3-ci.yml gates.
_V3_PACKAGES = (
    "nexus",
    "ingest",
    "explore",
    "process",
    "visualize",
    "testbench",
)

# The one legitimate home of the defaults (plus its testbench section).
_HOME_FILES = {"models.py", "models_testbench.py"}

_SRC_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Violation:
    """One restated default: where, what, and in which context."""

    filename: str
    lineno: int
    value: Any
    context: str

    def __str__(self) -> str:
        return (
            f"{self.filename}:{self.lineno}: S8 default {self.value!r} "
            f"restated ({self.context}); its one home is the ConfigModel"
        )


def s8_default_values() -> dict[str, Any]:
    """Every S8 default, read from a default-constructed model (derived
    values resolved) — never from a parallel list."""
    model = ConfigModel()
    return {
        f"{section}.{fld.name}": getattr(getattr(model, section), fld.name)
        for section, fld in iter_config_fields()
    }


def _split_by_distinctiveness() -> tuple[set[float], set[int]]:
    """(distinctive, common-int) value sets per the module's scan rules."""
    distinctive: set[float] = set()
    common: set[int] = set()
    for value in s8_default_values().values():
        if isinstance(value, float) and value not in (0.0, 1.0):
            distinctive.add(value)
        elif isinstance(value, int) and value >= 2048:
            distinctive.add(value)
        elif isinstance(value, int) and value >= 4:
            common.add(value)
    return distinctive, common


def check_one_default_site(root: Path | None = None) -> list[Violation]:
    """Scan the v3 tree; a non-empty result is an NFR-403 failure."""
    base = _SRC_ROOT if root is None else root
    violations: list[Violation] = []
    for package in _V3_PACKAGES:
        for path in sorted((base / package).rglob("*.py")):
            if path.name in _HOME_FILES and path.parent.name == "config":
                continue
            violations.extend(scan_source(path.read_text(encoding="utf-8"), str(path)))
    return violations


def scan_source(text: str, filename: str) -> list[Violation]:
    """Scan one module's source text for restated S8 defaults."""
    distinctive, common = _split_by_distinctiveness()
    tree = ast.parse(text, filename=filename)
    violations = [
        Violation(filename, node.lineno, node.value, "literal")
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and not isinstance(node.value, bool)
        and isinstance(node.value, (int, float))
        and node.value in distinctive
    ]
    violations.extend(
        Violation(filename, node.lineno, node.value, context)
        for node, context in _restatement_contexts(tree)
        if not isinstance(node.value, bool) and node.value in common
    )
    return sorted(violations, key=lambda v: v.lineno)


def _restatement_contexts(
    tree: ast.Module,
) -> Iterator[tuple[ast.Constant, str]]:
    """Constant nodes sitting where a default would be re-declared."""
    for statement in tree.body:
        yield from _constant_assignments(statement, "module constant")
        if isinstance(statement, ast.ClassDef):
            for member in statement.body:
                yield from _constant_assignments(member, "class constant")
    for walked in ast.walk(tree):
        if isinstance(walked, (ast.FunctionDef, ast.AsyncFunctionDef)):
            defaults = [*walked.args.defaults, *walked.args.kw_defaults]
            for default in defaults:
                if isinstance(default, ast.Constant):
                    yield default, "parameter default"


def _constant_assignments(
    node: ast.stmt, context: str
) -> Iterator[tuple[ast.Constant, str]]:
    value = getattr(node, "value", None)
    if isinstance(node, (ast.Assign, ast.AnnAssign)) and isinstance(
        value, ast.Constant
    ):
        yield value, context
