"""localdata_mcp/nexus/observability/purity_check.py — the T2 teeth.

An importable static check (run by pytest, hence by CI — the
default_site_check.py pattern) asserting the structural half of the T2
fix: the v3 tree contains no print() call, and no StreamHandler-shaped
construction outside NX-4's one handler home — with a stdout-bound
construction banned even there (NFR-303/304, ARCHITECTURE.md section 8
NX-4: "nothing calls print() or opens its own StreamHandler").
Neighbors: config.py is the one home this check protects; the gated
file set comes from nexus/gated_tree.py.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

from ..gated_tree import iter_v3_sources

# The only file allowed to construct a (stderr-bound) handler,
# relative to src/localdata_mcp/.
HANDLER_HOME = "nexus/observability/config.py"

# Constructor names that open a stream handler path. StderrHandler is
# NX-4's own subclass: constructing it elsewhere is still a second
# handler path — handlers are wired in the home, used via get_logger().
_HANDLER_NAMES = frozenset({"StreamHandler", "StderrHandler"})


@dataclass(frozen=True)
class Violation:
    """One purity break: where and what."""

    filename: str
    lineno: int
    description: str

    def __str__(self) -> str:
        return f"{self.filename}:{self.lineno}: {self.description}"


def check_purity(root: Path | None = None) -> list[Violation]:
    """Sweep the gated v3 tree; a non-empty result fails the build."""
    violations: list[Violation] = []
    for path in iter_v3_sources(root):
        violations.extend(scan_source(path.read_text(encoding="utf-8"), str(path)))
    return violations


def scan_source(text: str, filename: str) -> list[Violation]:
    """Scan one module's source text for print calls and handler
    constructions (stdout-shaped anywhere; any shape outside the home)."""
    is_home = filename.endswith(HANDLER_HOME)
    violations: list[Violation] = []
    for node in ast.walk(ast.parse(text, filename=filename)):
        if not isinstance(node, ast.Call):
            continue
        callee = _callee_name(node.func)
        if callee == "print":
            violations.append(
                Violation(filename, node.lineno, "print() call; log through NX-4")
            )
        elif callee in _HANDLER_NAMES:
            violations.extend(_handler_violation(node, filename, is_home))
    return sorted(violations, key=lambda v: v.lineno)


def _callee_name(func: ast.expr) -> str | None:
    """The called name: `print` for Name nodes, the final attribute for
    dotted calls like `logging.StreamHandler`."""
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _handler_violation(call: ast.Call, filename: str, is_home: bool) -> list[Violation]:
    """Classify one handler construction."""
    if any(_is_sys_stdout(arg) for arg in call.args):
        return [
            Violation(
                filename,
                call.lineno,
                "stream handler bound to sys.stdout; fd 1 carries only JSON-RPC frames",
            )
        ]
    if not is_home:
        return [
            Violation(
                filename,
                call.lineno,
                f"second handler path; handlers are constructed only in {HANDLER_HOME}",
            )
        ]
    return []


def _is_sys_stdout(arg: ast.expr) -> bool:
    """True for the exact `sys.stdout` attribute shape."""
    return (
        isinstance(arg, ast.Attribute)
        and arg.attr == "stdout"
        and isinstance(arg.value, ast.Name)
        and arg.value.id == "sys"
    )
