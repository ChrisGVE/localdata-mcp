"""tests/v3/test_fr305_primitive_gate.py — the FR-305 grep gate, live in CI.

FR-305 bars the whole host-code-execution-primitive equivalence class,
not `eval`/`exec` alone: `compile()`, `__import__`/
`importlib.import_module` on non-literal input, and pandas'
`DataFrame.query`/`DataFrame.eval`/`pd.eval` are all in scope. This
gate is that criterion as an AST scan over the gated v3 tree
(gated_tree.py — the same sweep as every static gate; pytest runs
per-PR, so the gate is live in CI):

- ANY reference to the `eval`/`exec`/`compile`/`__import__` builtins —
  a bare mention, not only a call, so aliasing (`f = eval`) cannot
  slip past. `re.compile` and friends are attribute accesses, not the
  builtin, and pass untouched.
- ANY attribute call named `import_module`, `eval`, or `query` —
  receiver types are unresolvable statically, so the scan is
  deliberately over-broad and fail-safe; the sanctioned sites are a
  DECLARED allow-list carrying its rationale, never a scan exemption
  by pattern.

The battery leg (host-escape payloads through every string-accepting
registered parameter, derived from the NX-1 registry) is E8+ work once
the real tool surface exists — this file is the static leg NX-6
asserts standalone.
"""

from __future__ import annotations

import ast
from pathlib import Path

from localdata_mcp.nexus.gated_tree import iter_v3_sources

_BANNED_BUILTINS = frozenset({"eval", "exec", "compile", "__import__"})
_BANNED_ATTRIBUTE_CALLS = frozenset({"import_module", "eval", "query"})

# The declared sanctioned sites: (path suffix, name) → rationale.
_ALLOWED: dict[tuple[str, str], str] = {
    (
        "nexus/contract/spec_modules.py",
        "import_module",
    ): "iterates the declared literal SPEC_MODULES roster — never caller input",
    (
        "nexus/chokepoint/expr_eval.py",
        "eval",
    ): "asteval Interpreter.eval — the closed numeric grammar FR-305 mandates",
}


def _is_allowed(path: Path, name: str) -> bool:
    return any(
        str(path).endswith(suffix) and name == allowed_name
        for (suffix, allowed_name) in _ALLOWED
    )


def _violations_in(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in _BANNED_BUILTINS:
            found.append(f"{path}:{node.lineno}: reference to builtin {node.id!r}")
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in _BANNED_ATTRIBUTE_CALLS
            and not _is_allowed(path, node.func.attr)
        ):
            found.append(f"{path}:{node.lineno}: attribute call .{node.func.attr}()")
    return found


class TestFr305PrimitiveGate:
    def test_no_host_code_execution_primitive_in_the_gated_tree(self) -> None:
        offenders = [
            violation
            for path in iter_v3_sources()
            for violation in _violations_in(path)
        ]
        assert offenders == [], offenders

    def test_the_allow_list_names_only_live_sites(self) -> None:
        """A stale allow-list entry is a hole waiting for a new file —
        every entry must still match an existing gated source that
        actually uses the name it sanctions."""
        gated = list(iter_v3_sources())
        for (suffix, name), _rationale in _ALLOWED.items():
            matching = [path for path in gated if str(path).endswith(suffix)]
            assert matching, f"allow-list names a missing file: {suffix}"
            assert any(
                f".{name}(" in path.read_text(encoding="utf-8") for path in matching
            ), f"allow-list entry unused: {suffix} / {name}"
