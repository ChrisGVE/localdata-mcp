"""localdata_mcp/nexus/chokepoint/sql_validate/walker.py — the shared walk.

E6.2's mechanical half: parse with the pinned sqlglot per S4.1's
dialect mapping, then walk EVERY node against the dialect's declared
policy — a genuine allow-list, fail-safe in every disposition (§7):
parse failure → refuse; more than one statement → refuse; a node class
outside the applicable allow-set → refuse (including `Command`/
`Anonymous`/unknown, the escape hatch for constructs newer than the
pinned parser); the deny-set is applied first as the redundant second
layer. The walk is ENTRYPOINT-INDEPENDENT: it produces a cacheable
`SqlClassification` (statement category, mutation-node presence,
extracted path literals) and guard.py applies the per-call rules —
entrypoint, posture, containment — outside any cache (E6.3).
Neighbors: policy.py supplies the data; cache.py memoizes the
classification; guard.py is the only caller.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

import sqlglot
from sqlglot import expressions as exp

from .policy import BACKEND_TO_SQLGLOT_DIALECT, POLICIES, DialectPolicy

Category = Literal["query", "mutation", "local_file_read", "local_file_write"]


class SqlRefusedError(ValueError):
    """The statement failed the allow-list — a structured refusal
    (never a partial verdict); guard.py shapes it through NX-3."""


@dataclass(frozen=True)
class SqlClassification:
    """The cacheable outcome of one parse/walk (§7: statement category,
    node verdict, extracted path literals — nothing per-call)."""

    category: Category
    contains_mutation_nodes: bool
    path_literals: tuple[str, ...]


def classify(sql: str, backend_kind: str) -> SqlClassification:
    """Parse and walk `sql` for `backend_kind`, or refuse.

    Everything entrypoint- and posture-independent happens here; the
    refusals raised are deterministic per (dialect, text) and therefore
    cacheable alongside the classifications (E6.3).
    """
    policy = _policy_for(backend_kind)
    statement = _the_one_statement(sql, policy)
    category = _root_category(statement, policy)
    paths: list[str] = []
    contains_mutation = False
    saw_read_construct = False
    for node in statement.walk():
        mutation, read_construct = _walk_one(cast(exp.Expression, node), policy, paths)
        contains_mutation |= mutation
        saw_read_construct |= read_construct
    # A read-side local-file table function under an otherwise-plain
    # SELECT lifts the category so guard.py knows to contain its paths.
    if category == "query" and saw_read_construct:
        category = "local_file_read"
    return SqlClassification(
        category=category,
        contains_mutation_nodes=contains_mutation,
        path_literals=tuple(paths),
    )


def _policy_for(backend_kind: str) -> DialectPolicy:
    try:
        return POLICIES[backend_kind]
    except KeyError:
        raise SqlRefusedError(
            f"no validation policy for backend {backend_kind!r} — refused"
        ) from None


def _the_one_statement(sql: str, policy: DialectPolicy) -> exp.Expression:
    """Exactly one parseable statement, else refuse — never a vacuous
    pass over a partial tree (§7)."""
    dialect = BACKEND_TO_SQLGLOT_DIALECT[policy.backend_kind]
    try:
        statements = sqlglot.parse(sql, dialect=dialect)
    except Exception as failure:
        raise SqlRefusedError(f"statement does not parse: {failure}") from failure
    real = [s for s in statements if s is not None]
    if len(real) != 1:
        raise SqlRefusedError(f"exactly one statement is permitted, got {len(real)}")
    return cast(exp.Expression, real[0])


def _root_category(statement: exp.Expression, policy: DialectPolicy) -> Category:
    """The statement's top-level disposition — one of the three
    enumerated categories (§7), else refused. A write-side local-file
    statement node (DuckDB `Copy`, SQLite `Attach`) is `local_file_write`;
    read-side lifting happens during the walk."""
    root = type(statement).__name__
    if root in policy.denied_nodes:
        raise SqlRefusedError(f"construct {root} is denied outright — refused")
    if root in policy.query_statements:
        return "query"
    if root in policy.mutation_statements:
        return "mutation"
    if root in policy.local_file_nodes:
        return "local_file_write"
    raise SqlRefusedError(
        f"statement type {root} is outside the enumerated categories — refused"
    )


def _walk_one(
    node: exp.Expression, policy: DialectPolicy, paths: list[str]
) -> tuple[bool, bool]:
    """Validate one node against the class allow-list.

    Returns (is_mutation_node, is_read_local_file_construct). Order:
    deny-set first (the redundant second layer reports precisely), then
    function-name handling for `Anonymous`, then local-file statement
    nodes, then mutation nodes (recorded, not refused — guard.py's
    entrypoint decides), then the shared allow-list.
    """
    name = type(node).__name__
    if name in policy.denied_nodes:
        raise SqlRefusedError(f"construct {name} is denied outright — refused")
    if isinstance(node, exp.Anonymous):
        return False, _walk_anonymous(node, policy, paths)
    if name in policy.local_file_nodes:
        paths.extend(_string_literal_args(node))
        return False, False
    if name in policy.mutation_nodes:
        return True, False
    if name not in policy.allowed_nodes:
        raise SqlRefusedError(
            f"construct {name} is not on the {policy.backend_kind} allow-list — refused"
        )
    return False, False


def _walk_anonymous(
    node: exp.Anonymous, policy: DialectPolicy, paths: list[str]
) -> bool:
    """A function sqlglot could not type: permitted only as a declared
    local-file function (paths collected, returns True for read-side),
    refused otherwise. Returns whether it is a read-side construct."""
    function_name = (node.name or "").lower()
    if function_name in policy.denied_functions:
        raise SqlRefusedError(
            f"function {function_name}() is denied outright — refused"
        )
    if function_name in policy.local_file_functions:
        paths.extend(_string_literal_args(node))
        return policy.local_file_direction("function", function_name) == "read"
    raise SqlRefusedError(
        f"function {function_name}() is not on the {policy.backend_kind} "
        "allow-list — refused"
    )


def _string_literal_args(node: exp.Expression) -> list[str]:
    """Every string literal directly argued to a local-file construct —
    the paths NFR-108 containment must clear. Nested query expressions
    under a `COPY (SELECT …) TO` are walked separately as their own
    nodes; only this node's own direct literal args are paths."""
    literals: list[str] = []
    for value in node.args.values():
        for item in value if isinstance(value, list) else [value]:
            if isinstance(item, exp.Literal) and item.is_string:
                literals.append(item.this)
            elif isinstance(item, exp.Alias) and isinstance(item.this, exp.Literal):
                if item.this.is_string:
                    literals.append(item.this.this)
    return literals
