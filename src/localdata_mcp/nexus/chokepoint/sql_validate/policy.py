"""localdata_mcp/nexus/chokepoint/sql_validate/policy.py — the policy, as data.

E6.2's declarative half (§7 SQL-AST detail): the schema one dialect
policy fills, the `backend_kind → sqlglot` dialect mapping (S4.1 —
`postgresql→postgres`, `mssql→tsql`, identities elsewhere; verified
against the pinned sqlglot: the unmapped names raise Unknown dialect),
the SHARED node vocabulary every dialect starts from, and the one
aggregated mapping assembled from the per-dialect data fragments in
`dialects/*.py`. No per-dialect control flow exists anywhere — adding
a dialect is a one-file data edit (§9's pre-split). Neighbors:
walker.py is the only consumer; guard.py never reads policy directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Mapping

# S4.1's declared mapping — the two renames plus identities. An
# NFR-104 battery row asserts a bare parse_one succeeds for every
# mapped key at the pinned sqlglot version.
BACKEND_TO_SQLGLOT_DIALECT: Mapping[str, str] = {
    "sqlite": "sqlite",
    "postgresql": "postgres",
    "mysql": "mysql",
    "duckdb": "duckdb",
    "mssql": "tsql",
    "oracle": "oracle",
}

# E8.3's store families validate as the engine they physically are:
# a kv/tree/graph endpoint IS a SQLite file carrying the store schema
# (nexus/persistence/store_schemas.py), so its statements walk the
# sqlite policy — declared here as data, resolved by walker.py.
STORE_BACKEND_ALIASES: Mapping[str, str] = {
    "kv": "sqlite",
    "tree": "sqlite",
    "graph": "sqlite",
}

Direction = Literal["read", "write"]


@dataclass(frozen=True)
class LocalFileConstruct:
    """One explicitly-conditioned local-file capability (NFR-104's one
    exception class): a statement node class (`kind="node"`, e.g.
    DuckDB `Copy`, SQLite `Attach`) or a table-function name
    (`kind="function"`, e.g. `read_csv_auto`) with its declared
    direction — read-side through `guarded_query` on any posture,
    write-side through `guarded_mutation` on read-write posture only;
    every extracted path literal passes NFR-108 containment either way.
    """

    kind: Literal["node", "function"]
    name: str
    direction: Direction


@dataclass(frozen=True)
class DialectPolicy:
    """One dialect's complete allow-list — pure data, no behavior."""

    backend_kind: str
    query_statements: frozenset[str] = field(default_factory=frozenset)
    mutation_statements: frozenset[str] = field(default_factory=frozenset)
    allowed_nodes: frozenset[str] = field(default_factory=frozenset)
    mutation_nodes: frozenset[str] = field(default_factory=frozenset)
    denied_nodes: frozenset[str] = field(default_factory=frozenset)
    # Function names (lowercase) denied outright regardless of posture
    # — the redundant second layer inside the allow-list (§7).
    denied_functions: frozenset[str] = field(default_factory=frozenset)
    local_file_constructs: tuple[LocalFileConstruct, ...] = ()

    @property
    def local_file_functions(self) -> frozenset[str]:
        return frozenset(
            c.name for c in self.local_file_constructs if c.kind == "function"
        )

    @property
    def local_file_nodes(self) -> frozenset[str]:
        return frozenset(c.name for c in self.local_file_constructs if c.kind == "node")

    def local_file_direction(self, kind: str, name: str) -> Direction:
        for construct in self.local_file_constructs:
            if construct.kind == kind and construct.name == name:
                return construct.direction
        raise KeyError((kind, name))


# --- The shared vocabulary (every dialect starts here) ---------------

# Root statement classes per entrypoint category.
SHARED_QUERY_STATEMENTS: frozenset[str] = frozenset(
    {"Select", "Union", "Intersect", "Except"}
)
SHARED_MUTATION_STATEMENTS: frozenset[str] = frozenset({"Insert", "Update", "Delete"})

# Node classes only the mutation walk accepts — the read walk refuses
# any of these AT ANY DEPTH (the data-modifying-CTE rule, NFR-104/113).
SHARED_MUTATION_NODES: frozenset[str] = frozenset(
    {"Insert", "Update", "Delete", "Returning", "Values"}
)

# Denied outright in every dialect: sqlglot's escape hatches for
# constructs newer than the pinned parser (`Command`), and capability
# loading (`Install`). `Anonymous` is NOT here — it is handled by name
# against denied_functions/local_file_functions and refused otherwise.
SHARED_DENIED_NODES: frozenset[str] = frozenset({"Command", "Install", "Use"})

# The enumerated expression vocabulary of a permitted read statement —
# harvested from representative legitimate queries across all six
# dialects at the pinned sqlglot version. Anything outside it (DDL,
# transaction control, unknown constructs) is refused by absence.
SHARED_ALLOWED_NODES: frozenset[str] = frozenset(
    {
        # bind-parameter slots (`:name`, `?`, `$n`) — the guard's
        # mandated pattern: values bind through the driver, never
        # splice into text; the slot itself is inert.
        "Placeholder",
        "Parameter",
        # statement scaffolding
        "Select",
        "Union",
        "Intersect",
        "Except",
        "With",
        "CTE",
        "Subquery",
        "From",
        "Where",
        "Group",
        "Having",
        "Order",
        "Ordered",
        "Limit",
        "Offset",
        "Distinct",
        "Join",
        "Alias",
        "TableAlias",
        "Table",
        "Schema",
        "Column",
        "Identifier",
        "Star",
        "Tuple",
        "Paren",
        "Var",
        # structural children of a contained local-file construct
        # (COPY's credentials clause) — data, not a reachable capability.
        "Credentials",
        # literals and types
        "Literal",
        "Null",
        "Boolean",
        "DataType",
        "Interval",
        "Cast",
        "TryCast",
        # predicates and operators
        "And",
        "Or",
        "Not",
        "EQ",
        "NEQ",
        "GT",
        "GTE",
        "LT",
        "LTE",
        "Is",
        "In",
        "Between",
        "Like",
        "ILike",
        "Exists",
        "Any",
        "All",
        "Add",
        "Sub",
        "Mul",
        "Div",
        "Mod",
        "Pow",
        "Neg",
        "DPipe",
        # conditional and null handling
        "Case",
        "If",
        "Coalesce",
        "Nullif",
        # aggregates and math
        "Count",
        "Sum",
        "Avg",
        "Min",
        "Max",
        "Abs",
        "Round",
        "Floor",
        "Ceil",
        "Sqrt",
        "Exp",
        "Ln",
        "Log",
        "Stddev",
        "Variance",
        "ArrayAgg",
        "GroupConcat",
        "PercentileCont",
        "WithinGroup",
        # strings
        "Lower",
        "Upper",
        "Trim",
        "Length",
        "Substring",
        "Concat",
        "Replace",
        # dates
        "CurrentDate",
        "CurrentTime",
        "CurrentTimestamp",
        "Date",
        "Extract",
        "TimeToStr",
        "TsOrDsToDate",
        "TsOrDsToTimestamp",
        # window machinery
        "Window",
        "WindowSpec",
        "RowNumber",
        "Rank",
        "Lag",
        "Lead",
        "Ntile",
        "FirstValue",
        "LastValue",
    }
)


def _aggregate_policies() -> Mapping[str, DialectPolicy]:
    """The one mapping, assembled from the dialect data fragments."""
    from .dialects import duckdb, mssql, mysql, oracle, postgresql, sqlite

    policies = {
        fragment.POLICY.backend_kind: fragment.POLICY
        for fragment in (sqlite, postgresql, mysql, duckdb, mssql, oracle)
    }
    assert set(policies) == set(BACKEND_TO_SQLGLOT_DIALECT)
    return policies


POLICIES: Mapping[str, DialectPolicy] = _aggregate_policies()
