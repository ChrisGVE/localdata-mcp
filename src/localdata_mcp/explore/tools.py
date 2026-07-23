"""localdata_mcp/explore/tools.py — X-1 schema discovery (E9.1).

FR-201's three tools (names carried from `main` — DR GP2):
`describe_database` answers for EVERY declared endpoint kind — SQL
kinds with the table catalog, store kinds with their semantic shape
(kv key spaces, graph schema summary, rdf triple shape) — through the
guard's introspection seam; `describe_table` and `find_table` work
the SQL-kind table catalog (a store's schema is semantic — the guard
refuses the catalog view and points at the store tools). Local files
are NOT addressed here: X-1's signatures take an endpoint only, and
file-shaped discovery is X-2's `path=` half (profile_data) — the
I-1/I-2 separate-surfaces precedent. Uses the ingest runtime's one
guard handle (the tool layer shares it) and the one refusal wording
home. Neighbors: quality.py (X-2), search.py (X-3), categorical.py
(X-4) follow.
"""

from __future__ import annotations

from fnmatch import fnmatch
from typing import Any

from localdata_mcp.nexus.chokepoint.guard import UnknownEndpointError
from localdata_mcp.nexus.contract.spec import Param, TypeShape, tool_spec

from ..ingest.refusals import missing_entity_refusal, unknown_endpoint_refusal
from ..ingest.runtime import chokepoint


@tool_spec(
    name="describe_database",
    summary=(
        "Describe a declared endpoint's schema: SQL kinds return the "
        "table catalog (columns, keys, row counts), kv/tree stores "
        "their key-space shape, graph stores their node/edge shape, "
        "rdf stores their triple shape."
    ),
    params=(Param("endpoint", str, "The operator-declared endpoint name."),),
    input_shape=TypeShape.NONE,
    output_shape=TypeShape.SCALAR,
    domain="explore",
)
def describe_database(endpoint: str) -> Any:
    try:
        return chokepoint().describe_endpoint(endpoint)
    except UnknownEndpointError:
        raise unknown_endpoint_refusal(endpoint) from None


@tool_spec(
    name="describe_table",
    summary=(
        "Describe one table of a declared SQL-kind endpoint: columns "
        "with types and nullability, primary key, row count."
    ),
    params=(
        Param("endpoint", str, "The operator-declared endpoint name."),
        Param("table", str, "The table name (as the catalog lists it)."),
    ),
    input_shape=TypeShape.NONE,
    output_shape=TypeShape.SCALAR,
    domain="explore",
)
def describe_table(endpoint: str, table: str) -> Any:
    try:
        described = chokepoint().describe_endpoint_table(endpoint, table)
    except UnknownEndpointError:
        raise unknown_endpoint_refusal(endpoint) from None
    if described is None:
        raise missing_entity_refusal(
            f"Table {table!r} does not exist on endpoint {endpoint!r}.",
            "Call describe_database(endpoint) to list the tables, or "
            "find_table(endpoint, pattern) to search by name.",
        )
    return described


@tool_spec(
    name="find_table",
    summary=(
        "Find tables on a declared SQL-kind endpoint whose names match "
        "a glob pattern (e.g. 'sales_*')."
    ),
    params=(
        Param("endpoint", str, "The operator-declared endpoint name."),
        Param("name_pattern", str, "A glob pattern matched against table names."),
    ),
    input_shape=TypeShape.NONE,
    output_shape=TypeShape.SCALAR,
    domain="explore",
)
def find_table(endpoint: str, name_pattern: str) -> Any:
    try:
        names = chokepoint().endpoint_table_names(endpoint)
    except UnknownEndpointError:
        raise unknown_endpoint_refusal(endpoint) from None
    matched = [name for name in names if fnmatch(name, name_pattern)]
    return {
        "endpoint": endpoint,
        "name_pattern": name_pattern,
        "matches": matched,
        "searched_tables": len(names),
    }
