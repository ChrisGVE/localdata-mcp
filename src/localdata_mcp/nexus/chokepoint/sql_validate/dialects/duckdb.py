"""sql_validate/dialects/duckdb.py — the DuckDB fragment (data).

DuckDB's dialect deltas per NFR-104: the read-side table functions
(`read_csv_auto` family) are LOCAL-file constructs — permitted through
`guarded_query` on any posture with every path NFR-108-contained; the
write-side `COPY … TO` is the same class through `guarded_mutation` on
read-write posture only (declared direction `write` covers COPY FROM
too — it mutates the database, so the stricter entrypoint applies to
both forms). `EXPORT DATABASE` does not parse at the pinned sqlglot
version and is therefore refused structurally (fail-safe capability
gap, revisited at a pin bump). Extension loading is the deny layer:
`Install` is shared-denied, `LOAD` lowers to `Command` (shared-denied).
"""

from __future__ import annotations

from ..policy import (
    SHARED_ALLOWED_NODES,
    SHARED_DENIED_NODES,
    SHARED_MUTATION_NODES,
    SHARED_MUTATION_STATEMENTS,
    SHARED_QUERY_STATEMENTS,
    DialectPolicy,
    LocalFileConstruct,
)

POLICY = DialectPolicy(
    backend_kind="duckdb",
    query_statements=SHARED_QUERY_STATEMENTS,
    mutation_statements=SHARED_MUTATION_STATEMENTS,
    allowed_nodes=SHARED_ALLOWED_NODES,
    mutation_nodes=SHARED_MUTATION_NODES,
    denied_nodes=SHARED_DENIED_NODES,
    # httpfs never loads (Install/LOAD denied above), and any URL a
    # read function names fails NFR-108 containment — no function-name
    # deny entries are needed here.
    denied_functions=frozenset(),
    local_file_constructs=(
        LocalFileConstruct(kind="function", name="read_csv_auto", direction="read"),
        LocalFileConstruct(kind="function", name="read_csv", direction="read"),
        LocalFileConstruct(kind="function", name="read_parquet", direction="read"),
        LocalFileConstruct(kind="function", name="read_json_auto", direction="read"),
        LocalFileConstruct(kind="function", name="read_json", direction="read"),
        LocalFileConstruct(kind="node", name="Copy", direction="write"),
    ),
)
