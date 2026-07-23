"""sql_validate/dialects/postgresql.py — the PostgreSQL fragment (data).

PostgreSQL's dialect deltas per NFR-104: `COPY` names a SERVER-side
path (the file lives on the database host, not the client), so it is
a network/capability construct here — `Copy` is DENIED outright, the
explicit deny-vs-contain disposition that distinguishes this fragment
from DuckDB's (where `Copy` reaches the LOCAL filesystem and is
containable). The server-side file functions are the denied-function
second layer.
"""

from __future__ import annotations

from ..policy import (
    SHARED_ALLOWED_NODES,
    SHARED_DENIED_NODES,
    SHARED_MUTATION_NODES,
    SHARED_MUTATION_STATEMENTS,
    SHARED_QUERY_STATEMENTS,
    DialectPolicy,
)

POLICY = DialectPolicy(
    backend_kind="postgresql",
    query_statements=SHARED_QUERY_STATEMENTS,
    mutation_statements=SHARED_MUTATION_STATEMENTS,
    allowed_nodes=SHARED_ALLOWED_NODES,
    mutation_nodes=SHARED_MUTATION_NODES,
    denied_nodes=SHARED_DENIED_NODES | frozenset({"Copy"}),
    denied_functions=frozenset(
        {
            "pg_read_file",
            "pg_read_binary_file",
            "pg_ls_dir",
            "pg_stat_file",
            "lo_import",
            "lo_export",
            "dblink",
            "dblink_connect",
        }
    ),
)
