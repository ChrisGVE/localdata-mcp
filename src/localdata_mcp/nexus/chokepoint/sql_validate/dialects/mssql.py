"""sql_validate/dialects/mssql.py — the MSSQL/T-SQL fragment (data).

T-SQL's dialect deltas per NFR-104: the rowset/file-bridging functions
(`OPENROWSET`, `OPENDATASOURCE`, `OPENQUERY`) are denied outright —
each reaches beyond the connected database (files, linked servers).
`BULK INSERT`/`xp_cmdshell` lower to `Command` at the pinned sqlglot
version and are refused by the shared deny layer.
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
    backend_kind="mssql",
    query_statements=SHARED_QUERY_STATEMENTS,
    mutation_statements=SHARED_MUTATION_STATEMENTS,
    allowed_nodes=SHARED_ALLOWED_NODES,
    mutation_nodes=SHARED_MUTATION_NODES,
    denied_nodes=SHARED_DENIED_NODES,
    denied_functions=frozenset(
        {"openrowset", "opendatasource", "openquery", "xp_cmdshell"}
    ),
)
