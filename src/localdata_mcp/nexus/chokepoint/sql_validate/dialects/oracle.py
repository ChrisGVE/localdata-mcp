"""sql_validate/dialects/oracle.py — the Oracle fragment (data).

Oracle's dialect deltas per NFR-104: `BFILENAME` (server-side file
locator) and the `UTL_FILE`/`DBMS_LOB` entry points callable in SQL
context are denied outright — server-side file and network reach.
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
    backend_kind="oracle",
    query_statements=SHARED_QUERY_STATEMENTS,
    mutation_statements=SHARED_MUTATION_STATEMENTS,
    allowed_nodes=SHARED_ALLOWED_NODES,
    mutation_nodes=SHARED_MUTATION_NODES,
    denied_nodes=SHARED_DENIED_NODES,
    denied_functions=frozenset(
        {
            "bfilename",
            "utl_file.fopen",
            "dbms_lob.loadfromfile",
            "utl_http.request",
        }
    ),
)
