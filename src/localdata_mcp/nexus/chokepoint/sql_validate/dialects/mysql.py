"""sql_validate/dialects/mysql.py — the MySQL fragment (data).

MySQL's dialect deltas per NFR-104: `LOAD_FILE` denied outright
(server-side file read). `INTO OUTFILE`/`INTO DUMPFILE` do not parse
at the pinned sqlglot version, so they are refused structurally by the
parse-failure rule — the denied-function entry documents the intent
and stays as the second layer should a future pin parse them.
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
    backend_kind="mysql",
    query_statements=SHARED_QUERY_STATEMENTS,
    mutation_statements=SHARED_MUTATION_STATEMENTS,
    allowed_nodes=SHARED_ALLOWED_NODES,
    mutation_nodes=SHARED_MUTATION_NODES,
    denied_nodes=SHARED_DENIED_NODES | frozenset({"IntoOutfile", "LoadData"}),
    denied_functions=frozenset({"load_file"}),
)
