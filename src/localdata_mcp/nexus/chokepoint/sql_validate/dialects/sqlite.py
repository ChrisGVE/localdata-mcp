"""sql_validate/dialects/sqlite.py — the SQLite policy fragment (data).

SQLite's dialect deltas per NFR-104: `load_extension` denied outright
(capability loading); `ATTACH DATABASE` is the write-side local-file
construct — permitted only through `guarded_mutation` on read-write
posture with the attached path NFR-108-contained.
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
    backend_kind="sqlite",
    query_statements=SHARED_QUERY_STATEMENTS,
    mutation_statements=SHARED_MUTATION_STATEMENTS,
    allowed_nodes=SHARED_ALLOWED_NODES,
    mutation_nodes=SHARED_MUTATION_NODES,
    denied_nodes=SHARED_DENIED_NODES,
    denied_functions=frozenset({"load_extension"}),
    local_file_constructs=(
        LocalFileConstruct(kind="node", name="Attach", direction="write"),
    ),
)
